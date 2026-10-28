#!/usr/bin/env bash
# G1 machine setup, no token present (A29, "The machine"):
#   bash ~/ov-vidvrd-lab/studies/rpt/g1/box/setup.sh
# Preflight against G0's stack (RTX 4090, driver 580.142, Python 3.12.15, torch 2.11.0+cu128 in
# /venv/main; >= 192 GB RAM); records the cuDNN versions and what the container shows of its image
# (A29: the image digest and cuDNN are recorded, check 0 decides); the uploaded files against the
# laptop's SHA256; both package sets installed under G0's `pip freeze` as constraints and then
# required to equal it exactly; CLIP from its public URL, hash-checked. Any mismatch stops.
set -euo pipefail
BOX=~/rpt_g1_box; WORK=~/ov-vidvrd-lab; EXP=$BOX/expected
mkdir -p "$BOX"
exec > >(tee -a "$BOX/setup.log") 2>&1
say() { printf '\n== %s  [%s]\n' "$*" "$(date --iso-8601=seconds)"; }
stop() { echo "!! STOP: $*"; exit 1; }
source /venv/main/bin/activate

say "0. preflight (G0's stack)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | awk 'NR==2{printf "system RAM : %s GiB total, %s available\n", $2, $7}'
df -h --output=avail / | tail -1 | xargs echo "disk avail :"
echo "cores      : $(nproc)"
python -c "import torch,sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| cudnn', torch.backends.cudnn.version(), '| gpu ok', torch.cuda.is_available())"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX 4090" || stop "not an RTX 4090"
[ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" = "580.142" ] || stop "driver is not G0's 580.142"
[ "$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)" -ge 178 ] || stop "< 192 GB RAM"
[ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 60 ] || stop "< 60 GB free disk"
python -c "import sys, torch; assert sys.version.split()[0] == '3.12.15' and torch.__version__ == '2.11.0+cu128', (sys.version, torch.__version__)" \
  || stop "/venv/main is not G0's Python 3.12.15 + torch 2.11.0+cu128"
echo "image      : $(. /etc/os-release && echo "$PRETTY_NAME") | VAST_CONTAINERLABEL=${VAST_CONTAINERLABEL:-unset}"
echo "            (the image digest is not visible inside the container: read it from the Vast.ai instance page)"
bash "$WORK/studies/rpt/g0/box/token_scan.sh" "setup, before any token"

say "1. the uploaded files against the laptop's SHA256; no test file"
( cd "$WORK" && sha256sum -c --quiet "$EXP/expected_data.sha256" )
( cd "$WORK" && sha256sum -c --quiet "$EXP/expected_code.sha256" )
echo "  all match: $(wc -l < "$EXP/expected_data.sha256") data files, $(wc -l < "$EXP/expected_code.sha256") code files"
if find "$WORK/data" \( -path '*anno/test*' -o -name 'test_*' -o -name '*ECC_test*' \) | grep -q .; then stop "a test file is on the machine"; fi

say "2. EOV-MMP side: /venv/main, constrained to G0's freeze"
grep -vE '^-e |^#| @ ' "$EXP/pip_freeze_main.txt" > "$BOX/constraints_main.txt"
pip install -q -c "$BOX/constraints_main.txt" scipy scikit-learn matplotlib pyyaml ftfy regex einops timm fvcore \
    pycocotools opencv-python-headless huggingface_hub hf_transfer easydict six protobuf pytest
pip freeze > "$BOX/pip_freeze_main.txt"
diff "$EXP/pip_freeze_main.txt" "$BOX/pip_freeze_main.txt" || stop "/venv/main's packages differ from G0's freeze (diff above)"
echo "  /venv/main equals G0's freeze: $(sha256sum < "$BOX/pip_freeze_main.txt" | cut -c1-16)"

say "3. CLIP ViT-L/14@336px (public; the URL carries its SHA256)"
( cd "$WORK" && python -c "
from vlm.backbones.clip.clip import _MODELS, _download
print(_download(_MODELS['ViT-L/14@336px']))" 2>&1 | tail -1 )
echo "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02  $HOME/.cache/clip/ViT-L-14-336px.pt" | sha256sum -c

say "4. SAM 3 in its own venv, constrained to G0's freeze"
grep -vE '^-e |^#| @ ' "$EXP/pip_freeze_sam3env.txt" > "$BOX/constraints_sam3env.txt"
python3 -m venv ~/sam3env
~/sam3env/bin/python -m pip install -q --upgrade pip
~/sam3env/bin/python -m pip install -q -c "$BOX/constraints_sam3env.txt" torch torchvision --index-url https://download.pytorch.org/whl/cu128
( cd ~/sam3 && ~/sam3env/bin/python -m pip install -q -c "$BOX/constraints_sam3env.txt" -e ".[notebooks]" )
~/sam3env/bin/python -m pip install -q -c "$BOX/constraints_sam3env.txt" psutil setuptools
~/sam3env/bin/python -m pip freeze > "$BOX/pip_freeze_sam3env.txt"
diff "$EXP/pip_freeze_sam3env.txt" "$BOX/pip_freeze_sam3env.txt" || stop "sam3env's packages differ from G0's freeze (diff above)"
~/sam3env/bin/python -c "import torch, sam3; from sam3.model_builder import build_sam3_video_predictor; print('sam3 OK | torch', torch.__version__, '| cudnn', torch.backends.cudnn.version())"

say "setup done (the weights come next, with the author's token: fetch_with_token.sh)"
