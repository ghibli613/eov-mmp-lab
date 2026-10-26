#!/usr/bin/env bash
# G0 machine setup, no token present (A28):
#   bash ~/ov-vidvrd-lab/studies/rpt/g0/box/setup.sh
# Preflight; the uploaded files against the laptop's SHA256; the EOV-MMP side's dependencies in the
# image's own /venv/main (torch untouched: session 3's 2.11.0+cu128, as setup_rented_box.sh step 2);
# CLIP ViT-L/14@336px from its public URL, hash-checked; SAM 3 in its own venv (torch 2.10.0+cu128 as
# SAM 3's README installs it, `pip install -e ".[notebooks]"`, psutil, setuptools 78.1.0).
set -euo pipefail
BOX=~/rpt_g0_box; WORK=~/ov-vidvrd-lab
mkdir -p "$BOX"
[ -f "$BOX/t0" ] || date +%s > "$BOX/t0"              # the machine's clock starts here (run.sh measure)
exec > >(tee -a "$BOX/setup.log") 2>&1
say() { printf '\n== %s  [%s]\n' "$*" "$(date --iso-8601=seconds)"; }
source /venv/main/bin/activate

say "0. preflight"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | awk 'NR==2{printf "system RAM : %s GiB total, %s available\n", $2, $7}'
df -h --output=avail / | tail -1 | xargs echo "disk avail :"
echo "cores      : $(nproc)"
python -c "import torch,sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu ok', torch.cuda.is_available())"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "4090" || { echo "!! STOP: not an RTX 4090"; exit 1; }
[ "$(free -g | awk 'NR==2{print $2}')" -ge 31 ] || { echo "!! STOP: < 32 GB RAM"; exit 1; }
python -c "import torch; assert torch.__version__.startswith('2.11.0') and torch.version.cuda == '12.8', torch.__version__" \
  || { echo "!! STOP: the image's torch is not 2.11.0+cu128 (session 3's)"; exit 1; }
bash "$WORK/studies/rpt/g0/box/token_scan.sh" "setup, before any token"

say "1. the uploaded files against the laptop's SHA256"
( cd "$WORK" && sha256sum -c --quiet "$BOX/expected/expected_data.sha256" )
( cd "$WORK" && sha256sum -c --quiet "$BOX/expected/expected_code.sha256" )
echo "  all match: $(wc -l < "$BOX/expected/expected_data.sha256") data files, $(wc -l < "$BOX/expected/expected_code.sha256") code files"
find "$WORK/data" -path '*anno/test*' -o -name 'test_*' | grep -q . && { echo "!! STOP: a test file is on the machine"; exit 1; } || true

say "2. EOV-MMP side: dependencies in /venv/main, not torch (setup_rented_box.sh step 2)"
pip install -q scipy scikit-learn matplotlib pyyaml ftfy regex einops timm fvcore pycocotools \
               opencv-python-headless huggingface_hub hf_transfer easydict six protobuf pytest
pip freeze > "$BOX/pip_freeze_main.txt"

say "3. CLIP ViT-L/14@336px (public; the URL carries its SHA256)"
( cd "$WORK" && python -c "
from vlm.backbones.clip.clip import _MODELS, _download
print(_download(_MODELS['ViT-L/14@336px']))" )
echo "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02  $HOME/.cache/clip/ViT-L-14-336px.pt" | sha256sum -c

say "4. SAM 3 in its own venv"
python3 -m venv ~/sam3env
~/sam3env/bin/python -m pip install -q --upgrade pip
~/sam3env/bin/python -m pip install -q torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
( cd ~/sam3 && ~/sam3env/bin/python -m pip install -q -e ".[notebooks]" )
~/sam3env/bin/python -m pip install -q psutil setuptools==78.1.0
~/sam3env/bin/python -c "import torch, sam3; from sam3.model_builder import build_sam3_video_predictor; print('sam3 OK | torch', torch.__version__)"
~/sam3env/bin/python -m pip freeze > "$BOX/pip_freeze_sam3env.txt"

say "setup done (the weights come next, with the author's token: fetch_with_token.sh)"
