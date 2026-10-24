#!/usr/bin/env bash
# A26 machine setup, no token: preflight, the uploaded files against the laptop's hashes, the
# dependencies, the CUDA operator, CLIP's public weights (as D2's setup.sh).
#   bash ~/METOR/studies/a26/box/setup.sh
set -euo pipefail
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
WORK=~/ov-vidvrd; BOX=~/a26_box; A26=~/METOR/studies/a26
mkdir -p "$BOX"
exec > >(tee -a "$BOX/setup.log") 2>&1
say() { printf '\n== %s  [%s]\n' "$*" "$(date --iso-8601=seconds)"; }

say "0. preflight"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
free -g | awk 'NR==2{printf "system RAM : %s GiB total, %s available\n", $2, $7}'
df -h --output=avail / | tail -1 | xargs echo "disk avail :"
echo "cores      : $(nproc)"
command -v nvcc >/dev/null || { echo "!! STOP: no nvcc (need a -devel PyTorch image)"; exit 1; }
nvcc --version | tail -2
python -c "import torch,sys; print('python', sys.version.split()[0], '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu ok', torch.cuda.is_available())"
nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "4090" || { echo "!! STOP: not an RTX 4090"; exit 1; }
[ "$(free -g | awk 'NR==2{print $2}')" -ge 60 ] || { echo "!! STOP: < 64 GB RAM"; exit 1; }
bash "$A26/box/token_scan.sh" "setup, before any token"

say "1. the uploaded files against the laptop's SHA256"
( cd "$WORK" && sha256sum -c --quiet "$BOX/expected/expected_eov.sha256" )
( cd ~/METOR && sha256sum -c --quiet "$BOX/expected/expected_metor.sha256" )
echo "  all match: $(wc -l < "$BOX/expected/expected_eov.sha256") EOV + data, $(wc -l < "$BOX/expected/expected_metor.sha256") METOR ($(cat "$BOX/metor_commit.txt"))"

say "2. dependencies (not torch; as D2)"
pip install -q scipy scikit-learn matplotlib pyyaml ftfy regex einops timm fvcore pycocotools \
               opencv-python-headless gdown huggingface_hub hf_transfer \
               easydict tensorboard six protobuf pytest
pip freeze > "$BOX/pip_freeze.txt"

say "3. CUDA operator (the EOV tree's ops/, identical to METOR's)"
ARCH=$(python -c "import torch;c=torch.cuda.get_device_capability();print(f'{c[0]}.{c[1]}')")
( cd "$WORK/ops" && rm -rf build ./*.egg-info && TORCH_CUDA_ARCH_LIST="$ARCH" pip install -q --no-build-isolation . )
python -c "import torch, MultiScaleDeformableAttention; print('operator OK')"

say "4. CLIP ViT-L/14@336px (public; CLIP's own download checks the SHA256 in its URL)"
( cd ~/METOR && python -c "
from vlm.backbones.clip.clip import _MODELS, _download
print(_download(_MODELS['ViT-L/14@336px']))" )
echo "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02  $HOME/.cache/clip/ViT-L-14-336px.pt" | sha256sum -c

say "setup done (frames, weights, CLIP bank and ECC come next, with the token)"
