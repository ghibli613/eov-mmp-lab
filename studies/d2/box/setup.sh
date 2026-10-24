#!/usr/bin/env bash
# D2 machine setup, no token (A25c). Session 3's setup, minus the clone, the token and the data
# download (tools/setup_rented_box.sh steps 0, 2, 3, as session 3 ran them), then the hash check
# of everything upload.sh copied, and CLIP's public weights.
#   bash ~/METOR/studies/d2/box/setup.sh
set -euo pipefail
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
WORK=~/ov-vidvrd; BOX=~/d2_box; D2=~/METOR/studies/d2
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
[ "$(free -g | awk 'NR==2{print $2}')" -ge 30 ] || { echo "!! STOP: < 32 GB RAM"; exit 1; }
bash "$D2/box/token_scan.sh" "setup, before any token"

say "1. the uploaded files: $(wc -l < "$D2/box/expected_upload.sha256") files against the laptop's SHA256"
( cd "$WORK" && sha256sum -c --quiet "$D2/box/expected_upload.sha256" )
echo "  all match"

say "2. dependencies (not torch; as tools/setup_rented_box.sh step 2)"
pip install -q scipy scikit-learn matplotlib pyyaml ftfy regex einops timm fvcore pycocotools \
               opencv-python-headless gdown huggingface_hub hf_transfer \
               easydict tensorboard six protobuf pytest
pip freeze > "$BOX/pip_freeze.txt"

say "3. CUDA operator (as step 3)"
ARCH=$(python -c "import torch;c=torch.cuda.get_device_capability();print(f'{c[0]}.{c[1]}')")
( cd "$WORK/ops" && rm -rf build ./*.egg-info && TORCH_CUDA_ARCH_LIST="$ARCH" pip install -q --no-build-isolation . )
python -c "import torch, MultiScaleDeformableAttention; print('operator OK')"

say "4. CLIP ViT-L/14@336px (public; CLIP's own download checks the SHA256 in its URL)"
( cd "$WORK" && python -c "
from vlm.backbones.clip.clip import _MODELS, _download
print(_download(_MODELS['ViT-L/14@336px']))" )
echo "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02  $HOME/.cache/clip/ViT-L-14-336px.pt" | sha256sum -c

say "setup done (frames, weights, CLIP bank and ECC come next, with the token)"
