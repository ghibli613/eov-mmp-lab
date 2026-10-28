#!/usr/bin/env bash
# G1 machine, after setup.sh: EOV-MMP's CUDA operator (Deformable DETR's MultiScaleDeformableAttention),
# which its detector imports. Built exactly as the repo's own tools/setup_rented_box.sh:144-149 builds
# it, with nvcc 12.8 to match torch 2.11.0+cu128 (torch refuses a different CUDA major version).
# This image ships CUDA 13.2, so CUDA 12.8's toolkit comes from NVIDIA's apt repository, which the
# image configures (cuda-toolkit-12-8 only: no driver package). G0 never ran the detector, so G0's
# machine never needed this step. /venv/main then holds G0's freeze plus this one built package.
#   bash ~/ov-vidvrd-lab/studies/rpt/g1/box/build_ops.sh
set -euo pipefail
BOX=~/rpt_g1_box; WORK=~/ov-vidvrd-lab
exec > >(tee -a "$BOX/build_ops.log") 2>&1
trap 'echo "!! STOP: build_ops.sh failed at line $LINENO: $BASH_COMMAND"' ERR
source /venv/main/bin/activate
if ! [ -x /usr/local/cuda-12.8/bin/nvcc ]; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends cuda-toolkit-12-8
fi
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc --version | tail -2
ARCH=$(python -c "import torch; c = torch.cuda.get_device_capability(0); print(f'{c[0]}.{c[1]}')")
echo "TORCH_CUDA_ARCH_LIST=$ARCH"
# --no-build-isolation: ops/setup.py imports torch at module level (setup_rented_box.sh:144-148)
( cd "$WORK/ops" && rm -rf build ./*.egg-info && TORCH_CUDA_ARCH_LIST="$ARCH" pip install -q --no-build-isolation . )
( cd "$WORK" && python -c "import torch, MultiScaleDeformableAttention; print('operator OK | torch', torch.__version__)" )
pip freeze > "$BOX/pip_freeze_main_with_ops.txt"
echo "freeze vs G0's (only the operator may be added):"
diff "$BOX/expected/pip_freeze_main.txt" "$BOX/pip_freeze_main_with_ops.txt" || true
