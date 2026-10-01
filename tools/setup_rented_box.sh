#!/usr/bin/env bash
# One-shot setup + full pilot run on a rented GPU box (Vast.ai, RunPod, ...).
#
#   export HF_TOKEN=hf_...            # read access to both private repos
#   bash tools/setup_rented_box.sh    # from anywhere; it clones into $WORK
#
# Requirements, learned the hard way on Colab:
#   VRAM       >= 16 GB   (24 on a 3090/4090)
#   SYSTEM RAM >= 32 GB   <- the binding constraint. dataset.__getitem__ builds a
#                            list of per-frame CLIP tensors and torch.cat's it, so
#                            it holds the list AND the output at once: ~5 GB
#                            transient for the longest test video (1234 frames).
#                            Colab's 12.7 GB died reproducibly on a 645-frame one.
#   DISK       >= 60 GB
#
# Does NOT install torch -- it compiles the CUDA operator against whatever the
# image ships, so any recent PyTorch image works and nothing here is pinned.
#
# VERIFIED IMAGE: Vast.ai's own PyTorch template (RTX 4090, 2026-09-02):
#   Ubuntu 24.04.4, Python 3.12, venv /venv/main
#   torch 2.11.0+cu128, torchvision 0.26.0+cu128, numpy 2.5.2
#   nvcc 12.8 == torch CUDA 12.8, cudnn 91900
# which matches requirements.txt exactly. It ships only torch, torchvision,
# numpy and PyYAML -- step 2 installs the rest.
#
# Do NOT use a bare pytorch/pytorch:*-devel from Docker Hub: it has nvcc but no
# sshd, so the host's launch script loops on "ssh: command not found" and the
# instance is unreachable. A "-runtime" image has sshd but no nvcc.
set -euo pipefail

WORK="${WORK:-$HOME/ov-vidvrd}"
REPO="${REPO:-https://github.com/ghibli613/ov-vidvrd-lab.git}"
WEIGHTS_MANIFEST="https://huggingface.co/ghibli613/ov-vidvrd-weights/resolve/main/MANIFEST.json"
SHARDS="https://huggingface.co/datasets/ghibli613/ov-vidvrd-frames/resolve/main/SHARDS.json"
# Results stay on the box by default -- fetch them with scp (the command is
# printed at the end). Set PREDS_REPO=<user>/<repo> to also publish them to a
# private HuggingFace dataset, which is worth doing if you cannot copy them off
# promptly: the container disk dies with the instance.
PREDS_REPO="${PREDS_REPO:-}"
CKPT="baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

: "${HF_TOKEN:?set HF_TOKEN first -- both HuggingFace repos are private}"

say "0. preflight -- adapts to whatever the image ships"
# Nothing here is pinned to a torch or CUDA version. The operator is built
# against the image's own torch, so any recent PyTorch -devel image works; this
# step just reports what it found and fails fast on the things that genuinely
# break the run.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | awk 'NR==2{printf "system RAM : %s GiB total, %s available\n", $2, $7}'
df -h --output=avail / | tail -1 | xargs echo "disk       :"

if ! command -v nvcc >/dev/null 2>&1; then
  echo
  echo "!! nvcc NOT FOUND. The CUDA operator compiles from source, so this image"
  echo "!! cannot build it -- you have a -runtime image, not a -devel one."
  echo "!! Either re-rent with a '-devel' PyTorch image, or:"
  echo "!!   apt-get update && apt-get install -y cuda-toolkit-\$(python -c \"import torch;print(torch.version.cuda.replace('.','-'))\")"
  exit 1
fi

python - <<'PYCHK'
import shutil, subprocess, sys, torch
tv, tc = torch.__version__, torch.version.cuda
cc = torch.cuda.get_device_capability()
out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
nv = next((w.rstrip(",") for w in out.split() if w.startswith("V") and w[1:2].isdigit()), "?")[1:]
print(f"torch      : {tv}  (built for CUDA {tc})")
print(f"nvcc       : {nv}")
print(f"GPU        : {torch.cuda.get_device_name(0)}, compute {cc[0]}.{cc[1]}")
if not torch.cuda.is_available():
    sys.exit("!! torch.cuda.is_available() is False -- the image cannot see the GPU")
if tc and nv != "?" and tc.split(".")[0] != nv.split(".")[0]:
    print(f"!! nvcc is CUDA {nv} but torch was built for {tc}. Major versions differ,")
    print("!! so the operator may fail to build or import. If it does, install a torch")
    print(f"!! matching nvcc:  pip install torch --index-url "
          f"https://download.pytorch.org/whl/cu{nv.replace('.','')[:3]}")
PYCHK

RAM_GB=$(free -g | awk 'NR==2{print $2}')
if [ "$RAM_GB" -lt 30 ]; then
  echo
  echo "!! system RAM is ${RAM_GB} GiB. dataset.__getitem__ holds a whole video's"
  echo "!! per-frame tensors twice during torch.cat -- ~2.5 GiB for a 645-frame"
  echo "!! video, and the longest test video is 1,234 frames. 12.7 GiB failed"
  echo "!! reproducibly (PILOT-STATUS.md SS B.22). Want >= 32 GB."
  echo "!! Continuing in 10s; Ctrl-C to stop." && sleep 10
fi

say "1. clone"
mkdir -p "$(dirname "$WORK")"
if [ -d "$WORK/.git" ]; then
  # hard-sync rather than pull: the history upstream may have been amended or
  # force-pushed, and `pull --ff-only` refuses that, leaving you to delete the
  # checkout by hand. reset --hard only touches TRACKED files, so preds/,
  # output/ckpt/ and data/ (all untracked or gitignored) survive -- which is why
  # there is deliberately no `git clean` here.
  git -C "$WORK" fetch --quiet origin
  git -C "$WORK" reset --hard --quiet origin/main
  echo "  hard-synced to origin/main (untracked preds/, output/, data/ kept)"
else
  git clone --quiet "$REPO" "$WORK"
fi
cd "$WORK"
git log --oneline -1

say "2. dependencies (NOT torch -- the image's build stays)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# scipy is NOT optional: the tracker and the detector's Hungarian matcher both
# use linear_sum_assignment, and third_party/vidvrd_ii_helper imports interp1d.
# Leaving it out fails at step 4 with ModuleNotFoundError: No module named 'scipy'.
# transformers/tokenizers are in requirements.txt but imported nowhere, so skipped.
pip install -q scipy scikit-learn matplotlib pyyaml ftfy regex einops timm fvcore pycocotools \
               opencv-python-headless gdown huggingface_hub hf_transfer \
               easydict tensorboard six protobuf pytest
python - <<'PYCHK'
import importlib.util, sys
missing = [m for m in ("scipy", "sklearn", "matplotlib", "yaml", "cv2", "numpy",
                       "PIL", "tqdm",
                       "einops", "timm", "ftfy", "regex", "easydict", "fvcore",
                       "pycocotools", "huggingface_hub")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing after install: {missing}")
print("all imports present")
PYCHK
export HF_HUB_ENABLE_HF_TRANSFER=1

# Cap torch's intra-op threads. MEASURED on a 255-core box: leaving this to
# torch's default (one thread per core) gave 228 s/video; capping it at 8 gave
# ~60 -- roughly 4x, taking a 12-hour run down to under 3. The pipeline's CPU
# work is many small per-frame tensors, so hundreds of threads spend more time
# synchronising than computing. Scale with the box, but do not remove: a big
# machine is where this hurts most.
THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
echo "  OMP_NUM_THREADS=$THREADS (of $(nproc) cores) -- see the comment above"

say "3. CUDA operator"
ARCH=$(python -c "import torch;c=torch.cuda.get_device_capability();print(f'{c[0]}.{c[1]}')")
echo "compute capability $ARCH"
# --no-build-isolation is REQUIRED: ops/setup.py imports torch at module level,
# and pip's isolated build env has no torch. Without it the build dies with
#   ModuleNotFoundError: No module named 'torch'
( cd ops && rm -rf build ./*.egg-info \
    && TORCH_CUDA_ARCH_LIST="$ARCH" pip install --no-build-isolation . )
python -c "import torch, MultiScaleDeformableAttention; print('operator OK')"

say "4. annotations, trajectories, class splits, GT"
# NOT --steps frames: frames stream per batch during the run.
# `|| true` because prepare_data reports on EVERY step and exits 1 if any is
# incomplete -- including videos, frames, bank and weights, which we skip on
# purpose. Its exit code is therefore always 1 here, and under `set -e` that
# killed the script after a perfectly successful data prep. Verify what we
# actually need instead.
python tools/prepare_data.py --steps anno,meta,gt || true
python - <<'PYCHK'
import glob, os, sys
sys.path.insert(0, os.getcwd())
from utils import paths
tr = len(glob.glob(os.path.join(paths.ANNO_TRAIN_DIR, "*.json")))
te = len(glob.glob(os.path.join(paths.ANNO_TEST_DIR, "*.json")))
meta = len(glob.glob(os.path.join(paths.META_DIR, "*.json"))) \
     + len(glob.glob(os.path.join(paths.META_DIR, "*.pkl")))
gt = [os.path.join(paths.META_DIR, "test_relation_gt.json"),
      os.path.join(paths.META_DIR, "test_object_trajectories_gt.json")]
print(f"  annotations {tr} train + {te} test")
print(f"  meta files  {meta}")
bad = []
if tr != 800 or te != 200: bad.append(f"annotations ({tr}/800, {te}/200)")
for g in gt:
    if not os.path.exists(g): bad.append(f"missing {os.path.basename(g)}")
if bad:
    sys.exit("  data prep INCOMPLETE: " + "; ".join(bad))
print("  data prep OK")
PYCHK

say "5. weights (eval subset, 2.92 GB)"
python tools/hugging_download.py --manifest "$WEIGHTS_MANIFEST" --only eval

say "6. the run -- ~3-4 h on a 3090, both predicate splits in one pass"
python pilot_analysis/scripts/dump_predictions.py \
    --ckpt_path "output/ckpt/$CKPT" \
    --path_AFLink output/ckpt/AFLink_epoch20.pth \
    --shards "$SHARDS" \
    --out preds/full \
    --disk-budget 4.0 --flush-every 5 \
    --frame_stride 1 --fuse-splits

say "7. phases 2 and 3 (CPU)"
python pilot_analysis/scripts/phase2_phase3.py --preds preds/full \
    2>&1 | tee preds/phase2_phase3_results.txt

say "8. publishing the results (off by default)"
# The container disk dies with the instance, so get them somewhere durable
# BEFORE anyone reaches for the trash icon. Non-fatal: an upload problem must not
# be how a finished 3-hour run gets lost.
if [ -n "$PREDS_REPO" ]; then
  STAGE="$WORK/_preds_bundle"
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  cp preds/full/*.json "$STAGE"/ 2>/dev/null || true
  cp preds/phase2_phase3_results.txt "$STAGE"/ 2>/dev/null || true
  if python tools/hugging_upload.py --repo "$PREDS_REPO" --repo-type dataset \
        --bundle "$STAGE" --dest pilot_analysis/preds \
        --title "pilot predictions (stride 1, fused)"; then
    echo "  results are safe at https://huggingface.co/datasets/$PREDS_REPO (private)"
  else
    echo "  !! upload FAILED. The results are still on this box at $WORK/preds --"
    echo "  !! copy them off before destroying the instance."
  fi
else
  echo "  skipped -- results stay on this box. COPY THEM OFF before destroying it."
fi

say "DONE -- results in $WORK/preds"
ls -la preds/full/
cat <<MSG

Results are in $WORK/preds. THEY DIE WITH THIS INSTANCE -- fetch them first:

  scp -i ~/.ssh/<key> -P <port> -r root@<host>:$WORK/preds ./preds_from_pod

Then re-run the analysis locally, no GPU needed:

  python pilot_analysis/scripts/phase2_phase3.py --preds <dir>

!! DESTROY the instance when you are done -- Vast bills while it runs, and a
!! stopped instance still bills for its disk.
MSG
