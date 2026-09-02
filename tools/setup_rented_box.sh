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
# image ships. Use a PyTorch 2.x + CUDA 12.x template.
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

say "0. machine"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
free -g | awk 'NR==2{printf "system RAM: %s GiB total, %s available\n", $2, $7}'
df -h --output=avail "$(dirname "$WORK")" | tail -1 | xargs echo "disk available:"
RAM_GB=$(free -g | awk 'NR==2{print $2}')
if [ "$RAM_GB" -lt 30 ]; then
  echo "!! system RAM is ${RAM_GB} GiB. The longest videos need ~5 GB transient and"
  echo "!! 12.7 GiB is known to fail. Consider a box with >= 32 GB."
  echo "!! Continuing anyway in 10s; Ctrl-C to stop." && sleep 10
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
pip install -q scipy matplotlib pyyaml ftfy regex einops timm fvcore pycocotools \
               opencv-python-headless gdown huggingface_hub hf_transfer \
               easydict tensorboard six protobuf pytest
python - <<'PYCHK'
import importlib.util, sys
missing = [m for m in ("scipy", "matplotlib", "yaml", "cv2", "numpy", "PIL", "tqdm",
                       "einops", "timm", "ftfy", "regex", "easydict", "fvcore",
                       "pycocotools", "huggingface_hub")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit(f"missing after install: {missing}")
print("all imports present")
PYCHK
export HF_HUB_ENABLE_HF_TRANSFER=1

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
# NOT --steps frames: frames stream per batch during the run
python tools/prepare_data.py --steps anno,meta,gt

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
