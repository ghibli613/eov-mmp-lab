#!/usr/bin/env bash
# Stage 1, session 3 on a rented GPU box: setup, frames, and Task 2 (the 5-video
# check). It STOPS after Task 2 -- the full cache (Task 3) needs approval first.
#
#   git clone https://github.com/ghibli613/eov-mmp-lab.git ~/ov-vidvrd && cd ~/ov-vidvrd
#   export HF_TOKEN=hf_...          # read access to the private weights + frames repos
#   bash pilot_analysis/stage1c/setup_session3.sh
#
# Box: RTX 4090 (24 GB), >= 32 GB system RAM, 60 GB disk is enough (Task 2 needs
# ~15 GB: only its 5 videos' frames are fetched or decoded), Vast.ai PyTorch
# template (the image verified in tools/setup_rented_box.sh).
#
# Steps 0-5 are tools/setup_rented_box.sh's own -- preflight, clone/hard-sync,
# dependencies, CUDA operator, annotations, weights -- run verbatim, cut just
# before its step 6 (the old pilot run, not wanted here). Then:
#   6  pinned-hash check: every artefact in stage1b/preregistration.md, incl. A5
#   7  pick the 5 videos (3 train, 2 test), deterministically
#   8  frames for those 5 only: test from the private bundle (the pilot's exact
#      inputs), train decoded on the box with tools/extract_frames.py's function
#   9  frame provenance: box-decoded vs bundle frames, 2 test videos (report only)
#  10  Task 2: cache 3 train + 2 test videos; dump-fidelity check on the 2 test ones
#  11  summary: per-video time, size, fidelity, projection for all 1,000 videos
#
# The dump-fidelity check needs the pilot's segments_raw_all.json, copied in from
# the local machine (it is not in git):
#   scp -P <port> pilot_analysis/preds_from_pod/full/segments_raw_all.json \
#       root@<host>:~/ov-vidvrd/pilot_analysis/preds_from_pod/full/
set -euo pipefail

WORK="${WORK:-$HOME/ov-vidvrd}"
export WORK REPO="${REPO:-https://github.com/ghibli613/eov-mmp-lab.git}"
CACHE="${CACHE:-$WORK/cache_s3}"
SHARDS="https://huggingface.co/datasets/ghibli613/ov-vidvrd-frames/resolve/main/SHARDS.json"
DUMP="$WORK/pilot_analysis/preds_from_pod/full/segments_raw_all.json"
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
: "${HF_TOKEN:?set HF_TOKEN first -- the weights and frames repos are private}"

# ---- steps 0-5: the existing setup script, verbatim, up to its step 6 --------
SRC="$WORK/tools/setup_rented_box.sh"
[ -f "$SRC" ] || SRC="$(cd "$(dirname "$0")/../.." && pwd)/tools/setup_rented_box.sh"
PART="$(mktemp)"
awk '/^say "6\./ { exit } { print }' "$SRC" > "$PART"
grep -q '^say "5\.' "$PART" || { echo "!! could not cut setup_rented_box.sh before step 6"; exit 1; }
bash "$PART"
rm -f "$PART"

cd "$WORK"
export HF_HUB_ENABLE_HF_TRANSFER=1
THREADS="${THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
mkdir -p "$CACHE"
git log --oneline -1 | tee "$CACHE/commit.txt"

say "6. pinned-hash check (preregistration + A5)"
python pilot_analysis/stage1c/session3_tasks.py verify-hashes

say "7. pick the 5 videos (3 train, 2 test; from annotation frame counts)"
python pilot_analysis/stage1c/session3_tasks.py select-videos --out "$CACHE"
read -r -a TEST_VIDS <<< "$(python -c "import json;print(' '.join(json.load(open('$CACHE/selected_videos.json'))['test']))")"
read -r -a TRAIN_VIDS <<< "$(python -c "import json;print(' '.join(json.load(open('$CACHE/selected_videos.json'))['train']))")"

say "8. frames for those 5 only: test from the private bundle (the pilot's exact"
say "   inputs), train decoded here from the public videos (4.3 GB)"
python pilot_analysis/stage1c/session3_tasks.py fetch-test-frames --shards "$SHARDS" --videos "${TEST_VIDS[@]}"
python tools/prepare_data.py --steps videos || true
python pilot_analysis/stage1c/session3_tasks.py decode --videos "${TRAIN_VIDS[@]}"

say "9. frame provenance: box-decoded vs bundle, the 2 test videos (report only)"
python pilot_analysis/stage1c/session3_tasks.py frame-provenance --videos "${TEST_VIDS[@]}" --out "$CACHE"

say "10. Task 2: cache 3 train + 2 test videos"
python pilot_analysis/stage1c/cache_wrapper.py --split train --videos "${TRAIN_VIDS[@]}" --out "$CACHE"
python pilot_analysis/stage1c/cache_wrapper.py --split test  --videos "${TEST_VIDS[@]}"  --out "$CACHE"
if [ -f "$DUMP" ]; then
  python pilot_analysis/stage1c/cache_wrapper.py --split test --videos "${TEST_VIDS[@]}" \
      --fidelity-from-dump "$DUMP" --out "$CACHE"
else
  echo "!! $DUMP missing -- copy it in (see the header) and rerun this step:"
  echo "   python pilot_analysis/stage1c/cache_wrapper.py --split test --videos ${TEST_VIDS[*]} --fidelity-from-dump $DUMP --out $CACHE"
fi

say "11. summary -- STOP here; the full cache needs approval"
python pilot_analysis/stage1c/session3_tasks.py summarize --cache "$CACHE"
echo
echo "Copy the results off the box:"
echo "  scp -P <port> -r root@<host>:$CACHE ./cache_s3_task2"
