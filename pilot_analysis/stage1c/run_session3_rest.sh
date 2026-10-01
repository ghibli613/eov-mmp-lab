#!/usr/bin/env bash
# Stage 1, session 3: Tasks 3 -> 4 -> 5 -> 6 back to back on the GPU box.
# Run after setup_session3.sh (Task 2), from the pushed commit:
#
#   cd ~/ov-vidvrd && git fetch && git reset --hard origin/main
#   nohup setsid bash pilot_analysis/stage1c/run_session3_rest.sh > ~/session3_rest.log 2>&1 &
#
# Stops at the first failure or gate, printing "SESSION3 STOPPED":
#   Task 3  any video failed, or a fidelity residual > 1e-4          (exit 4)
#   Task 4  novel PredCls differs from the paper's 21.65 by > 3.0     (exit 3)
#   Task 5  paraphrase file does not match its SHA256                 (exit 5)
#   Task 6  SGDet differs from the pilot's 15.79 / 27.18 by > 0.3     (exit 6)
# Ends with SHA256SUMS over every cached file, for the copy-off.
set -uo pipefail
source /venv/main/bin/activate
export OMP_NUM_THREADS="${THREADS:-8}" MKL_NUM_THREADS="${THREADS:-8}" HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN="$(tr -d '[:space:]' < ~/.cache/huggingface/token)"
cd ~/ov-vidvrd
CACHE="$HOME/ov-vidvrd/cache_s3"
SGDET="$HOME/ov-vidvrd/cache_s3_sgdet"
SHARDS="https://huggingface.co/datasets/ghibli613/ov-vidvrd-frames/resolve/main/SHARDS.json"
mkdir -p "$CACHE" "$SGDET"

stage() { printf '\n== %s  [%s]\n' "$*" "$(date '+%H:%M:%S')"; }
run() {
  "$@"; local rc=$?
  if [ "$rc" -ne 0 ]; then echo "SESSION3 STOPPED at: $* (exit $rc)"; exit "$rc"; fi
}

git log --oneline -1 | tee "$CACHE/commit_rest.txt"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "SESSION3 STOPPED: checkout differs from the pushed commit"; git status --short; exit 1; }

stage "pinned-hash check"
run python pilot_analysis/stage1c/session3_tasks.py verify-hashes

stage "Task 3: full cache, 1,000 videos (test first, then train)"
run python pilot_analysis/stage1c/run_task3.py --out "$CACHE" --shards "$SHARDS"

stage "Task 4: baseline PredCls from the cache"
run python pilot_analysis/stage1c/eval_from_cache.py --cache "$CACHE" --out "$CACHE/task4"

stage "Task 5: paraphrase stability"
run python pilot_analysis/stage1c/paraphrase_test.py --cache "$CACHE" --out "$CACHE/task5"

stage "Task 6: SGDet cache on detected trajectories, 200 test videos"
run python pilot_analysis/stage1c/sgdet_cache.py --out "$SGDET"

stage "checksums"
( cd ~/ov-vidvrd && find cache_s3 cache_s3_sgdet -type f ! -name SHA256SUMS -print0 | sort -z \
    | xargs -0 sha256sum > SHA256SUMS_session3.txt && wc -l SHA256SUMS_session3.txt )
echo "SESSION3 ALL DONE"
