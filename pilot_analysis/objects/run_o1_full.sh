#!/usr/bin/env bash
# Object check O1 (A22), step 2 -- all 800 training videos. Run ONLY after the small run was
# approved, on the same box, from the pushed commit:
#
#   cd ~/ov-vidvrd && git fetch && git reset --hard origin/main
#   nohup setsid bash pilot_analysis/objects/run_o1_full.sh > ~/o1_full.log 2>&1 &
#
# Resumable (a video whose manifest entry and file hash match is skipped). Ends with SHA256SUMS.
set -uo pipefail
source /venv/main/bin/activate
export OMP_NUM_THREADS="${THREADS:-8}" MKL_NUM_THREADS="${THREADS:-8}" HF_HUB_ENABLE_HF_TRANSFER=1
cd ~/ov-vidvrd
OUT="$HOME/ov-vidvrd/o1_full"
mkdir -p "$OUT"
git log --oneline -1 | tee "$OUT/commit.txt"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "O1 STOPPED: checkout differs from the pushed commit"; git status --short; exit 1; }
python pilot_analysis/objects/o1_dump.py --verify || { echo "O1 STOPPED: pinned-hash check"; exit 1; }
python pilot_analysis/objects/o1_dump.py --out "$OUT" --all
rc=$?
( cd "$OUT" && sha256sum ./*.pt manifest.jsonl videos.json > SHA256SUMS && wc -l SHA256SUMS )
[ "$rc" -eq 0 ] && echo "O1 FULL DONE" || echo "O1 FULL FINISHED WITH FAILURES (exit $rc): see $OUT/failures.jsonl"
