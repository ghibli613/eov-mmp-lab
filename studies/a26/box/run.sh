#!/usr/bin/env bash
# A26 on the machine, no token present:
#   bash ~/METOR/studies/a26/box/run.sh check        # pins only
#   bash ~/METOR/studies/a26/box/run.sh gate         # the 2-video gate: R0 identity, timing, projection
#   nohup setsid bash ~/METOR/studies/a26/box/run.sh full > ~/a26_box/run_full.log 2>&1 &
set -euo pipefail
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
BOX=~/a26_box
bash ~/METOR/studies/a26/box/token_scan.sh "run $1, start" | tee -a "$BOX/token_scans.log"
cd ~/METOR/studies
COMMON=(--data_root ~/ov-vidvrd/data/vidvrd --weights_root ~/ov-vidvrd --out ~/a26_out)
case "$1" in
  check) python -B -m a26.run "${COMMON[@]}" --check_only ;;
  gate)  python -B -m a26.run "${COMMON[@]}" --first_last ;;
  full)  python -B -m a26.run "${COMMON[@]}" ;;
  *) echo "usage: run.sh check|gate|full"; exit 2 ;;
esac
echo "RUN $1 EXIT 0 $(date --iso-8601=seconds)"
