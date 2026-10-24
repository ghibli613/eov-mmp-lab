#!/usr/bin/env bash
# D2 Step 2 on the machine (A25 / A25a / A25c), no token present:
#   bash ~/METOR/studies/d2/box/run.sh check        # pins only
#   bash ~/METOR/studies/d2/box/run.sh first_last   # the 2-video check (ground rule 3, timing)
#   nohup setsid bash ~/METOR/studies/d2/box/run.sh full > ~/d2_box/run_full.log 2>&1 &
set -euo pipefail
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
BOX=~/d2_box
bash ~/METOR/studies/d2/box/token_scan.sh "run $1, start" | tee -a "$BOX/token_scans.log"
cd ~/METOR/studies
case "$1" in
  check)      python -B -m d2.record_run --code_root ~/ov-vidvrd --out ~/d2_rec --check_only ;;
  first_last) python -B -m d2.record_run --code_root ~/ov-vidvrd --out ~/d2_rec --first_last ;;
  full)       python -B -m d2.record_run --code_root ~/ov-vidvrd --out ~/d2_rec ;;
  *) echo "usage: run.sh check|first_last|full"; exit 2 ;;
esac
echo "RUN $1 EXIT 0 $(date --iso-8601=seconds)"
