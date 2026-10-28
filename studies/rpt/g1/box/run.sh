#!/usr/bin/env bash
# G1 on the machine, no token present (A29, A29a):
#   bash ~/ov-vidvrd-lab/studies/rpt/g1/box/run.sh tests      # the RPT unit tests, on the machine's CPU (set-up)
#   bash ~/ov-vidvrd-lab/studies/rpt/g1/box/run.sh preflight  # EOV side on 2 smoke videos, on GPU 0 and GPU 1, compared (set-up)
#   bash ~/ov-vidvrd-lab/studies/rpt/g1/box/run.sh checks     # checks 0-5 (the cap's clock starts)
#   bash ~/ov-vidvrd-lab/studies/rpt/g1/box/run.sh videos     # the first 20, the projection, the other 140
#   bash ~/ov-vidvrd-lab/studies/rpt/g1/box/run.sh eov        # EOV-MMP's HELD tracks (R0w, R0) and box sensitivity, all GPUs
#   nohup setsid bash ~/ov-vidvrd-lab/studies/rpt/g1/box/run.sh all > ~/rpt_g1_box/run_all.log 2>&1 &
# `all` runs preflight, checks, videos, eov in order and stops at the first non-zero exit: every
# STOP_*.json in ~/rpt_g1_box is a stop condition of A29; then the author is asked.
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
BOX=~/rpt_g1_box; WORK=~/ov-vidvrd-lab; OUT=~/rpt_g1_out
mkdir -p "$OUT"
cd "$WORK"

step() {
  bash studies/rpt/g0/box/token_scan.sh "run $1, start" | tee -a "$BOX/token_scans.log"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  case "$1" in
    tests) ( cd studies && /venv/main/bin/python -m pytest -q -p no:cacheprovider rpt/tests ) ;;
    preflight) /venv/main/bin/python -B studies/rpt/g1/box/eov_parallel.py preflight ;;
    checks) /venv/main/bin/python -B studies/rpt/g1/box/g1_driver.py checks ;;
    videos) /venv/main/bin/python -B studies/rpt/g1/box/g1_driver.py videos ;;
    eov) /venv/main/bin/python -B studies/rpt/g1/box/eov_parallel.py full ;;
    *) echo "usage: run.sh tests|preflight|checks|videos|eov|all"; exit 2 ;;
  esac
  echo "RUN $1 EXIT 0 $(date --iso-8601=seconds)"
}

if [ "$1" = all ]; then
  for s in tests preflight checks videos eov; do step "$s" 2>&1 | tee -a "$BOX/run_$s.log"; done
else
  step "$1" 2>&1 | tee -a "$BOX/run_$1.log"
fi
