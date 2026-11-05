#!/usr/bin/env bash
# The D7 machine (A34), after setup.sh and the author's fetch_sam3_with_token.sh:
#   nohup setsid bash ~/ov-vidvrd-lab/studies/d7/check/run.sh > ~/d7_box/run.log 2>&1 &
# check 0 (G0's smoke video 2, bitwise against G0) -> the 35-name reference on the list, only if check 0
# is not bitwise -> arm A and arm B. One SAM 3 process at a time on the one GPU. Any failed video stops
# the run (no skips); a rerun of this script resumes after the finished videos.
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd ~/ov-vidvrd-lab
PY=~/sam3env/bin/python
for s in check0 ref35 arms; do
  bash studies/rpt/g0/box/token_scan.sh "d7 $s" | tee -a ~/d7_box/token_scans.log
  $PY -B studies/d7/check/d7_run.py "$s"
  echo "D7 $s EXIT 0 $(date --iso-8601=seconds)"
done
echo D7-END
