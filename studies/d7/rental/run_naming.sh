#!/usr/bin/env bash
# The naming check on the rented machine (A33, A33a, A33b), after G1's setup.sh (G1_MACHINE=l40) and the
# author's fetch_with_token.sh:
#   nohup setsid bash ~/ov-vidvrd-lab/studies/d7/rental/run_naming.sh > ~/d7_box/naming_run.log 2>&1 &
# /venv/main is G0's EOV-MMP stack (the one G1's R0w ran on). Resumable per video; outputs in
# ~/ov-vidvrd-lab/studies/d7/naming_check/.
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd ~/ov-vidvrd-lab
bash studies/rpt/g0/box/token_scan.sh "naming check" | tee -a ~/d7_box/token_scans.log
mkdir -p studies/d7/naming_check
/venv/main/bin/python -u studies/d7/naming_check.py run 2>&1 | tee -a studies/d7/naming_check/run.txt
echo "NAMING-END $(date --iso-8601=seconds)"
