#!/usr/bin/env bash
# The 48 GB machine (A29b), after G1_MACHINE=l40 setup.sh and the author's fetch_with_token.sh:
#   nohup setsid bash ~/ov-vidvrd-lab/studies/rpt/g1/box48/run48.sh > ~/rpt_g1_box/run48.log 2>&1 &
# check 0 (stop and ask if not identical to G0) -> ...00280000 -> the EOV-MMP preflight against the
# 8x machine's -> the EOV-MMP side for ...00280000 (box sensitivity only if the preflight matched).
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd ~/ov-vidvrd-lab
PY=/venv/main/bin/python
f=data/vidvrd/data/VidVRD_ECC_test.json                 # fetched with the eval weights; never used (as on the 8x machine)
if [ -f "$f" ]; then
  echo "$(date --iso-8601=seconds) removed $f (sha256 $(sha256sum < "$f" | cut -c1-64))" | tee -a ~/rpt_g1_box/test_file_removed.log
  rm -f "$f"
fi
$PY -c "import MultiScaleDeformableAttention" 2>/dev/null || bash studies/rpt/g1/box/build_ops.sh
for s in check0 video preflight eov; do
  bash studies/rpt/g0/box/token_scan.sh "run48 $s" | tee -a ~/rpt_g1_box/token_scans.log
  $PY -B studies/rpt/g1/box48/l40_run.py "$s"
  echo "RUN48 $s EXIT 0 $(date --iso-8601=seconds)"
done
echo RUN48-END
