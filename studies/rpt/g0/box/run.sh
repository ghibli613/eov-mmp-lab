#!/usr/bin/env bash
# G0 on the machine, no token present (A28):
#   bash ~/ov-vidvrd-lab/studies/rpt/g0/box/run.sh step8       # the reproduction (5 videos)
#   bash ~/ov-vidvrd-lab/studies/rpt/g0/box/run.sh sam3-first  # SAM 3 on the first smoke video, then the budget check
#   bash ~/ov-vidvrd-lab/studies/rpt/g0/box/run.sh sam3-rest   # SAM 3 on the other two
#   bash ~/ov-vidvrd-lab/studies/rpt/g0/box/run.sh relations   # adapter -> gen_feats_test + modelC (3 videos)
#   bash ~/ov-vidvrd-lab/studies/rpt/g0/box/run.sh measure     # A28's runtime measurement, last
#   nohup setsid bash ~/ov-vidvrd-lab/studies/rpt/g0/box/run.sh all > ~/rpt_g0_box/run_all.log 2>&1 &
# Budget check (A28): if SAM 3's time on the first smoke video projects the 3 videos beyond 1.25
# GPU-hours, the run stops (exit 3) and the author is asked. The measurement runs only if fewer than
# 95 minutes have passed since setup began, so the machine stays under 2 hours.
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
BOX=~/rpt_g0_box; WORK=~/ov-vidvrd-lab; OUT=~/rpt_g0_out
SMOKE=(ILSVRC2015_train_00008005 ILSVRC2015_train_00010024 ILSVRC2015_train_00024005)
SAM3_SHA=$(awk '$2 ~ /sam3\.pt$/ {print $1}' "$BOX/expected/expected_fetch.sha256")
mkdir -p "$OUT"
cd "$WORK"

sam3() {                                    # sam3 OUT_SUBDIR VIDEO... [runner options]
  local sub="$1"; shift
  ~/sam3env/bin/python -B studies/rpt/sam3_runner.py --checkpoint ~/sam3_ckpt/sam3.pt --sha256 "$SAM3_SHA" \
    --frames "$WORK/data/vidvrd/frames" --out "$OUT/$sub" "$@"
}

step() {
  bash studies/rpt/g0/box/token_scan.sh "run $1, start" | tee -a "$BOX/token_scans.log"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  case "$1" in
    step8) /venv/main/bin/python -B studies/rpt/g0/step8_reproduce.py --out "$OUT" ;;
    sam3-first)
      sam3 sam3_out "${SMOKE[0]}"
      /venv/main/bin/python - "$OUT/sam3_out/${SMOKE[0]}.json.gz" <<'PY'
import gzip, json, sys
d = json.load(gzip.open(sys.argv[1], "rt"))
frames_total = 130 + 64 + 981                       # the 3 smoke videos (A28)
h = d["seconds_total"] / d["frames"] * frames_total / 3600
print(f"SAM 3 on {d['video']}: {d['seconds_total']:.0f} s for {d['frames']} frames x {d['prompts_run']} prompts;"
      f" projected for the 3 smoke videos: {h:.2f} GPU-hours (stop above 1.25)")
sys.exit(3 if h > 1.25 else 0)
PY
      ;;
    sam3-rest) sam3 sam3_out "${SMOKE[1]}" "${SMOKE[2]}" ;;
    relations) /venv/main/bin/python -B studies/rpt/g0/step9_relations.py --sam3 "$OUT/sam3_out" --out "$OUT" ;;
    measure)
      mins=$(( ($(date +%s) - $(cat "$BOX/t0")) / 60 ))
      if [ "$mins" -ge 95 ]; then
        echo "measurement SKIPPED: $mins minutes since setup began (A28: only below 95)"
      else
        echo "measurement: $mins minutes since setup began"
        # determinism control: the default run again, first 3 prompts of smoke video 1
        sam3 sam3_out_control "${SMOKE[0]}" --prompts airplane,antelope,ball
        # the backbone reused across the 35 prompts of smoke video 1
        sam3 sam3_out_reuse "${SMOKE[0]}" --reuse-backbone
      fi ;;
    *) echo "usage: run.sh step8|sam3-first|sam3-rest|relations|measure|all"; exit 2 ;;
  esac
  echo "RUN $1 EXIT 0 $(date --iso-8601=seconds)"
}

if [ "$1" = all ]; then
  for s in step8 sam3-first sam3-rest relations measure; do step "$s" 2>&1 | tee -a "$BOX/run_$s.log"; done
else
  step "$1" 2>&1 | tee -a "$BOX/run_$1.log"
fi
