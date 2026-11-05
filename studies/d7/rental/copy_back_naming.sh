#!/usr/bin/env bash
# The rented machine -> laptop, then verify (A33b). Run on the laptop:
#   bash studies/d7/rental/copy_back_naming.sh root@HOST PORT
# To studies/d7/naming_check/: the 78 per-video outputs, the manifest and the run log; and the machine's
# setup, fetch and auto-stop logs to studies/d7/rental/box_logs/. Every output must match the manifest's SHA256.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
D="$(cd "$HERE/.." && pwd)/naming_check"
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"
mkdir -p "$D/videos" "$HERE/box_logs"
rsync -a -e "$SSH" --exclude '*.part' "$T:ov-vidvrd-lab/studies/d7/naming_check/videos/" "$D/videos/"
rsync -a -e "$SSH" "$T:ov-vidvrd-lab/studies/d7/naming_check/manifest.jsonl" "$T:ov-vidvrd-lab/studies/d7/naming_check/run.txt" "$D/"
rsync -a -e "$SSH" --exclude expected "$T:rpt_g1_box/" "$HERE/box_logs/rpt_g1_box/"
rsync -a -e "$SSH" "$T:d7_box/" "$HERE/box_logs/d7_box/"
python3 -I - "$D" <<'PY'
import hashlib, json, os, sys
d = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rows = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
bad = [r["video"] for r in rows if sha(os.path.join(d, "videos", f"{r['video']}.json.gz")) != r["sha256"]]
print(f"copy-back: {len(rows)} of 78 outputs checked against the machine's SHA256; not matching: {len(bad)} {bad}")
sys.exit(1 if bad or len(rows) != 78 else 0)
PY
