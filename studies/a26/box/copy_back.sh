#!/usr/bin/env bash
# Machine -> laptop, then verify. Run on the laptop:
#   bash studies/a26/box/copy_back.sh root@HOST PORT
# Predictions to studies/a26/data/rec/ (git-ignored), machine logs to studies/a26/logs/box/.
# Every file must match the manifest's SHA256, for all 200 videos and the four configurations.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
A26="$(cd "$HERE/.." && pwd)"
SSH="ssh -p $PORT -i $HOME/.ssh/vast -o StrictHostKeyChecking=accept-new"
mkdir -p "$A26/data/rec" "$A26/logs/box"
rsync -a -e "$SSH" "$T:a26_out/" "$A26/data/rec/"
rsync -a -e "$SSH" --exclude expected "$T:a26_box/" "$A26/logs/box/"
python3 - "$A26/data/rec" <<'PY'
import hashlib, json, os, sys
rec = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = [json.loads(l) for l in open(os.path.join(rec, "manifest.jsonl"))]
vids = {m["video"] for m in man}
bad = [(m["video"], n) for m in man for n, h in m["sha256"].items()
       if sha(os.path.join(rec, n, m["video"] + ".json.gz")) != h]
n_cfg = {len(m["sha256"]) for m in man}
print(f"manifest: {len(man)} rows, {len(vids)} videos, configurations per row {n_cfg}; files not matching: {len(bad)}")
if bad or len(vids) != 200 or len(man) != 200 or n_cfg != {4}:
    sys.exit("!! STOP: copy-off incomplete or corrupted")
print("copy-off verified: 200 videos x 4 configurations, every file matches its SHA256")
PY
