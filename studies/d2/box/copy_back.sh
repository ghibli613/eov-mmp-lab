#!/usr/bin/env bash
# Machine -> laptop, then verify (A25c). Run on the laptop:
#   bash studies/d2/box/copy_back.sh root@HOST PORT
# Recordings to studies/d2/data/rec/ (git-ignored), machine logs to studies/d2/logs/box/.
# Every recording must match the manifest's SHA256 and all 200 videos must be there.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
D2="$(cd "$HERE/.." && pwd)"
SSH="ssh -p $PORT -i $HOME/.ssh/vast -o StrictHostKeyChecking=accept-new"
mkdir -p "$D2/data/rec" "$D2/logs/box"
rsync -a -e "$SSH" "$T:d2_rec/" "$D2/data/rec/"
rsync -a -e "$SSH" "$T:d2_box/" "$D2/logs/box/"
python3 - "$D2/data/rec" <<'PY'
import hashlib, json, os, sys
rec = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
man = [json.loads(l) for l in open(os.path.join(rec, "manifest.jsonl"))]
vids = {m["video"] for m in man}
bad = [m["video"] for m in man for e in (".npz", ".json")
       if sha(os.path.join(rec, "test", m["video"] + e)) != m["sha256" + e]]
print(f"manifest: {len(man)} rows, {len(vids)} videos; files not matching: {len(bad)}")
if bad or len(vids) != 200:
    sys.exit("!! STOP: copy-off incomplete or corrupted")
print("copy-off verified: 200 videos, every file matches its SHA256")
PY
