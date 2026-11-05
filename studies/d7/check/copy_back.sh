#!/usr/bin/env bash
# The D7 machine -> laptop, then verify (A34). Run on the laptop:
#   bash studies/d7/check/copy_back.sh root@HOST PORT
# To studies/d7/check/box_out/: every output (check0/, ref35/ if any, armA/, armB/), the manifest, check0.json,
# any STOP file, and the machine's logs. Every output file must match the machine's SHA256 in the manifest.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
D="$HERE/box_out"
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"
mkdir -p "$D/logs"
rsync -a -e "$SSH" --exclude '*.part' "$T:d7_out/" "$D/"
rsync -a -e "$SSH" --exclude expected "$T:d7_box/" "$D/logs/"
cp "$D/logs/check0.json" "$D/check0.json" 2>/dev/null || true
python3 -I - "$D" <<'PY'
import hashlib, json, os, sys
d = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
rows = [json.loads(l) for l in open(os.path.join(d, "manifest.jsonl"))]
bad = [f"{r['stage']}/{r['video']}" for r in rows if sha(os.path.join(d, r["stage"], f"{r['video']}.json.gz")) != r["sha256"]]
print(f"copy-back: {len(rows)} files checked against the machine's SHA256; not matching: {len(bad)} {bad}")
sys.exit(1 if bad or not rows else 0)
PY
