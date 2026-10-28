#!/usr/bin/env bash
# The 48 GB machine -> laptop, then verify (A29b). Run on the laptop:
#   bash studies/rpt/g1/box48/copy_back48.sh root@HOST PORT
# To studies/rpt/g1/l40_out/: SAM 3's file for ...00280000, its EOV-MMP file, the checks' and the
# preflight's outputs, and the machine's logs. Every SAM 3 and EOV file must match its manifest.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
D="$(cd "$HERE/.." && pwd)/l40_out"
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"
mkdir -p "$D/sam3_out" "$D/eov_out" "$D/box_out" "$D/logs"
rsync -a -e "$SSH" --exclude '*.part' "$T:rpt_g1_out/sam3/" "$D/sam3_out/"
rsync -a -e "$SSH" --exclude '*.part' "$T:rpt_g1_out/eov/" "$D/eov_out/"
rsync -a -e "$SSH" --exclude sam3 --exclude eov "$T:rpt_g1_out/" "$D/box_out/"
rsync -a -e "$SSH" --exclude expected "$T:rpt_g1_box/" "$D/logs/"
python3 -I - "$D" <<'PY'
import glob, hashlib, json, os, sys
d = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad, n = [], 0
for pattern, folder in (("logs/sam3_manifest.jsonl", "sam3_out"), ("eov_out/manifest*.jsonl", "eov_out")):
    for p in sorted(glob.glob(os.path.join(d, pattern))):
        for line in open(p):
            r = json.loads(line)
            n += 1
            if sha(os.path.join(d, folder, f"{r['video']}.json.gz")) != r["sha256"]:
                bad.append(f"{folder}/{r['video']}")
print(f"copy-off (48 GB machine): {n} files checked against the machine's SHA256; not matching: {len(bad)} {bad}")
sys.exit(1 if bad or not n else 0)
PY
