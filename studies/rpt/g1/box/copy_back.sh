#!/usr/bin/env bash
# Machine -> laptop, then verify. Run on the laptop:
#   bash studies/rpt/g1/box/copy_back.sh root@HOST PORT
# SAM 3's files to studies/rpt/g1/sam3_out/; EOV-MMP's to studies/rpt/g1/eov_out/; the checks' and the
# preflight's outputs to studies/rpt/g1/box_out/; the machine's logs to studies/rpt/g1/logs/box/.
# Every SAM 3 and EOV file must match its manifest's SHA256.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
G1="$(cd "$HERE/.." && pwd)"
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"
mkdir -p "$G1/sam3_out" "$G1/eov_out" "$G1/box_out" "$G1/logs/box"
rsync -a -e "$SSH" --exclude '*.part' "$T:rpt_g1_out/sam3/" "$G1/sam3_out/"
rsync -a -e "$SSH" --exclude '*.part' "$T:rpt_g1_out/eov/" "$G1/eov_out/"
rsync -a -e "$SSH" --exclude sam3 --exclude eov "$T:rpt_g1_out/" "$G1/box_out/"
rsync -a -e "$SSH" --exclude expected "$T:rpt_g1_box/" "$G1/logs/box/"
python3 -I - "$G1" <<'PY'
import hashlib, json, os, sys
g1 = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad, n = [], 0
import glob
for pattern, folder in (("logs/box/sam3_manifest.jsonl", "sam3_out"), ("eov_out/manifest*.jsonl", "eov_out")):
    mans = sorted(glob.glob(os.path.join(g1, pattern)))
    if not mans:
        print(f"(no {pattern})")
        continue
    last = {}
    for p in mans:
        for line in open(p):
            r = json.loads(line)
            last[r["video"]] = r["sha256"]
    for v, h in last.items():
        n += 1
        if sha(os.path.join(g1, folder, f"{v}.json.gz")) != h:
            bad.append(f"{folder}/{v}")
print(f"copy-off: {n} files checked against the machine's SHA256; not matching: {len(bad)} {bad[:3]}")
sys.exit(1 if bad else 0)
PY
