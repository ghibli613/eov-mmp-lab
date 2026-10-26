#!/usr/bin/env bash
# Machine -> laptop, then verify. Run on the laptop:
#   bash studies/rpt/g0/box/copy_back.sh root@HOST PORT
# SAM 3's files to studies/rpt/g0/sam3_out/; step 8 and step 9 outputs to studies/rpt/g0/box_out/;
# the machine's logs to studies/rpt/g0/logs/box/. Every file must match its manifest's SHA256.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
G0="$(cd "$HERE/.." && pwd)"
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"
mkdir -p "$G0/sam3_out" "$G0/box_out" "$G0/logs/box"
rsync -a -e "$SSH" "$T:rpt_g0_out/sam3_out/" "$G0/sam3_out/"
rsync -a -e "$SSH" --exclude sam3_out "$T:rpt_g0_out/" "$G0/box_out/"
rsync -a -e "$SSH" --exclude expected "$T:rpt_g0_box/" "$G0/logs/box/"
python3 -I - "$G0" <<'PY'
import hashlib, json, os, sys
g0 = sys.argv[1]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad, n = [], 0
for step in ("step8", "step9"):
    man = json.load(open(os.path.join(g0, "box_out", step, "manifest.json")))
    for row in man["videos"]:
        n += 1
        if sha(os.path.join(g0, "box_out", row["file"])) != row["sha256"]:
            bad.append(row["file"])
logs = os.path.join(g0, "logs", "box")
for name in sorted(os.listdir(logs)):
    if name.startswith("run_") and name.endswith(".log"):
        for line in open(os.path.join(logs, name)):
            if line.startswith("{") and '"sha256"' in line and '"file"' in line and '"video"' in line:
                row = json.loads(line)
                rel = row["file"].split("rpt_g0_out/", 1)[-1]          # sam3_out/..., sam3_out_reuse/..., ...
                if rel.startswith("sam3_out"):
                    n += 1
                    p = os.path.join(g0, rel) if rel.startswith("sam3_out/") else os.path.join(g0, "box_out", rel)
                    if sha(p) != row["sha256"]:
                        bad.append(p)
print(f"copy-off: {n} files checked against the machine's SHA256; not matching: {len(bad)} {bad[:3]}")
sys.exit(1 if bad else 0)
PY
