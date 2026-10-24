#!/usr/bin/env bash
# Object check O1 (A22), step 2 -- the 5-video run on a rented GPU box. It STOPS after the
# runtime / cost estimate: the full 800-video run needs approval first.
#
#   git clone https://github.com/ghibli613/eov-mmp-lab.git ~/ov-vidvrd && cd ~/ov-vidvrd
#   export HF_TOKEN=hf_...          # read access to the private weights repo
#   bash pilot_analysis/objects/run_o1_small.sh
#
# Box: RTX 4090 (24 GB), >= 32 GB system RAM, >= 80 GB disk, Vast.ai PyTorch image (as session 3).
# Training videos only. The repo's data step also fetches the test annotations; nothing here
# opens them.
#   0-5  tools/setup_rented_box.sh's own steps, verbatim, cut before its step 6
#   6    preregistration hash + the pins this step uses (A22 scripts, A5 runs)
#   7    the public videos (4.3 GB) -- training frames are decoded per video and deleted after
#   8    5 training videos with novel objects; the repo's own forward is checked against the copy
#   9    runtime / cost / disk estimate for all 800 -- STOP
set -euo pipefail
WORK="${WORK:-$HOME/ov-vidvrd}"
export WORK REPO="${REPO:-https://github.com/ghibli613/eov-mmp-lab.git}"
OUT="${OUT:-$WORK/o1_small}"
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
: "${HF_TOKEN:?set HF_TOKEN first -- the weights repo is private}"

SRC="$WORK/tools/setup_rented_box.sh"
[ -f "$SRC" ] || SRC="$(cd "$(dirname "$0")/../.." && pwd)/tools/setup_rented_box.sh"
PART="$(mktemp)"
awk '/^say "6\./ { exit } { print }' "$SRC" > "$PART"
grep -q '^say "5\.' "$PART" || { echo "!! could not cut setup_rented_box.sh before step 6"; exit 1; }
bash "$PART"
rm -f "$PART"

cd "$WORK"
export HF_HUB_ENABLE_HF_TRANSFER=1 OMP_NUM_THREADS="${THREADS:-8}" MKL_NUM_THREADS="${THREADS:-8}"
mkdir -p "$OUT"
git log --oneline -1 | tee "$OUT/commit.txt"

say "6. pinned-hash check (preregistration + A22 + A5)"
python pilot_analysis/objects/o1_dump.py --verify

say "7. public videos (training frames are decoded per video on the fly)"
python tools/prepare_data.py --steps videos || true

say "8. the 5-video run, with the repo-identity check"
python pilot_analysis/objects/o1_dump.py --out "$OUT" --select-small --check-repo

say "9. estimate for all 800 -- STOP here; the full run needs approval"
python pilot_analysis/objects/o1_dump.py --out "$OUT" --estimate
( cd "$OUT" && sha256sum ./*.pt manifest.jsonl estimate.json videos.json > SHA256SUMS )
echo
echo "Copy the results off the box:"
echo "  scp -P <port> -r root@<host>:$OUT ./o1_small"
