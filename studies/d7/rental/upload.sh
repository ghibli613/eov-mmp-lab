#!/usr/bin/env bash
# Laptop -> the rented 48 GB machine (A33b), copy only, never push. Run on a clean tree:
#   bash studies/d7/rental/upload.sh root@HOST PORT
# The lists come from upload_lists.py (explicit code folders, the 78 HELD-A videos' SAM 3 files,
# annotations and frames; no HELD-B and no test file). The code goes as `git archive` of exactly those
# committed files; SAM 3 as `git archive` of its pinned commit; the data by rsync. The expected SHA256
# and G0's package lists go to ~/rpt_g1_box/expected, where G1's setup.sh and fetch_with_token.sh read them.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SAM3="${SAM3_DIR:-$HOME/Documents/thesis/source/sam3}"
SAM3_COMMIT=f1647e8d0c1d280991f4d1b7d974c2704537fe66
PY="${PY:-$HOME/miniconda3/envs/repro-next/bin/python}"
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"

[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "!! STOP: the repo has uncommitted changes"; exit 1; }
[ "$(git -C "$SAM3" rev-parse HEAD)" = "$SAM3_COMMIT" ] || { echo "!! STOP: $SAM3 is not at $SAM3_COMMIT"; exit 1; }
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
( cd "$REPO" && "$PY" studies/d7/rental/upload_lists.py "$STAGE" )

$SSH "$T" 'mkdir -p ~/ov-vidvrd-lab ~/sam3 ~/rpt_g1_box/expected ~/d7_box'
tr '\n' '\0' < "$STAGE/code_files.txt" | xargs -0 git -C "$REPO" archive --format=tar "$COMMIT" -- \
  | $SSH "$T" 'tar -x -C ~/ov-vidvrd-lab'
git -C "$SAM3" archive --format=tar "$SAM3_COMMIT" | $SSH "$T" 'tar -x -C ~/sam3'
rsync -a -e "$SSH" --files-from="$STAGE/data_files.txt" "$REPO/" "$T:ov-vidvrd-lab/"
rsync -a -e "$SSH" "$STAGE/expected_code.sha256" "$STAGE/expected_data.sha256" "$STAGE/pip_freeze_main.txt" \
  "$STAGE/pip_freeze_sam3env.txt" "$STAGE/expected_fetch.sha256" "$T:rpt_g1_box/expected/"
printf '%s\n%s\n' "$COMMIT" "$SAM3_COMMIT" | $SSH "$T" 'cat > ~/rpt_g1_box/commits.txt'
echo "$(date --iso-8601=seconds) upload done: repo $COMMIT, SAM 3 $SAM3_COMMIT, $(wc -l < "$STAGE/data_files.txt") data files"
