#!/usr/bin/env bash
# Laptop -> D2 machine (A25c: copy, don't push). Run from anywhere:
#   bash studies/d2/box/upload.sh root@HOST PORT
# Copies: the EOV code exactly as `git archive fdeeeb4`; the metadata and test annotations the
# released test path reads (upload_files.txt); and METOR's studies/ (pre-registration, d2 code,
# baseline tracks). Weights, CLIP bank, ECC and frames come from Hugging Face (fetch_with_token.sh).
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
STUDIES="$(cd "$HERE/../.." && pwd)"
EOV="$(cd "$STUDIES/../../ov-vidvrd-lab" && pwd)"
SSH="ssh -p $PORT -i $HOME/.ssh/vast -o StrictHostKeyChecking=accept-new"

$SSH "$T" 'mkdir -p ~/ov-vidvrd ~/METOR ~/d2_box'
git -C "$EOV" archive --format=tar fdeeeb4d20145abde48d2f64f9e39f85dcd1fca6 | $SSH "$T" 'tar -x -C ~/ov-vidvrd'
rsync -a -e "$SSH" --files-from="$HERE/upload_files.txt" "$EOV/" "$T:ov-vidvrd/"
rsync -a -e "$SSH" --exclude '__pycache__' --exclude 'd2/data/rec' "$STUDIES/" "$T:METOR/studies/"
echo "$(date --iso-8601=seconds) upload done: code fdeeeb4, $(wc -l < "$HERE/upload_files.txt") data files, studies/"
