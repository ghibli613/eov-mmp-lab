#!/usr/bin/env bash
# Laptop -> A26 machine (copy, don't push). Run from anywhere, on a clean METOR tree:
#   bash studies/a26/box/upload.sh root@HOST PORT
# Copies: the EOV code as `git archive fdeeeb4` (fetch tools, CUDA operator) and the data files the
# test path reads (make_expected.py: upload_files.txt) to ~/ov-vidvrd; METOR as `git archive HEAD`
# without studies/d2/data/rec to ~/METOR; the expected hashes to ~/a26_box. Weights, CLIP bank, ECC
# and frames come from Hugging Face (fetch_with_token.sh).
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
STUDIES="$(cd "$HERE/../.." && pwd)"
METOR="$(cd "$STUDIES/.." && pwd)"
EOV="$(cd "$METOR/../ov-vidvrd-lab" && pwd)"
SSH="ssh -p $PORT -i $HOME/.ssh/vast -o StrictHostKeyChecking=accept-new"
[ -z "$(git -C "$METOR" status --porcelain --untracked-files=no)" ] || { echo "!! STOP: METOR has uncommitted changes"; exit 1; }
COMMIT="$(git -C "$METOR" rev-parse HEAD)"
STAGE="$(mktemp -d)"
( cd "$STUDIES" && python3 -B -m a26.box.make_expected "$STAGE" "$COMMIT" )

$SSH "$T" 'mkdir -p ~/ov-vidvrd ~/METOR ~/a26_box'
git -C "$EOV" archive --format=tar fdeeeb4d20145abde48d2f64f9e39f85dcd1fca6 | $SSH "$T" 'tar -x -C ~/ov-vidvrd'
git -C "$METOR" archive --format=tar "$COMMIT" -- . ':(exclude)studies/d2/data/rec' | $SSH "$T" 'tar -x -C ~/METOR'
rsync -a -e "$SSH" --files-from="$STAGE/upload_files.txt" "$EOV/" "$T:ov-vidvrd/"
rsync -a -e "$SSH" "$STAGE/" "$T:a26_box/expected/"
echo "$COMMIT" | $SSH "$T" 'cat > ~/a26_box/metor_commit.txt'
echo "$(date --iso-8601=seconds) upload done: EOV fdeeeb4, METOR $COMMIT, $(wc -l < "$STAGE/upload_files.txt") data files"
rm -rf "$STAGE"
