#!/usr/bin/env bash
# Laptop -> the rented machine, mid-rental (A34): after A34, the video list and the guard are committed.
# Sends the D7 check's committed code and video list (git archive), and checks them on the machine.
# The frames are already there: the list's videos are among the 78 the naming check uploaded.
#   bash studies/d7/rental/upload_d7.sh root@HOST PORT
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "!! STOP: the repo has uncommitted changes"; exit 1; }
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
FILES=$(git -C "$REPO" ls-files -- studies/d7/check | grep -v '^studies/d7/check/tests/')
git -C "$REPO" archive --format=tar "$COMMIT" -- $FILES | $SSH "$T" 'tar -x -C ~/ov-vidvrd-lab'
( cd "$REPO" && sha256sum $FILES ) | $SSH "$T" 'cd ~/ov-vidvrd-lab && sha256sum -c --quiet && echo "D7 code and list: all match"'
$SSH "$T" "cd ~/ov-vidvrd-lab && python3 -c \"import json; v=json.load(open('studies/d7/check/video_list.json'))['final']; import os; m=[x for x in v if not os.path.isdir('data/vidvrd/frames/'+x)]; print(len(v), 'videos;', 'frames missing for', m) ; exit(1 if m else 0)\""
echo "$COMMIT" | $SSH "$T" 'cat >> ~/rpt_g1_box/commits.txt'
