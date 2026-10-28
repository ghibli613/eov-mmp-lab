#!/usr/bin/env bash
# Laptop -> G1 machine (copy, never push). Run from anywhere, on a clean tree:
#   bash studies/rpt/g1/box/upload.sh root@HOST PORT
# Copies: this repo as `git archive HEAD` (without studies/d2/data/rec and studies/metor_code) to
# ~/ov-vidvrd-lab; SAM 3 as `git archive` of the pinned commit to ~/sam3; and, by rsync, the data G1
# reads: the 160 HELD videos' frames and training annotations, HELD's base GT relations, and the class,
# predicate, prior, CLIP text-feature, train-trajectory and train-ECC files EOV-MMP's front end reads.
# No test file (the test ECC file is not sent: the model then starts with an empty test_ecc, which
# eov_held.py replaces with the train ECC, A29 item 1). Expected SHA256 of everything to ~/rpt_g1_box.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
SAM3="${SAM3_DIR:-$HOME/Documents/thesis/source/sam3}"
SAM3_COMMIT=f1647e8d0c1d280991f4d1b7d974c2704537fe66
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"

[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "!! STOP: the repo has uncommitted changes"; exit 1; }
[ "$(git -C "$SAM3" rev-parse HEAD)" = "$SAM3_COMMIT" ] || { echo "!! STOP: $SAM3 is not at $SAM3_COMMIT"; exit 1; }
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
(
  cd "$REPO"
  for f in object2id.json predicate2id.json id2predicate.json id2object.json openvoc_obj_class_spilt_info.json \
           openvoc_pred_class_spilt_info.json prior.pkl clip_L14_feat_vidvrd.pkl train_object_trajectories_gt.json \
           VidVRD_ECC_train.json; do echo "data/vidvrd/data/$f"; done
  echo "pilot_analysis/stage2/data/held_gt.json"
  python3 -c "import json; print('\n'.join(sorted(json.load(open('pilot_analysis/stage1/train_split.json'))['held'])))" \
  | while read -r v; do
    echo "data/vidvrd/anno/train/$v.json"
    find "data/vidvrd/frames/$v" -name '*.jpg' | sort
  done
) > "$STAGE/upload_files.txt"
[ "$(grep -c '^data/vidvrd/anno/train/' "$STAGE/upload_files.txt")" = 160 ] || { echo "!! STOP: not 160 HELD annotations"; exit 1; }
if grep -qE '/anno/test/|test_relation_gt|test_object_trajectories|ECC_test|/test_' "$STAGE/upload_files.txt"; then
  echo "!! STOP: a test file is on the upload list"; exit 1
fi
( cd "$REPO" && tr '\n' '\0' < "$STAGE/upload_files.txt" | xargs -0 sha256sum ) > "$STAGE/expected_data.sha256"
git -C "$REPO" ls-files -s -- pilot_analysis/stage1c pilot_analysis/stage1/train_split.json studies/rpt studies/d2/*.py \
    studies/d2/logs/code_pins.json studies/a26/*.py \
  | awk '{print $4}' | grep -vE '^studies/rpt/g[01]/(sam3_out|eov_out|box_out|results)/' \
  | (cd "$REPO" && xargs sha256sum) > "$STAGE/expected_code.sha256"

$SSH "$T" 'mkdir -p ~/ov-vidvrd-lab ~/sam3 ~/rpt_g1_box/expected'
git -C "$REPO" archive --format=tar "$COMMIT" -- . ':(exclude)studies/d2/data/rec' ':(exclude)studies/metor_code' \
  | $SSH "$T" 'tar -x -C ~/ov-vidvrd-lab'
git -C "$SAM3" archive --format=tar "$SAM3_COMMIT" | $SSH "$T" 'tar -x -C ~/sam3'
rsync -a -e "$SSH" --files-from="$STAGE/upload_files.txt" "$REPO/" "$T:ov-vidvrd-lab/"
cp "$HERE/expected_fetch.sha256" "$REPO/studies/rpt/g0/logs/box/pip_freeze_main.txt" \
   "$REPO/studies/rpt/g0/logs/box/pip_freeze_sam3env.txt" "$STAGE/"
rsync -a -e "$SSH" "$STAGE/" "$T:rpt_g1_box/expected/"
printf '%s\n%s\n' "$COMMIT" "$SAM3_COMMIT" | $SSH "$T" 'cat > ~/rpt_g1_box/commits.txt'
echo "$(date --iso-8601=seconds) upload done: repo $COMMIT, SAM 3 $SAM3_COMMIT, $(wc -l < "$STAGE/upload_files.txt") data files"
