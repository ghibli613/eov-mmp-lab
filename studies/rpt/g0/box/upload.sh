#!/usr/bin/env bash
# Laptop -> G0 machine (copy, never push). Run from anywhere, on a clean tree:
#   bash studies/rpt/g0/box/upload.sh root@HOST PORT
# Copies: this repo as `git archive HEAD` (without studies/d2/data/rec and studies/metor_code, which
# G0 does not use) to ~/ov-vidvrd-lab; SAM 3 as `git archive` of the pinned commit to ~/sam3; and,
# by rsync, the only data G0 reads there: the 5 HELD videos' frames and training annotations, and
# the class and predicate files. No test file. The expected SHA256 of everything goes to ~/rpt_g0_box.
# Weights come from Hugging Face with the author's token (fetch_with_token.sh); CLIP from its public URL.
set -euo pipefail
T="$1"; PORT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../../.." && pwd)"
SAM3="${SAM3_DIR:-$HOME/Documents/thesis/source/sam3}"
SAM3_COMMIT=f1647e8d0c1d280991f4d1b7d974c2704537fe66
SSH="ssh -p $PORT -i ${SSH_KEY:-$HOME/.ssh/vast} -o StrictHostKeyChecking=accept-new"
VIDEOS=(ILSVRC2015_train_00008005 ILSVRC2015_train_00010024 ILSVRC2015_train_00024005
        ILSVRC2015_train_00025029 ILSVRC2015_train_00027003)

[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "!! STOP: the repo has uncommitted changes"; exit 1; }
[ "$(git -C "$SAM3" rev-parse HEAD)" = "$SAM3_COMMIT" ] || { echo "!! STOP: $SAM3 is not at $SAM3_COMMIT"; exit 1; }
COMMIT="$(git -C "$REPO" rev-parse HEAD)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
(
  cd "$REPO"
  for f in object2id.json predicate2id.json id2predicate.json openvoc_obj_class_spilt_info.json \
           openvoc_pred_class_spilt_info.json; do echo "data/vidvrd/data/$f"; done
  for v in "${VIDEOS[@]}"; do
    echo "data/vidvrd/anno/train/$v.json"
    find "data/vidvrd/frames/$v" -name '*.jpg' | sort
  done
) > "$STAGE/upload_files.txt"
if grep -qE '/anno/test/|test_relation_gt|test_object_trajectories' "$STAGE/upload_files.txt"; then
  echo "!! STOP: a test file is on the upload list"; exit 1
fi
( cd "$REPO" && tr '\n' '\0' < "$STAGE/upload_files.txt" | xargs -0 sha256sum ) > "$STAGE/expected_data.sha256"
git -C "$REPO" ls-files -s -- pilot_analysis/stage1c studies/rpt | awk '{print $4}' | grep -v '^studies/rpt/g0/sam3_out' \
  | (cd "$REPO" && xargs sha256sum) > "$STAGE/expected_code.sha256"

$SSH "$T" 'mkdir -p ~/ov-vidvrd-lab ~/sam3 ~/rpt_g0_box/expected'
git -C "$REPO" archive --format=tar "$COMMIT" -- . ':(exclude)studies/d2/data/rec' ':(exclude)studies/metor_code' \
  | $SSH "$T" 'tar -x -C ~/ov-vidvrd-lab'
git -C "$SAM3" archive --format=tar "$SAM3_COMMIT" | $SSH "$T" 'tar -x -C ~/sam3'
rsync -a -e "$SSH" --files-from="$STAGE/upload_files.txt" "$REPO/" "$T:ov-vidvrd-lab/"
rsync -a -e "$SSH" "$STAGE/" "$T:rpt_g0_box/expected/"
cp "$HERE/expected_fetch.sha256" "$STAGE/" 2>/dev/null && rsync -a -e "$SSH" "$STAGE/expected_fetch.sha256" "$T:rpt_g0_box/expected/"
printf '%s\n%s\n' "$COMMIT" "$SAM3_COMMIT" | $SSH "$T" 'cat > ~/rpt_g0_box/commits.txt'
echo "$(date --iso-8601=seconds) upload done: repo $COMMIT, SAM 3 $SAM3_COMMIT, $(wc -l < "$STAGE/upload_files.txt") data files"
