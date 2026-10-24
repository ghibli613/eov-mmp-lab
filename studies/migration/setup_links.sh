#!/usr/bin/env bash
# Recreate METOR's local data/ and output/ inside studies/metor_code, as they were in the METOR
# repo: data/{vidvrd,gt_jsons} and the six output/ckpt/*.pth are links into ov-vidvrd-lab's
# data/ and output/; output/log is a real folder. All of it is git-ignored by metor_code/.gitignore.
#
#   bash studies/migration/setup_links.sh
set -euo pipefail
M="$(cd "$(dirname "$0")/../metor_code" && pwd)"
mkdir -p "$M/data" "$M/output/ckpt" "$M/output/log"
for d in vidvrd gt_jsons; do
  ln -sfn "../../../data/$d" "$M/data/$d"
done
for f in AFLink_epoch20.pth \
         baseline_fbce_vidvrd_bs1_lr0.0001_drop0.5_dim512_none_rel_mot_clip_bbox_stage2_new_L14_e2e.pth \
         baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_end2end_base-001.pth \
         baseline_fbce_vidvrd_bs1_lr1e-05_dim512_none_rel_mot_clip_bbox_train42-003.pth \
         checkpoint_vidvrd0059_new_1e-5.pth \
         "vidvrd_backboneViT-L_14@336px_lr0.01vision-guided.pth"; do
  ln -sfn "../../../../output/ckpt/$f" "$M/output/ckpt/$f"
done
for p in "$M"/data/* "$M"/output/ckpt/*; do
  [ -e "$p" ] || { echo "!! broken link: $p -> $(readlink "$p")"; exit 1; }
done
echo "links ok under $M (data: $(ls "$M/data" | wc -l), output/ckpt: $(ls "$M/output/ckpt" | wc -l))"
