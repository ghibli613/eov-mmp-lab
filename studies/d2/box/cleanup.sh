#!/usr/bin/env bash
# Right after the run (A25c): delete the test annotations from the machine and log it -- each
# file's SHA256 is recorded before deletion, then its absence is checked -- and scan for tokens.
#   bash ~/METOR/studies/d2/box/cleanup.sh
set -euo pipefail
cd ~/ov-vidvrd/data/vidvrd
LOG=~/d2_box/cleanup.log
files=$(ls data/test_relation_gt.json data/test_object_trajectories_gt.json \
           data/test_object_trajectories_meta.json anno/test/*.json 2>/dev/null || true)
n=$(echo "$files" | grep -c . || true)
{
  echo "$(date --iso-8601=seconds) deleting $n test annotation files from $(pwd)"
  [ "$n" -gt 0 ] && sha256sum $files > ~/d2_box/deleted_test_annotations.sha256
  [ "$n" -gt 0 ] && rm -f $files
  left=$(ls data/test_relation_gt.json data/test_object_trajectories_gt.json \
            data/test_object_trajectories_meta.json anno/test/*.json 2>/dev/null | wc -l)
  echo "$(date --iso-8601=seconds) deleted; files left: $left"
  bash ~/METOR/studies/d2/box/token_scan.sh "cleanup"
} 2>&1 | tee -a "$LOG"
