#!/usr/bin/env bash
# Right after the run: delete the test annotations from the machine and log it -- each file's
# SHA256 is recorded before deletion, then its absence is checked -- and scan for tokens.
#   bash ~/METOR/studies/a26/box/cleanup.sh
set -uo pipefail
cd ~/ov-vidvrd/data/vidvrd || exit 1
LOG=~/a26_box/cleanup.log
mapfile -t files < <(find data -maxdepth 1 \( -name test_relation_gt.json -o -name test_object_trajectories_gt.json \
                        -o -name test_object_trajectories_meta.json \) -type f; find anno/test -name '*.json' -type f 2>/dev/null)
{
  echo "$(date --iso-8601=seconds) deleting ${#files[@]} test annotation files from $(pwd)"
  if [ "${#files[@]}" -gt 0 ]; then
    sha256sum "${files[@]}" > ~/a26_box/deleted_test_annotations.sha256
    rm -f -- "${files[@]}"
  fi
  left=$( { find data -maxdepth 1 -name 'test_*' -type f; find anno/test -type f 2>/dev/null; } | wc -l)
  echo "$(date --iso-8601=seconds) deleted; test files left: $left"
  bash ~/METOR/studies/a26/box/token_scan.sh "cleanup"
} 2>&1 | tee -a "$LOG"
