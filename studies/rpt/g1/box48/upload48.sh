#!/usr/bin/env bash
# Laptop -> the 48 GB machine (A29b), on a clean tree, after the 8x machine's copy-back:
#   bash studies/rpt/g1/box48/upload48.sh root@HOST PORT
# The 8x upload (studies/rpt/g1/box/upload.sh) restricted to the 3 videos A29b uses (smoke videos 1
# and 2 for check 0 and the preflight, and ...00280000), plus the 8x machine's preflight outputs.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
export G1_VIDEOS_CMD="printf %s\n ILSVRC2015_train_00008005 ILSVRC2015_train_00010024 ILSVRC2015_train_00280000"
export G1_EXTRA_FILES="studies/rpt/g1/box_out/preflight_eov/g0/ILSVRC2015_train_00008005.json.gz studies/rpt/g1/box_out/preflight_eov/g0/ILSVRC2015_train_00010024.json.gz"
exec bash "$HERE/../box/upload.sh" "$@"
