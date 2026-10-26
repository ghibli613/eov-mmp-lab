#!/usr/bin/env bash
# RPT G0: fetch SAM 3's checkpoint (facebook/sam3, gated) with the AUTHOR's read-only token.
# Run by the author, in their own terminal, so the token passes through no one else:
#   bash studies/rpt/g0/fetch_sam3_ckpt.sh [DEST]      # DEST defaults to ~/Documents/thesis/source/sam3_ckpt
#
# The token is read hidden into HF_TOKEN in this process only. HF_HOME points at a fresh empty
# folder, so no stored token file (e.g. ~/.cache/huggingface/token) is read or written; that folder
# is deleted and HF_TOKEN unset when the script ends, also on failure. The revision is pinned to
# the November 2025 SAM 3 release (not SAM 3.1). Prints the size and SHA256 that the G0 amendment
# pins.
set -euo pipefail
REV=3c879f39826c281e95690f02c7821c4de09afae7
DEST="${1:-$HOME/Documents/thesis/source/sam3_ckpt}"
PY="${SAM3_PY:-$HOME/miniconda3/envs/sam3/bin/python}"
HFH="$(mktemp -d)"

forget() {
  unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
  rm -rf "$HFH"
  return 0
}
trap forget EXIT

read -rsp "Hugging Face read-only token (input hidden): " HF_TOKEN; echo
export HF_TOKEN HF_HOME="$HFH" HF_HUB_DISABLE_TELEMETRY=1
mkdir -p "$DEST"
"$PY" - "$DEST" "$REV" <<'PY'
import sys
from huggingface_hub import hf_hub_download
for name in ("sam3.pt", "config.json"):
    print(hf_hub_download("facebook/sam3", name, revision=sys.argv[2], local_dir=sys.argv[1]))
PY
forget
trap - EXIT
echo "$(date --iso-8601=seconds) token unset; temporary HF_HOME removed"
cd "$DEST"
stat -c '%s bytes  %n' sam3.pt config.json
sha256sum sam3.pt config.json | tee SHA256SUMS
echo "facebook/sam3 revision $REV" > REVISION
