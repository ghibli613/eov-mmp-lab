#!/usr/bin/env bash
# Run by the AUTHOR, interactively, so the token passes through no one else (A28, as A29 keeps it; as A25c and A26):
#   ssh -t -p PORT -i ~/.ssh/vast root@HOST 'bash ~/ov-vidvrd-lab/studies/rpt/g1/box/fetch_with_token.sh'
#
# The read-only token is read hidden into HF_TOKEN in this process only. HF_HOME is a fresh folder,
# so no token file is read or written; it is deleted, and HF_TOKEN unset, before this script ends,
# also on failure. Fetches the end-to-end checkpoint (the private weights repo, `--only eval`) and
# SAM 3's checkpoint (facebook/sam3 at the pinned revision); then scans for token-shaped strings and
# checks both files against the hashes A28 pins.
set -euo pipefail
source /venv/main/bin/activate
WORK=~/ov-vidvrd-lab; BOX=~/rpt_g1_box
WEIGHTS="https://huggingface.co/ghibli613/ov-vidvrd-weights/resolve/main/MANIFEST.json"
SAM3_REV=3c879f39826c281e95690f02c7821c4de09afae7
LOG="$BOX/fetch.log"
HFH="$(mktemp -d)"

forget() {
  unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
  rm -rf "$HFH"
  rm -f ~/.cache/huggingface/token ~/.cache/huggingface/stored_tokens
  return 0
}
trap forget EXIT

bash "$WORK/studies/rpt/g0/box/token_scan.sh" "fetch, before the token" | tee -a "$LOG"
read -rsp "Hugging Face read-only token (input hidden): " HF_TOKEN; echo
export HF_TOKEN HF_HOME="$HFH" HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_TELEMETRY=1
echo "$(date --iso-8601=seconds) token held in this process's environment only" >> "$LOG"

cd "$WORK"
python tools/hugging_download.py --manifest "$WEIGHTS" --only eval 2>&1 | tee -a "$LOG"
mkdir -p ~/sam3_ckpt
python - "$SAM3_REV" <<'PY' 2>&1 | tee -a "$LOG"
import os, sys
from huggingface_hub import hf_hub_download
for name in ("sam3.pt", "config.json"):
    print(hf_hub_download("facebook/sam3", name, revision=sys.argv[1], local_dir=os.path.expanduser("~/sam3_ckpt")))
PY

forget
trap - EXIT
echo "$(date --iso-8601=seconds) token unset; temporary HF_HOME and any token file removed" | tee -a "$LOG"
bash "$WORK/studies/rpt/g0/box/token_scan.sh" "fetch, after the token" | tee -a "$LOG"
sha256sum -c "$BOX/expected/expected_fetch.sha256" 2>&1 | tee -a "$LOG"
echo
echo "Done. If every count above is 0 and every file is OK, the token is gone from this machine:"
echo "you can delete it on Hugging Face now."
