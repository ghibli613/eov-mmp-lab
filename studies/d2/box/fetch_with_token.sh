#!/usr/bin/env bash
# Run by the AUTHOR, interactively, so the token passes through no one else (A25c):
#   ssh -t -p PORT -i ~/.ssh/vast root@HOST 'bash ~/METOR/studies/d2/box/fetch_with_token.sh'
#
# The token is read hidden into HF_TOKEN in this process only. It is never written to a file
# (HF_HOME included), and it is unset, with any token file deleted, before this script ends --
# also on failure. Fetches the eval weights subset (end-to-end checkpoint, AFLink, CLIP bank,
# ECC) and the 200 test videos' frames, each checked by SHA256; then scans for token-shaped
# strings again and checks everything against the laptop's hashes.
set -euo pipefail
[ -f /venv/main/bin/activate ] && source /venv/main/bin/activate
WORK=~/ov-vidvrd; BOX=~/d2_box; D2=~/METOR/studies/d2
WEIGHTS="https://huggingface.co/ghibli613/ov-vidvrd-weights/resolve/main/MANIFEST.json"
FRAMES="https://huggingface.co/datasets/ghibli613/ov-vidvrd-frames/resolve/main"
mkdir -p "$BOX/frames_meta"
LOG="$BOX/fetch.log"

forget() {
  unset HF_TOKEN HUGGINGFACE_HUB_TOKEN
  rm -f ~/.cache/huggingface/token ~/.cache/huggingface/stored_tokens
  [ -n "${HF_HOME:-}" ] && rm -f "$HF_HOME/token" "$HF_HOME/stored_tokens"
  return 0
}
trap forget EXIT

bash "$D2/box/token_scan.sh" "fetch, before the token" | tee -a "$LOG"
read -rsp "Hugging Face token (input hidden): " HF_TOKEN; echo
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=1
echo "$(date --iso-8601=seconds) token held in this process's environment only" >> "$LOG"

cd "$WORK"
python tools/hugging_download.py --manifest "$WEIGHTS" --only eval 2>&1 | tee -a "$LOG"
python - "$FRAMES" "$BOX/frames_meta" <<'PY' 2>&1 | tee -a "$LOG"
import sys; sys.path.insert(0, ".")
from tools.hugging_download import url_file
for f in ("SHARDS.json", "MANIFEST.json"):
    url_file(f"{sys.argv[1]}/{f}", f"{sys.argv[2]}/{f}")
print("frame bundle SHARDS.json and MANIFEST.json saved")
PY
python pilot_analysis/stage1c/session3_tasks.py fetch-test-frames --shards "$FRAMES/SHARDS.json" 2>&1 | tee -a "$LOG"

forget
trap - EXIT
echo "$(date --iso-8601=seconds) token unset; token files removed" | tee -a "$LOG"
bash "$D2/box/token_scan.sh" "fetch, after the token" | tee -a "$LOG"
( cd "$WORK" && sha256sum -c "$D2/box/expected_fetch.sha256" ) 2>&1 | tee -a "$LOG"
echo "frame folders: $(ls -1d data/vidvrd/frames/*/ | wc -l) (want 200)" | tee -a "$LOG"
echo
echo "Done. If every count above is 0 and every file is OK, the token is gone from this machine:"
echo "you can delete it on Hugging Face now."
