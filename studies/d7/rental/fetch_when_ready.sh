#!/usr/bin/env bash
# Run by the AUTHOR in tmux on the rented machine (A33b), so the token is typed once, early:
#   tmux new -s fetch 'bash ~/d7_tools/fetch_when_ready.sh'      # type the token, then detach: Ctrl-b d
# The token is read hidden into a shell variable of this process only: not exported, not written, never
# shown. The script waits until G1's setup.sh reports "setup done", then hands the token to G1's
# unchanged fetch_with_token.sh on its standard input and unsets it. If setup stops, the token is
# discarded and nothing is fetched.
set -euo pipefail
LOG=~/rpt_g1_box/setup.log
read -rsp "Hugging Face read-only token (input hidden): " TOK; echo
echo "$(date --iso-8601=seconds) token held in memory; waiting for setup to finish (you can detach: Ctrl-b d)"
until grep -q "setup done" "$LOG" 2>/dev/null; do
  if grep -q "!! STOP" "$LOG" 2>/dev/null; then
    unset TOK
    echo "$(date --iso-8601=seconds) setup stopped: the token is discarded, nothing fetched"
    exit 1
  fi
  sleep 15
done
printf '%s\n' "$TOK" | bash ~/ov-vidvrd-lab/studies/rpt/g1/box/fetch_with_token.sh
unset TOK
echo "$(date --iso-8601=seconds) fetch finished; token unset"
