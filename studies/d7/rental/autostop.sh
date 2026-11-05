#!/usr/bin/env bash
# The rental's auto-stop (A33b): a watchdog stops this Vast.ai instance through its per-instance key once a
# deadline passes. Stopping ends GPU billing and keeps the disk. The Vast CLI goes in its own venv
# (~/vastcli), so neither pinned environment changes. The key and the instance ID are read from the
# container's environment (/proc/1/environ); the key is never printed.
#   bash autostop.sh arm HOURS          # install, test with a no-op start, set the deadline, start the watchdog
#   bash autostop.sh extend HOURS       # move the deadline to now + HOURS
#   bash autostop.sh status
set -euo pipefail
BOX=~/d7_box; DL=$BOX/autostop_deadline; LOG=$BOX/autostop.log
mkdir -p "$BOX"
envof() { tr '\0' '\n' < /proc/1/environ | sed -n "s/^$1=//p" | head -1; }
export CONTAINER_ID="${CONTAINER_ID:-$(envof CONTAINER_ID)}"
export CONTAINER_API_KEY="${CONTAINER_API_KEY:-$(envof CONTAINER_API_KEY)}"
now() { date +%s; }
case "${1:-status}" in
  arm)
    [ -n "$CONTAINER_ID" ] || { echo "!! no CONTAINER_ID: the auto-stop cannot work; set a stop in the Vast console"; exit 1; }
    [ -n "$CONTAINER_API_KEY" ] || { echo "!! no CONTAINER_API_KEY: the auto-stop cannot work; set a stop in the Vast console"; exit 1; }
    [ -x ~/vastcli/bin/vastai ] || { python3 -m venv ~/vastcli && ~/vastcli/bin/pip install -q vastai; }
    ~/vastcli/bin/vastai start instance "$CONTAINER_ID" >/dev/null 2>&1 \
      || { echo "!! the instance key does not work (no-op start failed): set a stop in the Vast console"; exit 1; }
    echo $(( $(now) + ${2:?hours} * 3600 )) > "$DL"
    nohup setsid bash -c '
      while true; do
        if [ "$(date +%s)" -ge "$(cat '"$DL"')" ]; then
          echo "$(date --iso-8601=seconds) deadline reached: stopping instance $CONTAINER_ID" >> '"$LOG"'
          ~/vastcli/bin/vastai stop instance "$CONTAINER_ID" >> '"$LOG"' 2>&1
          sleep 600
        fi
        sleep 60
      done' > /dev/null 2>&1 &
    echo "$(date --iso-8601=seconds) armed: deadline $(date -d @"$(cat "$DL")" --iso-8601=seconds), watchdog pid $!" | tee -a "$LOG"
    ;;
  extend)
    echo $(( $(now) + ${2:?hours} * 3600 )) > "$DL"
    echo "$(date --iso-8601=seconds) deadline moved to $(date -d @"$(cat "$DL")" --iso-8601=seconds)" | tee -a "$LOG"
    ;;
  status)
    [ -f "$DL" ] && echo "deadline $(date -d @"$(cat "$DL")" --iso-8601=seconds); now $(date --iso-8601=seconds)" || echo "not armed"
    pgrep -f "vastai stop instance" >/dev/null && echo "a stop is in progress" || true
    ;;
esac
