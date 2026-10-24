#!/usr/bin/env bash
# Counts of Hugging Face token-shaped strings on the machine (A24h / A25c). Prints counts only,
# never a match. Files: /root /workspace /tmp /etc /home (binary files and the frame folders
# skipped); environments: every running process.
#   bash token_scan.sh <label>
pat='hf_[A-Za-z0-9]{30,}'
files=$(grep -rIlE "$pat" /root /workspace /tmp /etc /home 2>/dev/null \
          --exclude-dir=frames --exclude-dir=d2_rec --exclude-dir=_staging | wc -l)
envs=$(grep -alE "$pat" /proc/[0-9]*/environ 2>/dev/null | wc -l)
tokfiles=$(ls -1 ~/.cache/huggingface/token ~/.cache/huggingface/stored_tokens \
             ${HF_HOME:+$HF_HOME/token} ${HF_HOME:+$HF_HOME/stored_tokens} 2>/dev/null | wc -l)
echo "$(date --iso-8601=seconds) token scan [${1:-}]: files=$files process_environments=$envs token_files=$tokfiles"
