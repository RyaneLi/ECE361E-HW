#!/bin/zsh
set -euo pipefail

ROOT="/Users/slrpz/Downloads/ECE361E/ece_361e_fl"
STATE_FILE="$ROOT/logs/job_queue_state.json"
LOG_FILE="$ROOT/logs/m7_finalists_pipeline.log"

exec >>"$LOG_FILE" 2>&1

echo "[$(date)] Waiting for exp370 run2 to finish before launching M7 finalists."

cd "$ROOT"
source "$ROOT/.autolaunch_env"
source /Users/slrpz/miniconda3/etc/profile.d/conda.sh
conda activate ece_361e_fl

while true; do
  last_successful=$(python - <<'PY'
import json
from pathlib import Path
state = json.loads(Path("logs/job_queue_state.json").read_text())
last = state.get("last_successful") or {}
print(f"{last.get('exp', '')}:{last.get('run', '')}")
PY
)

  if [[ "$last_successful" == "370:2" ]]; then
    break
  fi

  echo "[$(date)] Still waiting. last_successful=$last_successful"
  sleep 30
done

echo "[$(date)] Launching M7 finalists."
python start_tmux_device_managers.py --devices rpi mc1
MPLCONFIGDIR='/tmp/mplconfig' python -u run_experiment_queue.py \
  --skip_manager_start \
  --config_script config_m7_finalists.bash \
  --start_exp 371 \
  --end_exp 382 \
  --manager_boot_timeout_s 120
