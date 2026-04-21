#!/bin/zsh
set -euo pipefail

ROOT="/Users/slrpz/Downloads/ECE361E/ece_361e_fl"
LOG_FILE="$ROOT/logs/m8_pareto_pipeline.log"

exec >>"$LOG_FILE" 2>&1

echo "[$(date)] Waiting for exp382 run1 to finish before launching M8 Pareto refinement."

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

  if [[ "$last_successful" == "382:1" ]]; then
    break
  fi

  echo "[$(date)] Still waiting. last_successful=$last_successful"
  sleep 30
done

echo "[$(date)] Launching M8 Pareto refinement."
python start_tmux_device_managers.py --devices rpi mc1
MPLCONFIGDIR='/tmp/mplconfig' python -u run_experiment_queue.py \
  --skip_generate \
  --skip_manager_start \
  --config_script config_m8_pareto_refine.bash \
  --start_exp 383 \
  --end_exp 398 \
  --manager_boot_timeout_s 120
