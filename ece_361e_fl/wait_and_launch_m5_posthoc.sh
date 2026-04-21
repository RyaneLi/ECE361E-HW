#!/bin/zsh
set -euo pipefail

ROOT_DIR="/Users/slrpz/Downloads/ECE361E/ece_361e_fl"
QUEUE_LOG="${ROOT_DIR}/logs/m5_posthoc_pipeline.log"
PYTHON_BIN="/Users/slrpz/miniconda3/envs/ece_361e_fl/bin/python"

cd "${ROOT_DIR}"
mkdir -p "${ROOT_DIR}/logs"
echo "[$(date)] Starting M5 posthoc watcher." >> "${QUEUE_LOG}"

source "${ROOT_DIR}/.autolaunch_env"

echo "[$(date)] Waiting for anchor-capture queue to reach exp342." >> "${QUEUE_LOG}"

while true; do
  last_exp=$("${PYTHON_BIN}" -c "import json, pathlib; p = pathlib.Path('${ROOT_DIR}/logs/job_queue_state.json'); print(json.loads(p.read_text()).get('last_successful', {}).get('exp', 0) if p.exists() else 0)")
  if [[ "${last_exp}" -ge 342 ]]; then
    break
  fi
  sleep 30
done

echo "[$(date)] Anchor capture complete. Pruning archived checkpoints." >> "${QUEUE_LOG}"

mkdir -p artifacts/pruned_checkpoints artifacts/pruning_reports

"${PYTHON_BIN}" prune_checkpoint.py \
  --cloud_cfg artifacts/checkpoints/exp340_run1/cloud_cfg_exp340_run1.json \
  --source_checkpoint artifacts/checkpoints/exp340_run1/global_weights.pth \
  --output_checkpoint artifacts/pruned_checkpoints/exp340_run1_global_l1_020.pth \
  --report_json artifacts/pruning_reports/exp340_run1_global_l1_020.json \
  --amount 0.20 --method global_l1 --include_conv --include_linear --eval >> "${QUEUE_LOG}" 2>&1

"${PYTHON_BIN}" prune_checkpoint.py \
  --cloud_cfg artifacts/checkpoints/exp340_run1/cloud_cfg_exp340_run1.json \
  --source_checkpoint artifacts/checkpoints/exp340_run1/global_weights.pth \
  --output_checkpoint artifacts/pruned_checkpoints/exp340_run1_global_l1_035.pth \
  --report_json artifacts/pruning_reports/exp340_run1_global_l1_035.json \
  --amount 0.35 --method global_l1 --include_conv --include_linear --eval >> "${QUEUE_LOG}" 2>&1

"${PYTHON_BIN}" prune_checkpoint.py \
  --cloud_cfg artifacts/checkpoints/exp341_run1/cloud_cfg_exp341_run1.json \
  --source_checkpoint artifacts/checkpoints/exp341_run1/global_weights.pth \
  --output_checkpoint artifacts/pruned_checkpoints/exp341_run1_global_l1_020.pth \
  --report_json artifacts/pruning_reports/exp341_run1_global_l1_020.json \
  --amount 0.20 --method global_l1 --include_conv --include_linear --eval >> "${QUEUE_LOG}" 2>&1

"${PYTHON_BIN}" prune_checkpoint.py \
  --cloud_cfg artifacts/checkpoints/exp341_run1/cloud_cfg_exp341_run1.json \
  --source_checkpoint artifacts/checkpoints/exp341_run1/global_weights.pth \
  --output_checkpoint artifacts/pruned_checkpoints/exp341_run1_global_l1_035.pth \
  --report_json artifacts/pruning_reports/exp341_run1_global_l1_035.json \
  --amount 0.35 --method global_l1 --include_conv --include_linear --eval >> "${QUEUE_LOG}" 2>&1

"${PYTHON_BIN}" prune_checkpoint.py \
  --cloud_cfg artifacts/checkpoints/exp342_run1/cloud_cfg_exp342_run1.json \
  --source_checkpoint artifacts/checkpoints/exp342_run1/global_weights.pth \
  --output_checkpoint artifacts/pruned_checkpoints/exp342_run1_global_l1_020.pth \
  --report_json artifacts/pruning_reports/exp342_run1_global_l1_020.json \
  --amount 0.20 --method global_l1 --include_conv --include_linear --eval >> "${QUEUE_LOG}" 2>&1

"${PYTHON_BIN}" prune_checkpoint.py \
  --cloud_cfg artifacts/checkpoints/exp342_run1/cloud_cfg_exp342_run1.json \
  --source_checkpoint artifacts/checkpoints/exp342_run1/global_weights.pth \
  --output_checkpoint artifacts/pruned_checkpoints/exp342_run1_global_l1_035.pth \
  --report_json artifacts/pruning_reports/exp342_run1_global_l1_035.json \
  --amount 0.35 --method global_l1 --include_conv --include_linear --eval >> "${QUEUE_LOG}" 2>&1

echo "[$(date)] Generating M5 configs and starting managers." >> "${QUEUE_LOG}"
PYTHON_BIN="${PYTHON_BIN}" bash config_m5_posthoc_pruning.bash >> "${QUEUE_LOG}" 2>&1
"${PYTHON_BIN}" start_tmux_device_managers.py --devices rpi mc1 >> "${QUEUE_LOG}" 2>&1

echo "[$(date)] Launching M5 post-hoc pruning queue." >> "${QUEUE_LOG}"
MPLCONFIGDIR='/tmp/mplconfig' "${PYTHON_BIN}" -u run_experiment_queue.py \
  --skip_generate \
  --skip_manager_start \
  --config_script config_m5_posthoc_pruning.bash \
  --start_exp 343 \
  --end_exp 349 \
  --manager_boot_timeout_s 120 >> "${QUEUE_LOG}" 2>&1
