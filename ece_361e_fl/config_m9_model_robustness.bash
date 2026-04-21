#!/bin/bash

# M3-compliant robustness + averaging batch.
#
# Why this batch:
# - The best single-run frontier is still dominated by champion_sword and
#   refined_simplecnn_small_4_8_deep, but the official M3 result is likely to
#   depend on the 3-run average.
# - The latest sweeps showed a real weakness on run3 / seed 26, so the main
#   remaining opportunity is robustness across seeds rather than more tiny
#   run1-only hyperparameter tweaks.
# - Most alternate champion families already failed clearly (fighter/blade/
#   knife/dagger/rogue), so the alternative-model side focuses only on
#   architectures that still have a plausible robustness story.
# - In parallel, we complete run2/run3 for the strongest distinct run1
#   configurations so we can compare real 3-run averages instead of relying on
#   single-seed winners.
#
# Included families / tracks:
# 1. refined_simplecnn_small_4_8_deeper
#    - already qualifies from scratch
#    - slower than the current leaders, but could be more stable across seeds
# 2. simplecnn_small_4_8_deep_groupnorm
#    - previously missed qualification by only ~1%
#    - replaces batch norm with group norm, which may help cross-device / seed
#      stability in FL while remaining lightweight
# 3. best distinct run1 frontier configs
#    - champion_sword fedprox (baseline + low-energy variants)
#    - refined_simplecnn_small_4_8_deep fedprox
#    - champion_explorer_v2 fedavg/fedmax
#    - these are run2/run3 completions for average-based comparison
#
# All runs in this file:
# - start from scratch
# - stay at 30 communication rounds
# - use the M3 non-IID / 1-local-epoch setup

cloud_ip="${FL_CLOUD_IP:-172.29.204.147}"
python_bin="${PYTHON_BIN:-python}"

loss_func_name="cross_entropy"
verbose='false'
laptop_number='laptop_1'
cloud_port="9090"
cloud_cuda="cpu"
comm_rounds=30
num_devices=2

declare -a experiment_configs=(
# experiment | run | data_iid | model_name | learning_rate | loss_type | mu | beta | rpi_local_epochs | mc1_local_epochs

# Alternate robust challenger: deeper 4->8->16->24 with fedavg
  "399 1 false refined_simplecnn_small_4_8_deeper 0.0100 fedavg 1.0 10.0 1 1"
  "399 2 false refined_simplecnn_small_4_8_deeper 0.0100 fedavg 1.0 10.0 1 1"
  "399 3 false refined_simplecnn_small_4_8_deeper 0.0100 fedavg 1.0 10.0 1 1"

# Alternate robust challenger: deeper 4->8->16->24 with fedprox
  "400 1 false refined_simplecnn_small_4_8_deeper 0.0100 fedprox 1.0 10.0 1 1"
  "400 2 false refined_simplecnn_small_4_8_deeper 0.0100 fedprox 1.0 10.0 1 1"
  "400 3 false refined_simplecnn_small_4_8_deeper 0.0100 fedprox 1.0 10.0 1 1"

# Different-normalization challenger: same lightweight depth, but GroupNorm
  "401 1 false simplecnn_small_4_8_deep_groupnorm 0.0100 fedprox 1.0 10.0 1 1"
  "401 2 false simplecnn_small_4_8_deep_groupnorm 0.0100 fedprox 1.0 10.0 1 1"
  "401 3 false simplecnn_small_4_8_deep_groupnorm 0.0100 fedprox 1.0 10.0 1 1"

# Complete 3-run averages for the strongest distinct run1 frontier configs.
  "402 2 false champion_sword 0.0100 fedprox 1.0 10.0 1 1"
  "402 3 false champion_sword 0.0100 fedprox 1.0 10.0 1 1"

  "403 2 false refined_simplecnn_small_4_8_deep 0.0100 fedprox 1.0 10.0 1 1"
  "403 3 false refined_simplecnn_small_4_8_deep 0.0100 fedprox 1.0 10.0 1 1"

  "404 2 false champion_explorer_v2 0.0100 fedavg 1.0 10.0 1 1"
  "404 3 false champion_explorer_v2 0.0100 fedavg 1.0 10.0 1 1"

  "405 2 false champion_explorer_v2 0.0100 fedmax 1.0 10.0 1 1"
  "405 3 false champion_explorer_v2 0.0100 fedmax 1.0 10.0 1 1"

  "406 2 false champion_sword 0.0090 fedprox 1.0 10.0 1 1"
  "406 3 false champion_sword 0.0090 fedprox 1.0 10.0 1 1"

  "407 2 false champion_sword 0.0100 fedprox 1.1 10.0 1 1"
  "407 3 false champion_sword 0.0100 fedprox 1.1 10.0 1 1"
)

declare -a devices_configs=(
  "rpi sld-rpi-09.ece.utexas.edu 9090 cpu"
  "mc1 sld-mc1-09.ece.utexas.edu 9090 cpu"
)

for experiment_config in "${experiment_configs[@]}"
do
  read -a exp_config <<< "$experiment_config"
  experiment="${exp_config[0]}"
  run="${exp_config[1]}"
  data_iid="${exp_config[2]}"
  model_name="${exp_config[3]}"
  learning_rate="${exp_config[4]}"
  loss_type="${exp_config[5]}"
  mu="${exp_config[6]}"
  beta="${exp_config[7]}"
  rpi_local_epochs="${exp_config[8]}"
  mc1_local_epochs="${exp_config[9]}"

  if [ "$run" -eq 1 ]; then
    seed=2
  elif [ "$run" -eq 2 ]; then
    seed=14
  elif [ "$run" -eq 3 ]; then
    seed=26
  else
    echo "Invalid run value. Please specify 1, 2, or 3."
    exit 1
  fi

  dev_hw_types=()
  hosts=()
  ports=()
  cuda_names=()
  model_names=()
  dev_local_epochs=()

  for devs_configs in "${devices_configs[@]}"
  do
    read -a dev_config <<< "$devs_configs"
    hw_type="${dev_config[0]}"
    dev_hw_types+=("$hw_type")
    hosts+=("${dev_config[1]}")
    ports+=("${dev_config[2]}")
    cuda_names+=("${dev_config[3]}")
    model_names+=("$model_name")
    if [ "$hw_type" = "rpi" ]; then
      dev_local_epochs+=("$rpi_local_epochs")
    else
      dev_local_epochs+=("$mc1_local_epochs")
    fi
  done

  cloud_config_filename="cloud_cfg_exp${experiment}_run${run}.json"
  dev_config_filename="dev_cfg_exp${experiment}_run${run}.json"

  "$python_bin" generate_configs.py \
    --cloud_config_filename "${cloud_config_filename}" \
    --dev_config_filename "${dev_config_filename}" \
    --cloud_ip "${cloud_ip}" \
    --cloud_port "${cloud_port}" \
    --cloud_cuda "${cloud_cuda}" \
    --model_name "${model_name}" \
    --loss_func_name "${loss_func_name}" \
    --loss_type "${loss_type}" \
    --mu "${mu}" \
    --beta "${beta}" \
    --comm_rounds "${comm_rounds}" \
    --learning_rate "${learning_rate}" \
    --verbose "${verbose}" \
    --experiment "${experiment}" \
    --run "${run}" \
    --seed "${seed}" \
    --laptop_number "${laptop_number}" \
    --data_iid "${data_iid}" \
    --num_devices "${num_devices}" \
    --dev_hw_types "${dev_hw_types[*]}" \
    --hosts "${hosts[*]}" \
    --ports "${ports[*]}" \
    --cuda_names "${cuda_names[*]}" \
    --model_names "${model_names[*]}" \
    --dev_local_epochs "${dev_local_epochs[*]}"
done
