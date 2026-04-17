#!/bin/bash

# Focused frontier sweep around the current best qualifying runs.
# Rationale from benchmark_tracking.json:
# - champion_sword + fedprox is the current wall-clock leader (exp199).
# - champion_explorer_v2 is the current lowest-energy family (exp208/209/210).
# - The next useful move is not a broad architecture search; it is a targeted
#   structured-pruning / width-tuning sweep around those proven winners.
#
# This script intentionally mixes:
# - Anchor reruns of the best current configurations.
# - Sword-family speed variants: try to reach >=90% in fewer rounds.
# - Explorer_v2-family eco variants: try to cut per-round energy further.
#
# Launch example:
#   PYTHON_BIN=/Users/slrpz/miniconda3/envs/ece_361e_fl/bin/python \
#   FL_CLOUD_IP=172.29.204.66 \
#   bash config_m3_pruning.bash

############### TODO CHANGE
cloud_ip="${FL_CLOUD_IP:-172.29.204.66}" # Override when VPN/cloud IP changes.
python_bin="${PYTHON_BIN:-python}"

default_model_name="champion_sword"
default_learning_rate=0.01
loss_func_name="cross_entropy"
default_loss_type="fedprox"
default_mu=1.0
default_beta=10.0

declare -a experiment_configs=(
# experiment | run | data_iid | model_name | learning_rate | loss_type | mu | beta | rpi_local_epochs | mc1_local_epochs

# Time-frontier anchors and Sword-family variants
  "301 1 false champion_sword 0.010 fedprox 1.0 10.0 1 1"
  "302 1 false champion_sword 0.009 fedprox 1.0 10.0 1 1"
  "303 1 false vanguard_sword_midboost 0.010 fedprox 1.0 10.0 1 1"
  "304 1 false vanguard_sword_turbo 0.010 fedprox 1.0 10.0 1 1"
  "305 1 false vanguard_sword_eco 0.009 fedprox 1.0 10.0 1 1"
  "306 1 false vanguard_sword_eco 0.010 fedprox 1.1 10.0 1 1"

# Energy-frontier anchors and Explorer_v2-family variants
  "307 1 false champion_explorer_v2 0.010 fedavg 1.0 10.0 1 1"
  "308 1 false champion_explorer_v2 0.010 fedmax 1.0 10.0 1 1"
  "309 1 false vanguard_explorerv2_eco 0.010 fedavg 1.0 10.0 1 1"
  "310 1 false vanguard_explorerv2_eco 0.010 fedmax 1.0 10.0 1 1"
  "311 1 false vanguard_explorerv2_depthwise_eco 0.010 fedmax 1.0 10.0 1 1"
  "312 1 false vanguard_explorerv2_turbo 0.010 fedprox 1.0 10.0 1 1"
)

declare -a devices_configs=(
# hw_type | host | port | cuda_name
  "rpi sld-rpi-09.ece.utexas.edu 9090 cpu"
  "mc1 sld-mc1-09.ece.utexas.edu 9090 cpu"
)
############### TODO END CHANGE

verbose='false'
laptop_number='laptop_1'
cloud_port="9090"
cloud_cuda="cpu"
comm_rounds=30
num_devices=2

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

  [ -z "$model_name" ] && model_name="$default_model_name"
  [ -z "$learning_rate" ] && learning_rate="$default_learning_rate"
  [ -z "$loss_type" ] && loss_type="$default_loss_type"
  [ -z "$mu" ] && mu="$default_mu"
  [ -z "$beta" ] && beta="$default_beta"

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

    host="${dev_config[1]}"
    hosts+=("$host")

    port="${dev_config[2]}"
    ports+=("$port")

    cuda_name="${dev_config[3]}"
    cuda_names+=("$cuda_name")

    model_names+=("$model_name")
    if [ "$hw_type" = "rpi" ]; then
      dev_local_epochs+=("$rpi_local_epochs")
    elif [ "$hw_type" = "mc1" ]; then
      dev_local_epochs+=("$mc1_local_epochs")
    else
      echo "Unsupported hw_type: $hw_type. Expected rpi or mc1."
      exit 1
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
