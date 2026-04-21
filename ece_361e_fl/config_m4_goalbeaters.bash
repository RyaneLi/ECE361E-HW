#!/bin/bash

# Focused M4 frontier sweep around the best goal-beating runs so far.
#
# Winning families from benchmark_tracking.json:
# - Wall-clock leaders: champion_sword (exp197-201, 204) and
#   refined_simplecnn_small_4_8_deep (exp164-166).
# - Energy leaders: champion_explorer_v2 (exp208-210).
#
# This batch intentionally mixes three approaches:
# 1. Anchor reruns of the best proven settings.
# 2. Pruning-style eco descendants close to the winning architectures.
# 3. Different-approach low-cost head variants that keep the same useful trunk
#    but replace the expensive classifier with adaptive pooling.
#
# Launch example:
#   source .autolaunch_env
#   source /Users/slrpz/miniconda3/etc/profile.d/conda.sh
#   conda activate ece_361e_fl
#   PYTHON_BIN=$(which python) bash config_m4_goalbeaters.bash

############### TODO CHANGE
cloud_ip="${FL_CLOUD_IP:-172.29.204.147}"
python_bin="${PYTHON_BIN:-python}"

default_model_name="champion_sword"
default_learning_rate=0.01
loss_func_name="cross_entropy"
default_loss_type="fedprox"
default_mu=1.0
default_beta=10.0

declare -a experiment_configs=(
# experiment | run | data_iid | model_name | learning_rate | loss_type | mu | beta | rpi_local_epochs | mc1_local_epochs

# Time / balanced anchors from the current Pareto frontier
  "319 1 false champion_sword 0.010 fedprox 1.0 10.0 1 1"
  "320 1 false champion_sword 0.009 fedprox 1.0 10.0 1 1"
  "321 1 false champion_sword 0.010 fedprox 1.1 10.0 1 1"
  "322 1 false refined_simplecnn_small_4_8_deep 0.010 fedprox 1.0 10.0 1 1"

# Different approach: same useful trunk, cheaper head / smaller transfer cost
  "323 1 false champion_knife 0.010 fedprox 1.0 10.0 1 1"
  "324 1 false champion_dagger_v2 0.010 fedprox 1.0 10.0 1 1"
  "325 1 false champion_rogue 0.010 fedprox 1.0 10.0 1 1"

# Pruning-style descendants near the time winner
  "326 1 false vanguard_sword_eco 0.010 fedprox 1.0 10.0 1 1"
  "327 1 false vanguard_sword_eco 0.009 fedprox 1.0 10.0 1 1"

# Energy anchors and pruning-style descendants near the energy winner
  "328 1 false champion_explorer_v2 0.010 fedavg 1.0 10.0 1 1"
  "329 1 false champion_explorer_v2 0.010 fedmax 1.0 10.0 1 1"
  "330 1 false champion_explorer_v2 0.010 fedprox 1.0 10.0 1 1"
  "331 1 false vanguard_explorerv2_eco 0.010 fedprox 1.0 10.0 1 1"
  "332 1 false vanguard_explorerv2_depthwise_eco 0.010 fedprox 1.0 10.0 1 1"
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
