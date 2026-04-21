#!/bin/bash

# Anchor-capture reruns for post-hoc pruning.
# These reproduce the strongest historical winners so cloud_manager.py can
# archive fresh checkpoints under artifacts/checkpoints/exp340_run1...exp342_run1.

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
  "340 1 false champion_sword 0.010 fedprox 1.0 10.0 1 1"
  "341 1 false champion_explorer_v2 0.010 fedavg 1.0 10.0 1 1"
  "342 1 false champion_explorer_v2 0.010 fedmax 1.0 10.0 1 1"
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
