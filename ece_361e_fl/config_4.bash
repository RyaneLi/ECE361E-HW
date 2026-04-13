#!/bin/bash

# This script is used to generate commands to run cloud.py and device.py on edge devices
# Change the parameters below and run "bash run.bash" on terminal
# It will also run "generate_configs.py" based on the given parameters

############### TODO CHANGE
cloud_ip="" #Change every time when new VPN is connected

model_name="conv5small"
loss_func_name="cross_entropy"
learning_rate=0.01 # Keep a single global LR (no per-device LR in this milestone setup)

declare -a experiment_configs=(
# experiment | run | data_iid | loss_type | mu | beta | rpi_local_epochs | mc1_local_epochs
  "10 1 false fedprox 1.5 10.0 2 1" # Non-IID: FedProx, changed mu + local epochs per device
  "11 1 false fedmax 1.0 20.0 3 1" # Non-IID: FedMAX, changed beta + local epochs per device
)

declare -a devices_configs=(
# hw_type | host | port | cuda_name
  "rpi sld-rpi-09.ece.utexas.edu 9090 cpu" #Change number of device
  "mc1 sld-mc1-09.ece.utexas.edu 9090 cpu" #Change number of device
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
  loss_type="${exp_config[3]}"
  mu="${exp_config[4]}"
  beta="${exp_config[5]}"
  rpi_local_epochs="${exp_config[6]}"
  mc1_local_epochs="${exp_config[7]}"

  # Seeds for all runs are predetermined.
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

  declare -a dev_hw_types
  declare -a hosts
  declare -a ports
  declare -a cuda_names
  declare -a model_names
  declare -a dev_local_epochs
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

  python generate_configs.py \
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
