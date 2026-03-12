#!/bin/bash
# Deploy pruned + PTQ (INT8) MobileNet-v1 ONNX models on Raspberry Pi 3B+.
# Execute from HW4_files/: bash run_problem3_deploy.sh

MODELS_DIR="./pruned_ptq_models"

# pruning_fraction = 0.05
python deploy_onnx.py \
    --model mobilenet \
    --onnx_path "${MODELS_DIR}/l1_0.05_5_5_MBNv1_int8.onnx" \
    --device raspi \
    --output_dir ./results/ptq_0.05

# pruning_fraction = 0.1
python deploy_onnx.py \
    --model mobilenet \
    --onnx_path "${MODELS_DIR}/l1_0.1_5_5_MBNv1_int8.onnx" \
    --device raspi \
    --output_dir ./results/ptq_0.1

# pruning_fraction = 0.2
python deploy_onnx.py \
    --model mobilenet \
    --onnx_path "${MODELS_DIR}/l1_0.2_5_5_MBNv1_int8.onnx" \
    --device raspi \
    --output_dir ./results/ptq_0.2

# pruning_fraction = 0.3
python deploy_onnx.py \
    --model mobilenet \
    --onnx_path "${MODELS_DIR}/l1_0.3_5_5_MBNv1_int8.onnx" \
    --device raspi \
    --output_dir ./results/ptq_0.3

# pruning_fraction = 0.4
python deploy_onnx.py \
    --model mobilenet \
    --onnx_path "${MODELS_DIR}/l1_0.4_5_5_MBNv1_int8.onnx" \
    --device raspi \
    --output_dir ./results/ptq_0.4

# pruning_fraction = 0.5
python deploy_onnx.py \
    --model mobilenet \
    --onnx_path "${MODELS_DIR}/l1_0.5_5_5_MBNv1_int8.onnx" \
    --device raspi \
    --output_dir ./results/ptq_0.5
