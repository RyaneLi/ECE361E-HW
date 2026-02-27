# Problem 2: ONNX Deployment - Complete Guide

This is your complete guide for Problem 2: Deployment on Edge Devices Using ONNX. Everything you need is in this document.

## Table of Contents
1. [Quick Command Reference](#quick-command-reference)
2. [Critical Requirements](#critical-requirements)
3. [Step-by-Step Workflow](#step-by-step-workflow)
4. [Troubleshooting](#troubleshooting)

---

## Quick Command Reference

### Question 1: Convert Models to ONNX

**CRITICAL**: Create separate ONNX files for each device (different opset versions)!
- MC1: opset 13
- RaspberryPi: opset 16

```bash
# For Odroid MC1 (opset 13)
python convert_onnx.py --model vgg11 --model_path ./vgg11.pt --device mc1 --output_path ./onnx_models
python convert_onnx.py --model vgg16 --model_path ./vgg16.pt --device mc1 --output_path ./onnx_models

# For RaspberryPi 3B+ (opset 16)
python convert_onnx.py --model vgg11 --model_path ./vgg11.pt --device raspi --output_path ./onnx_models
python convert_onnx.py --model vgg16 --model_path ./vgg16.pt --device raspi --output_path ./onnx_models
```

Output: `vgg11_mc1.onnx`, `vgg16_mc1.onnx`, `vgg11_raspi.onnx`, `vgg16_raspi.onnx`

### Question 2: Deploy on Edge Devices

**On Odroid MC1** (MUST use taskset for 4 big cores):
```bash
# VGG11
taskset --all-tasks 0xF0 python deploy_onnx.py \
    --model vgg11 --onnx_path ./onnx_models/vgg11_mc1.onnx \
    --device mc1 --output_dir ./results_mc1

# VGG16
taskset --all-tasks 0xF0 python deploy_onnx.py \
    --model vgg16 --onnx_path ./onnx_models/vgg16_mc1.onnx \
    --device mc1 --output_dir ./results_mc1
```

**On RaspberryPi 3B+**:
```bash
# VGG11
python deploy_onnx.py \
    --model vgg11 --onnx_path ./onnx_models/vgg11_raspi.onnx \
    --device raspi --output_dir ./results_raspi

# VGG16
python deploy_onnx.py \
    --model vgg16 --onnx_path ./onnx_models/vgg16_raspi.onnx \
    --device raspi --output_dir ./results_raspi
```

**Note**: The script looks for `./test_deployment` folder by default (in the same directory as the script)

**If Smart Power 2 unavailable**: Add `--no-power` flag

### Question 3: Generate Plots and Tables

```bash
# Compile Tables 2 and 3
python compile_tables.py --results_dir ./results --show_stats

# Generate comparison plots for VGG11
python plot_results.py \
    --mc1_data ./results/mc1/vgg11_timeseries_data.csv \
    --raspi_data ./results/raspi/vgg11_timeseries_data.csv \
    --output_dir ./plots
```

---

## Critical Requirements

### 1. Device-Specific ONNX Versions ⚠️

**YOU MUST CREATE 4 SEPARATE ONNX FILES!**
- Odroid MC1: opset version 13
- RaspberryPi 3B+: opset version 16

Each device needs its own version of each model.

### 2. Odroid MC1: MUST Use Taskset ⚠️

Always run with `taskset --all-tasks 0xF0` to use 4 big cores (cores 4-7)

**VERIFY**: Use `htop` to confirm all 4 cores at ~100% utilization during inference!

### 3. Power Measurement

Both devices connect to **Smart Power 2** at 192.168.4.1 via telnet (same as HW2)
- Verify connection: `telnet 192.168.4.1`
- If unavailable, use `--no-power` flag

### 4. Temperature Measurement

**Odroid MC1**:
- Average of all 4 big core temperatures (thermal zones 0-3)
- Cores 1 and 3 are swapped (hardware quirk)

**RaspberryPi 3B+**:
- Uses `gpiozero.CPUTemperature().temperature`

### 5. RAM Measurement

**IMPORTANT**: Report inference-only RAM, not total RAM!
- Measure idle RAM before inference loop
- Measure peak RAM during inference
- Report: Peak RAM - Idle RAM

Use `htop` or `watch -n0.1 free -m` to monitor during inference.

### 6. Energy Consumption

**Different from RAM**: Energy uses entire script execution time
- Energy (J) = Average Power (W) × Total Script Time (s)
- Includes loading, preprocessing, and inference
- Same methodology as HW2 benchmarks

---

## Step-by-Step Workflow

### Prerequisites

1. **Install dependencies on edge devices**:
```bash
pip install numpy onnxruntime pillow tqdm psutil

# RaspberryPi only (usually pre-installed on Raspbian)
pip install gpiozero
```

2. **Ensure test_deployment folder** exists on the edge devices with all 10,000 CIFAR10 test images (should already be present)

### Step 1: Train Models (Problem 1)

Train VGG11 and VGG16 on TACC, resulting in:
- `vgg11.pt`
- `vgg16.pt`

### Step 2: Convert to ONNX (Question 1)

**On TACC or local machine** (NOT on edge devices):

```bash
# Create 4 ONNX files (2 models × 2 devices)
python convert_onnx.py --model vgg11 --model_path ./vgg11.pt --device mc1 --output_path ./onnx_models
python convert_onnx.py --model vgg16 --model_path ./vgg16.pt --device mc1 --output_path ./onnx_models
python convert_onnx.py --model vgg11 --model_path ./vgg11.pt --device raspi --output_path ./onnx_models
python convert_onnx.py --model vgg16 --model_path ./vgg16.pt --device raspi --output_path ./onnx_models
```

### Step 3: Transfer Files to Edge Devices

**Transfer to Odroid MC1**:
```bash
scp -r onnx_models/ student@mc1:/home/student/HW3_files/
scp deploy_onnx.py student@mc1:/home/student/HW3_files/
scp sysfs_paths.py student@mc1:/home/student/HW3_files/
```

**Transfer to RaspberryPi 3B+**:
```bash
scp -r onnx_models/ student@raspi:/home/student/HW3_files/
scp deploy_onnx.py student@raspi:/home/student/HW3_files/
```

**Note**: test_deployment folder is already present on the edge devices

### Step 4: Run Deployment on Devices (Question 2)

**On Odroid MC1** (via SSH):
```bash
# Verify Smart Power 2 connection
telnet 192.168.4.1

# Run VGG11 on 4 big cores
taskset --all-tasks 0xF0 python deploy_onnx.py \
    --model vgg11 --onnx_path ./mc1/vgg11_mc1.onnx \
    --device mc1 --output_dir ./results_mc1

# Open another SSH session and verify with htop
htop  # Check cores 4-7 are at ~100%

# Run VGG16
taskset --all-tasks 0xF0 python deploy_onnx.py \
    --model vgg16 --onnx_path ./mc1/vgg16_mc1.onnx \
    --device mc1 --output_dir ./results_mc1
```

**On RaspberryPi 3B+** (via SSH):
```bash
# Verify Smart Power 2 connection
telnet 192.168.4.1

# Run VGG11
python deploy_onnx.py \
    --model vgg11 --onnx_path ./raspi/vgg11_raspi.onnx \
    --device raspi --output_dir ./results_raspi

# Monitor with htop if desired
htop

# Run VGG16
python deploy_onnx.py \
    --model vgg16 --onnx_path ./raspi/vgg16_raspi.onnx \
    --device raspi --output_dir ./results_raspi
```

### Step 5: Collect Results

**Copy results back to local machine**:
```bash
scp -r student@mc1:/home/student/HW3_files/results_mc1 ./results/mc1
scp -r student@raspi:/home/student/HW3_files/results_raspi ./results/raspi
```

Expected directory structure:
```
results/
├── mc1/
│   ├── vgg11_summary_metrics.csv
│   ├── vgg11_timeseries_data.csv
│   ├── vgg16_summary_metrics.csv
│   └── vgg16_timeseries_data.csv
└── raspi/
    ├── vgg11_summary_metrics.csv
    ├── vgg11_timeseries_data.csv
    ├── vgg16_summary_metrics.csv
    └── vgg16_timeseries_data.csv
```

### Step 6: Generate Tables (Question 2)

```bash
python compile_tables.py --results_dir ./results --show_stats
```

This creates:
- `results/table2_compiled.csv` - Total inference time, RAM, Accuracy
- `results/table3_compiled.csv` - Total energy consumption

### Step 7: Generate Plots (Question 3)

```bash
python plot_results.py \
    --mc1_data ./results/mc1/vgg11_timeseries_data.csv \
    --raspi_data ./results/raspi/vgg11_timeseries_data.csv \
    --output_dir ./plots
```

This creates:
- `plots/vgg11_power_consumption_comparison.png`
- `plots/vgg11_temperature_comparison.png`

### Step 8: Analysis (Question 3)

Based on plots and tables, compare:
1. **Inference Time**: Which device is faster?
2. **Accuracy**: Same as Problem 1 results?
3. **Power Consumption**: Average power and patterns
4. **Temperature**: Thermal behavior during inference
5. **Energy Efficiency**: Energy per image = Total Energy / 10,000 images
6. **Memory**: RAM constraints on each device

**Device preference**: Consider speed, power, temperature, and your application requirements.

---

## Verification Checklist

### Before Running

**Odroid MC1**:
- [ ] Correct ONNX model with opset 13
- [ ] Will use `taskset --all-tasks 0xF0`
- [ ] Smart Power 2 connected (or use `--no-power`)

**RaspberryPi 3B+**:
- [ ] Correct ONNX model with opset 16
- [ ] `gpiozero` installed
- [ ] Smart Power 2 connected (or use `--no-power`)

### During Inference

**Odroid MC1**:
- [ ] Check `htop`: Cores 4-7 at ~100%
- [ ] Check power readings are updating

**RaspberryPi 3B+**:
- [ ] Check `htop`: CPUs active
- [ ] Check power readings are updating

### After Inference

**Both Devices**:
- [ ] Test accuracy matches Problem 1 results
- [ ] All 10,000 images processed
- [ ] CSV files generated
- [ ] Power measurements recorded (unless `--no-power`)
- [ ] Temperature measurements recorded

---

## Troubleshooting

### ONNX Runtime Errors

**Problem**: Model fails to load or run
**Solution**: Verify correct opset version for device (MC1=13, RaspberryPi=16)

### Power Meter Connection Failed

**Problem**: Cannot connect to 192.168.4.1
**Solution**: 
1. Verify Smart Power 2 is on and connected
2. Test: `telnet 192.168.4.1`
3. If unavailable, use `--no-power` flag

### Low CPU Utilization on MC1

**Problem**: Cores not at 100%
**Solution**: 
1. Ensure using `taskset --all-tasks 0xF0`
2. Check with `htop` during inference
3. Verify model is running on cores 4-7

### Temperature Reading Fails

**Problem**: Temperature shows 0.0
**Solution**:
- MC1: Check sysfs thermal sensor paths
- RaspberryPi: Ensure `gpiozero` installed

### Out of Memory

**Problem**: Edge device runs out of RAM
**Solution**: This is expected behavior - record the RAM usage and continue. Edge devices have limited memory.

### Import Errors

**Problem**: Missing Python packages
**Solution**: 
```bash
pip install numpy onnxruntime pillow tqdm psutil
pip install gpiozero  # RaspberryPi only
```

### Test Images Not Found

**Problem**: Cannot find test_deployment folder
**Solution**: Verify the test_deployment folder exists in the same directory as deploy_onnx.py with all 10,000 images

---

## Scripts Provided

1. **convert_onnx.py** - Converts PyTorch to ONNX with device-specific opset
2. **deploy_onnx.py** - Runs inference and collects all metrics automatically
3. **compile_tables.py** - Generates Table 2 and Table 3 from CSV results
4. **plot_results.py** - Creates power and temperature comparison plots

---

## Key Implementation Details

### Power Measurement (Telnet)
```python
import telnetlib as tel

telnet_connection = tel.Telnet("192.168.4.1", timeout=10)

def get_telnet_power(telnet_conn, last_pwr):
    tel_dat = str(telnet_conn.read_very_eager())
    # Parse CSV format from Smart Power 2
    # Returns power in Watts
```

### Temperature - MC1 (Average 4 Big Cores)
```python
def get_temps_mc1():
    templ = []
    for i in range(4):  # Thermal zones 0-3
        temp = float(open(sysfs.fn_thermal_sensor.format(i), 'r').read()) / 1000
        templ.append(temp)
    # Swap cores 1 and 3 (hardware quirk)
    templ[1], templ[3] = templ[3], templ[1]
    return sum(templ) / len(templ)  # Average
```

### Temperature - RaspberryPi
```python
import gpiozero

def get_temp_raspi():
    cpu_temp = gpiozero.CPUTemperature().temperature
    return cpu_temp
```

---

## Common Mistakes to Avoid

1. ❌ Using same ONNX file for both devices → ✅ Create device-specific files
2. ❌ Running MC1 without taskset → ✅ Always use `taskset --all-tasks 0xF0`
3. ❌ Not verifying CPU utilization → ✅ Check with `htop`
4. ❌ Reporting total RAM → ✅ Report inference RAM increase only
5. ❌ Using inference time for energy → ✅ Use entire script execution time
6. ❌ Single core temp for MC1 → ✅ Average of 4 big cores
7. ❌ Wrong opset version → ✅ MC1=13, RaspberryPi=16

---

## Summary

**Question 1**: Convert models with device-specific opset → 4 ONNX files  
**Question 2**: Deploy on both devices → Collect metrics, fill Table 2  
**Question 3**: Generate plots → Compare devices, fill Table 3, analyze

All metrics are collected automatically. Just run the commands, monitor with `htop`, and collect the CSV files!

Good luck! 🚀
