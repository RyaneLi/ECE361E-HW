import numpy as np
import onnxruntime as rt
from tqdm import tqdm
import os
from PIL import Image
import argparse
import time
import psutil
import sys
import telnetlib as tel

# Add path for sysfs_paths if using it
if os.path.exists('/home/student/HW3_files/sysfs_paths.py'):
    sys.path.append('/home/student/HW3_files')
    import sysfs_paths as sysfs

# For RaspberryPi temperature monitoring
try:
    import gpiozero
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False

# Create argument parser object
parser = argparse.ArgumentParser(description='Deploy ONNX models on edge devices')

# Add one argument for selecting VGG or MobileNet-v1 models
parser.add_argument('--model', type=str, required=True, 
                    choices=['vgg11', 'vgg16', 'mobilenet'],
                    help='Model to deploy (vgg11, vgg16, or mobilenet)')
parser.add_argument('--onnx_path', type=str, required=True,
                    help='Path to the ONNX model file')
parser.add_argument('--test_data_path', type=str, 
                    default='/home/student/HW3_files/test_deployment',
                    help='Path to test deployment dataset')
parser.add_argument('--output_dir', type=str, default='./results',
                    help='Directory to save results')
parser.add_argument('--device', type=str, default='mc1', choices=['mc1', 'raspi'],
                    help='Device type: mc1 (Odroid MC1) or raspi (RaspberryPi 3B+)')
parser.add_argument('--no-power', action='store_true',
                    help='Skip power measurements (Smart Power 2 unavailable)')

args = parser.parse_args()

# Use the arguments to set model path
onnx_model_name = args.onnx_path

# Create Inference session using ONNX runtime
sess_options = rt.SessionOptions()
sess_options.intra_op_num_threads = 4

sess = rt.InferenceSession(onnx_model_name, sess_options)

# Get the input name for the ONNX model
input_name = sess.get_inputs()[0].name
print("Input name  :", input_name)

# Get the shape of the input
input_shape = sess.get_inputs()[0].shape
print("Input shape :", input_shape)

# Mean and standard deviation 
mean = np.array((0.4914, 0.4822, 0.4465))
std = np.array((0.2023, 0.1994, 0.2010))

# Label names for CIFAR10 Dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize metrics tracking
total_correct = 0
total_samples = 0
inference_times = []
power_measurements = []
temperature_measurements = []
timestamps = []

# Get initial RAM usage
process = psutil.Process(os.getpid())
initial_ram = process.memory_info().rss / 1024 / 1024  # Convert to MB
peak_ram = initial_ram

# Initialize telnet connection for power measurement
telnet_connection = None
last_power = 0.0

if not args.no_power:
    print("\nConnecting to Smart Power 2 (192.168.4.1)...")
    try:
        telnet_connection = tel.Telnet("192.168.4.1", timeout=10)
        print("Connected to power meter")
    except Exception as e:
        print(f"\nWARNING: Could not connect to power meter: {e}")
        print("Continuing without power measurements...")
        telnet_connection = None
else:
    print("\nSkipping power measurements (--no-power flag)")

# Helper functions for reading sensors (edge device specific)
def get_telnet_power(telnet_conn, last_pwr):
    """Read power values using telnet from Smart Power 2."""
    if telnet_conn is None:
        return 0.0
    try:
        tel_dat = str(telnet_conn.read_very_eager())
        idx = tel_dat.rfind('\n')
        idx2 = tel_dat[:idx].rfind('\n')
        idx2 = idx2 if idx2 != -1 else 0
        ln = tel_dat[idx2:idx].strip().split(',')
        if len(ln) < 2:
            return last_pwr
        else:
            return float(ln[-2])
    except:
        return last_pwr

def get_temps_mc1():
    """Get temperature values for Odroid MC1 big cores (average of all 4)."""
    try:
        templ = []
        for i in range(4):
            temp = float(open(sysfs.fn_thermal_sensor.format(i), 'r').readline().strip()) / 1000
            templ.append(temp)
        # Swap cores 1 and 3 as per HW requirements
        t1 = templ[1]
        templ[1] = templ[3]
        templ[3] = t1
        # Return average of all 4 big cores
        return sum(templ) / len(templ)
    except:
        return 0.0

def get_temp_raspi():
    """Get CPU temperature for RaspberryPi 3B+."""
    try:
        # Primary method: Use gpiozero as per assignment requirements
        if GPIOZERO_AVAILABLE:
            cpu_temp = gpiozero.CPUTemperature().temperature
            return cpu_temp
        else:
            # Fallback: Read from thermal zone
            temp_paths = [
                '/sys/class/thermal/thermal_zone0/temp',
                '/sys/devices/virtual/thermal/thermal_zone0/temp'
            ]
            for path in temp_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        temp = float(f.read().strip())
                        # Convert from millidegrees to degrees Celsius
                        return temp / 1000.0
        return 0.0
    except Exception as e:
        print(f"Warning: Could not read temperature: {e}")
        return 0.0

def read_power():
    """Read power consumption from device sensors"""
    global last_power
    last_power = get_telnet_power(telnet_connection, last_power)
    return last_power

def read_temperature():
    """Read CPU temperature from device sensors"""
    if args.device == 'mc1':
        return get_temps_mc1()
    else:  # raspi
        return get_temp_raspi()

# Start time for total inference
start_time = time.time()
start_timestamp = start_time

# The test_deployment folder contains all 10,000 images from the testing dataset of CIFAR10 in .png format
print(f"\nStarting inference on test dataset from: {args.test_data_path}")
print(f"Model: {args.model}")

for filename in tqdm(os.listdir(args.test_data_path)):
    # Extract the true label from filename (assuming format: label_index.png)
    # CIFAR10 test images are typically named with their class index
    try:
        true_label_idx = int(filename.split('_')[0])
    except:
        # If filename doesn't contain label, skip accuracy computation for this image
        true_label_idx = None
    
    # Take each image, one by one, and make inference
    with Image.open(os.path.join(args.test_data_path, filename)).resize((32, 32)) as img:
        # Normalize image
        input_image = (np.float32(img) / 255. - mean) / std
        
        # Add the Batch axis in the data Tensor (C, H, W)
        input_image = np.expand_dims(np.float32(input_image), axis=0)

        # Change the order from (B, H, W, C) to (B, C, H, W)
        input_image = input_image.transpose([0, 3, 1, 2])
        
        # Measure power and temperature before inference
        power_before = read_power()
        temp_before = read_temperature()
        
        # Run inference and get the prediction for the input image
        inference_start = time.time()
        pred_onnx = sess.run(None, {input_name: input_image})[0]
        inference_end = time.time()
        
        # Measure power and temperature after inference
        power_after = read_power()
        temp_after = read_temperature()
        
        # Record measurements
        inference_times.append(inference_end - inference_start)
        power_measurements.append((power_before + power_after) / 2)
        temperature_measurements.append((temp_before + temp_after) / 2)
        timestamps.append(inference_end - start_timestamp)

        # Find the prediction with the highest probability
        top_prediction = np.argmax(pred_onnx[0])

        # Get the label of the predicted class
        pred_class = label_names[top_prediction]

        # Compute test accuracy of the model
        if true_label_idx is not None:
            total_samples += 1
            if top_prediction == true_label_idx:
                total_correct += 1
        
        # Update peak RAM usage
        current_ram = process.memory_info().rss / 1024 / 1024
        peak_ram = max(peak_ram, current_ram)

# End time for total inference
end_time = time.time()
total_inference_time = end_time - start_time

# Calculate metrics
test_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
avg_inference_time = np.mean(inference_times) if inference_times else 0.0
avg_power = np.mean(power_measurements) if power_measurements else 0.0
avg_temperature = np.mean(temperature_measurements) if temperature_measurements else 0.0

# Calculate total energy consumption (Power * Time)
# Energy in Joules = Average Power (Watts) * Total Time (seconds)
total_energy = avg_power * total_inference_time

# Print results
print("\n" + "="*60)
print(f"INFERENCE RESULTS FOR {args.model.upper()}")
print("="*60)
print(f"Total Inference Time: {total_inference_time:.2f} seconds")
print(f"Average Inference Time per Image: {avg_inference_time*1000:.2f} ms")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Peak RAM Memory Usage: {peak_ram:.2f} MB")
print(f"Average Power Consumption: {avg_power:.4f} W")
print(f"Total Energy Consumption: {total_energy:.4f} J")
print(f"Average CPU Temperature: {avg_temperature:.2f} °C")
print(f"Total Images Processed: {total_samples}")
print("="*60)

# Save detailed metrics to CSV files
import csv

# Save summary metrics
summary_file = os.path.join(args.output_dir, f'{args.model}_summary_metrics.csv')
with open(summary_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Metric', 'Value'])
    writer.writerow(['Model', args.model])
    writer.writerow(['Total Inference Time (s)', f'{total_inference_time:.2f}'])
    writer.writerow(['Average Inference Time per Image (ms)', f'{avg_inference_time*1000:.2f}'])
    writer.writerow(['Test Accuracy (%)', f'{test_accuracy:.2f}'])
    writer.writerow(['Peak RAM Memory (MB)', f'{peak_ram:.2f}'])
    writer.writerow(['Average Power (W)', f'{avg_power:.4f}'])
    writer.writerow(['Total Energy (J)', f'{total_energy:.4f}'])
    writer.writerow(['Average Temperature (°C)', f'{avg_temperature:.2f}'])
    writer.writerow(['Total Images', total_samples])

print(f"\nSummary metrics saved to: {summary_file}")

# Save time-series data for power and temperature plots
timeseries_file = os.path.join(args.output_dir, f'{args.model}_timeseries_data.csv')
with open(timeseries_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp (s)', 'Power (W)', 'Temperature (°C)', 'Inference Time (s)'])
    for i in range(len(timestamps)):
        writer.writerow([
            f'{timestamps[i]:.4f}',
            f'{power_measurements[i]:.4f}',
            f'{temperature_measurements[i]:.2f}',
            f'{inference_times[i]:.6f}'
        ])

print(f"Time-series data saved to: {timeseries_file}")

# Close telnet connection
if telnet_connection:
    telnet_connection.close()
    print("\nPower meter connection closed")

print("\n" + "="*60)
print("DEPLOYMENT COMPLETE!")
print("="*60)
