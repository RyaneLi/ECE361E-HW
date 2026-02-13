"""
Question 3: Blackscholes Benchmark Measurement Script

This script:
1. Sets LITTLE cluster to 0.2GHz, big cluster to 2GHz
2. Runs blackscholes on all 4 big cores (4 threads)
3. Measures power and temperature during execution
4. Saves data to q3_blackscholes_log.txt
"""

import psutil
import telnetlib as tel
import sysfs_paths as sysfs
import time
import subprocess
import argparse


def get_telnet_power(telnet_connection, last_power):
    """Read power values using telnet."""
    tel_dat = str(telnet_connection.read_very_eager())
    idx = tel_dat.rfind('\n')
    idx2 = tel_dat[:idx].rfind('\n')
    idx2 = idx2 if idx2 != -1 else 0
    ln = tel_dat[idx2:idx].strip().split(',')
    if len(ln) < 2:
        total_power = last_power
    else:
        total_power = float(ln[-2])
    return total_power


def get_temps():
    """Obtain temperature values for big cores."""
    templ = []
    for i in range(4):
        temp = float(open(sysfs.fn_thermal_sensor.format(i), 'r').readline().strip()) / 1000
        templ.append(temp)
    # Swap cores 1 and 3
    t1 = templ[1]
    templ[1] = templ[3]
    templ[3] = t1
    return templ


def set_user_space():
    """Set the system governor to 'userspace'."""
    clusters = [0, 4]
    for i in clusters:
        with open(sysfs.fn_cluster_gov.format(i), 'w') as f:
            f.write('userspace')


def set_cluster_freq(cluster_num, frequency):
    """Set cluster frequency (in KHz)."""
    with open(sysfs.fn_cluster_freq_set.format(cluster_num), 'w') as f:
        f.write(str(frequency))


# Main execution
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-power', action='store_true',
                        help='Skip power measurements (SP2 unavailable)')
    args = parser.parse_args()
    
    print("Question 3: Blackscholes Benchmark Measurement")
    if args.no_power:
        print("Running WITHOUT power measurements (--no-power flag)")
    
    # Set frequencies
    print("\nSetting userspace governor...")
    set_user_space()
    set_cluster_freq(0, 200000)    # LITTLE: 0.2 GHz
    set_cluster_freq(4, 2000000)   # big: 2 GHz
    print("Frequencies set: LITTLE=0.2GHz, big=2GHz")
    
    # Create log file
    out_fname = 'q3_blackscholes_log.txt'
    header = "time\tpower_W\ttemp4\ttemp5\ttemp6\ttemp7\tmax_temp"
    with open(out_fname, 'w') as out_file:
        out_file.write(header + "\n")
    print(f"Log file created: {out_fname}")
    
    # Start telnet connection
    telnet_connection = None
    total_power = 0.0
    
    if not args.no_power:
        print("\nConnecting to power meter (192.168.4.1)...")
        try:
            telnet_connection = tel.Telnet("192.168.4.1", timeout=10)
            print("Connected")
        except Exception as e:
            print(f"\nERROR: Could not connect to power meter!")
            print(f"   {str(e)}")
            print("\nTo run without power measurements, use: --no-power flag")
            print("Example: sudo python3 p1_q3_measure_blackscholes.py --no-power")
            exit(1)
    else:
        print("\nSkipping power measurements (power will be recorded as 0.0W)")
    
    # Start blackscholes benchmark on all big cores
    # Mask 0xF0 = binary 1111 0000 = cores 4,5,6,7
    # First arg is number of threads (4 for all big cores)
    command = "taskset --all-tasks 0xF0 /home/student/HW2_files/parsec_files/blackscholes 4 /home/student/HW2_files/parsec_files/in_10M_blackscholes.txt /home/student/HW2_files/parsec_files/prices.txt"
    print(f"\nStarting benchmark:")
    print(f"  {command}")
    start_time = time.time()
    proc_ben = subprocess.Popen(command.split())
    print("Blackscholes started on cores 4-7 with 4 threads")
    
    # Measurement loop
    print("\nMeasuring power and temperature every 0.2 seconds...")
    print("Benchmark will run until completion...\n")
    measurement_interval = 0.2  # seconds
    sample_count = 0
    
    while proc_ben.poll() is None:  # While benchmark is running
        current_time = time.time() - start_time
        
        # Get measurements
        if telnet_connection:
            total_power = get_telnet_power(telnet_connection, total_power)
        else:
            total_power = 0.0
        temps = get_temps()
        max_temp = max(temps)
        
        # Write to file
        with open(out_fname, 'a') as out_file:
            line = f"{current_time:.3f}\t{total_power:.3f}"
            for t in temps:
                line += f"\t{t:.2f}"
            line += f"\t{max_temp:.2f}"
            out_file.write(line + "\n")
        
        # Print progress every 10 samples
        if sample_count % 10 == 0:
            print(f"Time: {current_time:6.1f}s | Power: {total_power:5.2f}W | Max Temp: {max_temp:5.1f}°C")
        
        sample_count += 1
        time.sleep(measurement_interval)
    
    # Final measurement after benchmark completes
    current_time = time.time() - start_time
    if telnet_connection:
        total_power = get_telnet_power(telnet_connection, total_power)
    else:
        total_power = 0.0
    temps = get_temps()
    max_temp = max(temps)
    with open(out_fname, 'a') as out_file:
        line = f"{current_time:.3f}\t{total_power:.3f}"
        for t in temps:
            line += f"\t{t:.2f}"
        line += f"\t{max_temp:.2f}"
        out_file.write(line + "\n")
    
    if telnet_connection:
        telnet_connection.close()
    
    print("\nBLACKSCHOLES MEASUREMENT COMPLETE!")
    print(f"Data saved to: {out_fname}")
    print(f"Runtime: {current_time:.2f} seconds")
    print(f"Total samples: {sample_count + 1}")
