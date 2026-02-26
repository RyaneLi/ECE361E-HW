"""
Helper script to compile Table 2 and Table 3 from deployment results
Reads summary CSV files and generates formatted tables
"""

import pandas as pd
import argparse
import os
from pathlib import Path


def read_summary_metrics(csv_path):
    """
    Read summary metrics from CSV file
    Returns a dictionary of metric: value pairs
    """
    metrics = {}
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            metrics[row['Metric']] = row['Value']
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return metrics


def compile_table2(results_dir):
    """
    Compile Table 2: Inference time, RAM memory, and Accuracy
    
    Expected directory structure:
    results_dir/
        mc1/
            vgg11_summary_metrics.csv
            vgg16_summary_metrics.csv
        raspi/
            vgg11_summary_metrics.csv
            vgg16_summary_metrics.csv
    """
    print("\n" + "="*80)
    print("TABLE 2: Deployment Metrics")
    print("="*80)
    
    models = ['vgg11', 'vgg16']
    devices = ['mc1', 'raspi']
    
    # Create data structure for table
    table_data = {
        'Model': [],
        'MC1 Total Inference Time (s)': [],
        'RaspberryPi Total Inference Time (s)': [],
        'MC1 RAM Memory (MB)': [],
        'RaspberryPi RAM Memory (MB)': [],
        'MC1 Accuracy (%)': [],
        'RaspberryPi Accuracy (%)': []
    }
    
    for model in models:
        table_data['Model'].append(model.upper())
        
        for device in devices:
            csv_path = os.path.join(results_dir, device, f'{model}_summary_metrics.csv')
            
            if os.path.exists(csv_path):
                metrics = read_summary_metrics(csv_path)
                
                # Extract metrics
                inference_time = metrics.get('Total Inference Time (s)', 'N/A')
                ram_memory = metrics.get('Peak RAM Memory (MB)', 'N/A')
                accuracy = metrics.get('Test Accuracy (%)', 'N/A')
                
            else:
                print(f"Warning: {csv_path} not found")
                inference_time = 'N/A'
                ram_memory = 'N/A'
                accuracy = 'N/A'
            
            if device == 'mc1':
                table_data['MC1 Total Inference Time (s)'].append(inference_time)
                table_data['MC1 RAM Memory (MB)'].append(ram_memory)
                table_data['MC1 Accuracy (%)'].append(accuracy)
            else:
                table_data['RaspberryPi Total Inference Time (s)'].append(inference_time)
                table_data['RaspberryPi RAM Memory (MB)'].append(ram_memory)
                table_data['RaspberryPi Accuracy (%)'].append(accuracy)
    
    # Create DataFrame and display
    df = pd.DataFrame(table_data)
    print("\n", df.to_string(index=False))
    
    # Save to CSV
    output_file = os.path.join(results_dir, 'table2_compiled.csv')
    df.to_csv(output_file, index=False)
    print(f"\nTable 2 saved to: {output_file}")
    
    return df


def compile_table3(results_dir):
    """
    Compile Table 3: Total energy consumption
    
    Expected directory structure:
    results_dir/
        mc1/
            vgg11_summary_metrics.csv
            vgg16_summary_metrics.csv
        raspi/
            vgg11_summary_metrics.csv
            vgg16_summary_metrics.csv
    """
    print("\n" + "="*80)
    print("TABLE 3: Energy Consumption")
    print("="*80)
    
    models = ['vgg11', 'vgg16']
    devices = ['mc1', 'raspi']
    
    # Create data structure for table
    table_data = {
        'Model': [],
        'MC1 Total Energy (J)': [],
        'RaspberryPi Total Energy (J)': []
    }
    
    for model in models:
        table_data['Model'].append(model.upper())
        
        for device in devices:
            csv_path = os.path.join(results_dir, device, f'{model}_summary_metrics.csv')
            
            if os.path.exists(csv_path):
                metrics = read_summary_metrics(csv_path)
                energy = metrics.get('Total Energy (J)', 'N/A')
            else:
                print(f"Warning: {csv_path} not found")
                energy = 'N/A'
            
            if device == 'mc1':
                table_data['MC1 Total Energy (J)'].append(energy)
            else:
                table_data['RaspberryPi Total Energy (J)'].append(energy)
    
    # Create DataFrame and display
    df = pd.DataFrame(table_data)
    print("\n", df.to_string(index=False))
    
    # Save to CSV
    output_file = os.path.join(results_dir, 'table3_compiled.csv')
    df.to_csv(output_file, index=False)
    print(f"\nTable 3 saved to: {output_file}")
    
    return df


def generate_comparison_stats(results_dir):
    """
    Generate additional comparison statistics
    """
    print("\n" + "="*80)
    print("ADDITIONAL COMPARISON STATISTICS")
    print("="*80)
    
    models = ['vgg11', 'vgg16']
    devices = ['mc1', 'raspi']
    
    for model in models:
        print(f"\n{model.upper()} Comparison:")
        print("-" * 40)
        
        for device in devices:
            csv_path = os.path.join(results_dir, device, f'{model}_summary_metrics.csv')
            
            if os.path.exists(csv_path):
                metrics = read_summary_metrics(csv_path)
                
                device_name = "Odroid MC1" if device == 'mc1' else "RaspberryPi 3B+"
                print(f"\n{device_name}:")
                
                # Calculate additional metrics
                total_time = float(metrics.get('Total Inference Time (s)', 0))
                total_images = int(metrics.get('Total Images', 0))
                total_energy = float(metrics.get('Total Energy (J)', 0))
                avg_power = float(metrics.get('Average Power (W)', 0))
                
                if total_images > 0:
                    time_per_image = total_time / total_images * 1000  # ms
                    energy_per_image = total_energy / total_images  # J
                    print(f"  Average time per image: {time_per_image:.2f} ms")
                    print(f"  Average energy per image: {energy_per_image:.4f} J")
                    print(f"  Average power consumption: {avg_power:.4f} W")
                    
                    # Throughput
                    throughput = total_images / total_time if total_time > 0 else 0
                    print(f"  Throughput: {throughput:.2f} images/second")


def main():
    parser = argparse.ArgumentParser(description='Compile tables from deployment results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing results (with mc1/ and raspi/ subdirectories)')
    parser.add_argument('--show_stats', action='store_true',
                        help='Show additional comparison statistics')
    
    args = parser.parse_args()
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        print("\nExpected directory structure:")
        print("results_dir/")
        print("  mc1/")
        print("    vgg11_summary_metrics.csv")
        print("    vgg16_summary_metrics.csv")
        print("  raspi/")
        print("    vgg11_summary_metrics.csv")
        print("    vgg16_summary_metrics.csv")
        return
    
    # Compile tables
    table2_df = compile_table2(args.results_dir)
    table3_df = compile_table3(args.results_dir)
    
    if args.show_stats:
        generate_comparison_stats(args.results_dir)
    
    print("\n" + "="*80)
    print("Table compilation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
