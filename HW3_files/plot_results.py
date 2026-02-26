"""
Plotting script for Problem 2 Question 3
Generates plots for power consumption and temperature over time for both devices
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def plot_power_consumption(mc1_data, raspi_data, output_dir='./'):
    """
    Plot power consumption over time for both MC1 and RaspberryPi
    
    Args:
        mc1_data: DataFrame with MC1 time-series data
        raspi_data: DataFrame with RaspberryPi time-series data
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(mc1_data['Timestamp (s)'], mc1_data['Power (W)'], 
             label='Odroid MC1', linewidth=2, alpha=0.7)
    plt.plot(raspi_data['Timestamp (s)'], raspi_data['Power (W)'], 
             label='RaspberryPi 3B+', linewidth=2, alpha=0.7)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Power (W)', fontsize=12)
    plt.title('Power Consumption Over Time - VGG11 Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'vgg11_power_consumption_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Power consumption plot saved to: {output_file}")
    plt.show()


def plot_temperature(mc1_data, raspi_data, output_dir='./'):
    """
    Plot CPU temperature over time for both MC1 and RaspberryPi
    
    Args:
        mc1_data: DataFrame with MC1 time-series data
        raspi_data: DataFrame with RaspberryPi time-series data
        output_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(mc1_data['Timestamp (s)'], mc1_data['Temperature (°C)'], 
             label='Odroid MC1', linewidth=2, alpha=0.7)
    plt.plot(raspi_data['Timestamp (s)'], raspi_data['Temperature (°C)'], 
             label='RaspberryPi 3B+', linewidth=2, alpha=0.7)
    
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.title('CPU Temperature Over Time - VGG11 Model', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'vgg11_temperature_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Temperature plot saved to: {output_file}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Generate plots for Problem 2 Question 3')
    parser.add_argument('--mc1_data', type=str, required=True,
                        help='Path to MC1 time-series CSV file')
    parser.add_argument('--raspi_data', type=str, required=True,
                        help='Path to RaspberryPi time-series CSV file')
    parser.add_argument('--output_dir', type=str, default='./',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print(f"Loading MC1 data from: {args.mc1_data}")
    mc1_df = pd.read_csv(args.mc1_data)
    
    print(f"Loading RaspberryPi data from: {args.raspi_data}")
    raspi_df = pd.read_csv(args.raspi_data)
    
    # Generate plots
    print("\nGenerating power consumption plot...")
    plot_power_consumption(mc1_df, raspi_df, args.output_dir)
    
    print("\nGenerating temperature plot...")
    plot_temperature(mc1_df, raspi_df, args.output_dir)
    
    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
