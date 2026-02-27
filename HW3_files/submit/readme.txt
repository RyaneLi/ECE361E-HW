ECE 361E - Homework 3 Submission
Team Members: Ryane Li & Resul Ovezov
===============================================================================

This submission contains all the required files for HW3 on deploying PyTorch 
models to edge devices using ONNX format.

TEAM MEMBER CONTRIBUTIONS:
-------------------------------------------------------------------------
Ryane Li:     Problem 1 - Created VGG16 model architecture, trained VGG11, 
              VGG16, and MobileNet-v1 models on Lonestar6, generated Table 1 
              metrics, and created test accuracy comparison plots.

Resul Ovezov: Problems 2 & 3 - Developed ONNX conversion code, deployed all 
              models (VGG11, VGG16, MobileNet-v1) on RaspberryPi 3B+ and 
              Odroid MC1 edge devices, collected inference metrics, measured 
              power consumption and temperature variations.


FILE DESCRIPTIONS:

0. HW3_Solutions.pdf 
    PDF that contains our answers, tables, and charts.
=========================================================================

PROBLEM 1: PyTorch Evaluation of VGG Models
-------------------------------------------------------------------------

1. main.py
   Training script for VGG11, VGG16, and MobileNet-v1 on CIFAR10 that 
   calculates all Table 1 metrics (accuracy, training time, parameters, 
   FLOPs, GPU memory).

2. config.slurm
   SLURM job script for running parallel model training on TACC Lonestar6 
   GPU nodes.

3. p1_q3_plot.py
   Script that generates the test accuracy comparison plot (VGG11 vs VGG16) 
   for Problem 1 Question 3.

4. p1_q2_vgg11.csv
   Per-epoch test accuracy data for VGG11 model, used for plotting.

5. p1_q2_vgg16.csv
   Per-epoch test accuracy data for VGG16 model, used for plotting.

6. p1_q2_mobilenet.csv
   Per-epoch test accuracy data for MobileNet-v1 model, used for plotting.

7. p1_q3_vgg11_vgg16.png
   Plot showing test accuracy over epochs for VGG11 and VGG16 models (Problem 
   1 Question 3).

8. p1_table1.ipynb
   Jupyter notebook that parses training logs and generates Table 1 with 
   analysis and explanations for Problem 1 Question 3.

9. table1.csv
   Final compiled Table 1 with all training metrics for VGG11, VGG16, and 
   MobileNet-v1.


PROBLEM 2: Deployment on Edge Devices Using ONNX
-------------------------------------------------------------------------

10. convert_onnx.py
    Implements PyTorch to ONNX model conversion for deployment on edge devices 
    (Problem 2 Question 1).

11. deploy_onnx.py
    Deployment script that performs inference on edge devices and measures 
    performance metrics (inference time, RAM, accuracy, power, temperature, 
    energy).

12. vgg11_power_consumption_comparison.png
    Plot comparing power consumption over time for VGG11 on MC1 and RaspberryPi 
    (Problem 2 Question 3).

13. vgg11_temperature_comparison.png
    Plot comparing CPU temperature over time for VGG11 on MC1 and RaspberryPi 
    (Problem 2 Question 3).

14. p2.ipynb
    Jupyter notebook that compiles Tables 2 & 3, generates power/temperature 
    plots, and contains explanations for Problems 2 & 3.

15. table2.csv
    Final compiled Table 2 with inference performance metrics (time, RAM, 
    accuracy) for all models on both devices.

16. table3.csv
    Final compiled Table 3 with energy consumption data for all models on 
    both devices.

===============================================================================
End of README
