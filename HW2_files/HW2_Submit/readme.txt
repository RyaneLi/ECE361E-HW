================================================================================
ECE361E - Homework 2 Submission
================================================================================

Authors: Resul Ovezov, Ryane Li
Resul: Completed Problems 1 and 2
Ryane: Completed Problem 3

================================================================================
CONTENTS
================================================================================

HW2_Complete_Solutions.pdf
    Complete homework solutions document containing all answers, plots, tables,
    and analysis for Problems 1, 2, and 3.

--------------------------------------------------------------------------------
PYTHON MEASUREMENT SCRIPTS
--------------------------------------------------------------------------------

p1_q1_measure_tpbench.py
    Measurement script for Problem 1, Question 1.
    Runs TPBench benchmark on core 4 with LITTLE cluster at 0.2GHz and big 
    cluster at 2GHz. Collects system power, big cores temperature, and big 
    cores usage data.

p1_q3_measure_blackscholes.py
    Measurement script for Problem 1, Question 3.
    Runs the blackscholes benchmark on all 4 big cores (4 threads) with big 
    cluster at 2GHz and LITTLE at 0.2GHz. Collects system power and maximum 
    big core temperature data.

p1_q3_measure_bodytrack.py
    Measurement script for Problem 1, Question 3.
    Runs the bodytrack benchmark on all 4 big cores (4 threads) with big 
    cluster at 2GHz and LITTLE at 0.2GHz. Collects system power and maximum 
    big core temperature data.

--------------------------------------------------------------------------------
JUPYTER NOTEBOOKS
--------------------------------------------------------------------------------

p1.ipynb
    Problem 1 analysis notebook containing:
    - Question 1: TPBench measurement plots (power, temperature, usage)
    - Question 2: Benchmark phase identification analysis
    - Question 3: Blackscholes and bodytrack benchmark analysis with plots
                  and Table 1 metrics

p2.ipynb
    Problem 2 analysis notebook containing:
    - Question 1: SVM classifier for big cluster state (active/idle) with
                  Table 2 metrics and confusion matrices
    - Question 2: Linear regression for power prediction with Table 3 metrics
                  and prediction plots
    - Question 3: Feature engineering with Vdd^2*f and feature importance
                  analysis

p3.ipynb
    Problem 3 analysis notebook containing:
    - Question 1: MLPRegressor models for temperature prediction with plots
                  and Table 5 MSE results
    - Question 2: Discussion of techniques to improve regressor performance

--------------------------------------------------------------------------------
SOURCE FOLDER
--------------------------------------------------------------------------------

source/
    Contains all output data generated from the measurement scripts and 
    analysis notebooks, including:
    
    - Raw measurement logs (.txt files)
    - Generated plots and figures (.png files)
    - Metrics and results tables (.csv files)
    
    This folder contains all the source material referenced in the notebooks
    and compiled in the final PDF document.

