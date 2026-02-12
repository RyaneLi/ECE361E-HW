g ECE361E HW1 - Team9 Submission
==============================

This readme describes the contents of the zip file.


HW1_solution.pdf
  Solution write-up: tables, plots, and written answers for all problems.


p1_q1.py
  Problem 1, Question 1: Logistic Regression on MNIST. Prints Table 1 (training/testing accuracy, total training time, inference time, inference per image, GPU memory).
  Run: python p1_q1.py --epochs 25

p1_q2_plot.py
  Problem 1, Question 2: Loss and accuracy plots for Logistic Regression. Saves plot_data.csv and PNGs.
  Run: python p1_q2_plot.py --epochs 25

p2_q1.py
  Problem 2, Question 1: SimpleFC on MNIST with device (GPU/CPU). Loss and accuracy plots.
  Run: python p2_q1.py --epochs 25

p2_q2.py
  Problem 2, Question 2: SimpleFC with dropout (0.0, 0.2, 0.5, 0.8). One loss plot per experiment.
  Run: python p2_q2.py --epochs 25

p2_q3.py
  Problem 2, Question 3: Table 2. Best dropout (0.2) with and without normalization.
  Run: python p2_q3.py --epochs 25 --dropout 0.2

p3_q1.py
  Problem 3, Question 1.

p3_q2.py
  Problem 3, Question 2.


readme.txt
  This file; describes every item in the zip.
