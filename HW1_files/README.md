# ECE361E HW1 Files

## Problem 1, Question 1: Table 1

Table 1 reports the Logistic Regression model (Problem 1) using accuracy from the final epoch, plus timing and GPU memory. Run `python p1_q1.py --epochs 25` to reproduce.

| Training accuracy [%] | Testing accuracy [%] | Total time for training [s] | Total time for inference [s] | Average time for inference per image [ms] | GPU memory during training [MB] |
|-----------------------|----------------------|-----------------------------|------------------------------|------------------------------------------|----------------------------------|
| 92.56                 | 92.35                | 199.62                      | 1.26                         | 0.1261                                   | 17.74                             |

- **Accuracy**: From the final epoch of a 25-epoch run (same setup as `plot_data.csv`).
- **Training time**: Measured with `time.time()` over the training phase only (excluding the testing phase each epoch).
- **Inference time**: One pass over the test set after training; only the line `outputs = model(images)` is timed, with `batch_size=1`, `model.eval()`, and `torch.no_grad()`.
- **Average time per image**: Total inference time (seconds) × 1000 / number of test images.
- **GPU memory**: Peak memory during training (`torch.cuda.max_memory_allocated()`); 0 on CPU.

---

## Problem 2, Question 1: Does the SimpleFC model overfit?

**Yes, the model overfits.**

After viewing the resulting plots of p2_q1 (training/test loss and accuracy per epoch):

- **Training loss** decreases steadily to nearly zero (e.g., from ~0.95 at epoch 1 to ~0.0015 at epoch 25), while **test loss** improves at first (down to ~0.07) but then **plateaus** around 0.07–0.08 and does not improve (and even increases slightly) as training continues.
- **Training accuracy** rises to almost 100% (e.g., ~99.99% by epoch 25), while **test accuracy** improves to about 97–98% and then **plateaus** there for the rest of training.

The widening gap between training and test loss (and between very high train accuracy and lower, flat test accuracy) indicates that the model is memorizing the training set rather than learning features that generalize well. The SimpleFC network has many parameters (four fully connected layers), so without regularization (e.g., dropout) it tends to overfit. Adding dropout (as in Problem 2, Question 2) helps reduce this overfitting.
