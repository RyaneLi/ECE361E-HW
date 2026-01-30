# ECE361E HW1 Files

## Problem 1, Question 1: Table 1

Table 1 reports the Logistic Regression model (Problem 1) using accuracy from the final epoch, plus timing and GPU memory. Run `python p1_q1.py --epochs 25` to reproduce.

| Training accuracy [%] | Testing accuracy [%] | Total time for training [s] | Total time for inference [s] | Average time for inference per image [ms] | GPU memory during training [MB] |
|-----------------------|----------------------|-----------------------------|------------------------------|------------------------------------------|----------------------------------|
| 92.56                 | 92.35                | 199.62                      | 1.26                         | 0.1261                                   | 17.74                             |

---

## Problem 2, Question 1: Does the SimpleFC model overfit?

**Yes, the model overfits.**

After viewing the resulting plots of p2_q1 (training/test loss and accuracy per epoch):

- **Training loss** decreases steadily to nearly zero (e.g., from ~0.95 at epoch 1 to ~0.0015 at epoch 25), while **test loss** improves at first (down to ~0.07) but then **plateaus** around 0.07–0.08 and does not improve (and even increases slightly) as training continues.
- **Training accuracy** rises to almost 100% (e.g., ~99.99% by epoch 25), while **test accuracy** improves to about 97–98% and then **plateaus** there for the rest of training.

The widening gap between training and test loss (and between very high train accuracy and lower, flat test accuracy) indicates that the model is memorizing the training set rather than learning features that generalize well. The SimpleFC network has many parameters (four fully connected layers), so without regularization (e.g., dropout) it tends to overfit. Adding dropout (as in Problem 2, Question 2) helps reduce this overfitting.

---

## Problem 2, Question 2: Dropout experiments — what do you observe? Best/worst dropout?

**What do you observe from the loss plots?**

Across the four experiments (dropout 0.0, 0.2, 0.5, 0.8), the loss plots show that as dropout increases, the **gap between training loss and test loss shrinks**: training loss no longer drops to nearly zero while test loss stays high. So dropout reduces overfitting. If dropout is too high, both losses stay higher and the model underfits.

**Which dropout probability gives the best results?**

The **best** results (no overfitting, i.e., almost equal train and test loss values) typically come from **dropout 0.2 or 0.5**. In those plots, the training and test loss curves are close together and both decrease to a similar level, so the model generalizes well without memorizing the training set.

**Which dropout probability gives the worst results?**

The **worst** results come from two extremes:

- **Dropout 0.0**: Strong overfitting. Training loss drops to nearly zero while test loss plateaus or increases; the gap between the two curves is large. The model memorizes the training data and does not generalize well.
- **Dropout 0.8**: Underfitting. Both training and test loss stay relatively high because too many neurons are dropped each epoch; the model cannot learn the task well.

**Explanation:** Dropout randomly turns off neurons during training, so the model cannot rely on any single neuron and is encouraged to learn more robust features. With no dropout (0.0), the model overfits. With moderate dropout (0.2 or 0.5), train and test loss stay close and performance is best. With very high dropout (0.8), the model is over-regularized and underfits.

---

## Problem 2, Question 3: Table 2 (dropout 0.2 vs 0.2 + norm)

Table 2 uses the dropout rate that led to the best results in Problem 2, Question 2 (**0.2**). Run `python p2_q3.py --epochs 25 --dropout 0.2` to reproduce.

| Dropout    | Training accuracy [%] | Testing accuracy [%] | Total time for training [s] | First epoch reaching 96% train acc |
|------------|------------------------|----------------------|-----------------------------|-------------------------------------|
| 0.2        | 99.43                  | 98.30                | 196.49                      | 5                                    |
| 0.2 + norm | 99.63                  | 98.30                | 283.47                      | 3                                    |

**Normalization:** For the row "0.2 + norm", both training and testing datasets use `Compose(ToTensor(), Normalize(mean=0.1307, std=0.3081))` (MNIST mean and std). This standardizes inputs to roughly zero mean and unit variance.

**Compare and contrast — normalized vs unnormalized:**

- **Convergence speed:** With normalization (0.2 + norm), the model reaches 96% training accuracy by **epoch 3**, versus **epoch 5** without normalization (0.2). So normalization helps the model learn faster; inputs in a consistent scale make optimization easier.
- **Final accuracy:** Both setups reach about the same **test accuracy (98.30%)**. Final training accuracy is slightly higher with norm (99.63% vs 99.43%), but the main gain is faster convergence.
- **Training time:** Total training time is higher with normalization (283.47 s vs 196.49 s) because the Normalize transform adds a bit of work per batch; the benefit of normalization is fewer epochs needed to reach a given accuracy, not necessarily lower wall-clock time per run.
- **Explanation:** Normalization (here, standardization with mean 0.1307 and std 0.3081) puts inputs in a range that networks handle well, so gradients and updates are better behaved. That typically speeds up convergence (e.g., 96% train acc in 3 epochs instead of 5) and can improve generalization; in this experiment test accuracy is the same, but the normalized run gets there earlier in terms of epochs.
