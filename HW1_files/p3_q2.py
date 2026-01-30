
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import random
import numpy as np

# For model summary and FLOPs/MACs
try:
    from torchsummary import summary
except ImportError:
    summary = None
try:
    from thop import profile
except ImportError:
    profile = None
import os

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW1 - MyCNN')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()


# The number of target classes, you have 10 digits to classify
num_classes = 10

# Set device to CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

# Each experiment you will do will have slightly different results due to the randomness
# of 1. the initialization value for the weights of the model, 2. sampling batches of training data
# 3. numerical algorithms for computation (in CUDA.) In order to have reproducible results,
# we have fixed a random seed to a specific value such that we "control" the randomness.
random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(random_seed) # for data loader shuffling

# MNIST Dataset (Images and Labels)
# TODO: Insert here the normalized MNIST dataset
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
]), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class MyCNN(nn.Module):
    def __init__(self, num_classes):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(7 * 7 * 32, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out




model = MyCNN(num_classes).to(device)

# Print model summary (torchsummary)
print("\n--- Model Summary (torchsummary) ---")
if summary is not None:
    summary(model, (1, 28, 28))
else:
    print("torchsummary not installed. Run 'pip install torchsummary' to see model summary.")

# Print MACs, FLOPs, and parameter count (thop)
print("\n--- Model Complexity (thop) ---")
if profile is not None:
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    print(f"MACs: {macs:,}")
    print(f"FLOPs (approx): {2*macs:,}")
    print(f"# Parameters: {params:,}")
else:
    print("thop not installed. Run 'pip install thop' to see MACs/FLOPs.")

# Print total parameters (alternative way)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters (manual count): {total_params:,}")

# Print model (state_dict) size in KB
model_path = "mycnn.pth"
torch.save(model.state_dict(), model_path)
model_size_kb = os.path.getsize(model_path) / 1024
print(f"Saved model size: {model_size_kb:.2f} KB (state_dict)")


# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


# Lists to store loss and accuracy for each epoch
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (
                epoch + 1, num_epochs, batch_idx + 1,
                len(train_dataset) // batch_size,
                train_loss / (batch_idx + 1),
                100. * train_correct / train_total))
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100. * train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    model = model.eval()
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total
    test_losses.append(avg_test_loss)
    test_accuracies.append(test_acc)

    print('Test accuracy: %.2f %% Test loss: %.4f' % (test_acc, avg_test_loss))


# Print lists for plotting
print("\nTrain losses per epoch:", train_losses)
print("Train accuracies per epoch:", train_accuracies)
print("Test losses per epoch:", test_losses)
print("Test accuracies per epoch:", test_accuracies)

# Plot loss and accuracy curves
try:
    import matplotlib.pyplot as plt
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 5))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss vs. Epochs')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Test Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not installed. Run 'pip install matplotlib' to see plots.")

# ------------------------------------------------------------
# Model Summary Table
# ------------------------------------------------------------
# | Model name |    MACs    |   FLOPs    | # parameters | torchsummary (total) size [KB] | Saved model size [KB] |
# |------------|------------|------------|--------------|-------------------------------|-----------------------|
# | SimpleCNN  | 3,869,824  | 7,739,648  |   50,186     | 550                           | 199.60                |
# | MyCNN      | 1,031,744  | 2,063,488  |   20,490     | 260                           |  83.24                |
# ------------------------------------------------------------
# torchsummary (total) size is from "Estimated Total Size (MB)" (e.g., 0.26 MB = 260 KB)
# Saved model size is from the state_dict file size
# The saved model size is smaller than the torchsummary size because torchsummary 
# estimates the in-memory size required for all parameters (and sometimes activations), 
# while the saved model only contains the raw parameter values stored efficiently in a 
# binary file.
#

# BONUS QUESTION 3: The model architecture for MyCNN is exactly the same as the SimpleCNN
# except that we have reduced the number of channels in each convolutional layer by a factor of 2.
# This results in a significant reduction in MACs, FLOPs, number of parameters, and model size,
# while still maintaining a reasonable accuracy on the MNIST dataset. The memory utilized for storing
# the model is also reduced. The accuracy and loss difference between SimpleCNN and MyCNN is negligible.
# This demonstrates that we can design efficient models with fewer resources while still achieving
# good performance. Since performance between the two models is basically the same and MyCNN is more efficient,
# MyCNN is a better choice for deployment in resource-constrained environments. Another reason why MyCNN would
# be better is because it decreases the likelihood of overfitting due to having fewer parameters to learn. 

