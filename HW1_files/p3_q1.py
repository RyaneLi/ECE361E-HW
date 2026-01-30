
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
parser = argparse.ArgumentParser(description='ECE361E HW1 - SimpleCNN')
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
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.linear1 = nn.Linear(7 * 7 * 64, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool1(out)
        out = F.relu(self.conv2(out))
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out




model = SimpleCNN(num_classes).to(device)

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
model_path = "simplecnn.pth"
torch.save(model.state_dict(), model_path)
model_size_kb = os.path.getsize(model_path) / 1024
print(f"Saved model size: {model_size_kb:.2f} KB (state_dict)")


# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Training loop
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)
        # Sets the gradients to zero
        optimizer.zero_grad()
        # The actual inference
        outputs = model(images)
        # Compute the loss between the predictions (outputs) and the ground-truth labels
        loss = criterion(outputs, labels)
        # Do backpropagation to update the parameters of your model
        loss.backward()
        # Performs a single optimization step (parameter update)
        optimizer.step()
        train_loss += loss.item()
        # The outputs are one-hot labels, we need to find the actual predicted
        # labels which have the highest output confidence
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
        # Print every 100 steps the following information
        if (batch_idx + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f Acc: %.2f%%' % (epoch + 1, num_epochs, batch_idx + 1,
                                                                             len(train_dataset) // batch_size,
                                                                             train_loss / (batch_idx + 1),
                                                                             100. * train_correct / train_total))
    # Testing phase
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            # Perform the actual inference
            outputs = model(images)
            # Compute the loss
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            # The outputs are one-hot labels, we need to find the actual predicted
            # labels which have the highest output confidence
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    print('Test accuracy: %.2f %% Test loss: %.4f' % (100. * test_correct / test_total, test_loss / (batch_idx + 1)))

# ------------------------------------------------------------
# Model Summary Table
# ------------------------------------------------------------
# | Model name | MACs      | FLOPs     | # parameters | torchsummary (total) size [KB] | Saved model size [KB] |
# |------------|-----------|-----------|--------------|-------------------------------|-----------------------|
# | SimpleCNN  | 3,869,824 | 7,739,648 | 50,186       | 550                           | 199.60                |
# ------------------------------------------------------------
# torchsummary (total) size is from "Estimated Total Size (MB): 0.55" = 550 KB
# Saved model size is from the state_dict file size
# The saved model size is smaller than the torchsummary size because torchsummary 
# estimates the in-memory size required for all parameters (and sometimes activations), 
# while the saved model only contains the raw parameter values stored efficiently in a 
# binary file.

