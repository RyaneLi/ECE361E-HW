import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
import time
import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from vgg11 import VGG11
from vgg16 import VGG16
from mobilenet import MobileNetv1

try:
    from thop import profile
except ImportError:
    profile = None

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW3 - Starter PyTorch code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
# TODO: Add argument for choosing the model
parser.add_argument('--model', type=str, default='vgg11', choices=['vgg11', 'vgg16', 'mobilenet'], help='Model to train')
args = parser.parse_args()

# Always make assignments to local variables from your args at the beginning of your code for better
# control and adaptability
num_epochs = args.epochs
batch_size = args.batch_size
model_name = args.model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Get model
if model_name == 'vgg11':
    model = VGG11()
elif model_name == 'mobilenet':
    model = MobileNetv1(num_classes=10)
else:
    model = VGG16()
# TODO: Put the model on the GPU
model = model.to(device)

# Compute Table 1 metrics (params, FLOPs) before training
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
if profile is not None:
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    macs, _ = profile(model, inputs=(dummy_input,), verbose=False)
    flops = 2 * macs
else:
    flops = None

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

# For Table 1 and Q3 plot: track test accuracy per epoch
test_acc_history = []

# For Table 1: track peak GPU memory via nvidia-smi during training
gpu_memory_samples = []

def get_gpu_memory_mb():
    """Get GPU memory used in MB via nvidia-smi."""
    if device.type != 'cuda':
        return 0
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    if ',' in str(gpu_id):
        gpu_id = str(gpu_id).split(',')[0].strip()
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', '-i', str(gpu_id)],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip().split('\n')[0].strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return 0

# Training loop
start_time = time.time()
for epoch in range(num_epochs):
    # Training phase
    train_correct = 0
    train_total = 0
    train_loss = 0
    # Sets the model in training mode.
    model = model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # TODO: Put the images and labels on the GPU
        images, labels = images.to(device), labels.to(device)

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
        # Sample GPU memory via nvidia-smi during training (after first batch when memory is high)
        if device.type == 'cuda' and batch_idx == 0:
            gpu_memory_samples.append(get_gpu_memory_mb())
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
            # TODO: Put the images and labels on the GPU
            images, labels = images.to(device), labels.to(device)

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
    test_acc = 100. * test_correct / test_total
    test_acc_history.append(test_acc)
    print('Test loss: %.4f Test accuracy: %.2f %%' % (test_loss / (batch_idx + 1), test_acc))

# Training complete - compute Table 1 metrics
total_time = time.time() - start_time
train_acc = 100. * train_correct / train_total

gpu_memory_mb = max(gpu_memory_samples) if gpu_memory_samples else 0

# Print Table 1 results
print('\n--- Table 1 Results ---')
print('Model: %s' % model_name.upper())
print('Training accuracy [%%]: %.2f' % train_acc)
print('Test accuracy [%%]: %.2f' % test_acc)
print('Total time for training [s]: %.2f' % total_time)
print('Number of trainable parameters: %s' % f'{num_params:,}')
if flops is not None:
    print('FLOPs: %.1fM' % (flops / 1e6))
print('GPU memory during training [MB]: %.2f' % gpu_memory_mb)

# TODO: Save the PyTorch model in .pt format
model_path = os.path.join(os.path.dirname(__file__), '%s.pt' % model_name)
torch.save(model.state_dict(), model_path)
print('Model saved to %s' % model_path)

# Save CSV for Q3 plot (epoch, test_accuracy)
csv_path = os.path.join(os.path.dirname(__file__), 'p1_q2_%s.csv' % model_name)
with open(csv_path, 'w') as f:
    f.write('epoch,test_accuracy\n')
    for ep, acc in enumerate(test_acc_history, 1):
        f.write('%d,%.4f\n' % (ep, acc))
print('Plot data saved to %s' % csv_path)