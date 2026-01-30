import time
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import random
import numpy as np

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW1 - Starter code')
# Define the mini-batch size, here the size is 128 images per batch
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
# Define the number of epochs for training
parser.add_argument('--epochs', type=int, default=1, help='Number of epoch to train')
# Define the learning rate of your optimizer
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

# The size of input features
input_size = 28 * 28
# The number of target classes, you have 10 digits to classify
num_classes = 10

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
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


# Define your model
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    # Your model only contains a single linear layer
    def forward(self, x):
        out = self.linear(x)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LogisticRegression(input_size, num_classes)
model = model.to(device) 

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Reset GPU memory stats before training (for Table 1; see Appendix A1.3)
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
total_train_time_s = 0.0

# Training loop
for epoch in range(num_epochs):
    # Training phase (only this phase is timed for "Total time for training")
    model = model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0
    train_phase_start = time.time()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Here we vectorize the 28*28 images as several 784-dimensional inputs
        images = images.view(-1, input_size).to(device)
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
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    total_train_time_s += time.time() - train_phase_start
    # Testing phase (not included in training time)
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # Here we vectorize the 28*28 images as several 784-dimensional inputs
            images = images.view(-1, input_size).to(device)
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
    train_acc_final = 100. * train_correct / train_total
    test_acc_final = 100. * test_correct / test_total
    print('Test accuracy: %.2f %% Test loss: %.4f' % (test_acc_final, test_loss / (batch_idx + 1)))

# GPU memory during training (Table 1; Appendix A1.3)
gpu_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0

# Inference time: after training, measure only the line outputs = model(images)
# Use batch_size=1, model.eval(), and torch.no_grad() (one image at a time)
test_loader_bs1 = torch.utils.data.DataLoader(
    dataset=test_dataset, batch_size=1, shuffle=False
)
model = model.eval()
num_test_images = len(test_dataset)
total_inference_time_s = 0.0
with torch.no_grad():
    for images, labels in test_loader_bs1:
        images = images.view(-1, input_size).to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.time()
        outputs = model(images)  # only this line is timed
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        total_inference_time_s += time.time() - t0
inference_per_image_ms = (total_inference_time_s * 1000) / num_test_images

# Table 1
print('')
print('Table 1')
print('Training accuracy [%%]     %.2f' % train_acc_final)
print('Testing accuracy [%%]      %.2f' % test_acc_final)
print('Total time for training [s]     %.2f' % total_train_time_s)
print('Total time for inference [s]    %.2f' % total_inference_time_s)
print('Average time for inference per image [ms]   %.4f' % inference_per_image_ms)
print('GPU memory during training [MB] %.2f' % gpu_mem_mb)