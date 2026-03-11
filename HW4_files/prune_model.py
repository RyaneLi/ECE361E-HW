import argparse
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import random

# Argument parser
parser = argparse.ArgumentParser(description='ECE361E HW4 Pruning')
# TODO: add arguments for model to load, prune_ratio, prune_metric, pruning_iter and finetuning_epochs
args = parser.parse_args()

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

# TODO: Define model and load weights
model = None

batch_size = 128

# CIFAR10 Dataset (Images and Labels)
train_dataset = dsets.CIFAR10(root='data', train=True, transform=transforms.Compose([
    transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),]), download=True)

test_dataset = dsets.CIFAR10(root='data', train=False, transform=transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),]))

# Dataset Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define your loss and optimizer
criterion = nn.CrossEntropyLoss()  # Softmax is internally computed.
optimizer = torch.optim.Adam(model.parameters())

# one epoch of training
def fine_tune(model):
    model = model.train()
    # TODO: Put the model on the GPU
    train_loss = 0
    train_total = 0
    train_correct = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # TODO: Put the images and labels on the GPU
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

# get test set accuracy
def test(model):
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    # TODO: Put the model on the GPU
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            # TODO: Put the images and labels on the GPU
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
    print('Test loss: %.4f Test accuracy: %.2f %%\n' %
          (test_loss / (batch_idx + 1), 100. * test_correct / test_total))


def prune_network(model, prune_metric, prune_ratio, pruning_iter, finetuning_epochs, ignored_layers=None):
    # TODO: Put the model and the example_inputs on the GPU
    example_inputs = torch.randn(1, 3, 32, 32)

    # choose pruning importance metric
    # TODO: Based on prune_metric select the corresponding importance for pruning
    importance = None

    if ignored_layers is None:
        ignored_layers = []
        for name, m in model.named_modules():
            # TODO: add code to ignore certain layers for pruning based on the strategy
            # Note: you can add an argument for pruning_strategy
            # Note: Do NOT prune the final linear layer
            pass

    # initialize the high level pruner
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance,
        iterative_steps=pruning_iter,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers,
    )

    # prune for some number of iterations
    for i in range(pruning_iter):
        pruner.step()  # the actual pruning step

        # TODO check accuracy before finetuning

        # TODO fine tune model
        
        # TODO check accuracy after finetuning

        # TODO Save the model

    # TODO Export pruned model to ONNX using the name
    #  {prune_metric}_{prune_ratio}_{pruning_iter}_{finetuning_epochs}_MBNv1.onnx
    #  Note: use opset_version=16 and make sure to put the model on CPU
