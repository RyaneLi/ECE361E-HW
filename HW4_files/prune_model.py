import argparse
import os
import torch
import torch.nn as nn
import torch_pruning as tp
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import random

from mobilenet import MobileNetv1

# Argument parser
# TODO: add arguments for model to load, prune_ratio, prune_metric, pruning_iter and finetuning_epochs
parser = argparse.ArgumentParser(description='ECE361E HW4 Pruning')
parser.add_argument('--model', type=str, default='MobilenetV1.pth', help='Path to model weights')
parser.add_argument('--prune_ratio', type=float, default=0.05, help='Pruning fraction')
parser.add_argument('--prune_metric', type=str, default='l1', help='Pruning metric (l1)')
parser.add_argument('--pruning_iter', type=int, default=5, help='Number of pruning iterations')
parser.add_argument('--finetuning_epochs', type=int, default=5, help='Fine-tuning epochs per iteration')
parser.add_argument('--pruning_strat', type=int, default=1, choices=[1, 2], help='Pruning strategy (1=all except linear, 2=all except conv1/block1/linear)')
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
# Define model and load weights
model = MobileNetv1()
model.load_state_dict(torch.load(args.model))

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

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# one epoch of training
def fine_tune(model, epoch=None, total_epochs=None):
    epoch_str = f' (Epoch {epoch}/{total_epochs})' if epoch is not None else ''
    print(f'  [Fine-tune{epoch_str}]', flush=True)
    model = model.train()
    model = model.to('cuda')
    train_loss = 0
    train_total = 0
    train_correct = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to('cuda'), labels.to('cuda')
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
        if (batch_idx + 1) % 50 == 0:
            print(f'    Step [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {train_loss/(batch_idx+1):.4f} '
                  f'Acc: {100.*train_correct/train_total:.2f}%', flush=True)
    print(f'=> Loss: {train_loss/len(train_loader):.4f} Acc: {100.*train_correct/train_total:.2f}%')

# get test set accuracy
def test(model):
    test_correct = 0
    test_total = 0
    test_loss = 0
    # Sets the model in evaluation mode
    model = model.eval()
    model = model.to('cuda')
    # Disabling gradient calculation is useful for inference.
    # It will reduce memory consumption for computations.
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to('cuda'), labels.to('cuda')
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
    accuracy = 100. * test_correct / test_total
    print('Test loss: %.4f Test accuracy: %.2f %%\n' %
          (test_loss / (batch_idx + 1), accuracy))
    return accuracy


def prune_network(model, prune_metric, prune_ratio, pruning_iter, finetuning_epochs, pruning_strat=1, ignored_layers=None):
    global optimizer
    strat_folder = f'pruned_models/prune_strat_{pruning_strat}'
    os.makedirs(strat_folder, exist_ok=True)

    model = model.to('cuda')
    example_inputs = torch.randn(1, 3, 32, 32).to('cuda')
    baseline_params = count_params(model)
    print(f'\n{"="*60}')
    print(f'  Pruning strategy: {pruning_strat}')
    print(f'  Pruning config: metric={prune_metric}, ratio={prune_ratio}, '
          f'iters={pruning_iter}, ft_epochs={finetuning_epochs}')
    print(f'  Baseline parameters: {baseline_params:,}')
    print(f'{"="*60}')

    print('  Baseline accuracy (before any pruning):')
    baseline_acc = test(model)

    # choose pruning importance metric based on prune_metric
    if prune_metric == 'l1':
        importance = tp.importance.MagnitudeImportance(p=1)
    else:
        importance = tp.importance.MagnitudeImportance(p=1)  # default to L1

    if ignored_layers is None:
        ignored_layers = []
        for name, m in model.named_modules():
            # Do NOT prune the final linear layer (both strategies)
            if isinstance(m, nn.Linear):
                ignored_layers.append(m)
        if pruning_strat == 2:
            # Strategy 2: also ignore the first conv layer and first conv block
            ignored_layers.append(model.conv1)
            ignored_layers.append(model.layers[0])

    # initialize the high level pruner
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=importance,
        iterative_steps=pruning_iter,
        pruning_ratio=prune_ratio,
        ignored_layers=ignored_layers,
    )

    final_acc = baseline_acc
    params_after_prune = baseline_params

    # prune for some number of iterations
    for i in range(pruning_iter):
        print(f'\n--- Pruning Iteration {i+1}/{pruning_iter} ---')
        pruner.step()  # the actual pruning step
        params_after_prune = count_params(model)
        print(f'  Parameters after pruning: {params_after_prune:,} '
              f'({100.*params_after_prune/baseline_params:.1f}% of baseline)')

        # recreate optimizer for the modified (pruned) model parameters
        optimizer = torch.optim.Adam(model.parameters())

        # check accuracy before finetuning
        print(f'  Accuracy before fine-tuning:')
        test(model)

        # fine tune model
        print(f'  Fine-tuning for {finetuning_epochs} epoch(s)...')
        for epoch in range(finetuning_epochs):
            fine_tune(model, epoch=epoch+1, total_epochs=finetuning_epochs)

        # check accuracy after finetuning
        print(f'  Accuracy after fine-tuning:')
        final_acc = test(model)

        # Save the model after this iteration
        ckpt_name = f'{strat_folder}/{prune_metric}_{prune_ratio}_{pruning_iter}_{finetuning_epochs}_MBNv1_iter{i+1}.pth'
        torch.save(model.state_dict(), ckpt_name)
        print(f'  Checkpoint saved: {ckpt_name}')

    # Save results summary CSV for plotting
    result_file = f'{strat_folder}/result_{prune_ratio}.csv'
    with open(result_file, 'w') as f:
        f.write('pruning_fraction,final_accuracy,final_params,baseline_accuracy,baseline_params\n')
        f.write(f'{prune_ratio},{final_acc:.2f},{params_after_prune},{baseline_acc:.2f},{baseline_params}\n')
    print(f'  Results saved: {result_file}')

    # Export pruned model to ONNX (opset_version=16, model must be on CPU)
    model = model.cpu()
    example_inputs_cpu = torch.randn(1, 3, 32, 32)
    onnx_name = f'{strat_folder}/{prune_metric}_{prune_ratio}_{pruning_iter}_{finetuning_epochs}_MBNv1.onnx'
    torch.onnx.export(model, example_inputs_cpu, onnx_name, opset_version=16)
    print(f'Exported ONNX model: {onnx_name}')


if __name__ == '__main__':
    prune_network(model, args.prune_metric, args.prune_ratio, args.pruning_iter, args.finetuning_epochs,
                  pruning_strat=args.pruning_strat)
