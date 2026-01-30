# Problem 2, Question 1: Run SimpleFC with target device (GPU/CPU), draw loss and accuracy plots.
# Overfitting: If training loss keeps decreasing while test loss increases or plateaus, and/or
# training accuracy becomes much higher than test accuracy, the model is overfitting.

import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='ECE361E HW1 P2 Q1 - SimpleFC with device and plots')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--csv', type=str, default='p2_q1_plot_data.csv', help='CSV file to save plot data')
parser.add_argument('--plot-only', action='store_true', help='Load data from CSV and only generate PNG plots (no training)')
args = parser.parse_args()

CSV_PATH = args.csv
PLOT_ONLY = args.plot_only

def save_plot_data(csv_path, train_losses, test_losses, train_accs, test_accs):
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'test_loss', 'train_acc', 'test_acc'])
        for i in range(len(train_losses)):
            w.writerow([i + 1, train_losses[i], test_losses[i], train_accs[i], test_accs[i]])

def load_plot_data(csv_path):
    train_losses, test_losses, train_accs, test_accs = [], [], [], []
    with open(csv_path, newline='') as f:
        r = csv.DictReader(f)
        for row in r:
            train_losses.append(float(row['train_loss']))
            test_losses.append(float(row['test_loss']))
            train_accs.append(float(row['train_acc']))
            test_accs.append(float(row['test_acc']))
    return train_losses, test_losses, train_accs, test_accs

def make_plots(train_losses, test_losses, train_accs, test_accs):
    num_epochs = len(train_losses)
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_losses, label='Training loss')
    plt.plot(epochs_range, test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('SimpleFC: Training and test loss per epoch')
    plt.tight_layout()
    plt.savefig('p2_q1_loss_plot.png', dpi=150)
    plt.close()
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_accs, label='Training accuracy')
    plt.plot(epochs_range, test_accs, label='Test accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('SimpleFC: Training and test accuracy per epoch')
    plt.tight_layout()
    plt.savefig('p2_q1_accuracy_plot.png', dpi=150)
    plt.close()

if PLOT_ONLY:
    train_losses, test_losses, train_accs, test_accs = load_plot_data(CSV_PATH)
    make_plots(train_losses, test_losses, train_accs, test_accs)
    print('Loaded %s and saved p2_q1_loss_plot.png and p2_q1_accuracy_plot.png' % CSV_PATH)
    raise SystemExit(0)

input_size = 28 * 28
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr

random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(random_seed)

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = F.relu(self.linear3(out))
        out = self.linear4(out)
        return out


# Target device for model, images, and labels
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleFC(input_size, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(num_epochs):
    model.train()
    train_correct = 0
    train_total = 0
    train_loss = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.view(-1, input_size).to(device)
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
    train_losses.append(train_loss / len(train_loader))
    train_accs.append(100.0 * train_correct / train_total)

    model.eval()
    test_correct = 0
    test_total = 0
    test_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.view(-1, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_losses.append(test_loss / len(test_loader))
    test_accs.append(100.0 * test_correct / test_total)
    print('Epoch [%d/%d], Train Loss: %.4f, Train Acc: %.2f%%, Test Loss: %.4f, Test Acc: %.2f%%' %
          (epoch + 1, num_epochs, train_losses[-1], train_accs[-1], test_losses[-1], test_accs[-1]))

save_plot_data(CSV_PATH, train_losses, test_losses, train_accs, test_accs)
print('Saved plot data to %s' % CSV_PATH)
make_plots(train_losses, test_losses, train_accs, test_accs)
print('Saved p2_q1_loss_plot.png and p2_q1_accuracy_plot.png')
