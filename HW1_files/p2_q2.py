# Problem 2, Question 2: SimpleFC with dropout. Run four experiments with dropout
# probabilities [0.0, 0.2, 0.5, 0.8]. Draw one loss plot per experiment.
#
# What to observe: As dropout increases, train and test loss tend to become closer
# (less overfitting), but too much dropout can underfit (both losses stay high).
# Best: dropout that gives almost equal train and test loss (e.g. 0.2 or 0.5).
# Worst: 0.0 often overfits (train loss << test loss); 0.8 may underfit (both high).

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

parser = argparse.ArgumentParser(description='ECE361E HW1 P2 Q2 - SimpleFC dropout experiments')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
args = parser.parse_args()

input_size = 28 * 28
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
DROPOUT_PROBS = [0.0, 0.2, 0.5, 0.8]

random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(random_seed)

train_dataset = dsets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = dsets.MNIST(root='data', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class SimpleFC(nn.Module):
    def __init__(self, input_size, num_classes, dropout_prob=0.0):
        super(SimpleFC, self).__init__()
        self.linear1 = nn.Linear(input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = self.dropout(out)
        out = F.relu(self.linear2(out))
        out = self.dropout(out)
        out = F.relu(self.linear3(out))
        out = self.dropout(out)
        out = self.linear4(out)
        return out


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()


def train_one_experiment(dropout_prob):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    model = SimpleFC(input_size, num_classes, dropout_prob=dropout_prob)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    train_losses = []
    test_losses = []
    for epoch in range(num_epochs):
        model.train()
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
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.view(-1, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
        test_losses.append(test_loss / len(test_loader))
        print('Dropout %.1f Epoch [%d/%d], Train Loss: %.4f, Test Loss: %.4f' %
              (dropout_prob, epoch + 1, num_epochs, train_losses[-1], test_losses[-1]))
    return train_losses, test_losses


def save_loss_plot(train_losses, test_losses, dropout_prob):
    num_epochs_plot = len(train_losses)
    epochs_range = range(1, num_epochs_plot + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, train_losses, label='Training loss')
    plt.plot(epochs_range, test_losses, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('SimpleFC dropout=%.1f: Training and test loss per epoch' % dropout_prob)
    plt.tight_layout()
    filename = 'p2_q2_loss_plot_%.1f.png' % dropout_prob
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


if __name__ == '__main__':
    for p in DROPOUT_PROBS:
        print('--- Dropout probability: %.1f ---' % p)
        train_losses, test_losses = train_one_experiment(p)
        fname = save_loss_plot(train_losses, test_losses, p)
        print('Saved %s' % fname)
    print('Done. Four loss plots: p2_q2_loss_plot_0.0.png, p2_q2_loss_plot_0.2.png, p2_q2_loss_plot_0.5.png, p2_q2_loss_plot_0.8.png')
