# Problem 2, Question 3: Table 2. Run P2 Q2-style code with only the best dropout rate (X).
# Row "X": no normalization. Row "X + norm": same dropout + Normalize(mean=0.1307, std=0.3081).
# Replace BEST_DROPOUT with the dropout that gave the best results in P2 Q2 (e.g. 0.2 or 0.5).

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import argparse
import random
import numpy as np

parser = argparse.ArgumentParser(description='ECE361E HW1 P2 Q3 - Table 2 (best dropout, with/without norm)')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per mini-batch')
parser.add_argument('--epochs', type=int, default=25, help='Number of epoch to train')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--dropout', type=float, default=0.2, help='Best dropout rate from P2 Q2 (X in Table 2)')
args = parser.parse_args()

input_size = 28 * 28
num_classes = 10
num_epochs = args.epochs
batch_size = args.batch_size
learning_rate = args.lr
BEST_DROPOUT = args.dropout  # X in Table 2

random_seed = 1
torch.manual_seed(random_seed)
random.seed(random_seed)
np.random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
g = torch.Generator()
g.manual_seed(random_seed)

# Transforms: no norm (ToTensor only) vs norm (ToTensor + Normalize)
transform_no_norm = transforms.ToTensor()
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()


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


def run_experiment(use_normalization, train_loader, test_loader):
    """Run one experiment; return train_acc_final, test_acc_final, total_train_time_s, first_epoch_96."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    model = SimpleFC(input_size, num_classes, dropout_prob=BEST_DROPOUT)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    total_train_time_s = 0.0
    train_accs = []
    test_accs = []
    first_epoch_96 = None  # first epoch where train_acc >= 96

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0.0
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        t0 = time.time()
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
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        total_train_time_s += time.time() - t0
        train_acc = 100.0 * train_correct / train_total
        train_accs.append(train_acc)
        if first_epoch_96 is None and train_acc >= 96.0:
            first_epoch_96 = epoch + 1

        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images = images.view(-1, input_size).to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100.0 * test_correct / test_total
        test_accs.append(test_acc)

        label = 'norm' if use_normalization else 'no norm'
        print('[%s] Epoch [%d/%d], Train Acc: %.2f%%, Test Acc: %.2f%%' %
              (label, epoch + 1, num_epochs, train_acc, test_acc))

    train_acc_final = train_accs[-1]
    test_acc_final = test_accs[-1]
    if first_epoch_96 is None:
        first_epoch_96 = '>' + str(num_epochs)
    return train_acc_final, test_acc_final, total_train_time_s, first_epoch_96


def main():
    # Row X: no normalization
    train_ds = dsets.MNIST(root='data', train=True, transform=transform_no_norm, download=True)
    test_ds = dsets.MNIST(root='data', train=False, transform=transform_no_norm)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print('--- Row X (dropout=%.1f, no norm) ---' % BEST_DROPOUT)
    train_acc_x, test_acc_x, time_x, epoch_96_x = run_experiment(False, train_loader, test_loader)

    # Row X + norm: with normalization
    train_ds_n = dsets.MNIST(root='data', train=True, transform=transform_norm, download=True)
    test_ds_n = dsets.MNIST(root='data', train=False, transform=transform_norm)
    train_loader_n = torch.utils.data.DataLoader(train_ds_n, batch_size=batch_size, shuffle=True, generator=g)
    test_loader_n = torch.utils.data.DataLoader(test_ds_n, batch_size=batch_size, shuffle=False)
    print('--- Row X + norm (dropout=%.1f, with norm) ---' % BEST_DROPOUT)
    train_acc_n, test_acc_n, time_n, epoch_96_n = run_experiment(True, train_loader_n, test_loader_n)

    # Table 2
    print('')
    print('Table 2 (X = dropout %.1f)' % BEST_DROPOUT)
    print('Dropout        | Training acc [%%] | Testing acc [%%] | Total time training [s] | First epoch reaching 96%% train acc')
    print('---------------|-------------------|------------------|--------------------------|--------------------------------------')
    print('X              | %16.2f | %15.2f | %24.2f | %s' %
          (train_acc_x, test_acc_x, time_x, epoch_96_x))
    print('X + norm       | %16.2f | %15.2f | %24.2f | %s' %
          (train_acc_n, test_acc_n, time_n, epoch_96_n))
    print('')
    print('Normalized (X + norm): data transformed with Compose(ToTensor(), Normalize(mean=0.1307, std=0.3081)).')
    print('Typically normalization helps the model learn faster (96%% train acc reached earlier) and can improve generalization.')


if __name__ == '__main__':
    main()
