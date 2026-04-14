import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFC(nn.Module):
    def __init__(self, loss_type='fedavg'):
        super(SimpleFC, self).__init__()
        self.input_size = 32 * 32
        self.num_classes = 10
        self.loss_type = loss_type
        
        self.linear1 = nn.Linear(self.input_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        # Flatten all dimensions except batch
        x = x.view(x.size(0), -1)
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        activations = F.relu(self.linear3(out))
        logits = self.linear4(activations)

        if self.loss_type == 'fedmax':
            return logits, activations
        return logits
