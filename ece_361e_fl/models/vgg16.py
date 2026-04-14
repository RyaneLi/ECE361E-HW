import torch.nn as nn

cfg = {
    'VGG16_MNIST': [16, 16, 'M', 32, 32, 'M', 64, 'M'],  # Lightweight version for MNIST
}


class VGG16(nn.Module):
    def __init__(self, loss_type='fedavg'):
        super(VGG16, self).__init__()
        self.loss_type = loss_type
        self.features = self._make_layers(cfg['VGG16_MNIST'])
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1  # MNIST grayscale
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU()]
                in_channels = x
        return nn.Sequential(*layers)

