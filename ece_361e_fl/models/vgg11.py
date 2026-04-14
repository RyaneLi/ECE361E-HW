import torch.nn as nn

cfg = {
    'VGG11_MNIST': [32, 'M', 64, 'M', 128, 'M'],  # Lightweight version for MNIST
}


class VGG11(nn.Module):
    def __init__(self, loss_type='fedavg'):
        super(VGG11, self).__init__()
        self.loss_type = loss_type
        self.features = self._make_layers(cfg['VGG11_MNIST'])
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
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

