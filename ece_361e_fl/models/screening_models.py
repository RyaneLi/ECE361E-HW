import torch
import torch.nn as nn


def _choose_group_norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def _make_activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1)
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def _make_norm(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "batchnorm":
        return nn.BatchNorm2d(channels)
    if norm_type == "groupnorm":
        return nn.GroupNorm(_choose_group_norm_groups(channels), channels)
    if norm_type == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm type: {norm_type}")


class _ScreeningModel(nn.Module):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__()
        self.loss_type = loss_type

    def _return(self, logits: torch.Tensor, activations: torch.Tensor):
        if self.loss_type == "fedmax":
            return logits, activations
        return logits


# Baseline dense model: flattens the image immediately and uses a single linear classifier.
class Dense1Anchor(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(32 * 32, 10)

    def forward(self, x):
        activations = self.flatten(x)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Smaller 2-conv CNN: same idea as SimpleCNN, but with fewer channels to cut compute and energy.
class SimpleCNNSmall(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(32 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Adds a compact MLP head before logits: flatten -> FC(128) -> ReLU -> FC(10).
class SimpleCNNSmallMLPHead(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pre_classifier = nn.Linear(32 * 8 * 8, 128)
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        flat = out.view(out.size(0), -1)
        activations = self.activation(self.pre_classifier(flat))
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Extra-small 2-conv CNN: channel width reduced from (16,32) to (12,24).
class SimpleCNNSmall1224(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(24 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Tiny 2-conv CNN: channel width reduced from (16,32) to (8,16).
class SimpleCNNSmall816(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(16 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Ultra-tiny 2-conv CNN: channel width reduced from (16,32) to (4,8) for lower-bound cost tests.
class SimpleCNNSmall48(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(8 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Single-conv CNN with flattened head: removes the second conv block but keeps spatial detail for classification.
class SimpleCNNSingleConv(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(32 * 16 * 16, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Single-conv variant with an MLP head: flatten -> FC(128) -> ReLU -> FC(10).
class SimpleCNNSingleConvMLPHead(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pre_classifier = nn.Linear(32 * 16 * 16, 128)
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        flat = out.view(out.size(0), -1)
        activations = self.activation(self.pre_classifier(flat))
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Single-conv width variant: channel mapping (1 -> 16) with the original linear head.
class SimpleCNNSingleConv16(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(16 * 16 * 16, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Further reduced single-conv width variant: channel mapping (1 -> 8).
class SimpleCNNSingleConv8(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(8 * 16 * 16, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Minimal practical single-conv width variant for this setup: channel mapping (1 -> 4).
class SimpleCNNSingleConv4(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(4 * 16 * 16, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Baseline single conv: one standard 3x3 conv followed by ReLU, GAP, and classifier.
class Conv2d1Anchor(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.conv = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        out = self.pool(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Adds max pooling before GAP: reduces spatial size more aggressively than the anchor.
class Conv2d1MaxPool(Conv2d1Anchor):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )


# Activation swap: uses LeakyReLU instead of ReLU to test if negative slope helps.
class Conv2d1LeakyReLU(Conv2d1Anchor):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.activation = nn.LeakyReLU(negative_slope=0.1)


# Normalization ablation: adds BatchNorm after the conv to test training stability effects.
class Conv2d1BatchNorm(Conv2d1Anchor):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.norm = nn.BatchNorm2d(16)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.pool(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)

# Normalization ablation: uses GroupNorm instead of BatchNorm, often better for non-IID FL.
class Conv2d1GroupNorm(Conv2d1Anchor):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.norm = nn.GroupNorm(_choose_group_norm_groups(16), 16)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.pool(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Residual-lite variant: adds a skip projection so the block can preserve input information.
class Conv2d1ResidualLite(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.main = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        out = self.main(x) + self.skip(x)
        out = self.activation(out)
        out = self.pool(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Depthwise-separable baseline: expands channels, applies depthwise conv, then mixes channels.
class Depthwise1Anchor(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.expand = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.depthwise = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        out = self.activation(self.expand(x))
        out = self.activation(self.depthwise(out))
        out = self.activation(self.pointwise(out))
        out = self.pool(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Adds max pooling before GAP: compares pooling cost/benefit to the depthwise anchor.
class Depthwise1MaxPool(Depthwise1Anchor):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )


# Activation swap: uses LeakyReLU instead of ReLU in the depthwise block.
class Depthwise1LeakyReLU(Depthwise1Anchor):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.activation = nn.LeakyReLU(negative_slope=0.1)


# Normalization ablation: adds BatchNorm to the depthwise-separable pipeline.
class Depthwise1BatchNorm(Depthwise1Anchor):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.expand_norm = nn.BatchNorm2d(16)
        self.depthwise_norm = nn.BatchNorm2d(16)
        self.pointwise_norm = nn.BatchNorm2d(16)

    def forward(self, x):
        out = self.activation(self.expand_norm(self.expand(x)))
        out = self.activation(self.depthwise_norm(self.depthwise(out)))
        out = self.activation(self.pointwise_norm(self.pointwise(out)))
        out = self.pool(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Normalization ablation: uses GroupNorm, which can be more robust under non-IID FL.
class Depthwise1GroupNorm(Depthwise1Anchor):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.expand_norm = nn.GroupNorm(_choose_group_norm_groups(16), 16)
        self.depthwise_norm = nn.GroupNorm(_choose_group_norm_groups(16), 16)
        self.pointwise_norm = nn.GroupNorm(_choose_group_norm_groups(16), 16)

    def forward(self, x):
        out = self.activation(self.expand_norm(self.expand(x)))
        out = self.activation(self.depthwise_norm(self.depthwise(out)))
        out = self.activation(self.pointwise_norm(self.pointwise(out)))
        out = self.pool(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Residual-lite depthwise variant: adds a skip projection to test residual benefits.
class Depthwise1ResidualLite(_ScreeningModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.expand = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.depthwise = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, groups=16, bias=False)
        self.pointwise = nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.skip = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.activation = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(16, 10)

    def forward(self, x):
        out = self.activation(self.expand(x))
        main = self.activation(self.depthwise(out))
        main = self.pointwise(main)
        out = self.activation(main + self.skip(x))
        out = self.pool(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)