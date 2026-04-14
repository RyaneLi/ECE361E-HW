import torch
import torch.nn as nn
from models.screening_models import (
    Dense1Anchor,
    SimpleCNNSingleConv,
    SimpleCNNSingleConv4,
    SimpleCNNSingleConv8,
    SimpleCNNSingleConv16,
    SimpleCNNSmall,
    SimpleCNNSmall48,
    SimpleCNNSmall816,
    SimpleCNNSmall1224,
)


class _ResearchModel(nn.Module):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__()
        self.loss_type = loss_type

    def _return(self, logits: torch.Tensor, activations: torch.Tensor):
        if self.loss_type == "fedmax":
            return logits, activations
        return logits



# Dense1Anchor variant: adds one hidden layer (1024 -> 256) before classification.
class Dense1AnchorHidden256(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(32 * 32, 256)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        out = self.flatten(x)
        activations = self.activation(self.hidden(out))
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Dense1Anchor variant: adds two hidden layers (1024 -> 512 -> 256) for higher MLP capacity.
class Dense1AnchorDeep(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(32 * 32, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        out = self.flatten(x)
        out = self.activation(self.hidden1(out))
        activations = self.activation(self.hidden2(out))
        logits = self.classifier(activations)
        return self._return(logits, activations)


# Dense1Anchor variant: adds a lightweight conv+pool front-end before a linear head.
class Dense1AnchorConvLite(_ResearchModel):
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

# Dense1AnchorConvLite variant: adds flatten->linear preprocessing (1024->900), reshape to 30x30, then conv+pool.
class Dense1AnchorConvLiteV2(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.flatten = nn.Flatten()
        self.pre_conv = nn.Linear(32 * 32, 900)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(4 * 15 * 15, self.num_classes)

    def forward(self, x):
        out = self.flatten(x)
        out = self.pre_conv(out)
        out = out.view(out.size(0), 1, 30, 30)
        out = self.activation(self.conv1(out))
        out = self.pool1(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNN variant: narrows channels from 32->64 to 20->40 to reduce compute.
class SimpleCNN2040(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(40 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNN2040 variant: same 20->40 width with BatchNorm after each conv.
class SimpleCNN2040BatchNorm(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(20)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(40)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(40 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.pool2(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNN variant: intermediate width 24->48 between 16->32 and 32->64 baselines.
class SimpleCNN2448(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Linear(48 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNN variant: keeps 32->64 conv trunk but inserts a 4096->128 bottleneck MLP head.
class SimpleCNNBottleneck(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = nn.Linear(64 * 8 * 8, 128)
        self.classifier = nn.Linear(128, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        flattened = out.view(out.size(0), -1)
        activations = self.activation(self.bottleneck(flattened))
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNNSmall48 variant: widens only conv2 from 8 to 16 while keeping conv1 at 4.
class SimpleCNNSmall416(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
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


# SimpleCNNSmall48 variant: adds a third conv layer (8->16) after the second pool.
class SimpleCNNSmall48Deep(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(16 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        out = self.activation(self.conv3(out))
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNNSmall48Deep variant: same depth but with GroupNorm after each conv.
class SimpleCNNSmall48DeepGroupNorm(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=4)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=8)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.classifier = nn.Linear(16 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.gn1(self.conv1(x)))
        out = self.pool1(out)
        out = self.activation(self.gn2(self.conv2(out)))
        out = self.pool2(out)
        out = self.activation(self.gn3(self.conv3(out)))
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNNSmall816 variant: adds a third conv layer (16->24) after existing two conv blocks.
class SimpleCNNSmall816Deep(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(24 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        out = self.activation(self.conv3(out))
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNNSmall1224 variant: adds a third conv layer (24->32) after existing two conv blocks.
class SimpleCNNSmall1224Deep(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(32 * 8 * 8, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        out = self.pool2(out)
        out = self.activation(self.conv3(out))
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


# SimpleCNNSingleConv4 variant: adds a second conv (4->8) while preserving low-width trunk.
class SimpleCNNSingleConv4TwoConv(_ResearchModel):
    def __init__(self, loss_type: str = "fedavg"):
        super().__init__(loss_type=loss_type)
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(8 * 16 * 16, self.num_classes)

    def forward(self, x):
        out = self.activation(self.conv1(x))
        out = self.pool1(out)
        out = self.activation(self.conv2(out))
        activations = out.view(out.size(0), -1)
        logits = self.classifier(activations)
        return self._return(logits, activations)


def get_research_model(model_name: str, loss_type: str = "fedavg") -> nn.Module:
    if model_name == "simplecnn_20_40":
        return SimpleCNN2040(loss_type=loss_type)
    if model_name == "simplecnn_20_40_batchnorm":
        return SimpleCNN2040BatchNorm(loss_type=loss_type)
    if model_name == "simplecnn_24_48":
        return SimpleCNN2448(loss_type=loss_type)
    if model_name == "simplecnn_bottleneck":
        return SimpleCNNBottleneck(loss_type=loss_type)
    if model_name == "simplecnn_singleconv":
        return SimpleCNNSingleConv(loss_type=loss_type)
    if model_name == "simplecnn_singleconv_1_16":
        return SimpleCNNSingleConv16(loss_type=loss_type)
    if model_name == "simplecnn_singleconv_1_8":
        return SimpleCNNSingleConv8(loss_type=loss_type)
    if model_name == "simplecnn_singleconv_1_4":
        return SimpleCNNSingleConv4(loss_type=loss_type)
    if model_name == "simplecnn_singleconv_1_4_twoconv":
        return SimpleCNNSingleConv4TwoConv(loss_type=loss_type)
    if model_name == "simplecnn_small":
        return SimpleCNNSmall(loss_type=loss_type)
    if model_name == "simplecnn_small_12_24":
        return SimpleCNNSmall1224(loss_type=loss_type)
    if model_name == "simplecnn_small_12_24_deep":
        return SimpleCNNSmall1224Deep(loss_type=loss_type)
    if model_name == "simplecnn_small_8_16":
        return SimpleCNNSmall816(loss_type=loss_type)
    if model_name == "simplecnn_small_8_16_deep":
        return SimpleCNNSmall816Deep(loss_type=loss_type)
    if model_name == "simplecnn_small_4_8":
        return SimpleCNNSmall48(loss_type=loss_type)
    if model_name == "simplecnn_small_4_8_deep":
        return SimpleCNNSmall48Deep(loss_type=loss_type)
    if model_name == "simplecnn_small_4_8_deep_groupnorm":
        return SimpleCNNSmall48DeepGroupNorm(loss_type=loss_type)
    if model_name == "simplecnn_small_4_16":
        return SimpleCNNSmall416(loss_type=loss_type)
    if model_name == "simplecnn":
        return SimpleCNN(loss_type=loss_type)
    if model_name == "dense1_anchor":
        return Dense1Anchor(loss_type=loss_type)
    if model_name == "dense1_anchor_hidden_256":
        return Dense1AnchorHidden256(loss_type=loss_type)
    if model_name == "dense1_anchor_deep":
        return Dense1AnchorDeep(loss_type=loss_type)
    if model_name == "dense1_anchor_conv_lite":
        return Dense1AnchorConvLite(loss_type=loss_type)
    if model_name == "dense1_anchor_conv_lite_v2":
        return Dense1AnchorConvLiteV2(loss_type=loss_type)

    raise NotImplementedError(f"[!] ERROR: Research model {model_name} not implemented yet")
