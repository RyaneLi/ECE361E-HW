import torch
import torch.nn as nn


class _RefinedModel(nn.Module):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__()
		self.loss_type = loss_type

	def _return(self, logits: torch.Tensor, activations: torch.Tensor):
		if self.loss_type == "fedmax":
			return logits, activations
		return logits


# Experiment: 142
# Config: model=simplecnn_small, run=1, data_iid=false, lr=0.01,
# loss_type=fedmax, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: increase accuracy by +1%.
# Suggestion: Add batch norm.
# experiment suggestions: simplecnn_small but beta = 15, then RefinedSimpleCNNSmall with fedmax
class RefinedSimpleCNNSmall(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(16)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(32 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Experiment: 143
# Config: model=simplecnn_small_12_24, run=1, data_iid=false, lr=0.01,
# loss_type=fedprox, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: increase accuracy by +1%.
# Suggestion: Add batch norm.
# experiment suggestions: simplecnn_small_12_24 but mu = 1.5, then RefinedSimpleCNNSmall1224 with fedprox and mu = 1
class RefinedSimpleCNNSmall1224(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 12, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(12)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(24)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(24 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Experiment: 153
# Config: model=simplecnn_20_40, run=1, data_iid=false, lr=0.01,
# loss_type=fedavg, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: increase accuracy by +1%.
# suggestion: Add batch norm.
# experiment suggestions: simplecnn_20_40 but fedmax with beta = 15, then RefinedSimpleCNN2040 with fedavg, then RefinedSimpleCNN2040 with fedmax, then fedprox
class RefinedSimpleCNN2040(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(20)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(40)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(40 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Experiment: 155
# Config: model=simplecnn_24_48, run=1, data_iid=false, lr=0.01,
# loss_type=fedavg, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: increase accuracy by +1%.
# suggestion: Add batch norm.
# experiment suggestions: simplecnn_24_48 but fedmax with beta = 15, then RefinedSimpleCNN2448 with fedavg, then RefinedSimpleCNN2448 with fedmax, then fedprox
class RefinedSimpleCNN2448(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(24)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(48)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(48 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Experiment: 145
# Config: model=simplecnn_small_8_16, run=1, data_iid=false, lr=0.01,
# loss_type=fedprox, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: increase accuracy by +2.5%.
# Suggestion: Add batch norm.
# experiment suggestions: simplecnn_small_8_16 but mu = 1.5, then RefinedSimpleCNNSmall816 with fedavg, fedprox and fedmax
class RefinedSimpleCNNSmall816(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(8)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(16)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(16 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Experiment: 147
# Config: model=simplecnn_small_4_8, run=1, data_iid=false, lr=0.01,
# loss_type=fedprox, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: special attention and increase accuracy by +2%.
# Suggestion: Add batch norm.
# experiment suggestions: simplecnn_small_4_8 but mu = 1.5, then fedmax with beta = 15, then RefinedSimpleCNNSmall48 with fedavg, then RefinedSimpleCNNSmall48 with fedmax, then fedprox
class RefinedSimpleCNNSmall48(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(8 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Experiment: 158
# Config: model=simplecnn_small_4_8_deep, run=1, data_iid=false, lr=0.01,
# loss_type=fedavg, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: special attention and increase accuracy by +2%.

# experiment suggestions: simplecnn_small_4_8_deep but fedmax with beta = 15, then RefinedSimpleCNNSmall48Deep with fedavg, then RefinedSimpleCNNSmall48Deep with fedmax, then fedprox
class RefinedSimpleCNNSmall48Deep(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(16)
		self.classifier = nn.Linear(16 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		out = self.activation(self.conv3(out))
		out = self.bn3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Experiment: 159
# Config: model=simplecnn_small_4_8_deep_groupnorm, run=1, data_iid=false, lr=0.01,
# loss_type=fedavg, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: special attention and increase accuracy by +2%.
# Suggestion: Add batch norm and another layer.
# experiment suggestions: simplecnn_small_4_8_deep_groupnorm but fedmax with beta = 15, then RefinedSimpleCNNSmall48Deeper with fedavg, then RefinedSimpleCNNSmall48Deeper with fedmax, then fedprox
class RefinedSimpleCNNSmall48Deeper(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.activation = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(16)
		self.conv4 = nn.Conv2d(16, 24, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(24)
		self.classifier = nn.Linear(24 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.conv2(out)))
		out = self.pool2(out)
		out = self.activation(self.bn3(self.conv3(out)))
		out = self.activation(self.bn4(self.conv4(out)))
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Experiment: 157
# Config: model=simplecnn_small_4_16, run=1, data_iid=false, lr=0.01,
# loss_type=fedavg, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: special attention and increase accuracy by +1%.
# Suggestion: Add batch norm.
# experiment suggestions: simplecnn_small_4_16 but fedmax with beta = 15, then RefinedSimpleCNNSmall416 with fedavg, then RefinedSimpleCNNSmall416 with fedmax, then fedprox
class RefinedSimpleCNNSmall416(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.activation = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(16)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(16 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.conv2(out)))
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)



# Experiment: 154
# Config: model=simplecnn_20_40_batchnorm, run=1, data_iid=false, lr=0.01,
# loss_type=fedavg, mu=1.0, beta=10.0, rpi_local_epochs=1, mc1_local_epochs=1
# Work target: special attention for 91.88% baseline; focus on speed and lower inference power.
# exploration suggestions:
# try: simplecnn_20_40_batchnorm with:
# 1. learning rate 0.02,
# 2. fedmax with beta = 15
# 3. fedmax with beta = 15 and learning rate 0.02
# Then try RefinedSimpleCNN2040BatchNorm with fedavg, then fedmax with beta = 15, then fedprox
class RefinedSimpleCNN2040BatchNorm(_RefinedModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(20)
		self.activation = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.depthwise2 = nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=1, groups=20)
		self.bn2 = nn.BatchNorm2d(20)
		self.pointwise2 = nn.Conv2d(20, 40, kernel_size=1, stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(40)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(40 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.depthwise2(out)))
		out = self.activation(self.bn3(self.pointwise2(out)))
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


REFINED_VARIANTS = {
	"refined_simplecnn_small": RefinedSimpleCNNSmall,
	"refined_simplecnn_small_12_24": RefinedSimpleCNNSmall1224,
	"refined_simplecnn_20_40": RefinedSimpleCNN2040,
	"refined_simplecnn_24_48": RefinedSimpleCNN2448,
	"refined_simplecnn_small_8_16": RefinedSimpleCNNSmall816,
	"refined_simplecnn_small_4_8": RefinedSimpleCNNSmall48,
	"refined_simplecnn_small_4_8_deep": RefinedSimpleCNNSmall48Deep,
	"refined_simplecnn_small_4_8_deeper": RefinedSimpleCNNSmall48Deeper,
	"refined_simplecnn_small_4_16": RefinedSimpleCNNSmall416,
	"refined_simplecnn_20_40_batchnorm": RefinedSimpleCNN2040BatchNorm,
}


def get_refined_model(model_name: str, loss_type: str = "fedavg") -> nn.Module:
	if model_name in REFINED_VARIANTS:
		return REFINED_VARIANTS[model_name](loss_type=loss_type)

	raise NotImplementedError(f"[!] ERROR: Refined model {model_name} not implemented yet")
