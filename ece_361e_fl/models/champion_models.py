import torch
import torch.nn as nn


class _ChampionModel(nn.Module):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__()
		self.loss_type = loss_type

	def _return(self, logits: torch.Tensor, activations: torch.Tensor):
		if self.loss_type == "fedmax":
			return logits, activations
		return logits


# ============================================================================
# Base Model: refined_simplecnn_small_4_8_deep
# ============================================================================
# Architecture: 1 -> 4 -> 8 -> 16 channels with batch norm (3 conv layers)
# Shared Config: lr=0.01, mu=1.0, beta=10.0, rounds=30, rpi_epochs=1, mc1_epochs=1
#
# ANALYSIS - Impact of Loss Function on Performance:
# 
# Experiment 164 (fedavg):
#   - Accuracy: 90.63% (baseline, slowest convergence)
#   - Time to 90%: 75.30s (5 rounds)
#   - Energy per round: 61.36J (lowest)
#   - Profile: Most energy-efficient, but weakest accuracy
#
# Experiment 165 (fedmax):
#   - Accuracy: 91.02% (+0.39% vs fedavg)
#   - Time to 90%: 93.11s (8 rounds, slower convergence)
#   - Energy per round: 63.78J (+2.42J)
#   - Profile: Mid-range performance, more computation
#
# Experiment 166 (fedprox) ⭐ BEST:
#   - Accuracy: 91.73% (+1.10% vs fedavg, +0.71% vs fedmax)
#   - Time to 90%: 56.19s (fastest convergence, -25.11s vs fedavg)
#   - Energy per round: 66.44J (+5.08J vs fedavg)
#   - Profile: Excellent balance - highest accuracy, fastest convergence despite higher per-round energy
#
# Key Insights:
#   - fedprox wins on accuracy and convergence speed
#   - fedavg is most energy-efficient per round but slowest overall
#   - fedmax provides middle ground but doesn't outperform alternatives
#
class Champion_Sword(_ChampionModel):
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



# ============================================================================
# Base Model: refined_simplecnn_small_4_8_deeper
# ============================================================================
# Architecture: 1 -> 4 -> 8 -> 16 -> 24 channels with batch norm (4 conv layers)
# Shared Config: lr=0.01, mu=1.0, beta=10.0, rounds=30, rpi_epochs=1, mc1_epochs=1
#
# ANALYSIS - Impact of Loss Function on Performance (Deeper Architecture):
#
# Experiment 168 (fedavg):
#   - Accuracy: 91.24% (higher than shallow 164: +0.61%)
#   - Time to 90%: 119.60s (10 rounds, slowest of three)
#   - Energy per round: 70.05J (baseline for deeper)
#   - Profile: Highest per-round energy due to extra layer, slow convergence
#
# Experiment 169 (fedmax):
#   - Accuracy: 91.21% (-0.03% vs fedavg, essentially equivalent)
#   - Time to 90%: 135.52s (12 rounds, slowest overall)
#   - Energy per round: 75.52J (+5.47J vs fedavg)
#   - Profile: Worst performer - slowest convergence, high energy costs
#
# Experiment 170 (fedprox) ⭐ BEST:
#   - Accuracy: 91.35% (+0.11% vs fedavg, +0.14% vs fedmax)
#   - Time to 90%: 119.58s (10 rounds, tied with fedavg)
#   - Energy per round: 77.09J (+7.04J vs fedavg)
#   - Profile: Best accuracy but expensive; extra layer adds energy without proportional accuracy gain
#
# Comparison vs Shallow (small_4_8_deep):
#   - Deeper does NOT consistently improve accuracy (+0.62% for best case)
#   - Deeper has worse energy profiles (+10.65J per round on average)
#   - Deeper convergence is slower (more rounds needed)
#   - Trade-off: marginal accuracy gain not worth the energy cost
#

# COME BACK TO THIS MODEL IF CONSITENT ACCURACY IS AN ISSUE
class Champion_Mace(_ChampionModel):
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



# ============================================================================
# Base Models: 416 variants
# ============================================================================
# Replacement family for the old 4->16 attempt.
# 1 = keep the 1-2-4-8 stem but make it shape-safe and cheap.
# 2 = balanced 1-4-8-12-16 stack with adaptive pooling.
# 3 = bottleneck 1-4-8-16 with a depthwise separable block.

class Champion_Explorer(_ChampionModel):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(2)
		self.activation = nn.ReLU()
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(4)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(8)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(8 * 4 * 4, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.conv2(out)))
		out = self.pool2(out)
		out = self.activation(self.bn3(self.conv3(out)))
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Champion_Explorer_v2(_ChampionModel):
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
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(16 * 4 * 4, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.conv2(out)))
		out = self.pool2(out)
		out = self.activation(self.bn3(self.conv3(out)))
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Champion_Fighter(_ChampionModel):
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
		self.conv3 = nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(12)
		self.conv4 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(16)
		self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Linear(16 * 6 * 6, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.conv2(out)))
		out = self.pool2(out)
		out = self.activation(self.bn3(self.conv3(out)))
		out = self.activation(self.bn4(self.conv4(out)))
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Champion_Fighter_v2(_ChampionModel):
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
		self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Linear(24 * 6 * 6, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.conv2(out)))
		out = self.pool2(out)
		out = self.activation(self.bn3(self.conv3(out)))
		out = self.activation(self.bn4(self.conv4(out)))
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Champion_Rogue(_ChampionModel):
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
		self.depthwise3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8)
		self.bn3 = nn.BatchNorm2d(8)
		self.pointwise3 = nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0)
		self.bn4 = nn.BatchNorm2d(16)
		self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Linear(16 * 6 * 6, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.conv2(out)))
		out = self.pool2(out)
		out = self.activation(self.bn3(self.depthwise3(out)))
		out = self.activation(self.bn4(self.pointwise3(out)))
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Champion_Rogue_v2(_ChampionModel):
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
		self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Linear(16 * 6 * 6, self.num_classes)

	def forward(self, x):
		out = self.activation(self.bn1(self.conv1(x)))
		out = self.pool1(out)
		out = self.activation(self.bn2(self.conv2(out)))
		out = self.pool2(out)
		out = self.activation(self.bn3(self.conv3(out)))
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# ============================================================================
# APPENDED LOW-COST CANDIDATES
# ============================================================================
# These are only additional ideas to try if the goal is to beat the current
# 4->8->16 baseline on both time and energy without moving to a larger model.
#
# Candidate 4: keep the shallow stem, but shrink the classifier with global
# average pooling. This usually reduces memory traffic and can help MC1 time.
class Champion_Knife(_ChampionModel):
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
		self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Linear(16 * 6 * 6, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		out = self.activation(self.conv3(out))
		out = self.bn3(out)
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Champion_Knife_v2(_ChampionModel):
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
		self.conv3 = nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(12)
		self.conv4 = nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1)
		self.bn4 = nn.BatchNorm2d(16)
		self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Linear(16 * 6 * 6, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		out = self.activation(self.conv3(out))
		out = self.bn3(out)
		out = self.activation(self.conv4(out))
		out = self.bn4(out)
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# Candidate 5: slightly narrower head, but keep the same feature extractor.
# This aims to reduce classifier cost while preserving most representational
# power in the convolutional trunk.
class Champion_Dagger(_ChampionModel):
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
		self.conv3 = nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(12)
		self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Linear(12 * 6 * 6, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		out = self.activation(self.conv3(out))
		out = self.bn3(out)
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Champion_Dagger_v2(_ChampionModel):
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
		self.pool3 = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Linear(16 * 6 * 6, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.pool2(out)
		out = self.activation(self.conv3(out))
		out = self.bn3(out)
		out = self.pool3(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)



# ============================================================================
# Base Model: simplecnn_small_4_8
# ============================================================================
# Architecture: 1 -> 4 -> 8 channels with batch norm (2 conv layers, minimal capacity)
# Config: lr=0.01, loss=fedprox, mu=1.5 (non-standard), beta=10.0, rounds=30, rpi_epochs=1, mc1_epochs=1
#
# ANALYSIS - Minimal Model with Modified Hyperparameters:
#
# Experiment 181 (fedprox with mu=1.5):
#   - Accuracy: 88.61% (FAILED to reach 90% target, lowest of all)
#   - Time to 90%: n/a (convergence below 90%)
#   - Energy per round: 47.23J (LOWEST energy consumption - most efficient)
#   - Early stopping: Triggered at round 27 due to accuracy convergence below 90%
#   - Profile: Severely underfitted model, increased mu=1.5 does not help
#
# Key Insights:
#   - Minimal 2-layer architecture fundamentally insufficient for task
#   - Increased mu parameter (1.5 vs standard 1.0) worsens performance
#   - Early stopping indicates model plateaus at 88.6% - structural limitation
#   - Most energy-efficient but fails to meet accuracy requirements
#   - Cannot be used despite efficiency - does not meet minimum accuracy target
#   - Architecture capacity (4->8 without depth) is the bottleneck, not hyperparameters
#
class Champion_Blade(_ChampionModel):
	def __init__(self, loss_type: str = "fedprox"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 12, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(12)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(12 * 8 * 8, self.num_classes)

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


class Champion_Blade_v2(_ChampionModel):
	def __init__(self, loss_type: str = "fedprox"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.activation = nn.ReLU()
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.conv3 = nn.Conv2d(8, 12, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(12)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(12 * 8 * 8, self.num_classes)

	def forward(self, x):
		out = self.activation(self.conv1(x))
		out = self.bn1(out)
		out = self.pool1(out)
		out = self.activation(self.conv2(out))
		out = self.bn2(out)
		out = self.activation(self.conv3(out))
		out = self.bn3(out)
		out = self.pool2(out)
		activations = out.view(out.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


CHAMPION_VARIANTS = {
	# Champion_Sword: refined 4->8->16 deep (baseline, frozen)
	"champion_sword": Champion_Sword,

	
	# Champion_Mace: refined 4->8->16->24 deeper (backup, frozen)
	"champion_mace": Champion_Mace,

	
	# Champion_Explorer: ultra-cheap 1-2-4-8 stem (original)
	"champion_explorer": Champion_Explorer,
	
	# Champion_Explorer_v2: upgraded to 1-4-8-16 (wider stem, more capacity)
	"champion_explorer_v2": Champion_Explorer_v2,

	
	# Champion_Fighter: balanced 4-8-12-16 with adaptive pooling (original)
	"champion_fighter": Champion_Fighter,
	
	# Champion_Fighter_v2: upgraded to 4-8-16-24 with adaptive pooling (wider channels)
	"champion_fighter_v2": Champion_Fighter_v2,

	
	# Champion_Rogue: depthwise-separable bottleneck (original)
	"champion_rogue": Champion_Rogue,
	
	# Champion_Rogue_v2: replaced depthwise with regular conv 4-8-16 (more capacity)
	"champion_rogue_v2": Champion_Rogue_v2,

	
	# Champion_Knife: 4->8->16 with adaptive pooling (original)
	"champion_knife": Champion_Knife,
	
	# Champion_Knife_v2: upgraded to 4-8-12-16 with adaptive pooling (intermediate layer)
	"champion_knife_v2": Champion_Knife_v2,
	
	# Champion_Dagger: lite head 4-8->12 with adaptive pooling (original)
	"champion_dagger": Champion_Dagger,
	
	# Champion_Dagger_v2: upgraded to 4-8->16 with adaptive pooling (wider final layer)
	"champion_dagger_v2": Champion_Dagger_v2,
	
	# Champion_Blade: minimal 4->12 (2 layers, original)
	"champion_blade": Champion_Blade,
	
	# Champion_Blade_v2: upgraded to 4-8-12 (3 layers, added intermediate depth)
	"champion_blade_v2": Champion_Blade_v2,
}


def get_champion_model(model_name: str, loss_type: str = "fedavg") -> nn.Module:
	if model_name in CHAMPION_VARIANTS:
		return CHAMPION_VARIANTS[model_name](loss_type=loss_type)

	raise NotImplementedError(f"[!] ERROR: Champion model {model_name} not implemented yet")


# ============================================================================
# COMPREHENSIVE PERFORMANCE ANALYSIS & SUMMARY
# ============================================================================
#
#
# KEY FINDINGS:
#
# 1. ARCHITECTURAL INSIGHTS:
#    - Shallow + Deep (4->8->16 with 3 layers) outperforms Wider (4->16 with 2 layers)
#    - Adding extra layers (4 layers) doesn't proportionally improve accuracy
#    - Depth wins over width for this task
#
# 2. LOSS FUNCTION IMPACT:
#    - fedprox CONSISTENTLY achieves best results across architectures (when successful)
#    - fedavg offers best energy efficiency but loses accuracy
#    - fedmax provides no clear advantage - middle ground without benefits
#
# 3. SPEED vs ACCURACY TRADE-OFF:
#    - Champion model (Exp 166) achieves best speed AND accuracy simultaneously
#    - Suggests fedprox + shallow deep architecture is Pareto optimal
#    - Adding depth improves accuracy marginally (+0.16%) but costs 63.39s more convergence
#
# 4. ENERGY vs PERFORMANCE TRADE-OFF:
#    - Energy efficiency alone is misleading (Exp 181: 47.23J but only 88.61%)
#    - Target accuracy (90%+) requires 60J+ per round minimum
#    - Champion model uses 66.44J per round - acceptable cost for reliability
#
# 5. ARCHITECTURAL FAILURES:
#    - 4->16 (2 layers): Max 89.9% - insufficient capacity despite efficiency
#    - 4->8 (2 layers, minimal): Max 88.6% - severely underfitted
#    - These architectures hit hard ceiling, no loss function can fix
#
# RECOMMENDATION:
# Use ChampionRefinedSimpleCNNSmall48Deep with fedprox loss
# - Highest accuracy: 91.73%
# - Fastest convergence: 56.19s
# - Balanced energy: 66.44J per round
# - Best all-around performance and reliability
#
