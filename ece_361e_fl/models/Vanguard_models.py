import torch
import torch.nn as nn


class _VanguardModel(nn.Module):
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__()
		self.loss_type = loss_type

	def _return(self, logits: torch.Tensor, activations: torch.Tensor):
		if self.loss_type == "fedmax":
			return logits, activations
		return logits


# ============================================================================
# SWORD FAMILY VARIANTS (base winner: Champion_Sword)
# Target: keep >=90% while exploring either faster qualification or lower energy.
# ============================================================================


class Vanguard_Sword_Turbo(_VanguardModel):
	# Design decision:
	# - Slightly widen the final stage (16 -> 18 channels) while preserving the
	#   known-good 2x pooling + flatten topology from Sword.
	# Expected outcome:
	# - Higher feature capacity should reduce rounds-to-90% (speed-first), with
	#   a moderate increase in per-round energy.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(8, 18, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(18)
		self.classifier = nn.Linear(18 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.pool2(self.act(self.bn2(self.conv2(x))))
		x = self.act(self.bn3(self.conv3(x)))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_Sword_MidBoost(_VanguardModel):
	# Design decision:
	# - Add capacity in the middle stage (8 -> 10) while preserving Sword's
	#   proven 8x8 flatten head and overall depth.
	# Expected outcome:
	# - A second speed-first Sword candidate that may converge in fewer rounds
	#   without the larger per-round jump of final-stage-only widening.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 10, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(10)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(10, 16, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(16)
		self.classifier = nn.Linear(16 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.pool2(self.act(self.bn2(self.conv2(x))))
		x = self.act(self.bn3(self.conv3(x)))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_Sword_Eco(_VanguardModel):
	# Design decision:
	# - Narrow the final stage (16 -> 14 channels) with moderate head reduction
	#   (8x8 -> 7x7) to avoid over-compressing a near-threshold model.
	# Expected outcome:
	# - Lower per-round energy than Sword with less qualification risk than a
	#   stronger 6x6 head compression.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(8, 14, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(14)
		self.head_pool = nn.AdaptiveAvgPool2d((7, 7))
		self.classifier = nn.Linear(14 * 7 * 7, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.pool2(self.act(self.bn2(self.conv2(x))))
		x = self.act(self.bn3(self.conv3(x)))
		x = self.head_pool(x)
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# ============================================================================
# EXPLORER_V2 FAMILY VARIANTS (base qualifier: Champion_Explorer_v2)
# Target: energy-first frontier with attempts to reduce rounds-to-90%.
# ============================================================================


class Vanguard_ExplorerV2_Turbo(_VanguardModel):
	# Design decision:
	# - Keep 3-pool topology but widen final stage (16 -> 18).
	# Expected outcome:
	# - Better class separation earlier in training (speed-first) while retaining
	#   the compact 4x4 head that made Explorer_v2 energy-competitive.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(8, 18, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(18)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(18 * 4 * 4, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.pool2(self.act(self.bn2(self.conv2(x))))
		x = self.pool3(self.act(self.bn3(self.conv3(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_ExplorerV2_DepthwiseEco(_VanguardModel):
	# Design decision:
	# - Replace dense conv3 with depthwise + pointwise projection.
	# Expected outcome:
	# - Lower MACs in the final block (energy-first) with enough channel capacity
	#   to remain in the qualifying zone after hyperparameter tuning.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.dw3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1, groups=8)
		self.bn3 = nn.BatchNorm2d(8)
		self.pw3 = nn.Conv2d(8, 16, kernel_size=1, stride=1, padding=0)
		self.bn4 = nn.BatchNorm2d(16)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(16 * 4 * 4, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.pool2(self.act(self.bn2(self.conv2(x))))
		x = self.act(self.bn3(self.dw3(x)))
		x = self.pool3(self.act(self.bn4(self.pw3(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_ExplorerV2_Eco(_VanguardModel):
	# Design decision:
	# - Slightly narrow the final stage (16 -> 14), keeping everything else fixed.
	# Expected outcome:
	# - Reduced per-round energy with minimal disturbance to the known convergent
	#   training dynamics of Explorer_v2.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv3 = nn.Conv2d(8, 14, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(14)
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(14 * 4 * 4, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.pool2(self.act(self.bn2(self.conv2(x))))
		x = self.pool3(self.act(self.bn3(self.conv3(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# ============================================================================
# BLADE_V2 FAMILY VARIANTS (base qualifier: Champion_Blade_v2)
# Target: keep >=90% while reducing Blade_v2 energy/time costs.
# ============================================================================


class Vanguard_BladeV2_Balanced(_VanguardModel):
	# Design decision:
	# - Make a stronger speed-oriented move by widening both intermediate and
	#   final stages (4->8->12 becomes 4->9->14).
	# Expected outcome:
	# - Better chance to reduce rounds-to-90% versus the base Blade_v2, with
	#   acceptable per-round cost increase for speed-first testing.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 9, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(9)
		self.conv3 = nn.Conv2d(9, 14, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(14)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(14 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.act(self.bn2(self.conv2(x)))
		x = self.pool2(self.act(self.bn3(self.conv3(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_BladeV2_SlimA(_VanguardModel):
	# Design decision:
	# - More aggressive slimming in the middle stage (8 -> 6) while keeping
	#   final stage at 12.
	# Expected outcome:
	# - Lower per-round energy than Blade_v2, with a higher risk of dropping
	#   below 90% if optimization is not tuned.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 6, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(6)
		self.conv3 = nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(12)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(12 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.act(self.bn2(self.conv2(x)))
		x = self.pool2(self.act(self.bn3(self.conv3(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_BladeV2_SlimB(_VanguardModel):
	# Design decision:
	# - Keep middle stage at 8 but trim final stage more aggressively (12 -> 10).
	# Expected outcome:
	# - Another energy-first point near the passing boundary to test whether
	#   Blade_v2 was over-provisioned for the 90% requirement.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(8)
		self.conv3 = nn.Conv2d(8, 10, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(10)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(10 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.act(self.bn2(self.conv2(x)))
		x = self.pool2(self.act(self.bn3(self.conv3(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


# ============================================================================
# BLADE FAMILY VARIANTS (base near miss: Champion_Blade)
# Target: 5 exploration variants to cross >=90% while retaining low energy.
# ============================================================================


class Vanguard_Blade_W13(_VanguardModel):
	# Design decision:
	# - Minimal widening (12 -> 13) from the near-miss baseline.
	# Expected outcome:
	# - Small capacity bump aimed at crossing 90% with minimal energy increase.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 13, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(13)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(13 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.pool2(self.act(self.bn2(self.conv2(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_Blade_W14(_VanguardModel):
	# Design decision:
	# - Moderate widening (12 -> 14), still a 2-conv model.
	# Expected outcome:
	# - Better chance to pass >=90% than W13, with controlled energy overhead.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 14, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(14)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(14 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.pool2(self.act(self.bn2(self.conv2(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_Blade_RefineDW(_VanguardModel):
	# Design decision:
	# - Replace the aggressive width jump idea with a cheap depthwise refinement
	#   block after a moderate 4->13 expansion.
	# Expected outcome:
	# - Better feature refinement than W13/W14 with a smaller cost increase than
	#   a full dense-width jump.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 13, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(13)
		self.dw_refine = nn.Conv2d(13, 13, kernel_size=3, stride=1, padding=1, groups=13)
		self.bn_refine = nn.BatchNorm2d(13)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(13 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.act(self.bn2(self.conv2(x)))
		x = self.act(self.bn_refine(self.dw_refine(x)))
		x = self.pool2(x)
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_Blade_DeepLite(_VanguardModel):
	# Design decision:
	# - Add a lightweight intermediate conv (4->10->12), giving more depth than
	#   Blade_v1 but less width than full Blade_v2 variants.
	# Expected outcome:
	# - Potentially cross 90% like Blade_v2 while retaining a better energy
	#   profile than wider/deeper alternatives.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 10, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(10)
		self.conv3 = nn.Conv2d(10, 12, kernel_size=3, stride=1, padding=1)
		self.bn3 = nn.BatchNorm2d(12)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(12 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.act(self.bn2(self.conv2(x)))
		x = self.pool2(self.act(self.bn3(self.conv3(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


class Vanguard_Blade_HeadTrim(_VanguardModel):
	# Design decision:
	# - Replace the prior capacity-reducing head trim with a lightweight capacity
	#   boost using a pointwise 1x1 expansion (12 -> 14) before pooling.
	# Expected outcome:
	# - Improve qualification odds over Blade_v1 while keeping the extra compute
	#   smaller than adding a full new 3x3 conv block.
	def __init__(self, loss_type: str = "fedavg"):
		super().__init__(loss_type=loss_type)
		self.num_classes = 10
		self.act = nn.ReLU(inplace=True)
		self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
		self.bn1 = nn.BatchNorm2d(4)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv2d(4, 12, kernel_size=3, stride=1, padding=1)
		self.bn2 = nn.BatchNorm2d(12)
		self.pw_expand = nn.Conv2d(12, 14, kernel_size=1, stride=1, padding=0)
		self.bn3 = nn.BatchNorm2d(14)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.classifier = nn.Linear(14 * 8 * 8, self.num_classes)

	def forward(self, x):
		x = self.pool1(self.act(self.bn1(self.conv1(x))))
		x = self.act(self.bn2(self.conv2(x)))
		x = self.pool2(self.act(self.bn3(self.pw_expand(x))))
		activations = x.view(x.size(0), -1)
		logits = self.classifier(activations)
		return self._return(logits, activations)


VANGUARD_VARIANTS = {
	# Sword-based (3 variants)
	"vanguard_sword_turbo": Vanguard_Sword_Turbo,
	"vanguard_sword_midboost": Vanguard_Sword_MidBoost,
	"vanguard_sword_eco": Vanguard_Sword_Eco,

	# Explorer_v2-based (3 variants)
	"vanguard_explorerv2_turbo": Vanguard_ExplorerV2_Turbo,
	"vanguard_explorerv2_depthwise_eco": Vanguard_ExplorerV2_DepthwiseEco,
	"vanguard_explorerv2_eco": Vanguard_ExplorerV2_Eco,

	# Blade_v2-based (3 variants)
	"vanguard_bladev2_balanced": Vanguard_BladeV2_Balanced,
	"vanguard_bladev2_slim_a": Vanguard_BladeV2_SlimA,
	"vanguard_bladev2_slim_b": Vanguard_BladeV2_SlimB,

	# Blade-based (5 variants)
	"vanguard_blade_w13": Vanguard_Blade_W13,
	"vanguard_blade_w14": Vanguard_Blade_W14,
	"vanguard_blade_refine_dw": Vanguard_Blade_RefineDW,
	"vanguard_blade_deeplite": Vanguard_Blade_DeepLite,
	"vanguard_blade_headtrim": Vanguard_Blade_HeadTrim,

	# Compatibility aliases for previously proposed names
	"vanguard_sword_balanced_head": Vanguard_Sword_MidBoost,
	"vanguard_blade_w15": Vanguard_Blade_RefineDW,
}


def get_vanguard_model(model_name: str, loss_type: str = "fedavg") -> nn.Module:
	if model_name in VANGUARD_VARIANTS:
		return VANGUARD_VARIANTS[model_name](loss_type=loss_type)

	raise NotImplementedError(f"[!] ERROR: Vanguard model {model_name} not implemented yet")
