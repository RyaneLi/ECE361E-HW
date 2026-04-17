from models.conv5 import Conv5, Conv5_small
from models.simplefc import SimpleFC
from models.simplecnn import SimpleCNN
from models.vgg11 import VGG11
from models.vgg16 import VGG16
from models.mobilenet import MobileNetv1
from models.refined_models import get_refined_model
from models.research_models import get_research_model
from models.champion_models import get_champion_model
from models.screening_models import (
    Conv2d1Anchor,
    Conv2d1BatchNorm,
    Conv2d1GroupNorm,
    Conv2d1LeakyReLU,
    Conv2d1MaxPool,
    Conv2d1ResidualLite,
    Dense1Anchor,
    Depthwise1Anchor,
    Depthwise1BatchNorm,
    Depthwise1GroupNorm,
    Depthwise1LeakyReLU,
    Depthwise1MaxPool,
    Depthwise1ResidualLite,
    SimpleCNNSingleConv,
    SimpleCNNSingleConv4,
    SimpleCNNSingleConv8,
    SimpleCNNSingleConv16,
    SimpleCNNSingleConvMLPHead,
    SimpleCNNSmall,
    SimpleCNNSmallMLPHead,
    SimpleCNNSmall1224,
    SimpleCNNSmall48,
    SimpleCNNSmall816,
)


RESEARCH_VARIANTS = {
    "dense1_anchor_hidden_256",
    "dense1_anchor_deep",
    "dense1_anchor_conv_lite",
    "dense1_anchor_conv_lite_v2",
    "simplecnn_20_40",
    "simplecnn_20_40_batchnorm",
    "simplecnn_24_48",
    "simplecnn_bottleneck",
    "simplecnn_small_4_16",
    "simplecnn_small_4_8_deep",
    "simplecnn_small_4_8_deep_groupnorm",
    "simplecnn_small_8_16_deep",
    "simplecnn_small_12_24_deep",
    "simplecnn_singleconv_1_4_twoconv",
}

REFINED_VARIANTS = {
    "refined_simplecnn_small",
    "refined_simplecnn_small_12_24",
    "refined_simplecnn_20_40",
    "refined_simplecnn_24_48",
    "refined_simplecnn_small_8_16",
    "refined_simplecnn_small_4_8",
    "refined_simplecnn_small_4_8_deep",
    "refined_simplecnn_small_4_8_deeper",
    "refined_simplecnn_small_4_16",
    "refined_simplecnn_20_40_batchnorm",
}

CHAMPION_VARIANTS = {
    "champion_sword",
    "champion_mace",
    "champion_explorer",
    "champion_explorer_v2",
    "champion_fighter",
    "champion_fighter_v2",
    "champion_rogue",
    "champion_rogue_v2",
    "champion_knife",
    "champion_knife_v2",
    "champion_dagger",
    "champion_dagger_v2",
    "champion_blade",
    "champion_blade_v2",
}


def get_model(model_name, loss_type='fedavg'):
    if model_name in CHAMPION_VARIANTS:
        return get_champion_model(model_name, loss_type=loss_type)
    
    if model_name in REFINED_VARIANTS:
        return get_refined_model(model_name, loss_type=loss_type)

    if model_name in RESEARCH_VARIANTS:
        return get_research_model(model_name, loss_type=loss_type)

    if model_name == "dense1_anchor":
        return Dense1Anchor(loss_type=loss_type)
    elif model_name == "simplecnn_small":
        return SimpleCNNSmall(loss_type=loss_type)
    elif model_name == "simplecnn_small_mlphead":
        return SimpleCNNSmallMLPHead(loss_type=loss_type)
    elif model_name == "simplecnn_small_12_24":
        return SimpleCNNSmall1224(loss_type=loss_type)
    elif model_name == "simplecnn_small_8_16":
        return SimpleCNNSmall816(loss_type=loss_type)
    elif model_name == "simplecnn_small_4_8":
        return SimpleCNNSmall48(loss_type=loss_type)
    elif model_name == "simplecnn_singleconv":
        return SimpleCNNSingleConv(loss_type=loss_type)
    elif model_name == "simplecnn_singleconv_mlphead":
        return SimpleCNNSingleConvMLPHead(loss_type=loss_type)
    elif model_name == "simplecnn_singleconv_1_16":
        return SimpleCNNSingleConv16(loss_type=loss_type)
    elif model_name == "simplecnn_singleconv_1_8":
        return SimpleCNNSingleConv8(loss_type=loss_type)
    elif model_name == "simplecnn_singleconv_1_4":
        return SimpleCNNSingleConv4(loss_type=loss_type)
    elif model_name == "conv2d1_anchor":
        return Conv2d1Anchor(loss_type=loss_type)
    elif model_name == "conv2d1_leakyrelu":
        return Conv2d1LeakyReLU(loss_type=loss_type)
    elif model_name == "conv2d1_maxpool":
        return Conv2d1MaxPool(loss_type=loss_type)
    elif model_name == "conv2d1_batchnorm":
        return Conv2d1BatchNorm(loss_type=loss_type)
    elif model_name == "conv2d1_groupnorm":
        return Conv2d1GroupNorm(loss_type=loss_type)
    elif model_name == "conv2d1_residual_lite":
        return Conv2d1ResidualLite(loss_type=loss_type)
    elif model_name == "depthwise1_anchor":
        return Depthwise1Anchor(loss_type=loss_type)
    elif model_name == "depthwise1_leakyrelu":
        return Depthwise1LeakyReLU(loss_type=loss_type)
    elif model_name == "depthwise1_maxpool":
        return Depthwise1MaxPool(loss_type=loss_type)
    elif model_name == "depthwise1_batchnorm":
        return Depthwise1BatchNorm(loss_type=loss_type)
    elif model_name == "depthwise1_groupnorm":
        return Depthwise1GroupNorm(loss_type=loss_type)
    elif model_name == "depthwise1_residual_lite":
        return Depthwise1ResidualLite(loss_type=loss_type)
    if model_name == "conv5":
        return Conv5(loss_type=loss_type)
    elif model_name == "conv5small":
        return Conv5_small(loss_type=loss_type)
    elif model_name == "simplefc":
        return SimpleFC(loss_type=loss_type)
    elif model_name == "simplecnn":
        return SimpleCNN(loss_type=loss_type)
    elif model_name == "vgg11":
        return VGG11(loss_type=loss_type)
    elif model_name == "vgg16":
        return VGG16(loss_type=loss_type)
    elif model_name == "mobilenet":
        return MobileNetv1(loss_type=loss_type)
    else:
        raise NotImplementedError(f'[!] ERROR: Model {model_name} not implemented yet')
