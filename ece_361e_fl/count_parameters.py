from models.get_model import get_model
from models.refined_models import get_refined_model
from models.research_models import get_research_model

models = [
    "dense1_anchor",
    "simplecnn_small",
    "simplecnn_small_mlphead",
    "simplecnn_small_12_24",
    "simplecnn_small_8_16",
    "simplecnn_small_4_8",
    "simplecnn_singleconv",
    "simplecnn_singleconv_mlphead",
    "simplecnn_singleconv_1_16",
    "simplecnn_singleconv_1_8",
    "simplecnn_singleconv_1_4",
    "conv2d1_anchor",
    "conv2d1_leakyrelu",
    "conv2d1_maxpool",
    "conv2d1_batchnorm",
    "conv2d1_groupnorm",
    "conv2d1_residual_lite",
    "depthwise1_anchor",
    "depthwise1_leakyrelu",
    "depthwise1_maxpool",
    "depthwise1_batchnorm",
    "depthwise1_groupnorm",
    "depthwise1_residual_lite",
    "simplecnn",
    "simplefc",
    "conv5small",
    "vgg11",
    "vgg16",
    "mobilenet",
]

research_models = [
    "dense1_anchor_hidden_256",
    "dense1_anchor_deep",
    "dense1_anchor_conv_lite",
    "dense1_anchor_conv_lite_v2",
    "simplecnn_20_40",
    "simplecnn_20_40_batchnorm",
    "simplecnn_24_48",
    "simplecnn_bottleneck",
    "simplecnn_small_12_24_deep",
    "simplecnn_small_8_16_deep",
    "simplecnn_small_4_8_deep",
    "simplecnn_small_4_8_deep_groupnorm",
    "simplecnn_small_4_16",
    "simplecnn_singleconv_1_4_twoconv",
]

refined_models = [
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
]

champion_models = [
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
]

vanguard_models = [
    "vanguard_sword_turbo",
    "vanguard_sword_midboost",
    "vanguard_sword_eco",
    "vanguard_explorerv2_turbo",
    "vanguard_explorerv2_depthwise_eco",
    "vanguard_explorerv2_eco",
    "vanguard_bladev2_balanced",
    "vanguard_bladev2_slim_a",
    "vanguard_bladev2_slim_b",
    "vanguard_blade_w13",
    "vanguard_blade_w14",
    "vanguard_blade_refine_dw",
    "vanguard_blade_deeplite",
    "vanguard_blade_headtrim",
]


def print_model_counts(model_names, model_factory, title):
    print(title)
    print("-" * 50)

    for model_name in model_names:
        model = model_factory(model_name)

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"{model_name:30} | Total: {total_params:>10,} | Trainable: {trainable_params:>10,}")

        if total_params > 1_000_000:
            print(f"  ⚠️  WARNING: Exceeds 1M parameter limit!")
        else:
            print(f"  ✓ OK (under 1M)")
        print()

print_model_counts(models, get_model, "Model Parameter Counts (Main Registry):")

print_model_counts(
    research_models,
    get_research_model,
    "Model Parameter Counts (Research Registry):",
)

print_model_counts(
    refined_models,
    get_refined_model,
    "Model Parameter Counts (Refined Registry):",
)

print_model_counts(
    champion_models,
    get_model,
    "Model Parameter Counts (Champion Registry):",
)

print_model_counts(
    vanguard_models,
    get_model,
    "Model Parameter Counts (Vanguard Registry):",
)

print("-" * 50)
print("Limit: 1,000,000 parameters")
