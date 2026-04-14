from models.get_model import get_model

models = ["simplecnn", "simplefc", "conv5small", "vgg11", "vgg16", "mobilenet"]

print("Model Parameter Counts:")
print("-" * 50)

for model_name in models:
    model = get_model(model_name)
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"{model_name:15} | Total: {total_params:>10,} | Trainable: {trainable_params:>10,}")
    
    if total_params > 1_000_000:
        print(f"  ⚠️  WARNING: Exceeds 1M parameter limit!")
    else:
        print(f"  ✓ OK (under 1M)")
    print()

print("-" * 50)
print("Limit: 1,000,000 parameters")
