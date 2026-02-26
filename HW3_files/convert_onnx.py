import torch
import torch.onnx
import argparse
import os
from models.vgg11 import VGG11
from models.vgg16 import VGG16


def convert_pytorch_to_onnx(model, model_name, input_shape=(1, 3, 32, 32), output_path="./", opset_version=11):
    """
    Convert a PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model to be converted
        model_name: Name of the model (used for output filename)
        input_shape: Shape of the input tensor (batch_size, channels, height, width)
        output_path: Directory where the ONNX model will be saved
        opset_version: ONNX opset version (13 for MC1, 16 for RaspberryPi)
    
    Returns:
        output_file: Path to the saved ONNX model
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape)
    
    # If model is on GPU, move dummy input to GPU
    if next(model.parameters()).is_cuda:
        dummy_input = dummy_input.cuda()
    
    # Define the output file path
    output_file = os.path.join(output_path, f"{model_name}.onnx")
    
    # Export the model to ONNX format
    torch.onnx.export(
        model,                          # Model being run
        dummy_input,                    # Model input (or a tuple for multiple inputs)
        output_file,                    # Where to save the model
        export_params=True,             # Store the trained parameter weights inside the model file
        opset_version=opset_version,    # The ONNX version to export the model to
        do_constant_folding=True,       # Whether to execute constant folding for optimization
        input_names=['input'],          # The model's input names
        output_names=['output'],        # The model's output names
        dynamic_axes={
            'input': {0: 'batch_size'},    # Variable length axes
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model successfully converted to ONNX format (opset {opset_version}) and saved at: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch models to ONNX format')
    parser.add_argument('--model', type=str, required=True, choices=['vgg11', 'vgg16', 'mobilenet'],
                        help='Model to convert (vgg11, vgg16, or mobilenet)')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained PyTorch model (.pt file)')
    parser.add_argument('--output_path', type=str, default='./',
                        help='Directory where the ONNX model will be saved')
    parser.add_argument('--device', type=str, default='mc1', choices=['mc1', 'raspi'],
                        help='Target device: mc1 (opset 13) or raspi (opset 16)')
    
    args = parser.parse_args()
    
    # Set opset version based on device
    opset_version = 13 if args.device == 'mc1' else 16
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load the appropriate model architecture
    if args.model.lower() == 'vgg11':
        model = VGG11()
    elif args.model.lower() == 'vgg16':
        model = VGG16()
    elif args.model.lower() == 'mobilenet':
        from models.mobilenet import MobileNetv1
        model = MobileNetv1(num_classes=10)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Load the trained weights
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    # Convert the model to ONNX
    print(f"Converting for {args.device.upper()} (opset version {opset_version})")
    convert_pytorch_to_onnx(
        model=model,
        model_name=f"{args.model.lower()}_{args.device}",
        input_shape=(1, 3, 32, 32),  # CIFAR10 input shape
        output_path=args.output_path,
        opset_version=opset_version
    )


if __name__ == "__main__":
    main()
