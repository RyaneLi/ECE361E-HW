import torch

from mobilenet import MobileNetv1


def export_mobilenet_fp32_to_onnx(
    weights_path: str = "MobilenetV1.pth",
    onnx_output_path: str = "MobilenetV1_fp32.onnx",
) -> None:
    model = MobileNetv1()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32)

    # Use opset 13 so the ONNX IR version is <= 9 (required by onnxruntime 1.16.3 on Raspberry Pi).
    # dynamo=False forces the legacy exporter so we get true opset 13 (new exporter exports at 18 then fails to convert down).
    torch.onnx.export(
        model,
        dummy_input,
        onnx_output_path,
        opset_version=13,
        input_names=["input"],
        output_names=["logits"],
        dynamo=False,
    )

    print(f"Exported FP32 MobileNet-v1 ONNX model to {onnx_output_path}")


if __name__ == "__main__":
    export_mobilenet_fp32_to_onnx()

