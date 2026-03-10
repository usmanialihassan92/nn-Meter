import torch
import torchvision.models as models


def export_model_to_onnx(output_path="onnx/mobilenetv3.onnx"):

    # Load pretrained model
    model = models.mobilenet_v3_small(pretrained=True)

    # Set to evaluation mode
    model.eval()

    # Dummy input (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)

    print("Exporting PyTorch model to ONNX...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

    print(f"ONNX model saved to {output_path}")


if __name__ == "__main__":
    export_model_to_onnx()