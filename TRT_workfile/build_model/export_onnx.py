"""
这是将pytorch模型导出为 ONNX 格式的实用脚本。它利用了 NVIDIA modelopt 工具包中的底层工具来完成导出过程。
"""

import argparse
import os

import timm
import torch

from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata


def export_to_onnx(model, input_shape, onnx_save_path, device, weights_dtype="fp32"):
    """Export the torch model to ONNX format."""
    # Create input tensor with same precision as model's first parameter
    input_dtype = model.parameters().__next__().dtype
    input_tensor = torch.randn(input_shape, dtype=input_dtype).to(device)
    model_name = os.path.basename(onnx_save_path).replace(".onnx", "")

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model=model,
        dummy_input=(input_tensor,),
        weights_dtype=weights_dtype,
        model_name=model_name,
    )
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)

    # Write the onnx model to the specified directory without cleaning the directory
    onnx_bytes_obj.write_to_disk(os.path.dirname(onnx_save_path), clean_dir=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and export example models to ONNX.")

    parser.add_argument(
        "--timm_model_name",
        type=str,
        required=True,
        help="Export any timm model to ONNX (e.g., vit_base_patch16_224, swin_tiny_patch4_window7_224).",
    )
    parser.add_argument(
        "--onnx_save_path", type=str, required=False, help="Path to save the final ONNX model."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the exported model.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to export the ONNX model in FP16.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model(args.timm_model_name, pretrained=True, num_classes=1000).to(device)
    data_config = timm.data.resolve_model_data_config(model)
    input_shape = (args.batch_size,) + data_config["input_size"]

    save_path = args.onnx_save_path or f"{args.timm_model_name}.onnx"
    weights_dtype = "fp16" if args.fp16 else "fp32"
    export_to_onnx(
        model,
        input_shape,
        save_path,
        device,
        weights_dtype=weights_dtype,
    )
    print(f"{args.timm_model_name} model exported to {save_path}")