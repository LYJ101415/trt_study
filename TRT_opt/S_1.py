"""
以下这段代码的功能是“评估” (Evaluation)，而不是“量化” (Quantization)。
它的作用是加载一个已经存在的模型（无论是 FP32 还是已经量化好的 ONNX），将其编译为 TensorRT 引擎，并测试其准确率和延迟。
"""

import argparse # 用于解析命令行参数
import csv # 用于将结果写入 CSV 文件

import timm  # PyTorch Image Models 库，用于加载模型配置和数据预处理
from evaluation import evaluate # 从本地 evaluation.py 文件导入评估函数

# 从 modelopt 库导入用于模型部署和运行的核心类
from modelopt.torch._deploy._runtime import RuntimeRegistry
from modelopt.torch._deploy.device_model import DeviceModel
from modelopt.torch._deploy.utils import OnnxBytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="""Path to the image classification ONNX model with input shape of
        [batch_size,3,224,224] and output shape of [1,1000]""",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="use for timm.create_model to load data config",
    )
    parser.add_argument(
        "--engine_path",
        type=str,
        default=None,
        help="Path to the TensorRT engine",
    )
    parser.add_argument(
        "--timing_cache_path",
        type=str,
        default=None,
        help="Path to the TensorRT timing cache",
    )
    parser.add_argument(
        "--imagenet_path",
        type=str,
        default="ILSVRC/imagenet-1k",
        help="HF dataset card or local path to the ImageNet dataset",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument(
        "--eval_data_size", type=int, default=None, help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--engine_precision",
        type=str,
        default="stronglyTyped",
        choices=["best", "fp16", "stronglyTyped"],
        help="Precision mode for the TensorRT engine. \
            stronglyTyped is recommended, all other modes have been deprecated in TensorRT",
    )
    parser.add_argument(
        "--results_path", type=str, default=None, help="Save the results to the specified path"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for DataLoader shuffle to ensure reproducible sampling",
    )

    args = parser.parse_args()

    # 4. 定义部署配置，指定运行时为 TensorRT (TRT) 和精度模式
    deployment = {
        "runtime": "TRT",
        "precision": args.engine_precision,
    }

    # 5. 读取 ONNX 模型文件并转换为字节流
    onnx_bytes = OnnxBytes(args.onnx_path).to_bytes()

    # 6. 根据部署配置获取对应的运行时客户端 (这里是 TensorRT 客户端)
    client = RuntimeRegistry.get(deployment)

    # 7. 准备编译参数
    compilation_args = {
        "engine_path": args.engine_path,
        "timing_cache_path": args.timing_cache_path,
    }
    # 8. 将 ONNX 模型字节流编译成 TensorRT 引擎
    compiled_model = client.ir_to_compiled(onnx_bytes, compilation_args)

    # 9. 创建一个 DeviceModel 对象，用于封装编译后的模型和运行时客户端
    device_model = DeviceModel(client, compiled_model, metadata={})

    top1_accuracy, top5_accuracy = 0.0, 0.0

    # 10. 使用 timm 库加载模型配置，以获取正确的数据预处理方式
    # 注意：这里只加载模型配置，不加载权重，因为评估的是已导出的 ONNX 模型
    model = timm.create_model(args.model_name, pretrained=False, num_classes=1000)
    data_config = timm.data.resolve_model_data_config(model)
    transforms = timm.data.create_transform(**data_config, is_training=False)

    # 11. 调用 evaluate 函数进行评估
    # 该函数会使用 device_model 进行推理，并返回 Top-1 和 Top-5 准确率
    top1_accuracy, top5_accuracy = evaluate(
        device_model,
        transforms,
        batch_size=args.batch_size,
        num_examples=args.eval_data_size,
        dataset_path=args.imagenet_path,
        seed=args.seed,
    )
    # 12. 打印评估结果
    print(f"The top1 accuracy of the model is {top1_accuracy}%")
    print(f"The top5 accuracy of the model is {top5_accuracy}%")

    # 13. 获取并打印推理延迟
    latency = device_model.get_latency()
    print(f"Inference latency of the model is {latency} ms")

    # 14. 如果指定了结果保存路径，则将结果写入 CSV 文件
    if args.results_path:
        results: list[list[str | float]] = [
            ["Metric", "Value"],
            ["Top 1", top1_accuracy],
            ["Top 5", top5_accuracy],
            ["Latency", latency],
        ]
        with open(args.results_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(results)


if __name__ == "__main__":
    main()