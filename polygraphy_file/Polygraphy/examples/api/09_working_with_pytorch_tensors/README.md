# Working With PyTorch Tensors

## Introduction

Some runners like `OnnxrtRunner` and `TrtRunner` can accept and return PyTorch tensors
in addition to NumPy arrays. When PyTorch tensors are provided in the inputs, the runner
will return the outputs as PyTorch tensors as well. This can be especially useful in
cases where PyTorch supports a data type that is not supported by NumPy, such as BFloat16.

Polygraphy's included TensorRT `Calibrator` can also accept PyTorch tensors directly.

This example uses PyTorch tensors on the GPU where possible (i.e. if a GPU-enabled version
of PyTorch is installed). When the tensors already reside on GPU memory, no additional copies
are required in the runner/calibrator.

## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`


2. Run the example:

    ```bash
    python3 example.py
    ```


## See Also

* [Inference With TensorRT](../00_inference_with_tensorrt/)
* [INT8 Calibration In TensorRT](../04_int8_calibration_in_tensorrt/)
=========================================================================
这个文件是 Polygraphy（NVIDIA 的推理调试工具库）的一个高级示例文档，核心目标是展示 如何在 Polygraphy 的推理流程中直接使用 PyTorch 张量（torch.Tensor）替代传统的 NumPy 数组（np.ndarray），尤其是在涉及 BFloat16 精度、GPU 内存零拷贝、INT8 校准等场景下。
为什么需要支持 PyTorch Tensors？
传统上，Polygraphy（以及 ONNX Runtime / TensorRT）主要通过 NumPy 数组传递输入/输出。但这存在两个关键限制：
问题	              说明
1. 数据类型缺失	      NumPy 不支持 bfloat16（Brain Floating Point 16），而 PyTorch 和现代 GPU（如 A100、H100）广泛使用它进行训练/推理
2. GPU 内存冗余拷贝	  若数据已在 GPU 上（如来自 PyTorch 模型中间结果），转为 NumPy 需先 tensor.cpu().numpy() → 再由 runner 复制回 GPU，浪费带宽和时间

Polygraphy 的解决方案：允许 OnnxrtRunner / TrtRunner 直接接收 torch.Tensor 作为输入，并原样返回 torch.Tensor 输出，实现：类型兼容性（支持 torch.bfloat16）；
内存零拷贝（GPU tensor → GPU inference → GPU tensor）

关键技术原理：
1. Runner 的自动类型推断：当调用 runner.infer(feed_dict) 时：
如果 feed_dict 中的值是 torch.Tensor → runner 内部将其视为 CUDA memory pointer + metadata
推理完成后，输出张量也以 torch.Tensor 形式返回（保持设备和 dtype 一致性）
# 示例：直接传入 GPU tensor
input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16, device="cuda")
outputs = trt_runner.infer({"input": input_tensor})
print(type(outputs["output"]))  # <class 'torch.Tensor'>
print(outputs["output"].device) # cuda:0

底层机制：
Polygraphy 利用 PyTorch 的 DLPack 协议 或 CUDA memory pointer extraction（通过 tensor.data_ptr()）与 TensorRT/ONNX Runtime 对接，避免序列化开销。

2. 与 INT8 校准器（Calibrator）集成：TensorRT 的 INT8 量化校准需要大量输入样本。传统方式需将 PyTorch 数据转为 NumPy，再喂给校准器。现在可直接传 torch.Tensor：
from polygraphy.backend.trt import Calibrator

# 校准数据：列表 of feed_dict，每个值可是 torch.Tensor
calib_data = [{"input": torch.randn(1, 3, 224, 224, device="cuda")} for _ in range(100)]

calibrator = Calibrator(
    data_loader=calib_data,
    cache="calib.cache"
)
优势:校准数据无需离开 GPU；支持 bfloat16 → int8 的混合精度校准（若硬件支持）

3. 设备与内存管理
输入 Tensor 位置	   Runner 行为
CPU (device="cpu")	  自动复制到 GPU（等效于传统 NumPy 路径）
GPU (device="cuda")	  零拷贝，直接绑定到 TensorRT I/O buffer
不同 GPU	           抛出错误（需显式 .to("cuda:0")）
注意：TensorRT 引擎必须在同一 GPU上构建和运行，否则指针无效。