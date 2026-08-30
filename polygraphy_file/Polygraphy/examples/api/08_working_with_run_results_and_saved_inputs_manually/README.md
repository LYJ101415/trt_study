# Working With Run Results And Saved Inputs Manually

## Introduction

Inference inputs and outputs from `Comparator.run` can be serialized and saved to JSON
files so they can be reused. Inputs are stored as `List[Dict[str, np.ndarray]]` while outputs
are stored in a `RunResults` object, which can keep track of the outputs of multiple runners
from multiple inference iterations.

Command-line tools providing `--save-inputs` and `--save-outputs` options generally use these formats.

Usually, you'll only use saved inputs or `RunResults` with other Polygraphy APIs or
tools (as in [this example](../../cli//run/06_comparing_with_custom_output_data/)
or [this one](../../cli/inspect/05_inspecting_inference_outputs/)), but sometimes,
you may want to work with the underlying NumPy arrays manually.

Polygraphy includes convenience APIs that make it easy to load and manipulate these objects.

This example illustrates how you can load saved inputs and/or `RunResults` from a file
using the Python API and then access the NumPy arrays stored within.

## Running The Example

1. Generate some inference inputs and outputs:

    ```bash
    polygraphy run identity.onnx --trt --onnxrt \
        --save-inputs inputs.json --save-outputs outputs.json
    ```

2. **[Optional]** Use `inspect data` to view the inputs on the command-line:

    ```bash
    polygraphy inspect data inputs.json --show-values
    ```

3. **[Optional]** Use `inspect data` to view the outputs on the command-line:

    ```bash
    polygraphy inspect data outputs.json --show-values
    ```

4. Run the example:

    ```bash
    python3 example.py
    ```
=======================================================================
这个文件是一个 Polygraphy（NVIDIA 的推理调试与分析工具库）的 Python 示例文档，旨在说明如何手动加载和操作通过 polygraphy run 命令保存下来的推理输入（inputs）和输出结果（outputs）。
一、核心目的：为什么需要这个功能？在模型部署和调试过程中，我们经常需要：
复现某次推理结果（用于 bug 定位）;对比不同推理引擎（如 ONNX Runtime vs TensorRT）的输出差异;离线分析输入/输出的数值分布、精度误差等;为此，Polygraphy 提供了：

--save-inputs inputs.json
--save-outputs outputs.json
但这些 .json 文件不是普通 JSON，而是序列化后的 NumPy 数组 + 元数据。本示例教你怎么用 Python 正确加载并操作这些数据。

二、关键数据结构解析
1. Saved Inputs 格式
类型：List[Dict[str, np.ndarray]]
含义：每个元素代表一次推理迭代的输入（支持多轮推理）
字典的 key 是输入张量名（如 "X"），value 是对应的 np.ndarray;示例（伪代码）：
inputs = [
    {"X": np.array([[1,2,3]], dtype=np.float32)},   # 第1次推理输入
    {"X": np.array([[4,5,6]], dtype=np.float32)}    # 第2次推理输入
]
适用于：多次运行同一模型（如 batch 不同、随机 seed 不同）

2. Saved Outputs 格式（RunResults 对象）
类型：polygraphy.RunResults;内部结构：

RunResults = Dict[
    runner_name: str, 
    List[Dict[str, np.ndarray]]
]
含义：外层字典：key 是 runner 名称（如 "onnxrt-runner", "trt-runner"）
内层列表：每项是该 runner 一次推理的输出字典;示例（伪代码）：

outputs = {
    "onnxrt-runner": [
        {"Y": np.array([[1,2,3]])},   # ONNX Runtime 第1次输出
        {"Y": np.array([[4,5,6]])}    # ONNX Runtime 第2次输出
    ],
    "trt-runner": [
        {"Y": np.array([[1,2,3]])},   # TensorRT 第1次输出
        {"Y": np.array([[4,5,7]])}    # 注意：这里有个微小差异！
    ]
}
这正是 模型一致性验证 的基础：对比不同 runner 的输出是否一致。
