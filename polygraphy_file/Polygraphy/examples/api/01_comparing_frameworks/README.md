# Comparing Frameworks


## Introduction

One of the core features of Polygraphy is comparison of model outputs across multiple
different backends. This makes it possible to check the accuracy of one backend with
respect to another.

In this example, we'll look at how you can use the Polygraphy API to run inference
with synthetic input data using ONNX-Runtime and TensorRT, and then compare the results.


## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. Run the example
    ```bash
    python3 example.py
    ```

3. **[Optional]** Inspect the inference outputs from the example:

    ```bash
    polygraphy inspect data inference_results.json
    ```
===============================================================================
这份文档标题为 “Comparing Frameworks”（比较不同框架），是 Polygraphy 官方示例中的一个典型用例，旨在展示其核心能力之一：跨推理后端的模型输出一致性验证。下面从目标、原理、流程和工程价值四个维度进行深度解析。

一、核心目标：验证 TensorRT 是否“正确”
问题背景：将模型从 ONNX 转换为 TensorRT 引擎时，可能因算子支持差异、精度模式（FP16/INT8）、优化策略等导致数值偏差。
Polygraphy 的解决方案：同时在 ONNX Runtime（参考实现） 和 TensorRT（待验证实现） 上运行完全相同的输入，对比输出是否在可接受误差范围内。
这是模型部署前精度验证（Accuracy Validation） 的黄金标准。

二、关键概念解析
1. “Backend”（后端）：指执行模型推理的运行时引擎，例如：
onnxrt：ONNX Runtime（CPU/GPU）
trt：TensorRT
torch：PyTorch
tf：TensorFlow
Polygraphy 提供统一接口调用这些后端。

2. “Synthetic input data”（合成输入数据）
自动生成的随机或全1/0张量（如 np.ones((1,3,224,224))）；
用于快速验证，无需真实数据集；
可通过 DataLoader 自定义（如固定种子保证可复现）。
3. “Compare the results”（结果比较）
默认使用 逐元素绝对误差（|a - b|） 和 相对误差（|a - b| / |b|）；
支持自定义容差阈值（如 --atol=1e-3 --rtol=1e-2）；
输出详细统计：最大误差、失败位置、可视化等。
