# Converting To TensorRT And Running Inference
## Introduction

Polygraphy includes a high-level Python API that can convert models
and run inference with various backends. For an overview of the Polygraphy
Python API, see [here](../../../polygraphy/).

In this example, we'll look at how you can leverage the API to easily convert an ONNX
model to TensorRT and run inference with FP16 precision enabled. We'll then save the
engine to a file and see how you can load it again and run inference.

## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. **[Optional]** Inspect the model before running the example:

    ```bash
    polygraphy inspect model identity.onnx
    ```
3. Run the script that builds and runs the engine:

    ```bash
    python3 build_and_run.py
    ```
4. **[Optional]** Inspect the TensorRT engine built by the example:

    ```bash
    polygraphy inspect model identity.engine
    ```
5. Run the script that loads the previously built engine, then runs it:

    ```bash
    python3 load_and_run.py
    ```
## Further Reading

For more details on the Polygraphy Python API, see the
[Polygraphy API reference](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html).

=========================================================================================
以上是一份关于如何使用 Polygraphy 工具将 ONNX 模型转换为 TensorRT 引擎并运行推理的官方示例教程说明。

下面我为你详细解释其含义、实现原理、用途以及相关的注意事项：

1. 核心含义与实现原理
该文档描述了一个完整的工作流：利用 Polygraphy 的高级 Python API，把深度学习中间表达格式（ONNX 模型）转换为针对 NVIDIA GPU 优化的底层运行时引擎，并执行推理计算。

实现原理如下：

模型转换（Build阶段）： TensorRT 是 NVIDIA 的高性能深度学习推理优化器和运行时引擎。它不能直接运行 ONNX 模型，需要先对其进行解析，然后进行层间融合、精度校准（如开启 FP16）、内核自动调优等图优化操作，最终序列化生成一个 .engine 文件。Polygraphy 在这里充当了一个便捷的封装层，让开发者无需编写冗长的 TensorRT C++ 或 Python 原生 API 代码，即可完成这些操作。
推理运行（Run阶段）： 生成引擎后，Polygraphy 可以自动分配 GPU 显存、管理输入/输出缓冲区，并将推理请求发送给 TensorRT 运行时执行计算。
引擎加载与反序列化： 由于模型转换（Build）通常非常耗时，转换后的引擎可以直接序列化保存到磁盘。下次运行时，直接从磁盘加载（反序列化）引擎即可跳过优化编译阶段，极大地缩短启动时间。
2. 用途
快速原型验证： 开发者可以快速验证 ONNX 模型在 TensorRT 上的兼容性以及 FP16 等半精度模式下的推理结果是否正确。
性能优化部署： 将训练好的模型转换为 TensorRT 引擎，以在 NVIDIA GPU 上获得极低的推理延迟和高吞吐量，适用于生产环境的部署。
模型诊断与调试： 使用 Polygraphy 提供的 inspect 工具，可以在转换前后查看模型结构、层信息和数据类型，方便排查转换过程中可能出现的算子不支持或精度溢出问题。
3. 代码步骤解析
文档中提供的运行步骤清晰地展示了这一流程：

环境准备： 确保安装了 TensorRT 和 Python 依赖。
检查原模型（可选）： polygraphy inspect model identity.onnx 用于查看原始 ONNX 模型的基本信息。
构建并运行： python3 build_and_run.py 对应“转换+推理”的完整流程，开启 FP16 精度，并将引擎保存为文件。
检查引擎（可选）： polygraphy inspect model identity.engine 查看生成的 TensorRT 引擎信息，确认网络结构和精度是否符合预期。
加载并运行： python3 load_and_run.py 演示了如何直接加载之前保存的 .engine 文件并执行推理，避免了重复编译。
4. 注意事项
硬件与平台强绑定： TensorRT 引擎是高度硬件绑定的。生成的 .engine 文件与生成它的 GPU 架构、TensorRT 版本、CUDA 版本强相关。在一台机器上生成的引擎，通常无法在另一台配置不同的机器上使用，跨机器部署时需要在目标机器上重新转换。
FP16 精度损失风险： 文档提到启用了 FP16 精度。虽然这能显著提升推理速度并降低显存占用，但对于某些对精度敏感的模型（如部分大语言模型的生成、极小数值的回归任务），可能会出现精度溢出（NaN）或结果偏差。必要时需要结合 Polygraphy 的精度调试工具进行排查，或保留部分关键层为 FP32 精度。
TensorRT 环境依赖： 运行此示例必须安装 NVIDIA TensorRT，且需要适配的 NVIDIA 显卡驱动和 CUDA 环境，这限制了其跨平台能力（无法在 AMD GPU 或纯 CPU 环境下运行）。
动态维度问题： 虽然示例中用的是 identity.onnx（通常是固定形状的简单模型），但在实际应用中，如果 ONNX 模型具有动态输入维度（如变长序列），在 Build 阶段需要通过 Polygraphy 或配置文件明确指定优化配置，否则可能无法达到最优性能或转换失败。
