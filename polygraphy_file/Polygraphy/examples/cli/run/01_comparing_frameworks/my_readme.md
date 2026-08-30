# 框架间比较
## 简介
您可以使用 run 子工具在不同框架之间对模型进行比较。
在最简单的情况下，您只需提供一个模型以及一个或多个框架标志即可。
默认情况下，该工具会生成合成输入数据，使用指定的框架运行推理，然后比较这些框架的输出结果。
## 运行示例:
在本例中，我们将概述 run 子工具的几种常见用例：
比较 TensorRT 与 ONNX-Runtime 的输出
比较 TensorRT 的不同精度
调整容差
更改比较指标
（# 比较 ONNX-Runtime 与 TensorRT 的逐层输出）

### 比较 TensorRT 与 ONNX-Runtime 的输出
要在 Polygraphy 中使用两个框架运行模型并执行输出比较：
polygraphy run dynamic_identity.onnx --trt --onnxrt

dynamic_identity.onnx 模型具有动态输入形状。默认情况下，Polygraphy 会将模型中的所有动态输入维度覆盖为 constants.DEFAULT_SHAPE_VALUE（定义为 1），并发出警告：
[W]     Input tensor: X (dtype=DataType.FLOAT, shape=(1, 2, -1, -1)) | No shapes provided; Will use shape: [1, 2, 1, 1] for min/opt/max in profile.
[W]     This will cause the tensor to have a static shape. If this is incorrect, please set the range of shapes for this input tensor.

若要抑制此消息并向 Polygraphy 显式提供输入形状，请使用 --input-shapes 选项：
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --input-shapes X:[1,2,4,4]

### 比较 TensorRT 的不同精度
若要构建包含降低精度层的 TensorRT 引擎并与 ONNXRT 进行比较，请使用支持的精度标志之一（例如 --tf32、--fp16、--int8 等）。
例如：
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4]

:warning: 要在 INT8 精度下获得可接受的准确率，通常需要额外的校准步骤：
请参阅开发者指南 (https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
以及关于如何在命令行中使用 Polygraphy 进行校准的说明。(../../../../examples/cli/convert/01_int8_calibration_in_tensorrt)

# 调整容差
run 使用的默认容差通常适用于 FP32 精度，但可能不适用于降低的精度。为了放宽容差，您可以分别使用 --atol 和 --rtol 选项来设置绝对容差和相对容差。

## 更改比较指标
您可以使用 --check-error-stat 选项来更改用于比较的指标。默认情况下，Polygraphy 使用“逐元素”指标（--check-error-stat elemwise）。
--check-error-stat 的其他可选指标包括 mean、median 和 max，它们分别比较整个张量的平均、中位数和最大绝对/相对误差。
为了更好地理解这一点，假设我们正在比较两个输出 out0 和 out1。Polygraphy 会计算这两个张量的逐元素绝对差值和相对差值：
absdiff = out0 - out1
reldiff = absdiff / abs(out1)

然后，对于输出中的每个索引 i，Polygraphy 会检查是否满足 absdiff[i] > atol and reldiff[i] > rtol。如果任意索引满足该条件，则比较失败。这比比较整个张量上的最大绝对和相对误差（--check-error-stat max）要宽松，因为如果不同的索引 i 和 j 分别满足 absdiff[i] > atol 和 reldiff[j] > rtol，那么 max 比较将会失败，但 elemwise 比较可能会通过。
综合以上内容，下面的示例在 FP16 精度的 TensorRT 与 ONNX-Runtime 之间运行 median（中位数）比较，并将绝对容差和相对容差均设为 0.001：
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4] \
    --atol 0.001 --rtol 0.001 --check-error-stat median

您也可以为 --atol/--rtol/--check-error-stat 指定针对每个输出的独立值。
更多信息请参阅 run 子工具的帮助输出。

### 比较 ONNX-Runtime 与 TensorRT 的逐层输出
当网络输出不匹配时，比较逐层输出有助于定位误差引入的位置。为此，您可以分别使用 --trt-outputs 和 --onnx-outputs 选项。这些选项接受一个或多个输出名称作为参数。特殊值 mark all 表示应比较模型中的所有张量：
 polygraphy run dynamic_identity.onnx --trt --onnxrt \
     --trt-outputs mark all \
     --onnx-outputs mark all

为了更轻松地找到第一个不匹配的输出，您可以使用 --fail-fast 选项，该选项会使工具在发现第一个输出不匹配后立即退出。
请注意，使用 --trt-outputs mark all 有时可能会因时序差异、层融合选择及格式约束等原因干扰生成的引擎，从而掩盖故障。在这种情况下，您可能需要采用更复杂的方法来二分查找故障模型，并生成一个能复现错误的简化测试用例。有关如何使用 Polygraphy 执行此操作的教程，请参阅缩减失败的 ONNX 模型。
# 延伸阅读
在某些情况下，您可能需要在多次 Polygraphy 运行之间进行比较（例如，将预构建的 TensorRT 引擎或 Polygraphy 网络脚本的输出与 ONNX-Runtime 进行比较）。(../../../../examples/cli/run/04_defining_a_tensorrt_network_or_config_manually)。

有关如何实现这一点的教程，请参阅跨运行比较。(../../../../examples/cli/run/02_comparing_across_runs)

有关在 TensorRT 中处理动态形状的更多详情：
参阅 TensorRT 中的动态形状，(../../../../examples/cli/convert/03_dynamic_shapes_in_tensorrt/)；

了解如何使用 Polygraphy CLI 指定优化配置文件以供引擎使用。(../../../../examples/api/07_tensorrt_and_dynamic_shapes/)
参阅 TensorRT 与动态形状，了解如何使用 Polygraphy API 实现此操作。

有关如何提供真实输入数据的详情，请参阅使用自定义输入数据进行比较。(../05_comparing_with_custom_input_data/).

有关如何使用 Polygraphy 调试精度问题的更广泛教程，请参阅调试 TensorRT 精度问题。(../../../../how-to/debug_accuracy.md)