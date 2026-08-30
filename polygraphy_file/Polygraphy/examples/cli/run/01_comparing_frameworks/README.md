# Comparing Frameworks

## Introduction

You can use the `run` subtool to compare a model between different frameworks.
In the simplest case, you can supply a model, and one or more framework flags.
By default, it will generate synthetic input data, run inference using the
specified frameworks, then compare outputs of the specified frameworks.

## Running The Example

In this example, we'll outline various common use-cases for the `run` subtool:

- [Comparing TensorRT And ONNX-Runtime Outputs](#comparing-tensorrt-and-onnx-runtime-outputs)
- [Comparing TensorRT Precisions](#comparing-tensorrt-precisions)
- [Changing Tolerances](#changing-tolerances)
- [Changing Comparison Metrics](#changing-comparison-metrics)
- [Comparing Per-Layer Outputs Between ONNX-Runtime And TensorRT](#comparing-per-layer-outputs-between-onnx-runtime-and-tensorrt)

### Comparing TensorRT And ONNX-Runtime Outputs

To run the model in Polygraphy with both frameworks and perform an output
comparison:

```bash
polygraphy run dynamic_identity.onnx --trt --onnxrt
```

The `dynamic_identity.onnx` model has dynamic input shapes. By default,
Polygraphy will override any dynamic input dimensions in the model to
`constants.DEFAULT_SHAPE_VALUE` (defined as `1`) and warn you:

<!-- Polygraphy Test: Ignore Start -->
```
[W]     Input tensor: X (dtype=DataType.FLOAT, shape=(1, 2, -1, -1)) | No shapes provided; Will use shape: [1, 2, 1, 1] for min/opt/max in profile.
[W]     This will cause the tensor to have a static shape. If this is incorrect, please set the range of shapes for this input tensor.
```
<!-- Polygraphy Test: Ignore End -->

In order to suppress this message and explicitly provide input shapes to
Polygraphy, use the `--input-shapes` option:

```
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --input-shapes X:[1,2,4,4]
```

### Comparing TensorRT Precisions

To build a TensorRT engine with reduced precision layers for comparison against
ONNXRT, use one of the supported precision flags (e.g. `--tf32`, `--fp16`,`--int8`, etc.).
For example:

```bash
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4]
```

> :warning: Getting acceptable accuracy with INT8 precision typically requires an additional calibration step:
  see the [developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)
  and instructions on [how to do calibration](../../../../examples/cli/convert/01_int8_calibration_in_tensorrt)
  with Polygraphy on the command line.

### Changing Tolerances

The default tolerances used by `run` are usually appropriate for FP32 precision
but may not be appropriate for reduced precisions. In order to relax tolerances,
you can use the `--atol` and `--rtol` options to set absolute and relative
tolerance respectively.

### Changing Comparison Metrics

You can use the `--check-error-stat` option to change the metric used for
comparison. By default, Polygraphy uses an "elementwise" metric
(`--check-error-stat elemwise`).

Other possible metrics for `--check-error-stat` are `mean`, `median`, and `max`, which
compares the mean, median, and maximum absolute/relative error across the tensor, respectively.

To better understand this, suppose we are
comparing two outputs `out0` and `out1`. Polygraphy takes
the elementwise absolute and relative difference of these tensors:

<!-- Polygraphy Test: Ignore Start -->
```
absdiff = out0 - out1
reldiff = absdiff / abs(out1)
```
<!-- Polygraphy Test: Ignore End -->

Then, for each index `i` in the output, Polygraphy checks whether
`absdiff[i] > atol and reldiff[i] > rtol`. If any index satisfies this,
then the comparison will fail.  This is less stringent than comparing the maximum
absolute and relative error across the entire tensor (`--check-error-stat max`) since if
*different* indices `i` and `j` satisfy `absdiff[i] > atol` and `reldiff[j] > rtol`,
then the `max` comparison will fail but the `elemwise` comparison may
pass.

Putting it all together, the below example runs a `median` comparison between
TensorRT using FP16 and ONNX-Runtime, using absolute and relative tolerances of `0.001`:

```bash
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes X:[1,2,4,4] \
    --atol 0.001 --rtol 0.001 --check-error-stat median
```

> You can also specify per-output values for `--atol`/`--rtol`/`--check-error-stat`.
  See the help output of the `run` subtool for more information.

### Comparing Per-Layer Outputs Between ONNX-Runtime And TensorRT

When network outputs do not match, it can be useful to compare per-layer outputs
to see where the error is introduced. To do so, you can use the `--trt-outputs`
and `--onnx-outputs` options respectively. These options accept one or more
output names as their arguments. The special value `mark all` indicates that all
tensors in the model should be compared:

```bash
 polygraphy run dynamic_identity.onnx --trt --onnxrt \
     --trt-outputs mark all \
     --onnx-outputs mark all
```

To find the first mismatched output more easily, you can use the `--fail-fast`
option which will cause the tool to exit after the first mismatch between
outputs.

Note that use of `--trt-outputs mark all` can sometimes perturb the generated
engine due to differences in timing, layer fusion choices, and format
constraints, which can hide the failure.  In that case, you may have to use a
more sophisticated approach to bisect the failing model and generate a reduced
test case that reproduces the error. See [Reducing Failing ONNX
Models](../../../../examples/cli/debug/02_reducing_failing_onnx_models) for a tutorial on
how to do this with Polygraphy.

## Further Reading

* In some cases you may need to do comparisons across multiple Polygraphy runs
  (for example, when comparing the output of a pre-built TensorRT engine or
  [Polygraphy network script](../../../../examples/cli/run/04_defining_a_tensorrt_network_or_config_manually)
  against ONNX-Runtime). See [Comparing Across Runs](../../../../examples/cli/run/02_comparing_across_runs) for a tutorial on how to
  accomplish this.

* For more details on working with dynamic shapes in TensorRT:
  * See [Dynamic Shapes in TensorRT](../../../../examples/cli/convert/03_dynamic_shapes_in_tensorrt/) for how to specify
    optimization profiles for use with the engine using the Polygraphy CLI
  * See [TensorRT and Dynamic Shapes](../../../../examples/api/07_tensorrt_and_dynamic_shapes/) for details on
    how to do this with the Polygraphy API

* For details on how to supply real input data, see [Comparing with Custom Input Data](../05_comparing_with_custom_input_data/).

* See [Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md) for a broader tutorial on how to debug accuracy failures using Polygraphy.
==============================================================================
这份文档是 Polygraphy 工具中 run 子命令的使用指南，核心目标是：在不同深度学习推理框架（如 TensorRT、ONNX Runtime）之间对同一个模型进行输出对比，以验证精度一致性或调试精度问题。

下面我将逐部分详细解析其内容、原理和工程价值。

 一、整体定位与用途
 Polygraphy 是什么？NVIDIA 开发的 模型调试与验证工具集;专注于 TensorRT 与其他框架（ONNX Runtime、PyTorch 等）的兼容性验证
 polygraphy run 的核心功能：
功能	          说明
多框架推理	    同时用 TRT、ONNXRT 等运行同一模型
自动输入生成	  自动生成符合模型输入 shape/dtype 的随机数据
输出对比	      比较各框架输出是否“足够接近”
精度分析	      支持 FP32/FP16/INT8 等不同精度下的对比
逐层调试	      可比较中间层输出，定位误差来源

 典型使用场景：验证 TensorRT 引擎是否与 ONNX 模型等价;调试 FP16/INT8 精度下降问题
定位模型转换（ONNX → TRT）中的 bug
 二、关键特性详解
1 基础用法：跨框架输出对比
polygraphy run model.onnx --trt --onnxrt
自动行为：生成随机输入（默认值 1 填充动态维度）;分别用 TensorRT 和 ONNX Runtime 推理
比较最终输出张量
输出：若一致：显示 PASSED;若不一致：显示误差统计 + 失败原因

 注意动态 shape 警告：若模型有 -1 维度（如 [1, 3, -1, -1]），Polygraphy 默认用 1 替代，并警告你这会变成静态 shape。

正确做法：显式指定 --input-shapes X:[1,3,224,224]

2 精度对比：验证低精度是否可接受
polygraphy run model.onnx --trt --fp16 --onnxrt --input-shapes X:[1,3,224,224]
--fp16：让 TensorRT 构建 FP16 引擎（自动启用 mixed-precision）
对比 FP16 (TRT) vs FP32 (ONNXRT) 输出
 INT8 特别说明：INT8 需要校准（calibration）才能保证精度;直接加 --int8 会使用默认校准（可能不准）
正确做法：先用 校准脚本 生成校准表

 工程建议：FP16 通常无需校准即可用于对比；INT8 必须校准后才可信任。

3 容差控制：如何定义“足够接近”？
默认容差（适用于 FP32）：atol = 1e-5（绝对误差容忍）;rtol = 1e-5（相对误差容忍）
降低精度时需放宽容差：
# FP16 允许更大误差
polygraphy run ... --atol 1e-3 --rtol 1e-3
误差判断逻辑（关键！）：对每个元素 i，检查是否 同时 满足：
abs(out_trt[i] - out_onnx[i]) > atol   AND
rel_error = abs(diff) / max(abs(out_onnx[i]), eps) > rtol
→ 只要有一个元素同时超限，就判定失败

 这比“最大误差”更宽松：允许某些点绝对误差大（但相对小），或相对误差大（但绝对小）。

4 误差统计方式：--check-error-stat
选项	            含义	                           适用场景
elemwise (默认)	 按上述“逐元素 AND”规则判断	    通用，避免个别 outlier 导致失败
max	     检查整个张量的最大绝对/相对误差	      严格模式，确保全局误差可控
mean / median	   检查平均/中位误差	           忽略局部噪声，关注整体趋势

示例：
# 使用中位误差 + 宽松容差
polygraphy run ... --atol 0.001 --rtol 0.001 --check-error-stat median
 推荐：调试初期用 median 或 mean 快速判断整体趋势;上线前用 max 或 elemwise 确保无严重错误
5 逐层输出对比：定位误差源头;当最终输出不一致时，需知道从哪一层开始出错：
polygraphy run model.onnx --trt --onnxrt \
    --trt-outputs mark all \
    --onnx-outputs mark all
mark all：强制保存所有中间层输出（即使不是原始输出）
Polygraphy 会自动匹配同名 tensor 并逐一对比
 注意事项：
可能影响 TRT 优化：标记所有输出会禁用某些融合优化（如 conv+relu 合并），导致 engine 行为改变;若仍不一致：需用 模型简化工具 生成最小复现 case
实用技巧：
--fail-fast：遇到第一个不匹配层就退出，加速调试
可指定具体 layer 名：--trt-outputs layer1_out,layer2_out
 三、高级用法与最佳实践
 场景 1：使用真实输入数据（非随机）;默认用随机数据，但可能无法触发特定 bug;
可通过 --load-inputs data.pkl 加载预存输入（见 Custom Input Data）
 场景 2：对比已构建的 TRT engine;不是从 ONNX 构建，而是直接加载 .engine 文件
需配合 --model-type trt 和 --load-outputs（见 Comparing Across Runs）
 场景 3：动态 shape 支持;必须显式提供 optimization profile：
--input-shapes X:[1,3,128,128]  # min/opt/max 相同
# 或
--trt-min-shapes X:[1,3,64,64] \
--trt-opt-shapes X:[1,3,256,256] \
--trt-max-shapes X:[1,3,512,512]
 四、相关资源指引;文档末尾提供了精准的进阶路径：
需求               对应教程
调试 TRT 精度问题	  Debugging TensorRT Accuracy Issues
动态 shape 处理	    Dynamic Shapes in TensorRT
自定义输入数据	    Comparing with Custom Input Data
跨次运行对比	      Comparing Across Runs
模型简化复现 bug	  Reducing Failing ONNX Models
 五、总结：这份文档的核心价值
维度	         说明
目标用户	    TensorRT 开发者、模型部署工程师
解决痛点	    “TRT 结果和 ONNX 不一样，怎么 debug？”
方法论	     从 整体输出 → 逐层输出 → 最小复现 的系统化调试流程
工程思想	  量化误差（atol/rtol）、控制变量（精度/输入）、分层验证（per-layer）

 一句话精髓：不要只看“结果对不对”，而要问“哪里开始不对、为什么不对、能否接受这种不对”。
Polygraphy 的 run 命令正是实现这一思想的利器。如果你正在做 TensorRT 部署，这份文档就是你的精度验证操作手册 —— 建议收藏并结合实际案例反复使用。

