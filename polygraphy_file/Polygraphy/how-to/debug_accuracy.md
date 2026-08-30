# Debugging TensorRT Accuracy Issues

Accuracy issues in TensorRT, especially with large networks, can be challenging to debug.
One way to make them manageable is to reduce the problem size or pinpoint the source of failure.

This guide aims to provide a general approach to doing so; it is structured as a flattened flowchart -
at each branch, two links are provided so you can choose the one that best matches your situation.

If you're using an ONNX model, try [sanitizing it](../examples/cli/surgeon/02_folding_constants/) before
proceeding, as this may solve the problem in some cases.


## Does Real Input Data Make A Difference?

Some models may be sensitive to input data. For example, real inputs may result in better accuracy
than randomly generated ones. Polygraphy offers multiple ways to supply real input
data, outlined in [`run` example 05](../examples/cli/run/05_comparing_with_custom_input_data/).

Does using real input data improve the accuracy?

- Yes, accuracy is acceptable when using real input data.

    This likely means there is no bug; rather, your model is sensitive to input data.

- No, I still see accuracy issues even with real input data.

    Go To: [Intermittent Or Not?](#intermittent-or-not)


## Intermittent Or Not?

Is the issue intermittent between engine builds?

- Yes, sometimes the accuracy issue disappears when I rebuild the engine.

    Go To: [Debugging Intermittent Accuracy Issues](#debugging-intermittent-accuracy-issues)

- No, I see accuracy issues every time I build an engine.

    Go To: [Is Layerwise An Option?](#is-layerwise-an-option)


## Debugging Intermittent Accuracy Issues

Since the engine building process is non-deterministic, different tactics (i.e. layer implementations) may
be selected each time the engine is built. When one of the tactics is faulty, this may manifest as an intermittent
failure. Polygraphy includes a `debug build` subtool to help you find such tactics.

For more information, refer to [`debug` example 01](../examples/cli/debug/01_debugging_flaky_trt_tactics/).

Were you able to find the failing tactic?

- Yes, I know which tactic is faulty.

    Go To: [You Have A Minimal Failing Case!](#you-have-a-minimal-failing-case)

- No, the failure may not be intermittent.

    Go To: [Is Layerwise An Option?](#is-layerwise-an-option)


## Is Layerwise An Option?

If the accuracy issue is consistently reproducible, the best next step is to figure out which
layer is causing the failure. Polygraphy includes a mechanism to mark all tensors in the network
as outputs so that they can be compared; however, this can potentially affect TensorRT's optimization
process. Hence, we need to determine if we still observe the accuracy issue when all output tensors are marked.

Refer to [this example](../examples/cli/run/01_comparing_frameworks/README.md#comparing-per-layer-outputs-between-onnx-runtime-and-tensorrt) for details on how to compare
per-layer outputs before proceeding.

Were you able to reproduce the accuracy failure when comparing layer-wise outputs?

- Yes, the failure reprodces even if I mark other outputs in the network.

    Go To: [Extracting A Failing Subgraph](#extracting-a-failing-subgraph)

- No, marking other outputs causes the accuracy to improve OR I am not able to run the model at all when I mark other outputs.

    Go To: [Reducing A Failing Onnx Model](#reducing-a-failing-onnx-model)


## Extracting A Failing Subgraph

Since we're able to compare layerwise outputs, we should be able to determine which layer
first introduces the error by looking at the output comparison logs. Once we know which layer
is problematic, we can extract it from the model.

In order to figure out the input and output tensors for the layer in question, we can use
`polygraphy inspect model`. Refer to one of these examples for details:

- [TensorRT Networks](../examples/cli/inspect/01_inspecting_a_tensorrt_network/)
- [ONNX models](../examples/cli/inspect/03_inspecting_an_onnx_model/).

Next, we can extract a subgraph including just the problematic layer.
For more information, refer to [`surgeon` example 01](../examples/cli/surgeon/01_isolating_subgraphs/).

Does this isolated subgraph reproduce the problem?

- Yes, the subgraph fails too.

    Go To: [You Have A Minimal Failing Case!](#you-have-a-minimal-failing-case)

- No, the subgraph works fine.

    Go To: [Reducing A Failing Onnx Model](#reducing-a-failing-onnx-model)


## Reducing A Failing ONNX Model

When we're unable to pinpoint the source of failure using a layerwise comparison, we can
use a brute force method of reducing the ONNX model - iteratively generate smaller and smaller
subgraphs to find the smallest possible one that still fails. The `debug reduce` tools helps automate this process.

For more information, refer to [`debug` example 02](../examples/cli/debug/02_reducing_failing_onnx_models/).

Does the reduced model fail?

- Yes, the reduced model fails.

    Go To: [You Have A Minimal Failing Case!](#you-have-a-minimal-failing-case)

- No, the reduced model doesn't fail, or fails in a different way.

    Go To: [Double Check Your Reduce Options](#double-check-your-reduce-options)


## Double Check Your Reduce Options

If the reduced model no longer fails, or fails in a different way, ensure that your `--check` command
is correct. You may also want to use `--fail-regex` to ensure that you're only considering the accuracy
failure (and not other, unrelated failures) when reducing the model.

- Try reducing again.

    Go To: [Reducing A Failing Onnx Model](#reducing-a-failing-onnx-model)

## You Have A Minimal Failing Case!

If you've made it to this point, you now have a minimal failing case! Further debugging should
be significantly easier.

If you are a TensorRT developer, you'll need to dive into the code at this point.
If not, please report your bug!
================================================================================
这份文档是一份关于 Debugging TensorRT Accuracy Issues（调试 TensorRT 精度问题） 的实战指南。它提供了一套系统化的、基于决策树的排查流程，帮助开发者在遇到 TensorRT 推理结果与原始模型（如 PyTorch、ONNX Runtime）不一致时，能够快速定位问题根源。
以下是对该文档的详细解析，包括其含义、作用，以及结合实际场景和高级工具的应用示例：
1. 文档的含义与作用
核心含义：TensorRT 为了追求极致性能，会对模型进行图优化（如层融合、常量折叠）、内核自动调优（Tactic Selection）以及低精度量化（FP16/INT8）。这些优化不可避免地会引入微小的数值差异。当差异超出预期导致业务指标（如 mAP、Top-1 准确率）显著下降时，就需要进行精度调试。该文档提供了一个“扁平化流程图”，指导开发者一步步缩小问题范围。
核心作用：
降低调试复杂度：将庞大的神经网络黑箱拆解，通过控制变量法（如使用真实数据、逐层对比）隔离问题。
自动化排查：结合 Polygraphy 工具，自动完成逐层对比、子图提取、模型裁剪等繁琐操作。
生成最小复现用例（Minimal Failing Case）：最终目标是得到一个极小的、能稳定复现精度问题的模型片段，这对于自行修复或向 NVIDIA 提交 Bug 报告至关重要。
2. 核心排查流程解析与应用示例
文档的排查逻辑可以总结为以下四个关键阶段：
阶段一：排除输入数据干扰 (Does Real Input Data Make A Difference?)
含义：某些模型对输入数据分布非常敏感。使用随机生成的假数据进行测试时可能会出现精度问题，但换成真实的业务数据后问题消失。
实际应用示例：
在使用 Polygraphy 对比 ONNX Runtime 和 TensorRT 的输出时，默认可能会使用随机数据。你可以指定真实数据进行验证：
polygraphy run model.onnx \
    --onnxrt \
    --trt \
    --load-inputs real_data.json  # 加载真实业务数据
结果判断：如果换成真实数据后精度达标，说明模型本身没有 Bug，只是对输入分布敏感，无需继续深度调试。
阶段二：判断问题是否为间歇性 (Intermittent Or Not?)
含义：TensorRT 在构建引擎时会尝试多种 CUDA 内核实现（Tactics）。由于构建过程的非确定性，有时选中的某个 Tactic 存在缺陷，导致精度问题时有时无。
实际应用示例：
如果你发现每次重新构建 .engine 文件，精度表现都不一样，可以使用 Polygraphy 的 debug build 命令来自动定位有问题的 Tactic：
polygraphy debug build model.onnx \
    --check "polygraphy run model.onnx --onnxrt --trt --atol 1e-3"
结果判断：该命令会反复构建引擎并测试，最终找出导致精度失败的具体 Tactic。你可以选择禁用该 Tactic 来解决问题。
阶段三：逐层对比定位 (Is Layerwise An Option?)
含义：如果问题是稳定复现的，最有效的方法是让 TensorRT 输出网络中每一层的中间结果，并与 ONNX Runtime 的黄金标准进行逐层对比，找到第一个出现显著误差的层。
实际应用示例：
polygraphy run model.onnx \
    --onnxrt \
    --trt \
    --validate  # 逐层对比输出
结果判断：日志会列出每一层的误差。假设发现 Conv_15 层的误差突然从 1e-5 飙升到 1e-2，那么 Conv_15 就是嫌疑层。
阶段四：隔离子图与模型裁剪 (Extracting Subgraph / Reducing Model)
含义：找到嫌疑层后，尝试将其提取为独立子图进行验证。如果提取后问题消失（可能是因为上下文依赖或优化策略改变），则使用暴力裁剪工具 debug reduce，自动删减无关节点，直到得到最小的失败模型。
实际应用示例：
使用 surgeon 提取子图：
polygraphy surgeon extract model.onnx \
    --start Conv_14 \
    --end Conv_16 \
    -o subgraph.onnx
如果子图无法复现，使用 debug reduce 自动裁剪：
polygraphy debug reduce model.onnx \
    --check "polygraphy run model.onnx --onnxrt --trt --atol 1e-3" \
    -o minimal_fail.onnx
3. 高级应用示例与工业级实践
在实际的工业级部署中，精度问题往往不仅仅是找 Bug，还涉及量化策略和架构特性的权衡。结合更广泛的 TensorRT 生态，以下是高级应用示例：
高级示例 1：INT8 量化精度崩塌的系统性排查
当模型从 FP32 转为 INT8 后 mAP 暴跌 5% 以上，这通常是系统性数值退化，而非单纯的 Tactic 错误。
排查动作：
校准数据溯源：检查校准数据集（Calibration Dataset）是否具有代表性。如果校准集只有 50 张室内图，而线上场景包含大量夜间/模糊图像，会导致激活值统计失真。建议使用 500-1000 张覆盖真实场景分布的无标签数据进行校准。
敏感层强制回退：使用 Polygraphy debug precision 自动分析哪些层对 INT8 最敏感。通常 Softmax 前、残差连接处、LayerNorm 等层容易溢出或丢失精度。
代码级干预：通过 ONNX GraphSurgeon 或 TensorRT API，将这些敏感层强制保留为 FP16 或 FP32：
# 伪代码示例：强制特定层使用高精度
config.set_flag(trt.BuilderFlag.INT8)
# 对敏感层白名单强制 FP16
network.get_layer(i).precision = trt.DataType.HALF 
高级示例 2：Ampere 架构下的 TF32 精度陷阱
在 RTX 3090、A100 等 Ampere 架构显卡上，即使你指定了 FP32 精度，TensorRT 也可能默认开启 TF32 以加速计算。TF32 的精度介于 FP32 和 FP16 之间，可能导致误差从 1e-6 放大到 1e-4。
排查动作：
如果你在 Ampere 显卡上遇到莫名其妙的 FP32 精度下降，首先尝试关闭 TF32：
# 命令行方式
trtexec --onnx=model.onnx --noTF32
# API 方式
config.clear_flag(trt.BuilderFlag.TF32)
如果关闭后精度恢复正常，说明模型对 TF32 的截断误差敏感，需在精度和性能间做权衡。
高级示例 3：利用 TRT Engine Explorer (TREX) 进行可视化诊断
当逐层对比发现误差，但不知道 TensorRT 内部到底做了什么优化时，可以使用 TREX 工具。
排查动作：
使用 trtexec 导出性能 Profile 和引擎文件。
使用 TREX 加载引擎，可视化计算图。
观察重点：检查嫌疑层是否被与其他层（如 Conv+BN+ReLU）融合。层融合虽然提升性能，但在 BN 参数极端（如方差接近 0）时，FP16 下的 1/sqrt(var) 可能导致溢出或精度骤降。通过 TREX 确认融合情况后，可考虑在 ONNX 阶段手动拆分该节点。
总结
这份文档提供了一套严谨的“控制变量 -> 逐层定位 -> 最小化复现”的调试方法论。在实际应用中，建议：
先跑通 FP32：确保 FP32 下 TensorRT 与 ONNX Runtime 误差在 1e-3 ~ 1e-4 级别（受浮点运算顺序影响，无法做到完全位级一致）。
再碰低精度：FP16 通常无损或微损；INT8 必须配合高质量的校准数据和敏感层白名单策略。
善用工具链：Polygraphy 是精度调试的瑞士军刀，TREX 是性能与结构分析的显微镜，ONNX GraphSurgeon 是模型结构修改的手术刀。三者结合，能解决 99% 的 TensorRT 精度问题。