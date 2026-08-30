# Checking for Intermediate NaN or Infinities

## Introduction

When debugging model accuracy issues in Polygraphy, it can be helpful to check layerwise outputs for potential problems. Polygraphy's `run` subtool provides a helpful flag `--validate` which can quickly diagnose problematic intermediate outputs.

This example demonstrates use of this flag with a model which intentionally generates an
infinite output by adding infinity to the input tensor.

## Running The Example

 <!-- Polygraphy Test: XFAIL Start -->
```bash
polygraphy run add_infinity.onnx --onnx-outputs mark all --onnxrt --validate
```
 <!-- Polygraphy Test: XFAIL End -->

 <!-- Polygraphy Test: Ignore Start -->
You should see output like:
```
[I] onnxrt-runner-N0-05/13/22-22:35:48  | Completed 1 iteration(s) in 0.1326 ms | Average inference time: 0.1326 ms.
[I] Output Validation | Runners: ['onnxrt-runner-N0-05/13/22-22:35:48']
[I]     onnxrt-runner-N0-05/13/22-22:35:48  | Validating output: B (check_inf=True, check_nan=True)
[I]         mean=inf, std-dev=nan, var=nan, median=inf, min=inf at (0,), max=inf at (0,), avg-magnitude=inf
[E]         Inf Detected | One or more non-finite values were encountered in this output
[I]         Note: Use -vv or set logging verbosity to EXTRA_VERBOSE to display non-finite values
[E]         FAILED | Errors detected in output: B
[E]     FAILED | Output Validation
```
 <!-- Polygraphy Test: Ignore End -->

## See Also

* [Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md)
=============================================================================
这份文档《Checking for Intermediate NaN or Infinities》展示了 如何使用 Polygraphy 的 --validate 功能快速检测模型推理过程中是否出现数值异常（如 NaN、Inf），这是调试模型精度问题（尤其是部署后精度下降）的关键手段。

下面我将从问题背景、核心机制、命令解析、输出解读、扩展用法和工程价值六个维度，为你深度解析。
 一、解决的核心问题：为什么需要检测 NaN/Inf？
在深度学习模型部署中，以下情况会导致中间层或输出出现非有限值（NaN/Inf）：
原因	    说明
数值溢出	如 softmax 输入过大 → exp(x) = Inf
除零错误	LayerNorm、BatchNorm 中方差为 0
不合法操作	log(0) = -Inf, sqrt(-1) = NaN
权重损坏	模型文件损坏或量化错误引入 Inf/NaN
动态 shape 不匹配	张量 reshape 失败导致未初始化内存

 后果：一旦出现 Inf/NaN，后续所有计算结果都不可信，但模型可能“静默失败”（无报错，只输出乱码）。

Polygraphy 的 --validate 就是为此设计的自动哨兵。

 二、核心机制：--validate 如何工作？
工作流程:启用所有中间输出
--onnx-outputs mark all：强制 ONNX Runtime 输出每一层的中间结果
运行推理:使用指定后端（如 --onnxrt）执行前向传播;逐张量验证;对每个输出张量（包括中间层）检查：是否包含 NaN（check_nan=True）;是否包含 Inf 或 -Inf（check_inf=True）
报告异常:一旦发现非有限值，立即标记失败并打印统计信息
 无需修改模型代码！纯工具链级检测。

 三、命令行详解
polygraphy run add_infinity.onnx --onnx-outputs mark all --onnxrt --validate
参数	             作用
add_infinity.onnx	一个故意构造的故障模型（内部有 x + inf 操作）
--onnx-outputs mark all	关键！ 要求 ONNX Runtime 输出所有中间节点的值
（默认只输出最终输出）
--onnxrt	使用 ONNX Runtime 作为推理后端
--validate	启用输出验证（自动开启 NaN/Inf 检查）

 注意：--validate 本身不控制哪些张量被检查 —— 它检查所有被请求输出的张量。
因此必须配合 --onnx-outputs mark all 才能检测中间层。

 四、输出日志深度解读
[I] onnxrt-runner-... | Completed 1 iteration(s)...
[I] Output Validation | Runners: ['onnxrt-runner-...']
[I]     onnxrt-runner-... | Validating output: B (check_inf=True, check_nan=True)
[I]         mean=inf, std-dev=nan, var=nan, median=inf, min=inf at (0,), max=inf at (0,), avg-magnitude=inf
[E]         Inf Detected | One or more non-finite values were encountered in this output
[E]         FAILED | Errors detected in output: B
[E]     FAILED | Output Validation
关键信息提取：
问题张量名：B（这是 ONNX 模型中某个节点的输出名）
异常类型：Inf Detected（包含无穷大）
位置信息：min=inf at (0,) → 第 0 个元素是 Inf
统计特征：
mean=inf：均值为无穷 → 说明存在 Inf
std-dev=nan：标准差为 NaN → 因为 Inf 参与计算导致

 定位技巧：用 Netron 打开 add_infinity.onnx，搜索输出名为 B 的节点，即可定位到问题算子（很可能是 Add 节点）。

 五、扩展用法与高级技巧
1. 仅检查最终输出（不检查中间层）
# 默认行为：只验证模型最终输出
polygraphy run model.onnx --trt --validate
2. 自定义检查级别
# 仅检查 NaN（忽略 Inf）
polygraphy run ... --validate --no-check-inf

# 仅检查 Inf（忽略 NaN）
polygraphy run ... --validate --no-check-nan
3. 查看具体异常值位置（需更高日志级别）
# 显示哪些位置是 NaN/Inf
polygraphy run ... --validate -vv
# 或
POLYGRAPHY_LOG_LEVEL=EXTRA_VERBOSE polygraphy run ...
输出示例：
Non-finite values detected in output 'B':
    Inf at indices: [(0, 0, 0, 0), (0, 0, 0, 1)]
    NaN at indices: []
4. 结合 TensorRT 使用
# 检测 TRT 引擎中的数值异常
polygraphy run model.onnx --trt \
    --trt-outputs mark all \  # TRT 版本的 "mark all"
    --validate
5. 自动化回归测试
在 CI 脚本中加入：
if ! polygraphy run model.onnx --onnxrt --validate; then
    echo "ERROR: Model produces NaN/Inf!" >&2
    exit 1
fi
 六、注意事项
性能开销:
--onnx-outputs mark all 会显著降低推理速度（需保存所有中间张量），仅用于调试
内存占用:大模型 + 所有中间输出 → 可能 OOM，建议配合小 batch size 使用

动态 shape 模型:需确保输入 shape 能触发问题路径（如某些 shape 才会 overflow）

FP16 模型:Inf/NaN 在 FP16 中更常见（动态范围小），建议优先用 FP32 验证

 七、总结：这份文档的工程价值
维度	    说明
快速诊断	1 条命令定位 NaN/Inf 问题，无需手动插桩
全栈覆盖	支持 ONNX Runtime、TensorRT 等主流后端
精准定位	报告具体张量名、位置、统计特征
CI 友好	可集成到自动化测试，防止带病模型上线

 记住这个黄金命令：
polygraphy run model.onnx --onnx-outputs mark all --onnxrt --validate
这是你面对“模型输出全是 nan”时的第一响应武器。
在实际项目中，该方法已帮助无数工程师快速定位了因 LayerNorm 方差为零、Softmax 溢出、自定义 CUDA kernel bug 等导致的精度灾难，是 MLOps 工具箱中的必备利器。

