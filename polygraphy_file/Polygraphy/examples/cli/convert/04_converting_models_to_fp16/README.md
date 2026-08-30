# Converting ONNX Models To FP16

## Introduction

When debugging accuracy issues with using TensorRT reduced precision
optimizations (`--fp16` and `--tf32` flags) on an ONNX model trained in FP32,
it can be helpful to convert the model to FP16 and run it under ONNX-Runtime
to check if there are might be problems inherent to running the model
with reduced precision.

## Running The Example

1. Convert the model to FP16:

   ```bash
   polygraphy convert --fp-to-fp16 -o identity_fp16.onnx identity.onnx
   ```

2. **[Optional]** Inspect the resulting model:

   ```bash
   polygraphy inspect model identity_fp16.onnx
   ```

3. **[Optional]** Run the FP32 and FP16 models under ONNX-Runtime and then compare the results:

   ```bash
   polygraphy run --onnxrt identity.onnx \
      --save-inputs inputs.json --save-outputs outputs_fp32.json
   ```

   ```bash
   polygraphy run --onnxrt identity_fp16.onnx \
      --load-inputs inputs.json --load-outputs outputs_fp32.json \
      --atol 0.001 --rtol 0.001
   ```

4. **[Optional]** Check if any intermediate outputs of the FP16 model
   contain NaN or infinity (see [Checking for Intermediate NaN or Infinities](../../../../examples/cli/run/07_checking_nan_inf)):

   ```bash
   polygraphy run --onnxrt identity_fp16.onnx --onnx-outputs mark all --validate
   ```

## See Also

* [Comparing Across Runs](../../../../examples/cli/run/02_comparing_across_runs)
* [Checking for Intermediate NaN or Infinities](../../../../examples/cli/run/07_checking_nan_inf)
* [Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md)
=============================================================================
这份文档是 Polygraphy（NVIDIA 官方 TensorRT 调试工具集） 中关于 将 ONNX 模型从 FP32 转换为 FP16 以辅助精度调试 的典型工作流说明。其核心目标是：在将模型交给 TensorRT 做 FP16 推理前，先验证“模型本身是否能在纯 FP16 下正常工作”。

下面将从 背景动机、技术原理、命令详解、工程价值、常见误区 五个维度进行深度解析。

一、为什么需要这个流程？—— 背景与动机
问题场景：你有一个在 PyTorch/TensorFlow 中用 FP32 训练的 ONNX 模型，想用 TensorRT 的 --fp16 加速推理，但发现：输出结果偏差很大（如分类错误、检测框偏移），甚至出现 NaN / Inf。核心疑问：这是 TensorRT 的问题？还是模型本身就不适合 FP16？

关键洞察：TensorRT 的 FP16 优化包含两部分：计算精度降低（FP32 → FP16）；算子融合、kernel 选择等图优化；→ 如果直接在 TensorRT 中调试，无法区分问题是来自 精度损失 还是 优化 bug。
解决方案：
先在 ONNX Runtime（无图优化、纯精度测试）中运行纯 FP16 模型：
若 ONNX Runtime 的 FP16 结果 already bad → 模型本身不兼容 FP16
若 ONNX Runtime 的 FP16 结果 good，但 TensorRT bad → 可能是 TensorRT 优化问题
 这是一个 隔离变量（isolate the issue） 的经典调试策略。

二、技术原理：FP32 → FP16 转换做了什么？
polygraphy convert --fp-to-fp16 -o identity_fp16.onnx identity.onnx
该命令执行以下操作：
1. 遍历模型所有权重和常量
将 float32 类型的参数（如卷积核、BN 参数）转换为 float16，保留网络拓扑结构不变
2. 插入必要的类型转换节点（Cast）
在输入/输出或算子之间插入 Cast(to=10)（10 = FP16）或 Cast(to=1)（1 = FP32）
确保 ONNX Runtime 能正确解释数据流
3. 不改变计算逻辑
所有算子（Conv, MatMul, LayerNorm 等）仍按原语义执行，只是内部计算使用 FP16（由 ONNX Runtime 决定）。注意：这不是量化（Quantization），而是纯精度降级（Precision Downcasting）。

三、命令逐行详解
步骤 1：转换模型
polygraphy convert --fp-to-fp16 -o identity_fp16.onnx identity.onnx
--fp-to-fp16：启用 FP32→FP16 转换
输入：identity.onnx（FP32）
输出：identity_fp16.onnx（模型内嵌 FP16 权重）
步骤 2：检查模型（可选）
polygraphy inspect model identity_fp16.onnx
查看是否所有权重已变为 tensor(float16)；验证是否有异常节点
步骤 3：对比 FP32 vs FP16 结果
# 运行 FP32 模型，保存输入输出
polygraphy run --onnxrt identity.onnx \
   --save-inputs inputs.json --save-outputs outputs_fp32.json

# 用相同输入运行 FP16 模型，并与 FP32 结果比较
polygraphy run --onnxrt identity_fp16.onnx \
   --load-inputs inputs.json --load-outputs outputs_fp32.json \
   --atol 0.001 --rtol 0.001
关键参数：
参数	                     作用
--save-inputs	   保存随机生成的输入（确保两次运行输入一致）
--load-inputs	   复用之前保存的输入
--load-outputs	   加载 FP32 的输出作为参考
--atol / --rtol	设置绝对/相对误差容忍度（FP16 理论误差 ~1e-3）

 如果比较通过 → 模型本身支持 FP16

❌ 如果失败 → 模型存在数值敏感层（如 softmax、layer norm）

步骤 4：检查中间结果是否溢出：
polygraphy run --onnxrt identity_fp16.onnx --onnx-outputs mark all --validate
--onnx-outputs mark all：强制输出所有中间节点的值
--validate：自动检查是否有 NaN / Inf
帮助定位具体哪个算子导致数值不稳定
四、工程价值：为什么这个流程至关重要？
场景 1：模型训练未做 FP16 兼容设计；某些模型使用了对精度敏感的操作：
exp(x) 当 x > 10 → FP16 上溢为 Inf；log(sum(exp(x)))（log-sum-exp）在 FP16 下易失稳
通过此流程可提前发现
场景 2：避免误判 TensorRT 问题；曾有用户报告：“TensorRT FP16 结果错！”
实际是模型本身在 ONNX Runtime FP16 下就错，节省大量 debug 时间
场景 3：指导模型修复
若发现某层输出 NaN，可：在训练时加入 FP16 模拟（AMP）；修改模型结构（如加 clamp、改归一化方式）；对特定层禁用 FP16（TensorRT 支持 per-layer precision）

五、常见误区澄清
❌ 误区 1：“FP16 模型 = TensorRT FP16 引擎”
ONNX FP16 模型：仅权重和计算精度降级，无图优化；TensorRT FP16 引擎：包含 kernel 融合、内存优化、算法选择等，两者结果可能不同，但若 ONNX FP16 已错，TensorRT 必错
❌ 误区 2：“只要最终输出 ok，中间 NaN 无所谓”
中间 NaN 可能被后续算子“掩盖”（如 ReLU 把 -Inf 变 0）；但会导致结果不可复现或极端 case 崩溃；必须用 --validate 检查
❌ 误区 3：“所有模型都能无损转 FP16”
事实：很多 NLP 模型（尤其是含 softmax 的）在长序列下 FP16 会失稳
解决方案：使用 --tf32（Ampere+ GPU）作为折中
对 attention scores 保持 FP32（TensorRT 支持 mixed precision）

六、扩展：如何修复 FP16 不兼容问题？
若此流程发现精度问题，可采取：
1. 训练阶段修复：使用 PyTorch AMP（Automatic Mixed Precision）训练
显式对敏感层（如 loss computation）使用 FP32
2. 导出 ONNX 时修复：在模型代码中插入 .to(torch.float32) 保护关键计算
3. TensorRT 阶段修复
# 在构建 TensorRT 引擎时，强制某些层用 FP32
config.set_flag(trt.BuilderFlag.FP16)
config.set_precision_for_layer("Layer_123", trt.float32)
 总结：该流程的核心价值
目标	                实现方式	                         意义
隔离精度问题根源	在无优化的 ONNX Runtime 中测试纯 FP16	区分“模型问题” vs “TensorRT 问题”
量化 FP16 误差	   通过 --atol/--rtol 设定容忍度	        判断误差是否在可接受范围
定位数值不稳定点	检查所有中间输出的 NaN/Inf	            精准修复模型缺陷
指导后续优化策略	决定是否全 FP16、混合精度、或回退 FP32	  避免盲目部署

 一句话精髓：“在把模型交给 TensorRT 之前，先问自己：它配得上 FP16 吗？”
这个简单的预检流程，能避免 80% 的 FP16 部署陷阱。
因此，该文档不仅是一个工具使用指南，更是一种 系统性精度调试方法论 的体现。