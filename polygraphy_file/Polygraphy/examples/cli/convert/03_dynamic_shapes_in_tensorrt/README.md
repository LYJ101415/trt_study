# Working With Models With Dynamic Shapes In TensorRT

## Introduction

In order to use dynamic input shapes with TensorRT, we have to specify a range
(or multiple ranges) of possible shapes when we build the engine.
For details on how this works, refer to
[API example 07](../../../api/07_tensorrt_and_dynamic_shapes/).

When using the CLI, we can specify the per-input minimum, optimum, and maximum
shapes one or more times. If shapes are specified more than
once per input, multiple optimization profiles are created.

## Running The Example

1. Build an engine with 3 separate profiles:

    ```bash
    polygraphy convert dynamic_identity.onnx -o dynamic_identity.engine \
        --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[1,3,28,28] --trt-max-shapes X:[1,3,28,28] \
        --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[4,3,28,28] --trt-max-shapes X:[32,3,28,28] \
        --trt-min-shapes X:[128,3,28,28] --trt-opt-shapes X:[128,3,28,28] --trt-max-shapes X:[128,3,28,28]
    ```

    For models with multiple inputs, simply provide multiple arguments to each `--trt-*-shapes` parameter.
    For example: `--trt-min-shapes input0:[10,10] input1:[10,10] input2:[10,10] ...`

    *TIP: If we want to use only a single profile where min == opt == max, we can leverage the runtime input*
        *shapes option: `--input-shapes` as a conveneint shorthand instead of setting min/opt/max separately.*


2. **[Optional]** Inspect the resulting engine:

    ```bash
    polygraphy inspect model dynamic_identity.engine
    ```


## Further Reading

For more information on using dynamic shapes with TensorRT, see the
[developer guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#work_dynamic_shapes)
===========================================================================
这份文档是 NVIDIA TensorRT 中关于 动态形状（Dynamic Shapes）模型处理 的一个典型示例说明。它解决了深度学习推理中一个核心问题：如何让同一个 TensorRT 引擎高效支持不同尺寸的输入（如 batch size 变化、图像分辨率可变等）。
下面将从 原理、命令解析、工程意义、最佳实践 四个维度进行详细拆解。

一、背景与核心概念（Introduction 解析）
1. 什么是“动态形状”？
在训练阶段，模型通常使用固定输入尺寸（如 [1, 3, 224, 224]）。但在推理时，我们可能希望：
支持 可变 batch size（如 1~32）；支持 不同分辨率图像（如 28×28 到 512×512）；处理序列长度不一的 NLP 输入；→ 这些需求要求模型能处理 运行时才知道的输入形状，即动态形状（Dynamic Shapes）。

2. TensorRT 如何支持动态形状？
TensorRT 不能在运行时任意改变网络结构，因此它采用 “预定义形状范围 + 优化配置文件（Optimization Profiles）” 机制：构建引擎时，必须为每个输入张量指定：
min：最小可能形状；opt：最常用/性能最优的形状（用于 kernel 选择和内存规划）；
max：最大可能形状。TensorRT 基于这些信息 预先生成多个执行计划（execution plans），每个对应一个 Optimization Profile。
推理时，用户需先 激活某个 profile，再提供符合该 profile 范围的实际输入。
 关键点：动态 ≠ 任意；而是“在预定义范围内动态”。

二、命令详解：构建多 Profile 引擎：
polygraphy convert dynamic_identity.onnx -o dynamic_identity.engine \
    --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[1,3,28,28] --trt-max-shapes X:[1,3,28,28] \
    --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[4,3,28,28] --trt-max-shapes X:[32,3,28,28] \
    --trt-min-shapes X:[128,3,28,28] --trt-opt-shapes X:[128,3,28,28] --trt-max-shapes X:[128,3,28,28]
逐行解析：第一组（Profile 0）：
--trt-min-shapes X:[1,3,28,28]
--trt-opt-shapes X:[1,3,28,28]
--trt-max-shapes X:[1,3,28,28]
表示：只支持 batch=1；min = opt = max → 单一固定形状；适用于 低延迟单样本推理场景
第二组（Profile 1）：
--trt-min-shapes X:[1,3,28,28]
--trt-opt-shapes X:[4,3,28,28]   # ← 注意！opt 是 4
--trt-max-shapes X:[32,3,28,28]
支持 batch size 1 ~ 32，但 优化目标是 batch=4（TensorRT 会针对此形状选择最快 kernel、分配最优内存），适用于 吞吐优先的批量推理（如服务器端）
第三组（Profile 2）：
--trt-min-shapes X:[128,3,28,28]
--trt-opt-shapes X:[128,3,28,28]
--trt-max-shapes X:[128,3,28,28]
支持 大 batch=128，可能用于 离线大数据处理
 每连续出现一组 --trt-min/opt/max-shapes，就创建一个 新的 Optimization Profile。

本例共创建 3 个 profiles，索引为 0、1、2。

多输入模型示例（文档提示）
--trt-min-shapes input0:[10,10] input1:[5,20] \
--trt-opt-shapes input0:[20,20] input1:[10,40] \
--trt-max-shapes input0:[100,100] input1:[50,200]
每个 --trt-*-shapes 参数后可跟 多个输入名+形状对
所有输入必须在同一 profile 中同时指定 min/opt/max
小技巧：固定形状的简写：如果模型 不需要动态形状（即 min=opt=max），可用：
polygraphy convert model.onnx -o model.engine --input-shapes X:[1,3,224,224]
等价于：
--trt-min-shapes X:[1,3,224,224] \
--trt-opt-shapes X:[1,3,224,224] \
--trt-max-shapes X:[1,3,224,224]

三、为什么需要多个 Profiles？单一 Profile 的局限性；假设只定义一个 profile：min=[1,...], opt=[16,...], max=[32,...]。
当输入 batch=1 时：内存按 max=32 分配 → 浪费显存，kernel 为 batch=16 优化 → batch=1 性能不佳；当输入 batch=32 时：可能因内存碎片导致 无法达到理论吞吐。
多 Profile 的优势
Profile	                 适用场景	              优势
Profile 0 (batch=1)	  实时交互、边缘设备	     最低延迟、最小内存
Profile 1 (batch=1~32, opt=4)	通用服务器推理	 平衡吞吐与资源
Profile 2 (batch=128)	  离线批处理	         最大吞吐
→ 一个引擎覆盖全场景，无需维护多个 .engine 文件！

四、推理时如何使用？（虽未在文档中，但至关重要）
构建好引擎后，推理代码必须显式选择 profile：Python API 示例：
import tensorrt as trt
import pycuda.driver as cuda

# 加载引擎
with open("dynamic_identity.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 激活 Profile 1（支持 batch 1~32）
context.set_optimization_profile_async(1, stream_handle)

# 设置实际输入形状（必须在 profile 1 的 [1,32] 范围内）
context.set_binding_shape(0, [8, 3, 28, 28])  # batch=8

# 分配内存、推理...
 如果不设置 profile 或形状超出范围，会报错！

五、工程最佳实践
1. Profile 设计原则
不要过度细分：每个 profile 增加引擎大小和构建时间；覆盖真实使用分布：通过日志分析常见 batch size；opt 形状应代表 80%+ 场景：确保主流请求性能最优
2. 避免“万能 profile”： 错误做法：
--trt-min-shapes X:[1,...] --trt-opt-shapes X:[16,...] --trt-max-shapes X:[1024,...]
→ 导致：显存按 1024 分配，小 batch 浪费严重；kernel 为 16 优化，1 和 1024 都次优
 正确做法：分段覆盖
Profile 0: [1,1,1]
Profile 1: [2,8,32]
Profile 2: [64,128,256]
3. 与 Deterministic 构建结合
动态形状引擎同样受 kernel 计时影响；应在锁频环境下构建，并保存 timing cache
否则不同构建的 profile 可能选择不同 kernel
六、查看引擎信息（Inspect）
polygraphy inspect model dynamic_identity.engine
输出会包含：
Optimization Profiles:
    Profile 0:
        X: min=[1,3,28,28], opt=[1,3,28,28], max=[1,3,28,28]
    Profile 1:
        X: min=[1,3,28,28], opt=[4,3,28,28], max=[32,3,28,28]
    ...
帮助验证 profile 是否按预期创建。

✅ 总结
关键点	                                     说明
动态形状 ≠ 任意形状	              必须在构建时预定义 min/opt/max 范围
Optimization Profile	        每组 min/opt/max 定义一个 profile，引擎可含多个
opt 形状最关键	                  决定 kernel 选择和内存布局，应设为最常用形状
推理时需显式激活 profile	      否则无法使用动态功能
多 profile vs 单 profile	    多 profile 提供更优的资源/性能平衡
与 timing cache 兼容	        动态引擎同样需要 deterministic 构建保障一致性

 一句话精髓：“用多个 Optimization Profiles，让一个 TensorRT 引擎智能适配不同输入规模，兼顾延迟、吞吐与显存效率。”；这种机制是 TensorRT 在生产环境中支持灵活推理需求的核心能力之一，广泛应用于自动驾驶、视频分析、大模型推理等场景。