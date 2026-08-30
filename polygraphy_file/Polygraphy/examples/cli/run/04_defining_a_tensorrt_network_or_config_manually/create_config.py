
"""
Creates a TensorRT builder configuration and enables FP16 tactics.
"""
import tensorrt as trt # 导入 TensorRT Python API，用于访问 BuilderFlag 等常量
from polygraphy import func
# 导入 CreateConfig —— 这是一个可调用对象，用于创建默认的 TensorRT IBuilderConfig
from polygraphy.backend.trt import CreateConfig


# If we define a function called `load_config`, polygraphy can use it to
# create the builder configuration.
#
# TIP: If our function isn't called `load_config`, we can explicitly specify
# the name with the script argument, separated by a colon. For example: `create_config.py:my_func`.
"""CreateConfig() 是什么？CreateConfig 是 Polygraphy 提供的一个 “可调用工厂对象”
调用 CreateConfig()()（即 CreateConfig().__call__()）会：
创建 trt.Builder;创建 trt.IBuilderConfig;返回 (builder, network, config) —— 但在此上下文中，我们只关心 config
 在 @func.extend(CreateConfig()) 中，CreateConfig() 被实例化一次，作为“前置步骤”。"""
@func.extend(CreateConfig())
def load_config(config):
    # NOTE: func.extend() causes the signature of this function to be `(builder, network) -> config`
    # For details on how this works, see examples/api/03_interoperating_with_tensorrt
    """
    此函数的作用：在默认的 TensorRT 构建配置基础上启用 FP16 模式。关键点：
    - 虽然函数签名写的是 (config)，但 @func.extend 会将其“重映射”为
      接受 (builder, network) 作为输入，并返回修改后的 config。
    - 函数内部无需 return —— @func.extend 会自动返回最终的 config 对象。
    启用 FP16 精度模式（允许层使用 float16 计算）
     这会显著提升推理速度（尤其在 NVIDIA GPU 上），并减少显存占用
    """
    config.set_flag(trt.BuilderFlag.FP16)

    """注意：这里没有 return 语句！因为 @func.extend 会自动将传入的 config 对象（已就地修改）作为返回值"""
    # Notice that we don't need to return anything - `extend()` takes care of that for us!

"""
三、实际应用示例
步骤 1：保存代码为 create_config.py
步骤 2：准备 ONNX 模型（如 resnet50.onnx）
步骤 3：使用 CLI 运行（对比 FP32 vs FP16）

# 方式 1：直接运行，使用自定义 FP16 配置
polygraphy run resnet50.onnx --trt \
    --trt-config-script=create_config.py \
    --onnxrt \  # 同时用 ONNX Runtime 作为参考
    --input-shapes input:[1,3,224,224]

# 方式 2：仅构建 FP16 引擎（不运行推理）
polygraphy convert resnet50.onnx --trt \
    --trt-config-script=create_config.py \
    -o resnet50_fp16.engine
步骤 4：验证结果
输出应显示 FP16 引擎构建成功;若与 ONNX Runtime 对比，误差应在合理范围（通常 atol=1e-2 可接受）

 四、常见扩展场景（替换 set_flag(FP16)）
场景 1：启用 INT8 + 设置校准器
@func.extend(CreateConfig())
def load_config(config):
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = MyCalibrator()  # 自定义校准器
场景 2：设置 Workspace 大小
@func.extend(CreateConfig())
def load_config(config):
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
场景 3：配置动态 Shape（Optimization Profile）
@func.extend(CreateConfig())
def load_config(config, builder, network):  # 注意：需额外参数
    # 创建 profile
    profile = builder.create_optimization_profile()
    profile.set_shape("input", 
                      min=[1, 3, 224, 224],
                      opt=[4, 3, 224, 224],
                      max=[8, 3, 224, 224])
    config.add_optimization_profile(profile)

 注意：若需 builder/network，函数签名需为 load_config(config, builder, network)

场景 4：启用 Timing Cache（加速重复构建）
@func.extend(CreateConfig())
def load_config(config):
    cache = trt.TimingCache(None)
    config.set_timing_cache(cache, ignore_mismatch=False)

 五、注意事项
无需 return;@func.extend 会自动返回 config，不要写 return config（虽然写了也不报错，但冗余）
就地修改;所有操作都是对传入的 config 对象就地修改（in-place）
CLI 参数优先级;如果 CLI 中同时指定了 --fp16 和 --trt-config-script，脚本中的设置会覆盖 CLI 参数
错误调试;若配置无效（如硬件不支持 FP16），TRT 会在构建引擎时报错

 六、总结
这段代码虽短，却是 TensorRT 高级部署的基石：

特性	    说明
简洁性	   仅需 1 行代码启用 FP16
可组合性	@func.extend 自动处理前置/后置逻辑
CLI 无缝集成	无需改写推理脚本，直接用于 run/convert
可扩展性	轻松扩展为 INT8、动态 shape、性能优化等

 最佳实践工作流：
1. 用 polygraphy template trt-config --fp16 -o config.py 生成模板
2. 根据需求添加更多配置（如 workspace、profile）
3. 用 polygraphy run ... --trt-config-script=config.py 验证
4. 用 polygraphy convert ... 构建最终部署引擎
这种模式让你既能享受 命令行工具的便捷性，又能实现 Python API 的完全控制力，是工业级 TensorRT 部署的标准做法。
"""