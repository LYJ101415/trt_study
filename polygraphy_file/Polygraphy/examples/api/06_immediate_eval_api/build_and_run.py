"""以下代码使用 Polygraphy 的立即求值函数式 API（Immediately Evaluated Functional API），
从 ONNX 模型构建 TensorRT 引擎、修改网络结构、启用 FP16 精度、保存引擎并执行推理。
总结：该脚本的核心价值
---------------------------------------------------------
特性	             说明
立即求值	    无需理解 Lazy Loader 的 () 语义，代码线性易读
直接修改网络	无需 @func.extend，直接操作 network 对象
完整生命周期	演示了 构建 → 修改 → 保存 → 加载 → 推理 全流程
生产就绪	    保存的 .engine 文件可直接用于 C++/Python 部署
------------------------------------------------------------
最佳实践建议：
开发/调试阶段：使用此 Functional API 快速迭代
部署阶段：将构建逻辑封装为脚本，预生成 .engine 文件
复杂流水线：混合使用 Lazy Loader（部署） + Functional API（调试）"""

"""
This script uses Polygraphy's immediately evaluated functional APIs
to load an ONNX model, convert it into a TensorRT network, add an identity
layer to the end of it, build an engine with FP16 mode enabled,
save the engine, and finally run inference.
"""
import numpy as np
from polygraphy.backend.trt import (
    TrtRunner,
    create_config, # 创建 IBuilderConfig 对象，用于配置引擎构建选项（如 FP16、INT8、工作空间大小等）
    engine_from_network, # 从 (builder, network) 元组和配置构建 已序列化的 TensorRT 引擎（ICudaEngine）
    network_from_onnx_path, # 立即加载 ONNX 文件，返回 (builder, network, parser) 三元组（非懒加载）函数
    save_engine, # 将 ICudaEngine 序列化并保存为 .engine 文件，便于后续部署
)


def main():
    # In Polygraphy, loaders and runners take ownership of objects if they are provided
    # via the return values of callables. For example, we don't need to worry about object
    # lifetimes when we use lazy loaders.
    #
    # Since we are immediately evaluating, we take ownership of objects, and are responsible for freeing them.
    """步骤 1：加载 ONNX 模型 → 获取 TensorRT 网络:network_from_onnx_path：立即读取 "identity.onnx" 文件。内部自动创建 trt.Builder、INetworkDefinition 和 OnnxParser。
    返回三元组：builder: trt.Builder 对象（用于构建引擎）;network: trt.INetworkDefinition 对象（可直接修改！）
    parser: trt.OnnxParser 对象（可用于调试解析错误）
    内存管理注意：文档注释强调 —— “我们拥有这些对象的所有权，需自行负责释放”。
    实际上，由于 Python 的引用计数机制，只要不手动持有 builder/network 的长期引用，它们会在函数结束时自动析构。但在 C++ 或复杂场景中需谨慎。"""
    builder, network, parser = network_from_onnx_path("identity.onnx")

    # Extend the network with an identity layer (purely for the sake of example).
    #   Note that unlike with lazy loaders, we don't need to do anything special to modify the network.
    #   If we were using lazy loaders, we would need to use `func.extend()` as described
    #   in example 03 and example 05.
    """步骤 2：修改网络结构（添加 Identity 层）:
    1.获取原网络的第一个输出张量（假设 identity.onnx 只有一个输出）。"""
    prev_output = network.get_output(0)
    """2.取消标记原输出：TensorRT 默认只保留被 mark_output 的张量。若不取消，原输出仍存在，可能导致冗余计算或命名冲突。"""
    network.unmark_output(prev_output)
    """3.添加 Identity 层：add_identity(input) 是 TensorRT 的恒等映射层（输出 = 输入）。
    返回 IIdentityLayer，调用 .get_output(0) 获取其输出张量（类型 ITensor）。"""
    output = network.add_identity(prev_output).get_output(0)
    """4.重命名并标记为新输出：设置张量名为 "output"（推理时通过此键获取结果）。mark_output 确保该张量不会被优化器删除。
    为什么这么做？示例中仅为演示 “如何在 Functional API 中直接修改网络”。实际用途包括：替换不支持的算子（如将 GridSample 替换为 Plugin;
    添加后处理层（如 NMS、ArgMax）;修改输入/输出名以匹配部署规范"""
    output.name = "output"
    network.mark_output(output)

    # Create a TensorRT IBuilderConfig so that we can build the engine with FP16 enabled.
    """步骤 3：创建引擎配置（启用 FP16）:create_config：创建 IBuilderConfig 对象。fp16=True 启用 半精度（FP16）推理（需 GPU 支持，如 Volta 架构以上）。
    其他常用参数：tf32=True, int8=True, workspace_size=1<<30（1GB）等。优势：比原生 TensorRT 少写 5 行样板代码！"""
    config = create_config(builder, network, fp16=True)
    """步骤 4：构建并保存 TensorRT 引擎:engine_from_network：输入：(builder, network) 元组 + config
   输出：已构建完成的 ICudaEngine 对象（可直接用于推理）内部调用 builder.build_serialized_network(network, config)"""
    engine = engine_from_network((builder, network), config)

    # To reuse the engine elsewhere, we can serialize it and save it to a file.
    """序列化并保存引擎：将 engine 转为字节流并写入 "identity.engine";后续可通过 engine_from_bytes() 或 TrtRunner("identity.engine") 加载
    工业价值：预编译引擎可避免在线构建（耗时 10s~10min），实现 “一次构建，多次部署”。"""
    save_engine(engine, path="identity.engine")

    """步骤 5：执行推理并验证结果:"""
    with TrtRunner(engine) as runner:
        inp_data = np.ones((1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["output"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()

"""
实际应用示例（超越 Identity）:
示例 1：替换不支持的算子（如 GroupNorm）
# 假设 ONNX 中有 GroupNorm，TensorRT 不支持
builder, network, parser = network_from_onnx_path("model.onnx")

for i in range(network.num_layers):
    layer = network.get_layer(i)
    if "groupnorm" in layer.name.lower():
        inp = layer.get_input(0)
        # 用 InstanceNorm + Reshape 模拟（简化版）
        shuffle = network.add_shuffle(inp)
        shuffle.reshape_dims = (0, 32, -1)  # 分成32组
        inst_norm = network.add_normalization(shuffle.get_output(0), ...)
        # ...恢复 shape 并替换输出
        network.mark_output(new_output)

config = create_config(builder, network, fp16=True)
engine = engine_from_network((builder, network), config)
save_engine(engine, "fixed_model.engine")

示例 2：添加后处理（目标检测 NMS）
# 在 YOLO 输出后添加 NMS Plugin
builder, network, parser = network_from_onnx_path("yolov5.onnx")

boxes = network.get_output(0)   # (num_boxes, 4)
scores = network.get_output(1)  # (num_boxes, num_classes)

# 注册 NMS Plugin（需提前编译 .so）
registry = trt.get_plugin_registry()
nms_plugin = registry.get_plugin_creator("EfficientNMS_TRT", "1").create_plugin(...)

nms_layer = network.add_plugin_v2([boxes, scores], nms_plugin)
network.mark_output(nms_layer.get_output(0))
# 继续构建引擎...

示例 3：动态 Shape 配置
builder, network, parser = network_from_onnx_path(
    "model.onnx",
    flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
)

# 创建 Optimization Profile
profile = builder.create_optimization_profile()
profile.set_shape("input", min=(1,3,224,224), opt=(4,3,224,224), max=(8,3,224,224))
config = create_config(builder, network, profiles=[profile])
"""
