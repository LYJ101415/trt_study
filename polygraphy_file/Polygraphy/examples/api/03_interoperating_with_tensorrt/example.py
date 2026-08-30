

"""
This script demonstrates how to use Polygraphy in conjunction with APIs
provided by a backend. Specifically, in this case, we use TensorRT APIs
to print the network name and enable FP16 mode.
"""
import numpy as np
import tensorrt as trt # 直接导入原生 TensorRT API，用于访问底层功能（如 BuilderFlag.FP16）。
from polygraphy import func # 包含 @func.extend 装饰器，用于扩展 Polygraphy 的 lazy loader。
from polygraphy.backend.trt import (
    CreateConfig, # 创建 TensorRT 构建配置（IBuilderConfig）的 lazy loader。
    EngineFromNetwork, # 从网络和配置构建 TensorRT 引擎。
    NetworkFromOnnxPath, # 从 ONNX 文件构建 TensorRT 网络的 lazy loader（延迟加载器）
    TrtRunner, # 封装引擎，提供统一推理接口。
)
# 以上所有 *From* 类都是 callable 对象，调用时才执行实际操作（惰性求值），懒加载。


# TIP: The immediately evaluated functional API makes it very easy to interoperate
# with backends like TensorRT. For details, see example 06 (`examples/api/06_immediate_eval_api`).

"""以上注释提示：Polygraphy 提供两种 API 风格：Lazy API（本例使用）：返回可调用对象，延迟执行；
# Immediate API：立即执行并返回结果（适合脚本式快速验证）。"""

# We can use the `extend` decorator to easily extend lazy loaders provided by Polygraphy
# The parameters our decorated function takes should match the return values of the loader we are extending.
"""以上注释说明：关键原则：被装饰函数的参数必须与原始 loader 的返回值一一对应。"""

# For `NetworkFromOnnxPath`, we can see from the API documentation that it returns a TensorRT
# builder, network and parser. That is what our function will receive.
"""第一部分：使用 @func.extend 修改网络结构:NetworkFromOnnxPath("identity.onnx") 是一个 loader，
调用时返回 (builder, network, parser)。
@func.extend(...) 将这个 loader 包装，使得当它被调用时，先执行 load_network 函数。
参数 builder, network, parser 是原生 TensorRT 对象：
builder: trt.Builder
network: trt.INetworkDefinition
parser: trt.OnnxParser"""
@func.extend(NetworkFromOnnxPath("identity.onnx"))
def load_network(builder, network, parser):
    # Here we can modify the network. For this example, we'll just set the network name.
    """直接操作原生 TensorRT 网络对象！设置网络名称为 "MyIdentity"（可用于日志、调试或工具识别）。
     此处仅为演示，真实场景可做更复杂的修改（见后文应用举例）。本例中仅设置网络的名称。"""
    network.name = "MyIdentity"
    print(f"Network name: {network.name}")

    # Notice that we don't need to return anything - `extend()` takes care of that for us!
    """重要机制：@func.extend 会自动返回原始 loader 的返回值（即 (builder, network, parser)），即使你的函数没有 return。
    你只需就地修改传入的对象即可（Python 中对象是引用传递）。"""

# In case a builder configuration option is missing from Polygraphy, we can easily set it using TensorRT APIs.
# Our function will receive a TensorRT IBuilderConfig since that's what `CreateConfig` returns.
"""第二部分：使用 @func.extend 设置高级构建配置:CreateConfig() 是一个 loader，调用时返回 trt.IBuilderConfig 对象。
被装饰函数 load_config 接收该 config 对象。"""
@func.extend(CreateConfig())
def load_config(config):
    # Polygraphy supports the fp16 flag, but in case it didn't, we could do this:
    """直接调用原生 TensorRT API：config.set_flag(trt.BuilderFlag.FP16)
   即使 Polygraphy 未来未支持某个新 flag（如 BuilderFlag.SPARSE_WEIGHTS），你也能立即使用。
   此处以 FP16 为例（Polygraphy 实际已支持，但演示原理）。
   注意：CreateConfig() 默认不启用 FP16，此处显式开启。"""
    config.set_flag(trt.BuilderFlag.FP16)


def main():
    # Since we have no further need of TensorRT APIs, we can come back to regular Polygraphy.
    #
    # NOTE: Since we're using lazy loaders, we provide the functions as arguments - we do *not* call them ourselves.
    """关键点：传入的是函数对象 load_network 和 load_config，不是调用结果！EngineFromNetwork 内部会：
    调用 load_network() → 得到 (builder, network, parser)（已被修改）；调用 load_config() → 得到 config（已启用 FP16）；
    使用这些对象构建引擎。完美融合：高层抽象（Polygraphy） + 底层控制（TensorRT）"""
    build_engine = EngineFromNetwork(load_network, config=load_config)

    with TrtRunner(build_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer({"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()

"""
实际应用示例：为 YOLO 模型开启 FP16 并修改输入尺寸;在实际部署中，我们通常不会去修改网络名称，而是利用这个机制来做混合精度控制、插入自定义插件或调整动态 Shape 配置。
以下是一个实战场景：使用 best.onnx（目标检测模型），通过 @func.extend 强制开启 FP16，并设置最大 Workspace 为 2GB。

import tensorrt as trt
from polygraphy import func
from polygraphy.backend.trt import CreateConfig, EngineFromNetwork, NetworkFromOnnxPath, TrtRunner

# 1. 扩展网络加载：这里可以做一些图优化或节点替换
@func.extend(NetworkFromOnnxPath("./best.onnx"))
def load_yolo_network(builder, network, parser):
    # 例如：打印模型包含的层数，或者在特定节点后插入调试用的 Identity 层
    print(f"YOLO Network loaded. Total layers: {network.num_layers}")
    network.name = "YOLOv8_FP16_Engine"

# 2. 扩展构建配置：精细控制 TensorRT 的构建行为
@func.extend(CreateConfig())
def configure_yolo_trt(config):
    # 开启 FP16 半精度推理
    config.set_flag(trt.BuilderFlag.FP16)
    
    # 设置最大工作空间为 2GB (实际部署大模型时非常重要)
    config.max_workspace_size = 2 * (1 << 30) 
    
    # 如果模型有 INT8 需求，还可以在这里配置 Calibrator
    # config.set_flag(trt.BuilderFlag.INT8)
    # config.int8_calibrator = MyCustomCalibrator(...)

# 3. 构建引擎并执行推理
build_engine = EngineFromNetwork(load_yolo_network, config=configure_yolo_trt)

with TrtRunner(build_engine) as runner:
    # 假设 YOLO 模型的输入是 (1, 3, 640, 640)
    import numpy as np
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
    
    outputs = runner.infer({"images": dummy_input})
    
    # 获取输出并进行后处理（如 NMS）
    predictions = outputs["output0"]
    print(f"Raw output shape: {predictions.shape}")

补充说明和示例：
load_yolo_network 负责模型结构层面的修改（如打印结构、插入/替换节点、修改输入形状等）。
configure_yolo_trt 负责构建配置层面的设置（如开启 FP16、设置 Workspace、加载插件库等）。
若需求是（修改输入为 1028x1028 以及 在第5层卷积后插入插件），具体实现方式如下：
1. 修改模型输入为 1028x1028：在 load_yolo_network 函数中，可以直接修改网络输入张量（Input Tensor）的 shape：
@func.extend(NetworkFromOnnxPath("./best.onnx"))
def load_yolo_network(builder, network, parser):
    # 获取网络的第一个输入张量（通常名为 'images' 或 'input'）
    input_tensor = network.get_input(0)
    
    # 修改输入形状为 (Batch=1, Channels=3, Height=1028, Width=1028)
    # 注意：TRT 的 shape 必须是 tuple 格式
    input_tensor.shape = (1, 3, 1028, 1028)
    
    print(f"已将模型输入修改为: {input_tensor.name}, Shape: {input_tensor.shape}")

2. 在第5层卷积后插入插件；要在特定层后插入自定义插件，需要遍历网络找到该层，获取其输出张量，然后利用 TensorRT 的 add_plugin_v2 API 将插件接入计算图：
@func.extend(NetworkFromOnnxPath("./best.onnx"))
def load_yolo_network(builder, network, parser):
    # 1. 找到第5层卷积（假设索引为4，或者通过名称查找）
    target_layer = network.get_layer(4)  # 索引从0开始，第5层是index 4
    if target_layer.type != trt.LayerType.CONVOLUTION:
        raise ValueError(f"第5层不是卷积层，而是 {target_layer.type}")
        
    # 2. 获取该卷积层的输出张量
    conv_output = target_layer.get_output(0)
    
    # 3. 创建你的自定义插件实例
    # 假设你已经通过 C++ 编写并编译了插件，并在此处通过 Python 绑定实例化
    my_plugin = create_my_custom_plugin()  # 替换为你的插件创建函数
    
    # 4. 将插件添加到网络中
    # inputs 是一个列表，表示插件的输入；这里接入前面卷积的输出
    plugin_layer = network.add_plugin_v2(inputs=[conv_output], plugin=my_plugin)
    plugin_layer.name = "MyCustomPlugin_After_Conv5"
    
    # 5. 关键：如果这个插件是网络的最后一个节点，必须将其标记为网络输出
    # 否则 TensorRT 可能会在优化时将其作为死代码剔除
    # network.mark_output(plugin_layer.get_output(0))
    
    print(f"成功在 {target_layer.name} 后插入插件: {plugin_layer.name}")

3. 加载插件库（配合 configure_yolo_trt）；如果插入的是外部编译好的 .so 插件库，还需要在配置阶段将其加载：
@func.extend(CreateConfig())
def configure_yolo_trt(config):
    config.set_flag(trt.BuilderFlag.FP16)
    
    # 如果你的插件需要加载外部动态链接库，可以在这里加载
    # 或者在 EngineFromNetwork 外层使用 Polygraphy 的 LoadPlugins
    import ctypes
    ctypes.CDLL("./path/to/your/plugin.so")  # 替换为你的插件库路径

核心注意事项：
形状对齐：插入插件后，必须确保插件的输入/输出 Shape 与前后层完全匹配，否则 TensorRT 在构建引擎时会报 Shape 不兼容的错误。
图拓扑更新：使用 add_plugin_v2 后，TensorRT 会自动处理计算图的依赖关系，原卷积层的输出会自动重定向到插件的输入。
更复杂的图修改：如果你需要做更复杂的图重构（比如替换整个子图、批量修改节点），建议结合 ONNX GraphSurgeon 在保存 ONNX 阶段就完成修改，
然后再交给 Polygraphy 构建引擎，这样代码会更清晰。
"""
