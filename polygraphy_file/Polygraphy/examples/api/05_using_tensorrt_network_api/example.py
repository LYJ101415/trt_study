"""以下代码是 Polygraphy 中使用 TensorRT 原生 Network API 从零构建网络 的经典示例。
它展示了如何通过 @func.extend 装饰器，将手写的 TensorRT 网络逻辑无缝集成到 Polygraphy 的懒加载（lazy loading）体系中，
从而兼顾底层控制力与高层工程便利性。可用于替换算子，做一些自定义的优化，甚至实现自定义层。"""

"""
This script demonstrates how to use the extend() API covered in example 03
to construct a TensorRT network using the TensorRT Network API.
"""
import numpy as np
import tensorrt as trt # 访问 TensorRT 原生 API（如层类型、数据类型）。
from polygraphy import func # 提供 @func.extend 装饰器。
from polygraphy.backend.trt import CreateNetwork, EngineFromNetwork, TrtRunner

"""定义模型的输入/输出张量名称和形状，便于后续引用和验证。
此处为静态 shape（无 batch 维度），实际应用中通常包含 batch（如 (1, 64, 64)）。
"""
INPUT_NAME = "input"
INPUT_SHAPE = (64, 64)
OUTPUT_NAME = "output"


# Just like in example 03, we can use `extend` to add our own functionality to existing lazy loaders.
# `CreateNetwork` will create an empty network, which we can then populate ourselves.
"""核心：手写 TensorRT 网络（使用 @func.extend）:@func.extend(CreateNetwork())：
CreateNetwork() 返回一个 惰性加载对象（尚未真正创建网络）。@func.extend 将 create_network 函数注册为该加载器的扩展逻辑。
当后续需要网络时（如 EngineFromNetwork(create_network)），Polygraphy 会：调用 CreateNetwork() 内部逻辑 → 创建 (builder, network)
自动传入 (builder, network) 到 create_network 函数;执行用户自定义建图逻辑。参数说明：
builder：trt.Builder 对象（用于高级配置，此处未使用）
network：trt.INetworkDefinition 对象（核心：在此添加层）
关键优势：无需手动管理 IBuilder 生命周期或错误处理。"""
@func.extend(CreateNetwork())
def create_network(builder, network):
    # This network will add 1 to the input tensor.
    """添加输入张量：name：必须与推理时 feed_dict 的键一致（"input"）。shape：静态 shape (64, 64)（若需动态 shape，应设为 (-1, 64) 并启用 EXPLICIT_BATCH）。
    dtype：指定数据类型（trt.float32）。"""
    inp = network.add_input(name=INPUT_NAME, shape=INPUT_SHAPE, dtype=trt.float32)
    """添加常量层：weights：必须为 NumPy 数组（TensorRT 内部拷贝到 GPU）。返回 IConstantLayer，调用 .get_output(0) 获取输出张量（类型 ITensor）。
    此处创建一个全 1 张量，shape 与输入匹配。"""
    ones = network.add_constant(
        shape=INPUT_SHAPE, weights=np.ones(shape=INPUT_SHAPE, dtype=np.float32)
    ).get_output(0)
    """添加逐元素加法层：inp 和 ones 是两个输入张量（必须 shape 兼容）。op=trt.ElementWiseOperation.SUM：指定加法操作。
    返回 IElementWiseLayer，获取其输出张量。"""
    add = network.add_elementwise(
        inp, ones, op=trt.ElementWiseOperation.SUM
    ).get_output(0)
    """设置输出名并标记为网络输出：add.name = "output"：使推理结果可通过 "output" 键访问。
    network.mark_output(add)：强制保留该输出（否则 TensorRT 会因 DCE 优化移除未使用的层）。
    常见错误：忘记 mark_output 导致推理时输出为空！无需返回值！@func.extend 自动将修改后的 (builder, network) 传递给下游。"""
    add.name = OUTPUT_NAME
    network.mark_output(add)

    # Notice that we don't need to return anything - `extend()` takes care of that for us!


def main():
    # After we've constructed the network, we can go back to using regular Polygraphy APIs.
    #
    # NOTE: Since we're using lazy loaders, we provide the `create_network` function as
    # an argument - we do *not* call it ourselves.
    """传递函数对象（非调用结果）：EngineFromNetwork(create_network)：传入 函数本身（create_network 是经 @extend 包装的懒加载器）。
    Polygraphy 在需要时自动执行建图 → 构建引擎。懒加载优势：避免提前初始化 CUDA 上下文，支持组合式工作流。"""
    build_engine = EngineFromNetwork(create_network)

    with TrtRunner(build_engine) as runner:
        """构造输入数据：feed_dict 键必须匹配 INPUT_NAME（"input"）。数据类型必须为 np.float32（与网络定义一致）。"""
        feed_dict = {
            INPUT_NAME: np.random.random_sample(INPUT_SHAPE).astype(np.float32)
        }

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict)

        assert np.array_equal(outputs[OUTPUT_NAME], (feed_dict[INPUT_NAME] + 1))

        print("Inference succeeded!")


if __name__ == "__main__":
    main()

"""
实际工业应用示例
示例 1：带自定义 Plugin 的检测头
@func.extend(CreateNetwork())
def yolov5_head_network(builder, network):
    # 输入: (1, 255, 20, 20) 来自 backbone
    input_tensor = network.add_input("features", trt.float32, (1, 255, 20, 20))
    
    # 注册 YOLO 解码插件（需提前实现 IPluginV2IOExt）
    registry = trt.get_plugin_registry()
    creator = registry.get_plugin_creator("YoloLayer_TRT", "1")
    plugin = creator.create_plugin("yolo_decode", trt.PluginFieldCollection([]))
    
    # 添加插件层
    yolo_layer = network.add_plugin_v2([input_tensor], plugin)
    output_tensor = yolo_layer.get_output(0)
    output_tensor.name = "detections"
    network.mark_output(output_tensor)
用途：部署含 proprietary 后处理的检测模型。

示例 2：动态 Batch 的图像预处理网络
@func.extend(CreateNetwork(
    flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
))
def dynamic_preprocess_network(builder, network):
    # 支持任意 batch size: (-1, 3, H, W)
    input_img = network.add_input("image", trt.float32, (-1, 3, -1, -1))
    
    # 归一化: (img / 255.0 - mean) / std
    scale = network.add_constant((1, 3, 1, 1), np.array([1/255.0]*3, dtype=np.float32)).get_output(0)
    scaled = network.add_elementwise(input_img, scale, trt.ElementWiseOperation.PROD).get_output(0)
    
    mean = network.add_constant((1, 3, 1, 1), np.array([0.485, 0.456, 0.406], dtype=np.float32)).get_output(0)
    centered = network.add_elementwise(scaled, mean, trt.ElementWiseOperation.SUB).get_output(0)
    
    std_inv = network.add_constant((1, 3, 1, 1), np.array([1/0.229, 1/0.224, 1/0.225], dtype=np.float32)).get_output(0)
    normalized = network.add_elementwise(centered, std_inv, trt.ElementWiseOperation.PROD).get_output(0)
    
    normalized.name = "normalized_image"
    network.mark_output(normalized)
注意：需在 CreateConfig 中设置 Optimization Profile 以支持动态 shape。

示例 3：多输入融合网络（如双目视觉）
@func.extend(CreateNetwork())
def stereo_matching_network(builder, network):
    left = network.add_input("left_image", trt.float32, (1, 3, 480, 640))
    right = network.add_input("right_image", trt.float32, (1, 3, 480, 640))
    
    # 特征提取（简化）
    conv_w = np.random.rand(64, 3, 7, 7).astype(np.float32)
    conv_b = np.random.rand(64).astype(np.float32)
    
    left_feat = network.add_convolution_nd(left, 64, (7,7), conv_w, conv_b).get_output(0)
    right_feat = network.add_convolution_nd(right, 64, (7,7), conv_w, conv_b).get_output(0)
    
    # 特征拼接
    concat = network.add_concatenation([left_feat, right_feat])
    concat.axis = 1  # 沿通道维度拼接
    
    output = concat.get_output(0)
    output.name = "stereo_features"
    network.mark_output(output)
用途：自动驾驶中的立体匹配、光流估计等任务。
"""
