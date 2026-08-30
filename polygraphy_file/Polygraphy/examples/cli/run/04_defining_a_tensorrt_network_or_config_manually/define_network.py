
"""
Parses an ONNX model, and then extends it with an Identity layer.
"""
from polygraphy import func # 导入 Polygraphy 的 func 模块，用于函数装饰和扩展
from polygraphy.backend.trt import NetworkFromOnnxPath

# 创建一个可调用对象 parse_onnx，它封装了从 "identity.onnx" 构建 TRT network 的逻辑
# 注意：此时并未真正解析 ONNX，只是创建了一个“配方”
parse_onnx = NetworkFromOnnxPath("identity.onnx")


# If we define a function called `load_network`, polygraphy can
# use it directly in place of using a model file.
#
# TIP: If our function isn't called `load_network`, we can explicitly specify
# the name with the model argument, separated by a colon. For example, `define_network.py:my_func`.
""" 定义一个名为 load_network 的函数，并用 @func.extend 装饰
 这个函数将被 Polygraphy CLI 识别为网络定义入口。
 @func.extend(A) 表示：先执行 A()，再把结果作为参数传给被装饰函数
最终生成的新函数签名是 () -> 返回值，而不是原函数签名；等效于手动写：
def load_network():
    builder, network, parser = parse_onnx()  # 先解析 ONNX
    # ... 然后执行下面函数代码的修改逻辑 ...
    return builder, network, parser
"""
@func.extend(parse_onnx)
def load_network(builder, network, parser):
    # NOTE: func.extend() causes the signature of this function to be `() -> (builder, network, parser)`
    # For details on how this works, see examples/api/03_interoperating_with_tensorrt
    """
    此函数的作用：在原始 ONNX 网络基础上添加一个 Identity 层。关键点：
    - 虽然函数签名写的是 (builder, network, parser)，
      但 @func.extend 会自动将其转换为无参函数 () -> (builder, network, parser)
    - 实际调用时，parse_onnx() 会被执行，返回 (builder, network, parser)，
      然后传入此函数进行修改。
    """
    # Append an identity layer to the network
    """获取网络当前的唯一输出张量（假设 identity.onnx 只有一个输出）"""
    prev_output = network.get_output(0)
    """取消标记该输出（因为我们要替换成新的输出）"""
    network.unmark_output(prev_output)
    """添加一个 Identity 层（输入是 prev_output）; Identity 层不改变数据，常用于调试或占位；
    .get_output(0)并获取 Identity 层的输出张量"""
    output = network.add_identity(prev_output).get_output(0)
    """将新输出标记为网络的最终输出"""
    network.mark_output(output)

    """最后函数隐式返回 (builder, network, parser) —— 这是由 @func.extend 自动处理的"""
    # Notice that we don't need to return anything - `extend()` takes care of that for us!

"""
三、实际应用示例
步骤 1：准备 ONNX 模型:假设 identity.onnx 是一个简单模型：
输入：X，shape [1, 2, 3, 3]
输出：Y = X（Identity 操作）
步骤 2：保存上述代码为 define_network.py
步骤 3：用 Polygraphy CLI 运行该脚本
# 使用修改后的网络进行推理（对比 ONNX Runtime）
polygraphy run define_network.py --trt --onnxrt \
    --model-type=trt-network-script \
    --input-shapes X:[1,2,3,3]
预期行为：TRT 引擎会包含 两个 Identity 层（原始 + 新增）
最终输出名可能变为 output（而非原始 ONNX 的输出名）
输出数值应与 ONNX Runtime 完全一致
步骤 4：验证网络结构（可选）
# 生成脚本查看内部结构
polygraphy run define_network.py --trt --gen-script=debug_net.py
# 然后阅读 debug_net.py 中的网络构建逻辑

 四、扩展应用场景（替换 Identity 为实用操作）
场景 1：添加后处理层（如 Sigmoid）
@func.extend(parse_onnx)
def load_network(builder, network, parser):
    prev_output = network.get_output(0)
    network.unmark_output(prev_output)
    
    # 添加 Sigmoid 激活
    sigmoid = network.add_activation(prev_output, trt.ActivationType.SIGMOID)
    network.mark_output(sigmoid.get_output(0))
场景 2：修复 ONNX 导出问题（如 NMS 层缺失）
@func.extend(parse_onnx)
def load_network(builder, network, parser):
    # 假设 ONNX 缺少 NMS，我们手动添加
    boxes = network.get_output(0)   # [N, 4]
    scores = network.get_output(1)  # [N, C]
    
    network.unmark_output(boxes)
    network.unmark_output(scores)
    
    # 添加 EfficientNMS 插件（需注册插件）
    nms = network.add_plugin_v2(
        inputs=[boxes, scores],
        plugin=efficient_nms_plugin
    )
    network.mark_output(nms.get_output(0))  # detection_boxes
    network.mark_output(nms.get_output(1))  # detection_scores
场景 3：强制某层使用 FP16
@func.extend(parse_onnx)
def load_network(builder, network, parser):
    # 遍历所有卷积层，设为 FP16
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.CONVOLUTION:
            layer.precision = trt.float16
            layer.set_output_type(0, trt.float16)

 五、注意事项
输出数量必须匹配:如果原始 ONNX 有 2 个输出，修改后也应有 2 个（或明确知道下游如何处理）,
不要销毁 builder/network,Polygraphy 会自动管理生命周期，不要调用 del builder,动态 shape 处理
若 ONNX 有动态维度，需在 CLI 中指定 --trt-min/opt/max-shapes。
错误调试：若网络修改失败，TRT 会在 mark_output 或 build_engine 时报错

 六、总结
这段代码虽短，却体现了 Polygraphy 的核心设计哲学：“用最小的代码侵入，实现最大的网络控制力，并无缝集成到命令行工作流中。”
 @func.extend：优雅组合解析与修改逻辑
 load_network 约定：打通 Python API 与 CLI 工具
 Identity 示例：展示安全修改网络的标准范式
 
 实际工作中，你只需：用 polygraphy template trt-network model.onnx -o net.py 生成模板
在 load_network 中插入你的自定义逻辑；用 polygraphy run net.py --trt ... 直接验证；
这比手写完整 TRT 推理脚本效率提升 10 倍以上，是 TensorRT 高级用户的必备技能。
"""