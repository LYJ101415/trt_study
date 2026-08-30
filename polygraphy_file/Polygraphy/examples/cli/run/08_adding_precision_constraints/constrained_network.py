"""以下代码是一个 Polygraphy 网络加载脚本（Network Loader Script），它通过 扩展 ONNX 解析流程，在 TensorRT 网络构建阶段直接注入精度约束，
确保关键层（Add/Sub）以 FP32 运行，从而避免 FP16 量化导致的数值溢出问题。"""

"""
Parses an ONNX model, then adds precision constraints so specific layers run in FP32.
"""

from polygraphy import func
from polygraphy.backend.trt import NetworkFromOnnxPath # 导入 ONNX → TRT 网络解析器（返回可调用对象）
import tensorrt as trt

# Load the model, which implements the following network:
#
# x -> MatMul (I_rot90) -> Add (FP16_MAX) -> Sub (FP16_MAX) -> MatMul (I_rot90) -> out
#
# Without constraining the subgraph (Add -> Sub) to FP32, this model may
# produce incorrect results when run with FP16 optimziations enabled.
"""NetworkFromOnnxPath 是一个 可调用对象（Callable），当被调用时会：
创建 TRT Builder/Network/Parser;解析指定 ONNX 文件;返回 (network, parser) 元组
此处 并未立即解析模型，只是创建了解析器实例"""
parse_network_from_onnx = NetworkFromOnnxPath("./needs_constraints.onnx")


@func.extend(parse_network_from_onnx)
def load_network(builder, network, parser):
    """The below function traverses the parsed network and constrains precisions
    for specific layers to FP32.

    See examples/cli/run/04_defining_a_tensorrt_network_or_config_manually
    for more examples using network scripts in Polygraphy.

    @func.extend(target) 是 Polygraphy 的函数扩展装饰器;它将 load_network 附加到 parse_network_from_onnx 的执行流程末尾;
    当 parse_network_from_onnx() 被调用时，实际执行顺序为：原始 ONNX 解析逻辑（由 NetworkFromOnnxPath 实现）。
    自定义的 load_network(builder, network, parser) 函数;参数自动传递：builder: trt.Builder;
    network: trt.INetworkDefinition（已解析的网络）;parser: trt.OnnxParser。
    为什么需要 builder, network, parser 参数？builder：可用于创建新层（本例未使用）；
    network：核心操作对象，直接修改其层属性；parser：可用于检查解析错误（如 parser.num_errors > 0）
    """
    for layer in network:
        # Set computation precision for Add and Sub layer to FP32
        """遍历所有层，对名为 "Add" 或 "Sub" 的层：设置 计算精度 为 FP32（即使启用 --fp16）"""
        if layer.name in ("Add", "Sub"):
            layer.precision = trt.float32

        # Set the output precision for the Add layer to FP32.  Without this,
        # the intermediate output data of the Add may be stored as FP16 even
        # though the computation itself is performed in FP32.
        """额外设置 Add 层的输出张量精度为 FP32;原因：防止 TRT 将 FP32 计算结果转回 FP16 存储，导致后续 Sub 层输入仍是损坏数据"""
        if layer.name == "Add":
            layer.set_output_type(0, trt.float32)

"""
三、实际应用示例
场景：使用该脚本验证修复后的模型精度
Step 1: 保存代码为 constrained_network.py
Step 2: 生成黄金输入/输出（FP32 参考）
polygraphy run needs_constraints.onnx --onnxrt --val-range x:[1,2] \
    --save-inputs inputs.json --save-outputs golden_outputs.json
Step 3: 用自定义脚本运行 TRT 引擎并验证
polygraphy run constrained_network.py --trt --fp16 \
    --precision-constraints obey \
    --load-inputs inputs.json --load-outputs golden_outputs.json \
    --check-error-stat median
预期成功输出：
Comparing Output: 'out'
    Runner: trt-runner-...     | Stats: mean=1.500, min=1.000, max=2.000
    Runner: golden             | Stats: mean=1.500, min=1.000, max=2.000
    Absolute Difference | Max: 1.19e-07, Mean: 5.96e-08
 PASSED | All outputs are within tolerance.
若未设置 set_output_type，则会失败：
 FAILED | Absolute difference exceeds tolerance!
    Max absolute error: 65504.0  # = FP16_MAX
 四、工程最佳实践
1. 层名称验证（防错）
# 添加日志确保层存在
target_layers = {"Add", "Sub"}
found_layers = {layer.name for layer in network}
assert target_layers.issubset(found_layers), f"Missing layers: {target_layers - found_layers}"
2. 多输出层安全处理
if layer.name == "Add":
    for i in range(layer.num_outputs):
        layer.set_output_type(i, trt.float32)
3. 结合配置修改（进阶）
@func.extend(parse_network_from_onnx)
def load_network(builder, network, parser):
    # ... 精度约束 ...
    
    # 同时返回自定义 config（需配合 ConfigLoader 使用）
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    return network, config  # 注意：需显式返回
 注意：若函数有 return，必须返回 (network, config) 元组，否则 Polygraphy 无法获取网络。

 五、注意事项
文件路径硬编码风险:
"./needs_constraints.onnx" 应改为相对路径或参数化（生产环境建议用环境变量）

TRT 版本兼容性:layer.precision 和 set_output_type 在 TRT 7+ 支持，旧版本需用 set_must_run_in_int8/float16（已弃用）
调试技巧

临时添加打印语句查看层信息：
print(f"[DEBUG] Layer: {layer.name}, Type: {layer.type}")

 六、总结：这份代码的工程价值
维度	     说明
模式标准化	  展示了 Polygraphy “解析 ONNX + 扩展修改” 的标准范式
精度控制完备	同时设置计算精度和输出精度，避免常见陷阱
部署安全	  通过 obey 约束确保引擎行为符合预期
可扩展性强	   易于添加新约束（如动态 shape、插件集成）

 核心思想：“在模型解析的第一时间施加约束，比事后修补更可靠”
这种模式广泛应用于 自动驾驶感知模型、医疗影像分割 等对数值稳定性要求极高的场景，是 TensorRT 高级部署工程师的必备技能。
"""