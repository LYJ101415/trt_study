"""这段代码是一个 TensorRT 网络后处理脚本（Postprocessing Script），用于在 Polygraphy 解析 ONNX 模型生成 TensorRT 网络后，
对特定层强制施加精度约束（Precision Constraints），以解决 FP16 量化导致的数值溢出或精度损失问题。"""

"""
Postprocessing script to add precision constraints to a TensorRT network.
"""

import tensorrt as trt # 导入 TensorRT Python API，用于访问网络定义和精度类型（如 trt.float32）


def postprocess(network):
    """
    Traverses the parsed network and constrains precisions
    for specific layers to FP32.

    Args:
        network (trt.INetworkDefinition): The network to modify.

    Returns:
        None
    函数名必须为 `postprocess`（Polygraphy CLI 的约定）
    接收一个已解析的 TensorRT 网络对象（通常来自 ONNX 解析）
    该函数无返回值，直接修改传入的 network 对象（in-place）
    """
    for layer in network: # 遍历网络中的每一层（layer 是 trt.ILayer 子类实例）
        # Set computation precision for Add and Sub layer to FP32
        """对名为 "Add" 或 "Sub" 的层：   - 强制其**计算精度**为 FP32;
           - 即使启用了 --fp16，这些层也会用 FP32 执行运算"""
        if layer.name in ("Add", "Sub"):
            layer.precision = trt.float32

        # Set the output precision for the Add layer to FP32.  Without this,
        # the intermediate output data of the Add may be stored as FP16 even
        # though the computation itself is performed in FP32.
        """额外设置 "Add" 层的**输出张量精度**为 FP32
        原因：即使计算用 FP32，TRT 默认可能将结果转回 FP16 存储;这会导致后续 Sub 层输入仍是 FP16，无法解决问题！
        参数说明：  - 0: 输出索引（Add 只有一个输出）  - trt.float32: 目标数据类型
        仅设 precision=FP32 但不设 output_type=FP32 → 计算正确但结果被截断回 FP16！
        为何只对 "Add" 设置输出精度？Add 层：输出会作为 Sub 层的输入;若 Add 输出是 FP16，则 Sub 层即使设为 FP32，
        输入已是损坏的 FP16 数据; Sub 层：其输出是最终模型输出，通常由 TRT 自动匹配，无需显式设置"""
        if layer.name == "Add":
            layer.set_output_type(0, trt.float32)

"""
实际应用示例
场景：修复 needs_constraints.onnx 的 FP16 溢出问题
Step 1: 保存脚本为 add_constraints.py
Step 2: 使用 Polygraphy 运行并验证
# 生成参考输出（FP32 ONNX Runtime）
polygraphy run needs_constraints.onnx --onnxrt --val-range x:[1,2] \
    --save-inputs inputs.json --save-outputs golden_outputs.json

# 用后处理脚本构建 TRT 引擎并对比
polygraphy run needs_constraints.onnx --trt --fp16 \
    --trt-network-postprocess-script add_constraints.py \
    --precision-constraints obey \
    --load-inputs inputs.json --load-outputs golden_outputs.json \
    --check-error-stat median
预期结果：
成功：TRT 输出与 ONNX Runtime 一致（误差 < 1e-5）
失败（若未设 output_type）：Add 输出溢出为 Inf，Sub 无法恢复

 四、扩展应用场景
场景 1：批量约束多类层
# 约束所有 Reduce、Softmax、LayerNorm 相关层
for layer in network:
    if any(kw in layer.name for kw in ["Reduce", "Softmax", "LayerNorm"]):
        layer.precision = trt.float32
        for i in range(layer.num_outputs):
            layer.set_output_type(i, trt.float32)
场景 2：动态精度策略（基于层属性）
for layer in network:
    if isinstance(layer, trt.IReduceLayer) and layer.op == trt.ReduceOperation.AVG:
        # 对 AVG Pooling 强制 FP32（避免小值下溢）
        layer.precision = trt.float32
        layer.set_output_type(0, trt.float32)
场景 3：结合日志调试
print(f"Layer: {layer.name}, Type: {layer.type}, Precision: {layer.precision}")
# 输出示例：
# Layer: Add, Type: ElementWise, Precision: DataType.FLOAT

 五、注意事项与常见陷阱
1. 层名称来源
layer.name 是 TensorRT 解析 ONNX 后生成的名，可能与 ONNX 节点名不同
查看实际层名：
polygraphy inspect model needs_constraints.onnx --trt --show-layer-names
2. 输出索引范围
set_output_type(idx, ...) 的 idx 必须 < layer.num_outputs
多输出层（如 DetectionOutput）需遍历所有输出：
for i in range(layer.num_outputs):
    layer.set_output_type(i, trt.float32)
3. 精度类型匹配
trt.float32 对应 np.float32
不要混用：trt.DataType.HALF ≠ np.float16（虽然等价，但 API 要求用 TRT 类型）
4. obey vs prefer
若使用 --precision-constraints prefer，TRT 可能忽略约束（当无 FP32 实现时）
调试阶段建议用 obey，确保约束生效

 六、总结：这份代码的工程价值
特性	    说明
精准控制	针对性修复数值敏感层，避免全局 FP32 性能损失
解耦设计	约束逻辑独立于模型构建，便于维护
工业标准	NVIDIA 官方推荐的混合精度调试方法
CI 友好	脚本可纳入版本控制，实现自动化精度保障
 黄金法则：“计算精度 + 输出精度” 必须同时设置，否则约束无效！
在实际项目中，这类脚本常用于修复 BatchNorm 方差为零、Softmax 溢出、自定义插件数值不稳定 等问题，是 TensorRT 高级调试的必备技能。
    """
