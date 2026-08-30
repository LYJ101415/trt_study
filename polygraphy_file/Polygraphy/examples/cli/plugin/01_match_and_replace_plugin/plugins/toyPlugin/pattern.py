"""这段 Python 代码是 Polygraphy 插件系统中用于定义子图匹配模式（Graph Pattern）和提取匹配信息的核心逻辑，
专为将 ONNX 模型中的特定子图自动替换为自定义 TensorRT 插件而设计。"""

from polygraphy import mod # 导入 Polygraphy 的模块懒加载工具
gs = mod.lazy_import("onnx_graphsurgeon>=0.5.0") # 懒加载 onnx-graphsurgeon（ONNX 图操作库），要求版本 >=0.5.0
    # 使用 lazy_import 避免在未使用时引入依赖,为什么用 lazy_import？Polygraphy 是一个工具集，用户可能只用 CLI 而不用 API。懒加载避免不必要的依赖安装。
from typing import List,Dict # 类型提示：提高代码可读性和 IDE 支持

def get_plugin_pattern():
    """
    定义插件要匹配的 ONNX 子图拓扑结构。目标子图结构：
    Toy plugin pattern:
        A     B                ← 两个独立输入
        \   /
          C, attrs['x'] < 2.0  ← 中间节点，带属性约束: attrs['x'] < 2.0
        /   \
        D     E                ← 两个输出

    以下代码关键点：pattern.variable() 表示“任意输入”，不绑定具体来源;
    check_func 实现动态属性过滤，这是高级匹配的关键;
    节点名（如 "Anode"）是模式内部标识符，用于后续属性提取
    """
    pattern = gs.GraphPattern()  # 创建一个新的图模式对象（用于描述子图结构）
    # 定义两个输入变量（占位符，不绑定具体算子）
    in_0 = pattern.variable() # 第一个输入张量
    in_1 = pattern.variable() # 第二个输入张量
    # 添加节点 A: OpType="A", 输入=in_0, 输出=a_out
    a_out = pattern.add("Anode", "A", inputs=[in_0])
    # 添加节点 B: OpType="B", 输入=in_1, 输出=b_out
    b_out = pattern.add("Bnode", "B", inputs=[in_1])
    # 定义约束函数：仅当节点 C 的属性 x < 2.0 时才匹配
    check_function = lambda node : node.attrs["x"] < 2.0
    # 添加节点 C: OpType="C", 输入=[a_out, b_out], 并附加约束
    c_out = pattern.add("Cnode", "C", inputs=[a_out, b_out], check_func=check_function)
    # 添加节点 D: OpType="D", 输入=c_out
    d_out = pattern.add("Dnode", "D", inputs=[c_out])
    # 添加节点 E: OpType="E", 输入=c_out
    e_out = pattern.add("Enode", "E", inputs=[c_out])
    # 设置整个子图的输出张量（即 D 和 E 的输出）
    pattern.set_output_tensors([d_out, e_out])

    return pattern

def get_matching_subgraphs(graph) -> List[Dict[str,str]]:
    """
    在给定 ONNX 图中查找所有匹配上述模式的子图，并返回每个匹配子图的输入/输出/属性信息。
    返回格式供 Polygraphy plugin replace 使用。
    重要细节：m.get("Cnode")：通过模式中的节点名获取实际 ONNX 节点，
    attrs 字段将传递给最终的插件节点，输入/输出张量名必须与 ONNX 模型中的真实名称一致
    """
    gp = get_plugin_pattern() # 获取预定义的图模式
    matches = gp.match_all(graph) # 在 graph 中查找所有匹配实例
    ans = []  # 存储所有匹配结果
    for m in matches:
        # save the input and output tensor names of the matching subgraph(s)
        # 提取匹配子图的所有输入张量名（去重）
        input_tensors = list(set([ip_tensor.name for ip_tensor in m.inputs]))
        # 提取匹配子图的所有输出张量名（去重）
        output_tensors = list(set([op_tensor.name for op_tensor in m.outputs]))
        # 从匹配的 C 节点中提取属性 'x'，并计算插件属性
        # 注意：这里做了变换：ToyX = int(x) * 2
        attrs = {"ToyX": int(m.get("Cnode").attrs["x"]) * 2}
        # 构造标准替换描述字典
        ioa = {
            'inputs':input_tensors, # 子图输入张量名列表
            'outputs':output_tensors, # 子图输出张量名列表
            'attributes':attrs  # 插件所需属性
        }
        ans.append(ioa)
    return ans

def get_plugin_metadata() -> Dict[str,str]:
    """
    返回插件的元数据，用于生成最终的 ONNX 节点。
    命名规范：'op' 必须与你在 TensorRT 中注册插件时使用的 OpType 完全一致；例如：
    若 C++ 中 REGISTER_TENSORRT_PLUGIN(CustomToyPluginCreator);，则此处必须为 "CustomToyPlugin"
    """
    return {'name':'toyPlugin', # 插件逻辑名（用于 config.yaml）
            'op':'CustomToyPlugin', # ONNX 节点的 op_type（必须与 TRT 插件注册名一致）
            }

"""
真实使用示例：端到端流程
假设你已开发了一个名为 CustomToyPlugin 的 TensorRT 插件，并希望将其集成到 ONNX 模型中。
步骤 1：准备插件目录结构
./plugins/
└── toyPlugin/
    ├── __init__.py          # 空文件
    └── pattern.py           # ← 将上述代码保存为此文件
 文件名必须为 pattern.py，且包含这三个函数。

步骤 2：创建测试 ONNX 模型 (test_model.onnx);用以下 Python 脚本生成符合模式的模型：
# create_test_model.py
import onnx
from onnx import helper, TensorProto

# 创建节点
A = helper.make_node("A", ["input0"], ["a_out"])
B = helper.make_node("B", ["input1"], ["b_out"])
C = helper.make_node("C", ["a_out", "b_out"], ["c_out"], x=1.5)  # x=1.5 < 2.0 → 会匹配
D = helper.make_node("D", ["c_out"], ["output0"])
E = helper.make_node("E", ["c_out"], ["output1"])

# 创建图
graph = helper.make_graph(
    [A, B, C, D, E],
    "test_graph",
    [
        helper.make_tensor_value_info("input0", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1]),
    ],
    [
        helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1]),
    ]
)

# 保存模型
model = helper.make_model(graph)
onnx.save(model, "test_model.onnx")
运行：python create_test_model.py

步骤 3：使用 Polygraphy 匹配子图
polygraphy plugin match test_model.onnx \
    --plugin-dir ./plugins \
    -o config.yaml
预期输出：
[I] Found a matched subgraph!
生成的 config.yaml 内容：
name: toyPlugin
instances:
- inputs: [input0, input1]
  outputs: [output0, output1]
  attributes:
    ToyX: 3  # 因为 x=1.5 → int(1.5)=1 → 1*2=2? 等等！
 注意：代码中 int(m.get("Cnode").attrs["x"])

若 x=1.5，int(1.5) = 1，所以 ToyX = 1 * 2 = 2
（文档示例写 3 可能是笔误）

步骤 4：（可选）人工审查 config.yaml;确认输入/输出是否正确，属性值是否合理。

步骤 5：执行替换
polygraphy plugin replace test_model.onnx \
    --plugin-dir ./plugins \
    --config config.yaml \
    -o replaced_model.onnx
步骤 6：验证结果
用 Netron 打开 replaced_model.onnx，应看到：
Node {
  op_type: "CustomToyPlugin",
  input: ["input0", "input1"],
  output: ["output0", "output1"],
  attribute { name: "ToyX", i: 2 }
}
 原来的 5 个节点（A,B,C,D,E）被替换为 1 个插件节点！

步骤 7：TensorRT 推理验证
确保你的 TRT 环境已加载 CustomToyPlugin：
// C++ 中必须调用
initLibNvInferPlugins(nullptr, "");
// 或显式注册
REGISTER_TENSORRT_PLUGIN(CustomToyPluginCreator);
然后用 Polygraphy 运行：
polygraphy run replaced_model.onnx --trt
应成功执行，无 “Plugin not found” 错误。

 三、关键注意事项
问题	        解决方案
插件未注册	    确保 TRT 运行时加载了插件库
属性类型不匹配	 ONNX 属性 i (int), f (float) 必须与插件期望一致
多匹配冲突	    在 config.yaml 中手动删除不需要的匹配
模式太宽泛	    在 check_func 中增加更多约束（如输入形状）

 总结
这段代码是 Polygraphy 插件集成系统的“模式定义核心”，它实现了：
声明式子图描述：用代码清晰定义要替换的结构
动态属性推导：从原图自动计算插件参数
标准化输出：生成 plugin replace 可直接使用的配置
 最佳实践：将 pattern.py 视为 “插件与 ONNX 模型之间的契约” ——
它精确描述了：“在什么条件下，用我的插件替换哪部分计算”。掌握此机制，
你就能将任何 CUDA 高性能内核无缝集成到 ONNX/TensorRT 生态中，实现 “模型可移植性” 与 “推理极致性能” 的完美统一。
"""