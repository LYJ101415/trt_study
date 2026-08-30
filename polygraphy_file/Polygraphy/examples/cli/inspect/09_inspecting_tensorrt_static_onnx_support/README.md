# Inspecting TensorRT ONNX Support

## Introduction

The `inspect capability` subtool provides detailed information on TensorRT's ONNX operator support for a given ONNX graph.
It is also able to partition and save supported and unsupported subgraphs from the original model in order to report all the dynamically checked errors with a given model (see the example `08_inspecting_tensorrt_onnx_support`).

## Running The Example

1. Generate the capability report

    ```bash
    polygraphy inspect capability nested_local_function.onnx
    ```

2. This should display a summary table like:

    ```
    [I] ===== Summary =====
        Stack trace                       | Operator  | Node               | Reason
        -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        OuterFunction -> NestedLocalFake2 | Fake_2    | nested_node_fake_2 | In node 0 with name: nested_node_fake_2 and operator: Fake_2 (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"
        OuterFunction                     | Fake_1    | nested_node_fake_1 | In node 0 with name: nested_node_fake_1 and operator: Fake_1 (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"
    ```

## Understanding The Output

In this example, `nested_local_function.onnx` contains `Fake_1` and `Fake_2` nodes that are not supported by TensorRT. `Fake_1` node is located inside a local function `OuterFunction` and `Fake_2` node is located inside a nested local function, `NestedLocalFake2`.
The summary table shows the current stack trace consisting of local functions, the operator in which the error occurred and the reason it's unsupported.

For more information and options, see `polygraphy inspect capability --help`.
=========================================================================
这份文档展示了 Polygraphy 的 inspect capability 工具在处理含“嵌套本地函数（Nested Local Functions）”的 ONNX 模型时的能力。这是对前一版本能力检查的重要补充，专门解决 ONNX 中高级结构（如函数封装、嵌套调用）带来的算子支持性诊断难题。

下面从 ONNX 本地函数背景、技术突破点、输出结构解析、与分区模式对比、工程意义 五个维度进行深度解析。

一、背景：ONNX 中的“本地函数（Local Function）”是什么？
1. FunctionProto：ONNX 的“子程序”机制：ONNX 允许将一组节点封装为 可复用的函数（类似编程语言中的函数）；函数定义存储在模型的 functions 字段中。
调用方式：通过一个 特殊节点（OpType = 函数名）引用
2. 嵌套函数（Nested Local Function）：函数 A 内部调用函数 B → 形成调用栈
示例结构：
main_graph
  └── call OuterFunction()
        └── call NestedLocalFake2()  ← 嵌套层级
3. 为什么这很重要？
现代模型（如 HuggingFace Transformers 导出的 ONNX）大量使用函数封装
传统工具无法穿透函数边界 → 报错模糊（仅说“某函数不支持”）
二、技术突破：Polygraphy 如何实现嵌套函数诊断？
关键机制：递归遍历 + 调用栈追踪；静态分析所有 FunctionProto；不仅检查主图（graph），还递归检查每个 FunctionProto；对嵌套函数，记录完整调用路径；构建“错误上下文栈”；当在 NestedLocalFake2 中发现不支持算子；记录路径：OuterFunction -> NestedLocalFake2
精准定位节点；即使节点深藏在 3 层函数内，也能报告其原始名称和位置。

 与 --with-partitioning 模式的核心区别：
此模式（默认）能完整处理 FunctionProto，而分区模式不能！

三、输出结构逐字段解析；示例输出：
[I] ===== Summary =====
    Stack trace                       | Operator  | Node               | Reason
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    OuterFunction -> NestedLocalFake2 | Fake_2    | nested_node_fake_2 | In node 0 with name: nested_node_fake_2 and operator: Fake_2 (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"
    OuterFunction                     | Fake_1    | nested_node_fake_1 | In node 0 with name: nested_node_fake_1 and operator: Fake_1 (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"
字段详解：
1. Stack trace（调用栈）：OuterFunction -> NestedLocalFake2；表示错误发生在 NestedLocalFake2 函数内；该函数被 OuterFunction 调用
作用：快速定位问题在模型中的“逻辑位置”
2. Operator（算子类型）：Fake_2, Fake_1；实际不被 TRT 支持的算子名；
可能是自定义插件名或标准 ONNX 算子
3. Node（节点名称）：nested_node_fake_2；ONNX 节点的 name 属性（若未设置则为空）
关键标识：用于在 Netron 等工具中搜索定位
4. Reason（失败原因）：包含 TRT 原生错误信息："Plugin not found..." → 插件未注册
若是标准算子 → 会显示 “Unsupported operator”；精确到具体节点（In node 0 with name: ...）
四、与 --with-partitioning 模式的对比
特性	           默认模式（本例）	   --with-partitioning 模式
FunctionProto 支持	 完整支持（含嵌套）	 不支持（盲区）
错误来源	       静态规则匹配	             动态 TRT 解析器实际报错
子图保存	       不生成	           可保存支持/不支持子图
适用场景	诊断复杂模型结构	调试具体解析失败
推荐使用	首次检查必用	    已知无函数时使用

 官方建议：先用 默认模式 检查（确保覆盖所有函数），若无函数且需子图，再用 --with-partitioning

五、工程价值与典型应用场景
场景 1：调试 HuggingFace 模型导出问题
HF 模型导出的 ONNX 常含 ScatterElements, LayerNormalization 封装函数
使用此工具可精准定位：
BertModel -> EmbedLayerNormalization | LayerNorm | embed_ln | Unsupported operator
场景 2：验证自定义插件覆盖范围
开发 GridSampler_TRT 插件后；检查是否所有 GridSampler 节点（包括函数内）都被覆盖
场景 3：客户模型支持评估
客户提供含私有函数的 ONNX
生成报告明确告知：“需为 CustomOp_X 开发插件，位于 PreprocessingFunc 内”
场景 4：CI/CD 集成
# 在部署流水线中加入检查
result = polygraphy.inspect_capability("model.onnx")
if result.unsupported_ops:
    raise DeploymentError(f"Unsupported ops: {result.unsupported_ops}")
六、操作建议与最佳实践
1. 始终优先使用默认模式
polygraphy inspect capability your_model.onnx
确保不遗漏任何函数内的节点
2. 结合 Netron 可视化
用 Netron 打开 ONNX；搜索 nested_node_fake_2 快速定位
3. 处理插件缺失
若报错 "Plugin not found"：确认是否链接了插件库：
# C++ 中必须调用
initLibNvInferPlugins(nullptr, "");
检查插件注册名是否匹配 ONNX 节点的 domain 和 op_type
4. 展开函数（备用方案）
若需用 --with-partitioning，先展开函数：
polygraphy surgeon extract model.onnx --fold-functions --output unfolded.onnx
polygraphy inspect capability --with-partitioning unfolded.onnx
✅ 总结：此功能的核心价值
维度	    说明
问题域	    解决 ONNX 高级结构（函数/嵌套）的诊断黑盒
技术亮点	递归遍历 + 调用栈追踪 + 精准节点定位
不可替代性	是目前唯一能完整诊断嵌套函数内算子支持性的开源工具
使用原则	任何含函数的 ONNX 模型，必须用此模式检查！

 一句话精髓：“当你的 ONNX 模型像俄罗斯套娃一样嵌套时，Polygraphy 能一层层打开并告诉你：问题出在第几层的哪个零件上。”因此，这个看似简单的命令，实际上是处理现代复杂 ONNX 模型（尤其是 NLP/CV 大模型导出产物）的必备诊断利器。掌握它，你就拥有了透视 ONNX 模型内部结构的能力。

