检查 TensorRT 的 ONNX 支持情况
简介
inspect capability 子工具可提供有关 TensorRT 对指定 ONNX 图中 ONNX 算子支持情况的详细信息。
它还能够对原始模型中受支持和不受支持的子图进行划分并保存，以便报告给定模型的所有动态检查错误（参见示例 08_inspecting_tensorrt_onnx_support）。
运行示例:
生成能力报告：
polygraphy inspect capability nested_local_function.onn

这将显示一个如下所示的汇总表：
[I] ===== Summary =====
    Stack trace                       | Operator  | Node               | Reason
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    OuterFunction -> NestedLocalFake2 | Fake_2    | nested_node_fake_2 | In node 0 with name: nested_node_fake_2 and operator: Fake_2 (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"
    OuterFunction                     | Fake_1    | nested_node_fake_1 | In node 0 with name: nested_node_fake_1 and operator: Fake_1 (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?"
理解输出内容
在本例中，nested_local_function.onnx 包含 TensorRT 不支持的 Fake_1 和 Fake_2 节点。Fake_1 节点位于局部函数 OuterFunction 内部，而 Fake_2 节点位于嵌套的局部函数 NestedLocalFake2 内部。
汇总表显示了由局部函数组成的当前调用栈、发生错误的算子以及其不受支持的原因。
更多信息和选项，请参见 polygraphy inspect capability --help。