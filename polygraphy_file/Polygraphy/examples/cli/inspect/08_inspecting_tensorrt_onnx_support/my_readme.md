检查 TensorRT 的 ONNX 支持情况
简介
inspect capability 子工具可提供有关 TensorRT 对指定 ONNX 图中 ONNX 算子支持情况的详细信息。
它还能够对原始模型中受支持和不受支持的子图进行划分并保存，以便报告给定模型的所有动态检查错误。
运行示例:
生成能力报告：
polygraphy inspect capability --with-partitioning model.onnx

这将显示一个如下所示的汇总表：
[I] ===== Summary =====
    Operator | Count   | Reason                                                                                                                                                                    | Nodes
    -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Fake     |       1 | In node 0 with name:  and operator: Fake (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[2, 3]]
理解输出内容
在本例中，model.onnx 包含一个 TensorRT 不支持的 Fake 节点。
汇总表显示了不受支持的算子、不受支持的原因、其在图中出现的次数，
以及当存在多个连续的不受支持节点时，这些节点在图中的索引范围。
请注意，该范围采用包含起始索引、不包含结束索引的方式。
需要注意的是，图划分逻辑（--with-partitioning）目前无法暴露局部函数（FunctionProto）内部节点的问题。如需正确处理局部函数内部节点的静态错误报告，请参阅默认流程（不带 --with-partitioning 选项）的说明，详见示例 09_inspecting_tensorrt_static_onnx_support。
更多信息和选项，请参见 polygraphy inspect capability --help。