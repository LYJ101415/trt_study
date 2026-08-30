# Inspecting TensorRT ONNX Support

## Introduction

The `inspect capability` subtool provides detailed information on TensorRT's ONNX operator support for a given ONNX graph.
It is also able to partition and save supported and unsupported subgraphs from the original model in order to report all the dynamically checked errors with a given model.

## Running The Example

1. Generate the capability report

    ```bash
    polygraphy inspect capability --with-partitioning model.onnx
    ```

2. This should display a summary table like:

    ```
    [I] ===== Summary =====
        Operator | Count   | Reason                                                                                                                                                                    | Nodes
        -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        Fake     |       1 | In node 0 with name:  and operator: Fake (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[2, 3]]
    ```

## Understanding The Output

In this example, `model.onnx` contains a `Fake` node that is not supported by TensorRT.
The summary table shows the unsupported operator, the reason it's unsupported, how many times it appears in the graph,
and the index range of these nodes in the graph in case there are multiple unsupported nodes in a row.
Note that this range uses an inclusive start index and an exclusive end index.

It is important to note that the graph partitioning logic (`--with-partitioning`) currently does not support surfacing issues with nodes inside local functions (`FunctionProto`s). See the description of the default flow (without `--with-partitioning` option, described in the example `09_inspecting_tensorrt_static_onnx_support`) for static error reporting that properly handles nodes inside local functions.

For more information and options, see `polygraphy inspect capability --help`.
===========================================================================
这份文档介绍了 Polygraphy 的 inspect capability 工具，用于深度分析 TensorRT 对 ONNX 模型中算子的支持情况。这是模型部署前的关键诊断步骤，尤其适用于包含自定义算子、新版本 ONNX 算子或复杂控制流的模型。

下面从 核心目的、技术原理、命令详解、输出解析、分区机制、局限性、工程价值 七个维度进行系统性解析。

一、核心目的：为什么需要 Inspect Capability？
关键问题场景：模型导入 TRT 失败；报错模糊（如 “unsupported operator”），但不知道具体是哪个节点；部分算子不支持；模型大部分可运行，但含少量 TRT 不支持的算子（如 Einsum, NonMaxSuppression）；需要量化支持范围；团队评估是否能用 TRT 部署某类模型（如 DETR、ViT）；调试插件缺失；自定义插件未注册，导致报错 “Plugin not found”
 inspect capability 的核心价值：
精准定位不支持的算子 + 提供上下文信息 + 分离可运行/不可运行子图

二、技术原理：如何实现能力检查？
Polygraphy 采用 “模拟构建 + 图分割” 双重机制：

1. 静态分析（默认）；解析 ONNX 模型的 GraphProto；对比 TRT 内置的 ONNX 算子支持表（基于当前 TRT 版本）。
优点：快速、支持 FunctionProto（本地函数）
缺点：无法捕获动态依赖（如某些算子组合才失败）
2. 动态分区（--with-partitioning）；尝试将 ONNX 图 分割为多个子图；Supported Subgraph：TRT 可解析的部分；Unsupported Subgraph：含不支持算子的部分；对每个子图调用 TRT 解析器，捕获真实错误信息。
优点：错误信息更准确（来自 TRT 原生日志）
缺点：不支持 FunctionProto 内部的节点（见文档警告）

 关键区别：
无 --with-partitioning → 静态检查（全面但可能不够精确）
有 --with-partitioning → 动态检查（精确但有盲区）
三、命令详解与选项
基础命令：
polygraphy inspect capability model.onnx
仅做静态支持性检查
启用动态分区（推荐用于调试）：
polygraphy inspect capability --with-partitioning model.onnx
其他实用选项：
选项	             作用
--save-subgraphs	保存支持/不支持的子图为独立 ONNX 文件
--trt-version	    指定 TRT 版本（如 8.6）进行兼容性检查
--exclude-ops	    忽略特定算子（用于测试插件覆盖）
--verbose	        显示详细错误堆栈

四、输出结构深度解析
示例输出：
[I] ===== Summary =====
    Operator | Count   | Reason                           | Nodes
    Fake     |       1 | In node 0 with name:  and operator: Fake (checkFallbackPluginImporter): INVALID_NODE: creator && "Plugin not found, are the plugin name, version, and namespace correct?" | [[2, 3]]
字段逐项解读：
1. Operator（算子名）
Fake：ONNX 中的算子类型（OpType）；若为自定义算子，此处显示插件名（如 GridAnchor_TRT）
2. Count（出现次数）：该算子在图中出现的总次数；用于评估影响范围（1 个 vs 100 个）
3. Reason（失败原因）：最关键字段！包含：节点索引：In node 0；算子名：operator: Fake
TRT 原生错误："Plugin not found..."
错误来源：若是标准算子 → TRT 不支持该 OpType；若是插件算子 → 插件未注册或版本不匹配
4. Nodes（节点索引范围）：[[2, 3]] 表示 节点索引从 2（含）到 3（不含）
即 仅节点 2 不支持；若连续多个节点不支持：[[5, 8]] → 节点 5,6,7

 为什么用半开区间 [start, end)？与 Python 切片一致，便于程序处理（如 nodes[2:3]）

五、图分区（Partitioning）机制详解
当使用 --with-partitioning 时，Polygraphy 执行以下操作：

步骤 1：遍历 ONNX 图；识别所有 不支持的节点；将图分割为 最大连续支持子图
步骤 2：生成子图文件（若加 --save-subgraphs）
# 生成文件示例
model_supported_0.onnx    # 第一个支持子图
model_unsupported_0.onnx  # 第一个不支持子图
步骤 3：对每个子图尝试 TRT 解析：支持子图：应成功解析（验证分割正确性）
不支持子图：捕获 TRT 原生错误（提供精确原因）
应用场景：开发自定义插件：
用 model_unsupported_0.onnx 作为插件测试输入
模型重构：将不支持子图替换为 TRT 支持的等效结构

六、重要局限性与注意事项
1. FunctionProto 盲区（关键限制！）
ONNX 允许定义 本地函数（FunctionProto），类似子程序
--with-partitioning 无法检查 Function 内部的节点
解决方案：先用 polygraphy surgeon extract 展开函数，再检查：
polygraphy surgeon extract model.onnx --fold-constants --output unfolded.onnx
polygraphy inspect capability --with-partitioning unfolded.onnx
2. 动态 Shape 影响
某些算子在动态 shape 下不支持（如 Reshape with unknown dim），建议先固定 shape 再检查
3. TRT 版本依赖
支持列表随 TRT 版本变化（如 TRT 8.5+ 支持 Gelu）；使用 --trt-version 指定目标版本
七、工程价值与典型工作流
场景 1：部署前可行性评估
polygraphy inspect capability my_model.onnx --trt-version 8.6
输出支持率：95% operators supported；决策：是否值得投入插件开发
场景 2：调试插件注册问题
错误：Plugin not found
检查：插件是否在 libnvinfer_plugin.so 中；是否调用 initLibNvInferPlugins()
ONNX 节点属性是否匹配插件要求
场景 3：模型适配 TRT
步骤：用 --save-subgraphs 获取不支持子图；用 ONNX 修改工具（如 onnx-simplifier）替换算子；重新检查直至全支持
场景 4：客户支持：客户提供 model.onnx 报错；用此工具生成 精准诊断报告，附带不支持节点位置
 总结：inspect capability 的核心价值
维度	    说明
定位	    TensorRT ONNX 支持性的“CT 扫描仪”
核心能力	精准定位不支持算子 + 提供 TRT 原生错误 + 分割子图
不可替代性	比 trtexec --onnx=model.onnx 提供更结构化、可操作的诊断信息
最佳实践	任何 ONNX 模型导入 TRT 前，必先运行此检查！

 终极建议：当你面对 “Why does TensorRT fail to parse my ONNX model?” 时，
不要看 TRT 的模糊日志，运行 polygraphy inspect capability --with-partitioning。
它会直接告诉你：哪个节点、什么算子、为什么失败 —— 这就是高效调试的起点。
因此，尽管这是一个诊断工具，但它在 AI 模型部署流水线中扮演着“守门员”角色，能大幅降低集成风险和调试成本。