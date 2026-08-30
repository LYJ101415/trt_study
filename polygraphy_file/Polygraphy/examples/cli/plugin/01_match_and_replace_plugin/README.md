# Matching and replacing a subgraph with a plugin in an onnx model


## Introduction

The `plugin` tool offers subtools to find and replace subgraphs in an onnx model.

Subgraph substition is a three-step process:
1. Find matching subgraphs based on the plugin's graph pattern (pattern.py) and list the potential substitutions in a user-editable intermediate file (config.yaml)
2. Review and edit (if necessary) the list of potential substitutions (config.yaml)
3. Replace subgraphs with plugins based on the list of potential substitutions (config.yaml)


`original.onnx` -------> `match` -------> `config.yaml` -------> `replace` -------> `replaced.onnx`
`plugins` ----------------^   `usr input`---^    `plugins`--------^                     

## Details

### Match
Finding matchings subgraphs in a model is done based on a graph pattern description (`pattern.py`) provided by the plugins.
The graph pattern description (`pattern.py`) contains information about the topology and additional constraints for the graph nodes, and a way to calculate the plugin's attributes based on the matching subgraph. 
Only plugins which provide a graph pattern description (pattern.py) are considered for matching.

The result of the matching is stored in an intermediate file called `config.yaml`. 
The user should review and edit this file, as it serves as a TODO list for the replacement step. For example, if there are 2 matching subgraphs, but only one should be substituted, the result can be removed from the file.

As a preview/dry-run step, the `plugin list` subtool can show the list of potential substitutions without generating an intermediate file.

### Replace
Replacement of subgraphs with plugins uses the `config.yaml` file generated in the matching stage. Any matching subgraph listed in this file is going to be removed and replaced with a single node representing the plugin. The original file is kept, and a new file is saved where the replacements are done. This file by default is called `replaced.onnx`.

### Compare
The original and the replaced model can be compared to check if they behave the same way before and after plugin substitution:
`polygraphy run original.onnx --trt --save-outputs model_output.json`
`polygraphy run replaced.onnx --trt --load-outputs model_output.json`

## Running The Example

1. Find and save matches of toyPlugin in the example network:

    ```bash
    polygraphy plugin match toy_subgraph.onnx \
        --plugin-dir ./plugins -o config.yaml
    ```

    <!-- Polygraphy Test: Ignore Start -->
    This will display something like:

    ```
    checking toyPlugin in model
    [I] Start a subgraph matching...
    [I] 	Checking node: n1 against pattern node: Anode.
    [I] 	No match because: Op did not match. Node op was: O but pattern op was: A.
    [I] Start a subgraph matching...
    [I] Found a matched subgraph!
    [I] Start a subgraph matching...
    ```

    The resulting config.yaml will look like:

    ```
    name: toyPlugin
    instances:
    - inputs:
    - i1
    - i1
    outputs:
    - o1
    - o2
    attributes:
        x: 1
    ```
    <!-- Polygraphy Test: Ignore End -->

2. **[Optional]** List matches of toyPlugin in the example network, without saving config.yaml:

    ```bash
    polygraphy plugin list toy_subgraph.onnx \
        --plugin-dir ./plugins
    ```

    <!-- Polygraphy Test: Ignore Start -->
    This will display something like:

    ```
    checking toyPlugin in model
    [I] Start a subgraph matching...
    [I] 	Checking node: n1 against pattern node: Anode.
    [I] 	No match because: Op did not match. Node op was: O but pattern op was: A.
    [I] Start a subgraph matching...
    ...
    [I] Found a matched subgraph!
    [I] Start a subgraph matching...
    [I] 	Checking node: n6 against pattern node: Anode.
    [I] 	No match because: Op did not match. Node op was: E but pattern op was: A.
    the following plugins would be used:
    {'toyPlugin': 1}
    ```

    There will be no resulting config.yaml, as this command is only for printing the number of matches per plugin
    <!-- Polygraphy Test: Ignore End -->

The `plugin replace` subtool replaces subgraphs in an onnx model with plugins


3. Replace parts of the example network with toyPlugin:

    ```bash
    polygraphy plugin replace toy_subgraph.onnx \
        --plugin-dir ./plugins --config config.yaml -o replaced.onnx
    ```

    <!-- Polygraphy Test: Ignore Start -->
    This will display something like:

    ```
    [I] Loading model: /Users/pkisfaludi/Documents/git/Polygraphy/examples/cli/plugin/03_replace_subgraph_with_a_plugin/toy_subgraph.onnx
    ```

    The result file is replaced.onnx, where a subgraph in the example network is replaced by toyPlugin
    <!-- Polygraphy Test: Ignore End -->
==========================================================================
这份文档详细介绍了 Polygraphy 的 plugin 工具链，用于在 ONNX 模型中自动匹配并替换子图为自定义 TensorRT 插件节点。这是将高性能 CUDA 内核集成到 ONNX/TensorRT 推理流程中的标准化方法。

下面从 核心目标、三步工作流、关键技术细节、插件开发要求、验证机制、工程价值 六个维度进行系统性深度解析。

一、核心目标：为什么需要子图替换？
背景问题：TensorRT 原生算子性能不足；某些复杂操作（如 Deformable Conv、Custom Attention）无高效原生实现；模型含不支持算子；ONNX 中的 GridSample、NMS 等需用插件实现
融合优化需求；将多个算子（Conv+BN+ReLU）融合为单个插件，减少 kernel launch 开销。
解决方案：用高性能 CUDA 插件 替换 ONNX 中的等效子图，实现 “ONNX 可读性” + “TRT 高性能” 的统一。
 Polygraphy plugin 工具的核心价值：自动化、可审查、可验证 地完成子图 → 插件替换

二、三步标准工作流详解
整个过程是 “发现 → 审查 → 替换” 的闭环：

步骤 1：Match（发现匹配子图）
polygraphy plugin match toy_subgraph.onnx \
    --plugin-dir ./plugins -o config.yaml
关键机制：遍历所有插件：仅处理含 pattern.py 的插件。
子图匹配算法：基于 拓扑结构（节点连接关系）；检查 算子类型（OpType）；验证 约束条件（如输入张量形状、属性值）；生成中间文件：config.yaml 作为“待办清单”；
输出示例 (config.yaml)：
name: toyPlugin
instances:
  - inputs: [i1, i1]      # 匹配子图的输入张量名
    outputs: [o1, o2]     # 匹配子图的输出张量名
    attributes:           # 从子图推导的插件属性
      x: 1

 设计精妙之处：attributes 字段由 pattern.py 中的 infer_attributes() 动态计算，
例如从 Conv 的 kernel_shape 推导插件的 kernel_size 参数。

步骤 2：Review（人工审查）
这是最关键的一步！为什么需要人工干预？
场景	    说明
误匹配	    拓扑相同但语义不同（如两个 Add 节点 vs 插件要求的特定 Add）
部分替换	模型中有 5 个匹配子图，但只需替换 3 个
属性修正	自动推导的属性有误，需手动调整

操作方式：直接编辑 config.yaml；删除不需要的 instances 条目；修改 attributes 值

 最佳实践：将 config.yaml 纳入版本控制，作为替换决策的审计记录。

步骤 3：Replace（执行替换）
polygraphy plugin replace toy_subgraph.onnx \
    --plugin-dir ./plugins --config config.yaml -o replaced.onnx
执行过程：加载 original.onnx；对 config.yaml 中每个 instance：
删除 子图中的所有节点；插入 新节点：
Onnx
编辑
Node {
  op_type: "toyPlugin",
  domain: "com.example.plugins",  // 来自插件注册信息
  input: ["i1", "i1"],
  output: ["o1", "o2"],
  attribute { name: "x", i: 1 }
}
保存为 replaced.onnx

 安全机制：原始文件 original.onnx 永不修改，确保可回溯。

三、关键技术细节
1. 插件必须提供 pattern.py；这是匹配的“蓝图”，典型结构：
# plugins/toyPlugin/pattern.py
from polygraphy.backend.trt import TacticReplayData

def get_pattern():
    return {
        "nodes": [
            {"op": "A", "inputs": [], "outputs": ["t1"]},
            {"op": "B", "inputs": ["t1"], "outputs": ["o1", "o2"]}
        ],
        "inputs": ["i1"],
        "outputs": ["o1", "o2"]
    }

def infer_attributes(subgraph):
    # 从匹配的子图推导插件属性
    return {"x": subgraph.nodes[0].attribute["value"].i}
2. Dry-run 模式：plugin list
polygraphy plugin list model.onnx --plugin-dir ./plugins
仅打印匹配数量（如 {'toyPlugin': 1}）；不生成文件，适合快速验证
3. 多插件支持
同时扫描目录下所有插件；config.yaml 可包含多个插件的替换项
四、验证机制：确保功能正确性；替换后必须验证 语义等价性：

1. 精度验证（黄金标准）
# 运行原始模型
polygraphy run original.onnx --trt --save-outputs ref.json

# 运行替换后模型
polygraphy run replaced.onnx --trt --save-outputs test.json

# 比较结果
polygraphy compare ref.json test.json --rtol=1e-3
2. 结构验证
用 Netron 打开 replaced.onnx；确认插件节点已插入，且连接正确
3. 性能验证
polygraphy run replaced.onnx --trt --profile
检查是否达到预期加速比
五、工程价值与典型场景
场景 1：部署自研高性能算子；开发 FusedAttention_TRT 插件。
用此工具自动替换 ONNX 中的 MatMul + Softmax + MatMul 子图
场景 2：修复 TRT 不支持算子；ONNX 含 Einsum("bij,bjk->bik")；开发 Einsum_TRT 插件
自动替换所有匹配的 Einsum 表达式
场景 3：模型压缩与加速
将 Conv -> BN -> ReLU 融合为 FusedConvBNReLU；减少 66% 的 kernel launch 开销
场景 4：客户定制化部署；客户提供 ONNX 模型；用此工具注入私有插件，无需修改原始模型代码
六、最佳实践建议
阶段	     建议
插件开发	 确保 pattern.py 约束足够严格，避免误匹配
Match 阶段	 先用 plugin list 快速验证
Review 阶段	  必须人工检查！尤其关注输入/输出张量名
Replace 阶段	保留 original.onnx 和 config.yaml 用于审计
验证阶段	使用真实数据而非随机输入

 总结：plugin 工具链的核心价值
维度	    说明
定位	    ONNX 模型与 TensorRT 插件之间的“智能桥梁”
核心创新	将模式匹配 + 人工审查 + 自动化替换 结合
不可替代性	是目前唯一支持 “声明式子图替换” 的开源工具
工程意义	使插件集成从 “手动改图” 升级为 “可重复、可验证的流程”

 终极建议：当你需要将 CUDA 内核集成到 ONNX 模型时，不要手动编辑 ONNX 文件！
使用 Polygraphy 的 plugin match → review → replace 三步法，
你将获得：可靠性 + 可维护性 + 可审计性 —— 这就是专业部署的标志。
因此，尽管这是一个高级功能，但对于任何需要极致性能或支持特殊算子的生产环境，它都是不可或缺的利器。