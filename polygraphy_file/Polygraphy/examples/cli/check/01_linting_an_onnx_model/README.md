# checking An ONNX Model


## Introduction  简介

The `check lint` subtool validates ONNX Models and generates a JSON report detailing any bad/unused nodes or model errors.
@@@ check lint 子工具用于验证 ONNX 模型，并生成一份 JSON 报告，详细说明模型中存在的错误节点、未使用节点或模型级错误。

## Running The Example  @@@ 运行示例

### Lint the ONNX model: @@@ 对 ONNX 模型进行 Lint 检查：

<!-- Polygraphy Test: XFAIL Start -->
```bash
polygraphy check lint bad_graph.onnx -o report.json
```
<!-- Polygraphy Test: XFAIL End -->
The output should look something like this: @@@ 输出内容大致如下：
```bash
[I] RUNNING | Command: polygraphy check lint bad_graph.onnx -o report.json
[I] Loading model: bad_graph.onnx
[E] LINT | Field 'name' of 'graph' is required to be non-empty.
[I] Will generate inference input data according to provided TensorMetadata: {E [dtype=float32, shape=(1, 4)],
     F [dtype=float32, shape=(4, 1)],
     G [dtype=int64, shape=(4, 4)],
     D [dtype=float32, shape=(4, 1)],
     C [dtype=float32, shape=(3, 4)],
     A [dtype=float32, shape=(1, 3)],
     B [dtype=float32, shape=(4, 4)]}
[E] LINT | Name: MatMul_3, Op: MatMul |  Incompatible dimensions for matrix multiplication
[E] LINT | Name: Add_0, Op: Add |  Incompatible dimensions
[E] LINT | Name: MatMul_0, Op: MatMul |  Incompatible dimensions for matrix multiplication
[W] LINT | Input: 'A' does not affect outputs, can be removed.
[W] LINT | Input: 'B' does not affect outputs, can be removed.
[W] LINT | Name: MatMul_0, Op: MatMul | Does not affect outputs, can be removed.
[I] Saving linting report to report.json
[E] FAILED | Runtime: 1.006s | Command: polygraphy check lint bad_graph.onnx -o report.json
```

- This will create a `report.json` that contains information about what's wrong with the model.
- The above example uses a faulty ONNX Model `bad_graph.onnx` that has multiple errors/warnings captured by the linter.
The errors are:
    1. Model has an empty name.
    2. Nodes `Add_0`, `MatMul_0` and `MatMul_3` have incompatible input shapes.
The warnings are:
    1. Inputs `A` and `B` are unused output.
    2. Node `MatMul_0` is unused by output.

### Example Report:

The generated report looks as follows:

<!-- Polygraphy Test: Ignore Start -->
```json
{
    "summary": {
        "passing": [
            "MatMul_1",
            "cast_to_int64",
            "NonZero"
        ],
        "failing": [
            "MatMul_0",
            "MatMul_3",
            "Add_0"
        ]
    },
    "lint_entries": [
        {
            "level": "exception",
            "source": "onnx_checker",
            "message": "Field 'name' of 'graph' is required to be non-empty."
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions for matrix multiplication",
            "nodes": [
                "MatMul_3"
            ]
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions",
            "nodes": [
                "Add_0"
            ]
        },
        {
            "level": "exception",
            "source": "onnxruntime",
            "message": " Incompatible dimensions for matrix multiplication",
            "nodes": [
                "MatMul_0"
            ]
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Input: 'A' does not affect outputs, can be removed."
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Input: 'B' does not affect outputs, can be removed."
        },
        {
            "level": "warning",
            "source": "onnx_graphsurgeon",
            "message": "Does not affect outputs, can be removed.",
            "nodes": [
                "MatMul_0"
            ]
        }
    ]
}
```
<!-- Polygraphy Test: Ignore End -->

### Notes
Since it runs ONNX Runtime under the hood, it is possible to specify execution providers using `--providers`. Defaults to CPU.

It is also possible to override the input shapes using `--input-shapes`, or provide custom input data. For more details, refer [how-to/use_custom_input_data](../../../../how-to/use_custom_input_data.md).

For more information on usage, use `polygraphy check lint --help`.

================================================================
这份文档详细介绍了 Polygraphy 工具中 check lint 子命令的功能与使用方法，其核心目标是：对 ONNX 模型进行静态与动态分析，自动检测结构错误、算子兼容性问题、未使用节点等潜在缺陷，并生成结构化 JSON 报告。

一、核心目标：为什么需要 polygraphy check lint？ONNX 模型在导出、转换、优化过程中极易引入以下问题：结构不合法（如 graph 名为空）；
算子输入维度不匹配（如 MatMul 的 (1,3) × (4,1)）
存在“死代码”（unused inputs/nodes 不影响最终输出）
运行时崩溃（仅靠 ONNX checker 静态检查无法发现）

check lint 的价值：一站式诊断工具，结合 ONNX 官方校验器 + ONNX Runtime 动态执行 + ONNX GraphSurgeon 静态分析，提供比单一工具更全面的洞察。

二、技术原理：三层检测机制
检测层	            工具	         能力	               示例
1. 静态结构校验	onnx.checker	验证 ONNX 协议合规性	graph.name 不能为空
2. 动态执行校验	ONNX Runtime	实际运行模型，捕获 shape/type 错误	MatMul 维度不匹配
3. 数据流分析	ONNX GraphSurgeon	分析节点依赖关系	输入 A/B 未连接到任何输出

 关键创新：将 静态分析（快但浅）与 动态执行（慢但深）结合，既保证覆盖率，又提供可复现的错误上下文。

三、输出日志与 JSON 报告深度解读：控制台日志（人类可读）
[E] LINT | Field 'name' of 'graph' is required to be non-empty.  # ← 静态错误
[E] LINT | Name: MatMul_3, Op: MatMul | Incompatible dimensions...  # ← 动态错误
[W] LINT | Input: 'A' does not affect outputs...  # ← 静态警告
[E] = Exception（致命错误，模型无法运行）
[W] = Warning（非致命，但可能冗余/低效）

JSON 报告（机器可读）
{
  "summary": {
    "passing": ["MatMul_1", ...],   // 通过校验的节点
    "failing": ["MatMul_0", ...]    // 失败的节点
  },
  "lint_entries": [
    {
      "level": "exception",
      "source": "onnxruntime",       // 错误来源
      "message": "...",
      "nodes": ["MatMul_3"]          // 关联节点（若适用）
    },
    ...
  ]
}
关键字段说明：
字段	      说明
source	    错误检测器：• onnx_checker → ONNX 协议违规；• onnxruntime → 运行时错误
            • onnx_graphsurgeon → 结构冗余
nodes	    出错的具体节点名（用于定位）
level	    exception（阻断）或 warning（建议）

四、典型错误场景与修复建议：
错误 1：空 graph name
"Field 'name' of 'graph' is required to be non-empty."
原因：PyTorch 导出 ONNX 时未指定 example_outputs 或 dynamic_axes 配置错误
修复：torch.onnx.export(..., example_outputs=output, opset_version=13)
错误 2：MatMul 维度不匹配；
Incompatible dimensions for matrix multiplication
原因：前一个算子输出 shape=(1,3)，但 MatMul 期望 (3,4)
修复：检查模型定义中的线性层尺寸
使用 onnx.shape_inference.infer_shapes() 补全中间 shape

警告：未使用的输入/节点
Input: 'A' does not affect outputs, can be removed.
原因：模型导出时包含了未参与计算的 placeholder
价值：减小模型体积；避免推理时传入无用数据
修复：用 onnx-graphsurgeon 删除 dead nodes：
import onnx_graphsurgeon as gs
graph = gs.import_onnx(onnx.load("model.onnx"))
graph.cleanup().toposort()
onnx.save(gs.export_onnx(graph), "clean.onnx")

五、高级用法与生产实践
1. 指定执行设备（加速校验）
polygraphy check lint model.onnx --providers CUDAExecutionProvider
在 GPU 上运行 ONNX Runtime，更快暴露 CUDA 相关错误（如 unsupported op）
2. 覆盖输入 shape（测试边界条件）
polygraphy check lint model.onnx --input-shapes x:[1,3,224,224]
验证动态 shape 下的兼容性（如 batch=1 vs batch=64）
3. 集成到 CI/CD 流水线
# .gitlab-ci.yml 示例
lint-onnx:
  script:
    - polygraphy check lint model.onnx -o report.json
    - if grep -q '"level":"exception"' report.json; then exit 1; fi
阻断合并：若报告含 exception，CI 直接失败
4. 自动生成修复建议
结合 JSON 报告编写脚本：
report = json.load(open("report.json"))
for entry in report["lint_entries"]:
    if entry["level"] == "warning" and "can be removed" in entry["message"]:
        print(f"建议删除节点: {entry.get('nodes', ['INPUT'])}")

六、与其他工具对比
工具	                       能力	              局限
onnx.checker.check_model()	基础协议校验	    无法检测运行时 shape 错误
onnxruntime 直接推理	    捕获 runtime error	无结构分析，错误信息不友好
polygraphy check lint	   三合一诊断	        需安装 Polygraphy

 结论：对于生产级 ONNX 模型交付，check lint 应作为标准质检步骤。

七、总结：核心价值与适用场景
场景	           如何受益
模型导出后验证	  快速发现 PyTorch/TensorFlow → ONNX 的转换错误
模型优化前分析	  识别 dead nodes，指导剪枝
跨平台部署前检查	确保模型在 TRT/ORT/CUDA 上均可运行
团队协作规范	通过 CI 强制模型质量门禁

一句话总结：
“polygraphy check lint 是 ONNX 模型的‘CT 扫描仪’——既看结构合法性，也测功能正确性。”