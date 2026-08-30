检查 ONNX 模型
简介
check lint 子工具用于验证 ONNX 模型，并生成一份 JSON 报告，详细说明模型中存在的错误节点、未使用节点或模型级错误。
运行示例
对 ONNX 模型进行 Lint 检查：
polygraphy check lint bad_graph.onnx -o report.json

输出内容大致如下：
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

该命令会生成一个 report.json 文件，其中包含模型存在的问题信息。
上述示例使用了一个存在故障的 ONNX 模型 bad_graph.onnx，Linter 捕获了其中的多个错误和警告。
具体错误包括：
模型的名称字段为空。
节点 Add_0、MatMul_0 和 MatMul_3 的输入形状不兼容。
具体警告包括：
输入 A 和 B 未被任何输出使用。
节点 MatMul_0 未被任何输出使用。
报告示例：
生成的报告内容如下所示：
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
注意事项
由于该工具底层依赖 ONNX Runtime 运行，因此可以通过 --providers 参数指定执行提供程序（Execution Providers）。默认使用 CPU。
此外，还可以通过 --input-shapes 参数覆盖输入形状，或提供自定义输入数据。更多详情请参阅 how-to/use_custom_input_data。
如需了解更多用法信息，请运行 polygraphy check lint --help。

运行 polygraphy check lint --help输出如下：

[实验性功能] 对 ONNX 模型进行拓扑级“Lint”检查，以查找图中的故障节点。
所有依赖于故障节点的下游节点都将被标记为故障节点并被忽略。
所有错误信息和警告均会被记录在一份 JSON 报告中。
该 JSON 报告包含以下字段：
summary：汇总了被检查节点中通过和失败的节点列表。
（注意：此列表并非详尽无遗，因为某些节点可能由于依赖于先前的故障节点而被跳过）
lint_entries：Lint 检查条目列表，每个条目包含以下字段：
level：该条目的严重级别（error 或 warning）
source：生成该错误信息的底层检查器（onnx.checker 或 ONNX Runtime）
message：错误信息。该信息经过了初步解析/修剪，但可能仍保留底层检查器的原始格式。
（可选）nodes：与该错误信息相关的节点列表。如果不存在此字段，
则表示该条目是针对整个模型的全局错误/警告（例如缺少 opset 导入）。
JSON 输出的 Schema 如下：
{
    "summary": {
        "passing": [<通过 ORT 推理检查的节点列表>],
        "failing": [<未通过 ORT 推理检查的节点列表>]
    },
    "lint_entries": [
        { "level": <严重级别>, "source": <错误来源>, "message": <错误字符串>, "nodes": [<故障节点名称>] },
        ...
    ]
}
已知限制：
目前不支持 BFLOAT16 和 FLOAT8 数据类型。
JSON 报告中仅捕获相互独立的错误节点。依赖于故障节点的下游节点不会被检查。
不会递归检查嵌套在节点内部的子图。
自定义算子（Custom Ops）在 JSON 报告中被记录为警告，但在内部推理检查中被视为异常。因此，依赖于自定义算子的下游节点不会被检查是否存在错误或自定义算子问题。
该子工具根据用户提供的输入数据或为输入张量生成的随机数据来验证数据依赖型故障。因此，该子工具对子图的覆盖范围完全取决于输入数据，不保证 100% 覆盖。
例如，如果子图包含条件分支，该子工具只会检查基于当前输入数据所触发的那个分支。
大型模型（>2GB）要求外部数据文件与模型文件位于同一目录下，不支持自定义外部数据路径。
选项：
-h, --help 显示此帮助信息并退出
-o OUTPUT, --output OUTPUT
保存 JSON 报告的路径。
日志记录：
与日志记录和调试输出相关的选项
-v, --verbose 增加日志详细程度。可多次指定以提高详细级别
-q, --quiet 降低日志详细程度。可多次指定以降低详细级别
--verbosity VERBOSITY [VERBOSITY ...]
设置日志详细程度。优先级高于 -v 和 -q 选项，且与它们不同，允许按路径控制详细程度。详细程度值应来自 Polygraphy Logger 类中定义的日志级别，且不区分大小写。例如：--verbosity INFO 或 --verbosity verbose。要指定按路径的详细程度，请使用格式：<path>:<verbosity>。例如：--verbosity backend/trt:INFO backend/trt/loader.py:VERBOSE。路径应相对于 polygraphy/ 目录。例如，polygraphy/backend 应简写为 backend。系统会使用最匹配的路径来确定详细程度。例如，使用 --verbosity warning backend:info backend/trt:verbose 时，polygraphy/comparator 下的文件将使用 WARNING 级别，backend/onnx 下的文件将使用 INFO 级别，而 backend/trt 下的文件将使用 VERBOSE 级别。
--silent 禁用所有输出
--log-format {timestamp,line-info,no-colors} [{timestamp,line-info,no-colors} ...]
日志消息格式：{{'timestamp': 包含时间戳, 'line-info': 包含文件名和行号, 'no-colors': 禁用颜色}}
--log-file LOG_FILE Polygraphy 日志输出写入的文件路径。这可能不包含依赖项（如 TensorRT 或 ONNX Runtime）的日志输出。
模型：
与模型相关的选项
model_file 模型文件路径
--input-shapes INPUT_SHAPES [INPUT_SHAPES ...], --inputs INPUT_SHAPES [INPUT_SHAPES ...]
模型输入及其形状。用于确定生成推理输入数据时使用的形状。格式：--input-shapes :。
例如：--input-shapes image:[1,3,224,224] other_input:[10]
ONNX 模型加载：
与加载 ONNX 模型相关的选项。
--external-data-dir EXTERNAL_DATA_DIR, --load-external-data EXTERNAL_DATA_DIR, --ext EXTERNAL_DATA_DIR
包含模型外部数据的目录路径。通常仅在外部数据未存储在模型所在目录时才需要此选项。
--ignore-external-data
忽略外部数据，仅加载模型结构而不加载任何权重。该模型仅可用于不需要权重的用途，例如提取子图或检查模型结构。这在外部数据不可用的情况下很有用。
--fp-to-fp16 将 ONNX 模型中的所有浮点张量转换为 16 位精度。使用 TensorRT 的 fp16 精度不需要此选项，但对其他后端可能有用。需要安装 onnxmltools。
数据加载器：
与加载或生成推理输入数据相关的选项。
--seed SEED 用于生成随机输入的随机种子
--val-range VAL_RANGE [VAL_RANGE ...]
数据加载器中生成值的范围。要指定每个输入的范围，请使用格式：--val-range :[min,max]。如果未提供输入名称，则该范围适用于所有未明确指定的输入。例如：--val-range [0,1] inp0:[2,50] inp1:[3.0,4.6]
--int-min INT_MIN [已弃用：请使用 --val-range] 随机整数输入的最小整数值
--int-max INT_MAX [已弃用：请使用 --val-range] 随机整数输入的最大整数值
--float-min FLOAT_MIN
[已弃用：请使用 --val-range] 随机浮点输入的最小浮点值
--float-max FLOAT_MAX
[已弃用：请使用 --val-range] 随机浮点输入的最大浮点值
--iterations NUM, --iters NUM
默认数据加载器提供数据的推理迭代次数
--data-loader-backend-module {numpy,torch}
用于生成输入数组的模块。目前支持的选项：numpy, torch
--load-inputs LOAD_INPUTS_PATHS [LOAD_INPUTS_PATHS ...], --load-input-data LOAD_INPUTS_PATHS [LOAD_INPUTS_PATHS ...]
加载输入数据的路径。文件应为 JSON 序列化的 List[Dict[str, numpy.ndarray]]，即一个列表，其中每个元素是单次迭代的 feed_dict。使用此选项时，所有其他数据加载器参数将被忽略。
--data-loader-script DATA_LOADER_SCRIPT
定义加载输入数据函数的 Python 脚本路径。该函数不应接受任何参数，并返回一个生成器或可迭代对象，产出输入数据（Dict[str, np.ndarray]）。使用此选项时，所有其他数据加载器参数将被忽略。默认情况下，Polygraphy 会查找名为 load_data 的函数。您可以通过冒号分隔来指定自定义函数名。例如：my_custom_script.py:my_func
--data-loader-func-name DATA_LOADER_FUNC_NAME
[已弃用 - 可以使用 --data-loader-script 指定函数名，格式如：my_custom_script.py:my_func] 使用 data-loader-script 时，此选项指定加载数据的函数名称。默认为 load_data。
ONNX-Runtime 会话创建：
与创建 ONNX-Runtime 推理会话相关的选项
--providers PROVIDERS [PROVIDERS ...], --execution-providers PROVIDERS [PROVIDERS ...]
按优先级排序的执行提供程序列表。每个提供程序可以是 ONNX-Runtime 中可用执行提供程序的精确匹配或不区分大小写的部分匹配。例如，值 'cpu' 将匹配 'CPUExecutionProvider'