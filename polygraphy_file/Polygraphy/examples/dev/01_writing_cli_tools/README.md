# Writing Custom Command-Line Tools


## Introduction

Polygraphy includes various helper utilities to make it easier to write
new command-line tools from scratch.

In this example, we'll write a brand new tool called `gen-data` that will generate random data
using Polygraphy's default data loader, and then write it to an output file. The user will
be able to specify the number of values to generate as well as the output path.

To do this, we'll create a child class of `Tool` and use the `DataLoaderArgs` argument
group provided by Polygraphy.


## Running The Example

1. You can run the example tool from this directory. For example:

    ```bash
    ./gen-data -o data.json --num-values 25
    ```

2. We can even inspect the generated data with `inspect data`:

    ```bash
    polygraphy inspect data data.json -s
    ```

To see the other command-line options available in the example tool,
run:
```bash
./gen-data -h
```
=====================================================================

这份文档介绍了如何使用 Polygraphy 提供的工具和辅助类，从零开始编写一个自定义命令行工具（CLI tool）。下面是对该文件的逐层详细解析：
一、背景与目标
1. Polygraphy 的扩展能力：Polygraphy 不仅是一个推理调试/分析工具集，还提供了可复用的框架组件，方便开发者构建自己的 CLI 工具。
这些组件包括：命令行参数解析（基于 argparse 封装）；数据加载器（DataLoader）；日志系统；工具基类（Tool）。
2. 示例目标：创建 gen-data 工具：功能：生成指定数量的随机数据，并保存到文件中。
用户可配置：输出文件路径（--output 或 -o）；要生成的数据元素个数（--num-values）
使用 Polygraphy 内置的 默认数据加载器（Default DataLoader） 来生成符合 ONNX 张量格式的随机数据。
 注意：这里的“数据”指的是可用于模型输入的张量数据（如 float32 数组），而非普通文本。

二、关键技术点解析
1. 继承 Tool 基类
from polygraphy.tools.base import Tool

class GenData(Tool):
    def __init__(self):
        super().__init__("gen-data")  # 工具名称
所有 Polygraphy 自定义工具都应继承 polygraphy.tools.base.Tool
必须实现两个核心方法：
setup_parser(self, parser)：定义命令行参数
run(self, args)：执行主逻辑
2. 使用 DataLoaderArgs 参数组
from polygraphy.tools.args import DataLoaderArgs

# 在 setup_parser 中添加
DataLoaderArgs().add_to_parser(parser)
DataLoaderArgs 是 Polygraphy 提供的标准参数组，用于配置数据生成行为，例如：
--seed：随机种子
--val-range：数值范围（如 [0,1]）
--data-loader：选择数据加载器类型
通过复用此模块，避免重复造轮子，且保持与 Polygraphy 其他工具（如 run、surgeon）一致的用户体验。
3. 自定义参数：--num-values
除了复用已有参数组，还可添加专属参数：
parser.add_argument("--num-values", type=int, required=True, help="Number of values to generate")
4. 数据生成与保存
利用 DataLoaderArgs 构建出的 data_loader 对象生成数据：
from polygraphy.util import load_json

# 创建形状为 [num_values] 的张量描述
fake_input_meta = {"x": (np.float32, (args.num_values,))}

# 生成数据字典: {"x": np.array([...])}
data = args.data_loader.load("random", fake_input_meta)

# 保存为 JSON（Polygraphy 支持 .json / .npz 等格式）
save_json(data, args.output)
注意：虽然用户只指定“值的数量”，但内部需构造一个虚拟输入元信息（input metadata），因为 DataLoader 期望知道每个输入的 dtype 和 shape。
三、运行示例详解
1. 生成数据
./gen-data -o data.json --num-values 25
生成一个包含 25 个随机 float32 数值的张量
保存为 data.json，内容大致如下：
{
  "x": [0.123, -0.456, ..., 0.789]
}
2. 验证生成的数据
polygraphy inspect data data.json -s
-s 表示 show statistics（显示统计信息）
输出可能包括：数据形状：(25,)；数据类型：float32；最小值/最大值/均值等
 这体现了 Polygraphy 工具链的互操作性：自定义工具生成的数据可被官方工具直接消费。

3. 查看帮助
./gen-data -h
会显示：自定义参数（--num-values）；继承自 DataLoaderArgs 的参数（如 --seed, --val-range）；通用选项（如 --verbose）
四、为什么这样做？——设计优势
优势	说明
一致性	自定义工具与 Polygraphy 官方工具共享相同的参数风格和日志格式
复用性	直接使用经过验证的 DataLoader，无需自己实现随机数生成、类型处理等逻辑
可组合性	生成的数据可无缝用于 polygraphy run、inspect data 等其他工具
可维护性	若 Polygraphy 升级 DataLoader 功能（如支持新分布），你的工具自动受益
五、典型应用场景
测试数据生成；快速生成符合模型输入要求的随机数据，用于压力测试或功能验证。
数据预填充；为需要初始化输入的部署流程提供标准数据模板。
教学/演示：展示如何构造符合 ONNX 规范的输入数据。
扩展 Polygraphy 生态：团队可开发内部专用工具（如 gen-calibration-data、validate-output），并集成到统一 CLI 体系中。
六、总结
要素	内容
核心类	Tool（基类）、DataLoaderArgs（参数组）
关键方法	setup_parser(), run()
数据生成	基于虚拟 input metadata + Polygraphy DataLoader
输出格式	JSON / NPZ（与 Polygraphy 生态兼容）
最佳实践	复用标准参数组 + 保持接口一致性 + 提供验证手段

 一句话总结：通过继承 Tool 并组合 Polygraphy 提供的标准化组件（如 DataLoaderArgs），你可以快速构建出专业、可靠、可集成的自定义命令行工具，而无需从零处理命令行解析、数据格式、日志等底层细节。这种模式非常适合在模型部署、测试、调试等环节中扩展自动化能力。