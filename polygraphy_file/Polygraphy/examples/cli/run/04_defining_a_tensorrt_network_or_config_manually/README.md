# Defining A TensorRT Network Or Config Manually


## Introduction

In some cases, it can be useful to define a TensorRT network from scratch using the Python API,
or modify a network created by other means (e.g. a parser). Normally, this would restrict you
from using CLI tools, at least until you build an engine, since the network cannot be serialized
to disk and loaded on the command-line.

Polygraphy CLI tools provide a work-around for this - if your Python script defines a function
named `load_network`, which takes no parameters and returns a TensorRT builder, network,
and optionally parser, then you can provide your Python script in place of a model argument.

Similarly, we can create a custom TensorRT builder configuration using a script that defines
a function called `load_config` which accepts a builder and network and returns a builder configuration.

In this example, the included `define_network.py` script parses an ONNX model and appends an identity
layer to it. Since it returns the builder, network, and parser in a function called `load_network`,
we can build and run a TensorRT engine from it using just a single command. The `create_config.py`
script creates a new TensorRT builder configuration and enables FP16 mode.


### TIP: Generating Script Templates Automatically

Instead of writing the network script from scratch, you can use
`polygraphy template trt-network` to give you a starting point:

```bash
polygraphy template trt-network -o my_define_network.py
```

If you want to start from a model and modify the resulting TensorRT network instead
of creating one from scratch, simply provide the model as an argument to `template trt-network`:

```bash
polygraphy template trt-network identity.onnx -o my_define_network.py
```

Similarly, you can generate a template script for the config using `polygraphy template trt-config`:

```bash
polygraphy template trt-config -o my_create_config.py
```

You can also specify builder configuration options to pre-populate the script.
For example, to enable FP16 mode:

```bash
polygraphy template trt-config --fp16 -o my_create_config.py
```


## Running The Example

1. Run the network defined in `define_network.py`:

    ```bash
    polygraphy run --trt define_network.py --model-type=trt-network-script
    ```

2. Run the network from step (1) using the builder configuration defined in `create_config.py`:

    ```bash
    polygraphy run --trt define_network.py --model-type=trt-network-script --trt-config-script=create_config.py
    ```

    Note that we could have defined both `load_network` and `load_config` in the same script.
    In fact, we could have retrieved these functions from arbitrary scripts, or even modules.

*TIP: We can use the same approach with `polygraphy convert` to build, but not run, the engine.*
============================================================================
这份文档《Defining A TensorRT Network Or Config Manually》揭示了 Polygraphy CLI 工具与 Python API 深度集成的高级机制，其核心价值在于：“即使你用 Python 代码动态构建或修改了 TensorRT 网络/配置，依然能无缝使用 polygraphy run、polygraphy convert 等命令行工具进行推理或引擎构建。”

下面我将从设计思想、技术原理、使用方法、生成模板、典型场景和工程意义六个维度，为你深度解析。

 一、解决的核心痛点
 传统限制：TensorRT 的 INetworkDefinition 和 IBuilderConfig 是内存对象，无法直接序列化保存为文件；因此，如果你用 Python API 动态创建网络（如添加自定义层、修改结构），无法通过 CLI 工具（如 polygraphy run）直接加载使用；通常必须：手动写完整推理脚本；或先构建 engine 再用 CLI 加载 .engine 文件。
 Polygraphy 的创新方案；通过约定函数名（load_network, load_config），让 CLI 工具能“执行 Python 脚本并获取网络/配置对象”；这打破了 “CLI 只能处理静态模型文件” 的限制，实现了 “动态网络 + 命令行工具” 的无缝融合。

 二、核心技术机制详解
1 网络脚本：load_network() 函数约定
要求：脚本中必须定义一个 无参函数 load_network()；返回值：(builder, network, parser) 或 (builder, network)。
builder: trt.Builder 实例
network: trt.INetworkDefinition 实例
parser (可选): 如 trt.OnnxParser
CLI 调用方式：
polygraphy run --trt your_script.py --model-type=trt-network-script
--model-type=trt-network-script：告诉 Polygraphy 这不是 .onnx 或 .engine，而是一个网络定义脚本
内部流程：
CLI 导入 your_script.py；调用 load_network()；获取 (builder, network) 对象
用这些对象构建 TensorRT engine 并推理
 关键：网络是在运行时动态生成的，但 CLI 工具完全透明地处理了它。

2 配置脚本：load_config() 函数约定
要求：脚本中定义 load_config(builder, network) 函数
参数：builder 和 network（由 load_network 返回）
返回：trt.IBuilderConfig 实例
CLI 调用方式：
polygraphy run ... --trt-config-script=config.py
典型用途：启用 FP16/INT8；设置 workspace size；添加 timing cache；配置 optimization profile（动态 shape）

 三、示例脚本内容解析
示例 1：define_network.py（修改网络结构）
import tensorrt as trt
from polygraphy.backend.trt import CreateNetwork

def load_network():
    # 1. 创建 builder 和 network
    builder, network, parser = CreateNetwork().call_func()
    
    # 2. 解析 ONNX 模型（假设 identity.onnx 只有一个输出）
    parser.parse("identity.onnx")
    
    # 3. 获取原输出
    original_output = network.get_output(0)
    
    # 4. 添加 Identity 层（无实际作用，仅为演示）
    identity_layer = network.add_identity(original_output)
    identity_layer.get_output(0).name = "new_output"
    
    # 5. 将新输出设为网络输出
    network.mark_output(identity_layer.get_output(0))
    
    return builder, network, parser
 效果：最终 TRT 引擎的输出名为 "new_output"，而非原始 ONNX 的输出名。

示例 2：create_config.py（自定义配置）
import tensorrt as trt

def load_config(builder, network):
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)  # 启用 FP16
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    return config

 四、自动生成脚本模板（极大提升效率）
Polygraphy 提供了 template 子命令，避免手写样板代码。

1. 生成空白网络脚本
polygraphy template trt-network -o my_net.py
→ 生成包含 load_network() 框架的脚本，内部是空网络。

2. 从 ONNX 生成可修改的网络脚本（推荐！）
polygraphy template trt-network identity.onnx -o my_net.py
→ 自动生成：

解析 identity.onnx 的代码：
返回 (builder, network, parser)；你只需在解析后插入自定义逻辑（如加层、改属性）
3. 生成带预设配置的 config 脚本
polygraphy template trt-config --fp16 --workspace=2G -o my_config.py
→ 自动生成启用 FP16 和 2GB workspace 的配置脚本。

 这是最佳实践：“从 ONNX 生成模板 → 修改网络 → 用 CLI 直接运行”；无需手动处理 builder/network 生命周期管理。

 五、典型应用场景
场景	              实现方式
添加自定义插件层	   在 load_network() 中调用 network.add_plugin_v2(...)
修改层属性（如精度）	遍历 network 的 layers，设置 layer.precision = trt.float16
动态 shape 配置	      在 load_config() 中添加 optimization profile
INT8 校准	  在 load_config() 中设置 config.int8_calibrator = MyCalibrator()
多输入/输出重命名	在 load_network() 中调用 mark_input/mark_output

示例：为所有卷积层强制 FP16
def load_network():
    builder, network, parser = CreateNetwork().call_func()
    parser.parse("model.onnx")
    
    # 遍历所有层
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if layer.type == trt.LayerType.CONVOLUTION:
            layer.precision = trt.float16
    
    return builder, network, parser

 六、CLI 使用技巧
1. 同一脚本包含 load_network 和 load_config
# all_in_one.py
def load_network(): ...
def load_config(builder, network): ...
polygraphy run --trt all_in_one.py --model-type=trt-network-script
# config 会自动被发现并使用
2. 与 polygraphy convert 结合（只构建 engine）
# 构建 engine 而不运行推理
polygraphy convert define_network.py --model-type=trt-network-script -o model.engine
3. 指定输入 shape（动态 shape 必需）
polygraphy run ... --trt-min-shapes X:[1,3,224,224] \
                   --trt-opt-shapes X:[4,3,224,224] \
                   --trt-max-shapes X:[8,3,224,224]
注意：这些 shape 信息会传递给 load_config()，你可在其中创建 profile。

 七、注意事项
脚本必须可导入
不能有顶层代码（如 print("hello")），否则 CLI 导入时会执行
所有逻辑应封装在 load_network/load_config 中
资源管理
Polygraphy 会自动管理 builder/network 生命周期，不要手动 destroy
错误调试：若脚本报错，CLI 会打印完整 traceback，便于定位
与 ONNX 的兼容性：修改网络后，输入/输出名或 shape 可能变化，需确保后续处理匹配

 八、总结：这份文档的工程价值
维度	    说明
打破壁垒	让动态构建的 TensorRT 网络也能享受 CLI 工具的便利（run/convert/debug）
提升效率	通过 template 命令自动生成样板代码，避免重复劳动
增强灵活性	支持任意网络修改 + 任意配置定制，同时保持命令行工作流
促进标准化	提供统一接口（load_network/load_config），便于团队协作和脚本复用

 终极工作流建议：
1. 用 polygraphy template trt-network model.onnx -o net.py 生成模板
2. 在 net.py 中添加自定义逻辑
3. 用 polygraphy run --trt net.py --model-type=trt-network-script 直接验证
4. 用 polygraphy convert ... -o engine 构建最终部署文件
这套机制使得 “复杂网络定制” 与 “简单命令行验证” 不再矛盾，是 TensorRT 高级用户的必备技能。

