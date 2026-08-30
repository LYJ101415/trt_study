扩展 polygraphy run
简介
polygraphy run 允许您使用多个后端（包括 TensorRT 和 ONNX-Runtime）运行推理，并比较输出结果。
虽然它确实提供了加载自定义输出并与不受支持的后端进行比较的机制，
但通过扩展模块添加对该后端的支持可以使其更无缝地集成，
从而提供更好的用户体验。
在本例中，我们将为 polygraphy run 创建一个名为 polygraphy_reshape_destroyer 的扩展模块，
该模块将包含以下内容：
一个特殊的加载器（Loader），用于将 ONNX 模型中的空操作 Reshape 节点替换为 Identity 节点。
一个自定义运行器（Runner），支持仅包含 Identity 节点的 ONNX 模型。
命令行选项，用于：
在加载器应用变换时启用或禁用节点重命名。
以 slow（慢速）、medium（中速）或 fast（快速）模式运行模型。
在 slow 和 medium 模式下，我们将在推理过程中注入 time.sleep()
（这将在 fast 模式下带来巨大的性能提升！）。
背景知识
尽管本示例是自包含的，并且会在您遇到相关概念时进行解释，但仍建议您先熟悉
Polygraphy 的 `Loader` 和 `Runner` API、
`Argument Group` 接口
以及 `Script` 接口。
在此基础上，为 polygraphy run 创建扩展模块只需定义您的
自定义 Loader/Runner 和 Argument Group，并通过
setuptools 的 entry_points API 使它们对 Polygraphy 可见即可。
注意：严格来说，定义自定义 Loader 并非必需，但为了完整性，本示例仍将涵盖此内容。
按照惯例，Polygraphy 扩展模块的名称需以 polygraphy_ 为前缀。
阅读示例代码
我们构建示例扩展模块的方式在一定程度上镜像了 Polygraphy 仓库的结构。
这应该能让您更容易看出扩展模块中的功能与 Polygraphy 原生提供的功能之间的对应关系。
结构如下：
- extension_module/
    - polygraphy_reshape_destroyer/
        - backend/
            - __init__.py   # 控制子模块级别的导出
            - loader.py     # 定义我们的自定义加载器。
            - runner.py     # 定义我们的自定义运行器。
        - args/
            - __init__.py   # 控制子模块级别的导出
            - loader.py     # 为我们的自定义加载器定义命令行参数组。
            - runner.py     # 为我们的自定义运行器定义命令行参数组。
        - __init__.py       # 控制模块级别的导出
        - export.py         # 定义 `polygraphy run` 的入口点。
    - setup.py              # 构建我们的模块
    
建议按以下顺序阅读这些文件：
backend/loader.py
backend/runner.py
backend/\_\_init\_\_.py
args/loader.py
args/runner.py
args/\_\_init\_\_.py
\_\_init\_\_.py
export.py
setup.py

运行示例:
构建并安装扩展模块：
使用 setup.py 构建：
python3 extension_module/setup.py bdist_wheel

安装 wheel 包：
python3 -m pip install extension_module/dist/polygraphy_reshape_destroyer-0.0.1-py3-none-any.whl \
    --extra-index-url https://pypi.ngc.nvidia.com

提示：如果您对示例扩展模块进行了修改，可以通过重新构建（按照步骤 1）然后运行以下命令来更新已安装的版本：
python3 -m pip install extension_module/dist/polygraphy_reshape_destroyer-0.0.1-py3-none-any.whl \
    --force-reinstall --no-deps

安装扩展模块后，您应该能在 polygraphy run 的帮助输出中看到您添加的选项：
polygraphy run -h

接下来，我们可以用一个包含空操作 Reshape 的 ONNX 模型来试用自定义运行器：
polygraphy run no_op_reshape.onnx --res-des

我们还可以尝试添加的其他命令行选项：
重命名被替换的节点：
polygraphy run no_op_reshape.onnx --res-des --res-des-rename-nodes

不同的推理速度：
polygraphy run no_op_reshape.onnx --res-des --res-des-speed=slow

polygraphy run no_op_reshape.onnx --res-des --res-des-speed=medium

polygraphy run no_op_reshape.onnx --res-des --res-des-speed=fast

最后，让我们将我们的实现与 ONNX-Runtime 进行比较，以确保其功能正确：
polygraphy run no_op_reshape.onnx --res-des --onnxrt