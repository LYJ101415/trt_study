
# Generating A Script For Advanced Comparisons


## Introduction

For more advanced requirements, you may want to use the [API](../../../../polygraphy).
Instead of writing a script from scratch, you can use `run`'s `--gen-script` option
to create a Python script that you can use as a starting point.


## Running The Example

1. Generate a comparison script:

    ```bash
    polygraphy run identity.onnx --trt --onnxrt \
        --gen-script=compare_trt_onnxrt.py
    ```

    The generated script will do exactly what the `run` command would otherwise do.

2. Run the comparison script, optionally after modifying it:

    ```bash
    python3 compare_trt_onnxrt.py
    ```
=========================================================================
这份文档《Generating A Script For Advanced Comparisons》是 Polygraphy 工具链中一个极其高效且实用的功能指南，核心思想是：

“用命令行快速生成可定制的 Python 脚本，作为高级精度对比或调试任务的起点。”

下面我将从设计动机、技术原理、使用流程、生成内容解析、典型应用场景和工程价值六个维度，为你深度剖析。
 一、为什么需要这个功能？—— 设计动机
 问题背景:polygraphy run 命令虽然强大，但灵活性有限。当你需要：自定义输入数据（如从文件读取特定图像）；修改 TensorRT 构建配置（如自定义插件、优化策略）；添加日志、可视化、性能分析；循环测试多组输入；与现有 Python 代码集成。
→ 命令行参数无法满足，必须写 Python 脚本
 Polygraphy 的解决方案：“不要从零开始写脚本，而是让 Polygraphy 自动生成一个功能完整的模板！”；这体现了 “渐进式复杂度” 的工程哲学：

简单任务 → 用命令行；复杂任务 → 基于命令行生成脚本再修改
 二、核心技术机制：--gen-script
命令格式：
polygraphy run <model> [framework flags] --gen-script=<output.py>
执行过程：Polygraphy 解析命令行参数；模拟执行 整个推理+对比流程（但不实际运行）；反向生成 对应的 Python 代码，调用 Polygraphy API；将代码写入指定 .py 文件。

 关键特性：生成的脚本 = 命令行行为的 1:1 Python 实现；运行该脚本的结果，与直接运行原命令完全一致

 三、生成脚本内容深度解析（以示例为例）
假设执行：
polygraphy run identity.onnx --trt --onnxrt --gen-script=compare_trt_onnxrt.py
生成的 compare_trt_onnxrt.py 通常包含以下模块：
1 导入 Polygraphy 核心模块：
from polygraphy import constants, mod, util
from polygraphy.backend.trt import TrtRunner, CreateConfig
from polygraphy.backend.onnxrt import OnnxrtRunner
from polygraphy.comparator import Comparator, CompareFunc
from polygraphy.json import save_json
import numpy as np
2 定义模型加载函数：
def load_onnx_model():
    # 加载 ONNX 模型
    from polygraphy.backend.onnx import OnnxFromPath
    return OnnxFromPath("identity.onnx")()
3 定义 TensorRT 构建配置：
def create_trt_config():
    # 创建 TRT builder config（可能包含 FP16/INT8 设置）
    return CreateConfig()  # 若命令含 --fp16，则此处会启用 fp16_mode
4 定义 Runner（执行器）
# ONNX Runtime Runner
onnx_runner = OnnxrtRunner(load_onnx_model())

# TensorRT Runner
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath
trt_network = NetworkFromOnnxPath("identity.onnx")
trt_engine = EngineFromNetwork(trt_network, config=create_trt_config())
trt_runner = TrtRunner(trt_engine)
5 执行推理并对比：
# 生成输入数据（与命令行相同逻辑）
input_data = {"X": np.ones((1, 2, 1, 1), dtype=np.float32)}  # 动态维度被设为1

# 运行两个后端
with onnx_runner, trt_runner:
    onnx_outputs = onnx_runner.infer(input_data)
    trt_outputs = trt_runner.infer(input_data)

# 使用 Comparator 对比
comparator = Comparator([onnx_outputs, trt_outputs])
comparator.set_tolerance(atol=1e-5, rtol=1e-5)
comparator.check()
6 错误处理与报告：
if not comparator.check():
    print("Comparison failed!")
    # 可能包含详细误差打印

 注意：生成的脚本是完整、可独立运行的，不依赖原始命令行环境。

 四、典型使用流程（工作流）
Step 1: 用命令行验证基础行为
# 先确保命令行能跑通
polygraphy run model.onnx --trt --onnxrt --input-shapes X:[1,3,224,224]
Step 2: 生成脚本
polygraphy run model.onnx --trt --onnxrt \
    --input-shapes X:[1,3,224,224] \
    --gen-script=advanced_compare.py
Step 3: 修改脚本以满足高级需求
例如：替换输入数据：
# 原始：input_data = {"X": np.random.randn(...)}
# 修改为：
input_data = {"X": load_image("test.jpg")}  # 自定义图像加载
添加性能计时：
import time
start = time.time()
outputs = runner.infer(input_data)
print(f"Inference time: {time.time() - start:.3f}s")
自定义容差 per output：
comparator.set_tolerance(output_name="scores", atol=0.1)
comparator.set_tolerance(output_name="boxes", atol=1.0)
Step 4: 运行定制化脚本
python3 advanced_compare.py

 五、高级应用场景举例
需求	                 如何通过修改生成脚本实现
批量测试 100 张图片	    在脚本中加 for image in images: 循环
对比不同 TRT 精度	   复制 TRT runner 部分，分别构建 FP16/INT8 engine
保存中间层输出	      在 runner.infer 后添加 save_tensor(outputs["layer1"])
集成到 CI/CD	     脚本返回非零 exit code 表示失败，供自动化系统捕获
可视化误差热力图	 用 matplotlib 绘制 abs(out_trt - out_onnx)

 优势：你只需关注业务逻辑修改，无需重写底层推理框架集成代码。

 六、注意事项与最佳实践
1. 生成脚本是“快照”，不是“活模板”；修改后，若原命令变更（如换模型），需重新生成；
建议保留原始命令注释在脚本顶部：
# Generated from: polygraphy run model.onnx --trt --onnxrt ...
2. 动态 shape 处理；若命令含 --input-shapes，生成脚本会硬编码 shape；如需灵活 shape，需手动改为从外部传参
3. 依赖管理；脚本依赖 polygraphy 包，确保目标环境已安装；可用 pip install polygraphy 安装
4. 与 API 文档结合使用；生成脚本中的函数（如 CreateConfig, Comparator）均有详细 API 文档；参考 Polygraphy API Docs 进行深度定制

 七、总结：这份文档的核心价值
维度	    说明
降低门槛	让不熟悉 Polygraphy API 的用户也能快速上手高级功能
提升效率	避免重复编写框架集成样板代码（TRT/ONNXRT 初始化等）
保证正确性	生成的脚本经过 Polygraphy 官方验证，避免低级错误
促进探索	用户可通过阅读生成代码，学习 Polygraphy API 最佳实践
 
 终极建议：“先用命令行验证想法，再用 --gen-script 生成脚本进行扩展”；这是使用 Polygraphy 进行复杂模型验证的黄金工作流。如果你正在做 TensorRT 部署、精度调试或多模型对比，这个功能将极大提升你的开发效率和代码可靠性。

