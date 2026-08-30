# Validating Accuracy On A Real Dataset


## Introduction

The `Comparator` provided by Polygraphy can be useful for comparing a small number of
results across multiple runners, but is not well suited for validating a single runner
with a real dataset that includes labels or golden values - especially if the dataset is large.

In such cases, it is recommended to use a runner directly instead.

*NOTE: It is possible to provide custom input data to `Comparator.run()` using the `data_loader`*
    *parameter. This may be a viable option when using a smaller dataset.*

In this example, we use a `TrtRunner` directly to validate an identity model on
a trivial dataset. Unlike using the `Comparator`, using a runner gives you complete
freedom as to how you load your input data, as well as how you validate the results.

Since all runners provide the same interface, you can freely drop-in other runners
without touching the rest of your validation code. For example, in this case, validating
the model using ONNX-Runtime would require changing just 2 lines; this is left as an
exercise for the reader.


## Running The Example

1. Install prerequisites
    * Ensure that TensorRT is installed
    * Install other dependencies with `python3 -m pip install -r requirements.txt`

2. Run the example
    ```bash
    python3 example.py
    ```
=============================================================================
1.以上解析：
Polygraphy 的 Comparator 是一个用于比较多个推理后端（如 TensorRT、ONNX Runtime 等）输出结果是否一致的工具。
它适用于小规模、无标签的对比测试（例如验证不同引擎是否产生相同输出）。
但当面对真实数据集（real dataset）——即每个输入都有对应的真实标签（labels）或黄金标准输出（golden values）——尤其是数据量很大时，Comparator 并不是最佳选择。
原因：Comparator 设计初衷是做多后端一致性校验，而非与真实值进行精度评估（如计算准确率、误差等）。
2.推荐方案：直接使用 Runner
Runner 是 Polygraphy 中对各种推理后端（如 TensorRT、ONNX Runtime、PyTorch 等）的统一抽象接口。直接使用 TrtRunner（或其他 Runner）可以：
自由控制数据加载方式（如从磁盘读取图像、CSV、TFRecord 等）；
灵活定义验证逻辑（如计算 top-1 准确率、MSE、自定义指标等）；
高效处理大规模数据（可分批加载、流式处理，避免内存爆炸）。
3.最佳实践：
对于模型部署前的精度验证（如迁移 TensorRT 后是否掉点），应构建自己的验证脚本，使用 Runner + 自定义数据加载器 + 评估函数。
利用 Polygraphy 的统一 Runner 接口，轻松切换后端进行交叉验证。
4.鼓励用户：
理解 Polygraphy 的抽象设计；
实践“一次编写，多后端验证”的工程模式；
为后续自动化测试/CI 流程打下基础。
