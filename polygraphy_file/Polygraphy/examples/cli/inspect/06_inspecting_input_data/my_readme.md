检查输入数据
简介
inspect data 子工具可以显示由数据加载器生成的输入数据的相关信息。
运行示例:
通过运行推理生成一些输入数据：
polygraphy run identity.onnx --onnxrt --save-inputs inputs.json

检查输入数据：
polygraphy inspect data inputs.json --show-values

这将显示类似如下的内容：
[I] ==== Data (1 iterations) ====

    x [dtype=float32, shape=(1, 1, 2, 2)] | Stats: mean=0.35995, std-dev=0.25784, var=0.066482, median=0.35968, min=0.00011437 at (0, 0, 1, 0), max=0.72032 at (0, 0, 0, 1), avg-magnitude=0.35995, p90=0.62933, p95=0.62933, p99=0.71123
        [[[[4.17021990e-01 7.20324516e-01]
           [1.14374816e-04 3.02332580e-01]]]]