在 TensorRT 中处理动态形状模型
简介
要在 TensorRT 中使用动态输入形状，我们必须在构建引擎时指定一个或多个可能的形状范围。有关其工作原理的详细信息，请参阅 API 示例 07。
在使用命令行接口（CLI）时，我们可以为每个输入指定一次或多次最小、最优和最大形状。如果为同一输入指定了多组形状，则会创建多个优化配置文件（optimization profiles）。

运行示例:
构建包含 3 个独立配置文件的引擎：
polygraphy convert dynamic_identity.onnx -o dynamic_identity.engine \
    --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[1,3,28,28] --trt-max-shapes X:[1,3,28,28] \
    --trt-min-shapes X:[1,3,28,28] --trt-opt-shapes X:[4,3,28,28] --trt-max-shapes X:[32,3,28,28] \
    --trt-min-shapes X:[128,3,28,28] --trt-opt-shapes X:[128,3,28,28] --trt-max-shapes X:[128,3,28,28]
对于具有多个输入的模型，只需为每个 --trt-*-shapes 参数提供多个参数即可。例如：--trt-min-shapes input0:[10,10] input1:[10,10] input2:[10,10] ...
提示：如果我们只想使用单个配置文件且 min == opt == max，可以利用运行时输入形状选项 --input-shapes 作为便捷的简写方式，而无需分别设置 min/opt/max。

[可选] 检查生成的引擎：
polygraphy inspect model dynamic_identity.engine

延伸阅读
有关在 TensorRT 中使用动态形状的更多信息，请参阅开发者指南。
How does the timing cache improve engine builds?
What are alternatives to using a timing cache?
Can determinism impact performance in TensorRT?