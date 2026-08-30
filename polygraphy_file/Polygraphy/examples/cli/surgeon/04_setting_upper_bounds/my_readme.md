使用 Sanitize 为无界数据依赖形状 (DDS) 设置上限
简介
surgeon sanitize 子工具可用于为无界数据依赖形状（Data-Dependent Shapes, DDS）设置上限。
当张量的形状取决于另一个张量的运行时值时，这种形状被称为 DDS。
某些 DDS 具有有限的上限。例如，NonZero 算子的输出形状是一种 DDS，但其输出形状不会超过其输入的形状。
而另一些 DDS 则没有上限。例如，当 Range 算子的 limit 输入是运行时张量时，其输出就是一个无界 DDS。
对于具有无界 DDS 的张量，TensorRT 难以在构建阶段优化推理性能和内存使用。
在最坏的情况下，它们甚至会导致 TensorRT 引擎构建失败。
在本例中，我们将使用 Polygraphy 为图中的无界 DDS 设置上限：
运行示例:
首先对模型执行常量折叠：
polygraphy surgeon sanitize model.onnx -o folded.onnx --fold-constants

请注意，列出无界 DDS 以及为其设置上限均需要依赖常量折叠和符号形状推断。
查找具有无界 DDS 的张量：
polygraphy inspect model folded.onnx --list-unbounded-dds

Polygraphy 将显示所有具有无界 DDS 的张量。
为无界 DDS 设置上限：
polygraphy surgeon sanitize folded.onnx --set-unbounded-dds-upper-bound 1000 -o modified.onnx 

Polygraphy 会首先搜索所有具有无界 DDS 的张量。
然后，它会插入带有指定上限值的 Min 算子，以限制 DDS 张量的大小。
在本例中，一个 Min 算子被插入到了 Range 算子之前。
通过修改后的模型，TensorRT 将得知 Range 算子的输出形状不会超过 1000。
因此，后续的层可以选择更多的内核实现。
检查确认现在已不存在具有无界 DDS 的张量：
polygraphy inspect model modified.onnx --list-unbounded-dds
此时，modified.onnx 中应不再包含任何无界 DDS。