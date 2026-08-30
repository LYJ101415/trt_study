使用 Sanitize 折叠常量
简介
surgeon sanitize 子工具可用于折叠图中的常量、移除未使用的节点以及对节点进行拓扑排序。在形状已静态确定的情况下，它还可以简化涉及形状运算的子图。
在本例中，我们将对一个计算 output = input + ((a + b) + d) 的图进行常量折叠，其中 a、b 和 d 均为常量：
运行示例:
使用以下命令折叠常量：
polygraphy surgeon sanitize model.onnx \
    --fold-constants \
    -o folded.onnx

这会将 a、b 和 d 合并为一个常量张量，生成的图将计算 output = input + e：
提示：有时模型会包含诸如 Tile 或 ConstantOfShape 之类的算子，这些算子可能会生成大型常量张量。折叠这些张量可能会导致模型大小膨胀到不可接受的程度。您可以使用 --fold-size-threshold 来控制允许折叠的张量的最大字节数。任何生成超过此限制大小的张量的节点都不会被折叠，而是在运行时进行计算。
[可选] 您可以使用 inspect model 来确认结果是否正确：
polygraphy inspect model folded.onnx --show layers