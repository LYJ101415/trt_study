使用 Extract 隔离子图
简介
surgeon extract 子工具可以通过单条命令从模型中提取子图。
在本例中，我们将从一个计算 Y = x0 + (a * x1 + b) 的模型中提取子图：
假设我们希望隔离出计算 (a * x1 + b) 的子图，并且已经使用 polygraphy inspect model model.onnx --show layers 确定了该子图输入/输出张量的名称，但不知道其中任何张量的形状或数据类型。
当形状和数据类型未知时，可以使用 auto 指示 Polygraphy 尝试自动确定这些信息。
对于输入，必须同时指定形状和数据类型；而对于输出，则只需指定数据类型——因此 --inputs 需要两个 auto，而 --outputs 只需要一个。
运行示例:
提取子图：
polygraphy surgeon extract model.onnx \
    --inputs x1:auto:auto \
    --outputs add_out:auto \
    -o subgraph.onnx

如果我们已知形状和/或数据类型，则可以这样写，例如：
polygraphy surgeon extract model.onnx \
    --inputs x1:[1,3,224,224]:float32 \
    --outputs add_out:float32 \
    -o subgraph.onnx

生成的子图如下所示：
[可选] 此时，模型已可使用。您可以使用 inspect model 来确认其是否正确：
polygraphy inspect model subgraph.onnx --show layers

关于 auto 的说明
当将 auto 指定为形状或数据类型时，Polygraphy 会依赖 ONNX 形状推断来确定中间张量的形状和数据类型。
如果 ONNX 形状推断无法确定形状，Polygraphy 将使用合成输入数据通过 ONNX-Runtime 对模型运行推理。
您可以使用 --model-inputs 参数控制此输入数据的形状，并使用 Data Loader 选项控制其内容。
这将导致生成的子图的输入具有固定形状。您可以通过对子图再次使用 extract 命令并指定相同的输入，但使用包含动态维度的形状（例如 --inputs identity_out_0:[-1,-1]:auto），将其改回动态形状。
