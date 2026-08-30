检查 TensorRT 网络
简介
inspect model 子工具可以自动将支持的格式转换为 TensorRT 网络，并将其显示出来。
运行示例:
解析 ONNX 模型后显示 TensorRT 网络：
polygraphy inspect model identity.onnx \
    --show layers --display-as=trt

这将显示类似如下的内容：
[I] ==== TensorRT Network ====
    Name: Unnamed Network 0 | Explicit Batch Network

    ---- 1 Network Input(s) ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Network Output(s) ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Layer(s) ----
    Layer 0    | node_of_y [Op: LayerType.IDENTITY]
        {x [dtype=float32, shape=(1, 1, 2, 2)]}
         -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
         
也可以使用 --show layers attrs weights 来显示详细的层信息，包括层属性和权重。