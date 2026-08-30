检查 ONNX 模型
简介
inspect model 子工具可以显示 ONNX 模型。
运行示例:
检查 ONNX 模型：
polygraphy inspect model identity.onnx --show layers

这将显示类似如下的内容：
[I] ==== ONNX Model ====
    Name: test_identity | ONNX Opset: 8

    ---- 1 Graph Input(s) ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Graph Output(s) ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 0 Initializer(s) ----
    {}

    ---- 1 Node(s) ----
    Node 0    |  [Op: Identity]
        {x [dtype=float32, shape=(1, 1, 2, 2)]}
         -> {y [dtype=float32, shape=(1, 1, 2, 2)]}

也可以使用 --show layers attrs weights 来显示详细的层信息，包括层属性和权重。