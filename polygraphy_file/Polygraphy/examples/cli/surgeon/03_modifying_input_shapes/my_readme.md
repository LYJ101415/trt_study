修改输入形状
简介
surgeon sanitize 子工具可用于修改 ONNX 模型的输入形状。
此操作不会更改模型的中间层，因此如果模型对输入形状有特定假设（例如，Reshape 节点中硬编码了新的形状），则可能会引发问题。
输出形状可以通过推断得出，因此无需修改（也不需要进行修改）。
注意：强烈建议使用所需的形状重新导出 ONNX 模型。
仅在无法重新导出的情况下，才应使用此处展示的方法。
运行示例:
将模型的输入形状更改为具有动态批次维度的形状，同时保持其他维度不变：
polygraphy surgeon sanitize identity.onnx \
    --override-input-shapes x:['batch',1,2,2] \
    -o dynamic_identity.onnx

[可选] 您可以使用 inspect model 来确认结果是否正确：
polygraphy inspect model dynamic_identity.onnx --show layers
