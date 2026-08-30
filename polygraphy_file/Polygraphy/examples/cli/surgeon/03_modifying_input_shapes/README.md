# Modifying Input Shapes


## Introduction

The `surgeon sanitize` subtool can be used to modify the input shapes of an ONNX model.
This does not change the intermediate layers of the model, and as such, may cause issues if
the model makes assumptions about the input shapes (for example, a `Reshape` node with a hard-coded
new shape).

Output shapes can be inferred and so these are not modified (nor do they need to be).

*NOTE: Re-exporting the ONNX model with the desired shapes is strongly recommended.*
    *The method shown here should only be used when doing so is not possible.*

## Running The Example

1. Change the input shape of the model to a shape with a dynamic batch dimension,
    keeping other dimensions the same:

    ```bash
    polygraphy surgeon sanitize identity.onnx \
        --override-input-shapes x:['batch',1,2,2] \
        -o dynamic_identity.onnx
    ```

2. **[Optional]** You can use `inspect model` to confirm whether it looks correct:

    ```bash
    polygraphy inspect model dynamic_identity.onnx --show layers
    ```
==================================================================
这份文件是关于如何使用 Polygraphy 工具中的 surgeon sanitize 子命令来修改 ONNX 模型的输入形状（input shapes） 的说明文档。以下是对该文件的详细解析：

一、背景与目的
1. 核心功能：使用 polygraphy surgeon sanitize 命令可以修改 ONNX 模型的输入张量形状。
此操作不会改动模型内部的计算图结构（即中间层保持不变）。
2. 使用场景：当你无法重新导出（re-export）ONNX 模型（例如原始训练代码不可用、导出工具受限等），但又需要调整输入形状（如支持动态 batch）时，可采用此方法作为“应急手段”。
 重要提示：官方强烈建议优先通过重新导出 ONNX 模型来设置正确的输入形状。仅在无法重新导出时才使用 surgeon sanitize。

二、技术细节与潜在风险
1. 不修改中间层：该工具只改输入节点的 shape 信息，不更新后续依赖固定形状的操作（如 Reshape, Slice, Gather 等）。如果模型中存在硬编码的形状假设（例如 Reshape 节点指定了 [1, 64, 8, 8] 这样的固定维度），那么修改输入形状后可能导致：推理失败（shape mismatch）
输出结果错误
2. 输出形状无需手动修改：ONNX 支持形状推断（shape inference），因此输出形状会在加载或运行时自动推导，无需用户干预。
三、操作示例详解：示例命令：
polygraphy surgeon sanitize identity.onnx \
    --override-input-shapes x:['batch',1,2,2] \
    -o dynamic_identity.onnx
参数解析：
参数	            含义
identity.onnx	原始 ONNX 模型文件
--override-input-shapes x:['batch',1,2,2]	将名为 x 的输入张量的形状改为 [batch, 1, 2, 2]，其中 'batch' 是一个符号名（symbolic dimension），表示该维度是动态的（可在推理时指定任意正整数）
-o dynamic_identity.onnx	输出修改后的新模型文件

 符号维度（如 'batch'）是 ONNX 中表示动态维度的标准方式，推理引擎（如 TensorRT、ONNX Runtime）会据此生成支持变长 batch 的执行计划。

可选验证步骤：
polygraphy inspect model dynamic_identity.onnx --show layers
使用 polygraphy inspect model 查看模型结构，确认输入形状是否已成功修改为 ['batch', 1, 2, 2]。
--show layers 会显示每一层的输入/输出形状，便于人工核对。

四、最佳实践建议：优先重新导出模型；在 PyTorch/TensorFlow 中导出 ONNX 时，直接使用动态轴（如 dynamic_axes 参数）指定可变维度，这是最安全可靠的方式。
谨慎使用 surgeon sanitize

仅在以下情况使用：仅有 .onnx 文件，无源代码；第三方提供的模型无法重新生成；快速原型验证（非生产环境）；验证修改后的模型
务必使用不同 batch size 进行推理测试，确保：无 shape 相关错误；输出数值正确；注意符号命名一致性

若模型有多个输入需共享同一动态维度（如 batch），应使用相同的符号名（如都用 'batch'），否则推理引擎可能无法正确处理。
五、总结
项目	内容
工具	Polygraphy (surgeon sanitize)
功能	修改 ONNX 模型输入形状（支持动态维度）
风险	可能破坏模型内部形状依赖，导致运行错误
替代方案	重新导出 ONNX 模型（推荐）
验证方式	polygraphy inspect model --show layers
 一句话总结：surgeon sanitize 是一个“外科手术式”的 ONNX 输入形状修改工具，强大但需谨慎使用，务必配合充分测试。