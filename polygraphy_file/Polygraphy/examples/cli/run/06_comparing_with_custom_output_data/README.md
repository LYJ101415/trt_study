# Comparing With Custom Output Data

## Introduction

In some cases, it may be useful to compare against output values generated outside Polygraphy.
The simplest way to do so is to create a `RunResults` object and save it to a file.

This example illustrates how you can generate custom input and output data outside of Polygraphy
and seamlessly load it into Polygraphy for comparison.

## Running The Example

1. Generate the input and output data:

    ```bash
    python3 generate_data.py
    ```

2. **[Optional]** Inspect the data.
    For inputs:

    ```bash
    polygraphy inspect data custom_inputs.json
    ```

    For outputs:

    ```bash
    polygraphy inspect data custom_outputs.json
    ```

3. Run inference with the generated input data and then compare outputs against the custom outputs:

    ```bash
    polygraphy run identity.onnx --trt \
        --load-inputs custom_inputs.json \
        --load-outputs custom_outputs.json
    ```

## Further Reading

For details on how to access and work with the outputs stored in `RunResults` objects
using the Python API, refer to [API example 08](../../../api/08_working_with_run_results_and_saved_inputs_manually/).
=============================================================
这份文档《Comparing With Custom Output Data》展示了 如何将外部生成的“黄金标准”输出（Golden Outputs）引入 Polygraphy，用于验证 TensorRT 或其他后端的推理结果是否正确。这是模型部署中精度验证和回归测试的关键环节。

下面我将从设计思想、核心机制、代码实现、使用流程、典型场景和工程价值六个维度，为你深度解析。
 一、解决的核心问题
 为什么需要“自定义输出数据”？默认情况下，polygraphy run --trt --onnxrt 会自动对比两个后端的输出;但在以下场景中，这不够用：;没有参考后端（如只有 TRT 引擎，无 ONNX 模型）;需要与“权威结果”对比（如 PyTorch 训练脚本输出、硬件实测值、人工标注）;构建 CI/CD 回归测试：确保新版本引擎不破坏已有精度;调试特定样本：已知某输入应产生某输出，验证是否符合
 Polygraphy 的解决方案：允许用户预先保存期望的输出（称为 RunResults），并在运行时加载进行对比。
 二、核心机制：RunResults 对象
什么是 RunResults？Polygraphy 内部用于表示一次或多次推理结果的数据结构;本质是：List[Dict[str, np.ndarray]];外层 List：对应多组输入（batch of samples）;内层 Dict：key=输出张量名，value=NumPy 数组;如何保存/加载？
保存：用 save_json(run_results, "outputs.json")
加载：CLI 通过 --load-outputs outputs.json 自动反序列化

 关键点：RunResults 的格式与 --load-inputs 的输入格式完全对称，只是用途不同。

 三、完整工作流解析（结合 generate_data.py）
虽然文档未给出 generate_data.py 源码，但根据上下文可推断其典型实现：
推测的 generate_data.py 内容
import numpy as np
from polygraphy.json import save_json

# 1. 生成输入数据（与模型输入兼容）
input_data = [{"x": np.random.randn(1, 3, 224, 224).astype(np.float32)}]

# 2. 用“权威来源”生成输出（例如 PyTorch 模型）
#    假设我们有一个 pytorch_model，输入 input_data[0]["x"]
#    得到输出 output_tensor
output_tensor = ...  # 来自 PyTorch / MATLAB / 真实设备等

# 3. 构造 RunResults 格式：List[Dict]
output_data = [{"output": output_tensor}]  # 注意："output" 必须匹配模型输出名

# 4. 保存输入和输出
save_json(input_data, "custom_inputs.json")
save_json(output_data, "custom_outputs.json")
用户操作流程
步骤	    命令	               作用
1	python3 generate_data.py	生成 inputs.json + outputs.json
2	polygraphy inspect data custom_inputs.json	 可选：检查输入内容
3	polygraphy inspect data custom_outputs.json	 可选：检查输出内容
4	polygraphy run ... --load-inputs ... --load-outputs ...	  核心：用自定义输入跑 TRT，并与自定义输出对比

 四、实际应用示例
场景：验证 TensorRT 引擎 vs PyTorch “黄金标准”
Step 1: 用 PyTorch 生成黄金输出
# generate_golden.py
import torch
import numpy as np
from polygraphy.json import save_json

# 加载训练好的 PyTorch 模型
model = torch.load("model.pth").eval()

# 创建输入
inp = np.random.randn(1, 3, 224, 224).astype(np.float32)
torch_inp = torch.from_numpy(inp)

# 获取 PyTorch 输出（黄金标准）
with torch.no_grad():
    golden_out = model(torch_inp).numpy()  # shape: [1, 1000]

# 保存
save_json([{"input": inp}], "golden_inputs.json")
save_json([{"output": golden_out}], "golden_outputs.json")
Step 2: 用 Polygraphy 验证 TRT 引擎
Bash
编辑
# 假设已将 PyTorch 模型转为 identity.onnx
polygraphy run identity.onnx --trt \
    --load-inputs golden_inputs.json \
    --load-outputs golden_outputs.json \
    --atol 1e-3 --rtol 1e-3

 成功条件：TRT 输出与 PyTorch 输出在容忍误差内一致
 失败：Polygraphy 会打印详细误差报告（max abs error, mean rel error 等）

 五、高级用法与注意事项
1. 多输出模型支持
# RunResults 可包含多个输出张量
output_data = [{
    "boxes": np.array(...),
    "scores": np.array(...),
    "labels": np.array(...)
}]
2. 多样本批量验证
# 生成 100 个样本的黄金输出
output_data = []
for i in range(100):
    out = run_golden_model(inputs[i])
    output_data.append({"output": out})

save_json(output_data, "100_samples_golden.json")
→ Polygraphy 会逐样本对比，并汇总整体误差。

3. 动态形状兼容性
自定义输出的 shape 必须与实际推理输出 shape 一致;若模型有动态输出（如检测框数量可变），需确保：黄金输出 shape 与输入对应的 TRT 输出 shape 匹配;或使用 padding 统一 shape（需后处理对齐）
4. 精度容忍度控制
# 默认 atol=1e-5, rtol=1e-5，可调整
polygraphy run ... --atol 1e-2 --rtol 1e-2
对 FP16 模型，通常需放宽容忍度（如 atol=1e-2）

 六、与其他对比方式的对比
对比方式	                适用场景	    优点	          缺点
--onnxrt（自动对比）	 有 ONNX 模型	 无需额外步骤	 依赖 ONNX Runtime 精度
--load-outputs（自定义输出）	有黄金标准/无参考后端	灵活、权威、可 CI	需预先生成输出
手动写 Python 脚本	        复杂逻辑	  完全控制	     开发成本高

 --load-outputs 是工业级部署的必备技能：它解耦了“结果生成”和“结果验证”，使验证流程标准化、自动化。

 七、总结：这份文档的工程价值
维度	    说明
验证闭环	实现“外部生成 → 内部验证”的完整精度保障流程
CI/CD 友好	JSON 文件可纳入版本控制，实现自动化回归测试
多后端支持	不仅限 TRT，也适用于 TensorRT-LLM、CUDA kernels 等
调试利器	快速定位“是模型问题还是推理引擎问题”

 记住这个模式：
1. 用权威工具生成 golden_outputs.json
2. 用 polygraphy run ... --load-outputs golden_outputs.json 验证部署模型
这种做法在 NVIDIA、Tesla、Amazon 等公司的模型部署 pipeline 中被广泛采用，是专业 MLOps 实践的标准组成部分。

