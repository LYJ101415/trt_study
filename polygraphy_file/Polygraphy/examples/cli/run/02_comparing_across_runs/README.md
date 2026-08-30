# Comparing Across Runs

## Prerequisites
For a general overview of how to use `polygraphy run` to compare the outputs of
different frameworks, see the example on [Comparing Frameworks](../../../../examples/cli/run/01_comparing_frameworks).

## Introduction

There are situations where you may need to compare results across different invocations
of the `polygraphy run` command.  Some examples of this include:

* Comparing results across different platforms
* Comparing results across different versions of TensorRT
* Comparing different model types with compatible input(s)/output(s)

In this example, we'll demonstrate how to accomplish this with Polygraphy.

## Running The Example

### Comparing Across Runs

1. Save the input and output values from the first run:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-inputs inputs.json --save-outputs run_0_outputs.json
    ```

2. Run the model again, this time loading the saved inputs and outputs from
    the first run.  The saved inputs will be used as inputs for the current run, and
    the saved outputs will be used to compare against the first run.

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs.json --load-outputs run_0_outputs.json
    ```

    The `--atol/--rtol/--check-error-stat` options all work the same as in the
    [Comparing Frameworks](../../../../examples/cli/run/01_comparing_frameworks) example:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --load-inputs inputs.json --load-outputs run_0_outputs.json \
        --atol 0.001 --rtol 0.001 --check-error-stat median
    ```

### Comparing Different Models

We can also use this technique to compare different models, like TensorRT engines
and ONNX modles (if they have matching outputs).

1. Convert the ONNX model to a TensorRT engine and save it to disk:

    ```bash
    polygraphy convert identity.onnx -o identity.engine
    ```

2. Run the saved engine in Polygraphy, using the saved inputs from the ONNX-Runtime run as
    inputs to the engine, and compare the engine's outputs to the saved ONNX-Runtime outputs:

    ```bash
    polygraphy run --trt identity.engine --model-type=engine \
        --load-inputs inputs.json --load-outputs run_0_outputs.json
    ```


## Further Reading

For details on how to access and work with the saved outputs
using the Python API, refer to [API example 08](../../../api/08_working_with_run_results_and_saved_inputs_manually/).

For information on comparing against custom outputs, refer to [`run` example 06](../06_comparing_with_custom_output_data/).
========================================================================
这份文档《Comparing Across Runs》是 Polygraphy 工具链中关于 跨多次运行（cross-run）结果对比 的高级用法指南。它解决了这样一个核心问题：

“如何在不同时间、不同模型、不同环境之间，使用完全相同的输入数据进行公平、可复现的输出对比？”

下面我将从设计思想、技术实现、典型场景和工程价值四个维度，为你详细解析。

 一、核心目标与解决的问题
❓ 为什么需要“跨运行对比”？在模型部署和验证过程中，经常遇到以下场景：
想比较 TensorRT 引擎 vs 原始 ONNX 模型 的输出是否一致;想验证 升级 TensorRT 版本后 精度是否发生变化;想在 不同 GPU 平台 上运行同一模型，确认结果可复现;想对比 两个不同训练版本的模型（如 v1.onnx vs v2.onnx）;但这些对比有一个前提：必须使用完全相同的输入数据！

如果每次运行都用随机输入，即使模型完全相同，输出也会因输入不同而“看起来不一致”，导致误判。

 Polygraphy 的解决方案：通过 保存/加载输入输出数据，实现：

输入固定化 → 确保对比公平;输出基准化 → 第一次运行的结果作为“黄金标准”
 二、关键技术机制详解
1 保存输入与输出（第一次运行）
polygraphy run identity.onnx --onnxrt \
    --save-inputs inputs.json \
    --save-outputs run_0_outputs.json
内部发生了什么？
文件	         内容	                 格式
inputs.json	   自动生成的输入张量数据	JSON + base64 编码的 NumPy 数组
run_0_outputs.json	ONNX Runtime 的推理结果	   同上

 关键点：输入是 确定性生成 的（默认种子固定）;数据以 人类可读 + 机器可加载 的方式持久化
2 加载输入并对比输出（后续运行）
polygraphy run identity.onnx --onnxrt \
    --load-inputs inputs.json \
    --load-outputs run_0_outputs.json
执行流程：从 inputs.json 读取输入张量;用当前框架（如 ONNXRT）运行模型;将本次输出 与 run_0_outputs.json 中的基准输出 对比;使用默认或指定的容差（--atol/--rtol）判断是否通过
 优势：即使两次运行间隔数天、在不同机器上，只要加载相同输入，就能做精确对比。

3 扩展：对比不同模型类型（ONNX vs TRT Engine）
这是最典型的工业场景！步骤分解：
# Step 1: 用 ONNX Runtime 跑一次，保存 I/O
polygraphy run model.onnx --onnxrt \
    --save-inputs inputs.json --save-outputs onnx_outputs.json

# Step 2: 将 ONNX 转为 TensorRT Engine
polygraphy convert model.onnx -o model.engine

# Step 3: 用 TRT Engine 跑，加载 ONNX 的输入，对比 ONNX 的输出
polygraphy run model.engine --trt --model-type=engine \
    --load-inputs inputs.json \
    --load-outputs onnx_outputs.json
关键参数说明：
参数	                     作用
--model-type=engine	     告诉 Polygraphy 输入文件是 .engine 而非 .onnx
--trt	                 指定使用 TensorRT 后端
--load-inputs	        确保 TRT 和 ONNX 用完全相同的输入
--load-outputs	       将 ONNX 输出作为“正确答案”来验证 TRT

 这就是 TensorRT 部署前的标准验证流程！

 三、容差控制与误差分析（与单次运行一致）;跨运行对比完全继承了 polygraphy run 的误差分析能力：
polygraphy run ... \
    --load-inputs inputs.json \
    --load-outputs ref_outputs.json \
    --atol 1e-3 \
    --rtol 1e-3 \
    --check-error-stat median
--atol/--rtol：控制绝对/相对误差容忍度
--check-error-stat：选择误差统计方式（elemwise, max, median 等）
 一致性保证：无论单次对比还是跨运行对比，误差判断逻辑完全相同。

 四、典型应用场景总结
场景	        命令示例	                价值
TRT 引擎验证	polygraphy run engine --load-inputs ... --load-outputs onnx_out.json	确保转换无精度损失
版本回归测试	保存 v1 输出 → 测试 v2 是否一致	       防止升级引入 bug
跨平台验证	   在 A 机器保存输出 → 在 B 机器加载对比	确保部署可移植性
模型 A/B 测试	用相同输入跑 model_v1.onnx 和 model_v2.onnx	 客观评估模型改进效果

 五、数据格式与 API 扩展
数据存储格式:默认使用 JSON + base64 存储 NumPy 数组;也支持 Pickle（通过 --save-inputs inputs.pkl）;可被 Python 直接加载：
from polygraphy.json import load_json
inputs = load_json("inputs.json")

进阶用法指引:文档末尾提供了两个重要扩展方向：
Python API 控制:→ API Example 08
允许你在代码中手动加载/处理保存的数据，实现自定义对比逻辑。

自定义输出对比:→ run Example 06

支持将模型输出与任意外部数据（如 PyTorch 输出、真实标签）对比。
 六、工程最佳实践建议
 推荐工作流（TensorRT 部署验证）
# 1. 用 ONNX Runtime 生成黄金输入/输出
polygraphy run model.onnx --onnxrt \
    --save-inputs golden_inputs.json \
    --save-outputs golden_outputs.json

# 2. 转换为 TRT Engine
polygraphy convert model.onnx --fp16 -o model_fp16.engine

# 3. 验证 FP16 引擎
polygraphy run model_fp16.engine --trt --model-type=engine \
    --load-inputs golden_inputs.json \
    --load-outputs golden_outputs.json \
    --atol 1e-2 --rtol 1e-2  # FP16 允许更大误差
 注意事项
动态 shape 模型：需确保保存的输入 shape 在 TRT engine 的 profile 范围内
随机性操作（如 Dropout）：需在推理时关闭，否则无法复现
INT8 模型：校准过程也需固定，否则输出不可比
 七、总结：这份文档的核心价值
维度	     说明
问题定位	解决“跨环境/跨模型对比缺乏公平性”的痛点
方法论	    固定输入 + 基准输出 = 可复现验证
技术亮点	无缝衔接 ONNX / TRT Engine / 多次运行
工业意义	是 TensorRT 部署验证的标准流程 的官方推荐方案

 记住这个公式：可信的精度验证 = 相同输入 + 量化容差 + 可复现流程;而 polygraphy run --save/--load 正是实现这一公式的利器。如果你正在做模型部署、精度调试或版本回归测试，这份文档提供的模式将成为你日常工作的标准操作范式。

