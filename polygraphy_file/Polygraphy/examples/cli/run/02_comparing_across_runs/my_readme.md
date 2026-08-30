跨运行比较
前提条件
有关如何使用 polygraphy run 比较不同框架输出的总体概述，请参阅比较框架示例:
[Comparing Frameworks](../../../../examples/cli/run/01_comparing_frameworks).
简介
在某些情况下，您可能需要在不同的 polygraphy run 命令调用之间比较结果。此类场景包括：
跨不同平台比较结果
跨不同版本的 TensorRT 比较结果
比较具有兼容输入/输出的不同类型模型
在本例中，我们将演示如何使用 Polygraphy 实现这一目标。
运行示例:
跨运行比较
保存第一次运行的输入和输出值：
polygraphy run identity.onnx --onnxrt \
    --save-inputs inputs.json --save-outputs run_0_outputs.json

再次运行模型，这次加载第一次运行中保存的输入和输出。保存的输入将用作当前运行的输入，而保存的输出将用于与第一次运行进行比较。
polygraphy run identity.onnx --onnxrt \
    --load-inputs inputs.json --load-outputs run_0_outputs.json
--atol/--rtol/--check-error-stat 
选项的用法与比较框架示例中完全相同：
polygraphy run identity.onnx --onnxrt \
    --load-inputs inputs.json --load-outputs run_0_outputs.json \
    --atol 0.001 --rtol 0.001 --check-error-stat median

比较不同模型
我们还可以使用此技术来比较不同的模型，例如 TensorRT 引擎和 ONNX 模型（前提是它们具有匹配的输出）。
将 ONNX 模型转换为 TensorRT 引擎并保存到磁盘：
polygraphy convert identity.onnx -o identity.engine

在 Polygraphy 中运行保存的引擎，使用 ONNX-Runtime 运行中保存的输入作为引擎的输入，并将引擎的输出与保存的 ONNX-Runtime 输出进行比较：
polygraphy run --trt identity.engine --model-type=engine \
    --load-inputs inputs.json --load-outputs run_0_outputs.json

延伸阅读
有关如何使用 Python API 访问和处理已保存输出的详细信息，请参阅 API 示例 08。
(../../../api/08_working_with_run_results_and_saved_inputs_manually/).

有关与自定义输出进行比较的信息，请参阅 `run` 示例 06。
(../06_comparing_with_custom_output_data/).