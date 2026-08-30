使用自定义输出数据进行比较
简介
在某些情况下，将结果与在 Polygraphy 外部生成的输出值进行比较可能非常有用。
实现这一目标的最简单方法是创建一个 RunResults 对象并将其保存到文件中。
本示例说明了如何在 Polygraphy 外部生成自定义输入和输出数据，并将其无缝加载到 Polygraphy 中进行比较。
运行示例:
生成输入和输出数据：
python3 generate_data.py

[可选] 检查数据。
查看输入数据：
polygraphy inspect data custom_inputs.json

查看输出数据：
polygraphy inspect data custom_outputs.json

使用生成的输入数据运行推理，然后将输出与自定义输出进行比较：
polygraphy run identity.onnx --trt \
    --load-inputs custom_inputs.json \
    --load-outputs custom_outputs.json
延伸阅读
有关如何使用 Python API 访问和处理存储在 RunResults 对象中的输出的详细信息，请参阅 API 示例 08。
(../../../api/08_working_with_run_results_and_saved_inputs_manually/).