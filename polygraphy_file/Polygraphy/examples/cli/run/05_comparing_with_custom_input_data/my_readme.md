使用自定义输入数据进行比较
简介
在某些情况下，我们可能希望使用自定义输入数据运行比较。
Polygraphy 提供了多种实现方式，详情见此处。
在本例中，我们将演示两种不同的方法：
使用数据加载器脚本：在 Python 脚本（data_loader.py）中定义 load_data() 函数。
Polygraphy 将在运行时调用 load_data() 来生成输入数据。
使用包含预生成输入的 JSON 文件。
为方便起见，我们将使用上述脚本（data_loader.py）将 load_data() 生成的输入保存到名为 custom_inputs.json 的文件中。
提示：通常在处理大量输入数据时，首选数据加载器脚本，因为它避免了写入磁盘的操作。另一方面，JSON 文件更具可移植性，有助于确保结果的可复现性。
最后，我们将把自定义输入数据提供给 polygraphy run，并比较 ONNX-Runtime 和 TensorRT 的输出。
由于我们的模型具有动态形状，因此需要设置 TensorRT 优化配置文件（Optimization Profile）。
有关如何通过命令行执行此操作的详细信息，请参阅 `convert` 示例 03。
为简单起见，我们将创建一个 min == opt == max 的配置文件。
注意：务必确保优化配置文件与自定义数据加载器提供的形状兼容。在我们这个非常简单的例子中，数据加载器始终生成形状为 (1, 2, 28, 28) 的输入，因此我们只需确保该形状落在 [min, max] 范围内即可。
运行示例
运行脚本以将输入数据保存到磁盘。
注意：仅当选择方案 2 时才需要执行此步骤。
python3 data_loader.py

使用自定义输入数据通过 TensorRT 和 ONNX-Runtime 运行模型：
方案 1：使用数据加载器脚本：
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --trt-min-shapes X:[1,2,28,28] --trt-opt-shapes X:[1,2,28,28] --trt-max-shapes X:[1,2,28,28] \
    --data-loader-script data_loader.py

方案 2：使用包含已保存输入的 JSON 文件：
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --trt-min-shapes X:[1,2,28,28] --trt-opt-shapes X:[1,2,28,28] --trt-max-shapes X:[1,2,28,28] \
    --load-inputs custom_inputs.json