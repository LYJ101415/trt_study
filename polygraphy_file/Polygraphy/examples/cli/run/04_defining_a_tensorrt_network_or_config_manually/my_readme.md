手动定义 TensorRT 网络或配置
简介
在某些情况下，使用 Python API 从头开始定义 TensorRT 网络，或修改通过其他方式（例如解析器）创建的网络可能非常有用。通常，这会限制您使用 CLI 工具，至少在您构建引擎之前是如此，因为网络无法序列化到磁盘并在命令行中加载。
Polygraphy CLI 工具为此提供了一种解决方法——如果您的 Python 脚本定义了一个名为 load_network 的函数，该函数不接受任何参数并返回一个 TensorRT builder、network 以及可选的 parser，那么您就可以用此 Python 脚本替代模型参数。
同样，我们可以通过脚本创建自定义 TensorRT builder 配置，该脚本需定义一个名为 load_config 的函数，它接受 builder 和 network 作为参数并返回 builder 配置。
在本例中，附带的 define_network.py 脚本解析了一个 ONNX 模型并向其追加了一个 identity 层。由于它在名为 load_network 的函数中返回了 builder、network 和 parser，我们仅需一条命令即可从中构建并运行 TensorRT 引擎。create_config.py 脚本则创建了一个新的 TensorRT builder 配置并启用了 FP16 模式。
提示：自动生成脚本模板
您无需从头编写网络脚本，可以使用 polygraphy template trt-network 来获取一个起点：
polygraphy template trt-network -o my_define_network.py

如果您希望从现有模型开始并修改生成的 TensorRT 网络，而不是从头创建，只需将模型作为参数提供给 template trt-network：
polygraphy template trt-network identity.onnx -o my_define_network.py

同样，您可以使用 polygraphy template trt-config 生成配置的模板脚本：
polygraphy template trt-config -o my_create_config.py

您还可以指定 builder 配置选项来预填充脚本内容。例如，启用 FP16 模式：
polygraphy template trt-config --fp16 -o my_create_config.py

运行示例:
运行 define_network.py 中定义的网络：
polygraphy run --trt define_network.py --model-type=trt-network-script

使用 create_config.py 中定义的 builder 配置运行步骤 (1) 中的网络：
polygraphy run --trt define_network.py --model-type=trt-network-script --trt-config-script=create_config.py

请注意，我们本可以在同一个脚本中同时定义 load_network 和 load_config。实际上，我们甚至可以从任意脚本或模块中获取这些函数。
提示：我们可以对 polygraphy convert 使用相同的方法来仅构建引擎而不运行它。