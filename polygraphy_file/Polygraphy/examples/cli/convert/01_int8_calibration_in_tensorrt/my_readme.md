TensorRT 中的 Int8 校准
简介
在 API 示例 04 中，我们了解了如何利用 Polygraphy 内置的校准器，轻松地在 TensorRT 中执行 int8 校准。
但如果我们想在命令行中实现同样的功能呢？
为此，我们需要一种向命令行工具提供自定义输入数据的方法。Polygraphy 提供了多种方式来实现这一点，详见此处。
在本示例中，我们将使用一个数据加载脚本：在一个名为 data_loader.py 的 Python 脚本中定义 load_data 函数，然后使用 polygraphy convert 来构建 TensorRT 引擎。
提示：我们也可以对 polygraphy run 采用类似的方法来构建并运行引擎。
运行示例:
转换模型，使用自定义数据加载脚本提供校准数据，并保存校准缓存以供后续使用：
polygraphy convert identity.onnx --int8 \
    --data-loader-script ./data_loader.py \
    --calibration-cache identity_calib.cache \
    -o identity.engine
[可选] 使用缓存重新构建引擎以跳过校准步骤：
polygraphy convert identity.onnx --int8 \
    --calibration-cache identity_calib.cache \
    -o identity.engine
由于校准缓存已包含数据，校准过程将被跳过。因此，我们无需再提供输入数据。
[可选] 直接使用 API 示例中的数据加载器。
这里介绍的方法非常灵活，我们甚至可以直接复用 API 示例中定义的数据加载器！只需指定函数名即可，因为该示例中的函数并不叫 load_data：
polygraphy convert identity.onnx --int8 \
    --data-loader-script ../../../api/04_int8_calibration_in_tensorrt/example.py:calib_data \
    -o identity.engine
