# Using Custom Input Data

For any tools that use inference input data, such as `run` or `convert`, Polygraphy
provides 2 ways to supply custom input data:

1. `--load-inputs`/`--load-input-data`, which takes a path to a JSON file containing a
    `List[Dict[str, np.ndarray]]`.
    The JSON file should be created by using Polygraphy's JSON utilities, like `save_json`,
    in the `polygraphy.json` submodule.

    *NOTE: This will cause Polygraphy to load the entire object into memory and so may be*
        *impractical or impossible if the data is very large.*

2. `--data-loader-script`, which takes a path to a Python script that defines a `load_data` function
    that returns a data loader. The data loader can be any iterable or generator that yields
    `Dict[str, np.ndarray]`. By using a generator, we can avoid loading all the data
    at once, and instead limit it to just a single input at a time.

    *TIP: If you have an existing script that already defines such a function, you do **not** need to create*
        *a separate script just for the sake of `--data-loader-script`. You can simply use the existing script*
        *and specify the name of the function if it's not `load_data`*


## Further Reading

- See [`run` example 05](../examples/cli/run/05_comparing_with_custom_input_data/)
    for examples of both approaches highlighted above.

======================================================================
这份文档详细说明了在 Polygraphy 工具（如 run 或 convert）中，如何为模型推理提供自定义输入数据。它提供了两种不同的方法，分别适用于不同的数据规模和使用场景。
下面为你详细解析这两种方法的含义、作用，并提供相应的代码示例。
 方法一：使用 JSON 文件加载 (--load-inputs)
含义与作用
这种方法通过 --load-inputs (或 --load-input-data) 参数，指定一个 JSON 文件的路径。该文件必须包含一个特定格式的数据结构：List[Dict[str, np.ndarray]]。
List: 代表多次推理迭代。列表中的每个元素对应一次推理的输入。
Dict[str, np.ndarray]: 代表单次推理的所有输入。字典的键（str）是模型输入节点的名称，值（np.ndarray）是对应的 NumPy 数组数据。
 重要提示：Polygraphy 会将整个 JSON 文件一次性加载到内存中。因此，如果数据集非常大，这种方法可能会导致内存溢出，变得不切实际甚至无法使用。
实际应用示例
假设你有一个模型，输入名为 input_image，形状为 [1, 3, 224, 224]。你想用两组自定义数据进行测试。
1. 创建 JSON 文件
推荐使用 Polygraphy 自带的 save_json 工具来确保格式正确。
import numpy as np
from polygraphy.json import save_json

# 准备两次迭代的输入数据
input_data_1 = np.random.rand(1, 3, 224, 224).astype(np.float32)
input_data_2 = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 构造 List[Dict[str, np.ndarray]] 格式
custom_inputs = [
    {"input_image": input_data_1},  # 第1次迭代
    {"input_image": input_data_2}   # 第2次迭代
]

# 保存为 JSON 文件
save_json(custom_inputs, "my_custom_inputs.json")
2. 在命令行中使用
polygraphy run model.onnx --onnxrt --load-inputs my_custom_inputs.json
 方法二：使用自定义 Python 脚本 (--data-loader-script)
含义与作用
这种方法通过 --data-loader-script 参数，指定一个 Python 脚本的路径。该脚本需要定义一个函数（默认为 load_data），该函数返回一个数据加载器。
数据加载器：可以是任何可迭代对象或生成器 (generator)，它在每次迭代时产生一个 Dict[str, np.ndarray]。
核心优势：通过使用生成器，你可以实现懒加载 (lazy loading)。这意味着数据不需要一次性全部载入内存，而是每次只加载或生成一个输入，这对于处理大规模数据集至关重要。
 实用技巧：如果你已经有一个脚本定义了数据加载函数，但函数名不叫 load_data，你无需创建新脚本。只需在命令行中指定 脚本路径:函数名 即可，例如 my_script.py:my_custom_loader。
实际应用示例
同样针对 input_image 输入，我们使用生成器来避免内存问题。
1. 创建 Python 脚本 (data_loader.py)
import numpy as np

def load_data():
    """
    一个生成器函数，每次调用 yield 一个输入字典。
    这里演示生成10个随机输入，实际中可以替换为从磁盘读取文件的逻辑。
    """
    for i in range(10):
        print(f"正在加载第 {i+1} 个输入...")
        # 模拟从文件或数据库读取数据
        input_data = np.random.rand(1, 3, 224, 224).astype(np.float32)
        yield {"input_image": input_data}
2. 在命令行中使用
# 使用默认的 load_data 函数
polygraphy run model.onnx --onnxrt --data-loader-script data_loader.py

# 或者，如果函数名是 my_custom_loader
# polygraphy run model.onnx --onnxrt --data-loader-script data_loader.py:my_custom_loader
🚀 高级应用示例：结合模型量化校准
自定义输入数据在 TensorRT 的 INT8 量化校准（calibration）中非常关键。你需要提供一组具有代表性的真实数据，而不是随机数据，以获得最佳的量化精度。
场景：使用自定义的真实图片数据对 ONNX 模型进行 INT8 转换和校准。
1. 准备校准数据加载器 (calib_data_loader.py)
import numpy as np
# 假设你有一个读取图片并预处理的函数
# from your_preprocessing_lib import load_and_preprocess_image 

def load_data():
    # 从文件列表或数据库中读取真实图片路径
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"] # ... 更多图片
    
    for path in image_paths:
        # 读取、预处理并转换为 numpy 数组
        # img_array = load_and_preprocess_image(path)
        img_array = np.random.rand(1, 3, 224, 224).astype(np.float32) # 示例用随机数据代替
        
        yield {"input_image": img_array}
2. 执行带校准的转换
使用 polygraphy convert 命令，并通过 --data-loader-script 提供校准数据，同时保存校准缓存以便下次复用。
polygraphy convert model.onnx \
    --int8 \
    --data-loader-script calib_data_loader.py \
    --calibration-cache my_model_calib.cache \
    -o model_int8.engine
在这个例子中，--data-loader-script 的懒加载特性使得你可以轻松使用成千上万张图片进行校准，而无需担心内存限制。