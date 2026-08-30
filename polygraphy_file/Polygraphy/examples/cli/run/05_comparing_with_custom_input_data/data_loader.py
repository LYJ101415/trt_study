"""这段代码是 Polygraphy 中用于提供自定义输入数据的标准模板，它同时展示了两种主流方式：
动态生成输入（通过 load_data() 函数） → 适用于 --data-loader-script
静态保存输入到 JSON 文件 → 适用于 --load-inputs
"""

"""
Demonstrates two methods of loading custom input data in Polygraphy:

Option 1: Defines a `load_data` function that returns a generator yielding
    feed_dicts so that this script can be used as the argument for
    the --data-loader-script command-line parameter.

Option 2: Writes input data to a JSON file that can be used as the argument for
    the --load-inputs command-line parameter.
"""
import numpy as np
from polygraphy.json import save_json # 导入 Polygraphy 的 JSON 序列化工具，支持保存/加载含 NumPy 数组的数据

"""定义输入张量的形状（batch=1, channel=2, height=28, width=28）
 这通常与模型的输入要求一致，且需匹配 TensorRT 的 optimization profile"""
INPUT_SHAPE = (1, 2, 28, 28)


# Option 1: Define a function that will yield feed_dicts (i.e. Dict[str, np.ndarray])
def load_data():
    """
    此函数必须命名为 `load_data`（Polygraphy CLI 的约定）。
    它返回一个 **generator**，每次 yield 一个 "feed_dict"：
        - key: 输入张量名称（如 "x"）
        - value: 对应的 NumPy 数组（dtype 和 shape 必须合法）
    注意：这里循环 5 次 → 表示提供 5 组不同的输入样本。
    即使每组数据相同（np.ones），也视为 5 次独立推理。
    """
    for _ in range(5):
        yield {
            "x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)
        }  # Still totally real data


# Option 2: Create a JSON file containing the input data using the `save_json()` helper.
#   The input to `save_json()` should have type: List[Dict[str, np.ndarray]].
#   For convenience, we'll reuse our `load_data()` implementation to generate the list.
"""将 generator 转换为列表：[feed_dict1, feed_dict2, ..., feed_dict5]; 每个 feed_dict 是 {"x": np.ndarray}"""
input_data = list(load_data())
"""将输入数据序列化为 Polygraphy 自定义的 JSON 格式
 - 自动处理 NumPy dtype/shape/base64 编码;- 添加描述信息便于识别
 生成的文件可直接用于 `polygraphy run --load-inputs custom_inputs.json`"""
save_json(input_data, "custom_inputs.json", description="custom input data")

"""
三、实际应用示例
场景：验证动态 shape 模型在固定输入下的 TRT 精度
Step 1: 保存代码为 data_loader.py
Step 2: 生成 JSON（可选）
python3 data_loader.py  # 生成 custom_inputs.json
Step 3: 使用数据加载脚本运行对比
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --trt-min-shapes x:[1,2,28,28] \
    --trt-opt-shapes x:[1,2,28,28] \
    --trt-max-shapes x:[1,2,28,28] \
    --data-loader-script data_loader.py
 效果：Polygraphy 会运行 5 次推理（每次输入全1），并对比 TRT 与 ONNX Runtime 的输出。

Step 4: 使用 JSON 文件运行对比
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --trt-min-shapes x:[1,2,28,28] \
    --trt-opt-shapes x:[1,2,28,28] \
    --trt-max-shapes x:[1,2,28,28] \
    --load-inputs custom_inputs.json
 效果：与 Step 3 完全相同，但输入来自磁盘文件。

 四、扩展应用场景
场景 1：加载真实图像
import cv2

def load_data():
    image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224))
        img = img.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0
        yield {"input": img}  # 假设模型输入名为 "input"
场景 2：测试边界条件
def load_data():
    # 测试全零输入
    yield {"x": np.zeros(INPUT_SHAPE, dtype=np.float32)}
    # 测试极大值
    yield {"x": np.full(INPUT_SHAPE, 1e6, dtype=np.float32)}
    # 测试随机噪声
    yield {"x": np.random.randn(*INPUT_SHAPE).astype(np.float32)}
场景 3：多输入模型
def load_data():
    yield {
        "input1": np.ones((1, 3, 224, 224), dtype=np.float32),
        "input2": np.zeros((1, 10), dtype=np.int32)
    }
 五、注意事项
张量名称必须匹配;错误的输入名会导致 TRT/ONNX 报错 “invalid input name”;
dtype 必须正确;ONNX 模型若要求 float32，不能传 float64
动态 shape 兼容性;所有生成的输入 shape 必须落在 TRT profile 的 [min, max] 范围内
JSON 文件大小;大数据集（如 1000 张图像）会导致 JSON 文件巨大 → 此时优先用 --data-loader-script

 六、总结：这份代码的工程价值
特性	      说明
双模式支持	  同一份逻辑，同时支持流式（脚本）和静态（JSON）输入
极简 API	  只需实现 load_data()，无需关心底层序列化
无缝集成 CLI	直接用于 polygraphy run，无需修改推理流程
可复现性保障	JSON 模式确保问题可精确复现

 最佳实践建议：开发阶段：用 --data-loader-script 快速迭代
问题复现/CI 验证：用 --load-inputs + JSON 确保确定性
这种设计体现了 Polygraphy “简单场景极简，复杂场景可控” 的哲学，是模型部署验证环节的利器。
"""
