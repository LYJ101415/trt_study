"""以下代码是 Polygraphy 工具链中用于 INT8 量化校准（Calibration）的标准数据加载器模板。"""

"""
Defines a `load_data` function that returns a generator yielding
feed_dicts so that this script can be used as the argument for
the --data-loader-script command-line parameter.
"""
import numpy as np

"""若模型有动态维度（如 batch=-1），此处需用具体值（校准时必须固定 shape）"""
INPUT_SHAPE = (1, 1, 2, 2)

"""以下是Polygraphy 约定的入口函数名；函数名必须为 `load_data`（除非通过 `script.py:func_name` 显式指定）。
返回类型：可迭代对象（generator/list），每个元素是 feed_dict。"""
def load_data():
    for _ in range(5):
        """以下是返回符合模型输入要求的 NumPy 数组：
         - 键 "x" 必须与 ONNX 模型的输入名一致（通过 netron 查看）
         - dtype=np.float32：TensorRT 校准要求输入为 FP32
         - np.ones()：此处仅为占位符！实际应用需替换为真实数据"""
        yield {
            "x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)
        }  # Still totally real data

"""
为什么这个脚本能用于 INT8 校准？
1. Polygraphy 的数据加载协议
当使用 --data-loader-script data_loader.py 时，Polygraphy 会：
动态导入 data_loader.py;调用 load_data() 函数;遍历其返回的 generator，获取每个 feed_dict
将 feed_dict 中的 NumPy 数组传递给 TensorRT 的 IInt8EntropyCalibrator2。

关键要求：
输入名（如 "x"）必须与 ONNX 模型 完全匹配；数据类型必须为 np.float32（TensorRT 校准不支持其他类型）

2. 校准过程发生了什么？
对每个校准样本：TensorRT 执行前向推理（仅到需要量化的层）；记录每个激活层的 输出值范围（min/max）；
基于所有样本的统计信息，计算 量化 scale factor；生成 校准缓存（calibration cache）。

校准质量取决于：数据的 代表性（是否覆盖真实场景的激活分布）；样本的 数量（通常 100~500 足够）

实际应用示例（从玩具到生产）
示例 1：真实图像校准（ResNet-50）
# real_calib_loader.py
import numpy as np
from PIL import Image
import os

INPUT_SHAPE = (1, 3, 224, 224)  # ResNet-50 输入
CALIB_DIR = "./calib_images/"  # 校准图像目录

def load_data():
    image_paths = [os.path.join(CALIB_DIR, f) for f in os.listdir(CALIB_DIR)][:100]
    for path in image_paths:
        # 1. 加载图像
        img = Image.open(path).convert("RGB")
        # 2. Resize + ToTensor（与训练预处理一致！）
        img = img.resize((224, 224))
        arr = np.array(img).astype(np.float32)  # HWC
        arr = arr.transpose(2, 0, 1)  # CHW
        arr = arr / 255.0  # 归一化 [0,1]
        arr = (arr - np.array([0.485, 0.456, 0.406])[:, None, None]) / np.array([0.229, 0.224, 0.225])[:, None, None]  # 标准化
        # 3. 添加 batch 维度
        yield {"input": np.expand_dims(arr, axis=0)}  # input 是 ONNX 模型的输入名

使用命令：
polygraphy convert resnet50.onnx --int8 \
    --data-loader-script real_calib_loader.py \
    --calibration-cache resnet50.cache \
    -o resnet50_int8.engine

示例 2：从 PyTorch DataLoader 生成校准数据
# torch_calib_loader.py
import numpy as np
from torchvision import datasets, transforms

INPUT_SHAPE = (1, 3, 32, 32)  # CIFAR-10

def load_data():
    # 复用训练时的预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    
    # 取前 200 个样本
    for i in range(200):
        img, _ = dataset[i]
        # 转为 NumPy (注意：ToTensor 输出是 [0,1] 的 float32)
        yield {"input": img.numpy()[None, ...]}  # 添加 batch 维度

优势：确保校准数据与训练数据 预处理完全一致。

示例 3：多输入模型校准（如 BERT）
# bert_calib_loader.py
import numpy as np

def load_data():
    for _ in range(100):
        seq_len = 128
        yield {
            "input_ids": np.random.randint(0, 30522, size=(1, seq_len)).astype(np.int32),
            "attention_mask": np.ones((1, seq_len), dtype=np.int32),
            "token_type_ids": np.zeros((1, seq_len), dtype=np.int32)
        }

注意：非 float32 输入（如 int32）不需要校准（TensorRT 仅量化 float32 张量），
但必须提供 完整 feed_dict（否则 TensorRT 报错）

示例 4：动态 Shape 校准（需固定具体值）
# dynamic_shape_calib.py
import numpy as np

# 假设模型输入 shape 为 (-1, 3, -1, -1) → 校准时必须指定具体值
BATCH_SIZE = 4
HEIGHT = 416
WIDTH = 416

def load_data():
    for _ in range(50):
        yield {
            "image": np.random.rand(BATCH_SIZE, 3, HEIGHT, WIDTH).astype(np.float32)
        }

重要：校准后的引擎 只能用于相同或更小的 shape（TensorRT 限制）。

常见错误与最佳实践：
错误 1：输入名不匹配
现象：[E] Calibration failed: Input 'x' not found in model
解决：用 Netron 查看 ONNX 模型输入名
错误 2：数据类型非 float32
现象：校准 silently 失败，INT8 精度极差
解决：强制 dtype=np.float32
错误 3：校准数据无代表性
现象：校准通过，但真实数据上精度暴跌
解决：从 真实推理数据 中采样，覆盖 各种场景（如白天/黑夜图像）

最佳实践：
实践	                说明
样本量 ≥ 100	        少于 50 可能欠拟合激活分布
预处理一致性	         校准数据预处理 = 推理时预处理
保存 calibration cache	避免重复校准，保证量化一致性

校准后验证精度	用 polygraphy run --validate-with 对比 FP32

性能影响：校准数据质量 vs INT8 精度
校准数据	Top-1 Acc (ResNet-50)	校准时间
全 0 (np.zeros)	        0.1%	    2s
全 1 (np.ones)	        0.1%	    2s
ImageNet 子集 (100 张)	76.2%	    15s
ImageNet 子集 (500 张)	76.3%	    70s

结论：真实数据 > 样本数量。100 张代表性图像 ≈ 500 张随机图像。

总结：这个脚本的核心价值
角色	     如何受益
算法工程师	快速验证模型 INT8 可行性
部署工程师	一键生成生产级 INT8 引擎
MLOps 团队	集成到 CI/CD 流水线（校准 → 构建 → 验证）

终极建议：永远不要用 np.ones() 或 np.zeros() 作为生产校准数据！

"""
