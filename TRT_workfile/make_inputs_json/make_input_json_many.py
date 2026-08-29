"""
使用多张张图片生成真实图片输入的 JSON 文件，用于 Polygraphy 的推理测试
"""

import cv2
import json
import numpy as np
import os
from glob import glob

from polygraphy.json import save_json


# 1. 辅助类：用于将 NumPy 数组序列化为 JSON（必须放在最前面）
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# 2. 配置参数（请根据实际情况修改）
IMAGE_DIR = "/root/my_FILE/my_FILE/test_images"  # 图片文件夹路径
OUTPUT_JSON = "/root/my_FILE/real_image_inputs.json"
TARGET_SIZE = (640, 640)       # 模型期望的输入尺寸 [H, W]
INPUT_NAME = "images"          # ONNX 模型中实际输入节点的名称
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp")  # 支持的图片格式

# 自动扫描文件夹下所有匹配的图片，并按文件名排序（保证顺序可复现）
IMAGE_PATHS = sorted(
    p for ext in IMAGE_EXTENSIONS
    for p in glob(os.path.join(IMAGE_DIR, ext))
)

if not IMAGE_PATHS:
    raise FileNotFoundError(
        f"在 {IMAGE_DIR} 下未找到任何匹配的图片文件\n"
        f"已搜索的格式: {IMAGE_EXTENSIONS}"
    )
print(f"[INFO] 在 {IMAGE_DIR} 下共发现 {len(IMAGE_PATHS)} 张图片")


def preprocess_image(image_path: str, target_size: tuple) -> np.ndarray:
    """YOLO 标准预处理流程，返回 [1, 3, H, W] 格式的数组"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # BGR → RGB
    img = cv2.resize(img, target_size)                   # 缩放到目标尺寸
    img = img.astype(np.float32) / 255.0                 # 归一化到 [0, 1]
    img = np.transpose(img, (2, 0, 1))                   # HWC → CHW
    img = np.expand_dims(img, axis=0)                    # 增加 Batch 维度 → [1, 3, H, W]
    return img


# 3. 批量处理所有图片，构造 feed_dict 列表
feed_dicts = []
failed_images = []

for path in IMAGE_PATHS:
    try:
        processed = preprocess_image(path, TARGET_SIZE)
        feed_dicts.append({INPUT_NAME: processed})
        print(f"[OK]   {os.path.basename(path)} → shape {processed.shape}")
    except Exception as e:
        failed_images.append((path, str(e)))
        print(f"[FAIL] {os.path.basename(path)} → {e}")

if not feed_dicts:
    raise RuntimeError("没有成功处理任何图片，请检查路径和文件完整性")

# 4. 使用 Polygraphy 的 save_json 保存
# Polygraphy 期望 inputs 文件是一个 List[Dict[str, ndarray]]，每个元素对应一次推理
save_json(feed_dicts, OUTPUT_JSON)

print(f"\n{'='*60}")
print(f"[SUCCESS] 共处理 {len(feed_dicts)} 张图片，已保存至: {OUTPUT_JSON}")
if failed_images:
    print(f"[WARNING] {len(failed_images)} 张图片处理失败:")
    for p, err in failed_images:
        print(f"         - {os.path.basename(p)}: {err}")