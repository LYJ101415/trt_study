"""
使用单张图片生成真实图片输入的 JSON 文件，用于 Polygraphy 的推理测试
"""

import cv2
import json
import numpy as np

from polygraphy.json import save_json 

# 1. 辅助类：用于将 NumPy 数组序列化为 JSON（必须放在最前面）
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# 2. 配置参数（请根据实际情况修改）
IMAGE_PATH = "/root/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/train/00041010.jpg"  # 替换为你的真实图片路径
OUTPUT_JSON = "/root/my_FILE/real_image_inputs.json"
TARGET_SIZE = (640, 640)                     # 模型期望的输入尺寸 [H, W]
INPUT_NAME = "images"                        # ONNX 模型中实际输入节点的名称

# 3. 读取并预处理图片
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"无法读取图片: {IMAGE_PATH}")

# YOLO 标准预处理流程
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)           # BGR 转 RGB
img = cv2.resize(img, TARGET_SIZE)                   # 缩放到目标尺寸
img = img.astype(np.float32) / 255.0                 # 归一化到 [0, 1]
img = np.transpose(img, (2, 0, 1))                   # HWC 转 CHW
img = np.expand_dims(img, axis=0)                    # 增加 Batch 维度，变为 [1, 3, 640, 640]

# 3. 构造 feed_dict（注意：必须是字典格式）
feed_dict = {
    INPUT_NAME: img
}

# 4. 【核心修改】使用 Polygraphy 的 save_json 保存
# Polygraphy 期望 inputs 文件是一个列表，所以这里传入 [feed_dict]
save_json([feed_dict], OUTPUT_JSON)

print(f"[SUCCESS] 真实图片输入已生成并保存至: {OUTPUT_JSON}")
print(f"[INFO] 输入张量形状: {img.shape}")