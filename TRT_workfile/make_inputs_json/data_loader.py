"""
制作校准集的数据加载器函数，直接传给命令行量化的代码中，替代使用json文件，解决显存不足问题。
"""

import numpy as np
import torch
import os
import cv2
from modelopt.onnx.quantization import quantize

from polygraphy.backend.trt import Calibrator

# # 伪造校准数据
# dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ONNX_PATH = "/root/my_FILE/myStudy_flle/best_PCB_F.onnx"
CALIB_IMG_DIR = "/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val"  # 👈 替换为你的真实校准图片文件夹路径
# engine_path = "yolov8_int8.onnx"
CALIB_NUM = 200  # 校准图片数量，建议 100~500 张
IMG_SIZE = 640   # YOLOv8 输入尺寸

def letterbox_yolov8(im, new_shape=(640, 640), color=(114, 114, 114)):
    """
    YOLOv8 标准 Letterbox 预处理：保持比例缩放并使用灰色填充
    """
    shape = im.shape[:2]  # 获取当前图片的 [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 1. 计算缩放比例 (以长边为基准等比例缩小，避免变形)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # 2. 计算缩放后的实际尺寸
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

    # 3. 计算需要填充的像素量，并平分到两侧使图片居中
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    # 4. 执行缩放
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    # 5. 执行填充 (使用指定的灰色)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)) # 保证上下填充之和绝对等于目标需要填充的总像素，防止图像尺寸出现 1 个像素的偏差。
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return im

def preprocess_image(image_path, target_size=(640, 640)):
    """
    完整的 YOLOv8 图像预处理流水线
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    # 1. 执行 Letterbox 缩放与填充
    img_letterbox = letterbox_yolov8(img, target_size)
    
    # 2. BGR 转 RGB，并归一化到 [0, 1]
    img_rgb = cv2.cvtColor(img_letterbox, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    
    # 3. HWC 转 CHW，并增加 Batch 维度 -> NCHW
    img_chw = np.transpose(img_normalized, (2, 0, 1))
    img_nchw = np.expand_dims(img_chw, axis=0)

    return img_nchw


def load_data():
    """
    Polygraphy 要求的标准数据加载器生成器。
    每次 yield 一个字典，将输入名称映射到 NumPy 数组。
    """
    img_files = [f for f in os.listdir(CALIB_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if len(img_files) == 0:
        raise FileNotFoundError(f"在 {CALIB_IMG_DIR} 中未找到任何图片！")

    input_name = "images"  # 必须与 ONNX 模型的输入节点名称完全一致
    count = 0
    
    for img_file in img_files:
        if count >= CALIB_NUM:
            break
            
        img_path = os.path.join(CALIB_IMG_DIR, img_file)
        img_array = preprocess_image(img_path)  # 确保返回 shape 为 (1, 3, 640, 640) 或 (3, 640, 640)
        
        if img_array is None:
            continue  # 跳过损坏的图片
            
        # 确保是 numpy 数组
        if isinstance(img_array, np.ndarray):
            # 【关键】每次 yield 一个字典，而不是返回列表或大数组
            yield {input_name: img_array}
            count += 1
    
    print(f"[INFO] 成功加载 {count} 张校准图片。")