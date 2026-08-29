"""
忽略节点的量化为onnx模型脚本
"""

import numpy as np
import torch
import os
import cv2
from modelopt.onnx.quantization import quantize

import onnx  # 仅用于节点名校验

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ONNX_PATH = "/root/my_FILE/models/best_PCB.onnx"
CALIB_IMG_DIR = "/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val"
engine_path = "yolov8_int8_1.onnx"
CALIB_NUM = 200
IMG_SIZE = 640

# # 需要忽略(保持 FP32)的敏感节点名 —— 先校验再填精确名
# NODES_TO_EXCLUDE = ["Conv_3", "Conv_250"]

NODES_TO_EXCLUDE = ["^/model.1/conv/Conv$", "^/model.22/dfl/conv/Conv$"]


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
    

def calib_data():
    """
    真实数据集校准数据生成器，返回完整的 numpy 数组
    """
    img_files = [f for f in os.listdir(CALIB_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if len(img_files) == 0:
        raise FileNotFoundError(f"在 {CALIB_IMG_DIR} 中未找到任何图片！")

    input_name = "images"
    calib_list = []
    count = 0
    
    for img_file in img_files:
        if count >= CALIB_NUM:
            break
        img_path = os.path.join(CALIB_IMG_DIR, img_file)
        img_array = preprocess_image(img_path)
        if img_array is None:
            continue  # 跳过损坏的图片
            
        # 确保是 numpy 数组，且 shape 为 (1, 3, 640, 640)
        if isinstance(img_array, np.ndarray):
            calib_list.append(img_array)
            count += 1
    
    print(f"[INFO] 成功加载 {count} 张校准图片。")
    
    # 将列表拼接为 (N, 1, 3, 640, 640)，然后 squeeze 掉多余的维度，变成 (N, 3, 640, 640)
    # 注意：如果 preprocess_image 已经返回 (1,3,640,640)，stack 后是 (N,1,3,640,640)
    # 我们需要将其调整为 (N, 3, 640, 640)
    calib_array = np.concatenate(calib_list, axis=0) 

    return calib_array

def verify_node_names(onnx_path, patterns):
    """打印命中 patterns 的节点名+类型, 用于确认 nodes_to_exclude 会命中哪些节点."""
    import re
    model = onnx.load(onnx_path)
    matched = []
    for n in model.graph.node:
        for p in patterns:
            if re.search(p, n.name):
                matched.append((n.name, n.op_type))
                break
    if not matched:
        print(f"[WARN] 未找到匹配 {patterns} 的节点! 请用 Netron 核对真实节点名")
        return matched
    for name, op in matched:
        print(f"    {name:<40s} op={op}")
    return matched


def main():
    # 0) 先校验节点名, 防止正则过度匹配/漏匹配
    print("[INFO] 校验待忽略节点名:")
    verify_node_names(ONNX_PATH, NODES_TO_EXCLUDE)

    # 1) 校准数据
    calibrator_data = calib_data()

    # 2) 量化并忽略敏感节点
    quantize(
        ONNX_PATH,
        calibration_data=calibrator_data,
        calibration_method='entropy',
        output_path=engine_path,
        nodes_to_exclude=NODES_TO_EXCLUDE,   # 忽略节点
    )
    print(f"[DONE] 已输出到 {engine_path}")


if __name__ == "__main__":
    main()
