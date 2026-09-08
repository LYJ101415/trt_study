"""
忽略节点的量化为onnx模型脚本 —— 流式校准版（磁盘 memmap，不把校准集整体放进内存）

与 quant_int8_exclude.py 的区别:
    原版 calib_data() 把所有校准图片拼接成一个 (N, 3, 640, 640) float32 的普通
    ndarray 一次性放进主机内存（200 张约 1 GB，几千张可达几十 GB，容易 OOM）。

    本版的标准做法（与 TensorRT IInt8Calibrator::getBatch / ORT
    CalibrationDataReader 同思路）:
        1. 预处理阶段逐张读取图片，逐张写入磁盘上的 np.memmap 文件，
           任意时刻内存里只有一张图；
        2. 校准阶段把只读 memmap 直接交给 ModelOpt。memmap 是 ndarray 的子类，
           ModelOpt 内部 CalibrationDataProvider 用 np.array_split 对它切片时
           得到的仍是零拷贝的磁盘视图，只会按 batch 惰性读入内存，
           因此主机内存 / 显存都只占用单个 batch。

    另外支持校准缓存复用: 同一图片目录+数量+尺寸的 memmap 已存在时直接复用，
    重复量化无需重新做一遍预处理。

依赖: modelopt==0.42.0（quantize() 的 calibration_data 参数接受 np.ndarray，
      np.memmap 为其子类，可直接传入）
"""

import json
import os
import time

import cv2
import numpy as np
import onnx  # 仅用于节点名校验
import torch
from modelopt.onnx.quantization import quantize

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ONNX_PATH = "/root/my_FILE/models/best_PCB_dy.onnx"
CALIB_IMG_DIR = "/root/my_FILE/datasets/Data_DeepPCB_YOLO/images/val"
engine_path = "/root/my_FILE/models/yolov8_int8_dy_exclude.onnx"
CALIB_NUM = 200
IMG_SIZE = 640

# 校准数据 memmap 缓存目录（磁盘空间需求: CALIB_NUM * 3 * IMG_SIZE^2 * 4 字节，
# 例如 5000 张 640x640 约 24.6 GB，请确保该目录所在分区放得下）
CALIB_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_cache")

# # 需要忽略(保持 FP32)的敏感节点名 —— 先校验再填精确名
# NODES_TO_EXCLUDE = ["Conv_3", "Conv_250"]

NODES_TO_EXCLUDE = ["^/model.1/conv/Conv$","^/model.22/dfl/conv/Conv$"] # "^/model.1/conv/Conv$",


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


def _cache_paths():
    """返回 (数据文件路径, 元信息文件路径)"""
    data_path = os.path.join(CALIB_CACHE_DIR, f"calib_{IMG_SIZE}x{IMG_SIZE}_f32.dat")
    meta_path = data_path + ".meta.json"
    return data_path, meta_path


def _list_calib_images():
    """列出校准图片（排序保证可复现），并做最基本的可读性过滤"""
    if not os.path.isdir(CALIB_IMG_DIR):
        raise FileNotFoundError(f"校准图片目录不存在: {CALIB_IMG_DIR}")
    img_files = sorted(
        f for f in os.listdir(CALIB_IMG_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))
    )
    if not img_files:
        raise FileNotFoundError(f"在 {CALIB_IMG_DIR} 中未找到任何图片！")
    return img_files


def build_calib_memmap():
    """
    流式预处理校准图片并写入磁盘 memmap，返回 (只读 memmap, 有效样本数)。

    内存占用: 任意时刻只有一张图（预处理）+ 操作系统按页读入的少量数据。
    """
    os.makedirs(CALIB_CACHE_DIR, exist_ok=True)
    data_path, meta_path = _cache_paths()
    img_files = _list_calib_images()

    # 1) 命中已有缓存则直接复用（校准集、数量、尺寸均未变化时）
    meta = {
        "img_dir": CALIB_IMG_DIR,
        "img_size": IMG_SIZE,
        "calib_num": CALIB_NUM,
        "first_files": img_files[:CALIB_NUM],
    }
    if os.path.exists(meta_path) and os.path.exists(data_path):
        with open(meta_path) as f:
            old_meta = json.load(f)
        # old_meta 比构建 meta 多存一个 "valid_num"：先取值，比对时再剔除
        valid_num = int(old_meta.get("valid_num", 0))
        if old_meta.pop("valid_num", None) is not None and old_meta == meta:
            n = valid_num
            print(f"[INFO] 命中校准缓存 {data_path}，共 {n} 个样本，跳过预处理。")
            return np.memmap(data_path, dtype=np.float32, mode="r",
                             shape=(n, 3, IMG_SIZE, IMG_SIZE)), n
        print("[INFO] 校准缓存已过期（图片目录/数量/尺寸变化），重新生成。")

    # 2) 预分配 CALIB_NUM 大小的磁盘文件，逐张写入
    n_max = min(CALIB_NUM, len(img_files))
    mm = np.memmap(data_path, dtype=np.float32, mode="w+",
                   shape=(n_max, 3, IMG_SIZE, IMG_SIZE))

    t0 = time.time()
    count = 0
    for img_file in img_files:
        if count >= CALIB_NUM:
            break
        img_path = os.path.join(CALIB_IMG_DIR, img_file)
        try:
            img_array = preprocess_image(img_path)  # (1, 3, 640, 640)
        except Exception as e:  # 坏图/读不出的图跳过，而不是中断整个量化
            print(f"[WARN] 跳过无法处理的图片 {img_file}: {e}")
            continue

        mm[count] = img_array[0]  # 直接写进磁盘上的第 count 行，不累积到内存
        count += 1
        if count % 50 == 0:
            print(f"[INFO] 已预处理 {count}/{n_max} 张 ({time.time() - t0:.1f}s)")

    if count == 0:
        raise RuntimeError("没有任何一张图片预处理成功！")

    mm.flush()
    del mm

    # 3) 写元信息并重新以只读方式打开有效样本数对应的区域
    meta["valid_num"] = count
    with open(meta_path, "w") as f:
        json.dump(meta, f)
    print(f"[INFO] 预处理完成: {count} 张校准图已写入 {data_path} "
          f"({count * 3 * IMG_SIZE * IMG_SIZE * 4 / 1024**3:.2f} GB, {time.time() - t0:.1f}s)")

    calib_mm = np.memmap(data_path, dtype=np.float32, mode="r",
                         shape=(count, 3, IMG_SIZE, IMG_SIZE))
    return calib_mm, count


def calib_data():
    """
    真实数据集校准数据（磁盘 memmap，惰性按 batch 读取）。

    返回 shape 为 (N, 3, 640, 640) 的只读 np.memmap（ndarray 子类），
    可直接作为 modelopt quantize() 的 calibration_data 参数。
    """
    calib_mm, count = build_calib_memmap()
    print(f"[INFO] 校准数据: shape={calib_mm.shape}, dtype={calib_mm.dtype}, "
          f"存储=磁盘memmap（常驻内存≈0，校准时按batch惰性读入）")
    return calib_mm


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

    # 1) 校准数据（磁盘 memmap，不再整体加载进内存）
    calibrator_data = calib_data()

    # 2) 量化并忽略敏感节点
    quantize(
        ONNX_PATH,
        calibration_data=calibrator_data,
        calibration_method='entropy',
        output_path=engine_path,
        calibration_eps=['cpu'],   # 固定校准在 CPU 上跑，保证熵校准的 scale 确定可复现（消除 CPU/GPU 漂移导致的精度波动）；
                          # 传['cuda:0'] —— 快，但 cuDNN 的浮点实现 run-to-run 可能有微小差异。关键是别让它在 CPU/GPU 之间漂
        nodes_to_exclude=NODES_TO_EXCLUDE,   # 忽略节点
    )
    print(f"[DONE] 已输出到 {engine_path}")


if __name__ == "__main__":
    main()

"""
export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
python3 /root/my_FILE/my_trt_FILE/TRT_workfile/quant_model/quant_int8_exclude_stream.py
"""
