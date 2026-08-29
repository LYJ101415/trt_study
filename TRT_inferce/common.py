"""
common.py — 共享工具：CUDA Runtime 封装 + 参考 CPU 预处理/后处理 + 可视化。

GPU 与 CPU 两条链路都复用这里的 ctypes/cudart 封装（与 server_trtapi.py 一致），
并保留一份"参考 CPU 实现"用于 test_correctness.py 做逐像素/逐框对比。
"""

import ctypes
import numpy as np

IMG_SIZE = 640
CONF_THRESH = 0.45
NMS_THRESH = 0.65
NUM_CLASSES = 6
PAD_VALUE = 114 # Letterbox 填充时的像素灰度值（YOLO 官方默认灰色填充）


# ---------------- ctypes 封装 libcudart ----------------
# 设计意图：在不依赖 PyTorch/CuPy 的情况下，直接操作 GPU 显存。
# 这通常用于纯 C/C++ 推理引擎（如 TensorRT API）的 Python 绑定层，以减少内存拷贝和框架开销。
# 关键：必须显式声明 argtypes/restype，否则 ctypes 会把 64 位指针按 32 位 c_int
# 截断，导致 cudaMemcpy 报 cudaErrorInvalidValue(1)。
_cudart = ctypes.CDLL("libcudart.so") # 加载 CUDA Runtime 动态链接库
# cudaMalloc: 分配 GPU 显存
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int # 返回 cudaError_t 枚举值

# cudaFree: 释放 GPU 显存
_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int

# cudaMemcpy: Host↔Device 内存拷贝
_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cudart.cudaMemcpy.restype = ctypes.c_int # 第4个参数是 cudaMemcpyKind 枚举

# cudaHostRegister: 将 CPU 内存注册为 Page-Locked Memory
# 作用：使该内存可被 DMA 直接访问，加速 H2D/D2H 传输，并支持异步拷贝
_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int

# cudaHostUnregister: 取消 Page-Locked 注册
_cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudart.cudaHostUnregister.restype = ctypes.c_int

# 辅助函数封装
def _check_cuda(err):
    """统一检查 CUDA API 返回值，非零即抛异常，避免静默失败"""
    if err != 0:
        raise RuntimeError(f"CUDA runtime error: {err}")


def cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    """封装 cudaMalloc：分配 nbytes 字节显存，返回设备指针"""
    ptr = ctypes.c_void_p() # 创建空指针容器
    _check_cuda(_cudart.cudaMalloc(ctypes.byref(ptr), nbytes)) # byref 传递指针的地址
    return ptr   # 返回指向显存的 c_void_p 指针


def cuda_free(ptr: ctypes.c_void_p):
    """封装 cudaFree：释放显存"""
    _check_cuda(_cudart.cudaFree(ptr))


def device_addr(ptr: ctypes.c_void_p) -> int:
    """返回可传给 CUDA kernel / execute_v2 的整型设备地址（c_void_p.value 即地址）。
    提取指针的整数地址值；用途：TensorRT execute_v2 / CUDA kernel launch 需要整数形式的设备地址，
    而非 ctypes 对象"""
    return ptr.value


def cuda_memcpy_h2d(dst: ctypes.c_void_p, src: np.ndarray):
    """Host → Device 拷贝：dst: GPU 目标地址；src: CPU numpy 数组（必须 C-contiguous 以保证内存连续）
    """
    assert src.flags["C_CONTIGUOUS"], "src must be C-contiguous"
    # cudaMemcpyKind=1 表示 cudaMemcpyHostToDevice
    _check_cuda(_cudart.cudaMemcpy(dst.value, src.ctypes.data, src.nbytes, 1))  # 1=H2D


def cuda_memcpy_d2h(dst: np.ndarray, src: ctypes.c_void_p):
    """Device → Host 拷贝：dst: CPU numpy 数组（预分配好的接收缓冲区）；src: GPU 源地址
    """
    assert dst.flags["C_CONTIGUOUS"], "dst must be C-contiguous"
    # cudaMemcpyKind=2 表示 cudaMemcpyDeviceToHost
    _check_cuda(_cudart.cudaMemcpy(dst.ctypes.data, src.value, dst.nbytes, 2))  # 2=D2H


def cuda_host_register(arr: np.ndarray):
    """注册 Page-Locked Memory：性能意义：普通 malloc 的内存可能被 OS swap 到磁盘，DMA 无法直接读取；
    注册后保证物理驻留，H2D/D2H 带宽可提升 30%-50%"""
    _check_cuda(_cudart.cudaHostRegister(arr.ctypes.data, arr.nbytes, 0))


def cuda_host_unregister(arr: np.ndarray):
    """取消注册，释放 Page-Locked 资源"""
    _check_cuda(_cudart.cudaHostUnregister(arr.ctypes.data))


# ---------------- 参考 CPU 实现（用于对比测试） ----------------

def letterbox_params(h: int, w: int):
    """计算等比缩放因子和填充量（纯数学计算，无实际图像处理）
    返回值与 server_trtapi.py 保持一致，确保 CPU/GPU 两条链路对齐"""
    scale = min(IMG_SIZE / h, IMG_SIZE / w)
    nh, nw = int(h * scale), int(w * scale)
    pad_h = (IMG_SIZE - nh) // 2
    pad_w = (IMG_SIZE - nw) // 2
    return scale, pad_w, pad_h


def preprocess_cpu(image: np.ndarray):
    """参考实现：letterbox + BGR->RGB + normalize + HWC->CHW。"""
    h, w = image.shape[:2]
    scale, pad_w, pad_h = letterbox_params(h, w)
    nh, nw = int(h * scale), int(w * scale)

    resized = cv2_resize(image, (nw, nh)) # Step1: 双线性插值等比缩放
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), PAD_VALUE, dtype=np.uint8) # Step2: 创建灰色画布
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized # Step3: 居中粘贴

    # Step4: BGR→RGB（OpenCV 默认 BGR，模型训练用 RGB）
    # Step5: HWC→CHW（模型期望 NCHW 格式）
    # Step6: uint8→float32 + /255.0 归一化到 [0,1]
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    # Step7: 添加 batch 维度 → (1, 3, 640, 640)
    return blob[np.newaxis, ...], scale, pad_w, pad_h


def cv2_resize(image: np.ndarray, dsize):
    """隔离 cv2 导入，避免未安装 opencv 时整个模块加载失败"""
    import cv2
    return cv2.resize(image, dsize, interpolation=cv2.INTER_LINEAR)


def postprocess_cpu(output: np.ndarray, scale: float, pad_w: int, pad_h: int):
    """参考实现：decode + 阈值 + NMS（与 server_trtapi.py 一致）。
    CPU 后处理：解码 + 置信度过滤 + NMS"""
    import cv2
    # output[0].T 假设输出 shape 为 (1, N, C)，若模型导出时 batch 维度被 squeeze，此处会出错。确保导出时保留 batch 维度。
    preds = output[0].T  # [num_boxes, 4+6] ； Transpose: (num_boxes, 4+6)，每行一个候选框
    boxes_xywh = preds[:, :4] # 前4列：cx, cy, w, h（中心点坐标+宽高格式）
    scores_all = preds[:, 4:] # 后6列：每个类别的概率
    class_ids = scores_all.argmax(axis=1) # 每个框的最高概率类别索引
    confs = scores_all.max(axis=1) # 对应的最高概率值

    # 置信度过滤：丢弃低置信度框
    mask = confs > CONF_THRESH
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return []

    # 坐标还原：模型输出是 640×640 空间中的 xywh，需转回原图 xyxy
    x, y, w, h = boxes_xywh.T
    x1 = (x - w / 2 - pad_w) / scale # 减填充 → 除缩放 → 得到原图左x
    y1 = (y - h / 2 - pad_h) / scale # 减填充 → 除缩放 → 得到原图上y
    x2 = (x + w / 2 - pad_w) / scale
    y2 = (y + h / 2 - pad_h) / scale

    # OpenCV DNN 模块的 NMS 实现（高效且经过充分测试）
    indices = cv2.dnn.NMSBoxes(
        [[float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])] for i in range(len(x1))],
        confs.tolist(), CONF_THRESH, NMS_THRESH,
    )

    # 兼容不同 OpenCV 版本的返回格式（有的返回 list[list[int]]，有的返回 list[int]）
    indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

    # 组装最终结果：(x1, y1, x2, y2, confidence, class_id)
    return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
             float(confs[i]), int(class_ids[i])) for i in indices]


def draw_results(image: np.ndarray, detections: list) -> np.ndarray:
    """在原图上绘制检测框和标签（原地修改 image）；直接修改传入的 image 数组，若需保留原图应传入 .copy()"""
    import cv2
    for x1, y1, x2, y2, conf, cls_id in detections:
        # 绿色矩形框
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 类别名 + 置信度文字（位于框上方）
        cv2.putText(image, f"cls{cls_id} {conf:.2f}", (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image
