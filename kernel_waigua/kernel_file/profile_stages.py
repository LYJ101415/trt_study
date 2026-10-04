"""
profile_stages.py — 逐段耗时测试（CPU letterbox / H2D / GPU 归一化转通道 / TRT 推理 / GPU 后处理）。

用于量化「Resize 移到 CPU」后每一段的真实耗时，支撑改动分析：
    - 每段 GPU 操作后紧跟 cudaDeviceSynchronize，测的是该段的墙钟时间（含 launch 开销）；
    - 额外打印高分辨率输入下「旧方案 H2D(整幅原图) vs 新方案 H2D(640×640)」的字节量对比。

用法:
    python profile_stages.py
    python profile_stages.py --image <path> --engine <path> --n 500
"""

import sys
import time
import ctypes
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

# 让脚本从任意 cwd 运行都能找到 common / gpu_ops
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "dependency_file"))
sys.path.insert(0, str(_HERE / "kernels"))

from common import (cuda_malloc, cuda_free, cuda_memcpy_h2d, device_addr, letterbox_cpu)
from gpu_ops import GpuOps

# cudaDeviceSynchronize：用于测每段 GPU 操作的墙钟时间
_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaDeviceSynchronize.argtypes = []
_cudart.cudaDeviceSynchronize.restype = ctypes.c_int
_sync = _cudart.cudaDeviceSynchronize

IMG_SIZE = 640
N_ANCHORS = 8400
NUM_CLASSES = 6


def load_engine(engine_path: str):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()
    in_shape = tuple(engine.get_tensor_shape("images"))
    out_shape = tuple(engine.get_tensor_shape("output0"))
    n_anchors = out_shape[2]
    d_in = cuda_malloc(int(np.prod(in_shape)) * 4)
    d_out = cuda_malloc(int(np.prod(out_shape)) * 4)
    d_raw = cuda_malloc(IMG_SIZE * IMG_SIZE * 3)
    return ctx, in_shape, out_shape, n_anchors, d_in, d_out, d_raw


def main():
    ap = argparse.ArgumentParser(description="逐段耗时测试（CPU letterbox + GPU 管线）")
    ap.add_argument("--image", default="/root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test_scaled/00041201.jpg")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_static.engine")
    ap.add_argument("--n", type=int, default=200, help="测时迭代次数")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    assert img is not None, f"cannot read {args.image}"
    print(f"image: {args.image}  shape={img.shape}  iterations={args.n}")

    ctx, in_shape, out_shape, n_anchors, d_in, d_out, d_raw = load_engine(args.engine)
    g = GpuOps(num_classes=NUM_CLASSES, max_dets=300)
    print(f"engine: {args.engine}  input={in_shape}  output={out_shape}\n")

    def one_frame():
        canvas, scale, px, py = letterbox_cpu(img)
        cuda_memcpy_h2d(d_raw, canvas)
        g.preprocess(d_raw, d_in)
        ctx.execute_v2([device_addr(d_in), device_addr(d_out)])
        g.postprocess(d_out, n_anchors, 0.25, 0.65, scale, px, py)

    # warmup
    for _ in range(20):
        one_frame()
    _sync()

    n = args.n
    t_lb = t_h2d = t_pre = t_inf = t_post = 0.0
    for _ in range(n):
        t0 = time.perf_counter(); canvas, scale, px, py = letterbox_cpu(img); t_lb += time.perf_counter() - t0
        t0 = time.perf_counter(); cuda_memcpy_h2d(d_raw, canvas); _sync(); t_h2d += time.perf_counter() - t0
        t0 = time.perf_counter(); g.preprocess(d_raw, d_in); _sync(); t_pre += time.perf_counter() - t0
        t0 = time.perf_counter(); ctx.execute_v2([device_addr(d_in), device_addr(d_out)]); _sync(); t_inf += time.perf_counter() - t0
        t0 = time.perf_counter(); g.postprocess(d_out, n_anchors, 0.25, 0.65, scale, px, py); t_post += time.perf_counter() - t0

    print(f"=== 逐段耗时（n={n}，每段 GPU 操作后同步）===")
    print(f"  CPU letterbox (cv2.resize+pad) : {t_lb/n*1e3:7.3f} ms")
    print(f"  H2D 640x640x3 (cudaMemcpy)     : {t_h2d/n*1e3:7.3f} ms")
    print(f"  GPU normalize+transpose kernel : {t_pre/n*1e3:7.3f} ms")
    print(f"  TRT execute_v2 (INT8)          : {t_inf/n*1e3:7.3f} ms")
    print(f"  GPU postprocess (decode+NMS)   : {t_post/n*1e3:7.3f} ms")

    # 高分辨率输入下 H2D 数据量对比：旧方案传整幅原图 vs 新方案传 640×640
    big = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    canvas_big, _, _, _ = letterbox_cpu(big)
    print(f"\n=== 高分辨率 1920x1080 输入下的 H2D 数据量对比 ===")
    print(f"  旧方案 H2D(整幅原图) : {big.nbytes/1e6:.1f} MB")
    print(f"  新方案 H2D(640x640)  : {canvas_big.nbytes/1e6:.1f} MB")

    cuda_free(d_in); cuda_free(d_out); cuda_free(d_raw)


if __name__ == "__main__":
    main()
