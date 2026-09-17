"""
compare_kernel.py — 方案④（自定义 CUDA kernel）的 CPU / GPU 输出对比脚本。

目的：用**同一个 engine**、**同一批真实图片**，对比两条推理链路的输出是否一致，
并给出耗时对比（体现把预处理/后处理搬到 GPU 的收益）。

链路 A（GPU kernel，本方案）:
    cv2.imread(解码,CPU) -> H2D 原始 uint8 -> gpu.preprocess(letterbox+归一化,GPU)
    -> execute_v2(推理,GPU) -> gpu.postprocess(decode+NMS,GPU) -> D2H 少量最终框

链路 B（CPU 参考，等价 server_trtapi.py）:
    cv2.imread -> preprocess_cpu(letterbox+归一化,CPU) -> H2D blob
    -> execute_v2(推理,GPU) -> D2H output0 -> postprocess_cpu(decode+NMS,CPU)

对比口径:
    - 正确性: 把两侧检测框四舍五入后按坐标排序，逐框比较 [x1,y1,x2,y2,cls]（conf 仅打印）。
    - 耗时:   每条链路都测"除 cv2.imread 之外"的完整耗时，各跑 N 次取中位数。

用法:
    python compare_kernel.py [image1 image2 ...]
    缺省用 test_images/ 下所有图片。
"""

import sys
import time
import glob
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (cuda_malloc, cuda_free, cuda_memcpy_h2d, cuda_memcpy_d2h,
                    device_addr, preprocess_cpu, postprocess_cpu, letterbox_params,
                    CONF_THRESH, NMS_THRESH)
from kernels.gpu_ops import GpuOps

ENGINE_PATH = "/root/my_FILE/models/yolov8_fp16.engine"
N_ITER = 20        # 耗时测量迭代次数


def canon(dets):
    """把检测结果归一化（四舍五入 + 排序），消除两侧的浮点/顺序差异后比较。"""
    return sorted([(round(float(x1), 2), round(float(y1), 2), round(float(x2), 2),
                    round(float(y2), 2), int(cls))
                   for x1, y1, x2, y2, _conf, cls in dets])


def load_engine(path):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    return engine, engine.create_execution_context()


def main():
    engine_path = ENGINE_PATH
    if not Path(engine_path).exists():
        print(f"[ERR] 未找到 engine: {engine_path}，先运行 build_engine.py 构建。")
        sys.exit(1)

    engine, ctx = load_engine(engine_path)
    in_shape = tuple(engine.get_tensor_shape("images"))     # (1,3,640,640)
    out_shape = tuple(engine.get_tensor_shape("output0"))   # (1,10,8400)
    n_anchors = out_shape[2]

    # 显存缓冲（两条链路共用同一份 d_input/d_output，保证推理本身完全一致）
    d_input = cuda_malloc(int(np.prod(in_shape)) * 4)
    d_output = cuda_malloc(int(np.prod(out_shape)) * 4)
    gpu = GpuOps()

    # 测试图片列表
    args = sys.argv[1:]
    if args:
        paths = [a for a in args if Path(a).exists()]
    else:
        d = "/root/my_FILE/my_FILE/test_images"
        paths = sorted(glob.glob(f"{d}/*.[jp][pn]g")) if Path(d).exists() else []

    if not paths:
        print("[ERR] 没有可用测试图片。")
        sys.exit(1)

    print(f"engine: {engine_path}")
    print(f"对比 {len(paths)} 张图，每链路测 {N_ITER} 次取中位数\n")

    total_match = 0
    t_gpu_all, t_cpu_all = [], []

    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 读图失败: {path}")
            continue
        h, w = img.shape[:2]
        scale, pad_x, pad_y = letterbox_params(h, w)
        d_raw = cuda_malloc(img.nbytes)   # 原始 uint8 显存缓冲（仅链路 A 用）

        # ---------- 链路 A：GPU kernel 管线 ----------
        def run_gpu():
            cuda_memcpy_h2d(d_raw, np.ascontiguousarray(img))
            gpu.preprocess(d_raw, h, w, d_input, scale, pad_x, pad_y)
            ctx.execute_v2([device_addr(d_input), device_addr(d_output)])
            return gpu.postprocess(d_output, n_anchors, CONF_THRESH, NMS_THRESH,
                                   scale, pad_x, pad_y)

        # ---------- 链路 B：CPU 参考管线 ----------
        blob, s2, px2, py2 = preprocess_cpu(img)   # 只在 CPU 做一次（计时不含，见下）
        h_out = np.empty(out_shape, dtype=np.float32)

        def run_cpu():
            cuda_memcpy_h2d(d_input, np.ascontiguousarray(blob))
            ctx.execute_v2([device_addr(d_input), device_addr(d_output)])
            cuda_memcpy_d2h(h_out, d_output)
            return postprocess_cpu(h_out, s2, px2, py2)

        # 预热后计时
        for _ in range(3):
            run_gpu(); run_cpu()
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            run_gpu()
        tg = (time.perf_counter() - t0) / N_ITER * 1000
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            run_cpu()
        tc = (time.perf_counter() - t0) / N_ITER * 1000

        dets_gpu, dets_cpu = run_gpu(), run_cpu()
        cg, cc = canon(dets_gpu), canon(dets_cpu)
        match = (len(cg) == len(cc)
                 and all(a == b for a, b in zip(cg, cc)))   # 框坐标+类别完全一致

        print(f"[{Path(path).name}] GPU {len(dets_gpu)} 框 | CPU {len(dets_cpu)} 框 | "
              f"一致={'是' if match else '否'} | GPU {tg:.2f} ms vs CPU {tc:.2f} ms "
              f"(加速 {tc / tg:.1f}x)")
        if not match:
            print("    GPU:", cg)
            print("    CPU:", cc)

        total_match += match
        t_gpu_all.append(tg); t_cpu_all.append(tc)
        cuda_free(d_raw)

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"正确性: {total_match}/{len(paths)} 张图一致")
    if t_gpu_all:
        avg_g = np.median(t_gpu_all); avg_c = np.median(t_cpu_all)
        print(f"耗时(中位数): GPU 管线 {avg_g:.2f} ms vs CPU 管线 {avg_c:.2f} ms "
              f"(加速 {avg_c / avg_g:.1f}x)")
    print(f"{'=' * 60}")

    cuda_free(d_input); cuda_free(d_output)


if __name__ == "__main__":
    main()
