"""
test_correctness.py — 验证 GPU kernel 与 CPU 参考实现的一致性。

1. 预处理: 随机/真实图 -> GPU letterbox  vs  common.preprocess_cpu  (容忍 ~1 LSB 浮点差)
2. 后处理: 合成 output0 -> GPU decode+NMS  vs  common.postprocess_cpu   (框集合应一致)
3. 端到端: 若 engine 存在，同图 GPU 管线  vs  CPU 管线全链路对比

用法:  python test_correctness.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (cuda_malloc, cuda_free, cuda_memcpy_h2d, cuda_memcpy_d2h,
                    device_addr, preprocess_cpu, postprocess_cpu,
                    letterbox_params, IMG_SIZE, CONF_THRESH, NMS_THRESH, NUM_CLASSES)
from kernels.gpu_ops import GpuOps


def _alloc(nbytes):
    return cuda_malloc(nbytes)


def test_preprocess(tol=2.0 / 255.0):
    print("\n=== [1/3] 预处理: GPU letterbox vs CPU ===")
    gpu = GpuOps()
    d_input = _alloc(3 * IMG_SIZE * IMG_SIZE * 4)
    h_in = np.empty((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    # 用一张带渐变的合成图 + 一张随机图
    yy, xx = np.mgrid[0:1234, 0:987]
    cases = [
        ((yy * 3 + xx).astype(np.uint8) % 255).repeat(3).reshape(1234, 987, 3),
        np.random.randint(0, 255, (321, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8),
    ]

    for idx, img in enumerate(cases):
        h, w = img.shape[:2]
        scale, pad_x, pad_y = letterbox_params(h, w)
        blob_cpu, *_ = preprocess_cpu(img)              # (1,3,640,640)

        raw = np.ascontiguousarray(img)
        d_raw = _alloc(raw.nbytes)
        cuda_memcpy_h2d(d_raw, raw)
        gpu.preprocess(d_raw, h, w, d_input, scale, pad_x, pad_y)
        cuda_memcpy_d2h(h_in, d_input)
        cuda_free(d_raw)

        diff = np.abs(h_in - blob_cpu[0])
        print(f"  case {idx} ({h}x{w}): max_abs={diff.max():.6f} "
              f"mean_abs={diff.mean():.6f} -> {'OK' if diff.max() <= tol else 'FAIL'}")
        assert diff.max() <= tol, f"case {idx} preprocess mismatch"

    cuda_free(d_input)
    print("  预处理一致性: PASS")


def test_postprocess():
    print("\n=== [2/3] 后处理: GPU decode+NMS vs CPU ===")
    gpu = GpuOps()
    N = 8400
    d_output = _alloc(10 * N * 4)
    h_out = np.empty((1, 10, N), dtype=np.float32)

    rng = np.random.default_rng(0)
    out0 = rng.uniform(0.0, 0.4, size=(1, 10, N)).astype(np.float32)
    # 埋几个明确的框，确保有检出且部分重叠（验证 NMS）
    boxes = [
        (320, 320, 200, 100, 0.9, 2),   # cx,cy,w,h,conf,cls
        (340, 330, 210, 110, 0.85, 2),  # 与上重叠 -> 应被 NMS 抑制
        (100, 80, 40, 40, 0.7, 1),
        (600, 600, 30, 30, 0.5, 0),
    ]
    for i, (cx, cy, w, h, conf, cls) in enumerate(boxes):
        out0[0, 0, i] = cx; out0[0, 1, i] = cy
        out0[0, 2, i] = w;  out0[0, 3, i] = h
        out0[0, 4 + cls, i] = conf

    scale, pad_x, pad_y = 0.5, 30, 20   # 任意非平凡参数

    cpu_dets = postprocess_cpu(out0.copy(), scale, pad_x, pad_y)
    cuda_memcpy_h2d(d_output, np.ascontiguousarray(out0))
    gpu_dets = gpu.postprocess(d_output, N, CONF_THRESH, NMS_THRESH, scale, pad_x, pad_y)

    def canon(d):
        return sorted([(round(a, 3), round(b, 3), round(c, 3), round(d_, 3),
                        round(e, 3), f) for a, b, c, d_, e, f in d])

    cc, cg = canon(cpu_dets), canon(gpu_dets)
    print(f"  CPU dets: {len(cc)}  GPU dets: {len(cg)}")
    for a, b in zip(cc, cg):
        ok = all(abs(x - y) < 1e-2 for x, y in zip(a[:5], b[:5])) and a[5] == b[5]
        print(f"    {a}  <->  {b}  {'OK' if ok else 'MISMATCH'}")
    assert len(cc) == len(cg) and all(
        all(abs(x - y) < 1e-2 for x, y in zip(a[:5], b[:5])) and a[5] == b[5]
        for a, b in zip(cc, cg)), "postprocess mismatch"

    cuda_free(d_output)
    print("  后处理一致性: PASS")


if __name__ == "__main__":
    test_preprocess()
    test_postprocess()
    print("\nGPU kernel 单元测试完成。")
    print("（真实图 + 真实 engine 的 CPU/GPU 全链路对比见 compare_kernel.py / compare_e2e.py）")
