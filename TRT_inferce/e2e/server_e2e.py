"""
server_e2e.py — 端到端 engine 推理（预处理 + 后处理已烧进 engine，单次 execute_v2）。

与 server_gpu.py 的差别：这里连 CUDA kernel 都不需要 —— 原始 uint8 图直接进 engine，
engine 内部完成 letterbox/normalize/decode/NMS，输出 [max_dets, 6] 最终框。
宿主只做: cv2.imread -> H2D(原始 uint8) -> set_input_shape -> execute_v2 -> D2H(小数组)。

用法:  python server_e2e.py image.jpg
       python server_e2e.py images/
"""

import sys
import time
import glob
import ctypes
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (cuda_malloc, cuda_free, cuda_memcpy_h2d, cuda_memcpy_d2h,
                    device_addr, draw_results)

ENGINE_PATH = "/root/my_FILE/models/yolov8_e2e_fp16.engine"
MAX_DETS = 300
MAX_RAW_HW = 1920  # 与 build_engine.py 的 max_shape 对齐


class TRTInferenceE2E:
    """端到端 engine：原始 uint8 图直出最终框，全程无 CPU 前后处理。"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        self.in_name = "image_raw"
        self.out_name = "detections"
        self._d_input = cuda_malloc(MAX_RAW_HW * MAX_RAW_HW * 3)
        self._d_output = cuda_malloc(MAX_DETS * 6 * 4)
        self.h_out = np.empty((MAX_DETS, 6), dtype=np.float32)

        print(f"[TRT-E2E] Engine loaded | input {self.in_name} (dynamic H/W uint8) "
              f"| output {self.out_name} [{MAX_DETS}, 6]")

        print("[TRT-E2E] Warming up...")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(3):
            self.infer(dummy)
        print("[TRT-E2E] Warmup done")

    def infer(self, image: np.ndarray) -> list:
        h, w = image.shape[:2]
        raw = np.ascontiguousarray(image)
        self.context.set_input_shape(self.in_name, (1, h, w, 3))
        cuda_memcpy_h2d(self._d_input, raw)
        self.context.execute_v2([device_addr(self._d_input), device_addr(self._d_output)])
        cuda_memcpy_d2h(self.h_out, self._d_output)

        # 过滤 padding 行（空槽 conf=0）
        mask = self.h_out[:, 4] > 0
        dets = self.h_out[mask]
        return [(float(x1), float(y1), float(x2), float(y2), float(conf), int(cls))
                for x1, y1, x2, y2, conf, cls in dets]

    def __del__(self):
        for buf in (self._d_input, self._d_output):
            try:
                cuda_free(buf)
            except Exception:
                pass


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "/root/my_FILE/my_FILE/test_images"
    engine = TRTInferenceE2E(ENGINE_PATH)

    if Path(target).is_dir():
        paths = sorted(glob.glob(str(Path(target) / "*.[jp][pn]g")))
    else:
        paths = [target]

    print(f"\n{'=' * 50}\nProcessing {len(paths)} image(s)...\n{'=' * 50}\n")
    total = 0.0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Cannot read: {path}")
            continue
        t0 = time.perf_counter()
        detections = engine.infer(image)
        elapsed = (time.perf_counter() - t0) * 1000
        total += elapsed
        print(f"[{Path(path).name}] {len(detections)} objects | {elapsed:.1f} ms")
        for x1, y1, x2, y2, conf, cls in detections:
            print(f"  cls={cls} conf={conf:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        out_dir = Path("./images_results_e2e")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"result_{Path(path).stem}.jpg"),
                    draw_results(image.copy(), detections))

    avg = total / max(len(paths), 1)
    print(f"\n{'=' * 50}\nAverage: {avg:.1f} ms/image | FPS: {1000 / avg:.1f}\n{'=' * 50}")


if __name__ == "__main__":
    main()
