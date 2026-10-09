"""
server_e2e.py — 静态端到端 engine 推理（宿主 CPU 只做 decode + resize，其余全在 engine）。

宿主流程:
  cv2.imread(原始图) -> CPU letterbox 到 640×640 -> H2D(uint8) + H2D(两个阈值标量)
  -> execute_v2 -> D2H([max_dets, 6] 最终框)

engine 内部完成 BGR→RGB/归一化 + 主干推理 + decode/NMS；输出 [max_dets, 6]，
坐标为 640×640 输入空间（如需画回原图，宿主按 letterbox 参数反算，见 main）。

用法:  python server_e2e.py image.jpg
       python server_e2e.py images/
       python server_e2e.py image.jpg --conf 0.45 --iou 0.65
"""

import sys
import time
import glob
import argparse
import ctypes
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dependency_file"))
from common import cuda_malloc, cuda_free, cuda_memcpy_h2d, cuda_memcpy_d2h, device_addr  # noqa: E402

ENGINE_PATH = "/root/my_FILE/models/yolov8_int8_static_e2e.engine"
IMG_SIZE = 640
MAX_DETS = 300
PAD_VALUE = 114


def letterbox_640(image: np.ndarray):
    """CPU 侧 decode + resize：等比缩放 + 居中灰边到 640×640，返回 canvas 与反算参数。"""
    h, w = image.shape[:2]
    scale = min(IMG_SIZE / h, IMG_SIZE / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMG_SIZE, IMG_SIZE, 3), PAD_VALUE, dtype=np.uint8)
    pad_h, pad_w = (IMG_SIZE - nh) // 2, (IMG_SIZE - nw) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return canvas, scale, pad_w, pad_h


class TRTInferenceE2E:
    """静态端到端 engine：640×640 uint8 图 + 两个阈值标量 → 最终框。"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")
        self.context = self.engine.create_execution_context()

        # 按 engine 的 I/O 张量顺序记录名称（execute_v2 需要按此顺序传地址）
        self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.io_names
                       if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.io_names
                        if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        print(f"[TRT-E2E] inputs={self.inputs} outputs={self.outputs}")

        # 显存缓冲：输入图(640×640×3 uint8) + 两个阈值标量(4B) + 输出(max_dets×6×4B)
        self._d_raw = cuda_malloc(IMG_SIZE * IMG_SIZE * 3)
        self._d_iou = cuda_malloc(4)
        self._d_score = cuda_malloc(4)
        self._d_out = cuda_malloc(MAX_DETS * 6 * 4)
        self.h_out = np.empty((MAX_DETS, 6), dtype=np.float32)

        self._addr = {
            "image_raw": device_addr(self._d_raw),
            "iou_thresh": device_addr(self._d_iou),
            "score_thresh": device_addr(self._d_score),
        }
        for o in self.outputs:
            self._addr[o] = device_addr(self._d_out)
        self._bindings = [self._addr[n] for n in self.io_names]

        print("[TRT-E2E] Warming up...")
        dummy = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        for _ in range(3):
            self.infer(dummy, conf=0.45, iou=0.65)
        print("[TRT-E2E] Warmup done")

    def infer(self, image_640: np.ndarray, conf: float, iou: float) -> list:
        # image_640 必须是 640×640 BGR uint8（宿主已完成 decode + resize）
        cuda_memcpy_h2d(self._d_raw, np.ascontiguousarray(image_640))
        cuda_memcpy_h2d(self._d_iou, np.array([iou], dtype=np.float32))
        cuda_memcpy_h2d(self._d_score, np.array([conf], dtype=np.float32))
        self.context.execute_v2(self._bindings)
        cuda_memcpy_d2h(self.h_out, self._d_out)

        # 过滤 padding 行（空槽 conf=0）
        dets = self.h_out[self.h_out[:, 4] > 0]
        return [(float(x1), float(y1), float(x2), float(y2), float(c), int(k))
                for x1, y1, x2, y2, c, k in dets]

    def __del__(self):
        for buf in (self._d_raw, self._d_iou, self._d_score, self._d_out):
            try:
                cuda_free(buf)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", nargs="?", default="/root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test_scaled/12000001.jpg")
    ap.add_argument("--engine", default=ENGINE_PATH)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.65)
    args = ap.parse_args()

    engine = TRTInferenceE2E(args.engine)
    target = args.target
    if Path(target).is_dir():
        paths = sorted(glob.glob(str(Path(target) / "*.[jp][pn]g")))
    else:
        paths = [target]

    print(f"\n{'=' * 50}\nProcessing {len(paths)} image(s) at conf={args.conf}, iou={args.iou}\n{'=' * 50}\n")
    total = 0.0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Cannot read: {path}")
            continue
        # 宿主 CPU 唯一保留的预处理：decode + resize
        canvas, scale, pad_w, pad_h = letterbox_640(image)

        t0 = time.perf_counter()
        detections = engine.infer(canvas, args.conf, args.iou)
        elapsed = (time.perf_counter() - t0) * 1000
        total += elapsed
        print(f"[{Path(path).name}] {len(detections)} objects | {elapsed:.1f} ms")
        for x1, y1, x2, y2, conf, cls in detections:
            print(f"  cls={cls} conf={conf:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        # 画回原图：640 空间 → 原图空间
        def to_orig(x, y):
            return (x - pad_w) / scale, (y - pad_h) / scale

        out_dir = Path("/root/my_FILE/trt_study/TRT_inferce_v2/e2e_static/images_results_e2e_static")
        out_dir.mkdir(parents=True, exist_ok=True)
        drawn = image.copy()
        for x1, y1, x2, y2, conf, cls in detections:
            ox1, oy1 = to_orig(x1, y1)
            ox2, oy2 = to_orig(x2, y2)
            cv2.rectangle(drawn, (int(ox1), int(oy1)), (int(ox2), int(oy2)), (0, 255, 0), 2)
            cv2.putText(drawn, f"cls{cls} {conf:.2f}", (int(ox1), int(oy1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(out_dir / f"result_{Path(path).stem}.jpg"), drawn)

    avg = total / max(len(paths), 1)
    print(f"\n{'=' * 50}\nAverage: {avg:.1f} ms/image | FPS: {1000 / avg:.1f}\n{'=' * 50}")


if __name__ == "__main__":
    main()
