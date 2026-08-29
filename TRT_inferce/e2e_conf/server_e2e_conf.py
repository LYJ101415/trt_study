"""
server_e2e_conf.py — 端到端 engine（运行时阈值）推理服务。

与 e2e/server_e2e.py 的区别：conf / iou 不再是烧死在图里的常量，而是每次
execute 时作为标量输入传入。因此一个 engine 既能跑推理(conf=0.45)也能跑
mAP(conf=0.001)，现场调阈值不用重编。

用法:
    python server_e2e_conf.py image.jpg
    python server_e2e_conf.py images/ --conf 0.45 --iou 0.65
"""

import sys
import time
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cuda_malloc, cuda_free, cuda_memcpy_h2d, cuda_memcpy_d2h, device_addr, draw_results

ENGINE_PATH = "/root/my_FILE/models/yolov8_e2e_conf_fp16.engine"
MAX_DETS = 300
MAX_RAW_HW = 1920  # 与 build_engine.py 的 max_shape 对齐


class TRTInferenceE2EConf:
    """端到端 engine，conf/iou 运行时传入。"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

        # 按 engine 的 I/O 张量顺序记录名称（execute_v2 需要按此顺序传地址）
        self.io_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        self.inputs = [n for n in self.io_names
                       if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.io_names
                        if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        print(f"[E2E-conf] inputs={self.inputs} outputs={self.outputs}")

        # ---- 显存(Device)缓冲分配 ----
        # 为什么预处理/后处理已经烧进 engine，这里还要手动 cudaMalloc？
        #   engine 烧录的是「计算逻辑」(GPU kernel)，而不是「内存管理」。
        #   TensorRT 只负责按图执行计算，不负责分配/释放输入输出的显存——
        #   每个 tensor 的设备地址必须由调用方提供(execute_v2 按 io_names 顺序传入)。
        #   且 cudaMalloc/cudaFree 开销大，若每帧自动分配释放性能会崩，
        #   故在 __init__ 一次性分配、infer 循环复用、__del__ 统一释放(显存复用)。
        # 各 buffer 含义：
        #   _d_raw   : 输入原始图(uint8, 3 通道)。按 build 时 max_shape(1920) 开「最大容量」，
        #              每次用 set_input_shape 指定实际尺寸，只占用前 h*w*3 字节。
        #   _d_iou   : 运行时标量阈值 iou(float32, 4 字节)。本文件是 e2e_conf 变体，
        #              conf/iou 不再烧死在图里，改为每次推理时动态传入。
        #   _d_score : 运行时标量阈值 conf(float32, 4 字节)。
        #   _d_out   : 输出 detections，最多 MAX_DETS 个框 × 6 个 float32
        #              (x1,y1,x2,y2,confidence,class_id) = 300 × 6 × 4 字节。
        self._d_raw = cuda_malloc(MAX_RAW_HW * MAX_RAW_HW * 3)
        self._d_iou = cuda_malloc(4)
        self._d_score = cuda_malloc(4)
        self._d_out = cuda_malloc(MAX_DETS * 6 * 4)
        # h_out : CPU 侧接收缓冲区。GPU 结果必须 cudaMemcpy(D2H) 拷回 CPU 后
        #         numpy/python 才能读取；预分配避免每次新建数组。
        self.h_out = np.empty((MAX_DETS, 6), dtype=np.float32)

        self._addr = {
            "image_raw": device_addr(self._d_raw),
            "iou_thresh": device_addr(self._d_iou),
            "score_thresh": device_addr(self._d_score),
            "detections": device_addr(self._d_out),
        }
        self._bindings = [self._addr[n] for n in self.io_names]

        print("[E2E-conf] Warming up...")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(3):
            self.infer(dummy, conf=0.45, iou=0.65)
        print("[E2E-conf] Warmup done")

    def infer(self, image: np.ndarray, conf: float, iou: float) -> list:
        h, w = image.shape[:2]
        self.context.set_input_shape("image_raw", (1, h, w, 3))
        cuda_memcpy_h2d(self._d_raw, np.ascontiguousarray(image))
        cuda_memcpy_h2d(self._d_iou, np.array([iou], dtype=np.float32))
        cuda_memcpy_h2d(self._d_score, np.array([conf], dtype=np.float32))
        self.context.execute_v2(self._bindings)
        cuda_memcpy_d2h(self.h_out, self._d_out)

        dets = self.h_out[self.h_out[:, 4] > 0]   # 过滤 padding 行(conf=0)
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
    ap.add_argument("target", nargs="?", default="/root/my_FILE/my_FILE/test_images/00041200.jpg")
    ap.add_argument("--engine", default=ENGINE_PATH)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.65)
    args = ap.parse_args()

    engine = TRTInferenceE2EConf(args.engine)
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
        t0 = time.perf_counter()
        detections = engine.infer(image, args.conf, args.iou)
        elapsed = (time.perf_counter() - t0) * 1000
        total += elapsed
        print(f"[{Path(path).name}] {len(detections)} objects | {elapsed:.1f} ms")
        for x1, y1, x2, y2, conf, cls in detections:
            print(f"  cls={cls} conf={conf:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
        out_dir = Path("./images_results_e2e_conf")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"result_{Path(path).stem}.jpg"),
                    draw_results(image.copy(), detections))

    avg = total / max(len(paths), 1)
    print(f"\n{'=' * 50}\nAverage: {avg:.1f} ms/image | FPS: {1000 / avg:.1f}\n{'=' * 50}")


if __name__ == "__main__":
    main()
