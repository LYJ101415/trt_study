"""
TensorRT 本地推理 — 纯 TRT API 版（零 PyTorch / 零 PyCUDA）
最小部署体积: TRT wheel + numpy ≈ 180MB
用法: python infer.py image.jpg
      python infer.py images/
"""

import sys
import time
import glob
import ctypes
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

# ==================== 配置区 ====================
ENGINE_PATH = "/root/my_FILE/models/yolov8_int8_dynamic.engine"
INPUT_NAME = "images"
OUTPUT_NAME = "output0"
IMG_SIZE = 640
CONF_THRESH = 0.45
NMS_THRESH = 0.65
NUM_CLASSES = 6

# ---------- ctypes 封装 cudaMalloc / cudaFree / cudaMemcpy ----------
_cudart = ctypes.cdll.LoadLibrary("libcudart.so")

# 🔑 关键：必须声明函数签名，否则 64 位指针/size_t 被截断为 32 位
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int

_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int

_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cudart.cudaMemcpy.restype = ctypes.c_int

_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int

_cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudart.cudaHostUnregister.restype = ctypes.c_int


def _check_cuda(err: int):
    if err != 0:
        raise RuntimeError(f"CUDA runtime error: {err}")


def cuda_malloc(nbytes: int) -> ctypes.c_void_p:
    ptr = ctypes.c_void_p()
    _check_cuda(_cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)))
    return ptr


def cuda_free(ptr: ctypes.c_void_p):
    _check_cuda(_cudart.cudaFree(ptr))


def cuda_memcpy_h2d(dst: ctypes.c_void_p, src: np.ndarray):
    assert src.flags['C_CONTIGUOUS'], "src must be C-contiguous"
    _check_cuda(_cudart.cudaMemcpy(
        dst, src.ctypes.data, ctypes.c_size_t(src.nbytes), ctypes.c_int(1)
    ))


def cuda_memcpy_d2h(dst: np.ndarray, src: ctypes.c_void_p):
    assert dst.flags['C_CONTIGUOUS'], "dst must be C-contiguous"
    _check_cuda(_cudart.cudaMemcpy(
        dst.ctypes.data, src, ctypes.c_size_t(dst.nbytes), ctypes.c_int(2)
    ))


def cuda_host_register(arr: np.ndarray):
    _check_cuda(_cudart.cudaHostRegister(
        arr.ctypes.data, ctypes.c_size_t(arr.nbytes), ctypes.c_uint(0)
    ))


def cuda_host_unregister(arr: np.ndarray):
    _check_cuda(_cudart.cudaHostUnregister(arr.ctypes.data))


def _get_ptr(ptr: ctypes.c_void_p) -> int:
    """安全提取 GPU 指针地址"""
    if hasattr(ptr, 'contents'):
        return ctypes.addressof(ptr.contents)
    return ptr.value
# --------------------------------------------------------------------


class TRTInference:
    """纯 TRT API 轻量推理引擎，支持动态 batch（固定 max shape 执行策略）"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # 🔑 修复：只从 profile 读取 INPUT 的 max shape
        profile_idx = 0
        min_in, opt_in, max_in = self.engine.get_tensor_profile_shape(INPUT_NAME, profile_idx)
        self.max_in_shape = tuple(max_in)
        self.max_batch = self.max_in_shape[0]

        # 🔑 修复：OUTPUT shape 通过 set_input_shape(max) + context.get_tensor_shape 获取
        self.context.set_input_shape(INPUT_NAME, list(self.max_in_shape))
        raw_out_shape = self.context.get_tensor_shape(OUTPUT_NAME)
        if -1 in raw_out_shape:
            raise RuntimeError(
                f"Output '{OUTPUT_NAME}' still has dynamic dim after set_input_shape: {raw_out_shape}"
            )
        self.max_out_shape = tuple(raw_out_shape)

        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(INPUT_NAME))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(OUTPUT_NAME))

        print(f"[TRT] Engine loaded | Dynamic batch (fixed-max execution)")
        print(f"  Profile: min={tuple(min_in)}, opt={tuple(opt_in)}, max={self.max_in_shape}")
        print(f"  Input:   {self.max_in_shape} {self.input_dtype}")
        print(f"  Output:  {self.max_out_shape} {self.output_dtype}")

        # --- GPU 显存分配（按 max shape）---
        in_nbytes = int(np.prod(self.max_in_shape)) * np.dtype(self.input_dtype).itemsize
        out_nbytes = int(np.prod(self.max_out_shape)) * np.dtype(self.output_dtype).itemsize

        self._d_input = cuda_malloc(in_nbytes)
        self._d_output = cuda_malloc(out_nbytes)

        # # --- CPU Pinned Memory ---
        self._h_input = np.empty(self.max_in_shape, dtype=self.input_dtype)
        self._h_output = np.empty(self.max_out_shape, dtype=self.output_dtype)
        cuda_host_register(self._h_input)
        cuda_host_register(self._h_output)

        # Warmup
        print("[TRT] Warming up...")
        dummy = np.zeros(self.max_in_shape, dtype=self.input_dtype)
        for _ in range(3):
            self._infer_raw(dummy)
        print("[TRT] Warmup done")

    def _infer_raw(self, blob: np.ndarray) -> np.ndarray:
        """
        固定 max shape 执行 → CUDA Graph safe
        blob: (actual_batch, C, H, W), actual_batch <= max_batch
        """
        actual_batch = blob.shape[0]
        assert actual_batch <= self.max_batch, \
            f"Batch {actual_batch} exceeds max_batch {self.max_batch}"

        # 拷贝实际数据到 pinned buffer，剩余填零
        self._h_input[:actual_batch] = blob
        if actual_batch < self.max_batch:
            self._h_input[actual_batch:] = 0

        # H2D 全量拷贝（地址/大小恒定 → graph safe）
        cuda_memcpy_h2d(self._d_input, self._h_input)

        # 🔑 始终用 max shape 执行，不调 set_input_shape
        self.context.execute_v2([_get_ptr(self._d_input), _get_ptr(self._d_output)])

        # D2H 全量拷贝
        cuda_memcpy_d2h(self._h_output, self._d_output)

        # CPU 端 slice 实际有效结果
        return self._h_output[:actual_batch].copy()

    def preprocess(self, image: np.ndarray):
        """Letterbox resize + normalize → [1, C, H, W] float32"""
        h, w = image.shape[:2]
        scale = min(IMG_SIZE / h, IMG_SIZE / w)
        nh, nw = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        pad_h = (IMG_SIZE - nh) // 2
        pad_w = (IMG_SIZE - nw) // 2
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis, ...], scale, pad_w, pad_h

    def postprocess(self, output: np.ndarray, scale: float,
                    pad_w: int, pad_h: int, orig_shape: tuple) -> list:
        preds = output[0].T  # [num_boxes, 4+num_classes]
        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]
        class_ids = scores_all.argmax(axis=1)
        confs = scores_all.max(axis=1)

        mask = confs > CONF_THRESH
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        x, y, w, h = boxes_xywh.T
        x1 = (x - w / 2 - pad_w) / scale
        y1 = (y - h / 2 - pad_h) / scale
        x2 = (x + w / 2 - pad_w) / scale
        y2 = (y + h / 2 - pad_h) / scale

        indices = cv2.dnn.NMSBoxes(
            [[x1[i], y1[i], x2[i], y2[i]] for i in range(len(x1))],
            confs.tolist(), CONF_THRESH, NMS_THRESH
        )
        indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                 float(confs[i]), int(class_ids[i])) for i in indices]

    def infer(self, image: np.ndarray) -> list:
        orig_shape = image.shape[:2]
        blob, scale, pad_w, pad_h = self.preprocess(image)
        output = self._infer_raw(blob)
        return self.postprocess(output, scale, pad_w, pad_h, orig_shape)

    def __del__(self):
        try:
            cuda_host_unregister(self._h_input)
            cuda_host_unregister(self._h_output)
            cuda_free(self._d_input)
            cuda_free(self._d_output)
        except Exception:
            pass


def draw_results(image: np.ndarray, detections: list) -> np.ndarray:
    COLORS = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=int)
    for x1, y1, x2, y2, conf, cls_id in detections:
        color = tuple(COLORS[cls_id].tolist())
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"cls{cls_id} {conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def main():
    target = "/root/my_FILE/my_FILE/test_images"
    engine = TRTInference(ENGINE_PATH)

    if Path(target).is_dir():
        paths = sorted(glob.glob(str(Path(target) / "*.[jp][pn]g")))
    else:
        paths = [target]

    print(f"\n{'='*50}")
    print(f"Processing {len(paths)} image(s)...")
    print(f"{'='*50}\n")

    total_time = 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Cannot read: {path}")
            continue

        t0 = time.perf_counter()
        detections = engine.infer(image)
        elapsed = (time.perf_counter() - t0) * 1000
        total_time += elapsed

        print(f"[{Path(path).name}] {len(detections)} objects | {elapsed:.1f} ms")
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            print(f"  cls={cls} conf={conf:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        OUTPUT_DIR = Path("./images_results")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"result_{Path(path).stem}.jpg"
        counter = 1
        while out_path.exists():
            out_path = OUTPUT_DIR / f"result_{Path(path).stem}_{counter}.jpg"
            counter += 1

        vis = draw_results(image.copy(), detections)
        cv2.imwrite(str(out_path), vis)
        print(f"  → Saved: {out_path}")

    avg = total_time / max(len(paths), 1)
    print(f"\n{'='*50}")
    print(f"Average: {avg:.1f} ms/image | FPS: {1000/avg:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()