"""
infer_e2e_dyn.py — 全动态(batch,H,W) e2e INT8 引擎的构建与推理验证（引擎检查工具）。

配套模型: yolov8_int8_exclude_e2e_dyn.onnx（graphsurgeon_e2e_conf_iou_dyn.py 产出）
  输入  image_raw (B,H,W,3) uint8 BGR，B/H/W 全动态
        iou_thresh (1,) f32 / score_thresh (1,) f32 —— 静态，整批共享
  输出  detections (B,300,6) f32 = [x1,y1,x2,y2,conf,cls]，原图坐标、已 NMS，
        每张图占各自 rows[0:n_b]，空行 conf=0

用途: 每次【构建引擎后】跑一遍，检查引擎是否构建正确、动态维是否生效:
    1) batch=1 任意尺寸（不限于 640）单图推理，与 FP32 参考对齐
    2) batch=2 两张不同内容的图，验证各图结果不串扰
    3) 同一引擎切换 conf=0.001（mAP 场景），验证运行时阈值生效
注意: 对齐参考必须用 FP32 模型（best_PCB.onnx）——ORT CPU 跑 Q/DQ 量化模型
      的输出与 FP32/TRT 偏差很大，不能当基准。

实际场景推理（单图/文件夹/视频多 batch）用 e2e_dyn/infer_e2e_dyn.py。

用法:
    python infer_e2e_dyn.py            # 引擎不存在则构建，然后 batch=1/2 推理验证
    python infer_e2e_dyn.py --engine xxx.engine --onnx yyy.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # parents[2]上两级目录，下面导入的common.py在上两级目录中
from common import cuda_malloc, device_addr, cuda_memcpy_h2d, cuda_memcpy_d2h  # noqa: E402

MAX_BATCH = 8
MAX_HW = 1920          # profile 的 H/W 上限，超出会推理失败
MAX_DETS = 300


class E2EDynEngine:
    """全动态 e2e 引擎封装。显存按 profile 上限一次性分配，推理时只改 shape 不重新 malloc。"""

    def __init__(self, engine_path, logger=None):
        self.logger = logger or trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        # ---- 按 profile 上限预分配显存（一次分配，服务期复用）----
        self._d_raw = cuda_malloc(MAX_BATCH * MAX_HW * MAX_HW * 3)       # uint8
        self._d_iou = cuda_malloc(4)                                     # f32 [1]
        self._d_score = cuda_malloc(4)                                   # f32 [1]
        self._d_out = cuda_malloc(MAX_BATCH * MAX_DETS * 6 * 4)          # f32 (B,300,6)

        self.ctx.set_input_shape("iou_thresh", (1,))                     # 静态输入只声明一次
        self.ctx.set_input_shape("score_thresh", (1,))
        self.ctx.set_tensor_address("iou_thresh", device_addr(self._d_iou))
        self.ctx.set_tensor_address("score_thresh", device_addr(self._d_score))
        self.ctx.set_tensor_address("detections", device_addr(self._d_out))

        self._h_out = np.empty((MAX_BATCH, MAX_DETS, 6), np.float32)     # CPU 接收缓冲
        self.iou_in = np.zeros(1, np.float32)
        self.score_in = np.zeros(1, np.float32)

    def infer(self, images, iou=0.65, conf=0.45):
        """images: (B,H,W,3) uint8 BGR，B/H/W 任意(不超过 profile 上限)。返回 (B,300,6) float32。"""
        images = np.ascontiguousarray(images)
        b, h, w, _ = images.shape
        assert b <= MAX_BATCH and max(h, w) <= MAX_HW, \
            f"({b},{h},{w}) 超出 profile 上限 batch<={MAX_BATCH}, H/W<={MAX_HW}"

        # ★ 动态形状的核心两步：声明形状 → 绑定地址（地址固定则只需第一次，这里每次重绑无害）
        self.ctx.set_input_shape("image_raw", images.shape)
        self.ctx.set_tensor_address("image_raw", device_addr(self._d_raw))

        self.iou_in[0] = iou        # ★ 运行时阈值：同一引擎跑推理(0.45)或 mAP(0.001) 都不用重编
        self.score_in[0] = conf

        cuda_memcpy_h2d(self._d_raw, images.view(np.uint8).reshape(-1))
        cuda_memcpy_h2d(self._d_iou, self.iou_in)
        cuda_memcpy_h2d(self._d_score, self.score_in)

        ok = self.ctx.execute_async_v3(0)
        if not ok:
            raise RuntimeError("execute_async_v3 failed")

        cuda_memcpy_d2h(self._h_out[:b], self._d_out)
        return self._h_out[:b].copy()


def build_engine_if_needed(engine_path, onnx_path):
    """复用 build_engine_e2e.build：动态维范围由 build_engine_e2e.py 全局配置区自动生成。"""
    if Path(engine_path).exists():
        print(f"[skip] 引擎已存在: {engine_path}")
        return
    from build_engine_e2e import build
    build(onnx_path, engine_path, fp16=True, int8=True)


# ---------------- 验证：与 FP32 原模型+CPU后处理 对齐 ----------------

def _ref_dets(ref_sess, img, conf, iou):
    from common import preprocess_cpu, letterbox_params
    scale, pad_x, pad_y = letterbox_params(img.shape[0], img.shape[1])
    blob, *_ = preprocess_cpu(img)
    preds = ref_sess.run(None, {"images": blob})[0][0].T
    boxes, scores = preds[:, :4], preds[:, 4:]
    cls = scores.argmax(1); confs = scores.max(1)
    m = confs > conf
    boxes, confs, cls = boxes[m], confs[m], cls[m]
    if len(boxes) == 0:
        return []
    x, y, w, h = boxes.T
    x1 = (x - w/2 - pad_x) / scale; y1 = (y - h/2 - pad_y) / scale
    x2 = (x + w/2 - pad_x) / scale; y2 = (y + h/2 - pad_y) / scale
    idx = cv2.dnn.NMSBoxes([[float(a), float(b), float(c), float(d)]
                            for a, b, c, d in zip(x1, y1, x2, y2)],
                           confs.tolist(), conf, iou)
    idx = [i[0] if isinstance(i, (list, tuple)) else i for i in idx]
    return sorted((round(float(x1[i]), 2), round(float(y1[i]), 2), round(float(x2[i]), 2),
                   round(float(y2[i]), 2), round(float(confs[i]), 2), int(cls[i])) for i in idx)


def _canon(d):
    return sorted((round(float(a), 2), round(float(b), 2), round(float(c), 2),
                   round(float(d), 2), round(float(e), 2), int(f))
                  for a, b, c, d, e, f in d)


def _same(cc, ce, tol=2.0):
    """INT8 引擎 vs FP32 参考的坐标偏差正常在 1~2px 内(量化误差)，默认容差 2px。"""
    return len(cc) == len(ce) and all(
        all(abs(x - y) <= tol for x, y in zip(a[:4], b[:4])) and a[5] == b[5]
        for a, b in zip(cc, ce))


def main(args):
    build_engine_if_needed(args.engine, args.onnx)
    eng = E2EDynEngine(args.engine)
    print("engine 输入:",
          [(eng.engine.get_tensor_name(i),
            str(eng.engine.get_tensor_dtype(eng.engine.get_tensor_name(i))))
           for i in range(eng.engine.num_io_tensors)
           if eng.engine.get_tensor_mode(eng.engine.get_tensor_name(i)) == trt.TensorIOMode.INPUT])

    import onnxruntime as ort
    ref = ort.InferenceSession(args.src_ref, providers=["CPUExecutionProvider"])

    img = cv2.imread(args.image)
    conf, iou = 0.45, 0.65

    # ---- batch=1，且故意不用 640：验证 H/W 真动态 ----
    out1 = eng.infer(img[None], iou=iou, conf=conf)
    d1 = _canon(out1[0][out1[0][:, 4] > 0])
    r1 = _canon(_ref_dets(ref, img, conf, iou))
    print(f"batch=1 (H={img.shape[0]},W={img.shape[1]}): {len(d1)} 框 vs 参考 {len(r1)} 框 -> "
          f"{'OK' if _same(r1, d1, args.tol) else 'DIFF'} (容差 {args.tol}px)")

    # ---- batch=2，两张不同内容的图（原图 + 镜像），验证不串扰 ----
    img2 = cv2.flip(img, 1)
    out2 = eng.infer(np.stack([img, img2]), iou=iou, conf=conf)
    print(f"batch=2 输出 shape: {out2.shape}")
    d2a = _canon(out2[0][out2[0][:, 4] > 0])
    d2b = _canon(out2[1][out2[1][:, 4] > 0])
    r2b = _canon(_ref_dets(ref, img2, conf, iou))
    print(f"  图0: {len(d2a)} 框 vs 参考 {len(r1)} -> {'OK' if _same(r1, d2a, args.tol) else 'DIFF'}")
    print(f"  图1: {len(d2b)} 框 vs 参考 {len(r2b)} -> {'OK' if _same(r2b, d2b, args.tol) else 'DIFF'}")

    # ---- mAP 场景：同一引擎直接换低阈值，无需重编 ----
    out3 = eng.infer(img[None], iou=0.65, conf=0.001)
    print(f"conf=0.001(mAP 场景): {int((out3[0][:, 4] > 0.001).sum())} 框")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/root/my_FILE/models/yolov8_int8_exclude_e2e_dyn.onnx")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_e2e_dyn.engine")
    ap.add_argument("--src-ref", default="/root/my_FILE/models/best_PCB.onnx",
                    help="对齐验证用的参考 ONNX。注意: 必须用 FP32 模型！"
                         "ONNXRuntime CPU 对 Q/DQ 量化模型的推理结果与 FP32/TRT 偏差很大"
                         "(实测 max conf 0.407 vs FP32 0.901)，不能当精度基准")
    ap.add_argument("--image", default="/root/my_FILE/my_trt_FILE/my_FILE/test_images/00041200.jpg")
    ap.add_argument("--tol", type=float, default=2.0, help="INT8 vs FP32 的坐标对齐容差(像素)")
    args = ap.parse_args()
    main(args)
