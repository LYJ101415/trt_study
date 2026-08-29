"""
eval_kernel_map.py — 方案④（外挂 CUDA kernel）的 COCO mAP 评测。

用完整 GPU 管线（GPU 预处理 + engine + GPU 后处理）跑整份验证集，
conf/iou 阈值运行时可调，因此既能测推理(conf=0.45)也能测 mAP(conf=0.001)，
且切换 INT8 / FP16 engine 只需换 --engine，无需重编。

用法:
    python eval_kernel_map.py --engine /root/my_FILE/models/yolov8_int8_waigua.engine \
                              --images .../images/test --ann .../test_coco.json
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (cuda_malloc, cuda_free, cuda_memcpy_h2d, device_addr,
                    letterbox_params, NUM_CLASSES)
from kernels.gpu_ops import GpuOps


class KernelEngine:
    """方案④ 推理引擎：GPU 预处理 + TRT engine + GPU 后处理（conf/iou 运行时传入）。"""

    def __init__(self, engine_path: str, num_classes: int = NUM_CLASSES, max_dets: int = 300):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.context = engine.create_execution_context()

        in_shape = tuple(engine.get_tensor_shape("images"))
        out_shape = tuple(engine.get_tensor_shape("output0"))
        self.n_anchors = out_shape[2]
        self._d_input = cuda_malloc(int(np.prod(in_shape)) * 4)
        self._d_output = cuda_malloc(int(np.prod(out_shape)) * 4)
        self._d_raw = None
        self._raw_cap = 0
        self.gpu = GpuOps(num_classes=num_classes, max_dets=max_dets)
        print(f"[kernel-engine] loaded {engine_path} | input {in_shape} "
              f"| output {out_shape}")

    def infer(self, image: np.ndarray, conf_thresh: float, iou_thresh: float) -> list:
        h, w = image.shape[:2]
        scale, pad_x, pad_y = letterbox_params(h, w)
        raw = np.ascontiguousarray(image)
        if self._d_raw is None or raw.nbytes > self._raw_cap:
            if self._d_raw is not None:
                cuda_free(self._d_raw)
            self._d_raw = cuda_malloc(max(raw.nbytes, 4096 * 4096 * 3))
            self._raw_cap = raw.nbytes
        cuda_memcpy_h2d(self._d_raw, raw)
        self.gpu.preprocess(self._d_raw, h, w, self._d_input, scale, pad_x, pad_y)
        self.context.execute_v2([device_addr(self._d_input), device_addr(self._d_output)])
        return self.gpu.postprocess(self._d_output, self.n_anchors,
                                    conf_thresh, iou_thresh, scale, pad_x, pad_y)


def run_eval(engine_path, images_dir, ann_json, output_json, conf=0.001, iou=0.65):
    coco_gt = COCO(ann_json)
    image_ids = coco_gt.getImgIds()
    print(f"Found {len(image_ids)} images in validation set.")

    engine = KernelEngine(engine_path)
    results = []

    for img_id in tqdm(image_ids, desc="TRT Inference"):
        info = coco_gt.loadImgs(img_id)[0]
        img = cv2.imread(str(Path(images_dir) / info["file_name"]))
        if img is None:
            raise FileNotFoundError(info["file_name"])

        dets = engine.infer(img, conf, iou)   # 完整 GPU 管线

        for x1, y1, x2, y2, score, cls in dets:
            results.append({
                "image_id": img_id,
                "category_id": int(cls),
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "score": round(float(score), 4),
            })

    with open(output_json, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} detections to {output_json}")

    if not results:
        print("No detections, skip eval."); return

    coco_dt = coco_gt.loadRes(output_json)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"\n📊 mAP ({Path(engine_path).name}, conf={conf}):")
    print(f"  mAP@[0.50:0.95] = {ev.stats[0]:.4f}")
    print(f"  mAP@0.50        = {ev.stats[1]:.4f}")
    print(f"  mAP@0.75        = {ev.stats[2]:.4f}")
    return ev.stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="方案④ kernel 管线 COCO mAP 评测")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_waigua.engine")
    ap.add_argument("--images", default="/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/test")
    ap.add_argument("--ann", default="/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/test_coco.json")
    ap.add_argument("--output", default="detections.json")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    args = ap.parse_args()
    run_eval(args.engine, args.images, args.ann, args.output, args.conf, args.iou)
