"""
eval_map.py — 用「运行时阈值」端到端 engine 跑 COCO mAP。

一个 engine 同时服务推理和评测：conf 作为运行时输入，conf=0.001 跑 mAP，
conf=0.45 跑推理，无需重编。

用法:
    python eval_map.py --engine /root/my_FILE/models/yolov8_e2e_conf_fp16.engine \
                       --images .../images/test --ann .../test_coco.json
"""

import sys
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server_e2e_conf import TRTInferenceE2EConf


def run_eval(engine_path, images_dir, ann_json, output_json, conf=0.001, iou=0.65):
    coco_gt = COCO(ann_json)
    image_ids = coco_gt.getImgIds()
    print(f"Found {len(image_ids)} images. engine={Path(engine_path).name} conf={conf} iou={iou}")

    engine = TRTInferenceE2EConf(engine_path)
    results = []

    for img_id in tqdm(image_ids, desc="TRT Inference"):
        info = coco_gt.loadImgs(img_id)[0]
        img = cv2.imread(str(Path(images_dir) / info["file_name"]))
        if img is None:
            raise FileNotFoundError(info["file_name"])

        dets = engine.infer(img, conf, iou)   # 运行时阈值

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
    ap = argparse.ArgumentParser(description="运行时阈值端到端 engine 的 mAP 评测")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_e2e_conf.engine")
    ap.add_argument("--images", default="/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/test")
    ap.add_argument("--ann", default="/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/test_coco.json")
    ap.add_argument("--output", default="detections_conf.json")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.65)
    args = ap.parse_args()
    run_eval(args.engine, args.images, args.ann, args.output, args.conf, args.iou)
