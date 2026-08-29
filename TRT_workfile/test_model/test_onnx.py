import os
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def preprocess_image(image_path, input_size=(640, 640)):
    """预处理：BGR -> RGB, Resize with pad, Normalize to [0,1]"""
    img = cv2.imread(str(image_path))
    h0, w0 = img.shape[:2]
    r = min(input_size[0] / h0, input_size[1] / w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # Pad to square
    dw, dh = input_size[1] - new_w, input_size[0] - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # HWC -> CHW, BGR -> RGB, normalize
    padded = padded[:, :, ::-1].transpose(2, 0, 1)  # RGB, CHW
    img_tensor = np.ascontiguousarray(padded, dtype=np.float32) / 255.0
    return img_tensor, (r, (dw, dh))

def postprocess_yolo(pred, img_info, conf_thres=0.001, iou_thres=0.65, max_det=300):
    if pred.ndim == 3:
        pred = pred[0]  # [10, 8400]
    if pred.shape[0] == 10 and pred.shape[1] == 8400:
        pred = pred.T   # [8400, 10]

    boxes_xywh = pred[:, :4]      # cx, cy, w, h
    cls_scores = pred[:, 4:]      # (N, 6) ← 直接是类别置信度！

    # 获取最大类别和得分
    max_scores = cls_scores.max(axis=1)  # shape (N,)
    max_classes = cls_scores.argmax(axis=1)

    # 过滤低分
    mask = max_scores > conf_thres
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    max_classes = max_classes[mask]

    if len(max_scores) == 0:
        return []

    # Top-K
    topk_idx = np.argsort(-max_scores)[:max_det]
    boxes_xywh = boxes_xywh[topk_idx]
    max_scores = max_scores[topk_idx]
    max_classes = max_classes[topk_idx]

    # xywh → xyxy
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 【关键修复】坐标还原：先减去平移量，再除以缩放比例
    r, (left, top) = img_info
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - left/2) / r
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - top/2) / r

    # 限制在原图范围内
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, None)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, None)

    # NMS
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=max_scores.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    detections = []
    for i in indices:
        x1, y1, x2, y2 = boxes_xyxy[i]
        detections.append([
            float(x1), float(y1), float(x2), float(y2),
            float(max_scores[i]), int(max_classes[i])
        ])
    return detections

def run_inference_and_evaluate(
    onnx_model_path: str,
    images_dir: str,
    annotations_json: str,
    output_results_json: str = "detections.json",
    input_size: tuple = (640, 640),
    batch_size: int = 1,
    conf_thres: float = 0.001,
    iou_thres: float = 0.65,
    max_det: int = 300,
):
    assert batch_size == 1, "Only batch_size=1 is supported for simplicity"

    # Load COCO dataset
    coco_gt = COCO(annotations_json)
    image_ids = coco_gt.getImgIds()
    print(f"Found {len(image_ids)} images in validation set.")

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    print(f"ONNX model loaded. Input: {input_name}")
    print("当前使用的 Execution Provider:", ort_session.get_providers())

    results = []

    for img_id in tqdm(image_ids, desc="Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Preprocess
        img_tensor, scale_pad = preprocess_image(img_path, input_size)
        img_input = np.expand_dims(img_tensor, axis=0)  # Add batch dim

        # Inference
        ort_outs = ort_session.run(None, {input_name: img_input})
        output = ort_outs[0]  # Assume first output is detection

        # Postprocess
        detections = postprocess_yolo(
            output,
            scale_pad,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det
        )

        # Format for COCO
        for det in detections:
            x1, y1, x2, y2, score, cls_id = det
            w = x2 - x1
            h = y2 - y1
            results.append({
                "image_id": img_id,
                "category_id": int(cls_id),
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                "score": round(score, 4)
            })

    # Save results
    with open(output_results_json, "w") as f:
        json.dump(results, f)
    print(f"Saved {len(results)} detections to {output_results_json}")

    # Evaluate with COCO API
    coco_dt = coco_gt.loadRes(output_results_json)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Print main metrics
    stats = coco_eval.stats
    print("\n📊 COCO mAP Results:")
    print(f"mAP@[IoU=0.50:0.95] = {stats[0]:.4f}")
    print(f"mAP@IoU=0.50         = {stats[1]:.4f}")
    print(f"mAP@IoU=0.75         = {stats[2]:.4f}")

    # 在 run_inference_and_evaluate 最后加：  D:\CV_PPQ\models\yolov8_sym_int8.onnx
    print("\n🔍 Sample detections vs GT:")
    sample_img_id = image_ids[0]
    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=sample_img_id))
    preds = [d for d in results if d["image_id"] == sample_img_id]

    print("Ground Truth category_ids:", [ann["category_id"] for ann in gt_anns])
    print("Predicted category_ids   :", [p["category_id"] for p in preds[:10]])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate quantized ONNX model with COCO mAP")
    parser.add_argument("--onnx", type=str, default="/root/my_FILE/yolov8_int8_fixed_1.onnx", help="Path to quantized ONNX model")
    parser.add_argument("--images", type=str, default="/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/test", help="Path to validation images folder")
    parser.add_argument("--ann", type=str, default="/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/test_coco.json", help="Path to COCO annotations JSON")
    parser.add_argument("--output", type=str, default="detections.json", help="Output detections JSON")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold")

    args = parser.parse_args()

    run_inference_and_evaluate(
        onnx_model_path=args.onnx,
        images_dir=args.images,
        annotations_json=args.ann,
        output_results_json=args.output,
        input_size=(args.imgsz, args.imgsz),
        conf_thres=args.conf,
        iou_thres=args.iou
    )