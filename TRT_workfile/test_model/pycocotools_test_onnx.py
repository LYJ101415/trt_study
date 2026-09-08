import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def _num_anchors(input_size):
    """YOLOv8 三检测头（P3/P4/P5，stride 8/16/32）在给定输入尺寸下的 anchor 总数。

    例：640x640 -> 80^2 + 40^2 + 20^2 = 8400；320x320 -> 40^2 + 20^2 + 10^2 = 2100。
    """
    h, w = input_size
    return (h // 8) * (w // 8) + (h // 16) * (w // 16) + (h // 32) * (w // 32)


def preprocess_image(image_path, input_size=(640, 640)):
    """预处理：BGR -> RGB, Resize with pad, Normalize to [0,1]。

    返回 (tensor, meta)。meta 含缩放比例、单边 padding 与原图尺寸，供后处理还原坐标。
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像（损坏或路径错误）: {image_path}")

    h0, w0 = img.shape[:2]
    r = min(input_size[0] / h0, input_size[1] / w0)
    new_h, new_w = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    dw, dh = input_size[1] - new_w, input_size[0] - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # HWC -> CHW, BGR -> RGB, normalize
    padded = padded[:, :, ::-1].transpose(2, 0, 1)  # RGB, CHW
    img_tensor = np.ascontiguousarray(padded, dtype=np.float32) / 255.0

    meta = {
        "scale": r,
        "pad_top": top,    # 单边上 padding（不是总 padding）
        "pad_left": left,  # 单边左 padding
        "orig_h": h0,
        "orig_w": w0,
    }
    return img_tensor, meta


def postprocess_yolo(pred, meta, input_size=(640, 640), conf_thres=0.001, iou_thres=0.65, max_det=300):
    if pred.ndim == 3:
        pred = pred[0]  # 去掉 batch 维 -> [C, N] 或 [N, C]

    # 规整为 [N, C]（N=anchor 数，C=4+类别数）。按 input_size 推导 anchor 数，避免硬编码 8400。
    num_anchors = _num_anchors(input_size)
    if pred.shape[1] == num_anchors:      # [C, N] -> 转置为 [N, C]
        pred = pred.T
    elif pred.shape[0] != num_anchors:    # 两轴都不匹配，模型结构/输入尺寸与预期不符
        raise ValueError(
            f"输出形状 {pred.shape} 与预期不符：anchor 数应为 {num_anchors}（input_size={input_size}）"
        )

    boxes_xywh = pred[:, :4]      # cx, cy, w, h
    cls_scores = pred[:, 4:]      # (N, num_classes) 直接是类别置信度

    max_scores = cls_scores.max(axis=1)   # shape (N,)
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

    # xywh -> xyxy
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 坐标还原：减去单边 padding，再除以缩放比例
    scale = meta["scale"]
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - meta["pad_left"]) / scale
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - meta["pad_top"]) / scale

    # 限制在原图范围内（同时裁剪上下限）
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, meta["orig_w"])
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, meta["orig_h"])

    # 过滤无效框（宽/高 <= 0）
    valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
    boxes_xyxy = boxes_xyxy[valid]
    max_scores = max_scores[valid]
    max_classes = max_classes[valid]
    if len(max_scores) == 0:
        return []

    # NMS：cv2.dnn.NMSBoxes 要求 [x, y, w, h]（左上角 + 宽高），这里先把 xyxy 转成 xywh
    boxes_xywh_nms = np.stack(
        [boxes_xyxy[:, 0], boxes_xyxy[:, 1],
         boxes_xyxy[:, 2] - boxes_xyxy[:, 0], boxes_xyxy[:, 3] - boxes_xyxy[:, 1]],
        axis=1,
    )
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh_nms.tolist(),
        scores=max_scores.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
    )
    if indices is None or len(indices) == 0:
        return []
    indices = np.asarray(indices).flatten().astype(int)

    detections = []
    for i in indices:
        x1, y1, x2, y2 = boxes_xyxy[i]
        detections.append([
            float(x1), float(y1), float(x2), float(y2),
            float(max_scores[i]), int(max_classes[i])
        ])
    return detections


def run_inference_and_evaluate(
    onnx_model_path,
    images_dir,
    annotations_json,
    output_results_json="detections.json",
    input_size=(640, 640),
    batch_size=1,
    conf_thres=0.001,
    iou_thres=0.65,
    max_det=300,
    device="cpu",
):
    if batch_size != 1:
        raise ValueError(f"当前仅支持 batch_size=1，收到 {batch_size}")

    # 推理后端：默认 CPU，指定 --device cuda 且环境可用时优先 CUDA
    available = ort.get_available_providers()
    if device.lower() == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        if device.lower() == "cuda":
            print(f"警告: 未检测到 CUDAExecutionProvider，回退到 CPU。可用 providers: {available}")
        providers = ["CPUExecutionProvider"]

    # Load COCO dataset
    coco_gt = COCO(annotations_json)
    image_ids = coco_gt.getImgIds()
    print(f"Found {len(image_ids)} images in validation set.")

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    print(f"ONNX model loaded. Input: {input_name}, Providers: {ort_session.get_providers()}")

    results = []

    for img_id in tqdm(image_ids, desc="Inference"):
        img_info = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, img_info["file_name"])
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Preprocess
        img_tensor, meta = preprocess_image(img_path, input_size)
        img_input = np.expand_dims(img_tensor, axis=0)  # Add batch dim

        # Inference
        ort_outs = ort_session.run(None, {input_name: img_input})
        output = ort_outs[0]  # Assume first output is detection

        # Postprocess
        detections = postprocess_yolo(
            output,
            meta,
            input_size=input_size,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
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
    print("\nCOCO mAP Results:")
    print(f"mAP@[IoU=0.50:0.95] = {stats[0]:.4f}")
    print(f"mAP@IoU=0.50         = {stats[1]:.4f}")
    print(f"mAP@IoU=0.75         = {stats[2]:.4f}")

    # 抽样核对预测与真值的 category_id 是否对齐（防止类别映射错位导致 mAP=0）
    print("\nSample detections vs GT:")
    sample_img_id = image_ids[0]
    gt_anns = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=sample_img_id))
    preds = [d for d in results if d["image_id"] == sample_img_id]

    print("Ground Truth category_ids:", [ann["category_id"] for ann in gt_anns])
    print("Predicted category_ids   :", [p["category_id"] for p in preds[:10]])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate quantized ONNX model with COCO mAP")
    parser.add_argument("--onnx", type=str, default=r"/root/my_FILE/models/yolov8_int8_dy_exclude.onnx", help="Path to quantized ONNX model")
    parser.add_argument("--images", type=str, default=r"/root/my_FILE/datasets/Data_DeepPCB_YOLO/images/test_scaled", help="Path to validation images folder")
    parser.add_argument("--ann", type=str, default=r"/root/my_FILE/datasets/Data_DeepPCB_YOLO/images/test_scaled_coco.json", help="Path to COCO annotations JSON")
    parser.add_argument("--output", type=str, default="detections.json", help="Output detections JSON")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.65, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="Inference device")

    args = parser.parse_args()

    run_inference_and_evaluate(
        onnx_model_path=args.onnx,
        images_dir=args.images,
        annotations_json=args.ann,
        output_results_json=args.output,
        input_size=(args.imgsz, args.imgsz),
        conf_thres=args.conf,
        iou_thres=args.iou,
        device=args.device,
    )
