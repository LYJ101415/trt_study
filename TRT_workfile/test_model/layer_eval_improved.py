# -*- coding: utf-8 -*-
"""
逐层量化敏感度消融 (改进版)，解除了节点名为空的模型的问题。
对 INT8 QDQ 模型逐层移除量化 (Q/DQ) 并回退 FP32, 在 COCO 上评估 mAP,
按 ΔmAP 排序找出对量化最敏感的算子层, 指导后续混合精度 (FP16/FP32 fallback) 决策。
相对 v4 的关键修复:
  1. 类别 ID 重映射: 模型类别索引 -> COCO category_id (修复索引偏移导致的 mAP 错误)
  2. 用张量名(而非节点名)定位 Q/DQ, 删除子串匹配, 消除误配对
  3. 共享 Q/DQ 安全校验: 防止生成损坏模型
  4. 相邻性校验: 仅支持直接 Q->DQ, 检测到中间算子则跳过
  5. 基线缓存签名失效: 模型 MD5 变化自动重算
  6. 临时模型唯一路径 + 结果 JSON 原子写 + 显式 session 释放
  7. 清理 scale/zero_point 残留 initializer + shape inference
  8. 后处理动态解析输出形状 (不再硬编码 10/8400)
  9. 区分 weight/activation 输入 (记录 DQ 连到目标算子的 input 下标)
"""

import os
import sys
import gc
import json
import time
import csv
import argparse
import hashlib
import tempfile
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

import onnx
import onnx.shape_inference
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort

DEFAULT_CONFIG = {
    "int8_model": "/root/my_FILE/yolov8_int8.onnx",
    "images_dir": "/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val",
    "ann_json": "/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val_coco.json",
    "img_size": 640,
    "conf_thres": 0.001,
    "iou_thres": 0.65,
    "max_det": 300,
    "device_id": 0,
    "output_csv": "ablation_ranking.csv",
    "output_json": "ablation_results.json",
    "baseline_cache": "baseline_map_cache.json",
}

# 分级标签 -> 控制台图标 (CSV 中只写纯 ASCII 标签)
LABEL_ICON = {
    "HIGH_SENS": "🔴",
    "MED_SENS": "🟡",
    "LOW_SENS": "🟢",
    "SAFE": "⚪",
    "ANOMALY": "⛔",
    "MINUS": "⚠️",
    "OK": "✅",
}


def preprocess_image(image_path, input_size=(640, 640)):
    """YOLO letterbox 预处理. 返回 (CHW float32 张量, (ratio, pad_w_total, pad_h_total))."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h0, w0 = img.shape[:2]
    r = min(input_size[0] / h0, input_size[1] / w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = input_size[1] - new_w   # 总水平 padding
    pad_h = input_size[0] - new_h   # 总垂直 padding
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    padded = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    return np.ascontiguousarray(padded, dtype=np.float32) / 255.0, (r, pad_w, pad_h)


def postprocess_yolo(pred, img_info, conf_thres=0.001, iou_thres=0.65, max_det=300):
    """YOLOv8 后处理. 动态解析输出形状, 不再硬编码 10/8400."""
    if pred.ndim == 3:
        pred = pred[0]
    # YOLOv8 输出为 [4+nc, N] 或 [N, 4+nc]; 统一成 [N, 4+nc]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    boxes_xywh = pred[:, :4]
    cls_scores = pred[:, 4:]
    max_scores = cls_scores.max(axis=1)
    max_classes = cls_scores.argmax(axis=1)

    mask = max_scores > conf_thres
    boxes_xywh, max_scores, max_classes = boxes_xywh[mask], max_scores[mask], max_classes[mask]
    if len(max_scores) == 0:
        return []

    topk = np.argsort(-max_scores)[:max_det]
    boxes_xywh, max_scores, max_classes = boxes_xywh[topk], max_scores[topk], max_classes[topk]

    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    r, pad_w, pad_h = img_info
    # 单边 padding = 总 padding / 2
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w / 2) / r
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h / 2) / r
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, None)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, None)

    indices = cv2.dnn.NMSBoxes(boxes_xyxy.tolist(), max_scores.tolist(),
                               conf_thres, iou_thres)
    if len(indices) == 0:
        return []
    indices = indices.flatten()

    dets = []
    for i in indices:
        bx1, by1, bx2, by2 = boxes_xyxy[i]
        dets.append([float(bx1), float(by1), float(bx2), float(by2),
                     float(max_scores[i]), int(max_classes[i])])
    return dets


def _make_cat_id_map(coco_gt, explicit=None):
    """构建 模型类别索引 -> COCO category_id 的映射表.

    默认取 GT 的 category_id 升序列表 (同时兼容 0-based 与 1-based).
    若模型类别顺序与 GT category_id 顺序不一致, 用 --cat-ids 显式指定.
    """
    if explicit:
        return [int(x) for x in explicit]
    return sorted(coco_gt.getCatIds())


def evaluate_model_ort(onnx_model_path, images_dir, annotations_json,
                       input_size, conf_thres, iou_thres, max_det, device_id,
                       cat_id_map=None):
    coco_gt = COCO(annotations_json)
    image_ids = coco_gt.getImgIds()
    cat_id_map = _make_cat_id_map(coco_gt, cat_id_map)
    if cat_id_map and cat_id_map[0] != 0:
        print(f"    [INFO] 类别映射: 模型索引 -> category_id = {cat_id_map}")

    providers = [("CUDAExecutionProvider", {"device_id": device_id}),
                 "CPUExecutionProvider"]
    sess_opt = ort.SessionOptions()
    sess_opt.log_severity_level = 3

    session = None
    results = []
    try:
        session = ort.InferenceSession(onnx_model_path, sess_options=sess_opt,
                                       providers=providers)
        input_name = session.get_inputs()[0].name

        # warmup, 避免首帧把 CUDA 初始化耗时计入评估
        try:
            warm = np.zeros([1, 3, input_size[0], input_size[1]], dtype=np.float32)
            session.run(None, {input_name: warm})
        except Exception:
            pass

        for img_id in tqdm(image_ids, desc="    Inference", leave=False):
            meta = coco_gt.loadImgs(img_id)
            if not meta:
                continue
            meta = meta[0]
            img_path = os.path.join(images_dir, meta["file_name"])
            if not os.path.exists(img_path):
                continue
            tensor, scale_pad = preprocess_image(img_path, input_size)
            out = session.run(None, {input_name: np.expand_dims(tensor, 0)})[0]
            for det in postprocess_yolo(out, scale_pad, conf_thres, iou_thres, max_det):
                x1, y1, x2, y2, score, cls_id = det
                if cls_id >= len(cat_id_map):
                    continue  # 无对应 GT 类别, 跳过
                results.append({
                    "image_id": img_id,
                    "category_id": int(cat_id_map[cls_id]),
                    "bbox": [round(x1, 2), round(y1, 2),
                             round(x2 - x1, 2), round(y2 - y1, 2)],
                    "score": round(score, 4),
                })
    finally:
        if session is not None:
            del session
            gc.collect()

    if not results:
        print("    [WARN] 无检测结果, 返回零 mAP")
        return {"map50_95": 0.0, "map50": 0.0, "map75": 0.0,
                "ar100": 0.0, "num_detections": 0}

    fd, tmp_det = tempfile.mkstemp(suffix=".json", prefix="ablation_det_")
    os.close(fd)
    try:
        with open(tmp_det, "w") as f:
            json.dump(results, f)
        coco_dt = coco_gt.loadRes(tmp_det)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        stats = ev.stats
    finally:
        _safe_remove(tmp_det)

    return {"map50_95": float(stats[0]), "map50": float(stats[1]),
            "map75": float(stats[2]), "ar100": float(stats[8]),
            "num_detections": len(results)}


def _build_tensor_maps(graph):
    producer, consumers = {}, {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return producer, consumers


def _extract_base_name(name):
    if not name:
        return "unnamed"
    for suf in ("_QuantizeLinear_Output", "_QuantizeLinear"):
        if name.endswith(suf):
            return name[: -len(suf)]
    parts = name.rsplit("_", 2)
    return parts[0] if len(parts) > 1 else name


def get_all_qdq_layers(int8_model):
    """枚举所有 Q/DQ 层. 用张量名定位, 并检测相邻性 / 共享情况."""
    model = onnx.load(int8_model)
    graph = model.graph
    _, consumers = _build_tensor_maps(graph)

    layers = []
    for node in graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        if not node.output:
            continue
        q_output = node.output[0]
        q_name = node.name or f"Q_{q_output}"

        q_consumers = consumers.get(q_output, [])
        dq_nodes = [c for c in q_consumers if c.op_type == "DequantizeLinear"]
        other = [c for c in q_consumers if c.op_type != "DequantizeLinear"]

        if not dq_nodes:
            # Q 输出没有直接连 DQ (中间有 Reshape/Transpose 等), 本脚本不支持
            continue

        dq_node = dq_nodes[0]
        if not dq_node.output:
            continue
        dq_output = dq_node.output[0]

        target_name = None
        target_input_idx = None
        num_dq_consumers = 0
        for c in consumers.get(dq_output, []):
            if c.op_type in ("QuantizeLinear", "DequantizeLinear"):
                continue
            num_dq_consumers += 1
            if target_name is None:
                target_name = c.name or c.op_type
                for i, inp in enumerate(c.input):
                    if inp == dq_output:
                        target_input_idx = i
                        break

        base_name = _extract_base_name(node.name)
        skip_reason = None
        if other:
            skip_reason = f"Q输出被 {len(other)} 个非DQ节点消费(存在中间算子), 不支持"
        elif len(dq_nodes) > 1:
            skip_reason = f"Q输出被 {len(dq_nodes)} 个DQ共享, 移除会破坏模型"

        display = target_name or base_name
        if target_input_idx is not None:
            display = f"{display}#in{target_input_idx}"

        layers.append({
            "q_name": q_name,
            "dq_name": dq_node.name,
            "q_output": q_output,
            "dq_output": dq_output,
            "target_name": target_name,
            "target_input_idx": target_input_idx,
            "base_name": base_name,
            "display_name": display,
            "skip_reason": skip_reason,
            "num_dq_consumers": num_dq_consumers,
        })

    print(f"[INFO] 共发现 {len(layers)} 个量化层")
    return layers


def generate_fallback_model(int8_model, output_path, q_output, dq_output):
    """移除指定 Q/DQ 对并重连到上游 FP32 张量. 返回 (成功, info_or_err)."""
    try:
        model = onnx.load(int8_model)
        graph = model.graph

        q_node = next((n for n in graph.node
                       if n.op_type == "QuantizeLinear" and n.output and n.output[0] == q_output), None)
        dq_node = next((n for n in graph.node
                        if n.op_type == "DequantizeLinear" and n.output and n.output[0] == dq_output), None)
        if q_node is None or dq_node is None:
            return False, f"未找到 Q({q_output}) DQ({dq_output})"

        upstream_tensor = q_node.input[0] if q_node.input else None
        if not upstream_tensor:
            return False, "Q 节点无上游输入"

        # 安全校验: q_output 只应被本 DQ 消费
        other_q_consumers = [n.name or n.op_type for n in graph.node
                             if n is not dq_node and q_output in n.input]
        if other_q_consumers:
            return False, (f"Q输出 '{q_output}' 仍被其他节点消费, "
                           f"移除将破坏模型: {other_q_consumers}")

        def _is_removed(n):
            return ((n.op_type == "QuantizeLinear" and n.output and n.output[0] == q_output)
                    or (n.op_type == "DequantizeLinear" and n.output and n.output[0] == dq_output))

        # 重连: 所有下游对 dq_output 的引用 -> upstream_tensor
        replaced = 0
        affected = []
        for node in graph.node:
            if _is_removed(node):
                continue
            for i, inp in enumerate(node.input):
                if inp == dq_output:
                    node.input[i] = upstream_tensor
                    replaced += 1
                    affected.append(node.name or node.op_type)
        for out in graph.output:
            if out.name == dq_output:
                out.name = upstream_tensor

        # 删除 Q/DQ 节点
        keep = [n for n in graph.node if not _is_removed(n)]
        del graph.node[:]
        graph.node.extend(keep)

        # 清理 value_info (被删节点产生的张量)
        removed_tensors = set()
        for n in (q_node, dq_node):
            for o in n.output:
                removed_tensors.add(o)
        vi_keep = [v for v in graph.value_info if v.name not in removed_tensors]
        del graph.value_info[:]
        graph.value_info.extend(vi_keep)

        # 清理不再被引用的 initializer (Q/DQ 的 scale/zero_point)
        used = set()
        for n in graph.node:
            used.update(n.input)
            used.update(n.output)
        for o in graph.output:
            used.add(o.name)
        for i in graph.input:
            used.add(i.name)
        init_keep = [ini for ini in graph.initializer if ini.name in used]
        del graph.initializer[:]
        graph.initializer.extend(init_keep)

        # shape inference (best-effort)
        try:
            inferred = onnx.shape_inference.infer_shapes(model)
            model.CopyFrom(inferred)
        except Exception as e:
            print(f"      [WARN] shape inference 失败(非致命): {e}")

        onnx.save(model, output_path)

        test_opt = ort.SessionOptions()
        test_opt.log_severity_level = 3
        test_sess = ort.InferenceSession(output_path, sess_options=test_opt,
                                         providers=["CPUExecutionProvider"])
        del test_sess

        print(f"      ✅ ORT 加载验证通过 | 重连 {replaced} 处")
        return True, {"replaced": replaced, "affected": affected}
    except Exception as e:
        _safe_remove(output_path)
        return False, str(e)


def _save_results(results, path):
    """原子写 JSON (先写临时文件再 rename), 避免中断损坏缓存."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _model_signature(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _classify(delta):
    abs_d = abs(delta)
    if delta > 0:
        if abs_d >= 0.01:
            return "HIGH_SENS"
        if abs_d >= 0.005:
            return "MED_SENS"
        if abs_d >= 0.002:
            return "LOW_SENS"
        return "SAFE"
    if abs_d >= 0.01:
        return "ANOMALY"
    if abs_d >= 0.002:
        return "MINUS"
    return "OK"


def _layer_entries(results):
    """过滤掉内部元数据键 (以 _ 开头), 只返回逐层结果."""
    return {k: v for k, v in results.items()
            if not k.startswith("_") and isinstance(v, dict)}


def load_baseline(cache_path, int8_model, images_dir, ann_json,
                  input_size, conf_thres, iou_thres, max_det, device_id):
    sig = _model_signature(int8_model)
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("signature") == sig:
                print(f"[INFO] 加载缓存基线 mAP50-95={cached['map50_95']:.4f}")
                return cached
            print("[WARN] 模型已变化, 重新评估基线")
        except Exception:
            pass

    print("[INFO] ========== 评估基线 INT8 模型 ==========")
    t0 = time.time()
    metrics = evaluate_model_ort(int8_model, images_dir, ann_json, input_size,
                                 conf_thres, iou_thres, max_det, device_id)
    elapsed = time.time() - t0
    metrics["signature"] = sig
    metrics["config"] = {"input_size": list(input_size), "conf_thres": conf_thres,
                         "iou_thres": iou_thres, "max_det": max_det}
    print(f"[INFO] 基线 mAP50-95={metrics['map50_95']:.4f}, "
          f"mAP50={metrics['map50']:.4f} ({elapsed:.1f}s)")
    with open(cache_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def print_and_export(results, baseline, output_csv):
    entries = _layer_entries(results)
    valid = [v for v in entries.values() if v.get("status") == "OK"]
    invalid = [v for v in entries.values() if v.get("status") != "OK"]

    if not valid:
        print("\n[WARN] 没有有效评估结果!")
        return

    ranked = sorted(valid, key=lambda x: x["delta_map50_95"], reverse=True)

    sep = "=" * 140
    dash = "-" * 140
    title = "逐层消融精度敏感层排名 (Δ>0=敏感层·回退FP16可提升 | Δ<0=异常·需排查)"
    bl_50_95 = baseline["map50_95"]
    bl_50 = baseline["map50"]
    baseline_info = f"基线 INT8 mAP50-95={bl_50_95:.4f}  mAP50={bl_50:.4f}"

    print(f"\n{sep}")
    print(f"{title:^140}")
    print(f"{baseline_info:^140}")
    print(sep)

    hdr = (f"{'Rank':<5} {'Operator_Node':<42} {'mAP50-95':>10} {'ΔmAP50-95':>12} "
           f"{'mAP50':>10} {'ΔmAP50':>10} {'Label':>14}")
    print(hdr)
    print(dash)

    for i, r in enumerate(ranked):
        d = r["delta_map50_95"]
        d50 = r["delta_map50"]
        node_short = r["display_name"][:40]
        label = _classify(d)
        icon = LABEL_ICON.get(label, "")
        line = (f"{i + 1:<5} {node_short:<42} {r['map50_95']:>10.4f} "
                f"{d:>+12.4f} {r['map50']:>10.4f} {d50:>+10.4f} {icon + ' ' + label:>14}")
        print(line)

    if invalid:
        print(f"\n[WARN] {len(invalid)} 个层未成功:")
        for v in invalid:
            err_msg = v.get("error", "")
            err_part = f" ({err_msg})" if err_msg else ""
            status = v["status"]
            disp = v.get("display_name", v.get("layer", "?"))
            print(f"    {disp}: {status}{err_part}")

    with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "Operator_Node", "Base_Name", "mAP50_95", "Delta_mAP50_95",
                    "mAP50", "Delta_mAP50", "Label", "GenTime_s", "EvalTime_s"])
        for i, r in enumerate(ranked):
            d = r["delta_map50_95"]
            d50 = r["delta_map50"]
            label = _classify(d)
            gen_t = r.get("gen_time_sec", "")
            eval_t = r.get("eval_time_sec", "")
            w.writerow([
                i + 1, r["display_name"], r.get("base_name", ""),
                f"{r['map50_95']:.4f}", f"{d:.4f}",
                f"{r['map50']:.4f}", f"{d50:.4f}",
                label, gen_t, eval_t,
            ])

    print(f"\n[INFO] CSV: {output_csv}")

    high_sens = [r for r in ranked if r["delta_map50_95"] >= 0.01]
    med_sens = [r for r in ranked if 0.005 <= r["delta_map50_95"] < 0.01]
    anomalies = [r for r in ranked if r["delta_map50_95"] <= -0.01]

    section_sep = "=" * 80
    print(f"\n{section_sep}")
    print("  📋 行动建议")
    print(section_sep)

    if high_sens:
        print(f"  🔴 HIGH_SENS ({len(high_sens)} 层) — 强烈建议强制 FP16:")
        for r in high_sens:
            print(f"     • {r['display_name']}  (ΔmAP50-95={r['delta_map50_95']:+.4f})")

    if med_sens:
        print(f"  🟡 MED_SENS ({len(med_sens)} 层) — 建议 per-channel 量化或重新校准:")
        for r in med_sens:
            print(f"     • {r['display_name']}  (ΔmAP50-95={r['delta_map50_95']:+.4f})")

    if anomalies:
        print(f"  ⛔ ANOMALY ({len(anomalies)} 层) — 去除量化后精度暴跌, 需排查图拓扑:")
        for r in anomalies:
            print(f"     • {r['display_name']}  (ΔmAP50-95={r['delta_map50_95']:+.4f})")

    if not high_sens and not med_sens and not anomalies:
        print("  ✅ 所有层量化质量良好, 无需额外处理!")

    print()


def main():
    parser = argparse.ArgumentParser(description="逐层消融 (改进版, ORT+COCO)")
    parser.add_argument("--int8-model", default=DEFAULT_CONFIG["int8_model"])
    parser.add_argument("--images-dir", default=DEFAULT_CONFIG["images_dir"])
    parser.add_argument("--ann-json", default=DEFAULT_CONFIG["ann_json"])
    parser.add_argument("--img-size", type=int, default=DEFAULT_CONFIG["img_size"])
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONFIG["conf_thres"])
    parser.add_argument("--iou-thres", type=float, default=DEFAULT_CONFIG["iou_thres"])
    parser.add_argument("--max-det", type=int, default=DEFAULT_CONFIG["max_det"])
    parser.add_argument("--device-id", type=int, default=DEFAULT_CONFIG["device_id"])
    parser.add_argument("--tmp-model", default=None,
                        help="临时模型路径 (默认使用系统临时目录)")
    parser.add_argument("--output-csv", default=DEFAULT_CONFIG["output_csv"])
    parser.add_argument("--output-json", default=DEFAULT_CONFIG["output_json"])
    parser.add_argument("--baseline-cache", default=DEFAULT_CONFIG["baseline_cache"])
    parser.add_argument("--cat-ids", default=None,
                        help="模型类别顺序对应的 COCO category_id, 逗号分隔, 如 '1,2,3'")
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    args = parser.parse_args()

    input_size = (args.img_size, args.img_size)
    cat_id_map = None
    if args.cat_ids:
        cat_id_map = [int(x.strip()) for x in args.cat_ids.split(",") if x.strip()]

    if args.tmp_model:
        tmp_model = args.tmp_model
    else:
        fd, tmp_model = tempfile.mkstemp(suffix=".onnx", prefix="ablation_fallback_")
        os.close(fd)

    output_json = args.output_json
    model_sig = _model_signature(args.int8_model)

    print("\n" + "=" * 80)
    print("  逐层消融 (改进版 · ORT + COCO · 张量名定位 + 安全校验 + 类别重映射)")
    print("=" * 80)

    all_layers = get_all_qdq_layers(args.int8_model)
    if not all_layers:
        print("[ERROR] 未找到 QuantizeLinear 节点!")
        sys.exit(1)
    for i, layer_info in enumerate(all_layers):
        idx_str = f"{i + 1:>4d}"
        disp = layer_info["display_name"]
        base = layer_info["base_name"]
        flag = f"  [SKIP:{layer_info['skip_reason']}]" if layer_info["skip_reason"] else ""
        print(f"  {idx_str}. {disp:<40s} (base={base}){flag}")

    baseline = load_baseline(args.baseline_cache, args.int8_model, args.images_dir,
                             args.ann_json, input_size, args.conf_thres,
                             args.iou_thres, args.max_det, args.device_id)

    results = {}
    if args.resume and os.path.exists(output_json):
        try:
            with open(output_json) as f:
                results = json.load(f)
            meta = results.get("_meta", {})
            if meta.get("signature") != model_sig:
                print("[WARN] 模型已变化, 忽略旧缓存")
                results = {}
            else:
                done_cnt = sum(1 for k, v in _layer_entries(results).items()
                               if v.get("status") == "OK")
                print(f"[INFO] 断点续跑: {done_cnt} 条已完成")
        except Exception:
            results = {}
    results["_meta"] = {"signature": model_sig, "model": args.int8_model,
                        "config": {"img_size": args.img_size,
                                   "conf_thres": args.conf_thres,
                                   "iou_thres": args.iou_thres,
                                   "max_det": args.max_det}}

    total = len(all_layers)
    evaluated = skipped = failed = 0

    print(f"\n[INFO] ========== 开始逐层消融 ({total} 层) ==========\n")

    for idx, layer_info in enumerate(all_layers):
        display_name = layer_info["display_name"]
        q_output = layer_info["q_output"]
        dq_output = layer_info["dq_output"]
        cache_key = q_output  # 张量名唯一, 作为结果/续跑键

        progress = f"[{idx + 1}/{total}]"
        print(f"{progress} {display_name}")

        if layer_info["skip_reason"]:
            print(f"      ⚠️  跳过: {layer_info['skip_reason']}")
            results[cache_key] = {
                "display_name": display_name,
                "base_name": layer_info["base_name"],
                "q_output": q_output, "dq_output": dq_output,
                "status": "SKIP", "error": layer_info["skip_reason"],
                "timestamp": datetime.now().isoformat(),
            }
            skipped += 1
            _save_results(results, output_json)
            continue

        if cache_key in results and results[cache_key].get("status") == "OK":
            skipped += 1
            continue

        t_gen = time.time()
        ok, info = generate_fallback_model(args.int8_model, tmp_model, q_output, dq_output)
        gen_time = time.time() - t_gen

        if not ok:
            results[cache_key] = {
                "display_name": display_name,
                "base_name": layer_info["base_name"],
                "q_output": q_output, "dq_output": dq_output,
                "status": "GEN_FAIL", "error": str(info),
                "timestamp": datetime.now().isoformat(),
            }
            failed += 1
            _save_results(results, output_json)
            _safe_remove(tmp_model)
            continue

        t_eval = time.time()
        try:
            fb = evaluate_model_ort(tmp_model, args.images_dir, args.ann_json,
                                    input_size, args.conf_thres, args.iou_thres,
                                    args.max_det, args.device_id, cat_id_map)
            eval_time = time.time() - t_eval
        except Exception as e:
            eval_time = time.time() - t_eval
            print(f"      ❌ 评估异常: {e}")
            results[cache_key] = {
                "display_name": display_name,
                "base_name": layer_info["base_name"],
                "q_output": q_output, "dq_output": dq_output,
                "status": "EVAL_FAIL", "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            failed += 1
            _save_results(results, output_json)
            _safe_remove(tmp_model)
            continue

        d50_95 = fb["map50_95"] - baseline["map50_95"]
        d50 = fb["map50"] - baseline["map50"]

        results[cache_key] = {
            "display_name": display_name,
            "base_name": layer_info["base_name"],
            "q_output": q_output, "dq_output": dq_output,
            "target_name": layer_info["target_name"],
            "target_input_idx": layer_info["target_input_idx"],
            "num_dq_consumers": layer_info["num_dq_consumers"],
            "status": "OK",
            "map50_95": fb["map50_95"], "map50": fb["map50"],
            "map75": fb["map75"], "ar100": fb["ar100"],
            "num_detections": fb["num_detections"],
            "delta_map50_95": d50_95, "delta_map50": d50,
            "baseline_map50_95": baseline["map50_95"],
            "baseline_map50": baseline["map50"],
            "gen_time_sec": round(gen_time, 2),
            "eval_time_sec": round(eval_time, 2),
            "timestamp": datetime.now().isoformat(),
        }

        sign = "+" if d50_95 >= 0 else ""
        print(f"      mAP50-95={fb['map50_95']:.4f}(Δ={sign}{d50_95:.4f}) | "
              f"mAP50={fb['map50']:.4f}(Δ={sign}{d50:.4f}) | "
              f"生成{gen_time:.1f}s+评估{eval_time:.1f}s")

        evaluated += 1
        _save_results(results, output_json)
        _safe_remove(tmp_model)

    print_and_export(results, baseline, args.output_csv)

    _safe_remove(tmp_model)

    print(f"[STATS] 总={total} | 新评={evaluated} | 跳过={skipped} | 失败={failed}")
    print(f"[DONE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
