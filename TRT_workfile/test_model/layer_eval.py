#!/usr/bin/env python3
"""
layer_eval.py: 逐层消融精度敏感层定位 (ONNXRuntime + COCO mAP)
  - Layer 列显示 Q/DQ 所服务的算子节点名 (如 Conv_45, MatMul_130)
  - 保留 _classify 双向分级 + 降序排列 + f-string 安全 + tensor重连
依赖: pip install onnxruntime onnx onnx-graphsurgeon pycocotools opencv-python tqdm numpy
"""

import os
import sys
import json
import time
import csv
import argparse
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

import onnx
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import onnxruntime as ort

# ============================================================
#  默认配置
# ============================================================
DEFAULT_CONFIG = {
    "int8_model": "/root/my_FILE/yolov8_int8.onnx",
    "images_dir": "/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val",
    "ann_json": "/root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val_coco.json",
    "img_size": 640,
    "conf_thres": 0.001,
    "iou_thres": 0.65,
    "max_det": 300,
    "device_id": 0,
    "tmp_model_path": "/tmp/fallback_temp.onnx",
    "output_csv": "ablation_ranking.csv",
    "output_json": "ablation_results.json",
    "baseline_cache": "baseline_map_cache.json",
}

# ============================================================
#  预处理 / 后处理
# ============================================================

def preprocess_image(image_path, input_size=(640, 640)):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    h0, w0 = img.shape[:2]
    r = min(input_size[0] / h0, input_size[1] / w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = input_size[1] - new_w, input_size[0] - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    padded = padded[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(padded, dtype=np.float32) / 255.0, (r, (dw, dh))

def postprocess_yolo(pred, img_info, conf_thres=0.001, iou_thres=0.65, max_det=300):
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] == 10 and pred.shape[1] == 8400:
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

    r, (left, top) = img_info
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - left / 2) / r
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - top / 2) / r
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

# ============================================================
#  ORT + COCO 评估
# ============================================================

def evaluate_model_ort(onnx_model_path, images_dir, annotations_json,
                       input_size, conf_thres, iou_thres, max_det, device_id):
    # from pycocotools.coco import COCO
    # from pycocotools.cocoeval import COCOeval
    # import onnxruntime as ort

    coco_gt = COCO(annotations_json)
    image_ids = coco_gt.getImgIds()

    providers = [("CUDAExecutionProvider", {"device_id": device_id}),
                 "CPUExecutionProvider"]
    sess_opt = ort.SessionOptions()
    sess_opt.log_severity_level = 3
    session = ort.InferenceSession(onnx_model_path, sess_options=sess_opt,
                                   providers=providers)
    input_name = session.get_inputs()[0].name

    results = []
    for img_id in tqdm(image_ids, desc="    Inference", leave=False):
        meta = coco_gt.loadImgs(img_id)[0]
        img_path = os.path.join(images_dir, meta["file_name"])
        if not os.path.exists(img_path):
            continue
        tensor, scale_pad = preprocess_image(img_path, input_size)
        out = session.run(None, {input_name: np.expand_dims(tensor, 0)})[0]
        for det in postprocess_yolo(out, scale_pad, conf_thres, iou_thres, max_det):
            x1, y1, x2, y2, score, cls_id = det
            results.append({
                "image_id": img_id, "category_id": int(cls_id),
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(score, 4),
            })

    if not results:
        print("    [WARN] 无检测结果, 返回零 mAP")
        return {"map50_95": 0.0, "map50": 0.0, "map75": 0.0,
                "ar100": 0.0, "num_detections": 0}

    tmp_det = annotations_json + ".tmp_det.json"
    with open(tmp_det, "w") as f:
        json.dump(results, f)
    try:
        coco_dt = coco_gt.loadRes(tmp_det)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate(); ev.accumulate(); ev.summarize()
        stats = ev.stats
    finally:
        _safe_remove(tmp_det)

    return {"map50_95": float(stats[0]), "map50": float(stats[1]),
            "map75": float(stats[2]), "ar100": float(stats[8]),
            "num_detections": len(results)}

# ============================================================
#  ONNX 操作 (v4: 记录算子节点名)
# ============================================================

def get_all_qdq_layers(int8_model):
    """
    提取所有量化层信息, 返回 list of dict:
      {
        "display_name": 算子节点名 (如 Conv_45, MatMul_130),
        "base_name":    原始Q节点基础名 (用于 find_qdq_node_pair),
      }
    
    逻辑: Q → q_out → DQ → dq_out → ConsumerNode
    display_name = ConsumerNode.name
    若找不到消费者, fallback 到 DQ 节点名或 Q base_name
    """
    # import onnx
    model = onnx.load(int8_model)
    graph = model.graph

    # 构建 tensor → 消费者节点列表 的映射
    tensor_consumers = {}
    for node in graph.node:
        for inp in node.input:
            if inp:
                tensor_consumers.setdefault(inp, []).append(node)

    layers = []
    seen_base = set()

    for node in graph.node:
        if node.op_type != "QuantizeLinear":
            continue

        q_name = node.name
        # 提取 base_name
        base = None
        for suf in ("_QuantizeLinear_Output", "_QuantizeLinear"):
            if q_name.endswith(suf):
                base = q_name[:-len(suf)]
                break
        if base is None:
            parts = q_name.rsplit("_", 2)
            base = parts[0] if len(parts) > 1 else q_name

        if base in seen_base:
            continue
        seen_base.add(base)

        # 找 Q 的输出 → DQ → DQ 的输出 → 消费者
        display = None
        q_output = node.output[0] if node.output else None
        if q_output and q_output in tensor_consumers:
            # Q 的消费者应该是 DequantizeLinear
            for dq_candidate in tensor_consumers[q_output]:
                if dq_candidate.op_type == "DequantizeLinear":
                    dq_output = dq_candidate.output[0] if dq_candidate.output else None
                    if dq_output and dq_output in tensor_consumers:
                        # DQ 的消费者就是目标算子
                        consumers = tensor_consumers[dq_output]
                        # 取第一个非 Q/DQ 消费者
                        for c in consumers:
                            if c.op_type not in ("QuantizeLinear", "DequantizeLinear"):
                                display = c.name
                                break
                    # fallback: 用 DQ 节点名
                    if display is None:
                        display = dq_candidate.name
                    break

        # 最终 fallback
        if display is None:
            display = base

        layers.append({
            "display_name": display,
            "base_name": base,
        })

    print(f"[INFO] 共发现 {len(layers)} 个量化层")
    return layers


def find_qdq_node_pair(int8_model, layer_base_name):
    """根据 base_name 查找 Q/DQ 精确节点名"""
    # import onnx
    model = onnx.load(int8_model)
    q_node = dq_node = None
    for node in model.graph.node:
        if node.op_type == "QuantizeLinear" and layer_base_name in node.name:
            if "DequantizeLinear" not in node.name:
                q_node = node.name
        elif node.op_type == "DequantizeLinear" and layer_base_name in node.name:
            dq_node = node.name
    return q_node, dq_node


def generate_fallback_model(int8_model, output_path, q_node_name, dq_node_name):
    """删除 Q/DQ 并手动重连 tensor + ORT 预检"""
    try:
        # import onnx
        # import onnxruntime as ort

        model = onnx.load(int8_model)
        graph = model.graph

        q_node = dq_node = None
        for node in graph.node:
            if node.name == q_node_name:
                q_node = node
            elif node.name == dq_node_name:
                dq_node = node
        if q_node is None or dq_node is None:
            err = f"未找到 Q={q_node_name} DQ={dq_node_name}"
            print(f"      [ERROR] {err}")
            return False

        upstream_tensor = q_node.input[0]
        dq_output = dq_node.output[0]
        if not upstream_tensor or not dq_output:
            print("      [ERROR] Q/DQ 输入输出为空")
            return False

        info = f"重连: '{dq_output}' → '{upstream_tensor}'"
        print(f"      {info}")

        replaced = 0
        for node in graph.node:
            if node.name in (q_node_name, dq_node_name):
                continue
            for i, inp in enumerate(node.input):
                if inp == dq_output:
                    node.input[i] = upstream_tensor
                    replaced += 1
        for out in graph.output:
            if out.name == dq_output:
                out.name = upstream_tensor

        repl_info = f"替换了 {replaced} 处下游引用"
        print(f"      {repl_info}")

        keep = [n for n in graph.node if n.name not in (q_node_name, dq_node_name)]
        del graph.node[:]
        graph.node.extend(keep)

        removed_outputs = set()
        for n in (q_node, dq_node):
            for o in n.output:
                removed_outputs.add(o)
        vi_keep = [v for v in graph.value_info if v.name not in removed_outputs]
        del graph.value_info[:]
        graph.value_info.extend(vi_keep)

        onnx.save(model, output_path)

        test_opt = ort.SessionOptions()
        test_opt.log_severity_level = 3
        test_sess = ort.InferenceSession(output_path, sess_options=test_opt,
                                         providers=["CPUExecutionProvider"])
        del test_sess
        print("      ✅ ORT 加载验证通过")
        return True

    except Exception as e:
        err_msg = str(e)
        print(f"      ❌ fallback 生成失败: {err_msg}")
        _safe_remove(output_path)
        return False

# ============================================================
#  辅助函数
# ============================================================

def _save_results(results, path):
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _classify(delta):
    abs_d = abs(delta)
    if delta > 0:
        if abs_d >= 0.01:    return "🔴 HIGH_SENS"
        elif abs_d >= 0.005: return "🟡 MED_SENS"
        elif abs_d >= 0.002: return "🟢 LOW_SENS"
        else:                return "⚪ SAFE"
    else:
        if abs_d >= 0.01:    return "⛔ ANOMALY"
        elif abs_d >= 0.002: return "⚠️ MINUS"
        else:                return "✅ OK"


def load_baseline(cache_path, int8_model, images_dir, ann_json,
                  input_size, conf_thres, iou_thres, max_det, device_id):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        val = cached['map50_95']
        msg = f"[INFO] 加载缓存基线 mAP50-95={val:.4f}"
        print(msg)
        return cached

    print("[INFO] ========== 评估基线 INT8 模型 ==========")
    t0 = time.time()
    metrics = evaluate_model_ort(int8_model, images_dir, ann_json, input_size,
                                 conf_thres, iou_thres, max_det, device_id)
    elapsed = time.time() - t0
    m50_95 = metrics['map50_95']
    m50 = metrics['map50']
    msg = f"[INFO] 基线 mAP50-95={m50_95:.4f}, mAP50={m50:.4f} ({elapsed:.1f}s)"
    print(msg)
    with open(cache_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return metrics


def print_and_export(results, baseline, output_csv):
    valid = [v for v in results.values() if v.get("status") == "OK"]
    invalid = [v for v in results.values() if v.get("status") != "OK"]

    if not valid:
        print("\n[WARN] 没有有效评估结果!")
        return

    ranked = sorted(valid, key=lambda x: x["delta_map50_95"], reverse=True)

    sep = "=" * 140
    dash = "-" * 140
    title = "逐层消融精度敏感层排名 (Δ>0=敏感层·回退FP16可提升 | Δ<0=异常·需排查)"
    bl_50_95 = baseline['map50_95']
    bl_50 = baseline['map50']
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
        d50 = r['delta_map50']
        node_short = r['display_name'][:40]
        label = _classify(d)
        line = (f"{i+1:<5} {node_short:<42} {r['map50_95']:>10.4f} "
                f"{d:>+12.4f} {r['map50']:>10.4f} {d50:>+10.4f} {label:>14}")
        print(line)

    if invalid:
        warn_cnt = len(invalid)
        print(f"\n[WARN] {warn_cnt} 个层未成功:")
        for v in invalid:
            err_msg = v.get('error', '')
            err_part = f" ({err_msg})" if err_msg else ""
            status = v['status']
            disp = v.get('display_name', v.get('layer', '?'))
            print(f"    {disp}: {status}{err_part}")

    with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['Rank', 'Operator_Node', 'Base_Name', 'mAP50_95', 'Delta_mAP50_95',
                     'mAP50', 'Delta_mAP50', 'Label', 'GenTime_s', 'EvalTime_s'])
        for i, r in enumerate(ranked):
            d = r["delta_map50_95"]
            d50 = r['delta_map50']
            label = _classify(d).encode('ascii', 'ignore').decode().strip()
            gen_t = r.get('gen_time_sec', '')
            eval_t = r.get('eval_time_sec', '')
            w.writerow([
                i + 1, r['display_name'], r.get('base_name', ''),
                f"{r['map50_95']:.4f}", f"{d:.4f}",
                f"{r['map50']:.4f}", f"{d50:.4f}",
                label, gen_t, eval_t,
            ])

    csv_msg = f"[INFO] CSV: {output_csv}"
    print(f"\n{csv_msg}")

    high_sens = [r for r in ranked if r["delta_map50_95"] >= 0.01]
    med_sens = [r for r in ranked if 0.005 <= r["delta_map50_95"] < 0.01]
    anomalies = [r for r in ranked if r["delta_map50_95"] <= -0.01]

    section_sep = "=" * 80
    print(f"\n{section_sep}")
    print("  📋 行动建议")
    print(section_sep)

    if high_sens:
        cnt = len(high_sens)
        print(f"  🔴 HIGH_SENS ({cnt} 层) — 强烈建议强制 FP16:")
        for r in high_sens:
            d_str = f"{r['delta_map50_95']:+.4f}"
            print(f"     • {r['display_name']}  (ΔmAP50-95={d_str})")

    if med_sens:
        cnt = len(med_sens)
        print(f"  🟡 MED_SENS ({cnt} 层) — 建议 per-channel 量化或重新校准:")
        for r in med_sens:
            d_str = f"{r['delta_map50_95']:+.4f}"
            print(f"     • {r['display_name']}  (ΔmAP50-95={d_str})")

    if anomalies:
        cnt = len(anomalies)
        print(f"  ⛔ ANOMALY ({cnt} 层) — 去除量化后精度暴跌, 需排查图拓扑:")
        for r in anomalies:
            d_str = f"{r['delta_map50_95']:+.4f}"
            print(f"     • {r['display_name']}  (ΔmAP50-95={d_str})")

    if not high_sens and not med_sens and not anomalies:
        print("  ✅ 所有层量化质量良好, 无需额外处理!")

    print()

# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="逐层消融 v4 (ORT+COCO)")
    parser.add_argument("--int8-model", default=DEFAULT_CONFIG["int8_model"])
    parser.add_argument("--images-dir", default=DEFAULT_CONFIG["images_dir"])
    parser.add_argument("--ann-json", default=DEFAULT_CONFIG["ann_json"])
    parser.add_argument("--img-size", type=int, default=DEFAULT_CONFIG["img_size"])
    parser.add_argument("--conf-thres", type=float, default=DEFAULT_CONFIG["conf_thres"])
    parser.add_argument("--iou-thres", type=float, default=DEFAULT_CONFIG["iou_thres"])
    parser.add_argument("--max-det", type=int, default=DEFAULT_CONFIG["max_det"])
    parser.add_argument("--device-id", type=int, default=DEFAULT_CONFIG["device_id"])
    parser.add_argument("--tmp-model", default=DEFAULT_CONFIG["tmp_model_path"])
    parser.add_argument("--output-csv", default=DEFAULT_CONFIG["output_csv"])
    parser.add_argument("--output-json", default=DEFAULT_CONFIG["output_json"])
    parser.add_argument("--baseline-cache", default=DEFAULT_CONFIG["baseline_cache"])
    parser.add_argument("--resume", action="store_true", help="断点续跑")
    args = parser.parse_args()

    tmp_model = args.tmp_model
    output_json = args.output_json
    input_size = (args.img_size, args.img_size)

    print("\n" + "=" * 80)
    print("  逐层消融 v4 (ORT + COCO · 算子节点名 + 双向分级 + tensor重连)")
    print("=" * 80)

    # Step 0: 提取量化层 (返回 list of dict)
    all_layers = get_all_qdq_layers(args.int8_model)
    if not all_layers:
        print("[ERROR] 未找到 QuantizeLinear 节点!")
        sys.exit(1)
    for i, layer_info in enumerate(all_layers):
        idx_str = f"{i+1:>4d}"
        disp = layer_info['display_name']
        base = layer_info['base_name']
        print(f"  {idx_str}. {disp:<40s} (base={base})")

    # Step 1: 基线
    baseline = load_baseline(args.baseline_cache, args.int8_model, args.images_dir,
                             args.ann_json, input_size, args.conf_thres,
                             args.iou_thres, args.max_det, args.device_id)

    # Step 2: 断点续跑
    results = {}
    if args.resume and os.path.exists(output_json):
        with open(output_json) as f:
            results = json.load(f)
        done_cnt = sum(1 for v in results.values() if v.get("status") == "OK")
        resume_msg = f"[INFO] 断点续跑: {done_cnt} 条已完成"
        print(resume_msg)

    # Step 3: 逐层消融
    total = len(all_layers)
    evaluated = skipped = failed = 0

    start_msg = f"[INFO] ========== 开始逐层消融 ({total} 层) =========="
    print(f"\n{start_msg}\n")

    for idx, layer_info in enumerate(all_layers):
        display_name = layer_info['display_name']
        base_name = layer_info['base_name']
        progress = f"[{idx+1}/{total}]"
        print(f"{progress} {display_name}")

        # 用 base_name 作为 results key (兼容旧缓存)
        cache_key = base_name

        if cache_key in results and results[cache_key].get("status") == "OK":
            skipped += 1
            continue

        q_node, dq_node = find_qdq_node_pair(args.int8_model, base_name)
        if not q_node or not dq_node:
            print("      ⚠️  未找到 Q/DQ 对, 跳过")
            results[cache_key] = {
                "display_name": display_name, "base_name": base_name,
                "status": "NO_QDQ_PAIR",
                "timestamp": datetime.now().isoformat(),
            }
            failed += 1
            _save_results(results, output_json)
            continue

        pair_msg = f"Q={q_node}  DQ={dq_node}"
        print(f"      {pair_msg}")

        t_gen = time.time()
        ok = generate_fallback_model(args.int8_model, tmp_model, q_node, dq_node)
        gen_time = time.time() - t_gen

        if not ok:
            results[cache_key] = {
                "display_name": display_name, "base_name": base_name,
                "status": "GEN_FAIL",
                "timestamp": datetime.now().isoformat(),
            }
            failed += 1
            _save_results(results, output_json)
            continue

        t_eval = time.time()
        try:
            fb = evaluate_model_ort(tmp_model, args.images_dir, args.ann_json,
                                    input_size, args.conf_thres, args.iou_thres,
                                    args.max_det, args.device_id)
            eval_time = time.time() - t_eval
        except Exception as e:
            eval_time = time.time() - t_eval
            err_str = str(e)
            print(f"      ❌ 评估异常: {err_str}")
            results[cache_key] = {
                "display_name": display_name, "base_name": base_name,
                "status": "EVAL_FAIL", "error": err_str,
                "timestamp": datetime.now().isoformat(),
            }
            failed += 1
            _save_results(results, output_json)
            _safe_remove(tmp_model)
            continue

        d50_95 = fb["map50_95"] - baseline["map50_95"]
        d50 = fb["map50"] - baseline["map50"]

        results[cache_key] = {
            "display_name": display_name, "base_name": base_name,
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
        summary = (f"mAP50-95={fb['map50_95']:.4f}(Δ={sign}{d50_95:.4f}) | "
                   f"mAP50={fb['map50']:.4f}(Δ={sign}{d50:.4f}) | "
                   f"生成{gen_time:.1f}s+评估{eval_time:.1f}s")
        print(f"      {summary}")

        evaluated += 1
        _save_results(results, output_json)
        _safe_remove(tmp_model)

    # Step 4: 排序 & 输出
    print_and_export(results, baseline, args.output_csv)

    stats_msg = (f"[STATS] 总={total} | 新评={evaluated} | "
                 f"跳过={skipped} | 失败={failed}")
    done_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n{stats_msg}")
    print(f"[DONE] {done_ts}")


if __name__ == '__main__':
    main()

"""
python D:\CV_FILE\my_FILE\TRT_opt\layer_ablation_eval.py \
--int8-model /root/my_FILE/yolov8_int8_1.onnx \
--images-dir /root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val \
--ann-json /root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val_coco.json \
--device-id 0 \
--img-size 640

python D:\CV_FILE\my_FILE\TRT_opt\remove_qdq_for_node.py \
--model /root/my_FILE/fixed_1.onnx \
--node Conv_3 \
--output fixed.onnx \
--verbose
"""