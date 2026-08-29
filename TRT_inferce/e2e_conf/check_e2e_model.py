import cv2
import onnxruntime as ort
import numpy as np

def check_with_onnxruntime(onnx_path, src_onnx, img_path, conf=0.45, iou=0.65):
    """验证 e2e 模型与原模型+CPU后处理输出一致。"""

    # ★ 修复: 图片路径由参数传入，不再硬编码
    img = cv2.imread(img_path)
    assert img is not None, f"图片读取失败: {img_path}"
    print(f"  测试图片: {img_path} ({img.shape[1]}x{img.shape[0]})")

    e2e = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = ort.InferenceSession(src_onnx, providers=["CPUExecutionProvider"])
    print(f"  e2e 输入: {[(i.name, i.shape) for i in e2e.get_inputs()]}")
    print(f"  e2e 输出: {[(o.name, o.shape) for o in e2e.get_outputs()]}")

    # ---------- 参考后处理 ----------
    def letterbox_params(h, w, target=640):
        scale = target / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        pad_h = (target - nh) // 2
        pad_w = (target - nw) // 2
        return scale, pad_w, pad_h

    def preprocess_cpu(img, target=640):
        h, w = img.shape[:2]
        scale, pad_w, pad_h = letterbox_params(h, w, target)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, scale, pad_w, pad_h

    def postprocess_ref(output, scale, pad_w, pad_h, conf_th, iou_th):
        preds = output[0].T  # (8400, 4+num_cls)
        boxes = preds[:, :4]
        scores = preds[:, 4:]
        cls = scores.argmax(1)
        confs = scores.max(1)
        mask = confs > conf_th
        boxes, confs, cls = boxes[mask], confs[mask], cls[mask]
        if len(boxes) == 0:
            return []
        x, y, w, h = boxes.T
        x1 = (x - w/2 - pad_w) / scale
        y1 = (y - h/2 - pad_h) / scale
        x2 = (x + w/2 - pad_w) / scale
        y2 = (y + h/2 - pad_h) / scale
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(b), float(c), float(d)]
             for a, b, c, d in zip(x1, y1, x2, y2)],
            confs.tolist(), conf_th, iou_th)
        idx = [i[0] if isinstance(i, (list, tuple)) else i for i in idx]
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                 float(confs[i]), int(cls[i])) for i in idx]

    def canon(d):
        return sorted([(round(float(a), 2), round(float(b), 2),
                        round(float(c), 2), round(float(d), 2),
                        round(float(e), 2), int(f))
                       for a, b, c, d, e, f in d])

    scale, pad_x, pad_y = letterbox_params(img.shape[0], img.shape[1])

    # 1. 参考结果
    blob, *_ = preprocess_cpu(img)
    ref_input_name = ref.get_inputs()[0].name
    out0 = ref.run(None, {ref_input_name: blob})[0]
    cpu_dets = postprocess_ref(out0, scale, pad_x, pad_y, conf, iou)

    # 2. e2e 结果
    e2e_out = e2e.run(None, {
        "image_raw": img[None],
        "iou_thresh": np.array([iou], dtype=np.float32),
        "score_thresh": np.array([conf], dtype=np.float32),
    })[0]

    # ★ 增强调试: 打印原始统计信息
    valid_mask = e2e_out[:, 4] > 0
    e2e_valid = e2e_out[valid_mask]
    print(f"  e2e 原始输出: {e2e_out.shape}, "
          f"有效框={len(e2e_valid)}, "
          f"conf范围=[{e2e_valid[:, 4].min():.4f}, {e2e_valid[:, 4].max():.4f}]"
          if len(e2e_valid) > 0 else
          f"  e2e 原始输出: {e2e_out.shape}, 有效框=0 (全零)")

    dets = e2e_valid

    # 3. 比较
    cc, ce = canon(cpu_dets), canon(dets)
    same = (len(cc) == len(ce) and all(
        all(abs(x - y) <= 1.0 for x, y in zip(a[:4], b[:4])) and a[5] == b[5]
        for a, b in zip(cc, ce)))

    status = "✅ OK" if same else "❌ DIFF"
    print(f"  conf={conf} iou={iou}: 参考 {len(cc)} 框 vs e2e {len(ce)} 框 -> {status}")

    if not same:
        print("  --- 差异详情 ---")
        max_show = 10
        for i, (a, b) in enumerate(zip(cc, ce)):
            if i >= max_show:
                print(f"  ... 还有 {min(len(cc), len(ce)) - max_show} 对未显示")
                break
            match = (all(abs(x-y) <= 1.0 for x, y in zip(a[:4], b[:4])) and a[5] == b[5])
            tag = "✓" if match else "✗"
            print(f"  {tag} ref={a}  e2e={b}")
        if len(cc) != len(ce):
            print(f"  数量不匹配: ref={len(cc)}, e2e={len(ce)}")
            if len(ce) < len(cc):
                print(f"  e2e 缺失的参考框:")
                for a in cc[len(ce):]:
                    print(f"    {a}")

    return same

if __name__ == "__main__":
    onnx_path = r"D:\Onnxruntime_quant\best_PCB_e2e_conf_111.onnx"
    src_onnx = r"D:\Onnxruntime_quant\best_PCB.onnx"
    img_path = r"D:\Onnxruntime_quant\onnxruntime_quant\data_test\images\test\00041200.jpg"

    for conf in (0.45, 0.001): # 测试两组阈值
        check_with_onnxruntime(onnx_path, src_onnx, img_path, conf=conf, iou=0.65)