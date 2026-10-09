"""
graphsurgeon_e2e.py — 静态模型端到端：只烧入 BGR→RGB/归一化(预处理) + decode/NMS(后处理)。

输入图像在 CPU 侧已经完成「解码 + resize(letterbox 到 640×640)」，因此图中不再包含
任何 Resize / Pad 等 resize 相关操作（这些操作被剔除，改由宿主 CPU 完成）。

模型三个输入：
  image_raw     (1, 640, 640, 3) uint8 BGR NHWC —— CPU 解码并 resize 后的图（固定尺寸）
  iou_thresh    (1,) float32    NMS 的 IoU 阈值（运行时传）
  score_thresh  (1,) float32    置信度阈值（运行时传）
输出：
  detections    (max_dets, 6) float32 = [x1,y1,x2,y2,conf,cls]，坐标在 640×640 输入空间

与 e2e_dyn 版本的区别：
  - 输入是静态 [1,640,640,3]（不再动态 H/W），因为 resize 已经移到 CPU；
  - 预处理只保留 Cast/Transpose/Gather/Mul（颜色+归一化+布局），剔除 letterbox；
  - 后处理 decode 后不再「反 letterbox」（无 scale/pad 可复用），坐标即 640×640 空间。

用法:
    python graphsurgeon_e2e.py            # 生成 yolov8_int8_static_e2e.onnx
    python graphsurgeon_e2e.py --check    # 生成后用 onnxruntime 与 CPU 参考链路对比
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F32 = TensorProto.FLOAT   # ONNX 数据类型别名，简化书写
I64 = TensorProto.INT64


def build_e2e(src, dst, img_size=640, num_anchors=8400, num_classes=6, max_dets=300):
    model = onnx.load(src)
    g = model.graph
    opset = model.opset_import[0].version   # 决定 ReduceMax 的 axes 是属性(<18)还是输入(>=18)
    orig_in = g.input[0].name               # 原始模型输入名（images）
    orig_out = g.output[0].name             # 原始模型输出名（output0）

    num_box_cols = 4
    num_out_cols = num_box_cols + num_classes   # 4 + 6 = 10

    pre_nodes = []
    post_nodes = []
    cur = pre_nodes
    inits = []

    # 创建常量初始化器：ONNX 没有字面量，所有常量都必须是 named initializer。
    def C(name, value, dtype=F32):
        arr = np.asarray(value, dtype=np.float32) if dtype == F32 else np.asarray(value, dtype=np.int64)
        inits.append(numpy_helper.from_array(arr, name=name))
        return name

    # 创建算子节点。cur 是可切换的节点列表（先 pre_nodes，后 post_nodes）。
    def N(op, ins, outs, **attrs):
        cur.append(helper.make_node(op, ins, [outs] if isinstance(outs, str) else outs, **attrs))

    # ---- 常量 ----
    c_inv255  = C("c_inv255", 1.0 / 255.0, F32)      # 归一化系数 1/255
    c_rgb_idx = C("c_rgb_idx", [2, 1, 0], I64)       # BGR→RGB 通道重排索引
    c_half    = C("c_half", 0.5, F32)                # xywh→xyxy 用
    c_0_i64   = C("c_0_i64", [0], I64)
    c_0_f     = C("c_0_f", 0.0, F32)
    c_maxbox  = C("c_maxbox", max_dets, I64)         # NMS 最大保留框数
    c_idx2    = C("c_idx2", 2, I64)                  # NMS 输出第 2 列 = box index
    # 注意：没有 c_iou / c_score —— 两个阈值改为运行时图输入。

    # SLICE 语法糖：自动创建 starts/ends 两个常量，并生成 Slice 节点。
    def SLICE(x, starts, ends, out):
        s = C(f"{out}_st", starts, I64)
        e = C(f"{out}_en", ends, I64)
        N("Slice", [x, s, e], out)
        return out

    # ================= 预处理（仅颜色/归一化/布局，无 resize） =================
    # 输入 image_raw 已是 640×640 的 BGR uint8，这里只做：
    #   uint8→float32 → NHWC→NCHW → BGR→RGB → /255
    N("Cast", ["image_raw"], "img_f", to=F32)          # UINT8 → FLOAT32
    N("Transpose", ["img_f"], "tr", perm=[0, 3, 1, 2])  # NHWC → NCHW
    N("Gather", ["tr", c_rgb_idx], "rgb", axis=1)       # BGR → RGB
    N("Mul", ["rgb", c_inv255], "preprocessed")         # /255 归一化

    # ================= 后处理（decode + NMS，坐标即 640×640 空间） =================
    cur = post_nodes
    N("Transpose", [orig_out], "trans", perm=[0, 2, 1])   # (1,10,8400) → (1,8400,10)
    SLICE("trans", [0, 0, 0], [1, num_anchors, num_box_cols], "boxes")          # 前4列 = cx,cy,w,h
    SLICE("trans", [0, 0, num_box_cols], [1, num_anchors, num_out_cols], "scores")  # 后6列 = 类别分数

    # 拆分 cx,cy,w,h
    SLICE("boxes", [0, 0, 0], [1, num_anchors, 1], "cx")
    SLICE("boxes", [0, 0, 1], [1, num_anchors, 2], "cy")
    SLICE("boxes", [0, 0, 2], [1, num_anchors, 3], "bw")
    SLICE("boxes", [0, 0, 3], [1, num_anchors, 4], "bh")

    # xywh → xyxy（直接在 640×640 空间，无需反 letterbox）
    N("Mul", ["bw", c_half], "w2")
    N("Mul", ["bh", c_half], "h2")
    N("Sub", ["cx", "w2"], "x1")
    N("Add", ["cx", "w2"], "x2")
    N("Sub", ["cy", "h2"], "y1")
    N("Add", ["cy", "h2"], "y2")
    N("Concat", ["x1", "y1", "x2", "y2"], "boxes_final", axis=2)   # (1,8400,4)

    # 提取置信度与类别
    if opset >= 18:
        c_axes2 = C("c_axes2", [2], I64)
        N("ReduceMax", ["scores", c_axes2], "conf", keepdims=1)     # opset18+：axes 是输入
    else:
        N("ReduceMax", ["scores"], "conf", axes=[2], keepdims=1)
    N("ArgMax", ["scores"], "cls", axis=2, keepdims=1)
    N("Cast", ["cls"], "cls_f", to=F32)

    # NMS：iou_thresh / score_thresh 是运行时图输入（一个 engine 通吃推理/mAP 阈值）
    # center_point_box=0 表示 boxes 是 (y1,x1,y2,x2) 格式（ONNX NMS 约定）
    N("Transpose", ["conf"], "conf_t", perm=[0, 2, 1])               # (1,1,8400)
    N("NonMaxSuppression", ["boxes_final", "conf_t", c_maxbox, "iou_thresh", "score_thresh"],
      "selected", center_point_box=0)

    # 收集选中结果
    N("Gather", ["selected", c_idx2], "box_idx", axis=1)             # NMS 输出第 2 列 = box index
    N("Gather", ["boxes_final", "box_idx"], "boxes_sel", axis=1)
    N("Gather", ["conf", "box_idx"], "conf_sel", axis=1)
    N("Gather", ["cls_f", "box_idx"], "cls_sel", axis=1)
    N("Concat", ["boxes_sel", "conf_sel", "cls_sel"], "det", axis=2)  # (1,N,6)

    # 填充到固定 max_dets 行（TRT 要求输出 shape 固定），不足行补 0，消费端过滤 conf>0
    N("Shape", ["selected"], "shape_sel")
    N("Gather", ["shape_sel", c_0_i64], "num_sel", axis=0)
    N("Sub", [c_maxbox, "num_sel"], "pad_det")
    N("Concat", [c_0_i64, c_0_i64, c_0_i64, c_0_i64, "pad_det", c_0_i64], "det_pads", axis=0)
    N("Pad", ["det", "det_pads", c_0_f], "det_padded", mode="constant")
    N("Squeeze", ["det_padded", c_0_i64], "detections")              # (max_dets, 6)

    # ================= 重写 graph I/O =================
    # 把原模型第一个节点的输入从 "images" 改为 "preprocessed"（缝合主干与预处理子图）
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == orig_in:
                node.input[i] = "preprocessed"

    del g.input[:]
    g.input.append(helper.make_tensor_value_info(
        "image_raw", TensorProto.UINT8, [1, img_size, img_size, 3]))
    # NonMaxSuppression 要求阈值输入是 1-D tensor（而非 0 维标量）
    g.input.append(helper.make_tensor_value_info("iou_thresh", F32, [1]))
    g.input.append(helper.make_tensor_value_info("score_thresh", F32, [1]))

    del g.output[:]
    g.output.append(helper.make_tensor_value_info("detections", F32, [max_dets, 6]))

    # 拼接：预处理 + 原始模型 + 后处理
    all_nodes = pre_nodes + list(g.node) + post_nodes
    del g.node[:]
    g.node.extend(all_nodes)
    g.initializer.extend(inits)

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"[e2e-static] saved {dst}")
    return model


# ---------------- 参考实现（供 onnxruntime 校验，与宿主 CPU 侧行为一致） ----------------

def letterbox_640(image):
    """宿主 CPU 侧的「解码 + resize」：等比缩放 + 居中灰边，得到 640×640 BGR uint8。"""
    import cv2
    h, w = image.shape[:2]
    scale = min(640.0 / h, 640.0 / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    pad_h, pad_w = (640 - nh) // 2, (640 - nw) // 2
    canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized
    return canvas


def decode_nms(output, conf_thresh, nms_thresh):
    """参考后处理：decode(xywh→xyxy) + 置信度过滤 + NMS，坐标即 640×640 空间。"""
    import cv2
    preds = output[0].T                        # [num_boxes, 4+6]
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    cls = scores.argmax(1)
    confs = scores.max(1)

    mask = confs > conf_thresh
    boxes, confs, cls = boxes[mask], confs[mask], cls[mask]
    if len(boxes) == 0:
        return []

    x, y, w, h = boxes.T
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    # cv2.dnn.NMSBoxes 按 Rect(x,y,w,h) 语义解析，须传宽高而非 x2/y2
    idx = cv2.dnn.NMSBoxes(
        [[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in zip(x1, y1, x2, y2)],
        confs.tolist(), conf_thresh, nms_thresh)
    if idx is None or len(idx) == 0:
        return []
    idx = [i[0] if isinstance(i, (list, tuple)) else i for i in idx]
    return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
             float(confs[i]), int(cls[i])) for i in idx]


def canon(dets):
    return sorted([(round(float(a), 2), round(float(b), 2), round(float(c), 2),
                    round(float(d), 2), round(float(e), 2), int(f))
                   for a, b, c, d, e, f in dets])


def _pick_providers():
    """选择 onnxruntime 执行后端。

    Q/DQ(显式量化)模型在 CPUExecutionProvider 下数值会明显失真（实测分数 0.59 vs TRT 0.95），
    因此优先用 CUDAExecutionProvider（与 TRT 更接近），无 CUDA 时回退 CPU 并告警。
    """
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    print("  [WARN] 无 CUDAExecutionProvider，回退 CPU（注意：CPU EP 对 Q/DQ 量化模型数值会失真）")
    return ["CPUExecutionProvider"]


def check_with_onnxruntime(onnx_path, src_onnx, img_path, conf=0.45, iou=0.65):
    """用 onnxruntime 验证：e2e(运行时阈值) 与「原模型 + CPU 后处理(同阈值)」输出一致。"""
    import cv2
    import onnxruntime as ort

    providers = _pick_providers()
    e2e = ort.InferenceSession(onnx_path, providers=providers)
    ref = ort.InferenceSession(src_onnx, providers=providers)
    print("  providers:", e2e.get_providers())
    print("  e2e 输入:", [(i.name, i.shape) for i in e2e.get_inputs()])
    print("  e2e 输出:", [(o.name, o.shape) for o in e2e.get_outputs()])

    img = cv2.imread(img_path)
    canvas = letterbox_640(img)                 # CPU 侧 decode + resize

    # 1. 参考链路：640 图上只做颜色/归一化/布局 → 原模型 → decode + NMS
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    ref_in = ref.get_inputs()[0].name
    out0 = ref.run(None, {ref_in: blob[None]})[0]
    cpu_dets = decode_nms(out0, conf, iou)

    # 2. e2e 链路：640 图 + 运行时阈值直接出框
    dets = e2e.run(None, {
        "image_raw": canvas[None],
        "iou_thresh": np.array([iou], dtype=np.float32),
        "score_thresh": np.array([conf], dtype=np.float32),
    })[0]
    dets = dets[dets[:, 4] > 0]                 # 过滤 padding 零行

    cc, ce = canon(cpu_dets), canon(dets)
    same = len(cc) == len(ce) and all(
        all(abs(x - y) <= 1.0 for x, y in zip(a[:4], b[:4])) and a[5] == b[5]
        for a, b in zip(cc, ce))
    print(f"  conf={conf} iou={iou}: 参考 {len(cc)} 框 vs e2e {len(ce)} 框 -> "
          f"{'OK' if same else 'DIFF'}")
    if not same:
        for a, b in zip(cc, ce):
            print("     ref", a, " e2e", b)
    return same


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/my_FILE/models/yolov8_int8_static.onnx")
    ap.add_argument("--dst", default="/root/my_FILE/models/yolov8_int8_static_e2e.onnx")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--img", default="/root/my_FILE/infer_images/00041000.jpg")
    args = ap.parse_args()

    build_e2e(args.src, args.dst, max_dets=args.max_det)
    if args.check:
        for conf in (0.45, 0.001):            # 测试两组阈值：推理 / mAP
            check_with_onnxruntime(args.dst, args.src, args.img, conf=conf, iou=0.65)
