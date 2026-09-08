"""
graphsurgeon_e2e_conf_iou_dyn_v2.py — batch / H / W 全动态的端到端 e2e 模型（运行时阈值）。

相对 graphsurgeon_e2e_conf_iou_dyn.py 的改动（修复 batch=2 报
"/model.22/Expand_5 invalid expand shape"）:
  ★0' Reshape 补丁加 rank 守卫。
      v1 把源模型里所有 [1,...] 开头的 Reshape 首位 1 一律换成 Concat([batch,...])。
      但 YOLOv8 头 make_anchors/meshgrid 里有一类 (w,) -> (1,w) 的行向量 Reshape
      （/model.22/Reshape_9/14/19，shape 常量 [1,-1]），其首位 1 是"1 行"不是 batch：
      batch=1 时 1==batch 碰巧正确；batch=2 时 Reshape 出 (B,w)，随后 Expand 到
      (h,w) 就报 invalid expand shape（/model.22/Expand_5 即此）。
      v2 规则：仅当 Reshape 数据输入的声明 rank>=2 时才认为首位 1 是 batch；
      rank==1（meshgrid 坐标向量）跳过；rank 未知也跳过并打印提示。
      本源模型 yolov8_int8_dy_exclude.onnx 中真正带 batch 的 Reshape
      （DFL / 检测头 [1,64,-1] 类）本就已写成动态（首位来自 Shape->Unsqueeze），
      所以 v2 下一个都不需要改 —— 修补仅对"batch 被烧死"的导出模型生效。
  ★6' 修复检查函数 postprocess_ref: cv2.dnn.NMSBoxes 按 C++ Rect 语义解析输入框
      = (x, y, w, h)，原代码直接传 (x1,y1,x2,y2) 把 x2/y2 当宽高，IoU 全错。
      conf=0.45 时只有 1 个框侥幸看不出；conf=0.001 多框时 NMS 结果不对，
      曾误判 e2e 有问题。传 (x1,y1,x2-x1,y2-y1) 后 conf=0.001 下参考与 e2e 逐框一致。

  其余与 v1 相同:
  ★1 graph 输入 (batch, height, width, 3) uint8，batch 动态
  ★2 Resize 目标尺寸 [batch, nh, nw, 3]
  ★3 后处理所有 Slice 的 batch 维用 INT64_MAX 表示"到末尾"
  ★4 NMS 输出 (N,3)=[batch_idx,class_idx,box_idx]，flat_idx=b*8400+box 展平 Gather
  ★5 输出 (batch, max_dets, 6)：ScatterND 把 NMS 结果按图分散进 (B,300,6) 零张量

  注意: 一个 batch 内所有图必须同 H/W，scale/pad 是整批共享的一组值。

输入 :
  image_raw     (B, H, W, 3) uint8 BGR NHWC，B/H/W 全动态
  iou_thresh    (1,) float32   NMS 的 IoU 阈值（整批共享）
  score_thresh  (1,) float32   置信度阈值（整批共享）
输出 :
  detections    (B, max_det, 6) float32 = [x1,y1,x2,y2,conf,cls]，原图坐标、已 NMS，
                每张图占 rows[0:n_b]，其余为 0（conf=0 可过滤）

用法:
    python graphsurgeon_e2e_conf_iou_dyn_v2.py            # 构建 + batch=1/2 验证
    python graphsurgeon_e2e_conf_iou_dyn_v2.py --src xxx.onnx --dst yyy.onnx
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64
IMAX = int(np.iinfo(np.int64).max)   # Slice ends 用: 表示"该维到末尾"（动态 batch 维）


def build_e2e_conf_dyn(src, dst, max_dets=300, num_classes=6):

    model = onnx.load(src)
    g = model.graph

    # ★0' 源模型修补（v2，带 rank 守卫）: 只修"数据输入 rank>=2 且目标形状首位为 1"
    #    的 Reshape（首位 1 = batch，如 [1,64,-1]）；rank==1 的（meshgrid 的
    #    (w,)->(1,w) 行向量，[1,-1]）绝不能改。
    #    "batch" 张量由本脚本加的预处理节点（Gather(Shape(img_f),0)）产出。
    vi_map = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}

    def declared_rank(t):
        v = vi_map.get(t)
        if v is None:
            return None
        try:
            return len(v.type.tensor_type.shape.dim)
        except Exception:
            return None

    init_by_name = {i.name: i for i in g.initializer}
    inserts = []                       # (插入位置, 节点)
    for idx, n in enumerate(g.node):
        if not (n.op_type == "Reshape" and len(n.input) >= 2 and n.input[1] in init_by_name):
            continue
        arr = numpy_helper.to_array(init_by_name[n.input[1]])
        if not (arr.dtype == np.int64 and arr.size and arr.flat[0] == 1):
            continue
        rank = declared_rank(n.input[0])
        if rank is None:
            print(f"[e2e-dyn-v2] skip {n.name}: 数据输入 {n.input[0]} 无 shape 声明，"
                  f"首位 1 是否为 batch 无法判定，不改（shape={arr.tolist()}）")
            continue
        if rank < 2:
            print(f"[e2e-dyn-v2] skip {n.name}: 输入 rank={rank}（坐标向量，首位 1 非 batch，"
                  f"如 make_anchors/meshgrid），不改（shape={arr.tolist()}）")
            continue
        rest_name = n.input[1] + "_rest"
        g.initializer.append(numpy_helper.from_array(arr[1:].copy(), name=rest_name))
        cname = n.input[1] + "_dyn"
        inserts.append((idx, helper.make_node(
            "Concat", ["batch", rest_name], [cname], axis=0)))
        n.input[1] = cname
        print(f"[e2e-dyn-v2] patched {n.name}: {arr.tolist()} -> Concat([batch, {arr[1:].tolist()}])")
    for pos, node in reversed(inserts):        # 倒序插入，位置不漂移；紧邻其 Reshape 之前
        g.node.insert(pos, node)

    opset = model.opset_import[0].version   # 决定 ReduceMax 的 axes 是属性(<18)还是输入(>=18)
    orig_in = g.input[0].name
    orig_out = g.output[0].name

    pre_nodes = []
    post_nodes = []
    cur = pre_nodes
    inits = []

    def C(name, value, dtype=F32):
        arr = np.asarray(value, dtype=np.float32) if dtype == F32 else np.asarray(value, dtype=np.int64)
        inits.append(numpy_helper.from_array(arr, name=name))
        return name

    def N(op, ins, outs, **attrs):
        cur.append(helper.make_node(op, ins, [outs] if isinstance(outs, str) else outs, **attrs))

    # ---- 常量 ----
    c_1_i64   = C("c_1_i64", [1], I64)
    c_2_i64   = C("c_2_i64", [2], I64)
    c_3_i64   = C("c_3_i64", [3], I64)
    c_0_i64   = C("c_0_i64", [0], I64)
    c_6_i64   = C("c_6_i64", [num_classes], I64)   # ★5 类别数

    c_640_i64 = C("c_640_i64", [640], I64)
    c_640_f   = C("c_640_f", 640.0, F32)
    c_114_f   = C("c_114_f", 114.0, F32)
    c_inv255  = C("c_inv255", 1.0 / 255.0, F32)
    c_rgb_idx = C("c_rgb_idx", [2, 1, 0], I64)
    c_half    = C("c_half", 0.5, F32)
    c_0_f     = C("c_0_f", 0.0, F32)
    c_maxbox  = C("c_maxbox", [max_dets], I64)      # 1维: NMS 输入 + out_shape Concat 都可用

    c_imax    = C("c_imax", [IMAX], I64)            # ★3 Slice ends: 到末尾
    c_8400_s  = C("c_8400_s", 8400, I64)            # ★4 0-d 标量，展平索引乘子
    c_neg1    = C("c_neg1", [-1], I64)              # Reshape 到 1 维/[-1,4]
    c_bx4     = C("c_bx4", [-1, 4], I64)
    c_bx1     = C("c_bx1", [-1, 1], I64)
    c_ax1     = C("c_ax1", [1], I64)                # Unsqueeze axes

    def SLICE(x, starts, ends, out):
        s = C(f"{out}_st", starts, I64)
        e = C(f"{out}_en", ends, I64)
        N("Slice", [x, s, e], out)
        return out

    # ================= 预处理（batch 动态版） =================
    N("Cast", ["image_raw"], "img_f", to=F32)
    N("Shape", ["img_f"], "shape_img")                       # [B,H,W,3]
    N("Gather", ["shape_img", c_0_i64], "batch", axis=0)     # ★2 batch 标量(实际为[1]形状)
    N("Gather", ["shape_img", c_1_i64], "H", axis=0)
    N("Gather", ["shape_img", c_2_i64], "W", axis=0)
    N("Cast", ["H"], "Hf", to=F32)
    N("Cast", ["W"], "Wf", to=F32)
    N("Max", ["Hf", "Wf"], "maxhw")
    N("Div", [c_640_f, "maxhw"], "scale")
    N("Mul", ["Hf", "scale"], "hsc")
    N("Mul", ["Wf", "scale"], "wsc")
    N("Floor", ["hsc"], "hfl")
    N("Floor", ["wsc"], "wfl")
    N("Cast", ["hfl"], "nh", to=I64)
    N("Cast", ["wfl"], "nw", to=I64)

    N("Concat", ["batch", "nh", "nw", c_3_i64], "sizes", axis=0)  # ★2 [B,nh,nw,3]
    N("Resize", ["img_f", "", "", "sizes"], "resized",
      mode="linear", coordinate_transformation_mode="half_pixel")

    # 居中 Padding（batch 维 pad 0，对任意 B 成立）
    N("Sub", [c_640_i64, "nh"], "padh_tot")
    N("Sub", [c_640_i64, "nw"], "padw_tot")
    N("Div", ["padh_tot", c_2_i64], "pad_top")
    N("Div", ["padw_tot", c_2_i64], "pad_left")
    N("Sub", ["padh_tot", "pad_top"], "pad_bot")
    N("Sub", ["padw_tot", "pad_left"], "pad_right")
    N("Concat", [c_0_i64, "pad_top", "pad_left", c_0_i64,
                 c_0_i64, "pad_bot", "pad_right", c_0_i64], "pads", axis=0)
    N("Pad", ["resized", "pads", c_114_f], "padded", mode="constant")

    N("Transpose", ["padded"], "tr", perm=[0, 3, 1, 2])
    N("Gather", ["tr", c_rgb_idx], "rgb", axis=1)
    N("Mul", ["rgb", c_inv255], "preprocessed")              # (B,3,640,640)

    # ================= 后处理（batch 感知版） =================
    cur = post_nodes
    N("Transpose", [orig_out], "trans", perm=[0, 2, 1])      # (B,10,8400)→(B,8400,10)
    SLICE("trans", [0, 0, 0], [IMAX, 8400, 4], "boxes")      # ★3 batch 维到末尾
    SLICE("trans", [0, 0, 4], [IMAX, 8400, 10], "scores")

    SLICE("boxes", [0, 0, 0], [IMAX, 8400, 1], "cx")
    SLICE("boxes", [0, 0, 1], [IMAX, 8400, 2], "cy")
    SLICE("boxes", [0, 0, 2], [IMAX, 8400, 3], "bw")
    SLICE("boxes", [0, 0, 3], [IMAX, 8400, 4], "bh")

    N("Mul", ["bw", c_half], "w2")
    N("Mul", ["bh", c_half], "h2")
    N("Sub", ["cx", "w2"], "x1")
    N("Add", ["cx", "w2"], "x2")
    N("Sub", ["cy", "h2"], "y1")
    N("Add", ["cy", "h2"], "y2")

    N("Cast", ["pad_left"], "padl_f", to=F32)
    N("Cast", ["pad_top"], "padt_f", to=F32)
    N("Sub", ["x1", "padl_f"], "sub_x1")
    N("Sub", ["y1", "padt_f"], "sub_y1")
    N("Sub", ["x2", "padl_f"], "sub_x2")
    N("Sub", ["y2", "padt_f"], "sub_y2")
    N("Div", ["sub_x1", "scale"], "x1o")
    N("Div", ["sub_y1", "scale"], "y1o")
    N("Div", ["sub_x2", "scale"], "x2o")
    N("Div", ["sub_y2", "scale"], "y2o")
    N("Concat", ["x1o", "y1o", "x2o", "y2o"], "boxes_final", axis=2)   # (B,8400,4)

    if opset >= 18:
        c_axes2 = C("c_axes2", [2], I64)
        N("ReduceMax", ["scores", c_axes2], "conf", keepdims=1)   # opset>=18: axes 是输入
        N("ArgMax", ["scores"], "cls_i", axis=2, keepdims=1)
    else:
        N("ReduceMax", ["scores"], "conf", axes=[2], keepdims=1)       # (B,8400,1)
        N("ArgMax", ["scores"], "cls_i", axis=2, keepdims=1)
    N("Cast", ["cls_i"], "cls_f", to=F32)                              # (B,8400,1)

    N("Transpose", ["conf"], "conf_t", perm=[0, 2, 1])                 # (B,1,8400)
    N("NonMaxSuppression", ["boxes_final", "conf_t", c_maxbox, "iou_thresh", "score_thresh"],
      "selected", center_point_box=0)                                  # ★4 (N,3)=[b,c,box]

    # ---- ★4 batch 感知 gather: flat_idx = b*8400 + box，在展平张量上取 ----
    SLICE("selected", [0, 0], [IMAX, 1], "b_idx")                      # (N,1)
    SLICE("selected", [0, 2], [IMAX, 3], "box_idx")                    # (N,1)
    N("Mul", ["b_idx", c_8400_s], "b_off")
    N("Add", ["b_off", "box_idx"], "flat_2d")                          # (N,1)
    N("Reshape", ["flat_2d", c_neg1], "flat_idx")                      # (N,)

    N("Reshape", ["boxes_final", c_bx4], "boxes_flat")                 # (B*8400,4)
    N("Gather", ["boxes_flat", "flat_idx"], "boxes_sel", axis=0)       # (N,4)
    N("Reshape", ["conf", c_bx1], "conf_flat")                         # (B*8400,1)
    N("Gather", ["conf_flat", "flat_idx"], "conf_sel", axis=0)         # (N,1)
    N("Reshape", ["cls_f", c_bx1], "cls_flat")                         # (B*8400,1)
    N("Gather", ["cls_flat", "flat_idx"], "cls_sel", axis=0)           # (N,1)
    N("Concat", ["boxes_sel", "conf_sel", "cls_sel"], "det", axis=1)   # (N,6)

    # ---- ★5 分散回 (B,300,6)：ScatterND 按图归位，空位自然全零（取代 Pad） ----
    N("Shape", ["selected"], "shape_sel")
    N("Gather", ["shape_sel", c_0_i64], "num_sel", axis=0)             # (1,)
    N("Range", [c_0_i64, "num_sel", c_1_i64], "rows")                  # (N,)
    N("Unsqueeze", ["rows", c_ax1], "rows_u")                          # (N,1)
    N("Concat", ["b_idx", "rows_u"], "scatter_idx", axis=1)            # (N,2)
    N("Concat", ["batch", c_maxbox, c_6_i64], "out_shape", axis=0)     # [B,300,6]
    N("ConstantOfShape", ["out_shape"], "det_zeros",
      value=numpy_helper.from_array(np.array([0.0], np.float32)))      # (B,300,6) 全零
    N("ScatterND", ["det_zeros", "scatter_idx", "det"], "detections")  # (B,300,6)

    # ================= 重写 graph I/O =================
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == orig_in:
                node.input[i] = "preprocessed"

    del g.input[:]
    del g.value_info[:]   # ★0b 源模型导出的中间张量声明仍带 batch=1，清掉让运行时重新推断
    g.input.append(helper.make_tensor_value_info(
        "image_raw", TensorProto.UINT8, ["batch", "height", "width", 3]))   # ★1 batch 动态
    g.input.append(helper.make_tensor_value_info("iou_thresh", F32, [1]))
    g.input.append(helper.make_tensor_value_info("score_thresh", F32, [1]))

    del g.output[:]
    g.output.append(helper.make_tensor_value_info(
        "detections", F32, ["batch", max_dets, 6]))                        # ★5

    all_nodes = pre_nodes + list(g.node) + post_nodes
    del g.node[:]
    g.node.extend(all_nodes)
    g.initializer.extend(inits)

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"[e2e-dyn-v2] saved {dst}")
    return model


def check_with_onnxruntime(onnx_path, src_onnx, conf=0.45, iou=0.65):
    """onnxruntime 验证: batch=1 与参考后处理一致；batch=2 两张图结果互不串扰。"""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import onnxruntime as ort
    import cv2
    from common import preprocess_cpu, letterbox_params

    e2e = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = ort.InferenceSession(src_onnx, providers=["CPUExecutionProvider"])
    print("  e2e 输入:", [(i.name, i.shape) for i in e2e.get_inputs()])
    print("  e2e 输出:", [(o.name, o.shape) for o in e2e.get_outputs()])

    def postprocess_ref(output, scale, pad_w, pad_h, conf, iou):
        preds = output[0].T
        boxes = preds[:, :4]; scores = preds[:, 4:]
        cls = scores.argmax(1); confs = scores.max(1)
        m = confs > conf
        boxes, confs, cls = boxes[m], confs[m], cls[m]
        if len(boxes) == 0:
            return []
        x, y, w, h = boxes.T
        x1 = (x - w/2 - pad_w) / scale; y1 = (y - h/2 - pad_h) / scale
        x2 = (x + w/2 - pad_w) / scale; y2 = (y + h/2 - pad_h) / scale
        # 注意: cv2.dnn.NMSBoxes 按 C++ Rect 语义解析输入框 = (x, y, w, h)，
        # 必须传 (x1, y1, x2-x1, y2-y1)；直接传 (x1,y1,x2,y2) 会把 x2/y2 当宽高，
        # IoU 全错、多框场景下 NMS 结果不对（单框时侥幸看不出）。
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(b), float(c - a), float(d - b)]
             for a, b, c, d in zip(x1, y1, x2, y2)],
            confs.tolist(), conf, iou)
        idx = [i[0] if isinstance(i, (list, tuple)) else i for i in idx]
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                 float(confs[i]), int(cls[i])) for i in idx]

    def canon(d):
        return sorted([(round(float(a), 2), round(float(b), 2), round(float(c), 2),
                        round(float(d), 2), round(float(e), 2), int(f))
                       for a, b, c, d, e, f in d])

    def same(cc, ce):
        return len(cc) == len(ce) and all(
            all(abs(x - y) <= 1.0 for x, y in zip(a[:4], b[:4])) and a[5] == b[5]
            for a, b in zip(cc, ce))

    img = cv2.imread('/root/my_FILE/datasets/Data_DeepPCB_YOLO/images/test/00041200.jpg')
    scale, pad_x, pad_y = letterbox_params(img.shape[0], img.shape[1])
    blob, *_ = preprocess_cpu(img)
    ref_dets = postprocess_ref(ref.run(None, {"images": blob})[0], scale, pad_x, pad_y, conf, iou)
    cc = canon(ref_dets)

    feed1 = {"image_raw": img[None],
             "iou_thresh": np.array([iou], np.float32),
             "score_thresh": np.array([conf], np.float32)}

    # 1) batch=1：与参考一致
    d1 = e2e.run(None, feed1)[0]
    c1 = canon(d1[0][d1[0][:, 4] > 0])
    print(f"  batch=1: 参考 {len(cc)} 框 vs e2e {len(c1)} 框 -> {'OK' if same(cc, c1) else 'DIFF'}")

    # 2) batch=2：两张不同图（同一张 + 翻转）结果应与各自单张结果一致（不串扰）
    img2 = cv2.flip(img, 1)
    feed2 = {"image_raw": np.ascontiguousarray(np.stack([img, img2])),
             "iou_thresh": feed1["iou_thresh"], "score_thresh": feed1["score_thresh"]}
    d2 = e2e.run(None, feed2)[0]
    print(f"  batch=2 输出 shape: {d2.shape}")
    c2a = canon(d2[0][d2[0][:, 4] > 0])
    c2b = canon(d2[1][d2[1][:, 4] > 0])
    ok_a = same(cc, c2a)
    # 图2 的参考：单独跑一次
    blob2, *_ = preprocess_cpu(img2)
    ref2 = postprocess_ref(ref.run(None, {"images": blob2})[0], scale, pad_x, pad_y, conf, iou)
    ok_b = same(canon(ref2), c2b)
    print(f"  batch=2 图0: {len(c2a)} 框 vs 参考 {len(cc)} -> {'OK' if ok_a else 'DIFF'}")
    print(f"  batch=2 图1: {len(c2b)} 框 vs 参考 {len(ref2)} -> {'OK' if ok_b else 'DIFF'}")
    if not (ok_a and ok_b):
        for a, b in zip(canon(ref2), c2b):
            print("     ref", a, " e2e", b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/my_FILE/models/yolov8_int8_dy_exclude.onnx")
    ap.add_argument("--dst", default="/root/my_FILE/models/yolov8_e2e_dy.onnx")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--no-check", dest="check", action="store_false")
    args = ap.parse_args()
    build_e2e_conf_dyn(args.src, args.dst, max_dets=args.max_det)
    if args.check:
        for conf in (0.45, 0.001):
            check_with_onnxruntime(args.dst, args.src, conf=conf, iou=0.65)
