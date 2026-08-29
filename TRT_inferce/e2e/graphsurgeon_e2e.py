"""
graphsurgeon_e2e.py — 把预处理 + 后处理(NMS) 全部烧进 ONNX，得到"端到端"模型。

输入 : image_raw  (1, H, W, 3) uint8 BGR NHWC，H/W 动态
输出 : detections (num_dets, 6) float32 = [x1,y1,x2,y2,conf,cls]，坐标已在原图尺寸

预处理(等价 server_trtapi.py 的 letterbox) 与 后处理(decode + NonMaxSuppression)
被改写为 ONNX 计算图，随后由 TRT 整体编译进一个 engine —— 运行时只需:
    H2D(原始 uint8) -> execute_v2 -> D2H(少量最终框)，全程无 CPU 前后处理。

用法:
    python graphsurgeon_e2e.py            # 生成 best_PCB_e2e.onnx
    python graphsurgeon_e2e.py --check    # 生成后与 CPU 参考链路做数值对比
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F32 = TensorProto.FLOAT
I64 = TensorProto.INT64


def build_e2e(src, dst, conf_thresh=0.45, nms_thresh=0.65, max_dets=300, num_classes=6):
    model = onnx.load(src)
    g = model.graph
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
    c_640_i64 = C("c_640_i64", [640], I64)
    c_640_f   = C("c_640_f", 640.0, F32)
    c_114_f   = C("c_114_f", 114.0, F32)
    c_inv255  = C("c_inv255", 1.0 / 255.0, F32)
    c_rgb_idx = C("c_rgb_idx", [2, 1, 0], I64)
    c_half    = C("c_half", 0.5, F32)
    c_0_f     = C("c_0_f", 0.0, F32)
    c_maxbox  = C("c_maxbox", max_dets, I64)
    c_iou     = C("c_iou", nms_thresh, F32)
    c_score   = C("c_score", conf_thresh, F32)
    c_idx2    = C("c_idx2", 2, I64)   # NMS 输出的第 2 列 = box index

    def SLICE(x, starts, ends, out):
        s = C(f"{out}_st", starts, I64)
        e = C(f"{out}_en", ends, I64)
        N("Slice", [x, s, e], out)
        return out

    # ================= 预处理 =================
    N("Cast", ["image_raw"], "img_f", to=F32)
    N("Shape", ["img_f"], "shape_img")
    N("Gather", ["shape_img", c_1_i64], "H", axis=0)
    N("Gather", ["shape_img", c_2_i64], "W", axis=0)
    N("Cast", ["H"], "Hf", to=F32)
    N("Cast", ["W"], "Wf", to=F32)
    N("Max", ["Hf", "Wf"], "maxhw")
    N("Div", [c_640_f, "maxhw"], "scale")            # scale = 640 / max(H,W)
    N("Mul", ["Hf", "scale"], "hsc")
    N("Mul", ["Wf", "scale"], "wsc")
    N("Floor", ["hsc"], "hfl")
    N("Floor", ["wsc"], "wfl")
    N("Cast", ["hfl"], "nh", to=I64)                 # nh = int(H*scale)
    N("Cast", ["wfl"], "nw", to=I64)

    N("Concat", [c_1_i64, "nh", "nw", c_3_i64], "sizes", axis=0)
    N("Resize", ["img_f", "", "", "sizes"], "resized",
      mode="linear", coordinate_transformation_mode="half_pixel")

    # 居中 pad 到 640x640（与 CPU 参考完全一致）
    N("Sub", [c_640_i64, "nh"], "padh_tot")
    N("Sub", [c_640_i64, "nw"], "padw_tot")
    N("Div", ["padh_tot", c_2_i64], "pad_top")
    N("Div", ["padw_tot", c_2_i64], "pad_left")
    N("Sub", ["padh_tot", "pad_top"], "pad_bot")
    N("Sub", ["padw_tot", "pad_left"], "pad_right")
    N("Concat", [c_0_i64, "pad_top", "pad_left", c_0_i64,
                 c_0_i64, "pad_bot", "pad_right", c_0_i64], "pads", axis=0)
    N("Pad", ["resized", "pads", c_114_f], "padded", mode="constant")

    N("Transpose", ["padded"], "tr", perm=[0, 3, 1, 2])   # NHWC -> NCHW
    N("Gather", ["tr", c_rgb_idx], "rgb", axis=1)          # BGR -> RGB
    N("Mul", ["rgb", c_inv255], "preprocessed")            # normalize [0,1]

    # ================= 后处理（切换节点列表） =================
    cur = post_nodes
    N("Transpose", [orig_out], "trans", perm=[0, 2, 1])    # (1,8400,10)
    SLICE("trans", [0, 0, 0], [1, 8400, 4], "boxes")
    SLICE("trans", [0, 0, 4], [1, 8400, 10], "scores")

    SLICE("boxes", [0, 0, 0], [1, 8400, 1], "cx")
    SLICE("boxes", [0, 0, 1], [1, 8400, 2], "cy")
    SLICE("boxes", [0, 0, 2], [1, 8400, 3], "bw")
    SLICE("boxes", [0, 0, 3], [1, 8400, 4], "bh")

    N("Mul", ["bw", c_half], "w2")
    N("Mul", ["bh", c_half], "h2")
    N("Sub", ["cx", "w2"], "x1")
    N("Add", ["cx", "w2"], "x2")
    N("Sub", ["cy", "h2"], "y1")
    N("Add", ["cy", "h2"], "y2")

    # 反 letterbox 到原图坐标（复用预处理的 scale/pad）
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
    N("Concat", ["x1o", "y1o", "x2o", "y2o"], "boxes_final", axis=2)

    N("ReduceMax", ["scores"], "conf", axes=[2], keepdims=1)
    N("ArgMax", ["scores"], "cls", axis=2, keepdims=1)
    N("Cast", ["cls"], "cls_f", to=F32)

    N("Transpose", ["conf"], "conf_t", perm=[0, 2, 1])      # (1,1,8400)
    N("NonMaxSuppression", ["boxes_final", "conf_t", c_maxbox, c_iou, c_score],
      "selected", center_point_box=0)

    N("Gather", ["selected", c_idx2], "box_idx", axis=1)    # 第 2 列 = box index
    N("Gather", ["boxes_final", "box_idx"], "boxes_sel", axis=1)
    N("Gather", ["conf", "box_idx"], "conf_sel", axis=1)
    N("Gather", ["cls_f", "box_idx"], "cls_sel", axis=1)
    N("Concat", ["boxes_sel", "conf_sel", "cls_sel"], "det", axis=2)

    # 固定输出尺寸 [max_dets, 6]：把可变 num_sel 用 0 填充到 max_dets（便于 TRT 输出 shape 确定）
    N("Shape", ["selected"], "shape_sel")
    N("Gather", ["shape_sel", c_0_i64], "num_sel", axis=0)
    N("Sub", [c_maxbox, "num_sel"], "pad_det")
    N("Concat", [c_0_i64, c_0_i64, c_0_i64, c_0_i64, "pad_det", c_0_i64], "det_pads", axis=0)
    N("Pad", ["det", "det_pads", c_0_f], "det_padded", mode="constant")
    N("Squeeze", ["det_padded", c_0_i64], "detections")   # opset13+: axes 作为输入

    # ================= 重写 graph I/O =================
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == orig_in:
                node.input[i] = "preprocessed"

    del g.input[:]
    g.input.append(helper.make_tensor_value_info(
        "image_raw", TensorProto.UINT8, [1, "height", "width", 3]))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("detections", F32, [max_dets, 6]))

    all_nodes = pre_nodes + list(g.node) + post_nodes
    del g.node[:]
    g.node.extend(all_nodes)
    g.initializer.extend(inits)

    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"[e2e] saved {dst}")
    return model


def check_with_onnxruntime(onnx_path, src_onnx):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import onnxruntime as ort
    from common import preprocess_cpu, postprocess_cpu, letterbox_params

    e2e = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = ort.InferenceSession(src_onnx, providers=["CPUExecutionProvider"])
    print("  e2e input :", [(i.name, i.shape) for i in e2e.get_inputs()])
    print("  e2e output:", [(o.name, o.shape) for o in e2e.get_outputs()])

    rng = np.random.default_rng(0)
    for h, w in [(1234, 987), (640, 640), (1920, 1080)]:
        img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        scale, pad_x, pad_y = letterbox_params(h, w)

        # CPU 参考链路：预处理 -> 原模型 -> 后处理
        blob, *_ = preprocess_cpu(img)
        out0 = ref.run(None, {"images": blob})[0]
        cpu_dets = postprocess_cpu(out0, scale, pad_x, pad_y)

        # e2e 链路：原图直出（固定 [max_dets,6]，空槽 conf=0）
        dets = e2e.run(None, {"image_raw": img[None, ...]})[0]
        dets = dets[dets[:, 4] > 0]  # 丢弃 padding 行

        def canon(d):
            return sorted([(round(float(a), 2), round(float(b), 2), round(float(c), 2),
                            round(float(d_), 2), round(float(e), 2), int(f))
                           for a, b, c, d_, e, f in d])

        cc, ce = canon(cpu_dets), canon(dets)
        same = len(cc) == len(ce) and all(
            all(abs(x - y) <= 1.0 for x, y in zip(a[:4], b[:4]))
            and a[5] == b[5] for a, b in zip(cc, ce))
        print(f"  [{h}x{w}] cpu={len(cc)} e2e={len(ce)} -> {'OK' if same else 'DIFF'}")
        if not same:
            for a, b in zip(cc, ce):
                print("     cpu", a, " e2e", b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/my_FILE/models/best_PCB.onnx")
    ap.add_argument("--dst", default="/root/my_FILE/models/best_PCB_e2e.onnx")
    # 推理服务用 conf=0.45；COCO mAP 评测必须用 conf=0.001（否则低分真值被过滤，mAP 偏低）
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    build_e2e(args.src, args.dst, conf_thresh=args.conf,
              nms_thresh=args.iou, max_dets=args.max_det)
    if args.check:
        check_with_onnxruntime(args.dst, args.src)
