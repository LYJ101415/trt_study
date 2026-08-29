"""
graphsurgeon_e2e_conf.py — 端到端 engine，但 conf / iou 阈值是「运行时输入」。

与 e2e/graphsurgeon_e2e.py 的唯一区别：
  原来 NonMaxSuppression 的 iou_threshold / score_threshold 是常量（烧死在图里，
  改阈值要重编 engine）；这里把它们改成图输入，运行时每次 execute 传入，
  因此一个 engine 既能跑推理(conf=0.45) 也能跑 mAP(conf=0.001)，无需重编。

输入 :
  image_raw     (1, H, W, 3) uint8 BGR NHWC，H/W 动态
  iou_thresh    (1,) float32   NMS 的 IoU 阈值（运行时传）
  score_thresh  (1,) float32   置信度阈值（运行时传）
输出 :
  detections    (max_det, 6) float32 = [x1,y1,x2,y2,conf,cls]，原图坐标、已 NMS

用法:
    python graphsurgeon_e2e_conf.py --src best_PCB.onnx --dst best_PCB_e2e_conf.onnx
"""

import argparse
import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

F32 = TensorProto.FLOAT # ONNX 数据类型别名，简化后续书写
I64 = TensorProto.INT64


def build_e2e_conf(src, dst, max_dets=300, num_classes=6):

    model = onnx.load(src)
    g = model.graph
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

    # # ---- 常量 ------------------------------------
    # 常用整数索引（以下四个）
    c_1_i64   = C("c_1_i64", [1], I64)
    c_2_i64   = C("c_2_i64", [2], I64)
    c_3_i64   = C("c_3_i64", [3], I64)
    c_0_i64   = C("c_0_i64", [0], I64)

    # 目标尺寸 640（int/float 各一份（以下2个）
    c_640_i64 = C("c_640_i64", [640], I64)
    c_640_f   = C("c_640_f", 640.0, F32)

    c_114_f   = C("c_114_f", 114.0, F32) # Letterbox 灰色填充值（114）

    c_inv255  = C("c_inv255", 1.0 / 255.0, F32) # 归一化系数 1/255

    c_rgb_idx = C("c_rgb_idx", [2, 1, 0], I64) # BGR→RGB 通道重排索引

    c_half    = C("c_half", 0.5, F32) # xywh→xyxy 转换用于 NMS

    c_0_f     = C("c_0_f", 0.0, F32)
    c_maxbox  = C("c_maxbox", max_dets, I64) # NMS 最大保留框数

    c_idx2    = C("c_idx2", 2, I64)   # NMS 输出的第 2 列 = box index

    def SLICE(x, starts, ends, out):
        s = C(f"{out}_st", starts, I64)
        e = C(f"{out}_en", ends, I64)
        N("Slice", [x, s, e], out)
        return out

    # ================= 预处理（与原版完全一致） =================

    N("Cast", ["image_raw"], "img_f", to=F32) # UINT8 → FLOAT32
    N("Shape", ["img_f"], "shape_img")        # 获取 [1,H,W,3]
    N("Gather", ["shape_img", c_1_i64], "H", axis=0) # H
    N("Gather", ["shape_img", c_2_i64], "W", axis=0) # W
    N("Cast", ["H"], "Hf", to=F32)            # H → float
    N("Cast", ["W"], "Wf", to=F32)            # W → float
    N("Max", ["Hf", "Wf"], "maxhw")           # max(H,W)
    N("Div", [c_640_f, "maxhw"], "scale")     # scale = 640/max(H,W)
    N("Mul", ["Hf", "scale"], "hsc") # H * scale
    N("Mul", ["Wf", "scale"], "wsc") # W * scale
    N("Floor", ["hsc"], "hfl")       # 向下取整（保持整数像素）
    N("Floor", ["wsc"], "wfl")
    N("Cast", ["hfl"], "nh", to=I64) # 转回 int64（Resize 需要）
    N("Cast", ["wfl"], "nw", to=I64)

    N("Concat", [c_1_i64, "nh", "nw", c_3_i64], "sizes", axis=0) # [1,nh,nw,3]
    N("Resize", ["img_f", "", "", "sizes"], "resized",  
      mode="linear", coordinate_transformation_mode="half_pixel")
# ------------------------------------------------------------------------------------
    # 居中 Padding
    N("Sub", [c_640_i64, "nh"], "padh_tot")          # 总高度填充量
    N("Sub", [c_640_i64, "nw"], "padw_tot")          # 总宽度填充量
    N("Div", ["padh_tot", c_2_i64], "pad_top")       # 上填充 = total//2
    N("Div", ["padw_tot", c_2_i64], "pad_left")      # 左填充 = total//2
    N("Sub", ["padh_tot", "pad_top"], "pad_bot")     # 下填充 = total - top
    N("Sub", ["padw_tot", "pad_left"], "pad_right")  # 右填充 = total - left

    N("Concat", [c_0_i64, "pad_top", "pad_left", c_0_i64,
                 c_0_i64, "pad_bot", "pad_right", c_0_i64], "pads", axis=0)
    N("Pad", ["resized", "pads", c_114_f], "padded", mode="constant")

    # 通道变换与归一化
    N("Transpose", ["padded"], "tr", perm=[0, 3, 1, 2]) # NHWC → NCHW
    N("Gather", ["tr", c_rgb_idx], "rgb", axis=1)       # BGR → RGB（axis=1 是 C 维度）
    N("Mul", ["rgb", c_inv255], "preprocessed")         # /255.0 归一化

    # ================= 后处理（与原版一致，仅 NMS 阈值改成输入） =================
    # .1 解析原始输出
    cur = post_nodes # ★ 切换节点列表到后处理
    N("Transpose", [orig_out], "trans", perm=[0, 2, 1]) # (1,84,8400) → (1,8400,84)
    SLICE("trans", [0, 0, 0], [1, 8400, 4], "boxes")   # 前4列 = cx,cy,w,h
    SLICE("trans", [0, 0, 4], [1, 8400, 10], "scores") # 后6列 = 类别分数

    # .2 xywh → xyxy 并还原到原图坐标
    SLICE("boxes", [0, 0, 0], [1, 8400, 1], "cx") 
    SLICE("boxes", [0, 0, 1], [1, 8400, 2], "cy")
    SLICE("boxes", [0, 0, 2], [1, 8400, 3], "bw")
    SLICE("boxes", [0, 0, 3], [1, 8400, 4], "bh")

    # xywh → xyxy
    N("Mul", ["bw", c_half], "w2")
    N("Mul", ["bh", c_half], "h2")
    N("Sub", ["cx", "w2"], "x1")
    N("Add", ["cx", "w2"], "x2")
    N("Sub", ["cy", "h2"], "y1")
    N("Add", ["cy", "h2"], "y2")

    N("Cast", ["pad_left"], "padl_f", to=F32) # pad 值转 float
    N("Cast", ["pad_top"], "padt_f", to=F32)
    N("Sub", ["x1", "padl_f"], "sub_x1") # 减 padding
    N("Sub", ["y1", "padt_f"], "sub_y1")
    N("Sub", ["x2", "padl_f"], "sub_x2")
    N("Sub", ["y2", "padt_f"], "sub_y2")
    N("Div", ["sub_x1", "scale"], "x1o") # 除 scale
    N("Div", ["sub_y1", "scale"], "y1o")
    N("Div", ["sub_x2", "scale"], "x2o")
    N("Div", ["sub_y2", "scale"], "y2o")
    N("Concat", ["x1o", "y1o", "x2o", "y2o"], "boxes_final", axis=2)

    if opset >= 18:
        c_axes2 = C("c_axes2", [2], I64)
        N("ReduceMax", ["scores", c_axes2], "conf", keepdims=1)
    else:
        N("ReduceMax", ["scores"], "conf", axes=[2], keepdims=1)
    N("ArgMax", ["scores"], "cls", axis=2, keepdims=1)
    N("Cast", ["cls"], "cls_f", to=F32)

    N("Transpose", ["conf"], "conf_t", perm=[0, 2, 1])
    N("NonMaxSuppression", ["boxes_final", "conf_t", c_maxbox, "iou_thresh", "score_thresh"],
      "selected", center_point_box=0)

    N("Gather", ["selected", c_idx2], "box_idx", axis=1)  # 取 NMS 输出的 box 索引列
    N("Gather", ["boxes_final", "box_idx"], "boxes_sel", axis=1) # 选中的 box
    N("Gather", ["conf", "box_idx"], "conf_sel", axis=1)         # 选中的 conf
    N("Gather", ["cls_f", "box_idx"], "cls_sel", axis=1)         # 选中的 cls
    N("Concat", ["boxes_sel", "conf_sel", "cls_sel"], "det", axis=2) # (1,N,6)

    N("Shape", ["selected"], "shape_sel")
    N("Gather", ["shape_sel", c_0_i64], "num_sel", axis=0)
    N("Sub", [c_maxbox, "num_sel"], "pad_det") # 需补零的行数
    N("Concat", [c_0_i64, c_0_i64, c_0_i64, c_0_i64, "pad_det", c_0_i64], "det_pads", axis=0) # 只在第1维（N维度）末尾补零
    N("Pad", ["det", "det_pads", c_0_f], "det_padded", mode="constant")
    N("Squeeze", ["det_padded", c_0_i64], "detections") # 去掉 batch 维 → (max_dets, 6)

    # ================= 重写 graph I/O =================
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == orig_in:
                node.input[i] = "preprocessed"

    del g.input[:] # 这不是删除变量，而是清空列表内容。
                    # 因为 g.input 是 protobuf 的 RepeatedCompositeFieldContainer（一种特殊的容器类型），不是普通 Python list：
    g.input.append(helper.make_tensor_value_info(
        "image_raw", TensorProto.UINT8, [1, "height", "width", 3]))
    
    g.input.append(helper.make_tensor_value_info("iou_thresh", F32, [1]))
    g.input.append(helper.make_tensor_value_info("score_thresh", F32, [1]))

    del g.output[:]
    g.output.append(helper.make_tensor_value_info("detections", F32, [max_dets, 6]))

    # 拼接三部分节点：预处理 + 原始模型 + 后处理
    all_nodes = pre_nodes + list(g.node) + post_nodes 
    del g.node[:] 
    g.node.extend(all_nodes) # 写入合并后的节点
    g.initializer.extend(inits) 

    onnx.checker.check_model(model) # 合法性校验
    onnx.save(model, dst)
    print(f"[e2e-conf] saved {dst}")
    return model

def check_with_onnxruntime(onnx_path, src_onnx, conf=0.45, iou=0.65):
    """用 onnxruntime 验证：e2e(运行时阈值) 与原模型+CPU后处理(同阈值) 输出一致。"""
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

    # 参数化的参考后处理（与 common.postprocess_cpu 一致，但 conf/iou 可传）
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
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x1, y1, x2, y2)],
            confs.tolist(), conf, iou)
        idx = [i[0] if isinstance(i, (list, tuple)) else i for i in idx]
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                 float(confs[i]), int(cls[i])) for i in idx]

    def canon(d):
        return sorted([(round(float(a), 2), round(float(b), 2), round(float(c), 2),
                        round(float(d), 2), round(float(e), 2), int(f))
                       for a, b, c, d, e, f in d])

    img = cv2.imread('/root/my_FILE/my_FILE/test_images/00041200.jpg')
    scale, pad_x, pad_y = letterbox_params(img.shape[0], img.shape[1])

    # 1. 用原始模型 + CPU 后处理得到参考结果
    blob, *_ = preprocess_cpu(img)
    out0 = ref.run(None, {"images": blob})[0]
    cpu_dets = postprocess_ref(out0, scale, pad_x, pad_y, conf, iou)

    # 2. 用 e2e 模型直接推理
    dets = e2e.run(None, {
        "image_raw": img[None], # 原图直接送入
        "iou_thresh": np.array([iou], dtype=np.float32), # 运行时阈值
        "score_thresh": np.array([conf], dtype=np.float32),
    })[0]
    dets = dets[dets[:, 4] > 0]  # 过滤填充的零行

    cc, ce = canon(cpu_dets), canon(dets)
    # 3. 排序后逐框比较（容忍 1.0 像素误差）
    same = len(cc) == len(ce) and all(
        all(abs(x - y) <= 1.0 for x, y in zip(a[:4], b[:4])) and a[5] == b[5]
        for a, b in zip(cc, ce))
    print(f"  conf={conf} iou={iou}: 参考 {len(cc)} 框 vs e2e {len(ce)} 框 -> "
          f"{'OK' if same else 'DIFF'}")
    if not same:
        for a, b in zip(cc, ce):
            print("     ref", a, " e2e", b)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/my_FILE/models/best_PCB.onnx")
    ap.add_argument("--dst", default="/root/my_FILE/models/best_PCB_e2e_conf.onnx")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    build_e2e_conf(args.src, args.dst, max_dets=args.max_det)
    if args.check:
        for conf in (0.45, 0.001): # 测试两组阈值
            check_with_onnxruntime(args.dst, args.src, conf=conf, iou=0.65)
