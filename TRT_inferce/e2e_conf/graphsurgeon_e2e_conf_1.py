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

    """ONNX 模型的层级结构是：
    ModelProto
    ├── opset_import          # 算子版本
    ├── ir_version            # IR 版本
    └── graph: GraphProto     ← g 就是这个
            ├── input[]         # 图输入声明
            ├── output[]        # 图输出声明
            ├── node[]          # 所有算子节点
            ├── initializer[]   # 权重/常量张量
            └── value_info[]    # 中间张量的类型/形状信息
    model.graph 返回的是 GraphProto 对象的引用（不是拷贝），所以后续对 g 的所有修改都会直接反映到 model 上。
    """
    g = model.graph
    opset = model.opset_import[0].version   # 决定 ReduceMax 的 axes 是属性(<18)还是输入(>=18)
    orig_in = g.input[0].name
    orig_out = g.output[0].name

    pre_nodes = []
    post_nodes = []
    cur = pre_nodes
    inits = []

    # 创建常量初始化器（Initializer）。将 Python 数值转为 ONNX 权重张量并加入 inits 列表，返回其名称供节点引用。
    # 这是构建 ONNX 子图的基础设施——ONNX 中没有字面量，所有常量都必须是图中的 initializer(初始化器）。
    def C(name, value, dtype=F32):
        # Step 1: 将 Python 值转为 numpy 数组，dtype 由参数决定
        arr = np.asarray(value, dtype=np.float32) if dtype == F32 else np.asarray(value, dtype=np.int64)
        # Step 2: 将 numpy 数组转为 ONNX TensorProto，并赋予名称,numpy_helper.from_array 会自动推断 shape,
        # 加入外部 inits 列表（稍后一次性写入 g.initializer）
        inits.append(numpy_helper.from_array(arr, name=name))
        # Step 4: 返回字符串名称，供 N() 创建节点时作为输入引用
        return name
    """为什么需要它？ONNX 中没有"字面量"概念。 你不能写 N("Mul", ["x", 0.5], "y")，因为 ONNX 节点的所有输入都必须是字符串名称，
    指向某个 tensor。即使是常数 0.5，也必须先作为一个 named initializer 存在于图中，然后才能被节点引用。C() 就是把这个繁琐过程封装成一行调用。
    详细见my_readme.md.
    """

    # 创建算子节点。cur 是一个可变引用，指向当前正在构建的节点列表（先指 pre_nodes，后切换到 post_nodes）。
    # 自动将单个字符串输出包装为列表，简化调用语法。详细应用见my_readme.md.
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
    # 注意：没有 c_iou / c_score —— 这两个改成运行时图输入

    # SLICE() 函数是一个语法糖封装函数，它把 ONNX Slice 算子所需的"创建常量 + 创建节点"两步操作合并成了一行调用。
    # 详解见my_readme.md.
    def SLICE(x, starts, ends, out):
        s = C(f"{out}_st", starts, I64)
        e = C(f"{out}_en", ends, I64)
        N("Slice", [x, s, e], out)
        return out

    # ================= 预处理（与原版完全一致） =================
    # 二、预处理子图（Pre-processing）：这部分完全复刻了 common.py 中 preprocess_cpu + letterbox_params 的逻辑，
    # 但全部用 ONNX 算子表达：
    # 以下是对应 Python: scale = min(640/h, 640/w) 的等价形式 640/max(h,w)。（注意：原版代码中 scale 是 float，但 ONNX 中必须为 int）
    N("Cast", ["image_raw"], "img_f", to=F32) # UINT8 → FLOAT32
    N("Shape", ["img_f"], "shape_img")        # 获取 [1,H,W,3]
    N("Gather", ["shape_img", c_1_i64], "H", axis=0) # H
    N("Gather", ["shape_img", c_2_i64], "W", axis=0) # W
    N("Cast", ["H"], "Hf", to=F32)            # H → float
    N("Cast", ["W"], "Wf", to=F32)            # W → float
    N("Max", ["Hf", "Wf"], "maxhw")           # max(H,W)
    N("Div", [c_640_f, "maxhw"], "scale")     # scale = 640/max(H,W)
#------------------------------------------------------------------------
    # 计算缩放后尺寸并 Resize（注意：原版代码中 scale 是 float，但 ONNX 中必须为 int）
    N("Mul", ["Hf", "scale"], "hsc") # H * scale
    N("Mul", ["Wf", "scale"], "wsc") # W * scale
    N("Floor", ["hsc"], "hfl")       # 向下取整（保持整数像素）
    N("Floor", ["wsc"], "wfl")
    N("Cast", ["hfl"], "nh", to=I64) # 转回 int64（Resize 需要）
    N("Cast", ["wfl"], "nw", to=I64)

    N("Concat", [c_1_i64, "nh", "nw", c_3_i64], "sizes", axis=0) # [1,nh,nw,3]
    # "" 表示可选输入 roi/scales 留空，直接用 sizes 指定目标尺寸。half_pixel 模式与 OpenCV resize 行为一致。
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

    # ONNX Pad 格式为 [before_batch, before_H, before_W, before_C, after_batch, after_H, after_W, after_C]。
    # 填充值 114.0 是 YOLO 标准灰色。
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
    # 拆分 cx,cy,w,h
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

    # 减去 padding 再除以 scale → 原图坐标，这一步将模型输出的 640×640 空间中的 box 精确映射回原始图片坐标系。
    # 注意这里复用了预处理阶段计算的 scale、pad_left、pad_top，体现了端到端图的优势：中间变量天然共享。
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

    # .3 提取置信度与类别
    # opset 兼容处理
    if opset >= 18:
        # opset 18+：ReduceMax 的 axes 是输入
        c_axes2 = C("c_axes2", [2], I64)
        N("ReduceMax", ["scores", c_axes2], "conf", keepdims=1)
    else:
        # opset <=17：axes 是属性
        N("ReduceMax", ["scores"], "conf", axes=[2], keepdims=1)
    N("ArgMax", ["scores"], "cls", axis=2, keepdims=1)
    N("Cast", ["cls"], "cls_f", to=F32)

    # .4 NMS（核心亮点）：iou_thresh 和 score_thresh 是图输入而非 initializer。这使得：
    # 同一个 TRT 引擎支持运行时调参；验证时可以测试多组阈值而无需重建模型
    # center_point_box=0 表示 boxes 是 (y1,x1,y2,x2) 格式（ONNX NMS 约定）
    N("Transpose", ["conf"], "conf_t", perm=[0, 2, 1])
    # ★ iou_threshold / score_threshold 用运行时输入（图输入名）
    N("NonMaxSuppression", ["boxes_final", "conf_t", c_maxbox, "iou_thresh", "score_thresh"],
      "selected", center_point_box=0)

    # .5 收集选中结果并填充到固定形状
    N("Gather", ["selected", c_idx2], "box_idx", axis=1)  # 取 NMS 输出的 box 索引列
    N("Gather", ["boxes_final", "box_idx"], "boxes_sel", axis=1) # 选中的 box
    N("Gather", ["conf", "box_idx"], "conf_sel", axis=1)         # 选中的 conf
    N("Gather", ["cls_f", "box_idx"], "cls_sel", axis=1)         # 选中的 cls
    N("Concat", ["boxes_sel", "conf_sel", "cls_sel"], "det", axis=2) # (1,N,6)

    # # 填充到固定 max_dets 行（TRT 要求输出 shape 固定）
    N("Shape", ["selected"], "shape_sel")
    N("Gather", ["shape_sel", c_0_i64], "num_sel", axis=0)
    N("Sub", [c_maxbox, "num_sel"], "pad_det") # 需补零的行数
    N("Concat", [c_0_i64, c_0_i64, c_0_i64, c_0_i64, "pad_det", c_0_i64], "det_pads", axis=0) # 只在第1维（N维度）末尾补零
    N("Pad", ["det", "det_pads", c_0_f], "det_padded", mode="constant")
    N("Squeeze", ["det_padded", c_0_i64], "detections") # 去掉 batch 维 → (max_dets, 6)
    """为什么必须 Pad？ TensorRT 引擎的输出 buffer 大小在构建时确定。如果实际检测到 5 个框但输出声明为 (300,6)，
    不补零会导致未初始化内存被读取。补零后，消费端只需过滤 conf > 0 的行即可。
    """

    # ================= 重写 graph I/O =================
    # 将原模型第一个节点的输入从 "images" 改为 "preprocessed":
    # 这是"缝合"的关键：让原始 YOLO 主干网络的输入接上预处理子图的输出。
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == orig_in:
                node.input[i] = "preprocessed"

    # 以下这段代码的作用是彻底替换 ONNX 图的输入声明，将原始模型的单一 float32 图片输入改为三个新输入：原始 UINT8 图片 + 两个运行时阈值。
    del g.input[:] # 这不是删除变量，而是清空列表内容。
                    # 因为 g.input 是 protobuf 的 RepeatedCompositeFieldContainer（一种特殊的容器类型），不是普通 Python list：
    g.input.append(helper.make_tensor_value_info(
        "image_raw", TensorProto.UINT8, [1, "height", "width", 3]))
    
    # 添加运行时阈值输入:"iou_thresh","score_thresh",Shape 为 [1] 而非标量 []，是因为 ONNX NonMaxSuppression 算子规范要求这两个输入必须是 1-D tensor。
    g.input.append(helper.make_tensor_value_info("iou_thresh", F32, [1]))
    g.input.append(helper.make_tensor_value_info("score_thresh", F32, [1]))
    """原始 YOLOv8 模型的输入通常是：input[0]: name="images", type=FLOAT, shape=[1,3,640,640],将其移除，因为端到端模型不再接受预处理后的 float32 tensor。
    第2行：添加原始图片输入:g.input.append(helper.make_tensor_value_info("image_raw", TensorProto.UINT8, [1, "height", "width", 3])),
    make_tensor_value_info 创建一个 ValueInfoProto（图输入的声明），三个参数分别是：
    参数	      值	                  含义
    name	    "image_raw"	        张量名称，与预处理子图中 Cast(["image_raw"], ...) 对应
    elem_type	TensorProto.UINT8	数据类型：无符号8位整数（原始像素值 0-255）
    shape	   [1, "height", "width", 3]	  NHWC 格式，H/W 用字符串表示动态维度

    "height" 和 "width" 是符号化维度名，告诉 ONNX/TRT："这两个维度的大小在运行时才确定"。这使得同一个引擎可以处理任意分辨率的图片。
    """

    # 新输出：固定形状的检测结果,[max_dets, 6] 是完全静态的形状（没有符号维度），因为 TRT 要求输出 buffer 大小在构建时确定。
    # 每行 6 个值对应 [x1, y1, x2, y2, conf, cls]，不足 max_dets 的行用零填充。
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("detections", F32, [max_dets, 6]))

    # 拼接三部分节点：预处理 + 原始模型 + 后处理
    """pre_nodes-->>预处理子图（Cast→Div255→Gather通道重排→Resize)-->>cur=pre_nodes 阶段由 N() 积累
    list(g.node)-->原始 YOLOv8 模型的全部算子-->ONNX 加载时已存在于图中
    post_nodes-->>后处理子图（Slice→Sigmoid→NMS→Gather→Concat）-->>cur=post_nodes 阶段由 N() 积累
    """
    all_nodes = pre_nodes + list(g.node) + post_nodes # 三段拼接:为什么用 list(g.node) 而不是直接 g.node？
        # g.node 是 protobuf RepeatedCompositeFieldContainer，与 Python list 的 + 运算不兼容。list() 将其转为普通列表后才能拼接。
    del g.node[:] # 清空原始节点,必须先清空再 extend，否则原始节点会出现两次（一次在原始位置，一次在 all_nodes 中间）
    g.node.extend(all_nodes) # 写入合并后的节点
    g.initializer.extend(inits) # 写入新常量,将 C() 函数积累的所有常量张量注入图的 initializer 列表。
    # 注意是 extend 不是替换：原始模型的权重（conv weight、bn bias 等）仍然保留在 g.initializer 中，这里只是追加新增的常量。

    onnx.checker.check_model(model) # 合法性校验
    onnx.save(model, dst)
    print(f"[e2e-conf] saved {dst}")
    return model

# 验证函数 check_with_onnxruntime:这个函数的目的是双重验证：确保端到端模型的输出与"原始模型 + CPU 后处理"在相同阈值下完全一致。
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
