"""
脚本总览：本脚本读取一个静态 batch（通常为 batch=1）的 YOLOv8 ONNX 模型，
在其前后分别拼接预处理（图像缩放+居中Padding+归一化）和后处理（NMS+结果收集）ONNX 计算图节点，
同时将 batch 维度改为动态（batch 占位符），最终输出一个”输入原始图像 → 输出检测框”的端到端 ONNX 模型。

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
from onnx import helper, TensorProto, numpy_helper # - helper：ONNX 辅助函数，提供 make_node、make_tensor_value_info 等便捷构造器。 
                # - TensorProto：ONNX 张量数据类型枚举（如 FLOAT、INT64）。 - numpy_helper：在 numpy 数组和 ONNX TensorProto 之间互转。

F32 = TensorProto.FLOAT # 类型别名，F32 代表 32 位浮点类型（ONNX 中用于图像数据、坐标、分数等）
I64 = TensorProto.INT64 # 类型别名，I64 代表 64 位整数类型（ONNX 中用于索引、形状、切片起止点等）。
IMAX = int(np.iinfo(np.int64).max)   # Slice ends 用: 表示"该维到末尾"（动态 batch 维）,作用： - np.iinfo(np.int64).max 获取 int64 的最大值（即 9223372036854775807）。 
     # - 在 ONNX 的 Slice 操作中，ends 参数使用一个极大的正整数来等价于 Python 切片中的 :（即”取到该维末尾”）。因为 ONNX Slice 不支持省略号语法，只能用具体数值代替”到末尾”。


def build_e2e_conf_dyn(src, dst, max_dets=300, num_classes=6):

    model = onnx.load(src) # 从文件系统加载源 ONNX 模型到内存。加载后 model 对象包含图结构（graph）、初始化器（initializer）、节点列表（node）等。
    g = model.graph # 获取模型的计算图对象 g，后续所有图操作（添加节点、修改输入输出等）都通过 g 进行。

    """
    背景：YOLOv8 导出 ONNX 时，某些 Reshape 节点的第二个输入（目标形状）被固化为静态值（如 [1, 64, -1]），其中首位 1 代表 batch=1。
    要支持动态 batch，需要将这些形状中的 batch 维度 1 替换为真正的 batch 张量（由后续添加的 Gather(Shape(img), 0) 产出）。背景：YOLOv8 导出 ONNX 时，
    某些 Reshape 节点的第二个输入（目标形状）被固化为静态值（如 [1, 64, -1]），其中首位 1 代表 batch=1。要支持动态 batch，
    需要将这些形状中的 batch 维度 1 替换为真正的 batch 张量（由后续添加的 Gather(Shape(img), 0) 产出）。
    修补策略。只修改那些：(1) 操作类型为 Reshape；(2) 输入数据的 rank（张量维度数）>= 2；(3) 目标形状数组的首元素为 1 的 Reshape 节点。
    rank==1 的 Reshape（如 meshgrid 产生的坐标向量形状变换）不能改，因为它们的首位 1 不是 batch 维度。
    # ★0' 源模型修补（v2，带 rank 守卫）: 只修"数据输入 rank>=2 且目标形状首位为 1"
    #    的 Reshape（首位 1 = batch，如 [1,64,-1]）；rank==1 的（meshgrid 的
    #    (w,)->(1,w) 行向量，[1,-1]）绝不能改。
    #    "batch" 张量由本脚本加的预处理节点（Gather(Shape(img_f),0)）产出。
    """
    # 以下是构建一个以张量名为键、ValueInfoProto 为值的字典 vi_map。value_info 包含中间张量的 shape 声明，input/output 包含输入/输出张量声明。
    # 此映射用于后续查询任意张量的 rank（维度数）。
    
    vi_map = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}

    def declared_rank(t): # 定义辅助函数 declared_rank，根据张量名从 vi_map 中查找其 shape 信息，返回维度数（rank）。如果找不到或类型信息不完整，返回 None。
        v = vi_map.get(t)
        if v is None:
            return None
        try:
            return len(v.type.tensor_type.shape.dim)
        except Exception:
            return None

    init_by_name = {i.name: i for i in g.initializer} # 建初始化器字典，键为初始化器名称，值为 TensorProto 对象。用于快速查找 Reshape 节点的第二个输入（目标形状常量）。
    inserts = [] # (插入位置, 节点)；初始化一个列表，用于收集需要在图中插入的 Concat 节点及其插入位置。稍后统一插入，避免遍历过程中修改列表导致索引错乱。
    for idx, n in enumerate(g.node): # 遍历计算图中所有节点，idx 为节点索引，n 为节点对象。
        # 过滤条件——只处理满足以下三个条件的节点： 1. 操作类型是 Reshape； 2. 该节点至少有 2 个输入（第一个是待 reshape 的张量，第二个是目标形状）； 
        # 3. 第二个输入（目标形状）能在 init_by_name 中查到（说明它是一个常量初始化器，而非运行时张量）。 不满足则跳过。
        if not (n.op_type == "Reshape" and len(n.input) >= 2 and n.input[1] in init_by_name):
            continue
        arr = numpy_helper.to_array(init_by_name[n.input[1]]) # 将目标形状初始化器转换为 numpy 数组，以便检查其内容和首元素值。
        # 过滤条件——目标形状数组必须满足： 1. 数据类型为 int64； 2. 非空（arr.size 不为 0）； 3. 首元素为 1（表明可能是 batch 维度）。 不满足则跳过。
        if not (arr.dtype == np.int64 and arr.size and arr.flat[0] == 1):
            continue
        rank = declared_rank(n.input[0]) # 查询 Reshape 第一个输入（待 reshape 的张量）的声明 rank。
        if rank is None: # 如果 rank 信息不可用（模型导出时未保留 shape 信息），跳过该节点并打印警告。
            print(f"[e2e-dyn-v2] skip {n.name}: 数据输入 {n.input[0]} 无 shape 声明，"
                  f"首位 1 是否为 batch 无法判定，不改（shape={arr.tolist()}）")
            continue
        # 如果 rank < 2（即一维向量），说明这个 Reshape 不是用于 batch 维度的形状变换（如 meshgrid 产生的坐标向量 (w,) -> (1, w)），首位的 1 不是 batch，不能改。跳过。
        if rank < 2:
            print(f"[e2e-dyn-v2] skip {n.name}: 输入 rank={rank}（坐标向量，首位 1 非 batch，"
                  f"如 make_anchors/meshgrid），不改（shape={arr.tolist()}）")
            continue
        # 1. 生成一个新名字 rest_name（原名字 + _rest 后缀），用于存放目标形状中除首元素外的部分（如 [1, 64, -1] 的 [64, -1]）。 
        # 2. 创建一个新的初始化器，内容为 arr[1:]（去掉首元素 1），追加到图的初始化器列表中。
        rest_name = n.input[1] + "_rest" 
        g.initializer.append(numpy_helper.from_array(arr[1:].copy(), name=rest_name))
        #  1. 生成动态形状名字 cname（原名字 + _dyn）。 2. 创建一个 Concat 节点，将 "batch" 张量（后续由预处理部分的 Gather 产出）和 rest_name（如 [64, -1]）沿 axis=0 拼接，
        # 输出为 cname（如 [batch, 64, -1]）。 3. 记录插入位置 idx 和这个新节点到 inserts 列表中。
        cname = n.input[1] + "_dyn"
        inserts.append((idx, helper.make_node(
            "Concat", ["batch", rest_name], [cname], axis=0)))
        # 1. 将原 Reshape 节点的第二个输入从静态形状常量替换为动态形状名 cname。这样 Reshape 在运行时会根据实际 batch 大小动态计算目标形状。 2. 打印补丁信息日志。
        n.input[1] = cname
        print(f"[e2e-dyn-v2] patched {n.name}: {arr.tolist()} -> Concat([batch, {arr[1:].tolist()}])")
    # 倒序遍历 inserts 列表，在每个 Reshape 节点之前插入 Concat 节点。倒序插入是关键：因为每次 insert(pos, ...) 会在索引 pos 处插入新元素，导致后续索引偏移。倒序插入可以确保之前记录的索引位置不受影响。
    for pos, node in reversed(inserts):        # 倒序插入，位置不漂移；紧邻其 Reshape 之前
        g.node.insert(pos, node)

    # 获取模型使用的 ONNX 操作集（opset）版本。opset 版本影响某些操作（如 ReduceMax）的语法： - opset < 18：axes 作为节点属性（axes=[2]）。 - opset >= 18：axes 改为作为第二个输入张量传入。
    opset = model.opset_import[0].version   # 决定 ReduceMax 的 axes 是属性(<18)还是输入(>=18)
    # 保存源模型原始输入名（通常是 "images"）和原始输出名（通常是检测头输出的张量名），后续用于重写图的 I/O 连接。
    orig_in = g.input[0].name
    orig_out = g.output[0].name

    #  - pre_nodes：存放预处理阶段的 ONNX 节点。 - post_nodes：存放后处理阶段的 ONNX 节点。 - cur：当前写入指针，初始指向 pre_nodes（先构建预处理图）。 - inits：存放预处理/后处理中添加的常量初始化器。
    pre_nodes = []
    post_nodes = []
    cur = pre_nodes
    inits = []

    # 定义常量创建辅助函数 C。 - 参数 name：常量节点名。 - 参数 value：常量的数值（标量或列表）。 - 参数 dtype：数据类型（默认 F32，可指定 I64）。 
    # - 内部逻辑：将 value 转为对应类型的 numpy 数组 → 用 numpy_helper.from_array 转为 ONNX TensorProto → 追加到 inits 列表 → 返回常量名（供后续节点引用）。
    def C(name, value, dtype=F32):
        arr = np.asarray(value, dtype=np.float32) if dtype == F32 else np.asarray(value, dtype=np.int64)
        inits.append(numpy_helper.from_array(arr, name=name))
        return name

    # 定义节点创建辅助函数 N。 - 参数 op：操作类型名（如 "Resize"、"Mul"）。 - 参数 ins：输入张量名列表或单个字符串。 - 参数 outs：输出张量名列表或单个字符串。 
    # - 参数 **attrs：操作特定属性（如 mode、axis、keepdims 等）。 - 内部逻辑：统一将单个字符串转为列表 → 调用 helper.make_node 创建节点 → 追加到当前容器（pre_nodes 或 post_nodes）。
    def N(op, ins, outs, **attrs):
        cur.append(helper.make_node(op, ins, [outs] if isinstance(outs, str) else outs, **attrs))

    # ---- 常量 ----
    # 创建常用的整数常量： - c_1_i64 = [1]（标量 1，用于 batch 维度拼接等） - c_2_i64 = [2]（用于除以 2 计算 padding 等） - c_3_i64 = [3]（用于通道数 3 和 Concat 的 axis） 
    # - c_0_i64 = [0]（用于 Gather 取第一个元素等） - c_6_i64 = [num_classes] = [6]（类别数，用于最终输出形状拼接）
    c_1_i64   = C("c_1_i64", [1], I64)
    c_2_i64   = C("c_2_i64", [2], I64)
    c_3_i64   = C("c_3_i64", [3], I64)
    c_0_i64   = C("c_0_i64", [0], I64)
    c_6_i64   = C("c_6_i64", [num_classes], I64)   # ★5 类别数

    #  - c_640_i64 = [640]：目标分辨率的整数形式（用于 Shape/Slice 等需要 int64 的操作）。 - c_640_f = 640.0：目标分辨率的浮点形式（用于 Resize/Div 等浮点操作）。 
    # - c_114_f = 114.0：Padding 的填充值（灰度填充，YOLOv8 默认用 114 填充灰色边框）。 - c_inv255 = 1/255.0：归一化倒数，用于将像素值 [0, 255] 缩放到 [0, 1]。 
    # - c_rgb_idx = [2, 1, 0]：通道重排索引，将 RGB 转为 BGR（OpenCV 默认通道顺序）。 - c_half = 0.5：用于 anchor box 中心计算（w/2、h/2）。 - c_0_f = 0.0：浮点零。 - c_maxbox = [300]：NMS 最大输出框数。
    c_640_i64 = C("c_640_i64", [640], I64)
    c_640_f   = C("c_640_f", 640.0, F32)
    c_114_f   = C("c_114_f", 114.0, F32)
    c_inv255  = C("c_inv255", 1.0 / 255.0, F32)
    c_rgb_idx = C("c_rgb_idx", [2, 1, 0], I64)
    c_half    = C("c_half", 0.5, F32)
    c_0_f     = C("c_0_f", 0.0, F32)
    c_maxbox  = C("c_maxbox", [max_dets], I64)      # 1维: NMS 输入 + out_shape Concat 都可用

    #  - c_imax = [9223372036854775807]：Slice 操作的 ends 参数用此值表示”到末尾”。 - c_8400_s = 8400：YOLOv8 默认锚点数量（80 类 × 3 anchors × 80×80/4 = 8400）。
    # 用于 batch 索引的展平计算（batch_idx * 8400 + box_idx）。注意这里用的是 0-d 标量而非列表。 - c_neg1 = [-1]：Reshape 的 -1 占位符，表示”自动推断该维大小”。 
    # - c_bx4 = [-1, 4]：将张量 reshape 为 (任意, 4)，用于将 (B, 8400, 4) 展平为 (B*8400, 4)。 - c_bx1 = [-1, 1]：类似地 reshape 为 (任意, 1)。 
    # - c_ax1 = [1]：Unsqueeze 的 axes 参数，表示在第 1 维插入新维度。
    c_imax    = C("c_imax", [IMAX], I64)            # ★3 Slice ends: 到末尾
    c_8400_s  = C("c_8400_s", 8400, I64)            # ★4 0-d 标量，展平索引乘子
    c_neg1    = C("c_neg1", [-1], I64)              # Reshape 到 1 维/[-1,4]
    c_bx4     = C("c_bx4", [-1, 4], I64)
    c_bx1     = C("c_bx1", [-1, 1], I64)
    c_ax1     = C("c_ax1", [1], I64)                # Unsqueeze axes

    # 定义切片辅助函数 SLICE。 - ONNX 的 Slice 操作需要 3 个输入：[数据, starts, ends]。 - 此函数自动为每次切片创建 starts 和 ends 常量（通过 C 函数），然后创建 Slice 节点。 - 参数 x：被切片的张量名。 
    # - 参数 starts：起始索引列表。 - 参数 ends：结束索引列表。 - 参数 out：输出张量名。
    def SLICE(x, starts, ends, out):
        s = C(f"{out}_st", starts, I64)
        e = C(f"{out}_en", ends, I64)
        N("Slice", [x, s, e], out)
        return out

    # ================= 预处理（batch 动态版） =================
    # 将原始图像（任意尺寸）转换为模型需要的 (B, 3, 640, 640) 格式，包括：转 float、获取形状、缩放、居中 padding、转 CHW、RGB→BGR、归一化。
    N("Cast", ["image_raw"], "img_f", to=F32) # 将原始输入 "image_raw"（uint8 图像）转换为 float32 类型，输出为 "img_f"。
                                            # 因为后续的尺寸计算、缩放等操作需要浮点运算。
    N("Shape", ["img_f"], "shape_img")  # [B,H,W,3]；获取 "img_f" 的形状，输出为 "shape_img"（int64 向量，如 [1, 1920, 1080, 3]）。
                                        # 此形状信息后续用于动态计算缩放比例和 padding。
    N("Gather", ["shape_img", c_0_i64], "batch", axis=0) # ★2 batch 标量(实际为[1]形状)；从 shape_img 中收集（Gather）第 0 维的值，即 batch 大小，
                                    # 输出为 "batch"。这个 "batch" 张量就是前面 Reshape 修补中引用的动态 batch 变量。
    N("Gather", ["shape_img", c_1_i64], "H", axis=0) # 分别从 shape_img 中收集第 1 维（高度 H）和第 2 维（宽度 W）的值。
    N("Gather", ["shape_img", c_2_i64], "W", axis=0)

    N("Cast", ["H"], "Hf", to=F32) # 将整数形式的 H、W 转换为浮点形式 Hf、Wf，用于后续的浮点除法运算。
    N("Cast", ["W"], "Wf", to=F32)

    N("Max", ["Hf", "Wf"], "maxhw") # 取 H 和 W 中的较大值作为缩放基准，保证图像等比例缩放后能完整放入 640×640 的方框内而不裁剪。
    N("Div", [c_640_f, "maxhw"], "scale") # 计算缩放比例 scale = 640.0 / max(H, W)。例如原图 1920×1080，max=1920，scale=640/1920=1/3。
    # 计算缩放后的尺寸：hsc = H * scale，wsc = W * scale。例如 1080 × (640/1920) = 360，1920 × (640/1920) = 640。
    N("Mul", ["Hf", "scale"], "hsc")
    N("Mul", ["Wf", "scale"], "wsc")
    # 对缩放后的尺寸向下取整，得到整数像素尺寸（ONNX Resize 的 sizes 参数需要整数）。
    N("Floor", ["hsc"], "hfl")
    N("Floor", ["wsc"], "wfl")
    # 将整数高度和宽度转为 int64 类型，因为后续 Concat 构建 sizes 向量需要 int64。
    N("Cast", ["hfl"], "nh", to=I64)
    N("Cast", ["wfl"], "nw", to=I64)

    # 将 batch、缩放后高度、缩放后宽度、通道数 3 拼接为形状向量 [batch, nh, nw, 3]，作为 Resize 操作的目标尺寸。
    # 这里首次使用了动态 "batch" 张量，使得 Resize 支持任意 batch 大小。
    N("Concat", ["batch", "nh", "nw", c_3_i64], "sizes", axis=0)  # ★2 [B,nh,nw,3]

    # 执行双线性插值缩放（Resize）。 - 第一个输入：源图像。 - 第二、三个输入：空字符串 ""（表示不使用 scales/axes 参数，直接用第四个输入 sizes 指定目标尺寸）。 
    # - 第四个输入：目标尺寸 [batch, nh, nw, 3]。 - mode="linear"：双线性插值。 - coordinate_transformation_mode="half_pixel"：Half-pixel 坐标变换模式（YOLOv8/ONNX 标准做法，确保缩放后像素中心对齐）。
    N("Resize", ["img_f", "", "", "sizes"], "resized",
      mode="linear", coordinate_transformation_mode="half_pixel")

    # 居中 Padding（batch 维 pad 0，对任意 B 成立）
    # 计算需要填充的总像素数：垂直方向 padh_tot = 640 - nh，水平方向 padw_tot = 640 - nw。
    N("Sub", [c_640_i64, "nh"], "padh_tot")
    N("Sub", [c_640_i64, "nw"], "padw_tot")
    # 计算上半部分和左边的 padding 量（尽量均分，差 1 给下半/右边）：pad_top = padh_tot // 2，pad_left = padw_tot // 2。
    N("Div", ["padh_tot", c_2_i64], "pad_top")
    N("Div", ["padw_tot", c_2_i64], "pad_left")
    # 计算下半部分和右边的 padding：pad_bot = padh_tot - pad_top，pad_right = padw_tot - pad_left。这样总 padding = pad_top + pad_bot = padh_tot。
    N("Sub", ["padh_tot", "pad_top"], "pad_bot")
    N("Sub", ["padw_tot", "pad_left"], "pad_right")
    # 构建 ONNX Pad 操作需要的 pads 向量。ONNX Pad 的 pads 格式为 [before_dim0, before_dim1, before_dim2, before_dim3, after_dim0, after_dim1, after_dim2, after_dim3]。 
    # - dim0 (batch)：前后都 pad 0（不影响 batch 维度）。 - dim1 (height)：前 pad_top，后 pad_bot。 - dim2 (width)：前 pad_left，后 pad_right。 - dim3 (channel)：前后都 pad 0。 
    # 结果：[0, pad_top, pad_left, 0, 0, pad_bot, pad_right, 0]。
    N("Concat", [c_0_i64, "pad_top", "pad_left", c_0_i64,
                 c_0_i64, "pad_bot", "pad_right", c_0_i64], "pads", axis=0)
    # 对缩放后的图像执行常量填充。 - 输入：缩放后的图像 "resized"。 - pads 向量：控制各维填充量。 - 填充值：114.0（灰色，YOLOv8 默认）。 - 输出：(B, nh+padh_tot, nw+padw_tot, 3) = (B, 640, 640, 3) 的填充图像。
    N("Pad", ["resized", "pads", c_114_f], "padded", mode="constant")

    # 通道重排（NHWC → NCHW）。将 (B, 640, 640, 3) 的 HWC 格式转为 (B, 3, 640, 640) 的 CHW 格式，这是 PyTorch/ONNX 模型的标准输入格式。
    # perm=[0, 3, 1, 2] 表示：dim0→dim0(batch)，dim3→dim1(channel)，dim1→dim2(height)，dim2→dim3(width)。
    N("Transpose", ["padded"], "tr", perm=[0, 3, 1, 2])
    # 通道颜色重排（RGB → BGR）。c_rgb_idx = [2, 1, 0] 表示：取第 2 个通道（蓝色）作为新第 0 通道，取第 1 个通道（绿色）保持不变，取第 0 个通道（红色）作为新第 2 通道。
    # 这是因为 OpenCV 的 cv2.imread 默认以 BGR 顺序读取图像，而模型训练时可能用的是 RGB。
    N("Gather", ["tr", c_rgb_idx], "rgb", axis=1)
    # 像素值归一化。将 [0, 255] 范围的像素值乘以 1/255，缩放到 [0, 1] 范围。输出 "preprocessed" 即为最终预处理结果，形状为 (B, 3, 640, 640)，作为检测头的输入。
    N("Mul", ["rgb", c_inv255], "preprocessed")              # (B,3,640,640)

    # ================= 后处理（batch 感知版） =================
    # 将检测头输出（框坐标 + 类别分数）转换为标准检测格式 [x1, y1, x2, y2, confidence, class_id]，并执行 NMS（非极大值抑制）去除重叠框。
    cur = post_nodes # 将当前写入指针切换到 post_nodes，后续节点追加到后处理列表。
    # 转置源模型输出张量。YOLOv8 的检测头输出形状通常为 (B, num_classes+4, num_anchors) 即 (B, 10, 8400)，
    # 转置为 (B, 8400, 10) 使得每个锚点的信息（4 个坐标 + 10 个类别分数）在最后一个维度上连续排列，方便后续切片。
    N("Transpose", [orig_out], "trans", perm=[0, 2, 1])      # (B,10,8400)→(B,8400,10)
    # 从转置后的张量中切片出边界框坐标部分。 - 起始：[0, 0, 0]（从所有维度的起始位置开始）。 - 结束：[IMAX, 8400, 4]（batch 维到末尾用 IMAX 代替，实际取全部 batch；
    # 第 1 维取全部 8400 个锚点；第 2 维取前 4 个元素即坐标）。 - 输出："boxes" 形状为 (B, 8400, 4)，包含 [cx, cy, w, h]。
    SLICE("trans", [0, 0, 0], [IMAX, 8400, 4], "boxes")      # ★3 batch 维到末尾

    # 切片出类别分数部分。 - 起始：[0, 0, 4]（第 2 维从索引 4 开始，跳过前面的 4 个坐标）。 
    # - 结束：[IMAX, 8400, 10]（第 2 维取到索引 10，即 4~9 共 6 个？不对，应该是 4~10 共 6 个… 等等，这里是取 4 到 10，即索引 4,5,6,7,8,9 共 6 个元素？
    # 但 num_classes=6，所以应该是取 4 到 10 共 6 个元素，即 [4:10]）。 - 输出："scores" 形状为 (B, 8400, 6)，包含 6 个类别的原始分数（logits）。
    # 注意：这里 SLICE 的 ends 参数 [IMAX, 8400, 10] 中，IMAX 用于 batch 维表示”到末尾”，8400 表示取全部锚点，10 是硬编码的上限。但 num_classes=6，
    # 所以实际上 scores 应该是 (B, 8400, 6)。这里 ends 写 10 可能是因为源模型导出时 classes=6 但输出维度声明为 10？或者这是一个通用模板，classes 可配置但 slice ends 没同步更新？
    # 从代码注释 ★5 来看，c_6_i64 使用了 num_classes 参数，但 SLICE 的 ends 写的是硬编码的 10。这可能是一个潜在的 bug 或特定模型的适配。
    SLICE("trans", [0, 0, 4], [IMAX, 8400, 10], "scores")

    # 分别从 boxes (B, 8400, 4) 中切片出四个分量： - cx：锚点中心 x 坐标 (B, 8400, 1)。 - cy：锚点中心 y 坐标 (B, 8400, 1)。 - bw：锚点宽度 (B, 8400, 1)。 - bh：锚点高度 (B, 8400, 1)。
    SLICE("boxes", [0, 0, 0], [IMAX, 8400, 1], "cx")
    SLICE("boxes", [0, 0, 1], [IMAX, 8400, 2], "cy")
    SLICE("boxes", [0, 0, 2], [IMAX, 8400, 3], "bw")
    SLICE("boxes", [0, 0, 3], [IMAX, 8400, 4], "bh")

    # 计算宽高的一半，用于从中心点+宽高格式转换为左上角+右下角格式。
    N("Mul", ["bw", c_half], "w2")
    N("Mul", ["bh", c_half], "h2")
    # 将 [cx, cy, w, h]（中心点+宽高）格式转换为 [x1, y1, x2, y2]（左上角+右下角）： 
    # - x1 = cx - w/2（左边界） - x2 = cx + w/2（右边界） - y1 = cy - h/2（上边界） - y2 = cy + h/2（下边界）
    N("Sub", ["cx", "w2"], "x1")
    N("Add", ["cx", "w2"], "x2")
    N("Sub", ["cy", "h2"], "y1")
    N("Add", ["cy", "h2"], "y2")

    # 将 padding 的整数值转换为浮点型，因为后续要与浮点坐标进行加减运算。
    N("Cast", ["pad_left"], "padl_f", to=F32)
    N("Cast", ["pad_top"], "padt_f", to=F32)
    # 去除 padding 的影响，将坐标从填充后的 640×640 空间还原回缩放后的实际图像空间。
    N("Sub", ["x1", "padl_f"], "sub_x1")
    N("Sub", ["y1", "padt_f"], "sub_y1")
    N("Sub", ["x2", "padl_f"], "sub_x2")
    N("Sub", ["y2", "padt_f"], "sub_y2")
    # 除以缩放比例，将坐标从 640×640 缩放空间还原回原始图像的实际尺寸。至此，坐标已完全还原到原始输入图像的空间中。
    N("Div", ["sub_x1", "scale"], "x1o")
    N("Div", ["sub_y1", "scale"], "y1o")
    N("Div", ["sub_x2", "scale"], "x2o")
    N("Div", ["sub_y2", "scale"], "y2o")
    # 将四个分量沿最后一维拼接，得到最终的边界框坐标 (B, 8400, 4)，格式为 [x1, y1, x2, y2]。
    N("Concat", ["x1o", "y1o", "x2o", "y2o"], "boxes_final", axis=2)   # (B,8400,4)

    # 根据 opset 版本，从类别分数中取出最大置信度和对应的类别索引： - ReduceMax：沿第 2 维（类别维）取最大值，
    # 得到每个锚点的最高置信度 conf (B, 8400, 1)。 - opset >= 18：axes 作为输入张量传入（c_axes2 = [2]）。 
    # - opset < 18：axes 作为节点属性传入（axes=[2]）。 - ArgMax：沿第 2 维取最大值的索引，得到类别索引 cls_i (B, 8400, 1)。
    if opset >= 18:
        c_axes2 = C("c_axes2", [2], I64)
        N("ReduceMax", ["scores", c_axes2], "conf", keepdims=1)   # opset>=18: axes 是输入
        N("ArgMax", ["scores"], "cls_i", axis=2, keepdims=1)
    else:
        N("ReduceMax", ["scores"], "conf", axes=[2], keepdims=1)       # (B,8400,1)
        N("ArgMax", ["scores"], "cls_i", axis=2, keepdims=1)

    # 将类别索引（int64）转换为 float32，因为后续 Concat 要求所有输入类型一致（boxes_final 是 float32）。
    N("Cast", ["cls_i"], "cls_f", to=F32)                              # (B,8400,1)

    # 转置置信度张量，从 (B, 8400, 1) 转为 (B, 1, 8400)，以匹配 NonMaxSuppression 操作的输入格式要求。
    N("Transpose", ["conf"], "conf_t", perm=[0, 2, 1])                 # (B,1,8400)
    # 执行非极大值抑制（NMS），去除重叠的检测框。 - 输入 1：boxes_final (B, 8400, 4)，格式为 [x1, y1, x2, y2]。 
    # - 输入 2：conf_t (B, 1, 8400)，置信度分数。 - 输入 3：c_maxbox [300]，最大输出框数。 
    # - 输入 4：iou_thresh（运行时输入），NMS 的 IoU 阈值。 - 输入 5：score_thresh（运行时输入），置信度阈值，低于此值的框会被提前过滤。 
    # - center_point_box=0：表示输入框不是中心点+宽高格式（0=表示不是）。 - 输出："selected" 形状为 (N, 3)，每行为 [batch_idx, class_idx, box_idx]，
    # 其中 box_idx 是在展平的 (B, 8400) 网格中的线性索引。
    N("NonMaxSuppression", ["boxes_final", "conf_t", c_maxbox, "iou_thresh", "score_thresh"],
      "selected", center_point_box=0)                                  # ★4 (N,3)=[b,c,box]

    # ---- ★4 batch 感知 gather: flat_idx = b*8400 + box，在展平张量上取 ----
    # NMS 输出的是展平索引（将 (B, 8400) 展平为一维数组后的索引），需要从展平的张量中按索引取值。
    # 从 NMS 输出中分离出 batch 索引和 box 索引： - b_idx：(N, 1)，每张图片的 batch 编号。 
    # - box_idx：(N, 1)，在 (B, 8400) 展平空间中的索引。
    SLICE("selected", [0, 0], [IMAX, 1], "b_idx")                      # (N,1)
    SLICE("selected", [0, 2], [IMAX, 3], "box_idx")                    # (N,1)
    # 将 2D 索引 (batch_idx, box_idx) 转换为展平一维索引： - b_off = b_idx * 8400：batch 偏移量。 - flat_2d = b_off + box_idx：展平后的二维索引。 
    # - flat_idx = flat_2d.reshape(-1)：转为真正的一维向量 (N,)，用于 Gather 操作。
    N("Mul", ["b_idx", c_8400_s], "b_off")
    N("Add", ["b_off", "box_idx"], "flat_2d")                          # (N,1)
    N("Reshape", ["flat_2d", c_neg1], "flat_idx")                      # (N,)

    # 从展平的 boxes 中按索引选取最终框坐标： - 先将 (B, 8400, 4) reshape 为 (B*8400, 4)。 
    # - 用 flat_idx 在 axis=0 上 Gather，得到 (N, 4) 的最终检测框坐标。
    N("Reshape", ["boxes_final", c_bx4], "boxes_flat")                 # (B*8400,4)
    N("Gather", ["boxes_flat", "flat_idx"], "boxes_sel", axis=0)       # (N,4)
    # 类似地，从展平的置信度中选取： - reshape (B, 8400, 1) → (B*8400, 1)。 
    # - Gather 得到 (N, 1) 的最终置信度。
    N("Reshape", ["conf", c_bx1], "conf_flat")                         # (B*8400,1)
    N("Gather", ["conf_flat", "flat_idx"], "conf_sel", axis=0)         # (N,1)
    # 类似地，从展平的类别索引中选取： - reshape (B, 8400, 1) → (B*8400, 1)。 
    # - Gather 得到 (N, 1) 的最终类别索引（float32）。
    N("Reshape", ["cls_f", c_bx1], "cls_flat")                         # (B*8400,1)
    N("Gather", ["cls_flat", "flat_idx"], "cls_sel", axis=0)           # (N,1)
    # 将框坐标、置信度、类别索引沿最后一维拼接，得到最终检测结果 (N, 6)，
    # 每行为 [x1, y1, x2, y2, conf, class_id]。
    N("Concat", ["boxes_sel", "conf_sel", "cls_sel"], "det", axis=1)   # (N,6)

    # ---- ★5 分散回 (B,300,6)：ScatterND 按图归位，空位自然全零（取代 Pad） ----
    # 目标：将不定长的 NMS 结果（N 可能小于 300）填充/分散到固定形状 (B, 300, 6) 中，空位补零。
    # 获取 NMS 输出中实际检测到的框数 num_sel。
    N("Shape", ["selected"], "shape_sel")
    N("Gather", ["shape_sel", c_0_i64], "num_sel", axis=0)             # (1,)
    # 生成行索引 [0, 1, 2, ..., N-1]，用于 ScatterND 的目标位置。
    N("Range", [c_0_i64, "num_sel", c_1_i64], "rows")                  # (N,)
    N("Unsqueeze", ["rows", c_ax1], "rows_u")                          # (N,1)
    # 拼接 batch 索引和行索引，形成 ScatterND 的索引张量 (N, 2)，
    # 表示每个检测结果在输出矩阵中的二维坐标 [batch_idx, row_idx]。
    N("Concat", ["b_idx", "rows_u"], "scatter_idx", axis=1)            # (N,2)
    # 构建输出形状 [batch, 300, 6]，其中 batch 是动态的，300 是 max_dets，6 是每行 6 个值。
    N("Concat", ["batch", c_maxbox, c_6_i64], "out_shape", axis=0)     # [B,300,6]
    # 创建一个与输出形状相同的全零张量 (B, 300, 6)，作为分散操作的基础（未被填充的位置自然保持为零）。
    N("ConstantOfShape", ["out_shape"], "det_zeros",
      value=numpy_helper.from_array(np.array([0.0], np.float32)))      # (B,300,6) 全零

    # 将检测结果 det (N, 6) 按照 scatter_idx (N, 2) 指定的位置分散到全零张量中。 - 对于每个检测结果 i，将其 6 个值写入 det_zeros[scatter_idx[i]]。 
    # - 未被写入的位置保持为 0，自然实现了”变长结果填充到固定长度”的效果。 - 输出 "detections" 形状为 (B, 300, 6)。
    N("ScatterND", ["det_zeros", "scatter_idx", "det"], "detections")  # (B,300,6)

    # ================= 重写 graph I/O =================
    # 目标：将端到端模型的输入改为原始图像 + 超参数，输出改为检测框。
    # 遍历所有节点，将源模型原始输入名（如 "images"）替换为预处理输出的 "preprocessed"。这样预处理节点就被串联到了数据流中。
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == orig_in:
                node.input[i] = "preprocessed"

    del g.input[:] # 清空原始输入列表，准备重新定义端到端模型的输入。
    # 清空 value_info（中间张量的 shape 声明）。因为源模型导出时 batch=1 是静态的，这些 shape 声明中的 batch 维度都是 1，与新图的动态 batch 不一致。
    # 清空后，ONNX 运行时（如 TensorRT、ONNX Runtime）会自动重新进行 shape 推断，避免 shape 不匹配的错误。
    del g.value_info[:]   # ★0b 源模型导出的中间张量声明仍带 batch=1，清掉让运行时重新推断

    # 添加端到端模型的新输入： 1. "image_raw"：原始图像，类型 uint8，形状为动态的 [batch, height, width, 3]（高度和宽度也是动态的，支持任意分辨率输入）。
    g.input.append(helper.make_tensor_value_info(
        "image_raw", TensorProto.UINT8, ["batch", "height", "width", 3]))   # ★1 batch 动态
    
    # 添加两个运行时超参数输入： 2. "iou_thresh"：NMS 的 IoU 阈值（浮点标量），允许运行时动态调整。 
    # 3. "score_thresh"：置信度阈值（浮点标量），允许运行时动态调整。
    g.input.append(helper.make_tensor_value_info("iou_thresh", F32, [1]))
    g.input.append(helper.make_tensor_value_info("score_thresh", F32, [1]))

    del g.output[:] # 清空原始输出列表。
    # 定义端到端模型的输出： - "detections"：检测结果，类型 float32，形状为 (batch, 300, 6)。
    g.output.append(helper.make_tensor_value_info(
        "detections", F32, ["batch", max_dets, 6]))                        # ★5

    # 将预处理节点、源模型节点、后处理节点按顺序拼接为完整的计算图，替换原有的节点列表。
    all_nodes = pre_nodes + list(g.node) + post_nodes
    del g.node[:]
    g.node.extend(all_nodes)

    # 将预处理和后处理中添加的所有常量初始化器追加到图的初始化器列表中。这样，这些常量在模型加载时会被正确地初始化。
    g.initializer.extend(inits)

    #  1. 使用 ONNX 内置检查器验证模型图的正确性（节点连接、类型匹配、shape 推断等）。如果检查失败会抛出异常。 
    # 2. 将构建好的端到端模型保存到 dst 路径。 3. 返回模型对象。
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"[e2e-dyn-v2] saved {dst}")
    return model

# 定义验证函数，使用 ONNX Runtime 分别运行端到端模型和源模型，对比结果一致性。 - onnx_path：端到端模型路径。 
# - src_onnx：源模型路径（用于产生参考结果）。 - conf=0.45：置信度阈值。 - iou=0.65：IoU 阈值。
def check_with_onnxruntime(onnx_path, src_onnx, conf=0.45, iou=0.65):
    """onnxruntime 验证: batch=1 与参考后处理一致；batch=2 两张图结果互不串扰。"""
    # 导入验证所需的库：onnxruntime 用于推理，cv2 用于图像读取和 NMS 参考实现，pathlib 用于路径操作。
    import sys
    from pathlib import Path
    import onnxruntime as ort
    import cv2
    # 将项目的 dependency_file 目录加入 Python 路径，然后导入自定义的预处理函数 preprocess_cpu 和 letterbox_params（用于产生与端到端模型一致的预处理结果作为参考）。
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from common import preprocess_cpu, letterbox_params

    # 创建两个 ONNX Runtime 推理会话： - e2e：端到端模型（包含预处理+推理+后处理）。 - ref：源模型（仅包含推理，需要外部预处理和后处理）。
    e2e = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ref = ort.InferenceSession(src_onnx, providers=["CPUExecutionProvider"])
    print("  e2e 输入:", [(i.name, i.shape) for i in e2e.get_inputs()])
    print("  e2e 输出:", [(o.name, o.shape) for o in e2e.get_outputs()])

    # 定义参考后处理函数，用 NumPy 手动实现与 ONNX 后处理图相同的逻辑： 1. 转置输出，提取 boxes 和 scores。 2. ArgMax 取类别，Max 取置信度。 3. 按置信度阈值过滤。 
    # 4. 从 [cx, cy, w, h] 转为 [x1, y1, x2, y2]。 5. 去 padding、除 scale，还原到原始图像坐标。
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
        # 使用 OpenCV 的 NMSBoxes 函数执行 NMS。注意这里传入的框格式是 (x, y, w, h)（左上角+宽高），
        # 所以需要 (c-a, d-b) 计算宽高。注释中特别警告：如果直接传 (x1, y1, x2, y2) 会被误解析为 (x1, y1, x2, y2) 当作 (x, y, w, h)，导致 IoU 计算错误。
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(b), float(c - a), float(d - b)]
             for a, b, c, d in zip(x1, y1, x2, y2)],
            confs.tolist(), conf, iou)
        # 处理 NMS 返回的索引格式（OpenCV 的 NMSBoxes 返回嵌套列表），并格式化为统一的结果列表。
        idx = [i[0] if isinstance(i, (list, tuple)) else i for i in idx]
        return [(float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                 float(confs[i]), int(cls[i])) for i in idx]

    # 定义标准化函数，对检测结果进行四舍五入（精度 2 位小数）并排序，用于忽略浮点精度差异后的结果对比。
    def canon(d):
        return sorted([(round(float(a), 2), round(float(b), 2), round(float(c), 2),
                        round(float(d), 2), round(float(e), 2), int(f))
                       for a, b, c, d, e, f in d])

    # 定义结果对比函数。判断两组检测结果是否一致： - 框数相同。 - 每个框的坐标差值 <= 1.0（允许浮点精度误差）。 - 类别索引完全一致。
    def same(cc, ce):
        return len(cc) == len(ce) and all(
            all(abs(x - y) <= 1.0 for x, y in zip(a[:4], b[:4])) and a[5] == b[5]
            for a, b in zip(cc, ce))

    # 读取测试图像，用源模型的参考后处理流程产生参考检测结果 cc，作为对比基准。
    img = cv2.imread('/root/my_FILE/datasets/Data_DeepPCB_YOLO/images/test/00041200.jpg')
    scale, pad_x, pad_y = letterbox_params(img.shape[0], img.shape[1])
    blob, *_ = preprocess_cpu(img)
    ref_dets = postprocess_ref(ref.run(None, {"images": blob})[0], scale, pad_x, pad_y, conf, iou)
    cc = canon(ref_dets)

    # 构建端到端模型的输入字典： - "image_raw"：单张图像，加 batch 维 img[None] → (1, H, W, 3)。 
    # - "iou_thresh"：IoU 阈值。 - "score_thresh"：置信度阈值。
    feed1 = {"image_raw": img[None],
             "iou_thresh": np.array([iou], np.float32),
             "score_thresh": np.array([conf], np.float32)}

    # 1) batch=1：与参考一致；执行 batch=1 推理，过滤掉置信度为 0 的空位，与参考结果对比并打印 OK/DIFF。
    d1 = e2e.run(None, feed1)[0]
    c1 = canon(d1[0][d1[0][:, 4] > 0])
    print(f"  batch=1: 参考 {len(cc)} 框 vs e2e {len(c1)} 框 -> {'OK' if same(cc, c1) else 'DIFF'}")

    # 2) batch=2：两张不同图（同一张 + 翻转）结果应与各自单张结果一致（不串扰）
    # 执行 batch=2 推理验证： 1. 将同一张图及其水平翻转图堆叠为 batch=2 输入。 2. 分别提取两张图的检测结果。 
    # 3. 图 0 应与单张推理结果一致（验证 batch 维度不串扰）。4. 图 1（翻转图）单独用参考流程计算并对比。 5. 如果任一不一致，打印详细差异信息。
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
    ap = argparse.ArgumentParser() # 创建命令行参数解析器。
    ap.add_argument("--src", default="/root/my_FILE/models/yolov8_int8_dy_exclude.onnx")
    ap.add_argument("--dst", default="/root/my_FILE/models/yolov8_e2e_dy.onnx")
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--check", action="store_true", default=True)
    ap.add_argument("--no-check", dest="check", action="store_false")
    args = ap.parse_args()
    build_e2e_conf_dyn(args.src, args.dst, max_dets=args.max_det)
    # 如果开启了验证模式，用两个置信度阈值（0.45 正常阈值 和 0.001 极低阈值用于捕获所有框）分别运行验证，
    # 确保端到端模型在不同阈值下都与参考实现一致。
    if args.check:
        for conf in (0.45, 0.001):
            check_with_onnxruntime(args.dst, args.src, conf=conf, iou=0.65)
