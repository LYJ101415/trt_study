
from pycocotools.coco import COCO
from pathlib import Path
import random
import nncf
import onnx
import cv2
import numpy as np
import io
import onnxruntime as ort
from pycocotools.cocoeval import COCOeval
import contextlib

VAL_IMG_DIR=r"D:\dataset\yolo_PCBData_withoutgroup\images\val"
ANNOTATIONS_JSON = r"D:\dataset\yolo_PCBData_withoutgroup\val_coco.json"  # 必须已存在
coco_gt = COCO(ANNOTATIONS_JSON)  # COCO API 用于加载和操作 COCO 数据集

def preprocess_image(image_path,input_size):
    img=cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    h0,w0=img.shape[:2]
    r = min(input_size[0] / h0, input_size[1] / w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    # 插值方式（interpolation）是双线性插值（INTER_LINEAR），在速度和质量之间取得平衡，且与 YOLOv8 训练一致
    # 之所以要插值是因为缩放时新像素位置与原始像素不重合，需要计算中间值，不插值会产生锯齿、伪影，严重降低检测精度
    dw, dh = input_size[1] - new_w, input_size[0] - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    # BORDER_CONSTANT（常量填充），而非镜像/重复填充
    padded = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    # padded[:,:,::-1]指保持前两个维度不变，将第三个维度（通常是颜色通道）进行反转。
    img_tensor = np.ascontiguousarray(padded, dtype=np.float32) / 255.0 
    # np.ascontiguousarray() 确保数组在内存中连续存储，加速推理
    image = np.expand_dims(img_tensor, axis=0) # 增加 batch 维度（NCHW）
    return image, (r, (dw, dh)) # 返回缩放比例 r 和填充量 (dw, dh)，用于后处理坐标还原

def postprocess_yolo(pred, img_info, conf_thres, iou_thres, max_det):
    if pred.ndim == 3:
        pred = pred[0] # pred是模型推理输出的原始张量（NumPy 数组），移除 batch 维度,(1,10，8400,)->(10，8400)
    if pred.shape[0] == 10 and pred.shape[1] == 8400:
        pred = pred.T # pred现在是（8400，10）；作用：使每一行代表一个预测框（共 8400 个 anchor）
                    # 格式：每行 = [cx, cy, w, h, cls0_score, cls1_score, ..., cls5_score]（假设 6 类）

    boxes_xywh = pred[:, :4] # shape: (8400, 4)，每个预测框的 (center_x, center_y, width, height)
    cls_scores = pred[:, 4:] # shape: (8400, 6) 假设有 6 个类别，每个框对每个类别的置信度（未经过 softmax）
    max_scores = cls_scores.max(axis=1) # cls_scores.shape = (8400, 6);axis=0：沿行方向压缩（结果是每列的最大值 → 得到 6 个值）;
                                        # axis=1：沿列方向压缩（结果是每行的最大值 → 得到 8400 个值）
    max_classes = cls_scores.argmax(axis=1)  # 对应类别 ID

    mask = max_scores > conf_thres # mask 是一个 布尔数组（Boolean array）
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    max_classes = max_classes[mask]

    if len(max_scores) == 0:
        return []

    topk_idx = np.argsort(-max_scores)[:max_det] # # 降序排序，取前 max_det 个（300个）检测框
    boxes_xywh = boxes_xywh[topk_idx]
    max_scores = max_scores[topk_idx]
    max_classes = max_classes[topk_idx]

    """以下将将 (cx, cy, w, h) 转为 (x1, y1, x2, y2)，此坐标仍在 640×640 padded 图像 上。
    以下采用向量计算：boxes_xywh[:, 0] → 取所有行的第 0 列 → [100, 200, 300]；boxes_xywh[:, 2] → 取所有行的第 2 列 → [40, 80, 50]
    boxes_xywh[:, 2] / 2 → [20, 40, 25]；相减：[100 - 20, 200 - 40, 300 - 25] = [80, 160, 275]。
    所以，x1 = np.array([80, 160, 275])为一维数组。
    """
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1) # 将四个一维数组 x1, y1, x2, y2 拼接成一个二维数组，每行是一个边界框 [x1, y1, x2, y2]
    # axis=1意味着按列方向堆叠，使每个框的坐标在同一行
    # ### axis 参数的对比
    # axis 结果 形状 axis=0 按行堆叠 (4, N) axis=1 按列堆叠 (N, 4)
    # axis=0 的效果（不推荐）：
    # boxes_xyxy = [
    #     [100, 200, 300],  # 所有框的 x1
    #     [150, 250, 350],  # 所有框的 y1
    #     [180, 280, 380],  # 所有框的 x2
    #     [220, 300, 420]   # 所有框的 y2
    # ]
    # # shape: (4, N) ❌ 不符合后续处理要求
    # ``` axis=1 的效果（正确）
    # ```
    # boxes_xyxy = [
    #     [100, 150, 180, 220],  # 第0个框完整坐标
    #     [200, 250, 280, 300],  # 第1个框完整坐标
    #     [300, 350, 380, 420]   # 第2个框完整坐标
    # ]
    # # shape: (N, 4) ✅ 每行代表一个检测框


    # r, (left, top) = img_info
    # boxes_xyxy[:, [0, 2]] -= left
    # boxes_xyxy[:, [1, 3]] -= top
    # boxes_xyxy /= r
    # 【关键修复】坐标还原：先减去平移量，再除以缩放比例
    r, (left, top) = img_info
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - left/2) / r
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - top/2) / r

    # 限制在原图范围内
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, None) # 将数组值限制在 [0, ∞) 范围内
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, None) # 将数组值限制在 [0, ∞) 范围内

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(), # 这里boxes_xyxy和max_scores的类型还是numpy.array
        scores=max_scores.tolist(), # 因为numpy的向量化操作高效，适合张量计算
        score_threshold=conf_thres, # 分数阈值，低于此值的框会被 预先过滤 掉
        nms_threshold=iou_thres # IoU 阈值，高于此值的重叠框会被 抑制
    ) # 返回类型为numpy.ndarray（OpenCV 4.x），形状是(K, 1)，K为保留框数量；元素类型是int，表示原始框的索引
   
    if len(indices) == 0:
        return []
    indices = indices.flatten() # 将 cv2.dnn.NMSBoxes 返回的嵌套列表或二维数组展平为一维数组，方便后续遍历。
            # cv2.dnn.NMSBoxes 在 OpenCV 新版本中返回的是 (K, 1) 的 NumPy 数组（即使 K=1），例如：[[3], [0], [2]] 而不是 [3, 0, 2]
            # 此时的结果为：[3, 0, 2]  → shape: (3,)；现在可以直接用 for i in indices: 遍历每个索引。
    detections = []
    for i in indices:
        x1, y1, x2, y2 = boxes_xyxy[i]
        detections.append([
            float(x1), float(y1), float(x2), float(y2),
            float(max_scores[i]), int(max_classes[i])
        ])
    return detections # 输出格式：[[x1, y1, x2, y2, score, class_id], ...]
    """举例：假设经过 NMS 后：indices = [1, 0]（保留第 1 和第 0 个框）；
    boxes_xyxy = [[100, 150, 180, 220], [200, 250, 280, 300]]；
    max_scores = [0.95, 0.88]；max_classes = [2, 5]（假设类别 ID）；
    第一次迭代（i = 1）：boxes_xyxy[1] = [200, 250, 280, 300]；max_scores[1] = 0.88；max_classes[1] = 5；添加：detections[200.0, 250.0, 280.0, 300.0, 0.88, 5]
    第二次迭代（i = 0）：；boxes_xyxy[0] = [100, 150, 180, 220]；max_scores[0] = 0.95；max_classes[0] = 2；添加：detections[100.0, 150.0, 180.0, 220.0, 0.95, 2]
    最终 ：detections：[[200.0, 250.0, 280.0, 300.0, 0.88, 5],[100.0, 150.0, 180.0, 220.0, 0.95, 2]]
    在这里，detections是一个 Python 列表（list），其长度最多为 max_det = 300，每个元素是一个包含 6 个数值的子列表：[x1, y1, x2, y2, score, class_id]"""

def val_fn(onnx_model,validation_dataset):
    model_bytes=io.BytesIO() # io.BytesIO() 创建一个内存中的二进制流，避免每次验证都写磁盘
    onnx.save(onnx_model,model_bytes) # 将 ONNX 模型序列化到内存缓冲区，而不是磁盘文件
    ort_session = ort.InferenceSession(     
        model_bytes.getvalue(), # model_bytes.getvalue()的返回值：类型：bytes（Python 字节串）；
        # 内容：ONNX 模型的完整二进制序列化数据；大小：等于模型文件的大小（如 15MB）；格式：ONNX 标准格式（protobuf 序列化）
        providers=["CPUExecutionProvider"]
    ) # 加载模型并创建推理引擎，使用 CPU 执行器
    input_name = ort_session.get_inputs()[0].name # 从 ORT session 动态获取第一个输入张量的名称（如 images ）
    input_h, input_w = ort_session.get_inputs()[0].shape[2:]  # 从输入张量的 shape 中提取 height 和 width （索引 2: 因为 shape 是 [N, C, H, W] ）
    input_size = (input_h, input_w)     
    try:                    # 尝试直接将 nncf.Dataset 转为 list
        val_paths = list(validation_dataset)  # 尝试直接转换
    except TypeError:       # 如果直接转换失败（当前版本）
        val_paths = validation_dataset.get_data()  # 调用 get_data() 获取原始数据（当前版本的稳定接口）

    target_filenames = {Path(p).name for p in val_paths} # 从路径中提取纯文件名（如 000000123456.jpg）
    # 建立文件名与 COCO image_id 的映射
    target_img_ids = [ # 列表推导式->[表达式 for 元素 in 可迭代对象 if 条件]->返回一个列表
        img['id'] for img in coco_gt.dataset['images']
        if img['file_name'] in target_filenames
    ] # 遍历 COCO 标注的 images 字段，找出文件名匹配的图像 ID
    if not target_img_ids:
        return 0.0 # 如果没有匹配的图像 ID，直接返回 0.0（避免后续报错）

    # 推理
    results = []
    for img_id in target_img_ids:
        img_info = coco_gt.loadImgs(img_id)[0] # coco_gt 是一个 pycocotools.coco.COCO 对象，用于加载和操作 COCO 格式的标注数据（如 instances_val2017.json）。
                                      # img_id 是某张图像在 COCO 数据集中的唯一 ID，loadImgs(img_id)是 COCO API 的方法，用于根据图像 ID 加载图像元信息，
                                      # [0]：取列表中的第一个（也是唯一一个）元素，即该图像的元信息字典。
        img_path = Path(VAL_IMG_DIR) / img_info["file_name"] # 从标注文件中取文件名再拼接出图片路径
        if not img_path.exists():
            continue

        img_tensor, scale_pad = preprocess_image(img_path, input_size)
        # img_input = np.expand_dims(img_tensor, axis=0)
        pred = ort_session.run(None, {input_name: img_tensor})[0]
        #pred的输出形状为：(1, 10, 8400)

        detections = postprocess_yolo(
            pred, 
            scale_pad,
            conf_thres=0.001,  # 低阈值确保有输出
            iou_thres=0.7, # 官方默认0.7，如果任务需要更密集的检测（如小目标） → 可尝试 iou=0.5 或 0.45，但要注明
            max_det=300
        )

        for det in detections:
            x1, y1, x2, y2, score, cls_id = det # (x1, y1)：左上角坐标，( x2, y2)：右下角坐标
            w, h = x2 - x1, y2 - y1  # 将 YOLO 常用的 [x1, y1, x2, y2] 边界框格式转换为 COCO 评估要求的 [x, y, width, height] 格式
            results.append({
                "image_id": img_id,
                "category_id": int(cls_id),
                "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)], # 使用 round() 是为了：控制小数位数（避免浮点精度冗余）
                "score": round(score, 4)
            })

    # COCO 评估
    if not results:
        map50_95 = 0.0
    else:
        coco_dt = coco_gt.loadRes(results) # loadRes()：将检测结果 results（Python list）转换为 COCO 格式的“检测数据集”对象（cocoDt），
                                     # loadRes内部也有一些输出，进入源码进行注释，有需要再去取消
        coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox") # 创建COCOeval评估器，iouType="bbox"：评估边界框（也可选 "segm" 用于实例分割）
        coco_eval.params.imgIds = target_img_ids # 评估的图片集
    # ----以下是抑制 COCOeval 的 print 输出，如果要调试，可打开注释打开下方-----
        # with contextlib.redirect_stdout(None):  # 或 open(os.devnull, 'w')
        #     coco_eval.evaluate() # 计算每张图、每个类别、每个 IoU 阈值下的匹配情况（TP/FP）
        #     coco_eval.accumulate() # 聚合结果，计算 AP/AR 等指标
        #     coco_eval.summarize() # 打印标准评估摘要（如 mAP, mAP@0.5, mAP@0.75 等）。
    #-----------------------------------------------------------
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        map50_95 = coco_eval.stats[0]  # mAP@[0.5:0.95]

    return float(map50_95)

onnx_model=onnx.load(r"D:\CV_yolov11\yolov11s_PCB_best.onnx")
input_name = onnx_model.graph.input[0].name

def transform(data_item):
    img,_=preprocess_image(data_item,(640,640))
    return {input_name:img}

all_image_paths=sorted(Path(VAL_IMG_DIR).glob("*.jpg"))
calib_image_paths = random.sample(all_image_paths,min(200,len(all_image_paths))) 
# 从所有可用图像路径中，随机抽取最多 200 张图片的路径，作为量化校准（Calibration）数据集。
calibration_dataset = nncf.Dataset(calib_image_paths,transform)

image_paths = sorted(Path(VAL_IMG_DIR).glob("*.jpg")) 
validation_image_paths_str = [str(p) for p in image_paths]
validation_pairs = validation_image_paths_str[:50]
validation_dataset = nncf.Dataset(validation_pairs)

quantized_model=nncf.quantize_with_accuracy_control(
    model=onnx_model,
    calibration_dataset=calibration_dataset,
    validation_dataset=validation_dataset,
    validation_fn=val_fn,
    max_drop=0.01,
    drop_type=nncf.DropType.ABSOLUTE
)

output_path = "yolov11_quantized_acq_map50-95.onnx"
onnx.save(quantized_model, output_path)
print(f"✅ Quantization completed! Saved to: {output_path}")