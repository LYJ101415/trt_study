"""
TensorRT 本地推理 — 边缘设备专用
零网络 / 零服务框架 / 零多余依赖
用法: python infer.py image.jpg
      python infer.py images/          # 批量推理整个文件夹
"""

import sys
import time
import glob
from pathlib import Path

import cv2
import torch
import numpy as np
import tensorrt as trt


# ==================== 配置区 ====================
ENGINE_PATH = "/root/my_FILE/models/yolov8_int8_fixed.engine"
INPUT_NAME = "images"        # ← 改成你 Engine 的实际输入名
OUTPUT_NAME = "output0"      # ← 改成你 Engine 的实际输出名
IMG_SIZE = 640               # 模型输入尺寸
CONF_THRESH = 0.45            # 置信度阈值
NMS_THRESH = 0.65            # NMS 阈值
NUM_CLASSES = 6             # COCO 类别数
# ================================================


class TRTInference:
    """轻量 TRT 本地推理引擎"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        # 创建执行上下文（边缘设备通常单线程，一个 context 够用）
        self.context = self.engine.create_execution_context()

        # 获取 I/O 信息
        self.input_shape = tuple(self.engine.get_tensor_shape(INPUT_NAME))   # (-1,3,H,W)
        self.output_shape = tuple(self.engine.get_tensor_shape(OUTPUT_NAME))  # (-1,C+4,N)
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(INPUT_NAME))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(OUTPUT_NAME))

    # --- 在初始化时一次性分配 GPU 显存（pinned memory + device buffer）------------
        self._d_input = torch.empty(self.input_shape, dtype=torch.float32, device='cuda')
        self._d_output = torch.empty(self.output_shape, dtype=torch.float32, device='cuda')
        # CPU 端 pinned memory，加速 H2D/D2H 传输
        self._h_input = torch.empty(self.input_shape, dtype=torch.float32).pin_memory()
        self._h_output = torch.empty(self.output_shape, dtype=torch.float32).pin_memory()
    #-----------------------------------------------------------------------------

        print(f"[TRT] Engine loaded")
        print(f"  Input:  {self.input_shape} {self.input_dtype}")
        print(f"  Output: {self.output_shape} {self.output_dtype}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Letterbox resize + normalize → [1, C, H, W] float32"""
        h, w = image.shape[:2]
        scale = min(IMG_SIZE / h, IMG_SIZE / w)
        nh, nw = int(h * scale), int(w * scale)

        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # 居中填充灰色边框
        canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        pad_h = (IMG_SIZE - nh) // 2
        pad_w = (IMG_SIZE - nw) // 2
        canvas[pad_h:pad_h + nh, pad_w:pad_w + nw] = resized

        # BGR→RGB, HWC→CHW, normalize
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return blob[np.newaxis, ...], scale, pad_w, pad_h

    def postprocess(self, output: np.ndarray, scale: float,
                    pad_w: int, pad_h: int, orig_shape: tuple) -> list:
        """解析 YOLOv8 输出 → [(x1,y1,x2,y2,conf,class_id), ...]"""
        # output shape: [1, (4+num_classes), num_boxes]
        preds = output[0].T  # [num_boxes, 4+num_classes]

        boxes_xywh = preds[:, :4]
        scores_all = preds[:, 4:]
        class_ids = scores_all.argmax(axis=1)
        confs = scores_all.max(axis=1)

        # 过滤低置信度
        mask = confs > CONF_THRESH
        boxes_xywh = boxes_xywh[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # xywh → xyxy
        x, y, w, h = boxes_xywh.T
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # 还原到原图坐标
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale

        # NMS
        indices = cv2.dnn.NMSBoxes(
            [[x1[i], y1[i], x2[i], y2[i]] for i in range(len(x1))],
            confs.tolist(), CONF_THRESH, NMS_THRESH
        )
        indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

        results = []
        for i in indices:
            results.append((
                float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]),
                float(confs[i]), int(class_ids[i])
            ))
        return results

    # @torch.no_grad()
    # def infer(self, image: np.ndarray) -> list: # 使用pycuda来分配GPU内存
    #     """端到端：图片 → 检测结果"""
    #     orig_shape = image.shape[:2]
    #     blob, scale, pad_w, pad_h = self.preprocess(image)

    #     # 设置动态 batch shape
    #     self.context.set_input_shape(INPUT_NAME, blob.shape)

    #     # 分配 GPU 内存
    #     # import cuda  # pycuda; 或用 torch 替代（见下方说明）
    #     import pycuda.driver as cuda
    #     import pycuda.autoinit

    #     d_input = cuda.mem_alloc(blob.nbytes)
    #     out_shape = (blob.shape[0],) + tuple(self.output_shape[1:])
    #     output_np = np.empty(out_shape, dtype=self.output_dtype)
    #     d_output = cuda.mem_alloc(output_np.nbytes)

    #     # 推理
    #     cuda.memcpy_htod(d_input, blob)
    #     self.context.execute_v2([int(d_input), int(d_output)])
    #     cuda.memcpy_dtoh(output_np, d_output)

    #     # 释放显存
    #     d_input.free()
    #     d_output.free()

    #     return self.postprocess(output_np, scale, pad_w, pad_h, orig_shape)

    @torch.no_grad()
    def infer(self, image: np.ndarray) -> list: # 使用torch来分配GPU内存       
    #------显存的分配在这里做每次都需要分配，放到类的初始化上做一次即可，大大节省时间------------    
        # orig_shape = image.shape[:2]
        # blob, scale, pad_w, pad_h = self.preprocess(image)
        # self.context.set_input_shape(INPUT_NAME, blob.shape)
        # # PyTorch 替代 PyCUDA 的显存操作
        # d_input = torch.from_numpy(blob).contiguous().cuda()
        # out_shape = (blob.shape[0],) + tuple(self.output_shape[1:])
        # d_output = torch.empty(out_shape, dtype=torch.float32, device='cuda')
        # self.context.execute_v2([d_input.data_ptr(), d_output.data_ptr()])
        # output_np = d_output.cpu().numpy()
    #---------------------------------------------------------------------------------------
        orig_shape = image.shape[:2]
        blob, scale, pad_w, pad_h = self.preprocess(image)

        # 动态 batch 校验（防止越界）
        assert blob.shape[0] <= self.input_shape[0], \
            f"Batch {blob.shape[0]} exceeds pre-allocated {self.input_shape[0]}"

        # 设置实际输入形状（支持动态 batch）
        self.context.set_input_shape(INPUT_NAME, blob.shape)

        # ✅ 复用预分配缓冲区，零额外显存分配
        self._h_input[:blob.shape[0]].copy_(torch.from_numpy(blob))
        self._d_input[:blob.shape[0]].copy_(self._h_input[:blob.shape[0]], non_blocking=True)

        # 若输出 shape 随输入变化，需重新计算
        actual_out_shape = (blob.shape[0],) + tuple(self.output_shape[1:])
        d_out_view = self._d_output[:blob.shape[0]]

        self.context.execute_v2([self._d_input.data_ptr(), d_out_view.data_ptr()])

        # 异步拷贝回 CPU
        h_out_view = self._h_output[:blob.shape[0]]
        h_out_view.copy_(d_out_view, non_blocking=True)
        torch.cuda.synchronize()  # 确保拷贝完成
        output_np = h_out_view.numpy()

        return self.postprocess(output_np, scale, pad_w, pad_h, orig_shape)


def draw_results(image: np.ndarray, detections: list) -> np.ndarray:
    """在图上画框（可选，用于可视化验证）"""
    COLORS = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=int)
    for x1, y1, x2, y2, conf, cls_id in detections:
        color = tuple(COLORS[cls_id].tolist())
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"cls{cls_id} {conf:.2f}"
        cv2.putText(image, label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return image


def main():
    # if len(sys.argv) < 2:
    #     print("Usage: python infer.py <image_or_folder>")
    #     sys.exit(1)

    # target = sys.argv[1]
    target = "/root/my_FILE/my_FILE/test_images"
    engine = TRTInference(ENGINE_PATH)

    # 收集图片路径
    if Path(target).is_dir():
        paths = sorted(glob.glob(str(Path(target) / "*.[jp][pn]g")))
    else:
        paths = [target]

    print(f"\n{'='*50}")
    print(f"Processing {len(paths)} image(s)...")
    print(f"{'='*50}\n")

    total_time = 0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Cannot read: {path}")
            continue

        # 预热后计时
        t0 = time.perf_counter()
        detections = engine.infer(image)
        elapsed = (time.perf_counter() - t0) * 1000
        total_time += elapsed

        print(f"[{Path(path).name}] {len(detections)} objects | {elapsed:.1f} ms")
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            print(f"  cls={cls} conf={conf:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        # 固定输出文件夹（可根据需要修改路径）
        OUTPUT_DIR = Path("./images_results")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # 生成唯一输出路径，避免同名覆盖
        out_path = OUTPUT_DIR / f"result_{Path(path).stem}.jpg"
        counter = 1
        while out_path.exists():
            out_path = OUTPUT_DIR / f"result_{Path(path).stem}_{counter}.jpg"
            counter += 1

        # 可选：保存可视化结果;绘制并保存到固定文件夹（使用上面构建好的 out_path）
        vis = draw_results(image.copy(), detections)
        cv2.imwrite(str(out_path), vis)  # ← 直接用 out_path，不要重新赋值！
        print(f"  → Saved: {out_path}")

    avg = total_time / max(len(paths), 1)
    print(f"\n{'='*50}")
    print(f"Average: {avg:.1f} ms/image | FPS: {1000/avg:.1f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

"""
# 单张推理
python infer.py test.jpg
# 批量推理整个文件夹
python infer.py ./test_images/
"""