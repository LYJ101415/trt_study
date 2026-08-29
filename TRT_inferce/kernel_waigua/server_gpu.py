"""
server_gpu.py — GPU 预处理 + 推理 + GPU 后处理（保留现有 engine，零 PyTorch）。

对比 server_trtapi.py：把 letterbox/normalize 与 decode/NMS 全部搬到 GPU，
宿主只做 JPEG 解码 + 一次 H2D(原始 uint8) + execute_v2 + 一次小 D2H(最终框)。

用法:  python server_gpu.py image.jpg
       python server_gpu.py images/
"""

import sys # 修改 Python 搜索路径 + 读取命令行参数
import time
import glob # 批量匹配图片文件
import ctypes # 底层 CUDA API 调用需要（封装在 common / gpu_ops 中）
from pathlib import Path # 跨平台路径处理

import cv2
import numpy as np
import tensorrt as trt

# 将当前脚本所在目录的上一级目录插入 sys.path 最前面，确保能导入同项目下的 common 和 gpu_ops 模块。
# Path(__file__).resolve()：获取脚本绝对路径（消除符号链接）;.parents[1]：上两级 → 即项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""common 模块：封装了底层 CUDA 内存操作和通用工具
cuda_malloc / cuda_free：GPU 显存分配/释放（ctypes 调用 cudaMalloc/cudaFree）
cuda_memcpy_h2d：Host→Device 内存拷贝
device_addr：从自定义 buffer 对象提取 GPU 指针地址（供 TRT execute_v2 使用）
letterbox_params：计算等比缩放 + padding 参数
draw_results：在图像上绘制检测框
IMG_SIZE / CONF_THRESH / NMS_THRESH / NUM_CLASSES：全局超参常量
gpu_ops.GpuOps：封装了自定义 CUDA kernel（预处理 + 后处理），是本项目"全 GPU 管线"的核心"""
from common import (cuda_malloc, cuda_free, cuda_memcpy_h2d, device_addr,
                    letterbox_params, draw_results,
                    IMG_SIZE, CONF_THRESH, NMS_THRESH, NUM_CLASSES)
from gpu_ops import GpuOps

# ==================== 配置区 ====================
ENGINE_PATH = "/root/my_FILE/models/yolov8_int8_waigua.engine"   # 由 best_PCB.onnx 构建的静态 engine
# ================================================

MAX_RAW_BYTES = 4096 * 4096 * 3  # 原始图像 GPU 缓冲区的初始分配上限。对应一张 4K RGB 图的大小。超过此尺寸时 _ensure_raw 会自动扩容。预分配避免每帧重新 malloc。


class TRTInferenceGPU:
    """纯 TRT API + 自定义 CUDA kernel 的 GPU 端到端推理（预处理/后处理均在 GPU）。"""

    def __init__(self, engine_path: str,
                 conf_thresh: float = CONF_THRESH,   # ** 添加阈值参数用于控制，默认值仍取全局常量,向后兼容
                 nms_thresh: float = NMS_THRESH      # ** 添加阈值参数用于控制，默认值仍取全局常量,向后兼容
                 ):     
        self.logger = trt.Logger(trt.Logger.WARNING) # 创建 TRT 日志器，WARNING 级别过滤冗余信息。

        # **添加阈值参数用于控制后处理阈值，默认值仍取全局常量。
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        with open(engine_path, "rb") as f:
            """trt.Runtime：引擎反序列化器，与 Builder 对应
            f.read()：将整个 .engine 文件读入内存（IHostMemory）
            deserialize_cuda_engine：反序列化为 ICudaEngine 对象
            ⚠️ Engine 与 TRT 版本强绑定，版本不匹配会返回 None
            """
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None: # 加载失败保护。常见原因：TRT 版本不匹配、文件损坏、GPU 架构不支持。
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        # 创建执行上下文，用于执行推理;创建 ExecutionContext。Context 持有推理时的临时显存（workspace），
        # 一个 Engine 可创建多个 Context 支持多流并发。
        self.context = self.engine.create_execution_context()

        # 静态 engine：直接用实际 shape（best_PCB.onnx 是 [1,3,640,640] 固定）;
        # 获取输入/输出元信息（引擎的 Binding 名称）,硬编码输入/输出 tensor 名称。这些名称来自 ONNX 导出时的命名，必须与引擎一致。
        self.in_name = "images"
        self.out_name = "output0"
        in_shape = tuple(self.engine.get_tensor_shape(self.in_name)) # 从引擎查询 tensor shape。静态模型返回固定值如 (1,3,640,640)；动态模型会含 -1。
        out_shape = tuple(self.engine.get_tensor_shape(self.out_name))

        # 将 TRT 数据类型（如 trt.float32）转为 NumPy dtype（如 np.float32），用于后续显存大小计算。
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))
        # YOLOv8 输出 shape 为 [1, 84, 8400]（COCO 80类 + 4 bbox = 84 channels × 8400 anchors）。n_anchors=8400 是后处理解码的关键参数。
        self.n_anchors = out_shape[2]   # 8400

        # GPU 显存：模型输入 + 模型输出（后处理直接在 d_output 上算，无需拷回）;np.prod(shape)：计算元素总数
        # .itemsize：每个元素的字节数,两者相乘 = 总字节数;关键设计：d_output 既是 TRT 推理的输出缓冲，也是 GPU 后处理的输入缓冲 → 零拷贝衔接，省去一次 D2H + H2D
        self._d_input = cuda_malloc(int(np.prod(in_shape)) * np.dtype(self.input_dtype).itemsize)
        self._d_output = cuda_malloc(int(np.prod(out_shape)) * np.dtype(self.output_dtype).itemsize)

        # 原始图显存缓冲（按需扩容）;原始图像 GPU 缓冲延迟分配，首次推理时通过 _ensure_raw 按需创建。
        self._d_raw = None
        self._raw_cap = 0

        # 初始化 GPU 算子:创建自定义 CUDA kernel 管理器。
        # num_classes：类别数（COCO=80）;max_dets=300：NMS 后最多保留的检测框数量（影响 GPU 后处理输出缓冲大小）
        self.gpu = GpuOps(num_classes=NUM_CLASSES, max_dets=300) # 这里的NUM_CLASSES要到common.py里改。

        # 打印摘要，启动时确认引擎加载正确，便于排查 shape/dtype 不匹配问题。
        print(f"[TRT-GPU] Engine loaded | input {in_shape} {self.input_dtype} "
              f"| output {out_shape} | anchors {self.n_anchors}")

        # Warmup：为什么必须 Warmup？CUDA kernel 首次执行触发 JIT 编译 / lazy loading
        # TensorRT 内部可能有延迟初始化（如 cuDNN handle），GPU 频率从 idle 升到 boost clock，显存池预热，避免首帧 malloc 开销
        # 用 64×64 小图 warmup 即可触发所有初始化路径，耗时极短。不 warmup 会导致前几帧延迟虚高 2-10 倍。
        print("[TRT-GPU] Warming up...")
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(3):
            self.infer(dummy)
        print("[TRT-GPU] Warmup done")

    def _ensure_raw(self, nbytes: int): # 这个函数是原始图缓冲按需扩容
        """策略：首次调用或当前缓冲不够大时，释放旧缓冲，分配新缓冲
        新容量取 max(实际需求, MAX_RAW_BYTES)，避免小图频繁 realloc
        一旦扩容到较大值，后续小图复用不再释放 → 单调增长，避免抖动
        这是典型的 amortized allocation 策略，兼顾内存效率和性能稳定性。
        """
        if self._d_raw is None or nbytes > self._raw_cap:
            if self._d_raw is not None:
                cuda_free(self._d_raw)
            cap = max(nbytes, MAX_RAW_BYTES)
            self._d_raw = cuda_malloc(cap)
            self._raw_cap = cap

    # 单帧推理主流程：预处理 + 推理 + 后处理
    def infer(self, image: np.ndarray,
              conf_thresh: float = None,             # ** 新增的参数，用于控制后处理阈值
              nms_thresh: float = None) -> list: 
        """image: BGR uint8 HWC（cv2.imread 的结果）。返回 [(x1,y1,x2,y2,conf,cls), ...];
        接口约定：输入 OpenCV 标准格式（BGR uint8 HWC），输出检测列表。"""
        # 新增阈值参数的优先级: 推理参数 > 构造参数 > 全局常量
        ct = conf_thresh if conf_thresh is not None else self.conf_thresh
        nt = nms_thresh  if nms_thresh  is not None else self.nms_thresh

        # Step 1：计算 Letterbox 参数:scale：等比缩放因子（如 640/1920 ≈ 0.333）,pad_x, pad_y：居中 padding 像素数
        h, w = image.shape[:2]
        scale, pad_x, pad_y = letterbox_params(h, w)

        # Step 2：H2D 传输原始图
        raw = np.ascontiguousarray(image) # 确保内存连续。cv2.imread 通常已连续，但 ROI 裁剪等操作可能产生非连续数组。非连续内存传给 CUDA 会导致错误结果。
        self._ensure_raw(raw.nbytes) # 确保 GPU 缓冲足够容纳当前图像。
        cuda_memcpy_h2d(self._d_raw, raw) # 将原始 BGR uint8 图像从 CPU 拷贝到 GPU。这是整个管线中唯一的一次 H2D 传输。 

        # Step 3：在 GPU做预处理
        """自定义 CUDA kernel 在 GPU 上完成：Letterbox resize：等比缩放 + 居中 padding（灰色填充 114）,
        BGR → RGB 通道转换,uint8 → float32 归一化（÷255.0）,HWC → CHW 布局转换,输出直接写入 d_input（TRT 模型输入缓冲），零中间拷贝。
        相比 CPU 预处理，GPU 预处理在高分辨率下快 5-20 倍。
        """
        self.gpu.preprocess(self._d_raw, h, w, self._d_input,   # 预处理(letterbox+归一化)在 GPU
                            scale, pad_x, pad_y)

        # Step 4：TensorRT 推理
        """execute_v2：TensorRT 同步推理接口（v2 支持显式 batch）,参数是 GPU 指针列表，按 binding index 排列：[input_ptr, output_ptr]
        device_addr()：从自定义 buffer 对象提取原始 CUDA 指针（ctypes.c_void_p）,推理结果直接写入 d_output，仍在 GPU 上
        """
        self.context.execute_v2([device_addr(self._d_input),
                                 device_addr(self._d_output)])   # 推理在 GPU

        # Step 5：GPU 后处理
        """自定义 CUDA kernel 在 GPU 上完成：解码：将 [84, 8400] 输出解析为 (x1,y1,x2,y2,conf,cls),
        置信度过滤：剔除 conf < CONF_THRESH 的框,NMS：Non-Maximum Suppression，剔除重叠框
        坐标还原：用 scale, pad_x, pad_y 将坐标映射回原图尺寸,D2H 拷贝结果：仅拷贝最终检测结果（通常几十个框，几百字节）
        这是第二次也是最后一次 GPU↔CPU 数据传输，且数据量极小。
        """
        return self.gpu.postprocess(self._d_output, self.n_anchors,  # 后处理(NMS)在 GPU
                                    ct, nt, scale, pad_x, pad_y)

    def __del__(self): # 析构释放显存
        """Python GC 时释放所有 GPU 显存。try/except：防止解释器退出时 CUDA context 已销毁导致 segfault
        __del__ 不保证及时调用，生产环境建议显式调用 close() 方法或使用 context manager
        """
        for buf in (self._d_raw, self._d_input, self._d_output):
            try:
                if buf is not None:
                    cuda_free(buf)
            except Exception:
                pass


def main():
    # 命令行第一个参数为图片路径或目录，缺省使用默认测试图。
    target = sys.argv[1] if len(sys.argv) > 1 else "/root/my_FILE/my_FILE/test_images/00041200.jpg"
    # 创建推理引擎实例（触发加载 + warmup）。
    engine = TRTInferenceGPU(ENGINE_PATH, conf_thresh=0.25, nms_thresh=0.5) # ** 新增的参数，用于控制后处理阈值

    # 收集图片列表:
    if Path(target).is_dir():
        paths = sorted(glob.glob(str(Path(target) / "*.[jp][pn]g")))
    else:
        paths = [target]

    # 逐帧推理 + 统计:读取失败的图片跳过并警告，不中断批量测试。
    print(f"\n{'=' * 50}\nProcessing {len(paths)} image(s)...\n{'=' * 50}\n")
    total = 0.0
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            print(f"[WARN] Cannot read: {path}")
            continue

        # perf_counter 精度优于 time.time()，适合毫秒级测量。包含 H2D + 预处理 + 推理 + 后处理 + D2H 全流程耗时。
        t0 = time.perf_counter()
        detections = engine.infer(image, nms_thresh=0.7) # ** 新增的参数，用于控制后处理阈值,可一个或两个。
        elapsed = (time.perf_counter() - t0) * 1000
        total += elapsed

        # 打印每张图的检测数量和耗时，以及每个检测框的详细信息。
        print(f"[{Path(path).name}] {len(detections)} objects | {elapsed:.1f} ms")
        for x1, y1, x2, y2, conf, cls in detections:
            print(f"  cls={cls} conf={conf:.3f} box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        # 保存可视化结果:mkdir(parents=True, exist_ok=True)：递归创建目录，已存在不报错
        # image.copy()：避免在原图上绘制影响后续使用,draw_results：在图像上画框 + 标签 + 置信度
        out_dir = Path("./images_results_gpu")
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"result_{Path(path).stem}.jpg"),
                    draw_results(image.copy(), detections))

    # 计算平均延迟和等效 FPS。max(len(paths), 1) 防除零。
    avg = total / max(len(paths), 1)
    print(f"\n{'=' * 50}\nAverage: {avg:.1f} ms/image | FPS: {1000 / avg:.1f}\n{'=' * 50}")


if __name__ == "__main__":
    main()

"""
怎么查看一个引擎的所有 Binding 名称？,就是模型的输入与输出名：
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    shape = engine.get_binding_shape(i)
    dtype = engine.get_binding_dtype(i)
    is_input = engine.binding_is_input(i)
    print(f"[{'IN' if is_input else 'OUT'}] {name}: {shape} {dtype}")
"""
