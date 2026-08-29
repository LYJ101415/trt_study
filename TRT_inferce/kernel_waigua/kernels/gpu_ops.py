"""
gpu_ops.py — ctypes 封装 libgpu_ops.so，提供 GPU 预处理 / 后处理的高层接口。

依赖 common.py 的 cudart 封装做显存分配。所有 device 缓冲一次性分配、跨帧复用。
"""
"""
| 语句 |                              作用 |
|------|------|
| `import ctypes`           | Python 标准库，用于调用 C/C++ 动态链接库（.so/.dll） |
| `Path(__file__).resolve().parents[1]`    | 获取当前文件的上两级目录，即项目根目录 |
| `sys.path.insert(0, ...)` | 将项目根目录插入模块搜索路径最前面，确保能找到 `common` 模块 |
| `cuda_malloc / cuda_free` | 对 `cudaMalloc` / `cudaFree` 的 Python 封装，返回/接收显存指针 |
| `cuda_memcpy_d2h`         | 对 `cudaMemcpy(dst, src, size, DeviceToHost)` 的封装 |
| `device_addr`             | 从 cupy/cuda-python 等对象中提取原始设备指针（整数地址） |

"""
import ctypes   # Python 标准库，用于调用 C/C++ 动态链接库（.so/.dll
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import cuda_malloc, cuda_free, cuda_memcpy_d2h, device_addr


class GpuOps:
    def __init__(self, num_classes=6, max_dets=300): # 初始化与资源分配
        self.num_classes = num_classes # 类别数，默认6（COCO子集或自定义数据集）
        self.max_dets = max_dets # 最大检测框数，默认300

        lib_path = Path(__file__).resolve().parent / "libgpu_ops.so"
        if not lib_path.exists():
            raise FileNotFoundError(
                f"{lib_path} 不存在，请先运行 kernels/compile.sh 编译。")
        self.lib = ctypes.CDLL(str(lib_path)) # ctypes.CDLL() 加载共享库，返回一个可调用其中导出函数的对象

        # ---- 函数签名: gpu_letterbox 签名-----------
        # 以下对应gpu_ops.cu中gpu_letterbox函数的返回值和参数的类型。
        self.lib.gpu_letterbox.restype = None      # 返回值类型：void
        self.lib.gpu_letterbox.argtypes = [        # 参数类型列表（必须严格匹配C端）
            ctypes.c_void_p,                       # d_src  源图显存指针 (uint8*)
            ctypes.c_int, ctypes.c_int,            # src_h, src_w  原图高宽
            ctypes.c_int, ctypes.c_int,            # nw, nh  缩放后的目标尺寸
            ctypes.c_void_p, ctypes.c_int,         # d_dst, dst_size  目标显存指针 + 边长(640)
            ctypes.c_int, ctypes.c_int,            # pad_x, pad_y  letterbox 填充偏移
            ctypes.c_void_p,                       # stream  CUDA stream 句柄
        ]
        # ---- gpu_postprocess 签名 ----
        self.lib.gpu_postprocess.restype = ctypes.c_int # 返回实际检测框数量
        self.lib.gpu_postprocess.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,   # d_out0, N, n_classes
            ctypes.c_float, ctypes.c_float,                # conf_thresh, iou_thresh
            ctypes.c_float, ctypes.c_int, ctypes.c_int,    # scale, pad_x, pad_y
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # d_dets, d_confs, d_count
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # d_order, d_final, d_final_count
            ctypes.c_int, ctypes.c_void_p,                 # max_dets, stream
        ]
        """
        为什么必须声明 argtypes？Python 是动态类型语言，而 C 是静态类型语言。当你用 ctypes 调用 .so 里的函数时：
        ctypes 默认假设所有参数都是 c_int（32位整数），返回值也是 c_intctypes 。如果不声明：
        64位指针被截断为32位 → segfaultfloat 被当作 int 解释 → 结果全错;参数个数不匹配 → 栈破坏
        声明后 ctypes 会自动做类型检查和转换，相当于给 C 函数加了 Python 侧的类型注解。
        类型对照表：
ctypes 类型	      C 类型	      大小(64位)	            说明
c_void_p	void*, uint8_t*, float*, cudaStream_t	8 bytes	   通用指针/句柄
c_int	       int	              4 bytes	                   32位有符号整数
c_float	       float	          4 bytes	                 IEEE754单精度浮点
None (restype)	  void	           -	                     函数无返回值
注意：cudaStream_t 本质上是 CUstream_st*（一个不透明指针），所以用 c_void_p 接收。
        """

        # ---- 后处理 scratch 显存（一次性分配复用）每次推理复用----
        self._d_dets = cuda_malloc(max_dets * 6 * 4) # 临时bbox: 300×6×float32 = 7.2KB,NMS 前的候选框暂存（conf过滤后）
        self._d_confs = cuda_malloc(max_dets * 4)    # 置信度:   300×float32  = 1.2KB
        self._d_count = cuda_malloc(4)               # NMS输出计数: 1×int32
        self._d_order = cuda_malloc(max_dets * 4)    # 排序索引:  300×int32  = 1.2KB
        self._d_final = cuda_malloc(max_dets * 6 * 4) # 最终结果:  300×6×f32  = 7.2KB
        self._d_final_count = cuda_malloc(4)         # 最终计数:  1×int32

        # ---- D2H 输出缓冲（pinned，加速拷贝）----
        # 作为 D2H 拷贝的目标缓冲:注：这里用的是普通 np.empty，严格来说应该用 pinned memory（cudaMallocHost）才能真正异步拷贝加速。
        # 如果 cuda_memcpy_d2h 内部已处理则无碍
        self.h_final = np.empty(max_dets * 6, dtype=np.float32) # 在 CPU 内存 上分配的一个 numpy 数组，作为 D2H（Device to Host）拷贝的目标缓冲区

    # ------------------------------------------------------------------
    def preprocess(self, d_raw, h: int, w: int, d_input,
                   scale: float, pad_x: int, pad_y: int, stream=0):
        """d_raw(BGR uint8 HWC, 显存) -> d_input(RGB float CHW 640x640, 显存)。
        参数说明
        参数	   含义
        d_raw	  原始 BGR UINT8 图片的设备指针（已由解码器/采集卡直接写入显存）
        h, w	  原图尺寸
        d_input	  模型输入 tensor 的设备指针（FLOAT CHW 640×640）
        scale	  等比缩放因子 = min(640/w, 640/h)
        pad_x, pad_y	letterbox 居中填充的像素偏移量
        stream	        CUDA stream，支持与 TRT engine 执行重叠
GPU Kernel 内部做了什么? 一个 kernel 同时完成四件事（避免多次 launch + 全局内存往返）：
Resize：双线性插值从 (h,w) → (nh,nw);Pad：填充灰色(114)到 640×640
BGR→RGB：通道交换;UINT8→FLOAT32 ÷255：归一化 + 类型转换 + HWC→CHW 转置
 核心价值：传统 CPU 预处理需要 ~2-5ms + 一次 H2D 拷贝；GPU 融合 kernel 仅需 ~0.1ms 且零拷贝。
        """
        nw, nh = int(w * scale), int(h * scale)
        self.lib.gpu_letterbox(
            device_addr(d_raw), h, w, nw, nh, device_addr(d_input), 640,
            pad_x, pad_y, ctypes.c_void_p(stream))

    def postprocess(self, d_output, n_anchors: int,
                    conf_thresh: float, iou_thresh: float,
                    scale: float, pad_x: int, pad_y: int, stream=0):
        """d_output(engine 输出 output0, 显存) -> list[tuple]，D2H 已在内完成。"""
        # 调用 GPU 后处理
        n = self.lib.gpu_postprocess(
            device_addr(d_output),  # YOLOv8 engine 的输出 tensor 指针 (shape: [1, 84, 8400]
            n_anchors,              # anchor 数量 = 8400
            self.num_classes,       # 类别数 = 6
            conf_thresh,            # 置信度阈值，如 0.25
            iou_thresh,             # NMS IoU 阈值，如 0.45
            scale,                  # letterbox 缩放因子，如 0.8
            pad_x, pad_y,           # letterbox 填充偏移，如 (64, 0)
            device_addr(self._d_dets), device_addr(self._d_confs), # 临时bbox暂存，临时置信度暂存
            device_addr(self._d_count), device_addr(self._d_order), # conf filter 输出计数，argsort/NMS 排序索引
            device_addr(self._d_final), device_addr(self._d_final_count), # 最终检测结果，最终有效框数量
            self.max_dets, ctypes.c_void_p(stream) # 最大检测框数上限 = 300，CUDA stream 句柄（stream: 支持与 TRT engine 执行异步/重叠）
            )
        
        # 选择性 D2H 拷贝
        if n == 0:
            return []
        cuda_memcpy_d2h(self.h_final[:n * 6], self._d_final)  # 只拷有效部分
        # 转为 Python 结果
        arr = self.h_final[:n * 6].reshape(n, 6) # reshape 为 (n, 6) 矩阵，每行 [x1, y1, x2, y2, conf, cls]
        # 以下列表推导式是 O(n) 的 Python 循环。当 n 接近 300 时可能成为瓶颈。如果下游只需要 numpy 数组，可以直接返回 arr 避免此开销。
        return [(float(x1), float(y1), float(x2), float(y2), float(conf), int(cls))
                for x1, y1, x2, y2, conf, cls in arr]

    def __del__(self):  # 析构释放显存
        """__del__ 的局限性：Python 不保证 __del__ 何时调用（甚至不调用）。生产环境建议显式提供 close() / release() 方法，
        并用 context manager (__enter__/__exit__) 管理生命周期。"""
        for buf in (self._d_dets, self._d_confs, self._d_count,
                    self._d_order, self._d_final, self._d_final_count):
            try:
                cuda_free(buf)
            except Exception:
                pass
