# TRT_inferce — 把 YOLOv8 预处理/后处理搬到 GPU

对比 `trt_server/server_trtapi.py`：原来 letterbox 缩放、BGR→RGB、归一化、decode、NMS
全部在 CPU 用 OpenCV/NumPy 完成；本目录把这两段搬到 GPU，宿主只做
**JPEG 解码 → 一次 H2D(原始 uint8) → execute_v2 → 一次小 D2H(最终框)**。

## 方案对比（针对本机环境实测）

| 方案 | 预处理 | 后处理 NMS | 运行时依赖 | 结论 |
|---|---|---|---|---|
| ① 现状 CPU | cv2+NumPy | NumPy+cv2.NMS | 轻 | 基线，全分辨率 resize 在 CPU 最慢 |
| ② OpenCV CUDA | `cv2.cuda` | 无 | 需重编 OpenCV | ❌ 本机 cv2 无 CUDA |
| ③ PyTorch GPU | torch | torchvision.nms | +torch | ❌ 违背零 PyTorch 部署初衷 |
| ④ **自定义 CUDA kernel** | 融合 kernel | 融合 kernel+thrust | nvcc 编译 `.so` | ✅ 保留现有 engine |
| ⑤ **端到端 engine** | 烧进 ONNX | `NonMaxSuppression` 烧进 ONNX | 仅 TRT | ✅ 单 engine，理论最优 |

本机关键事实（已探测）：
- GPU：RTX 4080（sm_89），`nvcc` 在 `/usr/local/cuda-12.8/bin/nvcc`
- OpenCV 4.10 **无 CUDA**，无 cupy/pycuda，**无 TRT NMS 插件**（`get_plugin_registry` 里没有 EfficientNMS）
- TRT 10.3 的 ONNX parser **支持 `NonMaxSuppression`**（已验证）
- 静态模型 `best_PCB.onnx`：`[1,3,640,640] → [1,10,8400]`

④ 与 ⑤ 性能相当（瓶颈都是 GPU 上的全分辨率 resize；相差仅 ~2 次 kernel launch ≈ 10µs）。
**能重编 engine 选 ⑤**（运行时最干净、理论最低延迟）；**要保留已调优的 INT8/FP16 engine 选 ④**。

## 目录结构

```
TRT_inferce/
├── common.py                    # cudart ctypes 封装 + CPU 参考实现（对比基准）
├── build_engine.py              # 从静态 ONNX 构建 FP16 engine（④ 用）
├── test_correctness.py          # GPU kernel 单元测试（预处理/后处理 vs CPU 参考）
├── compare_kernel.py            # ④ CPU/GPU 全链路输出 + 耗时对比（真实图+真实 engine）
├── compare_e2e.py               # ⑤ 端到端 ONNX vs CPU 参考输出对比
├── kernels/                     # ④ 自定义 CUDA kernel 方案
│   ├── gpu_ops.cu               #   融合 letterbox/归一化 + decode/NMS kernel
│   ├── compile.sh               #   nvcc -> libgpu_ops.so
│   ├── gpu_ops.py               #   ctypes 封装
│   └── server_gpu.py            #   服务器（保留现有 engine）
└── e2e/                         # ⑤ 端到端 engine 方案
    ├── graphsurgeon_e2e.py      #   ONNX 图手术：预处理+后处理烧进模型
    ├── build_engine.py          #   动态输入 engine 构建
    └── server_e2e.py            #   服务器（原始 uint8 直出最终框）
```

## 方案 ④：自定义 CUDA kernel（保留现有 engine）

### 原理
- **预处理 kernel**：每个输出像素一个线程，letterbox(bilinear, half-pixel) + 灰度 pad(114)
  + BGR→RGB + /255 + HWC→CHW 全部融合，直接写进 engine 的 `d_input`。采样映射与
  `cv2.resize(INTER_LINEAR)` 逐位对齐（用真实比值 `src_w/nw`，不是 `1/scale`）。
- **后处理**：`decode_kernel`(8400 个 anchor 并行 decode+阈值+反 letterbox) →
  `thrust::sort_by_key` 按 conf 降序 → `nms_greedy_kernel`(单线程贪心 NMS)。
- 宿主只 `cv2.imread` + H2D 原始 uint8 + `execute_v2` + D2H 少量最终框。

### 使用
```bash
# 1. 编译 kernel
cd kernels && bash compile.sh        # 生成 libgpu_ops.so

# 2. 构建 engine（静态模型 -> FP16，约 6 分钟）
cd .. && python build_engine.py --onnx /root/my_FILE/models/best_PCB.onnx \
                                --engine /root/my_FILE/models/yolov8_fp16.engine

# 3. 推理
python kernels/server_gpu.py /root/my_FILE/my_FILE/test_images/
```

## 方案 ⑤：端到端 engine（单 engine，理论最优）

### 原理
用 `onnx_graphsurgeon`（本实现用原生 onnx.helper）改写计算图：
- **输入**改为 `image_raw (1,H,W,3) uint8`（动态 H/W），图内做 letterbox+归一化（复用
  Resize/Pad/Transpose/Gather/Mul）。
- **输出**改为 `detections (max_dets,6)` = `[x1,y1,x2,y2,conf,cls]`（原图坐标），图内做
  decode + `NonMaxSuppression` + 固定尺寸填充。
- 最终 `execute_v2` 一次跑完所有，无中间 host↔device 同步。

### 使用
```bash
# 1. 图手术生成端到端 ONNX（并可用 onnxruntime 校验）
cd e2e && python graphsurgeon_e2e.py --check

# 2. 构建动态输入 engine（约 6 分钟）
python build_engine.py --onnx /root/my_FILE/models/best_PCB_e2e.onnx \
                       --engine /root/my_FILE/models/yolov8_e2e_fp16.engine

# 3. 推理
python server_e2e.py /root/my_FILE/my_FILE/test_images/
```

## 正确性验证结果

- **方案 ④**：`python test_correctness.py` —— 预处理 max_abs ≤ 0.003（~1 LSB 内）；
  后处理 decode+NMS 与 CPU 参考框集合一致。
  `python compare_kernel.py` —— 真实图 + 真实 FP16 engine，**5/5 张图框坐标+类别逐值相等**；
  耗时 GPU 管线 1.21 ms vs CPU 管线 2.93 ms（**加速 2.4×**，640×640 图；图越大收益越大）。
- **方案 ⑤**：`python compare_e2e.py` —— onnxruntime 跑 `best_PCB_e2e.onnx`，真实图 **5/5 一致**。

## 进一步优化（可选）

1. **JPEG 解码也上 GPU**：用 NVIDIA nvjpeg（`nvDecode`）把 `cv2.imread` 换掉，
   这是当前唯一仍在 CPU 的重活；需额外链 `libnvjpeg`。
2. **批量 + CUDA stream 流水线**：多图时用多 stream 让 decode/preprocess 与上一个 batch
   的 execute 重叠，进一步提高吞吐（当前是单帧串行）。
3. **INT8 标定**：对端到端 ONNX 直接做 ModelOpt PTQ，把 FP16 engine 换成 INT8，进一步降延迟。
