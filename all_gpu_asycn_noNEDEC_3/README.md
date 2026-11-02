# all_gpu_asycn_noNEDEC_3 —— 多解码/保存线程 + 三流 GPU + 各阶段耗时测试

从 `all_gpu_async_noNEDEC_2` 演进而来，本版新增：

1. **CPU 侧并行**：解码线程 × N + 保存线程 × N（可配置），消除「后处理 IO（jpeg 编码+写盘）」瓶颈；
2. **各阶段单线程耗时测试**：`bench_stages` 独立测量解码 / 各 GPU 阶段 / 编码的耗时，定位瓶颈（通用计时工具见 `include/perf.h`）。

```
文件夹图片 → libjpeg 直写锁页内存 → 异步 H2D → NPP letterbox resize
           → e2e engine(已含归一化+NMS) → 画框 → NPP 恢复原尺寸 → 异步 D2H → 存图
```

## 1. 线程模型

```
解码线程 × num_decode_threads (CPU, libjpeg 解码直写锁页, 共享文件游标)
        │ ready_q (槽池, mutex+condvar)
GPU 线程 × 1 (三流流水: Stream_Pre / Stream_Infer / Stream_Post)
        │ save_q
保存线程 × num_save_threads (CPU, 等 event → libjpeg 编码 → 写盘 → 归还槽)
```

- 槽池深度 `num_buffers` 默认 12，且运行期自动兜底：`num_buffers ≥ 解码×N + 保存×N + 3(级GPU)`；
- 收尾顺序：join 解码线程 → 置 `done_loading` → join GPU → 置 `done_gpu` → join 保存线程。

## 2. 参数配置

优先级：`命令行 > 配置文件(--config) > 代码默认值`（代码默认值见 `include/config.h`）。

| 键 | CLI | 默认 | 说明 |
|---|---|---|---|
| `engine_path` | `--engine` | e2e engine | 模型 |
| `images_dir` | `--images` | test | 输入目录 |
| `output_dir` | `--output` | ./images_results | 输出目录 |
| `conf` / `iou` | `--conf` / `--iou` | 0.45 / 0.65 | 阈值 |
| `num_buffers` | `--num-buffers` | 12 | 槽深度 |
| `num_streams` | `--num-streams` | 3 | 1 单流 / 3 三流 |
| `num_decode_threads` | `--num-decode-threads` | **3** | 解码线程数 |
| `num_save_threads` | `--num-save-threads` | **3** | 保存线程数 |
| `draw_thickness` | `--thickness` | 2 | 线宽 |
| `print_dets` | `--print-dets` | false | 打印检测框 |

```bash
./image_infer                                   # 全默认(3 解码 + 3 保存 + 3 流)
./image_infer --config config.ini               # 读配置
./image_infer --num-decode-threads 6 --num-save-threads 6
```

## 3. 各阶段耗时测试（bench_stages）

```bash
./bench_stages [--images <dir>] [--engine <path>] [--iters <n>] [--conf f] [--iou f]
```

复用正式流水线组件（jpeg_loader / npp_ops / tensorrt_e2e / draw），单线程/单流孤立测各阶段，
打印每个阶段平均耗时。实测（test_scaled，样本 960×960）：

```
CPU 解码(jpeg→锁页)     1.844 ms/img
H2D(锁页→显存)          0.227 ms/img
NPP letterbox resize     0.011 ms/img
TRT 推理(e2e)            1.076 ms/img   ← GPU 主耗时
画框 kernel              0.013 ms/img
NPP 恢复原尺寸           0.011 ms/img
D2H(显存→锁页)          0.211 ms/img
CPU 编码(BGR→jpeg)       2.571 ms/img   ← 单图最慢（瓶颈）
```

通用计时工具：[include/perf.h](include/perf.h)（`CpuTimer` + `GpuTimer`），任意 CUDA 工程可复用。

## 4. 测试结果（500 张，test_scaled 混合 960/480）

| 配置 | 吞吐 |
|---|---|
| 1 解码 + 1 保存（旧） | ~622 img/s |
| **3 解码 + 3 保存（默认）** | **~713 img/s（+15%）** |
| 6 解码 + 6 保存 | ~727 img/s（收益递减，已接近 GPU 上限） |

结论：编码(2.57ms)/解码(1.84ms) 是原瓶颈，并行化后 CPU 不再卡，吞吐逼近 GPU 推理上限(~1.08ms)；
继续加线程收益递减，受 GPU 推理与 H2D/D2H 限制。检测结果与参考实现一致，输出 500 张带框 JPEG。

## 5. 构建

```bash
bash build.sh      # 产物 ./image_infer 与 ./bench_stages
```

依赖：CUDA 12.8 + NPP 12.3 + TensorRT 10.8 + libjpeg-turbo8。

## 6. 目录结构

```
├── build.sh / config.ini / README.md
├── include/  common.h · config.h · perf.h · jpeg_loader.h · npp_ops.h · tensorrt_e2e.h · draw.h · pipeline.h
├── src/      config.cpp · jpeg_loader.cpp · npp_ops.cpp · tensorrt_e2e.cpp · pipeline.cpp · main.cpp
├── kernels/  draw.cu
└── bench/    bench_stages.cpp    # 各阶段耗时测试
```
