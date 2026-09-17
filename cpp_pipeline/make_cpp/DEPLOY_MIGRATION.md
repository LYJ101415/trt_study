# DEPLOY_MIGRATION.md — C++ 推理程序迁移部署指南

> 适用对象: `cpp_pipeline/infer_e2e_b8_cpp` (e2e INT8 端到端 PCB 缺陷检测, batch=8 四级流水线)
> 基线机器: Ubuntu 22.04.5 / RTX 4080 (SM 8.9) / 实测 1185 img/s (500 张, 3238 框)
> 文档日期: 2026-09-02
>
> 迁移的本质只有三件事: **① 带上可执行程序; ② 带上/装上运行库; ③ engine 与目标 GPU 架构匹配。**

---

## 一、目录内容总览

```
cpp_pipeline/
├── main.cpp              C++ 源码 (改逻辑才需要)
├── build.sh              编译脚本 (仅编译机需要)
├── trt_include/          TensorRT 10.3.0 头文件 (仅编译机需要; 源机 tar 包 include 断链,
│                         已从 NVIDIA 官方仓库 v10.3.0 tag 补齐, 与本机 libnvinfer 同 ABI)
├── collect_deps.sh       依赖收集脚本 (自动生成自包含部署包)
├── deploy/               ✅ 自包含部署包 (已生成, 即拿即迁)
│   ├── infer_e2e_b8_cpp  可执行文件
│   ├── lib/              129 个依赖库 (不含 glibc 核心 / libcuda)
│   ├── engine/yolov8_int8_e2e_dyn.engine
│   └── run.sh            运行入口 (自动设 LD_LIBRARY_PATH)
└── deploy_pkg.tar.gz     部署包压缩档 (168 MB)
```

---

## 二、依赖清单（编译机实测版本）

### 2.1 硬件层

| 组件 | 版本/要求 | 备注 |
|---|---|---|
| GPU | **NVIDIA RTX 40 系（SM 8.9）** | 基线: RTX 4080。engine 与 SM 架构绑定，跨架构必须重新 build engine |
| 显存 | ≥ 1 GB 空闲 | 引擎+上下文+槽位缓冲合计 < 1 GB |
| 驱动 | 基线 **580.105.08**；最低要求支持 CUDA 12.x（驱动 ≥ 545） | `nvidia-smi` 能出结果即可；libcuda 由驱动提供，**不随包分发** |
| PCIe | 常规 x16 即可 | 持续 H2D 流量 ~1 GB/s，远低于 PCIe 3.0 x16 上限 |

### 2.2 系统层

| 组件 | 基线版本 | 部署机要求 |
|---|---|---|
| OS | Ubuntu 22.04.5 LTS | 二进制兼容：glibc ≥ 2.35（即 Ubuntu 22.04+ / Debian 12+） |
| 内核 | 5.15 | ≥ 5.15 即可 |
| glibc | **2.35** | **目标机版本必须 ≥ 2.35，且不要用包里的 libc**（collect 脚本已排除） |
| CPU | 有效 ≥ 4 核 | 流水线线程数 = workers+consumers+3；注意云容器 cgroup 配额 |

### 2.3 运行库层（deploy/lib/ 已随包携带 129 个）

| 库 | 基线版本 | 来源 | 部署方式 |
|---|---|---|---|
| libnvinfer.so.10 | **TensorRT 10.3.0.26** | TensorRT 官方 tar 包 | ✅ 随包 |
| libcudart.so.12 | **CUDA 12.8.90** | /usr/local/cuda | ✅ 随包 |
| libopencv_{core,imgcodecs,imgproc}.so.4.5d | **4.5.4+dfsg-9ubuntu4** | Ubuntu 22.04 apt | ✅ 随包（含其 gdal 等间接依赖共 129 个） |
| libstdc++.so.6 / libgcc_s.so.1 | GCC 11.4.0 | 编译机 | ✅ 随包 |
| 其余间接依赖 | libjpeg8 / libpng16 / libtiff5 / libwebp7 / libzstd / liblzma 等 | 系统 apt | ✅ 随包 |
| **libcuda.so.1** | 580.105.08 | **NVIDIA 驱动** | ❌ 不随包，目标机装驱动自带 |
| **glibc 核心库** | 2.35 | **目标机 OS** | ❌ 不随包，携带会破坏系统一致性 |

### 2.4 编译层（仅当需要在编译机重新 build 时）

| 组件 | 版本 |
|---|---|
| g++ / gcc | 11.4.0-1ubuntu1~22.04 |
| CUDA toolkit (nvcc + 头文件) | /usr/local/cuda → CUDA 12.8 |
| TensorRT 头文件 | trt_include/（v10.3.0） |
| OpenCV 开发头 | libopencv-dev 4.5.4+dfsg-9ubuntu4 |

### 2.5 明确不需要的

- ❌ Python / conda / PyTorch / onnxruntime / TensorRT python 绑定
- ❌ cuDNN（推理路径未用到）
- ❌ INT8 校准数据（scale 已烧进 engine）

---

## 三、迁移步骤

### 方式 A：自包含部署包（推荐，目标机可完全离线）

```bash
# ---------- 源机: 打包 ----------
cd /root/my_FILE/my_trt_FILE/TRT_inferce/cpp_pipeline
./collect_deps.sh infer_e2e_b8_cpp ./deploy /root/my_FILE/models/yolov8_int8_e2e_dyn.engine
tar czf deploy_pkg.tar.gz deploy          # 已生成: 168 MB

# 传输到目标机 (scp / U盘 / 内网均可)
scp deploy_pkg.tar.gz user@目标机:/opt/pcb/

# ---------- 目标机: 解包 ----------
cd /opt/pcb && tar xzf deploy_pkg.tar.gz

# 校验 (见第四节)
./deploy/run.sh --source /path/to/测试图片目录 \
                --engine ./deploy/engine/yolov8_int8_e2e_dyn.engine \
                --save-txt --out ./check_result
```

要点: `run.sh` 已自动把包内 `lib/` 放到 `LD_LIBRARY_PATH` 最前（优先级高于编译时 rpath），
所以包内库生效、不污染系统。**目标机唯一必装的东西只有 NVIDIA 驱动。**

### 方式 B：目标机 apt 装库（Ubuntu 22.04 专用）

```bash
# 目标机: 装 OpenCV 运行库 (其余库随包或 apt)
sudo apt update
sudo apt install -y libopencv-core4.5d libopencv-imgcodecs4.5d libopencv-imgproc4.5d
# 拷贝: infer_e2e_b8_cpp + libnvinfer.so.10 + libcudart.so.12 (或装 TensorRT 10.3 deb)
export LD_LIBRARY_PATH=/opt/pcb/lib:$LD_LIBRARY_PATH
./infer_e2e_b8_cpp --source ... --engine ...
```

### 方式 C：Docker（多机批量部署首选）

```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core4.5d libopencv-imgcodecs4.5d libopencv-imgproc4.5d libgdal30 \
 && rm -rf /var/lib/apt/lists/*
COPY deploy/ /opt/pcb/
ENTRYPOINT ["/opt/pcb/run.sh"]
CMD ["--engine", "/opt/pcb/engine/yolov8_int8_e2e_dyn.engine"]
```

```bash
docker build -t pcb-infer:1.0 .
docker run --gpus all -v /data:/data pcb-infer:1.0 --source /data/inbox --save-txt --out /data/results
```

### 换 GPU 架构时（如 30 系 / A100 / L4）

engine 不能直接用，必须重新构建（在目标架构的卡上、用同一份量化 ONNX）：

```bash
trtexec --onnx=yolov8_int8_e2e_dyn.onnx --saveEngine=新卡.engine \
        --fp16 --memPoolSize=workspace:4096 \
        --minShapes=image_raw:1x128x128x3 \
        --optShapes=image_raw:8x640x640x3 \
        --maxShapes=image_raw:8x1920x1920x3
```

再把新 engine 放进部署包 `engine/`，程序代码零改动。

---

## 四、迁移后校验（5 步，按序执行）

```bash
# 1. 驱动与 GPU 可见
nvidia-smi

# 2. 库完整性 (无输出 = 通过)
LD_LIBRARY_PATH=./deploy/lib ldd deploy/infer_e2e_b8_cpp | grep "not found"

# 3. 程序能加载 (打印用法 = 通过)
./deploy/run.sh

# 4. 推理跑通 + 框数对基线
./deploy/run.sh --source 测试集目录 --engine deploy/engine/yolov8_int8_e2e_dyn.engine
#    ✅ 验收基线: 500 张 → 3238 框 (与源机逐张一致)

# 5. 性能验收 (同硬件应 ≥ 1100 img/s; 不同硬件以 GPU 节拍为准)
```

---

## 五、常见故障排查

| 现象 | 原因 | 处置 |
|---|---|---|
| `error while loading shared libraries: libnvinfer.so.10` | 没用 `run.sh`，LD_LIBRARY_PATH 缺失 | 用 `./run.sh` 或 `export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH` |
| `version 'GLIBC_2.32' not found` | 目标机 glibc < 2.35 (老系统) | 换 Ubuntu 22.04+ / Debian 12+，或在老系统重新编译 |
| `no kernel image is available for execution on the device` | engine 与 GPU 架构不匹配 | 用目标架构的卡重新 build engine（见第三节末尾） |
| `CUDA driver version is insufficient` | 驱动太老 | 升级 NVIDIA 驱动到 ≥ 545 (建议 550+) |
| `Cannot Parse... engine` / deserialize 失败 | TRT 库版本与 build engine 的版本不一致 | 换回 10.3.0 的 libnvinfer，或重新 build engine |
| 吞吐明显低于基线 | 解码线程不足 / NUMA 跨节点 / CPU 配额被限 | 加 `--workers`；`taskset -c <GPU亲和核>` 绑核；查 cgroup 配额 |
| 结果文件里框数为 0 | 阈值设错或 engine 输入名不符 | 核对 `--conf/--iou`；`polygraphy inspect model engine --show layers` 查 IO 名称 |

---

## 六、基线数据（验收对照用）

| 项 | 值 |
|---|---|
| 测试集 | DeepPCB test 500 张 (640×640) |
| 检测框总数 | **3238**（与 Python 版、原版逐张一致） |
| 吞吐 | **1185 img/s**（batch=8, workers=7, consumers=2） |
| 参照 | Python b8 流水线 830 img/s; trtexec 纯 GPU 上限 1560 img/s |
