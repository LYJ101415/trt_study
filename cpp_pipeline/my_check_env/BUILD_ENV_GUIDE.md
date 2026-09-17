# BUILD_ENV_GUIDE.md — C++ 全链条（编译→打包→部署）环境配置指南

> 回答的问题: `install_env.sh` 只覆盖 Python 推理链。要让"Python 项目 → C++ 部署"
> 这条链能走通，编译机/部署机各自需要装什么、怎么装、怎么验证。
> 配套文件: `check_build_env.sh`（一键审计）、`DEPLOY_MIGRATION.md`（迁移细节）、
> `collect_deps.sh`（打包工具）
> 文档日期: 2026-09-02

---

## 一、结论与本机现状

`install_env.sh` 全部是 pip 安装（torch / tensorrt-cu12 / onnxruntime / opencv-python /
modelopt / polygraphy），是**纯 Python 推理链**，不含任何 C++ 编译要素。

本机（源机）实际 C++ 编译要素的审计结果（`./check_build_env.sh`，12 ✅ / 0 ❌）:

| 编译要素 | 本机状态 | 来源 |
|---|---|---|
| build-essential (g++ 11.4 / make / cmake) | ✅ 有 | **系统预装**（非 install_env.sh） |
| NVIDIA 驱动 580.105.08 + RTX 4080 (SM 8.9) | ✅ 有 | 宿主机/手动装 |
| CUDA toolkit 12.8（nvcc + cuda_runtime.h + libcudart） | ✅ 有 | /usr/local/cuda（runfile 单独装，非 pip） |
| OpenCV C++ 开发库 4.5.4（libopencv-dev） | ✅ 有 | apt（非 pip 的 opencv-python！） |
| TensorRT 10.3.0.26 库 + trtexec | ✅ 有 | /root/my_FILE/TensorRT-10.3.0.26 tar 包 |
| **TensorRT C++ 头文件** | ⚠️ tar 包 include 断链，已用 GitHub v10.3.0 头补齐 → `trt_include/` | **曾缺失的关键一环** |
| cgroup CPU 配额 | ⚠️ 12 核（nproc 96 是虚的） | 云容器限制 |

**结论**: 这台机器能编译纯属"系统预装+单独装过 CUDA/OpenCV 的运气"，`install_env.sh`
本没规划这条链。换新编译机按第三节从零走一遍即可复现。

---

## 二、两种机器角色，两份环境清单

```
┌─ 编译机 (开发/编译/打包) ────────────┐   ┌─ 部署机 (只运行) ──────────┐
│ ① 系统工具链 build-essential         │   │ ① NVIDIA 驱动              │
│ ② NVIDIA 驱动 + GPU(与engine同架构)  │   │ ② GPU 架构与 engine 匹配   │
│ ③ CUDA toolkit (nvcc+头文件+libcudart)│   │ ③ glibc ≥ 编译机版本      │
│ ④ libopencv-dev (C++ 版 OpenCV!)     │   │  (运行库全在 deploy/lib)   │
│ ⑤ TensorRT tar 包 + C++ 头文件       │   │  不需要: 编译器/CUDA toolkit│
│ ⑥ 功能库按需 (相机SDK/PLC/curl)      │   │           /OpenCV/TRT 头   │
│ ⑦ Python 推理链 (install_env.sh,     │   │                            │
│     用于量化/build engine, 见注)     │   │                            │
└──────────────────────────────────────┘   └────────────────────────────┘
```

> 注: Python 链（install_env.sh）在编译机上仍要保留——**engine 的构建（trtexec 或
> ModelOpt 量化）在编译机/有同架构 GPU 的机器上做**，这是 Python 链唯一参与 C++ 部署的环节。

---

## 三、编译机环境安装（Ubuntu 22.04，从零到能编译）

### ① 系统工具链

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config
```

### ② NVIDIA 驱动（驱动决定能用多新的 CUDA）

```bash
ubuntu-drivers devices                 # 看推荐
sudo ubuntu-drivers autoinstall        # 或手动装 runfile
sudo reboot && nvidia-smi              # 验证
```

### ③ CUDA toolkit（注意: pip 的 torch/tensorrt 自带 CUDA 运行库 ≠ 这套头文件）

```bash
# 到 developer.nvidia.com/cuda-downloads 下载 12.x runfile, 然后只装 toolkit:
sudo sh cuda_12.8.0_570.xx_linux.run --toolkit --silent
echo 'export PATH=/usr/local/cuda/bin:$PATH'      >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
# 验证
nvcc --version && ls /usr/local/cuda/include/cuda_runtime.h
```

### ④ OpenCV C++ 开发库（apt 版与 Ubuntu 22.04 最省事）

```bash
sudo apt install -y libopencv-dev      # 4.5.4, 头文件+库全齐
# 验证
pkg-config --modversion opencv4
```

### ⑤ TensorRT 10.3 C++（tar 包 + 头文件，本项目踩过的坑）

```bash
# 5.1 下载 TensorRT-10.3.0.26 tar 包 (developer.nvidia.com/tensorrt-download), 解压到 /opt 或 /root
#     → bin/trtexec + lib/libnvinfer.so.10

# 5.2 ⚠️ 头文件: 检查 include 是否完好, 断链则从官方仓库补 (8 个文件)
ls TensorRT-10.3.0.26/include/NvInfer.h 2>/dev/null \
|| mkdir -p trt_include && cd trt_include \
&& for h in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferRuntime.h \
            NvInferRuntimeBase.h NvInferRuntimeCommon.h NvInferRuntimePlugin.h NvInferVersion.h; do
      curl -sL -o $h https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.3.0/include/$h
   done

# 5.3 库路径注册 (二选一):
echo "/path/to/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib" | sudo tee /etc/ld.so.conf.d/trt.conf
sudo ldconfig
#    或不打 ldconfig, 编译时 -Wl,-rpath 指定 (本项目 build.sh 采用此法)
```

### ⑥ 功能库按需加（相机/PLC/通信等，扩展全链条时逐个补）

| 功能 | 常用库 | 安装 |
|---|---|---|
| 网络请求/HTTP 上报 | libcurl | `sudo apt install -y libcurl4-openssl-dev` |
| Modbus/TCP | libmodbus | `sudo apt install -y libmodbus-dev` |
| Siemens PLC (S7) | snap7 | 源码编译或 apt: `libsnap7-dev` |
| 海康/Basler 相机 | MVS SDK / pylon | 厂商官网 tar 包 + 官方安装脚本 |
| RTSP 拉流 | OpenCV(FFMPEG) 或 live555 | apt 版 opencv 已带 FFMPEG 后端 |
| MySQL/PG 结果入库 | libpq / libmysqlclient | `sudo apt install -y libpq-dev` |

### ⑦ 验证编译环境

```bash
./check_build_env.sh          # 一键审计, 0 ❌ 才算齐
cd cpp_pipeline && bash build.sh   # 全链编译本项目 (通过 = 环境可用)
./infer_e2e_b8_cpp --source 测试图目录 --engine xxx.engine   # 跑通
```

---

## 四、build.sh 扩展模板（加相机/PLC 环节时照此改）

```bash
g++ -O3 -std=c++17 main.cpp camera_grab.cpp plc_client.cpp -o infer_e2e_b8_cpp \
    -I./trt_include -I"$TRT_ROOT/include" -I"$CUDA_ROOT/include" -I/usr/include/opencv4 \
    -I/opt/MVS/include                        # ← 相机 SDK 头文件
    -L"$TRT_ROOT/lib" -L"$CUDA_ROOT/lib64" \
    -lnvinfer -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_imgproc \
    -lMvCameraControl                         # ← 相机 SDK 库
    -lsnap7                                   # ← PLC 库
    -lcurl                                    # ← HTTP 上报
    -lpthread -Wl,-rpath,"$TRT_ROOT/lib" -Wl,-rpath,'$ORIGIN/../lib'
# 收官: 重跑 collect_deps.sh —— 新库的依赖会被 ldd 自动扫进部署包
```

要点: 每加一个功能库 = 头文件路径 `-I` + 库名 `-l` 两行；打包交给 collect_deps.sh 自动化。

---

## 五、部署机清单（详见 DEPLOY_MIGRATION.md）

只需: **NVIDIA 驱动** + **同架构 GPU** + **glibc ≥ 2.35 的 64 位 Linux**。
运行库不装——用 `collect_deps.sh` 生成的 `deploy/` 包（内含 129 个库 + run.sh）。

---

## 六、常见坑（本项目实际踩过/绕过的）

| 坑 | 症状 | 根因与处置 |
|---|---|---|
| pip 的 CUDA ≠ 系统 CUDA | `import torch` 正常，但 nvcc/头文件找不到 | pip 包自带运行库不带编译器头文件；编译必须装 toolkit（③） |
| opencv-python ≠ libopencv-dev | `#include <opencv2/...>` 找不到 | pip 装的是 Python 绑定；C++ 要 apt 装 libopencv-dev（④） |
| TRT tar 包 include 断链 | 编译报 `NvInfer.h: No such file` | 下载/解压不完整；用 GitHub v10.3.0 tag 头补齐（⑤） |
| nproc 是虚的 | 96 核机器只给 12 核配额 | 云容器 cgroup；线程数按配额规划（check 脚本⑤项） |
| 部署机 glibc 老于编译机 | `GLIBC_2.32 not found` | 在低版本系统上重编译，或统一升级系统 |
| 直接拷二进制不带库 | `libnvinfer.so.10: cannot open` | 用 collect_deps.sh 打包 + run.sh 启动 |
| engine 换卡跑不了 | `no kernel image is available` | engine 绑 GPU 架构；新卡 trtexec 重 build |
