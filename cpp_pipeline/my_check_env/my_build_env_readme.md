# my_readme.md — 裸机 C++ 模型推理部署 编译依赖 完整安装手册

> 目的: 在一台**全新裸机**（无任何预装）上，从零把「Python 项目 → C++ 部署」这条链所需的
> 编译依赖全部装齐，装完后能直接 `g++` 编译本项目 `cpp_pipeline` 的推理二进制，并可用
> `collect_deps.sh` 打包成自包含部署包。
>
> 依据: [BUILD_ENV_GUIDE.md](BUILD_ENV_GUIDE.md)（本机 12 ✅/0 ❌ 的审计结论），版本与路径
> 与本项目实际对齐。
>
> 本文档分三部分：
> - **第一部分** —— 一键安装脚本（可直接存成 `install_build_env_cpp.sh` 运行）
> - **第二部分** —— 分步命令拆解 + 每步验证（脚本跑不动/要手动装时照此走）
> - **第三部分** —— 部署机最小清单 + 常见坑

---

## 0. 裸机前提

| 项 | 要求 |
|---|---|
| 操作系统 | Ubuntu 22.04 x86_64（或其它 glibc ≥ 2.35 的 64 位 Linux） |
| GPU | NVIDIA 独显，且**架构要与目标 engine 一致**（本项目 RTX 4080，SM 8.9）；换架构需重 build engine |
| 驱动 | 最终驱动版本 **≥ 570**（CUDA 12.8 的最低要求） |
| 权限 | root 或可 `sudo` 的用户 |
| 网络 | 能访问 apt 源、developer.nvidia.com、raw.githubusercontent.com |
| 磁盘 | 系统盘预留 ≥ 20G（CUDA + TensorRT + OpenCV + 编译产物） |

> 说明: 本文档只装「编译机」需要的东西。**部署机**（只运行）只需 NVIDIA 驱动 + 同架构 GPU，
> 见第三部分。

---

## 一、一键安装脚本 `install_build_env_cpp.sh`

把下面整段存成脚本执行。脚本做了幂等保护（每步先探测再装），可重复运行。

```bash
#!/bin/bash
# install_build_env_cpp.sh — 裸机 C++ 推理部署 编译依赖 一键安装 (Ubuntu 22.04 x86_64)
# 覆盖: 工具链 / NVIDIA驱动 / CUDA toolkit / OpenCV C++ / TensorRT + 头文件 / ldconfig 注册
# 用法: sudo bash install_build_env_cpp.sh
set -euo pipefail

# ===== 可调变量 (按需改版本/路径) =====
CUDA_RUNFILE="cuda_12.8.0_570.86.10_linux.run"   # 从 developer.nvidia.com 下载到的实际文件名
TRT_TARBALL="TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.8.tar.gz"  # 实际 tar 包名
TRT_DIR="TensorRT-10.3.0.26"                     # 解压出的目录名
INSTALL_BASE="/root/my_FILE"                     # 与项目 build.sh 的 TRT_ROOT 对齐
CUDA_ROOT="/usr/local/cuda"

log() { echo -e "\n\033[1;32m==> $*\033[0m"; }
need_root() { [ "$(id -u)" -eq 0 ] || { echo "请用 sudo 运行"; exit 1; }; }

# ---------------- ① 系统工具链 ----------------
log "① 系统工具链 build-essential / cmake / pkg-config"
apt update -y
apt install -y build-essential cmake pkg-config

# ---------------- ② NVIDIA 驱动 ----------------
log "② NVIDIA 驱动"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "已装驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    ubuntu-drivers devices || true
    ubuntu-drivers autoinstall || {
        echo "⚠️  ubuntu-drivers 自动装失败，请改用手动 runfile 或换驱动源"
        echo "   参考: https://www.nvidia.com/Download/index.aspx 选 .run 后:"
        echo "   sudo sh NVIDIA-Linux-x86_64-xxx.run --silent && sudo reboot"
    }
fi
# 驱动装完必须重启才生效，脚本在此提示但不强制重启
echo "💡 若本步刚装了驱动，请: sudo reboot 后重跑本脚本 (脚本会跳过已装项)"

# ---------------- ③ CUDA toolkit ----------------
log "③ CUDA toolkit (nvcc + 头文件 + libcudart)"
if [ -f "$CUDA_ROOT/include/cuda_runtime.h" ]; then
    echo "已装 CUDA: $(basename "$(readlink -f "$CUDA_ROOT")")"
else
    # 下载: https://developer.nvidia.com/cuda-downloads → Linux x86_64 → Ubuntu 22.04 → runfile(local)
    # 此处假设 runfile 已放到当前目录
    if [ ! -f "$CUDA_RUNFILE" ]; then
        echo "❌ 未找到 $CUDA_RUNFILE，请先到 developer.nvidia.com/cuda-downloads 下载 runfile(local)"
        echo "   放到本脚本同目录后重跑。仅下载 tool 选择 runfile 即可。"
        exit 1
    fi
    sh "$CUDA_RUNFILE" --toolkit --silent --override   # --toolkit: 只装 toolkit, 驱动已在②装
    # 环境变量
    grep -q '/usr/local/cuda/bin' ~/.bashrc || {
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    }
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi
# 注册 libcudart 到 ldconfig
echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
ldconfig

# ---------------- ④ OpenCV C++ 开发库 ----------------
log "④ OpenCV C++ (libopencv-dev, 头文件+库)"
dpkg -s libopencv-dev >/dev/null 2>&1 || apt install -y libopencv-dev

# ---------------- ⑤ TensorRT 10.3 C++ ----------------
log "⑤ TensorRT 10.3 tar 包 + C++ 头文件"
mkdir -p "$INSTALL_BASE"
if [ -f "$INSTALL_BASE/$TRT_DIR/targets/x86_64-linux-gnu/bin/trtexec" ]; then
    echo "已解压 TensorRT: $INSTALL_BASE/$TRT_DIR"
else
    if [ ! -f "$TRT_TARBALL" ]; then
        echo "❌ 未找到 $TRT_TARBALL"
        echo "   下载: https://developer.nvidia.com/tensorrt-download (需登录, 选 10.3.0.26 / Linux x86_64 / cuda 12.8)"
        echo "   放到本脚本同目录后重跑。"
        exit 1
    fi
    tar -xzf "$TRT_TARBALL" -C "$INSTALL_BASE"
fi
TRT_LIB="$INSTALL_BASE/$TRT_DIR/targets/x86_64-linux-gnu/lib"

# 5.1 ⚠️ 头文件补齐 (tar 包 include 可能断链, 本项目踩过的坑)
TRT_INC="$INSTALL_BASE/$TRT_DIR/include"
mkdir -p "$TRT_INC"
for h in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferRuntime.h \
         NvInferRuntimeBase.h NvInferRuntimeCommon.h NvInferRuntimePlugin.h NvInferVersion.h; do
    if [ ! -s "$TRT_INC/$h" ]; then
        echo "  补头文件: $h"
        curl -fsSL -o "$TRT_INC/$h" "https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.3.0/include/$h"
    fi
done

# 5.2 注册 TRT 库 (可选; 本项目 build.sh 用 -Wl,-rpath 也能链接, 此处注册便于系统级使用)
echo "$TRT_LIB" > /etc/ld.so.conf.d/trt.conf
ldconfig

# ---------------- ⑥ 功能库 (可选, 按需取消注释) ----------------
log "⑥ 功能库 (可选, 需要再装)"
# apt install -y libcurl4-openssl-dev   # HTTP 上报
# apt install -y libmodbus-dev          # Modbus/TCP
# apt install -y libsnap7-dev           # Siemens PLC (S7)
# apt install -y libpq-dev              # PostgreSQL 结果入库
# 相机 SDK (海康MVS / Basler pylon) 为厂商 tar 包, 走官方安装脚本, 不在此列出

# ---------------- 完成自检 ----------------
log "安装完成, 自检:"
echo "  g++       : $(g++ --version | head -1)"
echo "  cmake     : $(cmake --version | head -1 | awk '{print $NF}')"
echo "  nvidia    : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 || echo '未装(需重启)')"
echo "  nvcc      : $(nvcc --version | tail -1 | grep -oP 'release \S+' || echo '未找到')"
echo "  opencv    : $(pkg-config --modversion opencv4 2>/dev/null || echo '未找到')"
echo "  trtexec   : $INSTALL_BASE/$TRT_DIR/targets/x86_64-linux-gnu/bin/trtexec"
echo "  TRT 头    : $(ls "$TRT_INC" | grep -c '^NvInfer.*\.h$')/8 个"
echo
echo "✅ 下一步: 进入 my_build_check_env/ 运行 ./check_build_env.sh 做完整审计 (0 ❌ 才算齐)"
```

---

## 二、分步命令拆解 + 验证

> 不想跑脚本、或某步脚本失败时，按下面逐条手动执行。每步都带**验证命令**。

### ① 系统工具链（g++ / make / cmake）

```bash
sudo apt update
sudo apt install -y build-essential cmake pkg-config

# 验证
g++ --version          # 应 ≥ 11
make --version
cmake --version
```

### ② NVIDIA 驱动（驱动决定能用多新的 CUDA）

```bash
ubuntu-drivers devices                 # 看推荐版本
sudo ubuntu-drivers autoinstall        # 自动装推荐驱动
sudo reboot                            # 重启生效
nvidia-smi                             # 验证: 出驱动版本 + GPU 型号即成功

# 备选: 手动 runfile (autoinstall 失败时)
#   到 https://www.nvidia.com/Download/index.aspx 选型号下载 .run
#   sudo sh NVIDIA-Linux-x86_64-xxx.run --silent && sudo reboot
```

> 注意: 驱动版本 **必须 ≥ 570**，否则后面 CUDA 12.8 的 libcudart 加载会失败。

### ③ CUDA toolkit（nvcc + cuda_runtime.h + libcudart）

> 关键认知: `pip` 装的 torch/tensorrt 自带 CUDA **运行库**，但**不含 nvcc 和头文件**；
> 编译 C++ 必须单独装 toolkit。

```bash
# 1. 下载 runfile: https://developer.nvidia.com/cuda-downloads
#    → Linux / x86_64 / Ubuntu 22.04 / runfile (local)
#    得到类似 cuda_12.8.0_570.86.10_linux.run

# 2. 只装 toolkit (驱动已在②装好, --toolkit 跳过驱动, --override 跳过编译器校验)
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent --override

# 3. 环境变量
echo 'export PATH=/usr/local/cuda/bin:$PATH'               >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# 4. 注册 libcudart
echo '/usr/local/cuda/lib64' | sudo tee /etc/ld.so.conf.d/cuda.conf
sudo ldconfig

# 验证
nvcc --version                                        # release 12.8
ls /usr/local/cuda/include/cuda_runtime.h             # 头文件存在
ldconfig -p | grep libcudart.so.12                    # 库已注册
```

### ④ OpenCV C++ 开发库（apt 版，头文件+库全齐）

```bash
sudo apt install -y libopencv-dev      # Ubuntu 22.04 装的是 4.5.4

# 验证
pkg-config --modversion opencv4         # 输出 4.5.4
ls /usr/include/opencv4/opencv2/opencv.hpp
```

> 关键认知: `pip install opencv-python` 只给 Python 用，**C++ 必须 `libopencv-dev`**。

### ⑤ TensorRT 10.3 C++（库 + 头文件 + trtexec）

> 💡 **更简单的下载方式**：完整 tar 包 ~2GB，但项目实际只用很小一部分（实测：推理核心
> `libnvinfer.so.10` 仅 242MB + 头文件几十 KB + `trtexec` 1.7MB）。三种方式，按省事排序：
>
> **方式一（最省流量）：从已配好的机器 scp 拷贝** —— 只拷 3 个东西：
> ```bash
> SRC=/root/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu
> scp "$SRC/lib/libnvinfer.so.10.3.0" 新机:$SRC/lib/   # 推理库 242MB
> scp "$SRC/bin/trtexec"             新机:$SRC/bin/     # 1.7MB
> scp trt_include/*.h                新机:trt_include/   # 8 个头文件
> # 若新机还要 build engine，再加 builder_resource(1.4GB):
> # scp "$SRC/lib/libnvinfer_builder_resource.so.10.3.0" 新机:$SRC/lib/
> ```
>
> **方式二（省心免登录，推荐）：apt 分包**，精确 pin 到 10.3.0.26，头文件自动装好、不用补断链：
> ```bash
> wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
> sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
> sudo apt install -y libnvinfer10=10.3.0.26-1+cuda12.5 \
>                     libnvinfer-dev=10.3.0.26-1+cuda12.5 \
>                     libnvinfer-bin=10.3.0.26-1+cuda12.5
> trtexec --version && ls /usr/include/x86_64-linux-gnu/NvInfer.h   # 验证
> ```
> 注: apt 的 libnvinfer10 仍 ~1.2GB(把 builder_resource 打进去了)，下载量与 tar 接近，
> 省的是登录/补头文件/依赖这些麻烦，不是流量。
>
> **方式三（完整 tar 包，兜底）**：下面 5.1~5.3，下载 ~2GB，解压后删大文件可省磁盘（5.4）。
>
> ---

```bash
# 5.1 下载 tar 包: https://developer.nvidia.com/tensorrt-download (需登录)
#     → 10.3.0.26 / Linux x86_64 / cuda 12.8
#    得到类似 TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.8.tar.gz
#    解压到与项目 build.sh 一致的目录 (此处 /root/my_FILE):
mkdir -p /root/my_FILE
tar -xzf TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.8.tar.gz -C /root/my_FILE

# 5.2 ⚠️ 头文件: tar 包 include 可能断链, 必须从官方 GitHub 补齐 (8 个)
TRT_INC=/root/my_FILE/TensorRT-10.3.0.26/include
mkdir -p "$TRT_INC"
for h in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferRuntime.h \
         NvInferRuntimeBase.h NvInferRuntimeCommon.h NvInferRuntimePlugin.h NvInferVersion.h; do
    curl -fsSL -o "$TRT_INC/$h" "https://raw.githubusercontent.com/NVIDIA/TensorRT/v10.3.0/include/$h"
done

# 5.3 注册库 (二选一: 打 ldconfig, 或编译时 -Wl,-rpath; 本项目 build.sh 用 rpath)
echo '/root/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib' | sudo tee /etc/ld.so.conf.d/trt.conf
sudo ldconfig

# 5.4 (可选) 删掉用不到的大文件: lib/ 从 6.3G 降到 ~300MB
TRT_LIB=/root/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib
rm -f "$TRT_LIB"/*_static.a "$TRT_LIB"/*_win.so*
# 仅当本机只跑推理、不 build engine 时再删 builder_resource(1.4G):
# rm -f "$TRT_LIB"/libnvinfer_builder_resource*.so

# 验证
/root/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/bin/trtexec --version   # trtexec 可用
ls "$TRT_INC" | grep -c '^NvInfer.*\.h$'                                          # 应为 8
ldconfig -p | grep libnvinfer.so.10
```

### ⑥ 功能库（按需，扩展全链条时逐个补）

| 功能 | 库 | 命令 |
|---|---|---|
| HTTP 上报 | libcurl | `sudo apt install -y libcurl4-openssl-dev` |
| Modbus/TCP | libmodbus | `sudo apt install -y libmodbus-dev` |
| Siemens PLC(S7) | snap7 | `sudo apt install -y libsnap7-dev` |
| PostgreSQL 入库 | libpq | `sudo apt install -y libpq-dev` |
| 海康/Basler 相机 | MVS / pylon | 厂商官网 tar 包 + 官方安装脚本 |
| RTSP 拉流 | OpenCV(FFMPEG) | apt 版 opencv 已带 FFMPEG 后端 |

### ⑦ 完整验证（装完后必做）

```bash
cd /root/my_FILE/make_opt/cpp_pipeline/my_build_check_env
./check_build_env.sh                          # 一键审计, 0 ❌ 才算齐

cd /root/my_FILE/make_opt/cpp_pipeline
bash build.sh                                 # 编译本项目, 出 ./infer_e2e_b8_cpp = 环境可用
./infer_e2e_b8_cpp --source <测试图目录> --engine <xxx.engine>   # 跑通
```

> 若 build.sh 里 `TRT_ROOT`/`CUDA_ROOT` 与你实际路径不同，改 [build.sh](../build.sh) 顶部
> 两个变量即可（本手册默认与现有一致：`/root/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu`、
> `/usr/local/cuda`）。

---

## 三、部署机最小清单（详见 [DEPLOY_MIGRATION.md](../DEPLOY_MIGRATION.md)）

部署机**只运行不编译**，只需三样，运行库全部由 `collect_deps.sh` 打包进 `deploy/`：

| 项 | 命令 / 要求 |
|---|---|
| NVIDIA 驱动 | `sudo ubuntu-drivers autoinstall && sudo reboot` |
| GPU 架构 | 必须与 engine 一致（SM 8.9 等）；不一致要 trtexec 重 build |
| 系统 | glibc ≥ 2.35 的 64 位 Linux（`ldd --version` 查看） |

**不需要**装：编译器(build-essential)、CUDA toolkit、OpenCV、TensorRT 头文件。
部署方式：整个 `deploy/` 目录拷过去 → `./run.sh --source ...` 直接用。

---

## 四、常见坑（本项目实际踩过）

| 坑 | 症状 | 处置 |
|---|---|---|
| pip 的 CUDA ≠ 系统 CUDA | `import torch` 正常，但 `nvcc`/头文件找不到 | 编译必须装 toolkit runfile（③） |
| opencv-python ≠ libopencv-dev | `#include <opencv2/...>` 找不到 | C++ 要 apt 装 libopencv-dev（④） |
| TRT tar 包 include 断链 | 编译报 `NvInfer.h: No such file` | 用 GitHub v10.3.0 tag 头补齐（⑤.2） |
| 驱动版本 < 570 | libcudart 加载失败 / nvidia-smi 异常 | 升驱动 ≥ 570（②） |
| nproc 是虚的 | 96 核机器只给 12 核配额 | 云容器 cgroup；线程数按配额规划 |
| 部署机 glibc 老于编译机 | `GLIBC_2.32 not found` | 低版本系统重编译，或统一升级系统 |
| 直接拷二进制不带库 | `libnvinfer.so.10: cannot open` | 用 collect_deps.sh 打包 + run.sh 启动 |
| engine 换卡跑不了 | `no kernel image is available` | engine 绑 GPU 架构；新卡 trtexec 重 build |
