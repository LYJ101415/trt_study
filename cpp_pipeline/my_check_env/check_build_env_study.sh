#!/bin/bash
# ============================================================================
# check_build_env.sh —— 通用 C++ (TensorRT/CUDA/OpenCV) 编译环境审计脚本
#
# 用法:   bash check_build_env.sh
# 特点:
#   ① 通用 —— 不绑定具体项目，自动探测「系统 runfile / apt / pip / tar」多种安装来源
#   ② 每个已找到的包都打印【实际位置】(↳ 行)，方便直接填 -I / -L / -Wl,-rpath
#   ③ 结果分 ✅(必需) / ⚠️(可选或提示) / ❌(缺失)，结尾给汇总

# 特别注意：Linux 的 bash 只认 LF（\n），不认 CRLF（\r\n），所以在 Windows 下编辑的脚本文件必须转换为 LF，否则会报错：
# 检查文件是否输出包含 CRLF（如有就不行）：file /root/my_FILE/cpp_pipeline/my_check_env/check_build_env.sh
#        
# 文件可处理方法：sed -i 's/\r$//' /root/my_FILE/cpp_pipeline/my_check_env/check_build_env_study.sh
#    或（需要安装dos2unix)：dos2unix /root/my_FILE/cpp_pipeline/my_check_env/check_build_env_study.sh
# ============================================================================

pass=0; fail=0
ok()   { echo "    ✅ $1"; pass=$((pass+1)); }
bad()  { echo "    ❌ $1"; fail=$((fail+1)); }
warn() { echo "    ⚠️  $1"; }
at()   { echo "          ↳ $1"; }

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- 定位 pip 的 nvidia 包根目录 (cuda_runtime / tensorrt_libs / cublas ...) ----
PIP_NV=""
for sp in $(python3 -c "import site; print(' '.join(site.getsitepackages()))" 2>/dev/null) \
          /usr/lib/python3/dist-packages /usr/local/lib/python3.12/dist-packages; do
    [ -d "$sp/nvidia" ] && PIP_NV="$sp/nvidia" && break
done

echo "================ ① 编译工具链 ================"
for t in g++ make cmake pkg-config; do
    if P=$(command -v $t 2>/dev/null); then
        v=$($t --version 2>/dev/null | head -1 | awk '{print $NF}')
        ok "$t ${v:+($v)}"
        at "$P"
    else
        bad "$t 未安装 —— sudo apt install -y build-essential cmake pkg-config"
    fi
done

echo "================ ② OpenCV C++ 开发库 ================"
if pkg-config --exists opencv4 2>/dev/null; then
    ok "OpenCV $(pkg-config --modversion opencv4 2>/dev/null)"
    at "头文件: $(pkg-config --variable=includedir opencv4 2>/dev/null)"
    at "库目录: $(pkg-config --variable=libdir opencv4 2>/dev/null)"
elif [ -f /usr/include/opencv4/opencv2/opencv.hpp ]; then
    ok "OpenCV 头文件存在"
    at "/usr/include/opencv4"
else
    bad "OpenCV C++ 开发库未装 —— sudo apt install -y libopencv-dev"
fi

echo "================ ③ NVIDIA 驱动 / GPU ================"
if NS=$(command -v nvidia-smi 2>/dev/null); then
    ok "驱动 $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
    ok "GPU  $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null | head -1)"
    at "$NS"
else
    bad "NVIDIA 驱动未装 —— ubuntu-drivers autoinstall"
fi

echo "================ ④ CUDA (系统 toolkit + pip 运行时) ================"
# --- 4.1 系统 CUDA toolkit (/usr/local/cuda*) ---
SYS_CUDA=""
for c in /usr/local/cuda-12.* /usr/local/cuda-11.* /usr/local/cuda; do
    [ -f "$c/include/cuda_runtime.h" ] && SYS_CUDA=$(readlink -f "$c") && break
done
if [ -n "$SYS_CUDA" ]; then
    ok "系统 CUDA toolkit: $(basename $SYS_CUDA)"
    at "$SYS_CUDA"
    if [ -x "$SYS_CUDA/bin/nvcc" ]; then
        ok "nvcc $($SYS_CUDA/bin/nvcc --version 2>/dev/null | grep -oP 'release [0-9.]+')"
        at "$SYS_CUDA/bin/nvcc"
    else
        warn "nvcc 未找到（只编译链接不写 .cu 可不需要）"
    fi
    [ -f "$SYS_CUDA/include/cuda_runtime.h" ] && { ok "CUDA 头文件"; at "$SYS_CUDA/include/cuda_runtime.h"; }
    if ls "$SYS_CUDA/lib64/libcudart.so"* >/dev/null 2>&1; then
        ok "libcudart（系统）"; at "$(ls $SYS_CUDA/lib64/libcudart.so.12 2>/dev/null | head -1)"
    fi
else
    bad "系统 CUDA toolkit 未装(/usr/local/cuda)"
fi

# --- 4.2 pip 的 CUDA 运行时/数学库 (torch 常自带) ---
if [ -n "$PIP_NV" ]; then
    ok "pip CUDA 包目录"
    at "$PIP_NV"
    for pkg in cuda_runtime cublas cudnn cufft curand cusparse cusolver nccl; do
        d="$PIP_NV/$pkg"
        if [ -d "$d/include" ] || [ -d "$d/lib" ]; then
            ok "pip $pkg"
            [ -d "$d/include" ] && at "头目录: $d/include"
            [ -d "$d/lib" ]     && at "库目录: $d/lib"
        fi
    done
else
    warn "未找到 pip 的 nvidia 包目录（torch/tensorrt pip 包未装或 python 不同）"
fi

echo "================ ⑤ TensorRT ================"
# --- 5.1 trtexec 二进制 ---
TRTEXEC=$(command -v trtexec 2>/dev/null)
[ -z "$TRTEXEC" ] && for d in /usr/src/tensorrt/bin /usr/local/bin; do
    [ -x "$d/trtexec" ] && TRTEXEC="$d/trtexec" && break
done
if [ -n "$TRTEXEC" ]; then
    ver=$(trtexec --version 2>&1 | grep -oP 'TensorRT v[0-9]+' | head -1)
    ok "trtexec ${ver:+($ver)}"
    at "$TRTEXEC"
else
    warn "trtexec 未找到（性能/层分析工具，apt 的 libnvinfer-bin 或 tar 包提供）"
fi

# --- 5.2 libnvinfer.so.10 运行库 (pip / tar / ldconfig) ---
NVINFER=$(ldconfig -p 2>/dev/null | grep -m1 'libnvinfer.so.10' | awk '{print $NF}')
[ -z "$NVINFER" ] && [ -n "$PIP_NV" ] && [ -f "$PIP_NV/tensorrt_libs/libnvinfer.so.10" ] \
    && NVINFER="$PIP_NV/tensorrt_libs/libnvinfer.so.10"
[ -z "$NVINFER" ] && NVINFER=$(find /root /opt /usr/local -name 'libnvinfer.so.10' -type f 2>/dev/null | head -1)
if [ -n "$NVINFER" ]; then
    ok "libnvinfer.so.10"
    at "$NVINFER"
else
    bad "libnvinfer.so.10 未找到 —— pip install tensorrt-cu12 或装 TensorRT tar"
fi

# --- 5.3 TRT C++ 头文件 (12 个, v10.8.0) ---
HDR_DIR=""
for c in "$SCRIPT_DIR/trt_include" "$SCRIPT_DIR/../trt_include" /root/my_FILE/trt_include \
         /usr/include/x86_64-linux-gnu /usr/include; do
    [ -f "$c/NvInfer.h" ] && HDR_DIR="$c" && break
done
# tar 包 include 兜底
[ -z "$HDR_DIR" ] && HDR_DIR=$(find /root /opt /usr/local -path '*/include/NvInfer.h' 2>/dev/null | head -1 | xargs dirname)
TRT_HDRS="NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferPlugin.h NvInferPluginBase.h \
NvInferPluginUtils.h NvInferRuntime.h NvInferRuntimeBase.h NvInferRuntimeCommon.h \
NvInferRuntimePlugin.h NvInferVersion.h NvOnnxConfig.h"
if [ -n "$HDR_DIR" ] && [ -f "$HDR_DIR/NvInfer.h" ]; then
    miss=""
    for h in $TRT_HDRS; do [ -f "$HDR_DIR/$h" ] || miss="$miss $h"; done
    if [ -z "$miss" ]; then
        ok "TRT 头文件 12/12"
        at "$HDR_DIR"
    else
        warn "TRT 头文件不全，缺:$miss"
        at "$HDR_DIR"
    fi
else
    bad "TRT 头文件目录未找到 —— 从 github NVIDIA/TensorRT v10.8.0 include/ 拉 12 个头"
fi

# --- 5.4 builder_resource / plugin / onnxparser (build engine 与插件用) ---
if [ -n "$PIP_NV" ] && [ -d "$PIP_NV/tensorrt_libs" ]; then
    TL="$PIP_NV/tensorrt_libs"
    for so in libnvinfer_builder_resource libnvinfer_plugin libnvonnxparser; do
        if ls "$TL/$so"* >/dev/null 2>&1; then
            ok "$so"
            at "$(ls $TL/$so* 2>/dev/null | head -1)"
        fi
    done
fi

echo "================ ⑥ 系统 / 运行约束 ================"
ok "glibc $(ldd --version 2>/dev/null | head -1 | awk '{print $NF}')（编译产物只能跑在 ≥ 此版本的系统）"
if [ -f /sys/fs/cgroup/cpu.max ]; then
    read -r q p < /sys/fs/cgroup/cpu.max
    if [ "$q" != "max" ]; then
        warn "CPU 配额受限: $((q/p)) 核（nproc 显示 $(nproc) 是虚的，线程数按配额规划）"
    else
        ok "CPU 无配额限制 ($(nproc) 核)"
    fi
fi
avail=$(df --output=avail -BG "$SCRIPT_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
[ -n "$avail" ] && { [ "$avail" -lt 5 ] && warn "磁盘剩余 ${avail}G 偏少" || ok "磁盘剩余 ${avail}G"; }

echo
echo "================ 汇总: $pass 项通过, $fail 项缺失 ================"
[ $fail -eq 0 ] && echo "✅ 编译环境完整，可进行 C++ 开发/编译/打包" \
                || echo "❌ 请按上方 ❌ 项补齐后重跑本脚本"


: <<'COMMENT'
"""
这份脚本是一个用于 NVIDIA TensorRT C++ 部署环境 的自动化检查工具。它不仅检查软件是否安装，还深入验证了版本兼容性、头文件完整性、动态库链接状态以及云原生环境下的资源限制。
以下是对该脚本的详细逻辑解析及总结出的完整依赖清单。
一、 代码逻辑详细解析
脚本通过 ok (✅)、bad (❌)、warn (⚠️) 三个函数将检查结果分级，核心检查逻辑分为七个维度：
1. 系统工具链 (编译基础)
g++ / make: 使用 command -v 检查是否存在。这是 C++ 编译的绝对前提，缺失标记为 ❌。
cmake: 检查是否存在。对于大型 C++ 项目通常是必需的，但脚本将其标记为 ⚠️（警告），暗示可能存在非 CMake 的构建方式或仅做简单编译。
libopencv-dev: 使用 dpkg -s 检查 Debian/Ubuntu 包状态。OpenCV 是计算机视觉部署的核心库，缺失标记为 ❌。
2. NVIDIA 驱动与 GPU
nvidia-smi: 检查驱动管理工具。如果存在，进一步查询驱动版本和 GPU 型号/算力。
逻辑: 无驱动则无法使用 GPU 加速，标记为 ❌。
3. CUDA Toolkit (编译与运行时)
头文件 (cuda_runtime.h): 检查 /usr/local/cuda/include。这是编译 CUDA 代码的必要条件，缺失标记为 ❌。
编译器 (nvcc): 检查 /usr/local/cuda/bin/nvcc。缺失仅标记为 ⚠️，因为如果只调用预编译库或使用 g++ 链接 CUDA runtime，可能不需要 nvcc。
运行时库 (libcudart.so.12): 通过 ldconfig -p 检查系统是否注册了 CUDA 12.x 的动态库。未注册标记为 ⚠️ 并给出修复命令（配置 ld.so.conf），这影响运行时加载而非编译。
4. TensorRT C++ (核心推理引擎)
这是脚本最复杂的部分，进行了三层验证：
二进制工具 (trtexec): 在多个候选路径（脚本上级目录、/root/my_FILE、/opt、/usr/local）下递归搜索 TensorRT-* 目录，并验证 trtexec 可执行文件。缺失标记为 ❌。
头文件完整性:
优先查找脚本自带的 trt_include/，其次查找 tar 包内的 include/。
精确校验: 逐一检查 8 个关键头文件（NvInfer.h, NvInferVersion.h 等）。任意缺失都会导致 C++ 编译失败，标记为 ❌ 并列出具体的缺失文件名。
动态库 (libnvinfer.so.10): 检查 ldconfig 注册状态。未注册仅标记为 ⚠️，因为脚本注释说明构建脚本可能使用了 -Wl,-rpath 来指定库路径，不依赖系统全局注册。
5. cgroup CPU 配额 (云环境适配)
cpu.max: 读取 cgroup v2 的 CPU 限制文件。
逻辑: 如果配额不是 "max"，计算实际可用核数并发出 ⚠️ 警告。这是为了防止在 Docker/K8s 容器中，程序误用宿主机的 nproc 数值导致线程过载或性能抖动。
6. glibc 版本 (部署兼容性)
ldd --version: 获取当前系统的 glibc 版本。
意义: 这是一个隐性依赖。在此机器上编译的二进制文件，不能运行在 glibc 版本低于此版本的机器上。脚本仅作信息展示 (✅)，但这对生产环境部署至关重要。
7. 全链条资源检查
Engine 文件: 检查 deploy/engine/yolov8_int8_e2e_dyn.engine 是否存在。不存在仅警告，因为新架构 GPU 可能需要重新生成。
Python 量化链: 尝试 import torch, modelopt。这是用于模型量化和生成 engine 的上游工具链。纯 C++ 推理阶段不需要，故标记为 ⚠️。
磁盘空间: 检查脚本所在分区剩余空间。< 5GB 标记为 ❌，防止编译或解压过程中磁盘写满。
二、 完整依赖清单总结
根据上述分析，该环境所需的依赖清单如下表所示：
1. 系统级与编译依赖 (OS & Build)
表格
依赖项	类型	必需性	推荐版本/备注	安装/修复命令
g++	编译器	🔴 必需	GCC 9+ (建议 11+)	sudo apt install -y build-essential
make	构建工具	🔴 必需	GNU Make 4.x	sudo apt install -y build-essential
cmake	构建工具	🟡 推荐	≥ 3.18	sudo apt install -y cmake
libopencv-dev	开发库	🔴 必需	4.x (匹配 TRT 插件)	sudo apt install -y libopencv-dev
glibc	系统库	🔵 基准	≥ 编译机版本	不可升级，需匹配目标部署环境
磁盘空间	资源	🔴 必需	≥ 5 GB 可用	清理磁盘或扩容
2. NVIDIA 生态依赖 (GPU Stack)
表格
依赖项	类型	必需性	推荐版本/备注	安装/修复命令
NVIDIA Driver	驱动	🔴 必需	≥ 535 (配合 CUDA 12)	ubuntu-drivers autoinstall 或官网下载
CUDA Toolkit	SDK	🔴 必需	12.x (脚本硬编码检查 .so.12)	官网 Runfile 安装 (选 toolkit only)
cuda_runtime.h	头文件	🔴 必需	随 CUDA Toolkit 安装	重装 CUDA Toolkit
libcudart.so.12	运行时库	🟡 推荐	12.x	echo /usr/local/cuda/lib64 | sudo tee /etc/ld.so.conf.d/cuda.conf && sudo ldconfig
nvcc	编译器	🟡 可选	12.x	仅当需要编译自定义 CUDA kernel 时必需
3. TensorRT C++ 依赖 (Inference Engine)
表格
依赖项	类型	必需性	推荐版本/备注	安装/修复命令
TensorRT Tar Package	SDK	🔴 必需	10.x (脚本检查 .so.10)	下载 TensorRT-10.x tar.gz 解压至 /opt 或项目目录
trtexec	工具	🔴 必需	随 TRT Tar 包提供	确保在 TensorRT-*/targets/x86_64-linux-gnu/bin/ 下
TRT C++ Headers	头文件	🔴 必需	v10.3.0 (8个特定文件)	从 GitHub NVIDIA/TensorRT v10.3.0 补齐到 trt_include/
libnvinfer.so.10	运行时库	🟡 推荐	10.x	若构建脚本未设 rpath，需添加到 ldconfig
⚠️ 关键版本锁定提示：
脚本中硬编码了 libcudart.so.12 和 libnvinfer.so.10。这意味着该环境强绑定 CUDA 12.x + TensorRT 10.x。使用其他版本组合将无法通过检查。
4. 开发与量化辅助依赖 (Optional / Dev Only)
表格
依赖项	类型	必需性	用途	备注
Python 3	解释器	🟡 可选	模型量化/ONNX导出	纯 C++ 推理部署可忽略
PyTorch	Python库	🟡 可选	量化上游框架	需与 CUDA/TRT 版本对齐
ModelOpt	Python库	🟡 可选	NVIDIA 量化工具	pip install nvidia-modelopt
YOLOv8 Engine	模型文件	🟡 可选	INT8 动态 batch 引擎	新 GPU 需用 trtexec 重新构建
5. 运行时环境约束 (Runtime Constraints)
CPU Quota Awareness: 如果在容器内运行，必须感知 cgroup CPU 限制。应用程序应读取 /sys/fs/cgroup/cpu.max 而非 nproc 来决定线程池大小。
ABI 兼容性: 编译产物的向下兼容性受限于编译机的 glibc 版本。若需支持旧版 OS（如 Ubuntu 18.04），建议在对应旧版环境中编译或使用静态链接。
三、 总结建议
该脚本定义了一个 “CUDA 12 + TensorRT 10 + OpenCV” 的 C++ 高性能推理部署标准环境。
对于开发者：请严格按照清单中的版本号（特别是 TRT 10.3.0 头文件和 CUDA 12）配置环境，否则编译必然失败。
对于运维/DevOps：重点关注 libnvinfer 和 libcudart 的 ldconfig 注册，以及容器场景下的 CPU 配额问题。
对于部署：务必记录编译机的 glibc 版本，作为交付文档的一部分，避免在低版本系统上出现 GLIBC_2.xx not found 错误。
"""
COMMENT