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
