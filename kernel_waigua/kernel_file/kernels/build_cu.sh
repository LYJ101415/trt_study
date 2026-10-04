# #!/usr/bin/env bash
# # 编译 gpu_ops.cu -> libgpu_ops.so（零 PyTorch，仅依赖 CUDA toolkit 的 nvcc + thrust）

set -e

# 优先用环境变量；否则用当前 conda 环境的 CUDA 12.8
# $ 后面应该跟变量名，比如 $CUDA_HOME，不能直接放在路径前面。写成 $/root/...，Shell 会把它当成一个错误的变量展开
# CUDA_HOME="${CUDA_HOME:-/root/miniconda3}"
# NVCC="${NVCC:-${CUDA_HOME}/bin/nvcc}"
# CUDA_INCLUDE="${CUDA_INCLUDE:-${CUDA_HOME}/targets/x86_64-linux/include}"

# # RTX 4080 = sm_89
# ARCH="${ARCH:-sm_89}"

# cd "$(dirname "$0")"

# "$NVCC" -O3 -std=c++17 -arch="${ARCH}" -Xcompiler -fPIC -shared \
#     -I"${CUDA_INCLUDE}" \
#     gpu_ops.cu -o libgpu_ops.so

# echo "built libgpu_ops.so (arch=${ARCH})"


NVCC="/root/miniconda3/bin/nvcc"
CUDA_INCLUDE="/root/miniconda3/targets/x86_64-linux/include "

# RTX 4080 = sm_89
ARCH="${ARCH:-sm_89}"

cd "$(dirname "$0")"

"$NVCC" -O3 -std=c++17 -arch="${ARCH}" -Xcompiler -fPIC -shared \
    -I"${CUDA_INCLUDE}" \
    gpu_ops.cu -o libgpu_ops.so

echo "built libgpu_ops.so (arch=${ARCH})"
