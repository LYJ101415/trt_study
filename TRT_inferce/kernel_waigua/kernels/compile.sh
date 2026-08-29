#!/usr/bin/env bash
# 编译 gpu_ops.cu -> libgpu_ops.so（零 PyTorch，仅依赖 CUDA toolkit 的 nvcc + thrust）
set -e 

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" 

# RTX 4080 = sm_89 (Ada)。如需其他架构，改 ARCH 即可。
ARCH="${ARCH:-sm_89}" 

cd "$(dirname "$0")"

"$NVCC" -O3 -std=c++17 -arch="${ARCH}" -Xcompiler -fPIC -shared \
    -I"${CUDA_HOME}/include" \
    gpu_ops.cu -o libgpu_ops.so

echo "built libgpu_ops.so (arch=${ARCH})"
