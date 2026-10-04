#!/usr/bin/env bash
# find_cuda.sh — 查找本机可用的 CUDA 工具链（nvcc + 头文件 cuda_runtime.h / thrust）。
# 用途：定位 nvcc（尤其 12.8 的 conda 工具链）与匹配的 include 目录，供 compile.sh 参考。

echo "========== 1. 查找 nvcc =========="
found=0
# 常见安装位置（conda 优先）
for d in \
    /root/miniconda3/bin \
    /usr/local/cuda-12.8/bin \
    /usr/local/cuda/bin \
    /usr/local/cuda-12.4/bin \
    /opt/cuda/bin \
    /usr/bin ; do
  if [ -x "$d/nvcc" ]; then
    ver=$("$d/nvcc" --version 2>/dev/null | grep -ioE "release [0-9]+\.[0-9]+" | head -1)
    printf "  %-42s %s\n" "$d/nvcc" "${ver:-unknown}"
    found=1
  fi
done
[ "$found" = "0" ] && echo "  (未在常见路径找到 nvcc)"

# PATH 里的 nvcc（可能与上面重复，仅提示）
if command -v nvcc >/dev/null 2>&1; then
  echo "  PATH 中的 nvcc: $(command -v nvcc)"
fi

echo
echo "========== 2. 查找 CUDA 头文件 =========="
# cuda_runtime.h 所在目录；thrust 应在其同级子目录（gpu_ops.cu 需要 thrust/sort.h 等）
for f in $(find /root/miniconda3 /usr/local /opt -maxdepth 7 -name cuda_runtime.h 2>/dev/null | sort); do
  inc=$(dirname "$f")
  if [ -d "$inc/thrust" ]; then
    printf "  %-58s (含 thrust: yes)\n" "$inc"
  else
    printf "  %-58s (含 thrust: NO)\n" "$inc"
  fi
done

echo
echo "========== 3. 建议 =========="
echo "  本机可用的 12.8 工具链（conda）："
echo "    NVCC    = /root/miniconda3/bin/nvcc"
echo "    INCLUDE = /root/miniconda3/targets/x86_64-linux/include  （conda 的 nvcc 可自动检测此目录）"
