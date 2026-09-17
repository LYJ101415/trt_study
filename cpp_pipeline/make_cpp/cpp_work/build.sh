#!/bin/bash
# build.sh — 编译 cpp_work 下的两个可执行文件
#   infer_e2e_b8_cpp     : main.cpp         —— 方案 A 用的单发 worker
#   infer_e2e_b8_daemon  : main_daemon.cpp  —— 方案 B 用的常驻 daemon
#
# 头文件/库与 make_cpp/build.sh 一致：TRT 头用 ../trt_include，库用 pip 的 tensorrt_libs + 系统 CUDA + OpenCV。

set -e
cd "$(dirname "$0")"

# 依赖路径
TRT_INC=../trt_include                                     # TRT 头文件（NvInfer.h 等）
TRT_ROOT=/root/miniconda3/lib/python3.12/site-packages/tensorrt_libs
CUDA_ROOT=/usr/local/cuda

# 头文件搜索路径（编译期）
INCLUDE_FLAGS=(-I"$TRT_INC" -I"$CUDA_ROOT/include" -I/usr/include/opencv4)

# 库搜索与链接（链接期）
# [注意] 库(-l)必须放在源文件之后，否则 --as-needed 会在链接器还没遇到符号时就把库丢弃，
#        导致 undefined reference（本目录首次编译时的报错就是这个原因）。
LIB_FLAGS=(-L"$TRT_ROOT" -L"$CUDA_ROOT/lib64" \
    -lnvinfer -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lpthread \
    -Wl,-rpath,"$TRT_ROOT" -Wl,-rpath,"$CUDA_ROOT/lib64")

# [改进] 打开常用告警，便于发现未初始化变量/未用返回值等问题
WARN_FLAGS=(-Wall -Wextra)

echo "==> 编译单发 worker (main.cpp)"
g++ -O3 -std=c++17 -DNDEBUG "${WARN_FLAGS[@]}" \
    main.cpp -o infer_e2e_b8_cpp \
    "${INCLUDE_FLAGS[@]}" "${LIB_FLAGS[@]}"

echo "==> 编译常驻 daemon (main_daemon.cpp)"
g++ -O3 -std=c++17 -DNDEBUG "${WARN_FLAGS[@]}" \
    main_daemon.cpp -o infer_e2e_b8_daemon \
    "${INCLUDE_FLAGS[@]}" "${LIB_FLAGS[@]}"

echo "✅ 编译完成:"
echo "   ./infer_e2e_b8_cpp      (方案 A 单发 worker)"
echo "   ./infer_e2e_b8_daemon   (方案 B 常驻 daemon)"
