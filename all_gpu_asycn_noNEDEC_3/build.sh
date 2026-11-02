#!/usr/bin/env bash
# build.sh — 编译图片推理流水线（多解码/保存线程 + 三流 GPU）+ 各阶段耗时测试。
# 产物：./image_infer  与  ./bench_stages
set -e
cd "$(dirname "$0")"

CUDA="${CUDA:-/usr/local/cuda-12.8}"
TRT_LIB="${TRT_LIB:-/root/miniconda3/lib/python3.12/site-packages/tensorrt_libs}"
TRT_INC="${TRT_INC:-/root/my_FILE/trt_include}"
BUILD=build

NVCC="$CUDA/bin/nvcc"
CXX=g++

INCS="-Iinclude -I$TRT_INC -I$CUDA/include"
CXXFLAGS="-O3 -std=c++17 -DNDEBUG -Wno-deprecated-declarations $INCS"
# 目标设备 sm_86（RTX 3080 Ti）；如需多架构兼容可追加 sm_75/80/89
GENCODE="-gencode arch=compute_86,code=sm_86"

mkdir -p "$BUILD"

echo "[1/4] 编译 draw.cu (自定义画框 kernel，NPP 无画框原语) ..."
"$NVCC" -O3 -std=c++17 $GENCODE -Xcompiler -fPIC -c kernels/draw.cu -o "$BUILD/draw.o" -Iinclude

echo "[2/4] 编译 C++ 源码 ..."
for src in src/config.cpp src/jpeg_loader.cpp src/npp_ops.cpp src/tensorrt_e2e.cpp \
           src/pipeline.cpp src/main.cpp; do
    obj="$BUILD/$(basename "${src%.*}").o"
    $CXX $CXXFLAGS -c "$src" -o "$obj"
done

echo "[3/4] 编译 bench_stages ..."
$CXX $CXXFLAGS -c bench/bench_stages.cpp -o "$BUILD/bench_stages.o"

echo "[4/4] 链接 ..."
# NPP 只用到 core / 数据交换(Set) / 几何(Resize)；多链 nppial/nppicc 无副作用
NPP_LIBS="-lnppc -lnppial -lnppicc -lnppidei -lnppig"
TRT_DEPS="-L$TRT_LIB -L$CUDA/lib64 -lnvinfer -lcudart -lcuda $NPP_LIBS -ljpeg -lpthread -ldl \
          -Wl,-rpath,$TRT_LIB -Wl,-rpath,$CUDA/lib64"

# image_infer（主程序，不含 bench_stages.o）
$CXX "$BUILD"/config.o "$BUILD"/jpeg_loader.o "$BUILD"/npp_ops.o "$BUILD"/tensorrt_e2e.o \
    "$BUILD"/pipeline.o "$BUILD"/main.o "$BUILD"/draw.o -o image_infer $TRT_DEPS

# bench_stages（各阶段耗时测试）
$CXX "$BUILD"/jpeg_loader.o "$BUILD"/npp_ops.o "$BUILD"/tensorrt_e2e.o \
    "$BUILD"/bench_stages.o "$BUILD"/draw.o -o bench_stages $TRT_DEPS

echo "✅ 编译完成: ./image_infer 与 ./bench_stages"
