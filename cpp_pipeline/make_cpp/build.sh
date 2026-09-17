#!/bin/bash
# build.sh — 编译 C++ 流水线推理
# 头文件: 本目录 trt_include/ (TRT 10.3.0 官方头, 因本机 tar 包 include 为断链)
# 库:     TensorRT tar 包的 lib + 系统 CUDA + OpenCV

# 1. 脚本初始化与环境设置
set -e # 这是一个 Bash 选项，表示“遇到错误即退出”。如果脚本在执行过程中任何一条命令返回了非零的退出状态（即执行失败），脚本会立即终止，防止在错误状态下继续执行后续命令。
cd "$(dirname "$0")" # 将当前工作目录切换到该脚本文件所在的目录。这确保了后续使用的相对路径（如 ./trt_include 和 main.cpp）都是相对于脚本位置的，
                     # 避免了因用户在任意目录下执行脚本而导致的路径找不到的问题。

# 2. 定义依赖库路径：作用：通过变量管理路径，使得编译命令更加整洁，且如果未来路径发生变化，只需修改此处即可。
TRT_ROOT=/root/miniconda3/lib/python3.12/site-packages/tensorrt_libs  # pip 的 TensorRT 10.8 库目录（libnvinfer.so 等）
CUDA_ROOT=/usr/local/cuda  # 定义了 CUDA_ROOT 环境变量，指向系统默认的 CUDA 安装路径。

# 3. 核心编译命令 (g++)：这是脚本的核心部分，使用 g++ 编译器将 C++ 源代码编译为可执行文件。
g++ -O3 -std=c++17 -DNDEBUG main.cpp -o infer_e2e_b8_cpp \
    -I./trt_include -I"$CUDA_ROOT/include" -I/usr/include/opencv4 \
    -L"$TRT_ROOT" -L"$CUDA_ROOT/lib64" \
    -lnvinfer -lcudart -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lpthread \
    -Wl,-rpath,"$TRT_ROOT" -Wl,-rpath,"$CUDA_ROOT/lib64"

echo "✅ 编译完成: ./infer_e2e_b8_cpp"


: <<'COMMENT'
编译选项：
-O3：开启最高级别的代码优化，以最大化程序的运行性能（对于推理程序至关重要）。
-std=c++17：指定使用 C++17 标准进行编译（TensorRT 的 C++ API 通常需要 C++17 或更高版本）。
-DNDEBUG：定义 NDEBUG 宏，这会禁用 C++ 标准库中的 assert 断言检查，通常用于 Release 版本以提升运行效率。
输入与输出：
main.cpp：指定要编译的 C++ 源代码文件。
-o infer_e2e_b8_cpp：指定生成的可执行文件名称为 infer_e2e_b8_cpp（从命名推测，这可能是一个端到端(end-to-end)、batch size为8的推理程序）。
头文件搜索路径 (-I)：告诉编译器去哪里找 .h 头文件。
-I./trt_include：当前目录下的自定义 TensorRT 头文件。
-I"$TRT_ROOT/include"：TensorRT 官方头文件（如 NvInfer.h）。
-I"$CUDA_ROOT/include"：CUDA 相关头文件。
-I/usr/include/opencv4：OpenCV 4.x 版本的头文件路径。
库文件搜索路径 (-L)：告诉链接器去哪里找编译好的动态/静态库。
-L"$TRT_ROOT/lib"：TensorRT 的库文件路径。
-L"$CUDA_ROOT/lib64"：CUDA 的库文件路径。
链接的库文件 (-l)：明确指定程序运行时需要链接哪些底层库。
-lnvinfer：TensorRT 的核心推理库。
-lcudart：CUDA 运行时库。
-lopencv_core, -lopencv_imgcodecs, -lopencv_imgproc：OpenCV 的核心模块、图像编解码模块和图像处理模块（用于图像的读取、预处理等）。
-lpthread：POSIX 多线程库，用于支持并发执行。
运行时库路径 (-Wl,-rpath)：
-Wl,-rpath,"$TRT_ROOT/lib" 和 -Wl,-rpath,"$CUDA_ROOT/lib64"：将这两个路径硬编码到生成的可执行文件中。
这样在运行程序时，操作系统动态链接器能自动找到 TensorRT 和 CUDA 的 .so 动态库，而不需要用户手动配置 LD_LIBRARY_PATH 环境变量。
4. 编译结果提示
echo "✅ 编译完成: ./infer_e2e_b8_cpp"：如果代码执行到这里，说明前面的 g++ 编译成功（没有触发 set -e 的退出机制），
向用户输出编译成功的提示信息。
总结：这段脚本是一个标准的、用于深度学习 C++ 部署的编译脚本。它严谨地处理了路径、优化级别、依赖库链接以及运行时环境配置，
确保能够顺利生成一个高性能的 TensorRT + OpenCV 推理可执行文件。
COMMENT
