#!/usr/bin/env bash
# 编译 gpu_ops.cu -> libgpu_ops.so（零 PyTorch，仅依赖 CUDA toolkit 的 nvcc + thrust）
set -e # 作用：启用 Shell 的"立即退出"模式（errexit）。逻辑：当脚本中任何一条命令返回非零退出码（即执行失败）时，
       # Shell 立即终止整个脚本，不再执行后续命令。为什么需要：编译脚本中，如果 nvcc 编译失败但脚本继续执行，
       # 后续的 echo "built ..." 会给出错误的成功提示，导致使用者误以为构建成功。set -e 确保"要么全成功，要么早报错"。

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" # 语法：${VAR:-default} 是 Bash 参数扩展，含义为"如果 VAR 未设置或为空，则使用 default"。
       # 逻辑：如果用户在环境变量中已设置了 CUDA_HOME（例如 export CUDA_HOME=/opt/cuda-12.6），则使用该值。如果未设置，则回退到默认路径 /usr/local/cuda-12.8。
       # 设计意图：提供合理的默认值，同时允许用户通过环境变量覆盖，适配不同机器上 CUDA 安装路径不同的情况。这是一种常见的可配置性模式。
NVCC="${CUDA_HOME}/bin/nvcc" # 作用：基于上一步确定的 CUDA_HOME，拼接出 nvcc 编译器的完整绝对路径。为什么不直接用 nvcc：
       # 系统 PATH 中可能存在多个 CUDA 版本的 nvcc，直接调用可能用到错误版本。使用绝对路径确保一定使用与 CUDA_HOME 匹配的编译器，避免版本不一致导致的链接错误或运行时崩溃。
       # 在某些 CI/容器环境中，nvcc 可能不在 PATH 中，绝对路径更可靠。

# RTX 4080 = sm_89 (Ada)。如需其他架构，改 ARCH 即可。
"""
作用：设置目标 GPU 架构（compute capability），同样支持环境变量覆盖。sm_89 的含义：NVIDIA Ada Lovelace 架构（RTX 4090/4080/4070 等）。
为什么重要：nvcc 需要知道目标架构才能生成对应的 PTX/SASS 机器码。不同架构的指令集、寄存器数量、共享内存大小不同，编译产物不通用。
sm_89 的代码在 sm_80（Ampere）上可以运行（通过 JIT 重编译 PTX），但反之不行，且性能不是最优。
常见架构对照：表格:
ARCH	GPU系列
sm_75	RTX 20xx (Turing)
sm_80	A100/A30 (Ampere)
sm_86	RTX 30xx (Ampere)
sm_89	RTX 40xx (Ada)
sm_90	H100/H200 (Hopper)
sm_100	B200 (Blackwell)
"""
ARCH="${ARCH:-sm_89}" 

"""
$0：当前脚本的路径（可能是相对路径如 ./build.sh 或绝对路径如 /home/user/project/build.sh）。
dirname "$0"：提取 $0 所在的目录部分。$(...)：命令替换，将 dirname 的输出作为 cd 的参数。
作用：将工作目录切换到脚本自身所在的目录。为什么关键：
用户可能从任意目录调用此脚本（如 cd /tmp && /home/user/project/build.sh）。
如果不 cd，nvcc gpu_ops.cu 会在 /tmp 下找文件，必然找不到。这确保了无论从哪里调用脚本，都能正确找到同目录下的 gpu_ops.cu。
注意："$0" 加了双引号，防止路径中含空格时被 word splitting 破坏。
"""
cd "$(dirname "$0")"

"""
nvcc 编译命令（核心）:这是整个脚本的核心，逐个参数解析："$NVCC"-->>调用前面定义的 nvcc 绝对路径。加引号防止路径含空格。
-O3 -->>最高级别优化。启用循环展开、向量化、内联、常量传播等激进优化。对于 GPU kernel 来说，-O3 对性能影响显著（相比 -O0 可能有数倍差距），
因为很多寄存器分配和指令调度依赖编译器优化。代价是编译时间稍长，但对于生产构建完全值得。
-std=c++17 -->>指定 Host 端 C++ 标准为 C++17。为什么需要 C++17：std::optional, std::string_view, structured bindings 等现代特性。
CUDA 12.x 的 Device 端也支持部分 C++17 特性（如 constexpr if）。如果不指定，nvcc 默认使用 C++14，可能导致编译错误。
-arch="${ARCH}" -->>指定目标 GPU 架构，告诉 nvcc 生成哪种 SASS（二进制机器码）。等价于 --gpu-architecture。
也可以指定多个架构实现 fat binary：-gencode arch=compute_89,code=sm_89 -gencode arch=compute_80,code=sm_80，这样同一个 .so 可在多种 GPU 上原生运行。
-Xcompiler -fPIC-Xcompiler：将后面的标志透传给 Host 端的 C++ 编译器（通常是 gcc/g++），而不是给 nvcc 的设备编译器。
-fPIC：Position Independent Code（位置无关代码）。
为什么必须：我们要生成的是 .so 共享库，不是可执行文件。共享库在加载时地址不固定，所有代码必须是位置无关的。
如果不加 -fPIC，链接阶段会报错：relocation R_X86_64_PC32 against symbol ... can not be used when making a shared object。
这个标志只影响 Host 代码（.cu 中的非 __global__/__device__ 函数），Device 代码天然就是 PIC 的。
-shared -->>告诉 nvcc 生成共享库（.so）而非可执行文件。nvcc 会将此标志传递给底层链接器（ld），等同于 gcc -shared。
生成的 .so 可以被 dlopen() / ctypes.CDLL() 动态加载。
-I"${CUDA_HOME}/include" -->>添加 CUDA 头文件搜索路径。确保 #include <cuda_runtime.h>, #include <thrust/sort.h> 等能找到对应版本的头文件。
与 CUDA_HOME 联动，保证头文件和编译器版本一致。
gpu_ops.cu -->>输入源文件。.cu 后缀告诉 nvcc 这是 CUDA 源文件，需要经过 CUDA 前端解析 __global__、<<<>>> 等语法。
-o libgpu_ops.so -->>指定输出文件名。lib 前缀 + .so 后缀是 Linux 共享库的命名惯例，方便 ctypes.CDLL("./libgpu_ops.so") 或 -lgpu_ops 链接。
\ (行尾反斜杠)。 Shell 续行符，将一条长命令拆成多行以提高可读性。注意：\ 后面不能有任何字符（包括空格），否则续行失败。
"""
"$NVCC" -O3 -std=c++17 -arch="${ARCH}" -Xcompiler -fPIC -shared \
    -I"${CUDA_HOME}/include" \
    gpu_ops.cu -o libgpu_ops.so

"""
作用：构建成功后打印确认信息，包含实际使用的架构。为什么有用：视觉确认构建完成。
显示实际生效的 ARCH 值（可能是默认的 sm_89，也可能是用户覆盖的值），便于排查"编译了但跑不起来"的架构不匹配问题。
因为有 set -e：只有 nvcc 真正成功时才会执行到这行，所以这条消息是可信的。
"""
echo "built libgpu_ops.so (arch=${ARCH})"
