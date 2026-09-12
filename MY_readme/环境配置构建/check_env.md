## 1. 环境检查（只读，先跑这些看现状）

```bash
# 驱动 / GPU / 计算能力
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

# CUDA toolkit 版本
nvcc --version | grep -i release
ls /usr/local/cuda/include/cuda_runtime.h

# 工具链(C++编译需要)
g++ --version | head -1
cmake --version | head -1

# 系统
cat /etc/os-release | grep -E '^(ID|VERSION_ID)'
ldd --version | head -1

# 全盘找 trtexec 与 TensorRT 运行库
find / -name 'trtexec' -type f 2>/dev/null
find / -name 'libnvinfer.so*' 2>/dev/null

# pip 里的 TensorRT / torch 版本
python3 -m pip list | grep -iE 'tensorrt|torch|nvidia-cuda|onnx'
```

实测 pip 里的运行库位置（后续所有步骤的根路径）：

```bash
TL=/root/miniconda3/lib/python3.12/site-packages/tensorrt_libs        # libnvinfer.so.10 等
CU=/root/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib  # libcudart.so.12 (12.8)
```

## 2. C++编译所需的所有依赖自动检查脚本：
  check_build_env.sh  --->>列举25项和所在位置。