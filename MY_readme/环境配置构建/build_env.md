## 1. tensorrt的版本兼容性查找：

三个免登录的查找入口（都能查到版本，例如： 10.8.0）
GitHub 仓库 tag（查头文件）：github.com/NVIDIA/TensorRT/tags —— 有 v10.8.0，include/ 下就是 C++ 头文件。但 GitHub 只有源码+头文件，没有 trtexec 二进制，也没有 tar 包。
NVIDIA CUDA apt 源（查 trtexec/头文件/库，全版本都有）：https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/Packages.gz，里面明确有 libnvinfer-bin=10.8.0.43-1+cuda12.8、libnvinfer-dev=10.8.0.43-1+cuda12.8、libnvinfer10=10.8.0.43-1+cuda12.8。
PyPI（查 pip 包）：pypi.org/project/tensorrt-cu12/ —— 你的 10.8.0 就在这里。

## 2. 环境配置构建：
一：tensorrt部署远程开发的环境配置：

1. 采用基础镜像环境：PyTorch==2.8.0；Python==3.12(ubuntu22.04)；CUDA==12.8

2. 安装tensorrt: pip install tensorrt==10.8.0.43 --extra-index-url https://pypi.nvidia.com
  
3. 安装onnxruntime-gpu：pip install onnxruntime-gpu==1.20.2

（ 相关知识：拉取目前能安装的包的版本号：pip index versions onnxruntime-gpu
  查看未安装包的依赖：pip install onnxruntime-gpu==1.20.2 --dry-run
  查看已安装包的依赖：pip show onnxruntime-gpu
  想看更详细的层级依赖（即依赖的依赖），可以使用 pipdeptree 工具：pip install pipdeptree
  pipdeptree -p onnxruntime-gpu  <<---这会以树状图的形式展示 onnxruntime-gpu 及其所有子依赖。
）

4. 安装onnx: pip install onnx==1.18.0

5. 安装OpenCV：pip install opencv-python==4.10.0.84

6. 安装polygrapy: pip install polygraphy==0.49.26 colored

7. 安装nvidia-modelopt：pip install nvidia-modelopt==0.42.0 --no-deps

（ 相关知识：使用 --no-deps 跳过依赖自动安装，因为核心环境（PyTorch 2.12.1、TensorRT 10.3.0 等）已经搭建完美，先跳过依赖安装防止其依赖覆盖原始环境的python,tensorrt等版本，后续再装其普通依赖。
）

8. 补充nvidia-modelopt的依赖：
  pip install ninja nvidia-ml-py pulp "pydantic>=2.0" regex rich safetensors scipy tqdm

（ 相关知识：如果需要用到 Hugging Face Transformers 或 Diffusers 相关的优化功能，建议直接安装包含所有可选依赖的完整包： pip install "nvidia-modelopt[all]==0.42.0"  <<---环境要求：该版本要求 Python >= 3.8 且 CUDA >= 11.8，完全契合你当前的 Python 3.12 和 CUDA 12.4 环境。
）

9. 补充nvidia-modelopt的其它Onnx依赖：
 pip install onnx-graphsurgeon==0.6.1 onnxscript==0.7.1  onnxslim==0.1.95 lief==1.0.0   

10. 下载安装trtexec二进制包和 C++ 编译依赖如下：
# 安装 trtexec（只提二进制，复用 pip 的 .so），--> 不下载 1.3GB 的 apt 全家桶，只提 458KB 的 deb（里面就一个 1.7MB 的 trtexec）。

（1）下载 deb（实测 URL 可用，返回 301 后正常下载）：
  wget -O /tmp/trtbin.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/libnvinfer-bin_10.8.0.43-1+cuda12.8_amd64.deb

（2）解包并装到 /usr/local/bin ：
    rm -rf /tmp/trtbin && mkdir -p /tmp/trtbin
    dpkg -x /tmp/trtbin.deb /tmp/trtbin
    cp /tmp/trtbin/usr/src/tensorrt/bin/trtexec /usr/local/bin/trtexec
    chmod +x /usr/local/bin/trtexec

（3）验证文件：ls -la /usr/local/bin/trtexec

（4）补无版本 `.so` 软链（层分析必需）：

（ 相关知识：pip 的 `tensorrt_libs` 里只有带版本号的文件（如 `libnvinfer_builder_resource.so.10.8.0`），
 缺无后缀软链。`--dumpLayerInfo` / `--dumpProfile` 依赖 builder，运行时 `dlopen` 找的是
 无后缀名，不补软链会导致层分析直接失败。
 ）

    ```bash
    TL=/root/miniconda3/lib/python3.12/site-packages/tensorrt_libs
    ln -sfv libnvinfer_builder_resource.so.10.8.0 "$TL/libnvinfer_builder_resource.so"
    ln -sfv libnvinfer_plugin.so.10              "$TL/libnvinfer_plugin.so"
    ln -sfv libnvonnxparser.so.10                "$TL/libnvonnxparser.so"
    ```
    
（5） 配置 LD_LIBRARY_PATH

（ 相关知识：pip 的 `.so` 默认不在系统库搜索路径（pip 只配给 Python），trtexec 和 C++ 二进制都靠它找库。
 把 tensorrt_libs + pip 的 CUDA 12.8 runtime 一起加进 `~/.bashrc`。
 ）

```bash
TL=/root/miniconda3/lib/python3.12/site-packages/tensorrt_libs
CU=/root/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib

echo "export LD_LIBRARY_PATH=$TL:$CU:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

(6) 拉取 C++ 头文件（⚠️ 用 jsDelivr，别用 raw.github）

( 相关知识 **实测踩坑**：`raw.githubusercontent.com` 在本机**直接 GET 会一直挂起**（HEAD 通、GET 断），
 必须换 **jsDelivr** 镜像拉取，否则会卡死。
）
```bash
TRT_INC=/root/my_FILE/trt_include
mkdir -p "$INC"
BASE="https://cdn.jsdelivr.net/gh/NVIDIA/TensorRT@v10.8.0/include"
for h in NvInfer.h NvInferImpl.h NvInferLegacyDims.h NvInferPlugin.h \
         NvInferPluginBase.h NvInferPluginUtils.h NvInferRuntime.h \
         NvInferRuntimeBase.h NvInferRuntimeCommon.h NvInferRuntimePlugin.h \
         NvInferVersion.h NvOnnxConfig.h; do
  curl -fsSL -o "$TRT_INC/$h" "$BASE/$h"
done
ls "$TRT_INC"/*.h | wc -l   # 应为 12
```

> 若以后 C++ 还要用 ONNX 解析 / 插件，需再补 `NvOnnxParser.h`、`NvInferPlugin.h`（同目录拉取）。

---

（7）验证：

```bash
TL=/root/miniconda3/lib/python3.12/site-packages/tensorrt_libs
CU=/root/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH="$TL:$CU:$LD_LIBRARY_PATH"

which trtexec                       # /usr/local/bin/trtexec
trtexec --version 2>&1 | head -1    # 应见: TensorRT v100800 [b43]  (= 10.8.0.43)
trtexec --help >/dev/null 2>&1 && echo "OK: libnvinfer 加载正常"
```

> 说明：`trtexec --version` 在 TRT 10.8 里会打印版本 banner 后跟着打完整帮助、最后因
> 「无 model」报 `FAILED`，属正常现象；只要首行出现 `TensorRT v100800 [b43]` 即代表成功。
---

（8）补 `.so` 无版本软链（链接必需）

（ 相关知识：pip 只发 `libcudart.so.12` / `libnvinfer.so.10` 这类带版本文件，`-lcudart` / `-lnvinfer`
 找的是无版本名，必须补软链。
 ）

```bash
TL=/root/miniconda3/lib/python3.12/site-packages/tensorrt_libs
CU_LIB=/root/miniconda3/lib/python3.12/site-packages/nvidia/cuda_runtime/lib

ln -sfv libnvinfer.so.10 "$TL/libnvinfer.so"          # -lnvinfer
ln -sfv libcudart.so.12  "$CU_LIB/libcudart.so"       # -lcudart
# 若 C++ 里还要用 ONNX parser / 插件，再补：
ln -sfv libnvinfer_plugin.so.10  "$TL/libnvinfer_plugin.so"
ln -sfv libnvonnxparser.so.10    "$TL/libnvonnxparser.so"
```
