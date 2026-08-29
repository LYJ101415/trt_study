
在远程服务器上创建虚拟环境：
  初始化：conda init
  创建环境：conda create -n trt_env python=3.12
  重新加载配置文件：source ~/.bashrc
  然后再尝试激活虚拟环境：conda activate trt_env
  卸载环境：conda remove -n trt_poly_env --all

  1. 安装Pytorch,torchvision等: 
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124


一：tensorrt部署远程开发的环境配置
1. 采用基础镜像环境：PyTorch==2.8.0；Python==3.12(ubuntu22.04)；CUDA==12.8

2. 安装tensorrt: pip install tensorrt-cu12==10.3.0
  
3. 拉取目前能安装的包的版本号：pip index versions onnxruntime-gpu
查看未安装包的依赖：pip install onnxruntime-gpu==1.20.2 --dry-run

4. 安装onnxruntime-gpu：pip install onnxruntime-gpu==1.20.2
查看已安装包的依赖：pip show onnxruntime-gpu
想看更详细的层级依赖（即依赖的依赖），可以使用 pipdeptree 工具：pip install pipdeptree
  pipdeptree -p onnxruntime-gpu  <<---这会以树状图的形式展示 onnxruntime-gpu 及其所有子依赖。

5. 安装onnx: pip install onnx==1.18.0
6. 安装OpenCV：pip install opencv-python==4.10.0.84
7. 安装polygrapy: pip install polygraphy==0.49.26 colored

8. 安装nvidia-modelopt，使用 --no-deps 跳过依赖自动安装，因为核心环境（PyTorch 2.12.1、TensorRT 10.3.0 等）已经搭建完美：pip install nvidia-modelopt==0.42.0 --no-deps
注意！！！nvidia-modelopt==0.42.0 需要pytorch -->> torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
@@ 补充nvidia-modelopt的完整依赖：
pip install ninja nvidia-ml-py pulp "pydantic>=2.0" regex rich safetensors scipy tqdm
  如果需要用到 Hugging Face Transformers 或 Diffusers 相关的优化功能，建议直接安装包含所有可选依赖的完整包： pip install "nvidia-modelopt[all]==0.42.0"  <<---环境要求：该版本要求 Python >= 3.8 且 CUDA >= 11.8，完全契合你当前的 Python 3.12 和 CUDA 12.4 环境。

@@ nvidia-modelopt的其它Onnx依赖：onnx-graphsurgeon==0.6.1 onnxscript==0.7.1  onnxslim==0.1.95 lief==1.0.0   

怎么采用环境安装文件“install_env.sh”来装环境：
1.让系统知道这个文件是一个可以运行的程序：chmod +x install_env.sh
2.使用 ./ 加上文件名来运行它：./install_env.sh
3.如果脚本中包含需要管理员权限的命令，执行： sudo ./install_env.sh
  如果不想修改文件权限，也可以直接使用 bash 命令来运行它：bash install_env.sh。

安装trtexec:
1. # 下载 NGC tarball（4.1GB，但只提取一个文件），注意安装兼容版本的 CUDA 和 cuDNN和tensorrt版本。
wget -q https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz

2. 解压需要的文件（大概6G以上）：
tar xzf TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz TensorRT-10.3.0.26/targets/x86_64-linux-gnu/

3. 加上环境变量，严格以下操作：
# 1. 在解压目录下递归搜索 trtexec
find ~/my_FILE/TensorRT-10.3.0.26 -name "trtexec" -type f 2>/dev/null
  例如输出：/root/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/bin/trtexec
# 同时搜索 libnvinfer.so
find ~/my_FILE/TensorRT-10.3.0.26 -name "libnvinfer.so*" -type f 2>/dev/null
  例如输出：/root/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib/libnvinfer.so.10.3.0
           /root/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib/stubs/libnvinfer.so

# 2. 写入正确的环境变量
cat >> ~/.bashrc << 'EOF'
export TENSORRT_HOME=~/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu
export PATH=$TENSORRT_HOME/bin:$PATH
export LD_LIBRARY_PATH=$TENSORRT_HOME/lib:$LD_LIBRARY_PATH
EOF

# 3. 立即生效，重新进入运行终端：
source ~/.bashrc

# 4. 验证
trtexec --help | head -3   -->>检查trtexec有没有

最终验证 trtexec 可用：无输出->所有依赖库都已就位，可以直接使用；有输出 → 根据缺失的库名补解对应文件即可
export LD_LIBRARY_PATH=~/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib:$LD_LIBRARY_PATH
ldd ~/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/bin/trtexec | grep "not found"

查看文件大小：du -sh ~/my_FILE/TensorRT-10.3.0.26/targets/x86_64-linux-gnu/lib/

# 检查 gzip 完整性（会扫描整个文件，需几分钟）
gzip -t TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz && echo "OK" || echo "CORRUPTED"

6. include文件夹，用于TensorRT C/C++ 头文件（NvInfer.h 等），编译自定义插件、C++ 推理代码时需要；
   需要时重新安装。
   

二，一些检查等命令行操作：
1. 确认当前驱动版本：nvidia-smi

2. 确认文件真实路径和文件大小情况以确定文件传输是否出错（经常会出现）：
    ls -lh /root/my_FILE/polygraphy_file/myStudy_flle/best_PCB_F.onnx
3. 程序突然奔溃的检查命令;如果遇到的是普通的 Python 报错，直接看终端输出的常规 Traceback 即可；但如果遇到的是程序突然闪退,没有任何 Python 报错信息的 Segmentation fault，加上 -X faulthandler 就是最强大的“黑匣子”排查工具。
   python -X faulthandler /root/my_FILE/CV_yolov8/nncf_ACQ/pycocotools_test.py

4. 输出onnxruntime是否在使用GPU运行，命令如下：
python -c "import onnxruntime as ort; print('可用 Providers:', ort.get_available_providers())"

5. 输出模型的层信息：polygraphy inspect model /root/my_FILE/yolov8_int8.engine --show layers   attrs

三：模型的检测
1. 影响精度的节点排查
 步骤 1：准备并简化 ONNX 模型；如果您的模型有动态维度，先固定形状并折叠常量，这有助于后续的排查。
  polygraphy surgeon sanitize your_model.onnx -o folded.onnx --fold-constants \
    --override-input-shapes input_name:[1,3,640,640]
 步骤 2：生成 FP32 基准数据：
  用随机数据，inputs.json：
  polygraphy run folded.onnx --onnxrt \
    --save-inputs inputs.json \
    --onnx-outputs mark all --save-outputs layerwise_golden.json
  用真实数据，real_inputs.json：
  polygraphy run folded.onnx --onnxrt \
    --load-inputs real_inputs.json \
    --onnx-outputs mark all --save-outputs layerwise_golden.json

  将输入和逐层输出合并，供裁剪时的子图使用：
  polygraphy data merge inputs.json layerwise_golden.json -o layerwise_inputs.json

 步骤 3：使用 debug reduce 定位精度差异节点；这是最关键的一步。我们将量化后的模型（假设您将其转换为了 TensorRT 引擎，或者直接用 ONNX INT8 模型）与 FP32 的标准答案进行对比。如果精度差异超出容忍度，就视为“失败”。
  polygraphy debug reduce folded.onnx -o initial_reduced.onnx --mode=bisect --load-inputs layerwise_inputs.json \
    --check polygraphy run polygraphy_debug.onnx --trt --int8 \
            --load-inputs layerwise_inputs.json --load-outputs layerwise_golden.json --atol 0.01 --rtol 0.01

   # 最终完善的安全命令（强烈推荐）
polygraphy debug reduce /root/my_FILE/polygraphy_file/myStudy_flle/best_PCB_F.onnx -o initial_reduced.onnx --mode=bisect \
    --load-inputs layerwise_inputs.json \
    --check polygraphy run polygraphy_debug.onnx --trt --int8 \
            --load-inputs calibration_data_50.json \
            --load-outputs layerwise_golden.json --atol 0.5 --rtol 0.5

 步骤 4：进一步精简子图；经过上一步的 bisect（二分法），您会得到一个较小的子图，但可能还包含一些正常的节点。使用 linear（线性法）可以逐个节点剔除，找到最小故障子图：
  polygraphy debug reduce initial_reduced.onnx -o final_reduced.onnx --mode=linear --load-inputs layerwise_inputs.json \
    --check polygraphy run polygraphy_debug.onnx --trt \
            --load-inputs layerwise_inputs.json --load-outputs layerwise_golden.json --atol 0.01 --rtol 0.01
 步骤 5：查看最终导致精度崩溃的算子：
  polygraphy inspect model final_reduced.onnx --show layers


  