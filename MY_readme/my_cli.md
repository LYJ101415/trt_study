一：采用输入数据images_inputs.json进行对/root/my_FILE/yolov8_int8.onnx模型进行逐层输出数据检查：
    检查有无NAF或 Inf 异常值;并保存为outputs.json供离线检查。
1. 先运行 make_inputs_json.py 生成由原图片生成的输入数据文件 images_inputs.json
2. 运行以下命令进行检测：
polygraphy run /root/my_FILE/yolov8_int8.onnx --onnxrt \
    --load-inputs images_inputs.json \
    --onnx-outputs mark all \
    --validate \
    --save-outputs int8_outputs.json

二：使用fp32模型的输出做为基准，对比int8模型输出，找出精度损失层。
1. # 保存 FP32 模型的输出作为基准
polygraphy run /root/my_FILE/myStudy_flle/best_PCB_F.onnx --onnxrt \
--load-inputs images_inputs.json \
--onnx-outputs mark all \
--validate \
--save-outputs fp32_outputs.json

检查已经序列化好的.engine模型的层结构（大kernel的组成算子和数量多少）：
polygraphy inspect model yolov8.engine --show layers attrs

三：当量化时校准出现在CPU上运行时，表示cuDNN没有进入环境：
1. 用 find 全局搜索 cuDNN，看看是否在环境变量中
2. 使用用户级的 .bashrc 或 .bash_profile 文件，每次你登录 SSH 时，这个变量都会自动加载。
echo 'export LD_LIBRARY_PATH=/root/miniconda3/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc


