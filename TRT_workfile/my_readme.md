一：检查模型与数据：
1. 检查模型算子与层结构是否正确可用：
  polygraphy check lint /root/my_FILE/myStudy_flle/best_PCB_F.onnx -o report.json

2.检查 ONNX 模型
polygraphy inspect model /root/my_FILE/yolov8_int8.onnx --show layers > model_tensorrt.txt  -->>重定向输出到文件并保存
polygraphy inspect model /root/my_FILE/yolov8_int8.onnx --show layers attrs weights  -->>来显示详细的层信息，包括层属性和权重

3.检查 TensorRT 引擎：
polygraphy inspect model dynamic_identity.engine \
    --show layers
：：显示更详细的层信息。
polygraphy inspect model /root/my_FILE/yolov8_int8.engine \
    --show layers attrs 

4. 自动将支持的格式转换为 TensorRT 网络，并将其显示出来:
  polygraphy inspect model /root/my_FILE/yolov8_int8.onnx \
    --show layers --display-as=trt
：：显示更详细的层信息，包括层属性和权重：
  polygraphy inspect model root/my_FILE/yolov8_int8.onnx \
    --show layers attrs weights --display-as=trt

5. 检查输入、输出数据：
  polygraphy inspect data inputs.json --show-values
  polygraphy inspect data outputs.json --show-values

二：将 ONNX 模型转换为 FP16:
当调试以 FP32 训练的 ONNX 模型在 TensorRT 中使用降低精度优化（--fp16 和 --tf32 标志）时出现的精度问题，将模型转换为 FP16 并在 ONNX Runtime 下运行可能会有所帮助。这有助于检查是否存在因降低精度运行模型而固有的问题。
运行示例:
将模型转换为 FP16：
polygraphy convert --fp-to-fp16 -o identity_fp16.onnx identity.onnx

[可选] 检查生成的模型：
polygraphy inspect model identity_fp16.onnx

[可选] 在 ONNX Runtime 下分别运行 FP32 和 FP16 模型，然后比较结果：
polygraphy run --onnxrt identity.onnx \
   --save-inputs inputs.json --save-outputs outputs_fp32.json

polygraphy run --onnxrt identity_fp16.onnx \
   --load-inputs inputs.json --load-outputs outputs_fp32.json \
   --atol 0.001 --rtol 0.001

[可选] 检查 FP16 模型的任何中间输出是否包含 NaN 或无穷大值（参见检查中间层的 NaN 或无穷大值）：
polygraphy run --onnxrt identity_fp16.onnx --onnx-outputs mark all --validate

二：模型框架间的比较：对于算子都支持的onnx模型先进行fp16转换检测看精度是否可行。
1. 对比验证 TensorRT（FP16模式）与 ONNX Runtime 的推理结果是否一致
polygraphy run dynamic_identity.onnx --trt --fp16 --onnxrt \
    --input-shapes images:[1,3,640,640] \
    --load-inputs inputs.json
    --atol 0.001 --rtol 0.001 --check-error-stat median

--input-shapes images:[1,3,640,640] -->>用于动态模型固定输入形状，images要符合模型输入名。
--load-inputs inputs.json -->>采用真实数据输入，inputs.json的创建使用 make_input_json.py文件代码
--atol 0.001 --rtol 0.001 -->>绝对误差与相对误差
--check-error-stat median -->>统计采用的数据（mean、median、max），它们分别比较整个张量的平均、中位数和最大绝对/相对误差

2. 比较 ONNX-Runtime 与 TensorRT 的逐层输出:
polygraphy run /root/my_FILE/myStudy_flle/best_PCB_F.onnx --trt --fp16 --onnxrt \
     --load-inputs /root/my_FILE/images_inputs.json \
     --trt-outputs mark all \
     --onnx-outputs mark all \
     --atol 0.01 --rtol 0.01 \
     --check-error-stat median \
     --fail-fast

 --fail-fast -->>遇到第一个不通过的节点即退出

 3. 保存模型运行的输入和输出值：
    polygraphy run identity.onnx --onnxrt \
    --save-inputs inputs.json --save-outputs outputs.json

    保存模型运行的输入和输出值的用法：
    （1）当模型在不同环境和硬件中使用时，可以比较不同环境下的输入和输出值，以确定是否存在差异。
    polygraphy run identity.onnx --onnxrt \
    --load-inputs inputs.json --load-outputs outputs.json \
    --atol 0.001 --rtol 0.001 --check-error-stat median
    （2）可以使用此技术来比较不同的模型，例如 TensorRT 引擎和 ONNX 模型（前提是它们具有匹配的输出）。
        将 ONNX 模型转换为 TensorRT 引擎并保存到磁盘：
        polygraphy convert identity.onnx -o identity.engine
        在 Polygraphy 中运行保存的引擎，使用 ONNX-Runtime 运行中保存的输入作为引擎的输入，并将引擎的输出与保存的 ONNX-Runtime 输出进行比较：
    polygraphy run --trt identity.engine --model-type=engine \
    --load-inputs inputs.json --load-outputs outputs.json \
    --atol 0.001 --rtol 0.001 --check-error-stat median

4. 使用自定义输入数据进行比较:(D:\CV_FILE\my_FILE\polygraphy_file\tools\Polygraphy\examples\cli\run\05_comparing_with_custom_input_data)
方案 1：使用数据加载器脚本：
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --data-loader-script data_loader.py

：：data_loader.py是一个数据处理函数，内置的入口函数必须用load_data()这个函数名

方案 2：使用包含已保存输入的 JSON 文件：需要占用很大显存，图片数据大时不适用
polygraphy run dynamic_identity.onnx --trt --onnxrt \
    --load-inputs custom_inputs.json

三：对原始的onnx模型进行int8量化：
1. 采用quant_int8.py文件进行量化，导出的为插入Q/DQ节点的onnx模型。
2. 采用test_onnx.py进行这个模型的精度验证，精度不达要求，进行逐层精度验证。
3. 采用layer_eval.py进行逐层精度验证，原理为逐层回退为fp32后计算精度与基准精度对比，排序精度损失。
4. 找到精度敏感层后，直接去修改量化好的onnx模型，把插入的Q/DQ节点去除。
5. 采用build_engine_detailed.py进行trt引擎构建，这个文件可以支持以后的层结构输出信息很详细、完整。
6. 采用test_trt.py文件对trt引擎进行精度验证。

四：trt_Profiling文件夹是trtexec工具的一些调试命令，analyse_json文件夹是对输入输出保存的.json文件解读的代码，
   make_inputs_json文件夹是制作输入文件保存为.json文件的代码，还有加载数据的函数（可直接传入使用）
