一：检查模型与数据：
1. 检查模型算子与层结构是否正确可用：
  polygraphy check lint /root/my_FILE/myStudy_flle/best_PCB_F.onnx -o report.json

2.检查 ONNX 模型
polygraphy inspect model identity.onnx --show layers > model_tensorrt.txt  -->>重定向输出到文件并保存
polygraphy inspect model identity.onnx --show layers attrs weights  -->>来显示详细的层信息，包括层属性和权重

3.检查 TensorRT 引擎：
polygraphy inspect model dynamic_identity.engine \
    --show layers
：：显示更详细的层信息。
polygraphy inspect model dynamic_identity.engine \
    --show layers attrs

4. 自动将支持的格式转换为 TensorRT 网络，并将其显示出来:
  polygraphy inspect model identity.onnx \
    --show layers --display-as=trt
：：显示更详细的层信息，包括层属性和权重：
  polygraphy inspect model identity.onnx \
    --show layers attrs weights --display-as=trt

5. 检查输入、输出数据：
  polygraphy inspect data inputs.json --show-values
  polygraphy inspect data outputs.json --show-values

6. 检查中间层的 NaN 或无穷大值：
polygraphy run add_infinity.onnx --onnx-outputs mark all --onnxrt --validate

--onnx-outputs mark all -->>标记所有输出

二：模型框架间的比较：
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
  
