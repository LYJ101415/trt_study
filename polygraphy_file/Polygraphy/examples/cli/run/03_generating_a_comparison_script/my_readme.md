生成用于高级比较的脚本
简介
对于更高级的需求，您可能希望使用 API。
您无需从头开始编写脚本，而是可以使用 run 的 --gen-script 选项来创建一个 Python 脚本作为起点。
运行示例:
生成比较脚本：
polygraphy run identity.onnx --trt --onnxrt \
    --gen-script=compare_trt_onnxrt.py

生成的脚本将执行与 run 命令完全相同的操作。

运行比较脚本（可根据需要先对其进行修改）：
python3 compare_trt_onnxrt.py