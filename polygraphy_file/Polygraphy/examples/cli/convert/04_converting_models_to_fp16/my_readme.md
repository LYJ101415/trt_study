将 ONNX 模型转换为 FP16
简介
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

另请参阅:
* [Comparing Across Runs](../../../../examples/cli/run/02_comparing_across_runs)
* [Checking for Intermediate NaN or Infinities](../../../../examples/cli/run/07_checking_nan_inf)
* [Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md)