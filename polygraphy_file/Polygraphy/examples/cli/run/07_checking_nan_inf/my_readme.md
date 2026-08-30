检查中间层的 NaN 或无穷大值
简介
在使用 Polygraphy 调试模型精度问题时，逐层检查输出以发现潜在问题往往很有帮助。Polygraphy 的 run 子工具提供了一个实用的标志 --validate，可以快速诊断存在问题的中间层输出。
本示例演示了如何对一个故意产生无穷大输出的模型使用该标志，该模型通过向输入张量加上无穷大值来实现这一行为。
运行示例:
polygraphy run add_infinity.onnx --onnx-outputs mark all --onnxrt --validate

您应该会看到类似如下的输出：
[I] onnxrt-runner-N0-05/13/22-22:35:48  | 在 0.1326 ms 内完成 1 次迭代 | 平均推理时间：0.1326 ms。
[I] 输出验证 | 运行器：['onnxrt-runner-N0-05/13/22-22:35:48']
[I]     onnxrt-runner-N0-05/13/22-22:35:48  | 正在验证输出：B (check_inf=True, check_nan=True)
[I]         mean=inf, std-dev=nan, var=nan, median=inf, min=inf at (0,), max=inf at (0,), avg-magnitude=inf
[E]         检测到 Inf | 在此输出中遇到一个或多个非有限值
[I]         注意：使用 -vv 或将日志详细程度设置为 EXTRA_VERBOSE 以显示非有限值
[E]         失败 | 在输出中检测到错误：B
[E]     失败 | 输出验证

另请参阅
调试 TensorRT 精度问题
[Debugging TensorRT Accuracy Issues](../../../../how-to/debug_accuracy.md)