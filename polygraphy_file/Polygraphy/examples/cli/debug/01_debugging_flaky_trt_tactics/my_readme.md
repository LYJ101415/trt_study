调试不稳定的 TensorRT 策略
重要提示：由于较新版本的 TensorRT 做出了一些未通过 IAlgorithmSelector 接口暴露的策略选择，本示例在这些版本上不再可靠。因此，下文概述的方法无法保证引擎构建的确定性。对于 TensorRT 8.7 及更高版本，您可以使用策略计时候存（Polygraphy 中的 --save-timing-cache 和 --load-timing-cache）来确保确定性，但这些文件是不透明的，因此无法被 inspect diff-tactics 解析。
简介
有时，TensorRT 中的某个策略（tactic）可能会产生错误的结果，或表现出其他缺陷行为。由于 TensorRT 构建器依赖于对策略进行计时，引擎构建过程是非确定性的，这可能导致策略缺陷表现为不稳定/间歇性的故障。
解决该问题的一种方法是多次运行构建器，并保存每次运行的策略重放文件。一旦我们收集到一组已知正常和已知异常的策略，就可以通过比较它们来确定哪个策略可能是错误来源。
debug build 子工具允许您自动化这一过程。
有关 debug 工具工作原理的更多详情，请参阅帮助输出：polygraphy debug -h 和 polygraphy debug build -h。
运行示例:
从 ONNX Runtime 生成基准输出（golden outputs）：
polygraphy run identity.onnx --onnxrt \
    --save-outputs golden.json

使用 debug build 重复构建 TensorRT 引擎，并将结果与基准输出进行比较，同时每次保存一个策略重放文件：
polygraphy debug build identity.onnx --fp16 --save-tactics replay.json \
    --artifacts-dir replays --artifacts replay.json --until=10 \
    --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json

让我们逐步解析此命令：
与其他 debug 子工具一样，debug build 在每次迭代中都会生成一个中间产物（默认为 ./polygraphy_debug.engine）。在本例中，该产物是一个 TensorRT 引擎。
提示：debug build 支持其他工具（如 convert 或 run）所支持的所有 TensorRT 构建器配置选项。
为了让 debug build 判断每个引擎是通过还是失败，我们需要提供一个 --check 命令。由于我们正在排查一个（模拟的）精度问题，可以使用 polygraphy run 将引擎的输出与我们的基准值进行比较。
提示：与其他 debug 子工具一样，这里也支持交互模式，只需省略 --check 参数即可使用。
与其他 debug 子工具不同，debug build 没有自动终止条件，因此我们需要提供 --until 选项以告知工具何时停止。该选项可以是迭代次数，也可以是 "good" 或 "bad"。在后一种情况下，工具会在分别找到第一个通过或失败的迭代后停止。
由于我们最终需要比较正常和异常的策略重放记录，我们指定 --save-tactics 来保存每次迭代的策略重放文件，然后使用 --artifacts 让 debug build 对其进行管理。这包括将它们分类到由 --artifacts-dir 指定的主产物目录下的 good 和 bad 子目录中。

使用 inspect diff-tactics 确定哪些策略可能存在问题：
polygraphy inspect diff-tactics --dir replays

注意：最后一步应报告无法确定潜在的异常策略，因为此时我们的 bad 目录应为空（否则请提交 TensorRT 问题反馈！）：
[I] Loaded 2 good tactic replays.
[I] Loaded 0 bad tactic replays.
[I] Could not determine potentially bad tactics. Try generating more tactic replay files?

延伸阅读:
有关 debug 工具的更多信息，以及适用于所有 debug 子工具的提示和技巧，请参阅 `debug` 子工具使用指南。
[how-to guide for `debug` subtools](../../../../how-to/use_debug_subtools_effectively.md).