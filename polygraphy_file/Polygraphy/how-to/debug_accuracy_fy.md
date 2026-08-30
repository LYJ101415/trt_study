调试 TensorRT 精度问题
TensorRT 中的精度问题，尤其是在大型网络中，可能很难调试。
使其变得可控的一种方法是缩小问题规模或精确定位故障源。
本指南旨在提供一种通用的处理方法；其结构类似于扁平化的流程图——
在每个分支处，都会提供两个链接，以便您选择最符合当前情况的选项。
如果您使用的是 ONNX 模型，建议在继续之前先尝试对其进行清洗（sanitize），(../examples/cli/surgeon/02_folding_constants/);因为在某些情况下这可能会解决问题。

真实输入数据是否有影响？
某些模型可能对输入数据敏感。例如，使用真实输入可能比使用随机生成的输入获得更好的精度。Polygraphy 提供了多种提供真实输入数据的方法，详见 `run` 示例 05。(../examples/cli/run/05_comparing_with_custom_input_data/).

使用真实输入数据是否提高了精度？
是的，使用真实输入数据时精度可以接受。
这可能意味着没有 bug；而是您的模型对输入数据比较敏感。
否，即使使用真实输入数据，我仍然看到精度问题。
跳转至：问题是间歇性的吗？(#intermittent-or-not)
问题是间歇性的吗？
该问题是否在引擎构建之间间歇性出现？
是的，有时当我重新构建引擎时，精度问题会消失。
跳转至：调试间歇性精度问题 (#debugging-intermittent-accuracy-issues)
否，每次构建引擎时我都会遇到精度问题。
跳转至：逐层分析是否可行？(#is-layerwise-an-option)
调试间歇性精度问题
由于引擎构建过程是非确定性的，因此每次构建引擎时可能会选择不同的策略（即层实现）。当其中某个策略存在缺陷时，这可能表现为间歇性故障。Polygraphy 包含一个 debug build 子工具，可帮助您查找此类有问题的策略。
更多信息请参阅 `debug` 示例 01。(../examples/cli/debug/01_debugging_flaky_trt_tactics/).

您是否找到了导致失败的策略？
是的，我知道哪个策略有缺陷。
跳转至：您已拥有最小复现案例！(#you-have-a-minimal-failing-case)
否，故障可能不是间歇性的。
跳转至：逐层分析是否可行？(#is-layerwise-an-option)
逐层分析是否可行？
如果精度问题可以稳定复现，最好的下一步是找出导致故障的具体层。Polygraphy 提供了一种机制，可以将网络中的所有张量标记为输出以便进行比较；但这可能会影响 TensorRT 的优化过程。因此，我们需要确定在标记所有输出张量时是否仍能观察到精度问题。
有关如何在继续之前比较逐层输出的详细信息，请参阅此示例。(../examples/cli/run/01_comparing_frameworks/README.md#comparing-per-layer-outputs-between-onnx-runtime-and-tensorrt)
在比较逐层输出时，您是否能够复现精度故障？
是的，即使我在网络中标记了其他输出，故障仍然复现。
跳转至：提取故障子图 (#extracting-a-failing-subgraph)
否，标记其他输出后精度有所提高，或者标记其他输出后根本无法运行模型。
跳转至：缩减故障 ONNX 模型 (#reducing-a-failing-onnx-model)

提取故障子图
既然我们能够比较逐层输出，就应该能够通过查看输出比较日志来确定哪一层首先引入了错误。一旦知道哪一层有问题，我们就可以将其从模型中提取出来。
为了确定相关层的输入和输出张量，我们可以使用 polygraphy inspect model。详情请参阅以下示例之一：
TensorRT 网络 (../examples/cli/inspect/01_inspecting_a_tensorrt_network/)
ONNX 模型 (../examples/cli/inspect/03_inspecting_an_onnx_model/).

接下来，我们可以提取一个仅包含问题层的子图。
更多信息请参阅 `surgeon` 示例 01。(../examples/cli/surgeon/01_isolating_subgraphs/).
这个隔离的子图是否能复现问题？
是的，子图也失败了。
跳转至：您已拥有最小复现案例！(#you-have-a-minimal-failing-case)
否，子图运行正常。
跳转至：缩减故障 ONNX 模型 (#reducing-a-failing-onnx-model)
缩减故障 ONNX 模型
当我们无法通过逐层比较精确定位故障源时，可以使用暴力方法缩减 ONNX 模型——迭代生成越来越小的子图，以找到仍然失败的最小子图。debug reduce 工具有助于自动化这一过程。
更多信息请参阅 `debug` 示例 02。(../examples/cli/debug/02_reducing_failing_onnx_models/).

缩减后的模型是否失败？
是的，缩减后的模型失败了。
跳转至：您已拥有最小复现案例！(#you-have-a-minimal-failing-case)
否，缩减后的模型没有失败，或以不同的方式失败。
跳转至：仔细检查您的 Reduce 选项 (#double-check-your-reduce-options)
仔细检查您的 Reduce 选项
如果缩减后的模型不再失败，或以不同的方式失败，请确保您的 --check 命令是正确的。您可能还需要使用 --fail-regex 来确保在缩减模型时只考虑精度故障（而不是其他无关的故障）。
尝试再次缩减。
跳转至：缩减故障 ONNX 模型 (#reducing-a-failing-onnx-model)
您已拥有最小复现案例！
如果您进行到了这一步，那么您现在已经拥有了一个最小复现案例！进一步的调试应该会容易得多。
如果您是 TensorRT 开发人员，此时需要深入代码进行分析。
如果不是，请提交您的 bug 报告！