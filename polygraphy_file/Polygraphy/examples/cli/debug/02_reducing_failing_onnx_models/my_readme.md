缩减失败的 ONNX 模型
简介
当模型因任何原因失败时（例如 TensorRT 中的精度问题），将其缩减为触发该失败的最小子图通常非常有用。这有助于更轻松地定位失败的根本原因。
实现这一目标的一种方法是逐步生成原始 ONNX 模型的更小子图。在每次迭代中，我们可以检查子图是正常工作还是仍然失败；一旦我们得到一个正常工作的子图，就可以确定上一次迭代生成的子图即为最小的失败子图。
debug reduce 子工具允许我们将此过程自动化。
运行示例
在本示例中，我们假设模型 (./model.onnx) 在 TensorRT 中存在精度问题。由于该模型实际上在 TensorRT 中可以正常运行（如果不能，请提交错误报告！），我们将先列出正常情况下应运行的命令，随后提供模拟失败的命令，以便让您直观了解该工具的实际使用效果。
我们的模拟失败将在模型中包含 Mul 节点时触发：
因此，最终缩减后的模型应仅包含 Mul 节点（因为其他节点不会导致失败）。

对于使用动态输入形状或包含形状运算的模型，需冻结输入形状并折叠形状运算：
polygraphy surgeon sanitize model.onnx -o folded.onnx --fold-constants \
    --override-input-shapes x0:[1,3,224,224] x1:[1,3,224,224]

假设 ONNX Runtime 能给出正确的输出。我们首先为网络中的每个张量生成基准值（golden values），同时保存所使用的输入数据：
polygraphy run folded.onnx --onnxrt \
    --save-inputs inputs.json \
    --onnx-outputs mark all --save-outputs layerwise_golden.json

然后，使用 data to-input 子工具将输入和逐层输出合并为一个逐层输入文件（下一步将说明为何必须这样做）：
polygraphy data to-input inputs.json layerwise_golden.json -o layerwise_inputs.json

接下来，以 bisect（二分）模式使用 debug reduce：
polygraphy debug reduce folded.onnx -o initial_reduced.onnx --mode=bisect --load-inputs layerwise_inputs.json \
    --check polygraphy run polygraphy_debug.onnx --trt \
            --load-inputs layerwise_inputs.json --load-outputs layerwise_golden.json

让我们逐步解析此命令：
与其他 debug 子工具一样，debug reduce 在每次迭代中都会生成一个中间产物（默认为 ./polygraphy_debug.onnx）。在本例中，该产物是原始 ONNX 模型的某个子图。
为了让 debug reduce 判断每个子图是失败还是通过，我们需要提供一个 --check 命令。由于我们正在排查精度问题，可以使用 polygraphy run 与之前生成的基准输出进行比较。
提示：与其他 debug 子工具一样，这里也支持交互模式，只需省略 --check 参数即可使用。
在 --check 命令中，我们通过 --load-inputs 提供逐层输入数据。否则，polygraphy run 会为子图张量生成新的输入，这些新输入可能与我们生成基准数据时这些张量的值不匹配。另一种方法是在 debug reduce 的每次迭代期间运行参考实现（此处为 ONNX Runtime），而不是提前运行。

由于我们使用的是非默认输入数据，因此还需通过 --load-inputs 直接向 debug reduce 命令本身提供逐层输入（除了提供给 --check 命令之外）。这对于具有多个并行分支（指模型中的路径而非控制流）的模型非常重要，例如：
 inp0  inp1
  |     |
 Abs   Abs
    \ /
    Sum
     |
    out
在这种情况下，debug reduce 需要能够将其中一个分支替换为常量。为此，它需要知道您正在使用的输入数据，以便用正确的值进行替换。虽然我们在此处使用了文件，但也可以通过 CLI 用户指南 中介绍的任何其他 Polygraphy 数据加载器参数来提供输入数据。

如果您不确定是否需要提供数据加载器，当 debug reduce 尝试替换分支时会发出如下警告：
[W]     This model includes multiple branches/paths. In order to continue reducing, one branch needs to be folded away.
        Please ensure that you have provided a data loader argument to `debug reduce` if your `--check` command is using a non-default data loader.
        Not doing so may result in false negatives!
我们指定了 -o 选项，以便将缩减后的模型写入 initial_reduced.onnx。

模拟失败： 我们可以结合使用 polygraphy inspect model 和 --fail-regex，在模型包含 Mul 节点时触发失败：
polygraphy debug reduce folded.onnx -o initial_reduced.onnx --mode=bisect \
    --fail-regex "Op: Mul" \
    --check polygraphy inspect model polygraphy_debug.onnx --show layers

[可选] 作为完整性检查，我们可以检查缩减后的模型以确保其确实包含 Mul 节点：
polygraphy inspect model initial_reduced.onnx --show layers

由于上一步使用了 bisect 模式，模型可能尚未达到最简状态。为了进一步精简，我们将以 linear（线性）模式再次运行 debug reduce：
polygraphy debug reduce initial_reduced.onnx -o final_reduced.onnx --mode=linear --load-inputs layerwise_inputs.json \
    --check polygraphy run polygraphy_debug.onnx --trt \
            --load-inputs layerwise_inputs.json --load-outputs layerwise_golden.json

模拟失败： 我们将使用与之前相同的技术：
polygraphy debug reduce initial_reduced.onnx -o final_reduced.onnx --mode=linear \
    --fail-regex "Op: Mul" \
    --check polygraphy inspect model polygraphy_debug.onnx --show layers

[可选] 此时，final_reduced.onnx 应仅包含导致失败的节点——即 Mul 节点。我们可以使用 inspect model 进行验证：
polygraphy inspect model final_reduced.onnx --show layers

延伸阅读
有关 debug 工具工作原理的更多详情，请参阅帮助输出：polygraphy debug -h 和 polygraphy debug reduce -h。
另请参阅 `debug reduce` 操作指南，获取更多信息、提示和技巧。
[`debug reduce` how-to guide](../../../../how-to/use_debug_reduce_effectively.md)