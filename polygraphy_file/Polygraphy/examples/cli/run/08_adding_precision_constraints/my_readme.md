添加精度约束
简介
当使用 FP32 训练的模型构建 TensorRT 引擎并利用降低精度优化时，模型中的某些层可能需要被约束为在 FP32 下运行，以保持可接受的精度。
以下示例演示了如何选择性地约束网络中指定层的精度。提供的 ONNX 模型执行以下操作：
通过右乘一个旋转 90 度的单位矩阵，将输入水平翻转；
向翻转后的输入加上 FP16_MAX，然后从结果中减去 FP16_MAX；
通过右乘旋转后的单位矩阵，将减法的输出水平翻转。
如果 x 为正数，由于数值将超出 FP16 的表示范围（这是设计使然），步骤 (2) 需要在 FP32 中执行才能达到可接受的精度。然而，当启用 FP16 优化且未设置约束时，TensorRT 并不知道 x 将使用的数值范围，通常会选择在整个过程中以 FP16 运行所有步骤：
步骤 (1) 和 (3) 中的 GEMM 运算在 FP16 下比在 FP32 下运行更快（对于足够大的问题规模）。
步骤 (2) 中的逐点运算在 FP16 下运行更快，并且保持数据为 FP16 格式可以避免额外的 FP32 格式转换开销。
因此，您需要对 TensorRT 网络中允许的精度进行约束，以便 TensorRT 在引擎中分配层精度时做出适当的选择。
Polygraphy 命令行工具提供了多种约束层精度的方法：
--layer-precisions 选项允许您为单个层设置精度。
网络后处理脚本允许您以编程方式修改由 Polygraphy 解析或以其他方式生成的 TensorRT 网络。
网络加载器脚本允许您使用 TensorRT Python API 手动构建整个 TensorRT 网络。在网络构建过程中，您可以按需设置层精度。
运行示例
警告： 此示例需要 TensorRT 8.4 或更高版本。
使用 --layer-precisions 选项
运行以下命令，比较使用 FP16 优化的 TensorRT 与 FP32 下的 ONNX-Runtime 运行该模型的结果：
polygraphy run needs_constraints.onnx \
    --trt --fp16 --onnxrt --val-range x:[1,2] \
    --layer-precisions Add:float16 Sub:float32 --precision-constraints prefer \
    --check-error-stat median

为了增加此命令因上述原因而失败的可能性，我们将强制 Add 节点以 FP16 精度运行，随后的 Sub 节点以 FP32 运行。这将阻止它们被融合，并导致 Add 的输出溢出 FP16 范围。
使用网络后处理脚本约束精度
另一种选择是使用 TensorRT 网络后处理脚本对已解析的网络应用精度约束。
使用提供的网络后处理脚本 add_constraints.py 来约束模型中的精度：
polygraphy run needs_constraints.onnx --onnxrt --trt --fp16 --precision-constraints obey \
    --val-range x:[1,2] --check-error-stat median \
    --trt-network-postprocess-script ./add_constraints.py

提示：您可以使用 --trt-npps 作为 --trt-network-postprocess-script 的简写。
默认情况下，Polygraphy 会在脚本中查找名为 postprocess 的函数来执行。若要指定其他函数，请在脚本名称后加冒号及函数名，例如：
polygraphy run ... --trt-npps my_script.py:custom_func

使用网络加载器脚本约束精度
或者，您可以使用网络加载器脚本来手动定义整个网络，在此过程中您可以设置层精度。
以下部分假设您已阅读过手动定义 TensorRT 网络或配置示例，并对如何使用 TensorRT Python API 有基本了解。
首先，在模型上运行 ONNX-Runtime 以生成参考输入和基准输出：
polygraphy run needs_constraints.onnx --onnxrt --val-range x:[1,2] \
    --save-inputs inputs.json --save-outputs golden_outputs.json

接下来，运行提供的网络加载器脚本 constrained_network.py，该脚本约束模型中的精度，强制 TensorRT 遵守约束，并使用保存的输入与保存的基准输出进行比较：
polygraphy run constrained_network.py --precision-constraints obey \
    --trt --fp16 --load-inputs inputs.json --load-outputs golden_outputs.json \
    --check-error-stat median

请注意，除了显式约束的层之外，如果这样做能提高整体引擎性能，TensorRT 可能会选择以 FP32 运行网络中的其他层。
[可选]：运行网络脚本，但在必要时允许 TensorRT 忽略精度约束。如果 TensorRT 没有满足所请求精度约束的层实现，则可能需要这样做才能运行网络：
polygraphy run constrained_network.py --precision-constraints prefer \
    --trt --fp16 --load-inputs inputs.json --load-outputs golden_outputs.json \
    --check-error-stat median

另请参阅
使用降低精度：关于如何使用 Polygraphy 调试降低精度优化的更通用指南。
(../../../../how-to/work_with_reduced_precision.md)

手动定义 TensorRT 网络或配置：关于如何创建网络脚本模板的说明。
(../../../../examples/cli/run/04_defining_a_tensorrt_network_or_config_manually)

TensorRT Python API 参考文档
(https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html)
