# Using `debug reduce` Effectively


## Table Of Contents

- [Introduction](#introduction)
- [A Note On Models With Dynamic Input Shapes](#a-note-on-models-with-dynamic-input-shapes)
- [Debugging Accuracy Errors](#debugging-accuracy-errors)
    - [Tolerances](#tolerances)
    - [Generating Golden Values](#generating-golden-values)
- [Tips And Tricks](#tips-and-tricks)
    - [Saving Intermediate Models](#saving-intermediate-models)
    - [Insights From Minimum Good Models](#insights-from-minimum-good-models)
    - [Reduction Modes](#reduction-modes)
- [Further Reading](#further-reading)


## Introduction

The `debug reduce` subtool allows you to iteratively reduce a failing ONNX model
to find a minimal failing case, which might be easier to debug than the original model.
The fundamental steps undertaken by `debug reduce` are as follows:

1. Remove some nodes from the original graph and write a new model to `polygraphy_debug.onnx`
    (this path can be changed using the `--iter-artifact` option).

2. Evaluate the model either interactively or, if a `--check` command is provided, automatically.

3. Remove more nodes if the model still fails, otherwise add nodes back; Then, repeat the process.

This guide provides some general information as well as tips and tricks on
how to use `debug reduce` effectively.

Also see the [general how-to guide for `debug` subtools](./use_debug_subtools_effectively.md),
which includes information applicable to all the `debug` subtools.


## A Note On Models With Dynamic Input Shapes

For models with dynamic input shapes, you may not always know the shapes of all intermediate
tensors in the model. Thus, when you check subgraphs, you may end up using
incorrect tensor shapes.

There are two ways to get around this:

1. Use `polygraphy surgeon sanitize --override-input-shapes <shapes>` to freeze the input shapes in the model
2. Supply `--model-input-shapes` to `debug reduce`, which will use shape inference to infer shapes
    of intermediate tensors.

If your model uses shape operations, it is generally best to use option (1) and fold the shape
operations away with `--fold-constants`.

In either case, if there's a problem with shape inference, you can use
`--force-fallback-shape-inference` to infer shapes by running inference instead.

Alternatively, you can use `--no-reduce-inputs` so that the model inputs are not modified.
The `polygraphy_debug.onnx` subgraph generated during each iteration will always use the inputs
of the original model; only layers from the end will be removed.


## Debugging Accuracy Errors

Accuracy errors are especially complex to debug since errors introduced by early layers
in the graph might be amplified by subsequent layers, making it difficult to determine
which layer is the true root cause of the error. This section outlines some things to
keep in mind when using `debug reduce` to debug accuracy errors.

### Tolerances

In some model architectures, intermediate layers may have large errors without necessarily
causing accuracy issues in the final model output. Thus, make sure the tolerance you use for
comparison is high enough to ignore these kinds of false positives.

At the same time, tolerance must be low enough to catch real errors.

A good starting point is to set tolerances close to the error you observe in the full model.


### Generating Golden Values

There are two different approaches you can take when generating golden values for comparison,
each with their own advantages and disadvantages:

1. **Generating golden values for all layers ahead of time.**

    When generating golden values ahead of time, you need to make sure that the input values
    to each subgraph come from the golden values. Otherwise, comparing the outputs of
    the subgraph against the golden values will be meaningless.
    See [the example](../examples/cli/debug/02_reducing_failing_onnx_models/) for
    details on this approach.

2. **Generating golden values for each subgraph.**

    Regenerating golden values for each subgraph may require less manual effort, but
    has the disadvantage that it doesn't necessarily accurately replicate the behavior
    of the subgraph in the context of the larger graph.
    For example, if the error in your model was caused by an overflow in an intermediate
    layer of the original model, generating fresh input values for each subgraph may not
    reproduce it.


## Tips And Tricks


### Saving Intermediate Models

In some cases, it's useful to have access to every model generated during the reduction process.
This way, if reduction exits early or fails to generate a minimal model, you still have something
to work with. Additionally, you can manually compare the various passing and failing subgraphs
to identify patterns, which may help you determine the root cause of the error.

You can specify `--artifacts polygraphy_debug.onnx` to `debug reduce` to automatically sort models
from each iteration into `good` and `bad` directories. The file name will include the iteration
number so you can easily correlate it with the logging output during reduction.


### Insights From Minimum Good Models

In addition to minimum failing models, `debug reduce` can also generate minimum passing models.
Generally, this is the passing model that is closest in size to the minimal failing model.
Comparing this against the minimum failing model can yield additional insights on the root
cause of a failure.

To make `debug reduce` save minimum passing models, use the `--min-good <path>` option.


### Reduction Modes

`debug reduce` offers multiple strategies to reduce the model, which you can specify with the `--mode` option:
`bisect` operates in `O(log(N))` time, while `linear` operates in `O(N)` time but may lead to smaller models.
A good compromise is to use `bisect` on the original model, then further reduce the result using `linear`.


## Further Reading

- The [how-to guide for `debug` subtools](./use_debug_subtools_effectively.md),
    which includes information applicable to all the `debug` subtools.

- The [`debug reduce` example](../examples/cli/debug/02_reducing_failing_onnx_models/), which
    demonstrates some of the features outlined here.

=============================================================================
这份文档详细解析了 Polygraphy 工具中 debug reduce 子命令的高效使用方法。debug reduce 的核心作用是通过迭代删减，将一个庞大且报错的 ONNX 模型，精简为一个最小失败模型（Minimal Failing Case）。这个精简后的模型保留了原始错误，但结构更简单，极大地降低了排查问题的难度。
以下是文档核心内容的详细解析、实际应用及高级技巧：
 核心运行逻辑
debug reduce 的工作流程是一个不断循环的“删减-验证”过程：
删减节点：从原始计算图中移除部分节点，生成一个新的模型（默认保存为 polygraphy_debug.onnx）。
验证模型：运行新模型，检查它是否依然报错（可以交互式检查，或通过 --check 参数自动检查）。
调整策略：如果模型依然报错，继续删减节点；如果模型不再报错（说明删多了），则将部分节点加回来。不断重复此过程，直到找到最小的报错子图。
 针对动态输入形状（Dynamic Input Shapes）的处理
对于包含动态 Shape 的模型，在删减子图时极易因为中间张量 Shape 推断失败而报错。文档提供了以下解决方案：
冻结 Shape（推荐）：使用 polygraphy surgeon sanitize --override-input-shapes <shapes> 提前将模型输入 Shape 固定，并结合 --fold-constants 折叠掉 Shape 相关的算子。
强制 Shape 推断：在 debug reduce 时提供 --model-input-shapes，让工具自动推断中间张量 Shape；若推断失败，可加上 --force-fallback-shape-inference，通过实际运行推理来推断 Shape。
保持输入不变：使用 --no-reduce-inputs 参数，确保每次迭代生成的子图始终使用原始模型的完整输入，仅从模型尾部开始删减层。
 调试精度误差（Accuracy Errors）
精度问题非常复杂，因为早期层的微小误差会被后续层放大，导致难以定位真正的“罪魁祸首”。
1. 容差设置（Tolerances）
避免误报：某些架构的中间层本身误差较大，但不影响最终输出。因此，比较时的容差（Tolerance）必须设置得足够大，以忽略这些“假阳性”。
抓住真凶：容差也不能过大，否则无法捕捉到真正的精度问题。
最佳实践：将容差设置为与你在完整模型中观察到的误差相近的值。
2. 生成黄金标准值（Golden Values）
在对比精度时，你需要一个“正确答案”作为基准，有两种生成策略：
策略一：提前为所有层生成。优点是准确，但必须确保每个子图的输入数据也来自这份黄金标准，否则对比毫无意义。
策略二：为每个子图实时生成。优点是省事，缺点是可能无法复现原始大模型中的特定错误（例如中间层溢出导致的 NaN，用新生成的随机数据可能无法触发）。
 高级技巧与实战应用
1. 保存中间迭代模型（Saving Intermediate Models）
在漫长的删减过程中，工具可能会提前退出或未能找到绝对最小的模型。保存所有中间状态非常有用。
实战命令：添加 --artifacts polygraphy_debug.onnx 参数。
作用：工具会自动将每次迭代生成的模型按 good（验证通过）和 bad（验证失败）分类放入对应文件夹，文件名包含迭代次数。你可以手动对比这些模型，寻找规律。
2. 利用“最小通过模型”获取洞察（Insights From Minimum Good Models）
除了“最小失败模型”，debug reduce 还能生成“最小通过模型”（即体积最接近失败模型的、验证通过的模型）。
实战命令：使用 --min-good <path> 参数。
作用：将“最小通过模型”与“最小失败模型”进行对比，通常只差一两个算子，这能极其精准地帮你锁定导致问题的根源算子。
3. 灵活选择删减模式（Reduction Modes）
bisect（二分法）：时间复杂度为 O(logN)
O(logN) ，速度极快，适合初步快速缩小范围。
linear（线性法）：时间复杂度为 O(N)
O(N) ，速度较慢，但往往能精简出更小的模型。
高级组合策略：先对原始大模型使用 bisect 快速定位，再对得到的结果使用 linear 进一步极限精简。
 综合实战示例
假设你有一个 model.onnx，在开启 FP16 时出现了精度问题，你想定位是哪一层引起的：
第一步：生成基准数据（ONNX Runtime 的 FP32 结果）
polygraphy run model.onnx --onnxrt --save-outputs golden_outputs.json
第二步：使用 debug reduce 定位问题层
结合上述技巧，使用二分法快速定位，保存中间过程，并保存最小通过模型，同时指定容差和检查命令：
polygraphy debug reduce model.onnx \
    --check polygraphy run polygraphy_debug.onnx --trt --fp16 --load-outputs golden_outputs.json --atol 1e-3 --rtol 1e-3 \
    --mode bisect \
    --artifacts debug_artifacts.onnx \
    --min-good min_good_model.onnx \
    --model-input-shapes "input:[1,3,224,224]"
(注：上述命令中的 --check 后跟的是用于验证每次删减后模型精度的完整 Polygraphy 命令)
通过这套组合拳，你可以高效、系统地将一个复杂的精度或转换报错问题，缩小到一两个具体的算子上，从而对症下药（例如使用 ONNX GraphSurgeon 修改该算子精度或替换实现）。

# 进阶版：加入校准缓存，大幅提升 INT8 debug 速度
polygraphy debug reduce model.onnx \
    --check polygraphy run polygraphy_debug.onnx --trt --int8 --data-loader-script calib_data_loader.py --calibration-cache calib.cache --load-outputs golden_outputs.json --atol 1e-1 --rtol 1e-1 \
    --mode bisect \
    --artifacts debug_artifacts.onnx \
    --min-good min_good_model.onnx \
    --model-input-shapes "input_name:[1,3,224,224]"
