# Debugging Flaky TensorRT Tactics

**IMPORTANT: This example no longer works reliably for newer versions of TensorRT, since they make some**
    **tactic choices that are not exposed via the IAlgorithmSelector interface. Thus, the approach outlined below**
    **cannot guarantee deterministic engine builds. With TensorRT 8.7 and newer, you can use the**
    **tactic timing cache (`--save-timing-cache` and `--load-timing-cache` in Polygraphy) to ensure**
    **determinism, but these files are opaque and thus cannot be interpreted by `inspect diff-tactics`**

## Introduction

Sometimes, a tactic in TensorRT may produce incorrect results, or have
otherwise buggy behavior. Since the TensorRT builder relies on timing
tactics, engine builds are non-deterministic, which can make tactic bugs
manifest as flaky/intermittent failures.

One approach to tackling the problem is to run the builder several times,
saving tactic replay files from each run. Once we have a set of known-good and
known-bad tactics, we can compare them to determine which tactic
is likely to be the source of error.

The `debug build` subtool allows you to automate this process.

For more details on how the `debug` tools work, see the help output:
`polygraphy debug -h` and `polygraphy debug build -h`.


## Running The Example

1. Generate golden outputs from ONNX-Runtime:

    ```bash
    polygraphy run identity.onnx --onnxrt \
        --save-outputs golden.json
    ```

2. Use `debug build` to repeatedly build TensorRT engines and compare results against the golden outputs,
    saving a tactic replay file each time:

    ```bash
    polygraphy debug build identity.onnx --fp16 --save-tactics replay.json \
        --artifacts-dir replays --artifacts replay.json --until=10 \
        --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json
    ```

    Let's break this down:

    - Like other `debug` subtools, `debug build` generates an intermediate artifact each iteration
        (`./polygraphy_debug.engine` by default). This artifact in this case is a TensorRT engine.

        *TIP: `debug build` supports all the TensorRT builder configuration options supported*
            *by other tools, like `convert` or `run`.*

    - In order for `debug build` to determine whether each engine fails or passes,
        we provide a `--check` command. Since we're looking at a (fake) accuracy issue,
        we can use `polygraphy run` to compare the outputs of the engine to our golden values.

        *TIP: Like other `debug` subtools, an interactive mode is also supported, which you can*
            *use simply by omitting the `--check` argument.*

    - Unlike other `debug` subtools, `debug build` has no automatic terminating condition, so we need
        to provide the `--until` option so that the tool knows when to stop. This can either be a number
        of iterations, or `"good"` or `"bad"`. In the latter case, the tool will stop after finding the
        first passing or failing iteration respectively.

    - Since we eventually want to compare the good and bad tactic replays, we specify `--save-tactics`
        to save tactic replay files from each iteration, then use `--artifacts` to tell `debug build`
        to manage them, which involves sorting them into `good` and `bad` subdirectories under the
        main artifacts directory, specified with `--artifacts-dir`.


3. Use `inspect diff-tactics` to determine which tactics could be bad:

    ```bash
    polygraphy inspect diff-tactics --dir replays
    ```

    *NOTE: This last step should report that it could not determine potentially bad tactics since*
        *our `bad` directory should be empty at this point (please file a TensorRT issue otherwise!):*

    <!-- Polygraphy Test: Ignore Start -->
    ```
    [I] Loaded 2 good tactic replays.
    [I] Loaded 0 bad tactic replays.
    [I] Could not determine potentially bad tactics. Try generating more tactic replay files?
    ```
    <!-- Polygraphy Test: Ignore End -->


## Further Reading

For more information on the `debug` tool, as well as tips and tricks applicable
to all `debug` subtools, see the
[how-to guide for `debug` subtools](../../../../how-to/use_debug_subtools_effectively.md).

==================================================================
这是一份关于使用 Polygraphy 调试 TensorRT 中“策略抖动”导致精度间歇性异常的经典文档。在 TensorRT 编译引擎时，它会为每一层尝试多种不同的实现策略，并通过实际计时来选择运行最快的那一个。由于系统负载、GPU状态等因素，每次编译选出的最快策略可能不同，这就导致了非确定性构建。如果某个策略恰好存在 Bug（比如在特定形状下计算结果错误），就会表现为模型偶尔正确、偶尔错误，即所谓的“Flaky”。
对于 TensorRT 8.7 及以上版本，此方法已不再可靠。NVIDIA 官方推荐使用 Tactic Timing Cache（策略计时缓存）（即 --save-timing-cache 和 --load-timing-cache）来保证确定性构建。只要加载了相同的 cache，TRT 就会根据 cache 里的历史耗时做决定，而不是重新计时，从而保证每次构建选出的策略完全一致。
二、 实际应用流程：
Step 1: 生成 Golden Outputs（确立绝对正确的基准）；不要依赖有 Bug 的 TRT，使用 ONNX Runtime 在 FP32 下运行，保存正确结果：
polygraphy run yolov8.onnx --onnxrt \
    --save-outputs golden.json
作用：生成 golden.json，里面包含了每层输出的正确数值。

Step 2: 使用 debug build 反复构建并验证（收集证据）：
polygraphy debug build yolov8.onnx --fp16 --save-tactics replay.json \
    --artifacts-dir replays --artifacts replay.json --until=10 \
    --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json

参数详细剖析：
--fp16：模拟出问题的构建配置。
--save-tactics replay.json：每次构建完，把这次构建用到的所有层策略保存到 replay.json。
--artifacts-dir replays 和 --artifacts replay.json：让 debug 工具接管这个 JSON 文件。如果 --check 验证通过，文件被移入 replays/good/；如果失败，移入 replays/bad/。
--until=10：最多重复构建 10 次引擎。也可以写 --until=bad，表示只要抓到一次失败就立刻停止（非常适合间歇性 Bug，节省时间）。
--check ...：验证逻辑。运行刚刚生成的 polygraphy_debug.engine，把它的输出与 golden.json 对比。如果精度误差过大，该命令返回非零退出码，debug build 就会判定此次构建为 bad。

Step 3: 使用 inspect diff-tactics 找出坏策略（解剖证据）：
polygraphy inspect diff-tactics --dir replays
作用：读取 replays/good 和 replays/bad 目录下的所有 JSON。找出那些只在 bad 中出现，在 good 中从未出现的策略。
输出示例：Layer: Conv_45, Tactic: 123456789。这就告诉你，Conv_45 层使用了 123456789 号策略时会导致计算错误。

三、 高级应用与实战技巧
1. 结合 --until=bad 实现快速自动捕获
间歇性 Bug 可能跑 50 次才出现一次，手动盯着看很不现实。

# 持续运行，直到抓到一次 bad 引擎为止
polygraphy debug build model.onnx --fp16 \
    --save-tactics replay.json --artifacts-dir replays --artifacts replay.json \
    --until=bad \
    --check polygraphy run polygraphy_debug.engine --trt --load-outputs golden.json -v
加上 -v 可以在失败时看到详细的误差输出（如最大绝对误差、余弦相似度等）。

2. 交互式调试
文档提到如果省略 --check，会进入交互模式。
polygraphy debug build model.onnx --fp16 --save-tactics replay.json ...
应用场景：有时候精度误差很小，Polygraphy 默认的阈值判断为 PASS，但你凭借业务经验知道这个误差是不可接受的。在交互模式下，工具构建一次就会暂停，询问你这次结果是 good 还是 bad，你可以手动检查输出后输入 y 或 n。

3. 从“诊断”走向“治疗”：使用 IAlgorithmSelector 剔除坏策略
找到坏策略只是第一步，真正的目的是在正式生产环境中避开它。
假设通过 Step 3 找到了罪魁祸首是 Tactic: 999，你可以编写一个 TRT 的 IAlgorithmSelector（C++ API 或 Polygraphy 的 trt-network-script），将该策略设为 Never。
用 Polygraphy 的方式快速验证剔除效果：
# 强制使用 good 目录下的策略重放
polygraphy run model.onnx --trt \
    --load-tactics replays/good/replay_0.json \
    --load-outputs golden.json
如果结果正确，说明 good 策略集是安全的，你可以将这个 replay_0.json 作为后续构建的基准。

4. 现代 TRT (>=8.7) 下的替代工作流
正如文档警告，新版 TRT 中 diff-tactics 可能失效。现代的调试思路应转变为：
A. 使用 Timing Cache 锁定变量：
# 第一次构建，生成 cache
polygraphy convert model.onnx --to-trt --save-timing-cache cache.json

# 后续无限次构建，保证策略绝对一致
polygraphy convert model.onnx --to-trt --load-timing-cache cache.json
B. 结合 Layer Precision 逐层排查：
如果无法对比策略，且使用 Cache 锁定后依然有精度问题，可以使用 --layer-precisions 逐层回退到 FP32 来二分查找出问题的层：

# 怀疑前30层有问题，强制前30层用FP32
polygraphy run model.onnx --trt --fp16 \
    --layer-precisions "Conv_0:float32,Conv_1:float32,..."
总结
这份文档提供的是一种**“控制变量法”的自动化实现**。在 TensorRT 8.7 之前，这是排查间歇性精度/计算崩溃问题的神兵利器；在 8.7 之后，虽然策略重放的透明度降低，但 debug build 这种“反复构建 + 自动验证 + 证据归档”的框架思想，以及 Timing Cache 的使用，依然是我们处理 TRT 不确定性问题的核心方法论。
