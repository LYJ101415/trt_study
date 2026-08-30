TensorRT 中的确定性引擎构建
注意：本示例需要 TensorRT 8.7 或更高版本。
简介
在构建引擎期间，TensorRT 会运行并计时多个内核（kernel），以选择最优的内核。由于内核的计时结果在每次运行时可能会有细微差异，因此该过程本质上是非确定性的。
在许多情况下，确定性的引擎构建可能是必要的。实现这一目标的一种方法是使用计时候存（timing cache），以确保每次都选择相同的内核。
运行示例:
构建引擎并保存计时候存：
polygraphy convert identity.onnx \
    --save-timing-cache timing.cache \
    -o 0.engine

使用计时候存进行另一次引擎构建：
polygraphy convert identity.onnx \
    --load-timing-cache timing.cache --error-on-timing-cache-miss \
    -o 1.engine
我们指定了 --error-on-timing-cache-miss，以确保新引擎的每一层都使用了计时候存中的条目。

验证两个引擎是否完全相同：
diff <(polygraphy inspect model 0.engine --show layers attrs) <(polygraphy inspect model 1.engine --show layers attrs)
