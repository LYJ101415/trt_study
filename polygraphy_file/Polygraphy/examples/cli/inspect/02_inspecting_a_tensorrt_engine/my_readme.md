检查 TensorRT 引擎
简介
inspect model 子工具可以加载并显示 TensorRT 引擎（即 plan 文件）的相关信息：
运行示例:
生成一个具有动态形状和 2 个配置（profile）的引擎：
polygraphy run dynamic_identity.onnx --trt \
    --trt-min-shapes X:[1,2,1,1] --trt-opt-shapes X:[1,2,3,3] --trt-max-shapes X:[1,2,5,5] \
    --trt-min-shapes X:[1,2,2,2] --trt-opt-shapes X:[1,2,4,4] --trt-max-shapes X:[1,2,6,6] \
    --save-engine dynamic_identity.engine

检查该引擎：
polygraphy inspect model dynamic_identity.engine \
    --show layers
注意：仅当引擎构建时 profiling_verbosity 不为 NONE 时，--show layers 才有效。更高的详细程度会提供更多逐层信息。

这将显示类似如下的内容：
[I] ==== TensorRT Engine ====
    Name: Unnamed Network 0 | Explicit Batch Engine

    ---- 1 Engine Input(s) ----
    {X [dtype=float32, shape=(1, 2, -1, -1)]}

    ---- 1 Engine Output(s) ----
    {Y [dtype=float32, shape=(1, 2, -1, -1)]}

    ---- Memory ----
    Device Memory: 0 bytes

    ---- 2 Profile(s) (2 Tensor(s) Each) ----
    - Profile: 0
        Tensor: X          (Input), Index: 0 | Shapes: min=(1, 2, 1, 1), opt=(1, 2, 3, 3), max=(1, 2, 5, 5)
        Tensor: Y         (Output), Index: 1 | Shape: (1, 2, -1, -1)

    - Profile: 1
        Tensor: X          (Input), Index: 0 | Shapes: min=(1, 2, 2, 2), opt=(1, 2, 4, 4), max=(1, 2, 6, 6)
        Tensor: Y         (Output), Index: 1 | Shape: (1, 2, -1, -1)

    ---- 1 Layer(s) Per Profile ----
    - Profile: 0
        Layer 0    | node_of_Y [Op: Reformat]
            {X [shape=(1, 2, -1, -1)]}
             -> {Y [shape=(1, 2, -1, -1)]}

    - Profile: 1
        Layer 0    | node_of_Y [profile 1] [Op: MyelinReformat]
            {X [profile 1] [shape=(1, 2, -1, -1)]}
             -> {Y [profile 1] [shape=(1, 2, -1, -1)]}

也可以使用 --show layers attrs 来显示更详细的层信息。