# Inspecting A TensorRT Network


## Introduction

The `inspect model` subtool can automatically convert supported formats
into TensorRT networks, and then display them.


## Running The Example

1. Display the TensorRT network after parsing an ONNX model:

    ```bash
    polygraphy inspect model identity.onnx \
        --show layers --display-as=trt
    ```

    This will display something like:

    ```
    [I] ==== TensorRT Network ====
        Name: Unnamed Network 0 | Explicit Batch Network

        ---- 1 Network Input(s) ----
        {x [dtype=float32, shape=(1, 1, 2, 2)]}

        ---- 1 Network Output(s) ----
        {y [dtype=float32, shape=(1, 1, 2, 2)]}

        ---- 1 Layer(s) ----
        Layer 0    | node_of_y [Op: LayerType.IDENTITY]
            {x [dtype=float32, shape=(1, 1, 2, 2)]}
             -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
    ```

    It is also possible to show detailed layer information, including layer attributes, using `--show layers attrs weights`.
====================================================================
这份文档展示了 如何使用 Polygraphy 的 inspect model 工具将 ONNX 模型解析为 TensorRT 网络并可视化其内部结构。这是理解模型在 TensorRT 中实际执行形式的关键一步，尤其适用于调试、性能分析和精度问题排查。

下面从核心目的、技术原理、命令详解、输出解读、高级用法、工程价值 六个维度进行深度解析。
一、核心目的：为什么需要“Inspect a TensorRT Network”？
问题背景：你有一个 ONNX 模型（如 resnet50.onnx），但在 TensorRT 中推理时：性能不如预期
输出精度下降；甚至构建失败；根本原因往往隐藏在 TensorRT 对模型的内部表示中；关键洞察：
ONNX 模型 ≠ TensorRT 网络，TensorRT 在解析 ONNX 时会进行一系列转换：
算子融合（如 Conv + BN + ReLU → CBR）；类型转换（FP32 ↔ FP16）；图优化（常量折叠、死代码消除）；层重命名（node_123 → Conv_123）
→ 直接查看 TensorRT 内部网络结构，才能知道“真实发生了什么”

二、技术原理：Polygraphy 如何实现？
polygraphy inspect model identity.onnx --show layers --display-as=trt
内部流程：加载 ONNX 模型；创建 TensorRT Builder + Network；使用 onnx-tensorrt 解析器（nvonnxparser）将 ONNX 转为 TRT Network；此过程模拟了 trt.Builder.build_engine() 的前半部分；遍历 TRT Network 的所有层（ILayer）、输入/输出（ITensor）；格式化输出人类可读的结构。注意：此过程不构建引擎（不涉及 tactic 选择、kernel 编译），仅展示网络拓扑。

三、命令详解与参数说明
基础命令：
polygraphy inspect model <model> --show layers --display-as=trt
参数	              作用
--show layers	    显示网络中的所有层（必须指定，否则无输出）
--display-as=trt	强制以 TensorRT 网络形式显示（即使输入是 ONNX）
高级选项（文档末尾提及）
--show layers attrs weights
attrs：显示层的属性（如卷积的 stride、padding）
weights：显示层的权重值（谨慎使用，大模型会刷屏）

 其他 --show 选项：
--show inputs/outputs：单独显示输入输出（默认已包含）
--show mark-all-outputs：标记所有中间张量为输出（用于调试）

四、输出结构深度解读（以 Identity 模型为例）
[I] ==== TensorRT Network ====
    Name: Unnamed Network 0 | Explicit Batch Network

    ---- 1 Network Input(s) ----
    {x [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Network Output(s) ----
    {y [dtype=float32, shape=(1, 1, 2, 2)]}

    ---- 1 Layer(s) ----
    Layer 0    | node_of_y [Op: LayerType.IDENTITY]
        {x [dtype=float32, shape=(1, 1, 2, 2)]}
         -> {y [dtype=float32, shape=(1, 1, 2, 2)]}
逐段解析：
1. 网络元信息
Name: Unnamed Network 0 | Explicit Batch Network
Explicit Batch Network：表示使用显式 batch 维度（ONNX 标准），而非旧版 TRT 的 implicit batch
2. 输入/输出张量
{x [dtype=float32, shape=(1, 1, 2, 2)]}
张量名 x，类型 float32，形状 (1,1,2,2)
注意：TensorRT 可能重命名张量（如 ONNX 中叫 input，TRT 中叫 x）
3. 层（Layer）详情
Layer 0    | node_of_y [Op: LayerType.IDENTITY]
Layer 0：层索引（按添加顺序）
node_of_y：层名称（通常来自 ONNX 节点名）
Op: LayerType.IDENTITY：TensorRT 内部算子类型

 关键区别：ONNX 中的算子叫 Identity；TensorRT 中对应 LayerType.IDENTITY（枚举值）
4. 数据流
{x ...} -> {y ...} -->清晰展示 输入张量 → 输出张量 的流向

五、实战示例：ResNet50 的典型输出片段
运行：
polygraphy inspect model resnet50.onnx --show layers --display-as=trt | head -20
可能看到：
Layer 5    | Conv_5 [Op: LayerType.CONVOLUTION]
    {497 [dtype=float32, shape=(1, 64, 56, 56)]}
     -> {500 [dtype=float32, shape=(1, 64, 56, 56)]}
    Attributes:
        kernel_shape: (3, 3)
        strides: (1, 1)
        padding: (1, 1, 1, 1)
        num_output_maps: 64
你能获得的信息：
卷积核大小 3x3
步长 1
Padding 1（左右上下）
输出通道数 64
输入/输出形状完全匹配（验证无 reshape 错误）
 这比看 ONNX 的 Conv 节点更直观，因为 TRT 已解析所有属性。

六、工程价值：为什么这个功能不可或缺？
场景 1：验证算子是否被正确解析;ONNX 中有 GELU，但 TRT 不支持？查看输出：若变成 PluginV2IOExt → 说明用了插件;若变成多个基础算子（Div + Erf + Add + Mul）→ 说明被分解
场景 2：调试精度问题;发现某层输出 NaN;通过 inspect 定位到具体层（如 Layer 123）;
结合 --show weights 检查权重是否异常（全零、超大值）
场景 3：性能分析预处理;确认是否有冗余层（如连续两个 Identity）;检查是否发生预期的融合（如 Conv + Relu 是否合并）
场景 4：教学与文档;向团队展示 “TensorRT 实际看到的模型结构”;比 Netron 查看 ONNX 更贴近部署实际
七、高级技巧与注意事项
技巧 1：结合 --layer-info 获取更多细节
polygraphy inspect model model.onnx --layer-info
显示每层的 精度（precision） 和 输出类型（output type）;对混合精度调试至关重要
技巧 2：导出为文本文件便于对比
polygraphy inspect model a.onnx --display-as=trt > a_trt.txt
polygraphy inspect model b.onnx --display-as=trt > b_trt.txt
diff a_trt.txt b_trt.txt
快速对比两个模型在 TRT 中的差异

注意事项：
问题	                   说明
不显示权重默认值	出于性能考虑，需显式加 --show weights
动态形状显示为 -1	如 shape=(-1, 3, 224, 224)
插件层信息有限	    自定义插件可能只显示 PluginV2IOExt，无内部细节
不反映最终引擎优化	 如 kernel 融合、内存复用等发生在 build 阶段

 总结：inspect model --display-as=trt 的核心价值
维度	说明
本质	ONNX 到 TensorRT 的“翻译器”，揭示模型在 TRT 中的真实形态
用途	调试、验证、教学、性能预分析
优势	比直接读 ONNX 更贴近部署实际，比分析 engine 二进制更可读
定位	TensorRT 调试工作流的第一步 —— “先看清问题，再解决问题”

 一句话精髓：“不要假设 TensorRT 看到了什么，用 inspect 去确认它真正看到了什么。”
这是避免“我以为...”类调试陷阱的最有效方法。
因此，无论你是刚接触 TensorRT 的新手，还是经验丰富的部署工程师，polygraphy inspect model --display-as=trt 都应成为你日常工具箱中的标准配置。