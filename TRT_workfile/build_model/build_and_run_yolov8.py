"""
该脚本基于 YOLOv8 的 ONNX 模型，构建并运行一个启用了 FP16 精度的 TensorRT 引擎。
使用随机生成的假数据（Dummy Data）进行推理验证。
"""
import numpy as np
from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    SaveEngine,
    TrtRunner,
)

def main():
    # 1. 配置 ONNX 模型路径与 TensorRT 构建参数
    onnx_path = "/root/autodl-tmp/polygraphy_file/myStudy_flle/best_PCB_F.onnx"
    engine_path = "yolov8.engine"
    
    # 2. 构建 TensorRT 引擎 (启用 FP16 加速)
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(onnx_path), 
        config=CreateConfig(fp16=True)
    )
    
    # 3. 将构建好的引擎保存到本地
    build_engine = SaveEngine(build_engine, path=engine_path)
    print(f"[*] 正在构建并保存 TensorRT 引擎到: {engine_path} ...")

    # 4. 加载引擎并进行推理验证
    with TrtRunner(build_engine) as runner:
        # 生成简单的假数据代替真实图像 (Batch=1, Channels=3, H=640, W=640)
        # 注意：如果你的 YOLOv8 训练时使用的是其他尺寸(如 416x416)，请修改这里的 H 和 W
        dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
        
        # 执行推理 (YOLOv8 默认输入名为 'images')
        outputs = runner.infer(feed_dict={"images": dummy_input})
        
        # 获取输出结果 (YOLOv8 默认输出名为 'output0')
        # 输出形状通常为 (1, 84, 8400) -> [batch, num_classes+4, num_boxes]
        output_tensor = outputs["output0"]
        
        # 验证输出有效性
        assert output_tensor is not None, "模型输出为空！"
        assert len(output_tensor.shape) == 3, f"输出维度异常，期望3维，实际为: {output_tensor.shape}"
        
        print(f"[+] 推理成功！")
        print(f"    输入形状: {dummy_input.shape}")
        print(f"    输出形状: {output_tensor.shape}")
        print(f"    输出数据范围: [{output_tensor.min():.4f}, {output_tensor.max():.4f}]")

if __name__ == "__main__":
    main()

"""
代码运行后输出如下的说明信息：
[I] Building engine with configuration:
    Flags                  | [FP16]
    Engine Capability      | EngineCapability.STANDARD
    Memory Pools           | [WORKSPACE: 32228.81 MiB, TACTIC_DRAM: 32228.81 MiB, TACTIC_SHARED_MEMORY: 1024.00 MiB]
    Tactic Sources         | [EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
    Profiling Verbosity    | ProfilingVerbosity.DETAILED
    Preview Features       | [PROFILE_SHARING_0806]
[I] Finished engine building in 369.775 seconds
[I] Saving engine to yolov8.engine
[+] 推理成功！
    输入形状: (1, 3, 640, 640)
    输出形状: (1, 10, 8400)
    输出数据范围: [0.0000, 636.4922]

以下是输出日志的详细解读：
1. 引擎构建配置回顾
Flags | [FP16]：确认引擎成功启用了 FP16 半精度加速。
Memory Pools | [WORKSPACE: 32228.81 MiB...]：TensorRT 使用了约 32GB 的显存来进行算子搜索和内存分配，这充分利用了你 RTX 4080 的性能。
2. 构建耗时与保存
Finished engine building in 369.775 seconds：引擎构建总共耗时约 370 秒（约 6 分钟）。这是正常的耗时，因为 TensorRT 进行了大量的底层算子优化。
Saving engine to yolov8.engine：构建好的引擎已成功序列化并保存为 yolov8.engine 文件。以后再次使用时，直接加载这个文件即可，启动只需几秒钟，无需重新构建。
3. 推理验证结果
[+] 推理成功！：模型在 TensorRT 引擎上运行正常，没有报错。
输入形状: (1, 3, 640, 640)：输入数据的尺寸符合预期（1张图片，3通道，640x640分辨率）。
输出形状: (1, 10, 8400)：这是 YOLOv8 的标准输出格式：
1：Batch 大小。
10：代表 6个类别数 + 4个边界框坐标 (x, y, w, h)。这说明你的 PCB 缺陷检测模型有 6 个类别。
8400：代表在 640x640 分辨率下，模型预设的候选框（Anchor）数量。
输出数据范围: [0.0000, 636.4922]：模型输出的原始数值范围。最大值 636.49 看起来是合理的（包含了未归一化的坐标值和置信度）。

4. Engine Capability | EngineCapability.STANDARD
含义：这表示引擎构建采用的是标准模式（Standard flow）。
作用：在这种模式下，TensorRT 不对网络功能施加任何限制，生成的序列化引擎可以使用标准的 nvinfer1 命名空间下的运行时 API 进行执行。
这是绝大多数常规深度学习模型（如你的 YOLOv8）默认且最合适的模式。(注：与之相对的是 SAFETY 或 DLA_STANDALONE 模式，主要用于自动驾驶等需要安全认证或特定硬件加速的场景)。
5. Tactic Sources | [EDGE_MASK_CONVOLUTIONS, JIT_CONVOLUTIONS]
含义：这表示 TensorRT 在构建引擎时，主要启用了这两种特定的算子优化策略（Tactics）：
EDGE_MASK_CONVOLUTIONS：启用基于边缘掩码表实现的卷积算子。这种策略通过消耗与输入大小成正比的额外内存空间，来换取推理性能的提升。
JIT_CONVOLUTIONS：启用基于源代码即时编译（JIT）融合实现的卷积算子。这种策略会在构建阶段动态生成优化代码，可能会增加引擎的构建时间，但能提升运行效率。
作用：TensorRT 会利用这些策略来寻找最适合你 RTX 4080 硬件的卷积计算方式。
6. Profiling Verbosity | ProfilingVerbosity.DETAILED
含义：这表示引擎在构建时开启了详细级别的性能分析（Profiling）输出。
作用：TensorRT 会记录并打印出详细的层信息，包括层名称和具体的层参数。这通常用于开发者在调试模型、排查精度问题或分析网络结构时使用。在正式的生产部署中，为了节省内存和开销，
通常会改回默认的 LAYER_NAMES_ONLY（仅输出层名）或 NONE。
7. Preview Features | [PROFILE_SHARING_0806]
含义：这表示启用了 TensorRT 的预览特性（Preview Features）中的“Profile Sharing”功能。
作用：预览特性是 TensorRT 提供的新功能，它们已经过全面测试，但尚未达到其他核心功能那样完全稳定的状态。PROFILE_SHARING_0806 通常与内存优化和动态形状支持相关。
你在之前的日志中也看到过一条警告 [W] profileSharing0806 is on by default...，说明这是 TensorRT 10.0 版本默认开启的内存共享优化机制。

"""