"""
代码是 NVIDIA Polygraphy 工具包中用于在真实数据集上验证 TensorRT 模型精度的完整示例。
它演示了如何使用 TrtRunner 直接运行推理，并将输出与“黄金标准”（即期望输出）进行比较。
"""

"""
This script uses the Polygraphy Runner API to validate the outputs of an identity model using a trivial dataset.
EngineFromNetwork：Polygraphy 的一个可调用对象，用于从 ONNX 网络构建 TensorRT 引擎。
NetworkFromOnnxPath：从 .onnx 文件路径加载 ONNX 模型并转换为 TensorRT 网络定义。
TrtRunner：封装 TensorRT 引擎的推理接口，提供统一的 .infer() 方法。
这些类共同构成了一条从 ONNX 到 TensorRT 推理的流水线。
"""
import numpy as np
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner

# 以下是模拟一个“真实数据集”（实际是人工构造的极小数据集，仅用于演示）。虽然只有 4 个样本，但此模式可扩展到大规模数据集（如从文件系统或数据库加载）。
REAL_DATASET = [
    np.ones((1, 1, 2, 2), dtype=np.float32),
    np.zeros((1, 1, 2, 2), dtype=np.float32),
    np.ones((1, 1, 2, 2), dtype=np.float32),
    np.zeros((1, 1, 2, 2), dtype=np.float32),
]  # Definitely real data

# For an identity network, the golden output values are the same as the input values.
# Though such a network appears useless at first glance, it can be very useful in some cases (like here!).
# Identity 网络：输出 = 输入（即 y = x）。因此，“黄金输出”（golden outputs / expected outputs）完全等于输入数据。
# 注释幽默地指出：虽然 Identity 网络看似无用，但在测试推理管道是否正常工作时非常有用（例如验证数据流、内存布局、精度是否一致）。
EXPECTED_OUTPUTS = REAL_DATASET


def main():
    # 构建 TensorRT 引擎的两步组合：NetworkFromOnnxPath("identity.onnx")：从 identity.onnx 文件加载 ONNX 模型。
    # EngineFromNetwork(...)：将该网络编译为 TensorRT 引擎（未立即构建，而是返回一个可调用对象）。
    # 注意：build_engine 是一个延迟构建的 callable，实际引擎会在 TrtRunner 初始化时被创建。
    # 这种设计允许 Polygraphy 在需要时才构建引擎，节省资源。
    # 补充说明：关于 identity.onnx；该模型应是一个简单的 Identity 算子（如 ONNX 的 Identity 节点），输入名为 "x"，输出名为 "y"。
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("identity.onnx"))

    with TrtRunner(build_engine) as runner:
        # 遍历数据集中的每个样本及其对应的期望输出。zip 将输入和标签一一配对，便于逐样本验证。data是取至REAL_DATASET，
        # golden是取至EXPECTED_OUTPUTS的相应位置的数据，以完成一一对应。
        for data, golden in zip(REAL_DATASET, EXPECTED_OUTPUTS):
            # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
            #   Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
            outputs = runner.infer(feed_dict={"x": data})

            assert np.array_equal(outputs["y"], golden)

        print("Validation succeeded!")


if __name__ == "__main__":
    main()

# 以上脚本虽小，但完整展示了生产级模型验证的核心范式：加载模型 → 加载数据 → 推理 → 对比标签 → 报告结果。
