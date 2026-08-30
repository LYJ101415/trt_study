"""以下代码的主要功能是加载一个已经构建好的TensorRT引擎文件，并使用它执行推理。具体来说，它读取名为 identity.engine 的文件，
输入一个全为1的 1x1x2x2 张量，经过模型计算后，验证输出是否与输入完全一致（因为这是一个恒等映射/Identity模型），最后打印推理成功的提示。
用于TensorRT 模型部署与验证：在将深度学习模型转换为 TensorRT 引擎后，通常需要编写这样的脚本来验证引擎文件是否损坏、反序列化是否正常、以及推理结果是否符合预期。
"""

"""
该脚本加载由 build_and_run.py 构建的 TensorRT 引擎，并执行推理。它首先使用 polygraphy 读取引擎文件，然后使用 TensorRT 运行推理，并验证输出是否与输入一致。
"""
import numpy as np
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes, TrtRunner


def main():
    # 正如我们在构建时所做的那样，我们可以将多个加载器（loaders）组合在一起，以实现我们想要的行为。具体来说，我们希望从文件中加载一个序列化的引擎，然后将其反序列化为 TensorRT 引擎。
    
    # BytesFromPath("identity.engine")：是数据加载器。它从本地磁盘路径读取名为 identity.engine 的序列化引擎文件，并将其内容作为字节流返回。
    # EngineFromBytes(...)：这一步是反序列化器。它接收上一步得到的字节流，将其反序列化为一个可以在GPU上运行的 TensorRT 引擎对象（即 ICudaEngine 的封装）。
    # 代码中的注释提到，这种将多个加载器组合在一起的模式与构建引擎时是一致的。
    load_engine = EngineFromBytes(BytesFromPath("identity.engine"))

    # 和之前的构建时的推理过程几乎完全一样
    # TrtRunner(load_engine)：这是推理执行器。它接收加载好的引擎，负责管理推理所需的资源（如GPU显存分配、输入/输出缓冲区等）。
    # 使用 Python 的上下文管理器（with 语句）来确保推理结束后，Runner 占用的 GPU 资源被正确释放。
    with TrtRunner(load_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        # 执行真正的推理计算。feed_dict 是一个字典，键 "x" 是模型定义的输入张量名称，值 inp_data 是 NumPy 数组。Runner 会自动将 NumPy 数组传输到 GPU 进行计算，并将结果传回主机内存。
        outputs = runner.infer(feed_dict={"x": inp_data})
        # 断言检查。outputs 也是一个字典，键 "y" 是模型的输出张量名称。由于这是一个 Identity（恒等）模型，输出理应与输入完全一样，因此使用 np.array_equal 进行严格验证。
        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()
