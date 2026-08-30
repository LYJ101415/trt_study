"""代码的核心目的是：加载同一个 ONNX 模型，分别使用 ONNX Runtime 和 TensorRT 执行推理，
然后比较两者的输出结果是否在允许的误差范围内一致，最后将推理结果保存到文件中。
"""

"""
This script runs an identity model with ONNX-Runtime and TensorRT,
then compares outputs.
"""
from polygraphy.backend.onnxrt import OnnxrtRunner, SessionFromOnnx
from polygraphy.backend.trt import EngineFromNetwork, NetworkFromOnnxPath, TrtRunner
from polygraphy.comparator import Comparator, CompareFunc


def main():
    # The OnnxrtRunner requires an ONNX-RT session.
    # We can use the SessionFromOnnx lazy loader to construct one easily:
    # 1.构建 ONNX Runtime 推理器：使用 SessionFromOnnx 创建一个加载 ONNX 模型的会话对象。这是一个惰性加载器，此时模型并未真正加载到内存，而是准备好了加载的配置
    build_onnxrt_session = SessionFromOnnx("identity.onnx")

    # The TrtRunner requires a TensorRT engine.
    # To create one from the ONNX model, we can chain a couple lazy loaders together:
    # 2. 构建 TensorRT 推理器：TensorRT 不能直接运行 ONNX 文件，需要先将其解析为网络结构，再编译为 TensorRT 引擎。这里通过链式调用实现：
    build_engine = EngineFromNetwork(NetworkFromOnnxPath("identity.onnx"))

    # 3. 初始化运行器：将前面构建的引擎和会话传入对应的 Runner 中。Runner 负责管理推理的生命周期（如加载资源、分配内存、执行推理、释放资源）。
    runners = [
        TrtRunner(build_engine),
        OnnxrtRunner(build_onnxrt_session),
    ]
    # `Comparator.run()` will run each runner separately using synthetic input data and
    #   return a `RunResults` instance. See `polygraphy/comparator/struct.py` for details.
    # TIP: To use custom input data, you can set the `data_loader` parameter in `Comparator.run()``
    #   to a generator or iterable that yields `Dict[str, np.ndarray]`.
    
    # 4.执行推理：Comparator.run() 会遍历所有的 Runner，自动生成合成的虚拟输入数据，分别执行推理，并收集每个 Runner 的输出结果，返回一个 RunResults 对象。
    # RunResults 对象本质是一个 Dict[str, RunResult]，其中：key：runner 的名称（如 "onnxrt"、"trt"）；value：RunResult 对象，包含：
    # outputs: Dict[str, np.ndarray]（模型输出）；runtime: float（推理耗时）；error: Optional[Exception]（如果推理失败）；iteration: int（第几次运行，用于多次采样）。
    run_results = Comparator.run(runners)

    # `Comparator.compare_accuracy()` checks that outputs match between runners.
    #
    # TIP: The `compare_func` parameter can be used to control how outputs are compared (see API reference for details).
    #   The default comparison function is created by `CompareFunc.simple()`, but we can construct it
    #   explicitly if we want to change the default parameters, such as tolerance.
    # 5. 精度比较:调用 compare_accuracy 比较 ONNX Runtime 和 TensorRT 的输出。由于浮点数运算在不同框架间可能存在微小差异，
    # 这里使用 CompareFunc.simple(atol=1e-8) 设置了绝对误差容限为1e-8,如果误差超出此范围，断言会失败并抛出异常。
    assert bool(
        Comparator.compare_accuracy(
            run_results, compare_func=CompareFunc.simple(atol=1e-8)
        )
    )

    # We can use `RunResults.save()` method to save the inference results to a JSON file.
    # This can be useful if you want to generate and compare results separately.
    # 6. 保存结果:将推理结果保存到 JSON 文件中，方便后续比较。
    run_results.save("inference_results.json")


if __name__ == "__main__":
    main()


""" 用途:
模型转换验证： 当开发者将训练好的模型转换为 ONNX，再进一步转换为 TensorRT 引擎时，
TensorRT 经常会因为算子不支持、图优化或精度截断（如 FP16/INT8 量化）而导致输出结果与原模型不一致。
此脚本可用于验证转换后的 TensorRT 模型是否保持了原始 ONNX 模型的精度。
回归测试： 在升级 TensorRT 版本或修改模型结构后，运行此脚本可以快速检查新环境是否引入了意料之外的精度衰退。
调试排错： 如果线上部署的 TensorRT 模型输出异常，可以使用此脚本排除 TensorRT 运行时环境的问题，确认是否是模型转换阶段就出现了问题。
"""
