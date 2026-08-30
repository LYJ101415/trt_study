"""以下代码演示了如何：从 ONNX 模型构建支持 多个 Optimization Profiles 的 TensorRT 引擎；
为不同业务场景（低延迟、动态批处理、离线处理）创建 独立的推理上下文（Execution Context）；
使用 Polygraphy 的 TrtRunner 简化动态 shape 推理流程；在运行时 按需激活不同 runner，实现灵活高效的部署策略。
模型是 dynamic_identity.onnx：一个恒等映射模型（输出 = 输入），仅用于验证 shape 传递正确性。"""

"""
This script builds an engine with 3 separate optimization profiles, each
built for a specific use-case. It then creates 3 separate execution contexts
and corresponding `TrtRunner`s for inference.
"""
import numpy as np
from polygraphy.backend.trt import ( # polygraphy.backend.trt：TensorRT 高级封装工具
    CreateConfig,
    Profile,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
    save_engine,
)
from polygraphy.logger import G_LOGGER


def main():
    # A Profile maps each input tensor to a range of shapes.
    # The `add()` method can be used to add shapes for a single input.
    #
    # TIP: To save lines, calls to `add` can be chained:
    #     profile.add("input0", ...).add("input1", ...)
    #
    #   Of course, you may alternatively write this as:
    #     profile.add("input0", ...)
    #     profile.add("input1", ...)
    """步骤 1：定义多个 Optimization Profiles:关键点：每个 Profile 对应一种业务场景
    min == opt == max 表示 固定 shape，可获得最佳性能;opt 必须设为 最常出现的 shape（如线上 80% 请求是 batch=4）
    命名规范：输入张量名为 "X"，输出为 "Y"（由 ONNX 模型决定）"""
    profiles = [
        # The low-latency case. For best performance, min == opt == max.
        """场景1：低延迟在线服务（batch=1）"""
        Profile().add("X", min=(1, 3, 28, 28), opt=(1, 3, 28, 28), max=(1, 3, 28, 28)),
        # The dynamic batching case. We use `4` for the opt batch size since that's our most common case.
        """场景2：动态批处理（batch 范围 [1,32]，最常用 batch=4）"""
        Profile().add("X", min=(1, 3, 28, 28), opt=(4, 3, 28, 28), max=(32, 3, 28, 28)),
        # The offline case. For best performance, min == opt == max.
        """场景3：离线大批量处理（batch=128）"""
        Profile().add(
            "X", min=(128, 3, 28, 28), opt=(128, 3, 28, 28), max=(128, 3, 28, 28)
        ),
    ]

    # See examples/api/06_immediate_eval_api for details on immediately evaluated functional loaders like `engine_from_network`.
    # Note that we can freely mix lazy and immediately-evaluated loaders.
    """步骤 2：构建支持多 Profile 的 TensorRT Engine:network_from_onnx_path()：加载 ONNX 模型（必须含动态维度，如 batch=-1）;
    CreateConfig(profiles=profiles)：将多个 profile 注入构建配置;engine_from_network()：立即构建引擎（非 lazy 模式）
    前提：ONNX 模型必须使用 explicit batch 导出（PyTorch 默认满足）"""
    engine = engine_from_network(
        network_from_onnx_path("dynamic_identity.onnx"),
        config=CreateConfig(profiles=profiles),
    )

    # We'll save the engine so that we can inspect it with `inspect model`.
    # This should make it easy to see how the engine bindings are laid out.
    """步骤 3：保存引擎供调试:生成 .engine 文件，可用以下命令检查：polygraphy inspect model dynamic_identity.engine --mode=full"""
    save_engine(engine, "dynamic_identity.engine")

    # We'll create, but not activate, three separate runners, each with a separate context.
    #
    # TIP: By providing a context directly, as opposed to via a lazy loader,
    # we can ensure that the runner will *not* take ownership of it.
    """步骤 4：为每个 Profile 创建独立 Runner:创建 第一个 context（默认使用 profile 0）;TrtRunner 封装了内存分配、shape 设置等细节"""
    low_latency = TrtRunner(engine.create_execution_context())

    # NOTE: The following two lines may cause TensorRT to display errors since profile 0
    # is already in use by the first execution context. We'll suppress them using G_LOGGER.verbosity().
    """抑制日志：因多个 context 共享 engine，TRT 可能报 warning（实际无害）;optimization_profile=1：显式指定使用 第二个 profile（索引从 0 开始）;
    offline runner 未指定 profile，后续需手动设置;为什么需要多个 context？单个 context 无法同时持有多个 profile 的优化状态。多 context = 多套优化策略。"""
    with G_LOGGER.verbosity(G_LOGGER.CRITICAL):
        # We can use the `optimization_profile` parameter of the runner to ensure that the correct optimization profile is used.
        # This eliminates the need to call `set_profile()` later.
        dynamic_batching = TrtRunner(
            engine.create_execution_context(), optimization_profile=1
        )  # Use the second profile, which is intended for dynamic batching.

        # For the sake of example, we *won't* use `optimization_profile` here.
        # Instead, we'll use `set_profile()` after activating the runner.
        offline = TrtRunner(engine.create_execution_context())

    # Finally, we can activate the runners as we need them.
    #
    # NOTE: Since the context and engine are already created, the runner will only need to
    # allocate input and output buffers during activation.
    """步骤 5：按需激活 Runner 并推理:"""
    input_img = np.ones((1, 3, 28, 28), dtype=np.float32)  # An input "image"
    """场景1：低延迟推理（batch=1）"""
    with low_latency:
        outputs = low_latency.infer({"X": input_img})
        assert np.array_equal(outputs["Y"], input_img)  # It's an identity model!

        print("Low latency runner succeeded!")

        # While we're serving requests online, we might decide that we need dynamic batching
        # for a moment.
        #
        # NOTE: We're assuming that activating runners will be cheap here, so we can bring up
        # the dynamic batching runner just-in-time.
        #
        # TIP: If activating the runner is not cheap (e.g. input/output buffers are large),
        # it might be better to keep the runner active the whole time.
        #
        with dynamic_batching:
            # We'll create fake batches by repeating our fake input image.
            small_input_batch = np.repeat(input_img, 4, axis=0)  # Shape: (4, 3, 28, 28)
            outputs = dynamic_batching.infer({"X": small_input_batch})
            assert np.array_equal(outputs["Y"], small_input_batch)

    # If we need dynamic batching again later, we can activate the runner once more.
    #
    # NOTE: This time, we do *not* need to set the profile.
    """场景2：动态批处理（batch=4 和 batch=16）"""
    with dynamic_batching:
        # NOTE: We can use any shape that's in the range of the profile without
        # additional setup - Polygraphy handles the details behind the scenes!
        #
        large_input_batch = np.repeat(input_img, 16, axis=0)  # Shape: (16, 3, 28, 28)
        outputs = dynamic_batching.infer({"X": large_input_batch})
        assert np.array_equal(outputs["Y"], large_input_batch)

        print("Dynamic batching runner succeeded!")
    """场景3：离线大批量处理（batch=128）"""
    with offline:
        # NOTE: When we first activate this runner, we need to set the profile index (it's 0 by default).
        # Since we provided our own execution context when we created the runner, we need to do this *only once*.
        # Our settings persist since the context will remain alive even after the runner is deactivated.
        # If we had instead allowed the runner to own the context, we'd need to repeat this step each time we activated the runner.
        #
        # Alternatively, we could have used the `optimization_profile` parameter (see above).
        #
        offline.set_profile( # 手动切换到 profile 2（索引=2）
            2
        )  # Use the third profile, which is intended for the offline case.

        large_offline_batch = np.repeat(
            input_img, 128, axis=0
        )  # Shape: (128, 3, 28, 28)
        outputs = offline.infer({"X": large_offline_batch})
        assert np.array_equal(outputs["Y"], large_offline_batch)

        print("Offline runner succeeded!")


if __name__ == "__main__":
    main()
