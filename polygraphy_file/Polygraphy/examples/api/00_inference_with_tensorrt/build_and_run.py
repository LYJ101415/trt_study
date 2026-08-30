"""
该脚本基于一个ONNX恒等模型，构建并运行一个启用了FP16精度的TensorRT引擎。
"""
import numpy as np
from polygraphy.backend.trt import ( # 导入 Polygraphy 中用于 TensorRT 后端的关键组件。
               # 所有类都是惰性加载器（Lazy Loader）：它们本身不执行操作，只在被调用时才触发实际计算。
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    SaveEngine,
    TrtRunner,
)


def main():
    # 我们可以将多个惰性加载器组合在一起，以实现所需的转换。在这个例子中，我们希望完成：ONNX → TensorRT 网络 → TensorRT 引擎（启用 FP16 精度）。
    # “惰性加载器”（lazy loader）：指的是一种延迟执行机制——对象在创建时不立即执行耗时操作（如模型转换、引擎构建），而是在真正需要结果时才触发计算。
    
    # 注意：build_engine 是一个可调用对象（例如函数或类实例），它返回一个引擎，而不是引擎本身。这强调了惰性求值的设计：
    # build_engine 并不直接包含构建好的引擎；而是一个“工厂”或“构造器”，只有在调用它（如 engine = build_engine()）时，才会真正执行构建过程。
    # 好处：避免在初始化阶段就执行昂贵的操作，支持按需加载、缓存、重试等策略。
    
    # 如果你想直接获取引擎，可以使用立即求值的函数式 API。提供了一种替代方案：如果你不需要惰性加载的灵活性，可以直接调用某个函数，该函数会立刻构建并返回引擎。
    # 这种方式更简单直接，适合脚本化或一次性使用的场景。
    #  详见示例 examples/api/06_immediate_eval_api。指向一个具体的代码示例文件，展示如何使用“立即求值”的 API 来绕过惰性加载机制，直接获得引擎。
    
    # 1.构建转换管道
    # NetworkFromOnnxPath：读取指定路径的 ONNX 模型文件，并将其解析为 TensorRT 的网络结构。
    # CreateConfig(fp16=True)：创建一个 TensorRT 构建配置，并启用 FP16 精度模式。
    # 这允许 TensorRT 在 GPU 上使用半精度浮点数进行计算，从而显著提升推理速度并降低显存占用。
    # EngineFromNetwork：将上述网络和配置组合起来，准备生成 TensorRT 引擎。关键点：这里的 build_engine 并不是引擎本身，
    # 而是一个可调用对象。它代表了一个“计划”，只有当它被实际调用时，才会真正执行构建引擎的耗时操作
    build_engine = EngineFromNetwork( 
        NetworkFromOnnxPath("identity.onnx"), config=CreateConfig(fp16=True)
    )  # 注意：config 是一个可选参数。说明调用者不需要强制提供配置对象。
       # 在 TensorRT 中，config 通常指 IBuilderConfig，用于指定构建引擎时的各种选项，例如：
       # 精度模式（FP16、INT8）;工作空间大小（workspace size）;动态形状范围（dynamic shapes）
       # 优化策略等;如果未提供 config，系统会使用默认配置，简化调用流程。

    # 为了在其他地方复用该引擎，我们可以将其序列化并保存到文件中。TensorRT 引擎（ICudaEngine）是平台和硬件相关的——一旦在一个特定 GPU 上构建完成，
    # 就可以序列化为字节流（即“plan 文件”），然后在相同架构的设备上直接反序列化加载，跳过耗时的构建过程。
    # 序列化的好处：避免每次启动都重新从 ONNX 或网络定义构建引擎（构建可能耗时数秒甚至分钟）；
    # 便于部署：将 .engine 或 .plan 文件打包到生产环境；提升启动速度和资源效率。常见做法：首次运行时构建并保存引擎，后续运行直接加载。
    
    # SaveEngine 惰性加载器在被调用时会返回 TensorRT 引擎，SaveEngine 是一个惰性加载器（lazy loader），它本身不是引擎，而是一个可调用对象（如函数或带有 __call__ 方法的类实例）。
    # 当你执行 engine = SaveEngine() 时，它才会：触发上游依赖（如构建好的引擎）；将引擎序列化并写入指定文件；
    # 同时返回该引擎对象（而不是只保存不返回）。这种设计使得 SaveEngine 可以无缝嵌入到处理链中，而不中断数据流。
    # 这使得我们可以将它与其他加载器串联起来。“链式组合”（chaining） 是函数式编程和流水线设计的核心思想。
    
    # 2.保存引擎：SaveEngine 也是一个惰性加载器，它被插入到管道中。当整个管道被触发时，
    # 它会将构建好的 TensorRT 引擎序列化并保存到本地文件 identity.engine 中，同时返回该引擎供后续使用。这样下次运行时可以直接加载引擎，无需重新构建。
    build_engine = SaveEngine(build_engine, path="identity.engine")

    # 一旦我们的加载器准备就绪，推理就只需构造一个运行器（runner）。意思是：模型加载和引擎构建完成后，执行推理变得非常简单——只需创建一个 runner 实例即可。
    # 通过上下文管理器激活它（例如 with TrtRunner(...)），然后调用 infer() 方法。上下文管理器（context manager）：Python 中通过 with 语句实现，用于自动管理资源的获取与释放（如文件、GPU 内存、CUDA 上下文等）。
    # 注意：你可以使用 activate() 函数代替上下文管理器，但你必须确保调用 deactivate() 以避免内存泄漏。因此，上下文管理器是更安全的选择。

    # 3.执行推理：TrtRunner：负责管理 TensorRT 引擎的运行时环境。
    # with 上下文管理器：在进入 with 块时，会触发上述的惰性加载链（构建引擎 -> 保存引擎 -> 加载引擎），并分配推理所需的 GPU 显存等资源；
    # 在退出 with 块时，会自动释放这些资源，防止内存泄漏。runner.infer：执行真正的推理计算。feed_dict 指定了输入张量的名称和数据，返回值是一个包含输出张量的字典。
    with TrtRunner(build_engine) as runner:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # 注意：runner 拥有输出缓冲区，并可在多次 infer() 调用之间自由复用它们。这意味着：第一次调用 infer() 返回的输出数组（如 outputs['out']）指向某块内存 A；
        # 第二次调用 infer() 后，同一块内存 A 的内容会被新结果覆盖；如果你保留了第一次返回的引用，它现在会“神奇地”变成第二次的结果！
        # 设计动机：避免频繁的内存分配/释放（尤其是 GPU 内存），提升推理吞吐量。副作用：返回的输出对象是“瞬态的”（ephemeral），其内容会在下次推理时被修改。
        # 因此，如果你想保存多次推理的结果，应使用 copy.deepcopy()。
        outputs = runner.infer(feed_dict={"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()
