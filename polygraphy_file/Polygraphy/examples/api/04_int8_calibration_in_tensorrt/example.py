"""代码是 Polygraphy 中 INT8 量化校准（Calibration）的标准示例，
展示了如何用极简方式完成原本在原生 TensorRT 中非常繁琐的校准流程。"""

"""
This script demonstrates how to use the Calibrator API provided by Polygraphy
to calibrate a TensorRT engine to run in INT8 precision.
"""
import numpy as np
from polygraphy.backend.trt import (
    Calibrator, # Polygraphy 提供的 INT8 校准器工厂函数，自动处理 GPU 数据拷贝和缓存。
    CreateConfig, # 创建 TensorRT 构建配置（启用 INT8、传入校准器等）。
    EngineFromNetwork,
    NetworkFromOnnxPath,
    TrtRunner,
)
from polygraphy.logger import G_LOGGER # Polygraphy 日志系统，用于控制日志级别（如显示 TensorRT 内部日志）。


# The data loader argument to `Calibrator` can be any iterable or generator that yields `feed_dict`s.
# A `feed_dict` is just a mapping of input names to corresponding inputs.
"""第一部分：校准数据生成器:关键概念：Calibrator 接受任意 可迭代对象（iterable），每次迭代返回一个 feed_dict。
feed_dict：字典，键为模型输入名（如 "x"），值为对应输入张量。"""
def calib_data():
    for _ in range(4):
        # TIP: If your calibration data is already on the GPU, you can instead provide GPU pointers
        # (as `int`s), Polygraphy `DeviceView`s, or PyTorch tensors instead of NumPy arrays.
        #
        # For details on `DeviceView`, see `polygraphy/cuda/cuda.py`.
        """这里是生成 4 个校准样本（真实场景通常需 100~1000 个）。每个样本是 shape (1,1,2,2) 的全 1 张量（因模型是 Identity，无实际意义）。
       高级提示：若数据已在 GPU（如来自 PyTorch DataLoader with pin_memory=True），可直接传 GPU 指针（int）、DeviceView 或 PyTorch tensor，避免 CPU→GPU 拷贝开销。
       DeviceView 是 Polygraphy 对 GPU 内存的轻量封装（类似 torch.Tensor.data_ptr()）。设计优势：数据来源完全灵活——NumPy、PyTorch、自定义生成器均可。"""
        yield {"x": np.ones(shape=(1, 1, 2, 2), dtype=np.float32)}  # Totally real data


def main():
    # We can provide a path or file-like object if we want to cache calibration data.
    # This lets us avoid running calibration the next time we build the engine.
    #
    # TIP: You can use this calibrator with TensorRT APIs directly (e.g. config.int8_calibrator).
    # You don't have to use it with Polygraphy loaders if you don't want to.
    """创建校准器：data_loader=calib_data()：传入生成器（注意：不加括号是函数对象，加括号是调用生成器）。
    cache="identity-calib.cache"：指定缓存文件路径。首次运行：执行校准 → 生成 identity-calib.cache。后续运行：检测到缓存存在 → 跳过校准，直接加载 scale 值。
    重要提示：此 calibrator 对象完全兼容原生 TensorRT，可直接赋值给 config.int8_calibrator。"""
    calibrator = Calibrator(data_loader=calib_data(), cache="identity-calib.cache")

    # We must enable int8 mode in addition to providing the calibrator.
    """构建引擎：NetworkFromOnnxPath("identity.onnx")：加载 ONNX 模型。CreateConfig(int8=True, calibrator=calibrator)：
    int8=True：显式启用 INT8 模式（仅传校准器不够！）。calibrator=calibrator：传入校准器。
    常见错误：忘记设置 int8=True，导致校准器被忽略，引擎仍为 FP32。"""
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath("identity.onnx"),
        config=CreateConfig(int8=True, calibrator=calibrator),
    )

    # When we activate our runner, it will calibrate and build the engine. If we want to
    # see the logging output from TensorRT, we can temporarily increase logging verbosity:
    """上下文管理器：G_LOGGER.verbosity(G_LOGGER.VERBOSE)：临时提升日志级别，显示 TensorRT 内部校准日志（如各层 scale 计算过程）。
    TrtRunner(build_engine)：激活 Runner 时，触发引擎构建（含校准）。"""
    with G_LOGGER.verbosity(G_LOGGER.VERBOSE), TrtRunner(build_engine) as runner:
        # Finally, we can test out our int8 TensorRT engine with some dummy input data:
        inp_data = np.ones(shape=(1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer({"x": inp_data})

        assert np.array_equal(outputs["y"], inp_data)  # It's an identity model!


if __name__ == "__main__":
    main()

"""
真实工业级应用举例: 场景 1：使用真实图片校准 ResNet50
from PIL import Image
import os

def real_calib_data():
    image_dir = "/data/calib_set/"  # 包含 1000 张代表性图片
    for img_name in sorted(os.listdir(image_dir))[:1000]:
        img = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        img = img.resize((224, 224))
        img_np = np.array(img).astype(np.float32) / 255.0  # 归一化
        img_np = np.transpose(img_np, (2, 0, 1))[np.newaxis, :]  # (1,3,224,224)
        yield {"input": img_np}  # ResNet50 输入名为 "input"

calibrator = Calibrator(
    data_loader=real_calib_data(),
    cache="resnet50_int8.cache"
)

build_engine = EngineFromNetwork(
    NetworkFromOnnxPath("resnet50.onnx"),
    config=CreateConfig(int8=True, calibrator=calibrator)
)
关键点：校准数据需覆盖真实场景（如不同光照、遮挡）。

场景 2：与 PyTorch DataLoader 集成（避免 CPU→GPU 拷贝）
import torch

def pytorch_calib_data():
    # 假设 dataloader 返回 GPU 张量
    for batch in calib_dataloader:  # batch.shape = (1, 3, 224, 224) on GPU
        # 直接传递 PyTorch GPU tensor（Polygraphy 自动处理）
        yield {"input": batch}

calibrator = Calibrator(
    data_loader=pytorch_calib_data(),
    cache="model_int8.cache"
)
性能优势：省去 cudaMemcpy 开销，校准速度提升 20%+。

场景 3：混合使用 Polygraphy 校准器 + 原生 TensorRT
# 完全不用 Polygraphy 构建引擎，但用其校准器
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open("model.onnx", "rb") as f:
    parser.parse(f.read())

config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.INT8)
config.int8_calibrator = Calibrator(  # 直接赋值！
    data_loader=calib_data(),
    cache="cache"
)

engine = builder.build_engine(network, config)  # 原生 TensorRT 引擎
无缝互操作：Polygraphy 不强制绑定其生态。

"""
