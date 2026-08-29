"""
这是导出是动态模型的构建脚本，用于生成 TensorRT 引擎，在性能上比静态导出的模型差较大。
"""

import argparse
import tensorrt as trt

def build_engine(onnx_path, output_path, min_batch, opt_batch, max_batch,
                 img_size=640, workspace_gb=4, verbose=False):
    # ==================== Logger ====================
    log_level = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(log_level)

    # ==================== Builder & Network ====================
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ==================== Parse ONNX ====================
    print(f"[INFO] Parsing ONNX: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    # 打印输入信息，确认动态维度名称
    inp = network.get_input(0)
    print(f"[INFO] Input name: {inp.name}, shape: {list(inp.shape)}")

    # ==================== Builder Config ====================
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)           # FP16 混合精度（INT8 模型也建议开）
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # ==================== 🔑 动态 Batch Optimization Profile ====================
    profile = builder.create_optimization_profile()
    input_name = inp.name
    channels = inp.shape[1]  # 通常为 3，自动读取避免硬编码

    min_shape = (min_batch, channels, img_size, img_size)
    opt_shape = (opt_batch, channels, img_size, img_size)
    max_shape = (max_batch, channels, img_size, img_size)

    profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
    config.add_optimization_profile(profile)

    print(f"[INFO] Dynamic batch profile:")
    print(f"       min={min_shape}, opt={opt_shape}, max={max_shape}")

    # ==================== Build ====================
    print("[INFO] Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    # ==================== Save ====================
    with open(output_path, "wb") as f:
        f.write(serialized_engine)

    print(f"✅ Engine saved: {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build TensorRT engine with dynamic batch")
    p.add_argument("--onnx",      type=str, default="/root/my_FILE/models/yolov8_int8_dynamic_fidex.onnx", help="量化后 ONNX 路径")
    p.add_argument("--output",    type=str, default="/root/my_FILE/models/yolov8_int8_dynamic_1.engine", help="输出 .engine 路径")
    p.add_argument("--min-batch", type=int, default=1,     help="最小 batch size")
    p.add_argument("--opt-batch", type=int, default=1,     help="最优 batch size（最常用）")
    p.add_argument("--max-batch", type=int, default=1,     help="最大 batch size")
    p.add_argument("--img-size",  type=int, default=640,   help="输入图像尺寸")
    p.add_argument("--workspace", type=int, default=4,     help="Workspace 大小 (GB)")
    p.add_argument("--verbose",   action="store_true",     help="开启 VERBOSE 日志")
    args = p.parse_args()

    build_engine(
        onnx_path=args.onnx,
        output_path=args.output,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        img_size=args.img_size,
        workspace_gb=args.workspace,
        verbose=args.verbose,
    )


"""
以下是动态模型构建引擎时采用固定batch的命令：
trtexec \
  --onnx=yolov8_int8_dynamic.onnx \
  --saveEngine=yolov8_fixed.engine \
  --int8 \
  --calib=calibration.cache \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:1x3x640x640 \
  --fp16  # INT8 推理时中间层可开 FP16 提升精度/速度
"""