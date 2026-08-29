import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# 1. 解析量化后的 ONNX 模型
onnx_path = "/root/my_FILE/yolov8_int8.onnx "  # 确保这个文件名和你之前生成的文件名一致
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX model")

# 2. 配置构建参数
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB workspace
config.set_flag(trt.BuilderFlag.FP16)  # 开启 FP16 混合精度

print("Building TensorRT engine...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("Failed to build TensorRT engine")

# 3. 保存引擎文件
output_path = "yolov8_int8.trt"
with open(output_path, "wb") as f:
    f.write(serialized_engine)

print(f"✅ TensorRT engine saved as {output_path}")