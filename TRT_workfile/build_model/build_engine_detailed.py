"""
构建engine引擎，有加参数ProfilingVerbosity.DETAILED，打印层结构信息时详细输出
"""

import tensorrt as trt

#  将日志级别从 WARNING 改为 VERBOSE，否则 detailed 信息可能被过滤
# logger = trt.Logger(trt.Logger.VERBOSE) # 终端输出太多
logger = trt.Logger(trt.Logger.WARNING) # 屏蔽终端输出

builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# 1. 解析量化后的 ONNX 模型
onnx_path = "/root/my_FILE/yolov8_int8_fixed_1.onnx"
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX model")

# 2. 配置构建参数
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB workspace
config.set_flag(trt.BuilderFlag.FP16)  # 开启 FP16 混合精度

# 设置 profiling verbosity 为 DETAILED，在以后查看层信息时会有很详细信息
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

print("Building TensorRT engine...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("Failed to build TensorRT engine")

# 3. 保存引擎文件
output_path = "yolov8_int8_fixed_1.engine"
with open(output_path, "wb") as f:
    f.write(serialized_engine)

print(f"✅ TensorRT engine saved as {output_path}")