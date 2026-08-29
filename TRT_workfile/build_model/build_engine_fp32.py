"""
构建engine引擎，有加参数ProfilingVerbosity.DETAILED，打印层结构信息时详细输出;
并且指定忽略的层用fp32，能在构建引擎时增加一些精度。
"""

import tensorrt as trt

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))  
parser = trt.OnnxParser(network, logger)
onnx_path = "/root/my_FILE/yolov8_int8_1.onnx"
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise RuntimeError("Failed to parse ONNX model")

# ============ 第一步：先打印所有层的真实名字，核对目标层 ============
print("=" * 90)
for i in range(network.num_layers):
    layer = network.get_layer(i)
    print(f"[{i:3d}] name={layer.name!r:55s} type={layer.type}")
print("=" * 90)

# ============ 第二步：钉成 FP32 的目标层名（按上面打印结果填写）============
# 注意：这里必须是 network 里 layer.name 的"完整精确名"，不是 ModelOpt 的排除名
FP32_TARGETS = [
    "/model.1/conv/Conv",
    "/model.1/act/Sigmoid",   # 该层后接的激活（SiLU = Sigmoid + Mul）也一起钉
    "/model.1/act/Mul",
    # 第二个被排除层同理，例如：
    "/model.22/dfl/conv/Conv",
    # "/model.22/act/Sigmoid",
    # "/model.22/act/Mul",
]

def pin_to_fp32(layer):
    layer.precision = trt.float32          # 计算精度钉成 FP32
    for i in range(layer.num_outputs):     # 输出类型也钉成 FP32，防止 Reformat 再转 FP16
        layer.set_output_type(i, trt.float32)

pinned = 0
for i in range(network.num_layers):
    layer = network.get_layer(i)
    if layer.name in FP32_TARGETS:
        pin_to_fp32(layer)
        pinned += 1
        print(f"✅ Pinned to FP32: {layer.name}")
print(f"共钉扎 {pinned} 个层\n")

# ============ 第三步：构建配置 ============
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
config.set_flag(trt.BuilderFlag.FP16)                                # 其余非 INT8 层允许 FP16
config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)    # ★ 严格服从上面的精度钉扎;
         # 另一个 flag 是 PREFER_PRECISION_CONSTRAINTS（只是偏好，仍可违反），精度敏感场景用 OBEY，不要用 PREFER。
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

print("Building TensorRT engine...")
serialized_engine = builder.build_serialized_network(network, config)

if serialized_engine is None:
    raise RuntimeError("Failed to build TensorRT engine")

output_path = "yolov8_int8_fixed.engine"
with open(output_path, "wb") as f:
    f.write(serialized_engine)
print(f"✅ TensorRT engine saved as {output_path}")
