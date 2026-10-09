"""
build_engine.py — 从静态端到端 ONNX(yolov8_int8_static_e2e.onnx) 构建 engine。

端到端模型的三个输入都是静态 shape：
  image_raw     (1, 640, 640, 3) uint8   CPU 解码并 resize 后的图
  iou_thresh    (1,) float32
  score_thresh  (1,) float32
无需 optimization profile（没有动态维度）。

用法:
    python build_engine.py --onnx /root/my_FILE/models/yolov8_int8_static_e2e.onnx \
                           --engine /root/my_FILE/models/yolov8_int8_static_e2e.engine
"""

import argparse
import os
import tensorrt as trt


def build(onnx_path, engine_path, fp16=False, int8=False, workspace_gb=8):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  ONNX parse error:", parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    # Q/DQ(显式量化)模型：设 INT8 flag（无需 calibrator）。
    # 注意：ModelOpt PTQ 产出的 Q/DQ 模型常把部分层标成 FP16，需同时开 FP16，否则报错。
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"[build-e2e] parsing {onnx_path} (fp16={fp16}, int8={int8})...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build engine")

    with open(engine_path, "wb") as f:
        f.write(serialized)   # IHostMemory 支持 buffer 协议，直接写入
    print(f"[build-e2e] saved {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/root/my_FILE/models/yolov8_int8_static_e2e.onnx")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_static_e2e.engine")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="FP32 模型用：构建 FP16 engine")
    ap.add_argument("--int8", action="store_true", default=True,
                    help="Q/DQ(已量化)模型用：构建 INT8 engine（无需 calibrator）")
    ap.add_argument("--workspace-gb", type=int, default=8)
    args = ap.parse_args()
    build(args.onnx, args.engine, fp16=args.fp16, int8=args.int8,
          workspace_gb=args.workspace_gb)
