"""
build_engine.py — 从 best_PCB_e2e_conf.onnx 构建「运行时阈值」端到端 engine。

输入: image_raw(动态 H/W uint8) + iou_thresh([1] f32) + score_thresh([1] f32)
只有 image_raw 需要 optimization profile，两个标量输入是静态 shape [1]。

用法:
    python build_engine.py --onnx /root/my_FILE/models/best_PCB_e2e_conf.onnx \
                           --engine /root/my_FILE/models/yolov8_e2e_conf_fp16.engine [--int8]
"""

import argparse
import os
import tensorrt as trt


def build(onnx_path, engine_path, fp16=False, int8=False, workspace_gb=8,
          min_shape=(1, 64, 64, 3), opt_shape=(1, 640, 640, 3), max_shape=(1, 1920, 1920, 3)):
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
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 只有动态的 image_raw 需要 profile；iou_thresh/score_thresh 是静态 [1]
    profile = builder.create_optimization_profile()
    profile.set_shape("image_raw", min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"[build-e2e-conf] parsing {onnx_path} (fp16={fp16}, int8={int8}) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build engine")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build-e2e-conf] saved {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/root/my_FILE/models/best_PCB_e2e_conf.onnx")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_e2e_conf_fp16.engine")
    ap.add_argument("--fp16", action="store_true", default=True,
                    help="FP32 模型：构建 FP16 engine")
    ap.add_argument("--int8", action="store_true", default=False,
                    help="Q/DQ 模型：构建 INT8 engine")
    ap.add_argument("--workspace-gb", type=int, default=8)
    args = ap.parse_args()
    build(args.onnx, args.engine, fp16=args.fp16, int8=args.int8,
          workspace_gb=args.workspace_gb)
