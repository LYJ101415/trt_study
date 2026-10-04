"""
build_engine.py — 用原生 TensorRT API 从 ONNX 构建 engine（FP16）。

默认针对静态模型 best_PCB.onnx（[1,3,640,640] -> [1,10,8400]）。
用法:
  python build_engine.py                          # 构建 yolov8_fp16.engine
  python build_engine.py --onnx X.onnx --engine Y.engine --fp16
"""

import argparse  # 命令行参数解析库
import os            # 文件操作（获取文件大小等）
import numpy as np   # 数值计算库（本脚本实际未使用，可能是预留或历史遗留）
import tensorrt as trt  # TensorRT Python API 核心库

def build_engine(onnx_path, engine_path, fp16=False, int8=False, workspace_gb=8): # workspace_gb：TensorRT 构建时允许使用的最大 GPU 显存（GB）
    logger = trt.Logger(trt.Logger.WARNING) # 创建 TensorRT 日志器，级别设为 WARNING，只输出警告和错误，过滤掉 INFO/VERBOSE 噪音

    # 一、创建 Builder / Network / Parser 三件套
    builder = trt.Builder(logger) # 创建 Builder 对象，它是整个构建流程的入口，负责管理网络解析、优化策略选择和引擎序列化
    # 创建一个空的 NetworkDefinition 对象。EXPLICIT_BATCH 标志：必须设置。从 TensorRT 7+ 开始，显式 Batch 维度是默认要求，ONNX Parser 依赖它来正确解析动态/静态 batch 维度。
    # 1 << int(...) 是位运算写法，等价于直接传枚举值，这是 TensorRT Python API 的历史惯用写法。
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger) # 创建 ONNX 解析器，绑定到上面的 network 对象。解析器会把 ONNX 图中的节点逐个翻译成 TensorRT 内部 Layer。

    # 二、解析 ONNX 文件
    if not parser.parse_from_file(onnx_path): # 执行 ONNX 解析。返回 True 表示成功，False 表示失败。注意：即使返回 False，parser 内部可能记录了多个错误，需要逐一取出。
        for i in range(parser.num_errors): # 遍历并打印所有解析错误详情。常见错误包括：不支持的 Op、shape 不匹配、Q/DQ 节点属性异常等。
            print("  ONNX parse error:", parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX") # 解析失败则直接抛异常终止，避免后续在无网络的情况下继续构建导致更难排查的错误。

    # 三、配置 Builder Config（核心精度控制）
    config = builder.create_builder_config() # 创建构建配置对象，用于设置精度模式、显存限制、优化级别等。

    """设置 TensorRT 构建时的 Workspace 显存上限。(1 << 30) = 1 GB（字节），所以 workspace_gb * (1 << 30) 把 GB 转为 Bytes。
    Workspace 是 TensorRT 在构建阶段用来尝试不同算法（如 cuDNN conv algo）的临时显存。越大 → 搜索空间越大 → 可能找到更快的 kernel → 但构建时间更长。
    8 GB 是比较安全的工业默认值；显存紧张可降到 2-4 GB，充裕时可升到 16 GB。"""
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    # Q/DQ(显式量化)模型：设 INT8 flag（无需 calibrator，scale 已烘焙在 Q/DQ 节点里）。
    # 注意：ModelOpt PTQ 产出的 Q/DQ 模型常把部分层/层输出标成 FP16，因此还需同时开 FP16，
    # 否则报 "fp16 precision has been set ... but fp16 is not configured"。
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if fp16 and builder.platform_has_fast_fp16: # builder.platform_has_fast_fp16：检查当前 GPU 是否有原生 FP16 Tensor Core
        config.set_flag(trt.BuilderFlag.FP16)

    # 静态模型：无需 optimization profile;如果是动态 shape（如 [-1,3,-1,-1]），则需要额外调用 config.add_optimization_profile() 指定 min/opt/max shape。
    print(f"[build] parsing {onnx_path} (fp16={fp16}, int8={int8}, workspace {workspace_gb}GB)...")
    serialized = builder.build_serialized_network(network, config) # 执行构建
    if serialized is None: # 构建失败保护。常见原因：显存不足、不支持的 Op、精度冲突等。
        raise RuntimeError("Failed to build engine")

    with open(engine_path, "wb") as f:
        f.write(serialized)   # IHostMemory 支持 buffer 协议，直接写入,无需手动 .tobytes() 或拷贝; "wb" 必须以二进制模式写入。
    print(f"[build] saved {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="/root/my_FILE/yolov8_int8.onnx")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_waigua.engine")
    ap.add_argument("--fp16", action="store_true", default=False,
                    help="FP32 模型用：构建 FP16 engine") # action="store_true"：布尔开关，出现即为 True，不需要跟值。
    ap.add_argument("--int8", action="store_true", default=False,
                    help="Q/DQ(已量化)模型用：构建 INT8 engine（需与 --fp16 同用，无需 calibrator）")
    ap.add_argument("--workspace-gb", type=int, default=8)
    args = ap.parse_args()
    build_engine(args.onnx, args.engine, fp16=args.fp16, int8=args.int8,
                 workspace_gb=args.workspace_gb)
