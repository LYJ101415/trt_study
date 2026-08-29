import argparse
import os
import time
import json
import numpy as np
import tensorrt as trt


# ============================================================
# ★ NEW: 引擎元数据管理（版本校验 + 构建信息持久化）
# ============================================================
ENGINE_META_SUFFIX = ".meta.json"

def get_trt_version() -> str:
    """获取当前 TensorRT 运行时版本号字符串"""
    return trt.__version__

def save_engine_meta(engine_path: str, build_info: dict):
    """
    将构建信息保存为与 engine 同名的 .meta.json 文件。
    包含 TRT 版本、构建时间、精度模式、shape profile 等关键信息。
    """
    meta_path = engine_path + ENGINE_META_SUFFIX
    with open(meta_path, "w") as f:
        json.dump(build_info, f, indent=2)
    print(f"[meta] saved {meta_path}")

def load_engine_meta(engine_path: str) -> dict | None:
    """加载已保存的引擎元数据，不存在则返回 None"""
    meta_path = engine_path + ENGINE_META_SUFFIX
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "r") as f:
        return json.load(f)

def validate_engine(engine_path: str) -> bool:
    """
    ★ NEW: 引擎版本校验
    检查 .engine 文件对应的 .meta.json 中记录的 TRT 版本
    是否与当前运行环境一致。不一致时打印警告并返回 False。
    """
    meta = load_engine_meta(engine_path)
    if meta is None:
        print(f"[warn] No meta file found for {engine_path}, skip version check.")
        print(f"[warn] Current TRT version: {get_trt_version()}")
        return True  # 无元数据不阻塞，仅警告

    recorded_version = meta.get("trt_version", "unknown")
    current_version = get_trt_version()

    if recorded_version != current_version:
        print(f"[ERROR] TRT version mismatch!")
        print(f"  Engine built with : {recorded_version}")
        print(f"  Current runtime   : {current_version}")
        print(f"  Engine file       : {engine_path}")
        print(f"  Please rebuild the engine with current TRT version.")
        return False

    print(f"[ok] TRT version match: {current_version}")
    return True


# ============================================================
# ★ NEW: 动态 Shape Profile 解析
# ============================================================
def parse_shape(shape_str: str) -> tuple:
    """
    将命令行传入的 shape 字符串解析为整数元组。
    格式: "1x3x640x640" 或 "1,3,640,640"
    """
    sep = "x" if "x" in shape_str else ","
    return tuple(int(d) for d in shape_str.split(sep))


def add_optimization_profiles(config, network, min_shape, opt_shape, max_shape):
    """
    ★ NEW: 为网络的每个输入添加 Optimization Profile。
    动态 shape 模型必须至少提供一个 profile，否则 build 失败。
    """
    profile = config.create_optimization_profile()
    num_inputs = network.num_inputs

    for i in range(num_inputs):
        input_tensor = network.get_input(i)
        name = input_tensor.name

        # 仅对动态维度的输入设置 profile；静态输入自动跳过
        if -1 in input_tensor.shape:
            profile.set_shape(name, min=min_shape, opt=opt_shape, max=max_shape)
            print(f"  [profile] {name}: min={min_shape} opt={opt_shape} max={max_shape}")
        else:
            print(f"  [profile] {name}: static shape {input_tensor.shape} (skipped)")

    if not profile.is_valid():
        raise RuntimeError(
            "Optimization profile is invalid. "
            "Check that min <= opt <= max for all dimensions."
        )

    config.add_optimization_profile(profile)
    print(f"[build] Added optimization profile (total profiles: {config.num_optimization_profiles})")


# ============================================================
# 核心构建函数（增强版）
# ============================================================
def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = False,
    int8: bool = False,
    workspace_gb: int = 8,
    # ★ NEW: 动态 shape 参数
    min_shape: tuple | None = None,
    opt_shape: tuple | None = None,
    max_shape: tuple | None = None,
):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    # ---------- 解析 ONNX ----------
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  ONNX parse error:", parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

    # ---------- Builder Config ----------
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
    )

    # 精度标志
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # ★ NEW: 动态 shape → 添加 optimization profile
    is_dynamic = any(
        -1 in network.get_input(i).shape for i in range(network.num_inputs)
    )

    if is_dynamic:
        if min_shape is None or opt_shape is None or max_shape is None:
            raise RuntimeError(
                "Model has dynamic input shapes but --min-shape/--opt-shape/--max-shape "
                "were not provided. All three are required for dynamic models."
            )
        add_optimization_profiles(config, network, min_shape, opt_shape, max_shape)
    else:
        print("[build] Static model detected, no optimization profile needed.")

    # ---------- 构建引擎 ----------
    print(
        f"[build] Building from {onnx_path} "
        f"(fp16={fp16}, int8={int8}, workspace={workspace_gb}GB, "
        f"dynamic={is_dynamic})..."
    )

    t0 = time.time()                                    # ★ NEW: 计时
    serialized = builder.build_serialized_network(network, config)
    build_time = time.time() - t0                        # ★ NEW: 计时

    if serialized is None:
        raise RuntimeError("Failed to build engine")

    # ---------- 保存引擎 ----------
    with open(engine_path, "wb") as f:
        f.write(serialized)

    engine_size_mb = os.path.getsize(engine_path) / 1e6
    print(f"[build] Saved {engine_path} ({engine_size_mb:.1f} MB) in {build_time:.1f}s")

    # ★ NEW: 保存构建元数据（用于后续版本校验）
    build_info = {
        "trt_version": get_trt_version(),
        "onnx_path": os.path.abspath(onnx_path),
        "engine_path": os.path.abspath(engine_path),
        "fp16": fp16,
        "int8": int8,
        "workspace_gb": workspace_gb,
        "is_dynamic": is_dynamic,
        "min_shape": list(min_shape) if min_shape else None,
        "opt_shape": list(opt_shape) if opt_shape else None,
        "max_shape": list(max_shape) if max_shape else None,
        "build_time_sec": round(build_time, 2),
        "engine_size_mb": round(engine_size_mb, 2),
    }
    save_engine_meta(engine_path, build_info)


# ============================================================
# 命令行入口
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX (supports Q/DQ INT8+FP16 & dynamic shape)"
    )
    ap.add_argument("--onnx", default="/root/my_FILE/yolov8_int8.onnx")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_waigua.engine")

    # 精度控制
    ap.add_argument("--fp16", action="store_true", default=False,
                    help="Enable FP16 precision")
    ap.add_argument("--int8", action="store_true", default=False,
                    help="Enable INT8 for Q/DQ models (use with --fp16)")
    ap.add_argument("--workspace-gb", type=int, default=8,
                    help="Max GPU workspace in GB during build")

    # ★ NEW: 动态 shape 参数
    ap.add_argument("--min-shape", type=str, default=None,
                    help='Min input shape, e.g. "1x3x320x320"')
    ap.add_argument("--opt-shape", type=str, default=None,
                    help='Optimal input shape, e.g. "1x3x640x640"')
    ap.add_argument("--max-shape", type=str, default=None,
                    help='Max input shape, e.g. "4x3x1280x1280"')

    # ★ NEW: 仅校验已有引擎的版本
    ap.add_argument("--validate-only", action="store_true", default=False,
                    help="Only validate existing engine version, do not build")

    args = ap.parse_args()

    # ★ NEW: 纯校验模式
    if args.validate_only:
        ok = validate_engine(args.engine)
        exit(0 if ok else 1)

    # ★ NEW: 解析动态 shape（仅在提供时解析）
    min_s = parse_shape(args.min_shape) if args.min_shape else None
    opt_s = parse_shape(args.opt_shape) if args.opt_shape else None
    max_s = parse_shape(args.max_shape) if args.max_shape else None

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        int8=args.int8,
        workspace_gb=args.workspace_gb,
        min_shape=min_s,
        opt_shape=opt_s,
        max_shape=max_s,
    )