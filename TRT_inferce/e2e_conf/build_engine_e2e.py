"""
build_engine_e2e.py — 从含 Q/DQ 节点 + 烧录预/后处理的 e2e ONNX 构建 engine。

特性:
  1. INT8(Q/DQ) 模型自动同时启用 FP16（显式量化模型里有层被标记 fp16，
     不开 FP16 flag 会报 "fp16 is not configured in the builder" 构建失败）
  2. ★ optimization profile 自动生成: 解析 ONNX 后探测每个输入的动态维(-1)，
     按【全局配置区】的范围创建 profile —— 静态输入(iou_thresh/score_thresh [1])自动跳过，
     全静态模型不创建任何 profile。换任何模型都不需要改函数签名。
  3. --detailed 记录详细层信息（Precision/Tactic），供 engine inspector 逐层分析

动态维范围在下方【全局配置区】设置，按部署实际负载修改:
    全动态 (-1,-1,-1,3) 模型 → batch/H/W 均取配置范围
    (1,-1,-1,3) 模型        → batch 静态位自动固定为 1
    全静态 (1,3,640,640)    → 无需 profile

用法:
    python build_engine_e2e.py                      # 默认 INT8 e2e engine
    python build_engine_e2e.py --no-int8            # 不用量化(忽略 Q/DQ)
    python build_engine_e2e.py --detailed           # 详细层信息(调试)
"""

import argparse
import os
import tensorrt as trt

# ==================== 全局配置区（按部署情况修改这里） ====================

DEFAULT_ONNX = "/root/my_FILE/models/yolov8_e2e_dy.onnx"
DEFAULT_ENGINE = "/root/my_FILE/models/yolov8_e2e_dyn.engine"
WORKSPACE_GB = 8

# 动态维的取值范围 (min, opt, max)。
# ★ opt = 实际部署中最常见的输入形状（tactic/性能按 opt 优化），min/max 是允许的上下限。
# 位置对应 4 维 NHWC 输入: 0=batch, 1=height, 2=width（channel 等静态维不走这里）。
DYNAMIC_DIM_RANGES = {
    0: (1, 1, 8),            # batch: 部署多为单张则 opt=1；批量吞吐可改 (1, 4, 8)
    1: (128, 640, 1920),      # height: min=64（更小的输入会失败），opt=最常见尺寸
    2: (128, 640, 1920),      # width
}

# =========================================================================


def _add_profiles(builder, config, network):
    """为每个含动态维(-1)的输入自动创建 optimization profile。
    静态输入（如 iou_thresh/score_thresh [1]）自动跳过；全静态模型一个都不创建。
    返回是否创建了 profile。"""
    created = False
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        dims = list(inp.shape)
        if all(d >= 0 for d in dims):
            continue                                   # 全静态输入，无需 profile
        shapes = ([], [], [])                          # min, opt, max
        for pos, d in enumerate(dims):
            if d >= 0:                                 # 静态维: 三个形状都钉死为该值
                for s in shapes:
                    s.append(d)
            else:                                      # 动态维: 从全局配置取 (min,opt,max)
                if pos not in DYNAMIC_DIM_RANGES:
                    raise RuntimeError(
                        f"输入 {inp.name} 的第 {pos} 维是动态的，但 DYNAMIC_DIM_RANGES "
                        f"没有配置该维的范围，请在全局配置区补充")
                for s, v in zip(shapes, DYNAMIC_DIM_RANGES[pos]):
                    s.append(v)
        mn, od, mx = map(tuple, shapes)
        if not all(a <= b <= c for a, b, c in zip(mn, od, mx)):
            raise RuntimeError(f"{inp.name} profile 不满足 min<=opt<=max: {mn} {od} {mx}")
        profile = builder.create_optimization_profile()
        profile.set_shape(inp.name, mn, od, mx)
        config.add_optimization_profile(profile)
        print(f"[profile] {inp.name}: min={mn} opt={od} max={mx}")
        created = True
    return created


def build(onnx_path, engine_path, fp16=False, int8=False, workspace_gb=WORKSPACE_GB,
          detailed=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print("  ONNX parse error:", parser.get_error(i))
        raise RuntimeError("Failed to parse ONNX")

    # ★ 显式量化(Q/DQ)模型必须同时开 INT8 + FP16，
    #   否则被标记 fp16 的层会触发 "fp16 is not configured" 构建失败
    if int8 and not fp16:
        print("[build] INT8(Q/DQ) 模型自动同时启用 FP16 回退")
        fp16 = True

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 元信息详细程度（构建期写死，推理端只读）：
    #   LAYER_NAMES_ONLY(默认) — inspector 只能给出层名
    #   DETAILED              — 额外记录每层的 Precision/输入输出 dtype/Tactic 名等
    #   代价：构建略慢、engine 略大，不影响运行性能 → 调试用，默认关闭
    if detailed:
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    # ★ 自动 profile: 只给含动态维的输入建；全静态模型不建
    _add_profiles(builder, config, network)

    print(f"[build-e2e-conf] parsing {onnx_path} (fp16={fp16}, int8={int8}) ...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Failed to build engine")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[build-e2e-conf] saved {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=DEFAULT_ONNX)
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--fp16", action="store_true", default=False,
                    help="FP32 模型：构建 FP16 engine")
    ap.add_argument("--int8", action=argparse.BooleanOptionalAction, default=True,
                    help="Q/DQ 模型：构建 INT8 engine（--no-int8 可关闭）")
    ap.add_argument("--workspace-gb", type=int, default=WORKSPACE_GB)
    ap.add_argument("--detailed", action="store_true", default=False,
                    help="构建时记录详细层信息（Precision/Tactic 等），供 engine inspector 逐层分析")
    args = ap.parse_args()
    build(args.onnx, args.engine, fp16=args.fp16, int8=args.int8,
          workspace_gb=args.workspace_gb, detailed=args.detailed) # 参数detailed=args.detailed在真正部署时构建可去除，省一点模型文件大小。

"""
命令	                 int8	实际生效的 flag	适用模型
build_engine_e2e.py	    True	INT8 + FP16	带 Q/DQ 的量化 ONNX
... --no-int8 --fp16	False	仅 FP16	普通 FP32 ONNX
... --no-int8	False	都不开	 FP32 基线对比
核心逻辑一句话：Q/DQ 模型 = INT8 + FP16 必须成对出现；纯 FP32 模型想转 FP16 才需要手动加 --fp16
"""
