"""
make_stage2_float_onnx.py — 图手术: 把 uint8 输入的 e2e ONNX 改成 float 输入变体。

手术内容（数值逐位等价，权重/Q-DQ 零改动）:
    1. 输入 image_raw 的 elem_type: UINT8 → FLOAT（值语义不变，仍是 0~255 原始像素）
    2. 紧跟输入的 Cast(uint8→float) 节点 → Identity（float 已就位，直通）

为什么要做: TRT10 限制 uint8 不能被网络中间层消费，GPU 裁剪 plugin 输出 float，
二级引擎需要 float 输入版本（详见 PLUGIN_TUTORIAL.md 第 6 节）。

用法:
    python make_stage2_float_onnx.py                          # 默认路径，只做图手术产 ONNX
    python make_stage2_float_onnx.py --build-engine           # 手术后接着构建 TRT 引擎
    python make_stage2_float_onnx.py --src a.onnx --dst b.onnx
"""

import argparse
import sys
from pathlib import Path

import onnx


def graph_surgery_uint8_to_float(src_onnx, dst_onnx, input_name="image_raw"):
    """输入 dtype UINT8→FLOAT + Cast→Identity。返回手术摘要信息。"""
    model = onnx.load(src_onnx)
    g = model.graph

    # ---- 1. 输入类型 UINT8 → FLOAT ----
    changed_input = None
    for i in g.input:
        if i.name == input_name:
            assert i.type.tensor_type.elem_type == onnx.TensorProto.UINT8, \
                f"{input_name} 不是 UINT8，无需手术或模型不符"
            i.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
            changed_input = i.name

    # ---- 2. Cast(image_raw) → Identity ----
    # 原: image_raw(uint8) --Cast--> img_f(float) --...--> 后续
    # 改: image_raw(float) --Identity--> img_f(float) --...--> 后续
    #     (后续节点全部引用 img_f，改 Cast 自身即可，不动任何消费者)
    cast_nodes = []
    for n in g.node:
        if n.op_type == "Cast" and n.input and n.input[0] == input_name:
            cast_nodes.append(n)
    for n in cast_nodes:
        n.op_type = "Identity"
        del n.attribute[:]                      # 删除 to=FLOAT 属性(Identity 无属性)

    onnx.checker.check_model(model)
    onnx.save(model, dst_onnx)
    return {
        "input": changed_input,
        "cast_to_identity": len(cast_nodes),
        "nodes": len(g.node),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/root/my_FILE/models/yolov8_int8_exclude_e2e_dyn.onnx",
                    help="uint8 输入的源 ONNX")
    ap.add_argument("--dst", default="/root/my_FILE/models/yolov8_int8_exclude_e2e_dyn_float.onnx",
                    help="手术后的 float 输入 ONNX")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_e2e_dyn_float.engine",
                    help="[可选] 同时构建的 TRT 引擎输出路径")
    ap.add_argument("--build-engine", action="store_true",
                    help="图手术完成后接着构建 TRT 引擎（约 4 分钟）")
    args = ap.parse_args()

    info = graph_surgery_uint8_to_float(args.src, args.dst)
    print(f"[surgery] {args.src}")
    print(f"      ->  {args.dst}")
    print(f"      输入 {info['input']}: UINT8 -> FLOAT")
    print(f"      Cast -> Identity: {info['cast_to_identity']} 个节点 (共 {info['nodes']} 节点)")
    print(f"      onnx.checker: 通过")

    if args.build_engine:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "e2e_conf"))
        from build_engine_e2e import build
        build(args.dst, args.engine, fp16=True, int8=True)


if __name__ == "__main__":
    main()
