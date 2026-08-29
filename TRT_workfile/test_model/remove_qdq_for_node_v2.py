#!/usr/bin/env python3
"""
remove_qdq_for_node.py (增强稳健版 v3)：
删除指定算子节点上游的 Q/DQ 对, 修复了节点名重复、多输入中间节点悬空、孤立权重等隐患。

用法:
    python remove_qdq_for_node.py --model m.onnx --node Conv_250 --out o.onnx
    # 允许连带移除共享 DQ 的其他节点
    python remove_qdq_for_node.py --model m.onnx --node Conv_250 --out o.onnx --force-shared
"""

import argparse
import sys
import os

# ============================================================
#  拓扑分析工具
# ============================================================

def build_tensor_maps(graph):
    """构建 tensor→生产者 / tensor→消费者列表 双向映射"""
    producer = {}   # tensor_name → 生产该 tensor 的节点
    consumers = {}  # tensor_name → 消费该 tensor 的节点列表
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return producer, consumers

def trace_dq_to_q(dq_node, tensor_producer, visited=None):
    """
    从 DQ 节点反向追溯到 QuantizeLinear。
    【优化点】收紧了 PASSTHROUGH_OPS，仅允许严格的单输入单输出数据流算子，
    避免 Concat/Slice 等多输入算子的辅助输入成为悬空引用。
    """
    if visited is None:
        visited = set()

    # 严格的单输入单输出形状/类型变换节点 (移除了 Concat, Split, Slice, Pad 等多输入算子)
    PASSTHROUGH_OPS = {
        "Reshape", "Transpose", "Squeeze", "Unsqueeze",
        "Flatten", "Expand", "Cast", "Identity", "Tile"
    }

    q_input = dq_node.input[0] if dq_node.input else None
    if not q_input:
        return None, None, []

    intermediates = []
    current_tensor = q_input
    max_depth = 10

    for _ in range(max_depth):
        if current_tensor in visited:
            break
        visited.add(current_tensor)

        prod = tensor_producer.get(current_tensor)
        if prod is None:
            break

        if prod.op_type == "QuantizeLinear":
            upstream = prod.input[0] if prod.input else None
            return prod, upstream, intermediates

        if prod.op_type in PASSTHROUGH_OPS:
            intermediates.append(prod)
            current_tensor = prod.input[0] if prod.input else None
            if not current_tensor:
                break
        else:
            break

    return None, None, []


def find_all_qdq_chains_for_node(model, consumer_node_name, verbose=False):
    """查找目标节点的所有 Q/DQ 链"""
    graph = model.graph
    producer, consumers_map = build_tensor_maps(graph)

    consumer = None
    for node in graph.node:
        if node.name == consumer_node_name:
            consumer = node
            break

    if consumer is None:
        print(f"[ERROR] 未找到节点: {consumer_node_name}")
        return []

    chains = []
    for idx, inp_tensor in enumerate(consumer.input):
        if not inp_tensor:
            continue

        dq_candidate = producer.get(inp_tensor)
        if dq_candidate is None or dq_candidate.op_type != "DequantizeLinear":
            continue

        q_node, upstream, intermediates = trace_dq_to_q(dq_candidate, producer)

        if q_node is None or upstream is None:
            continue

        shared = []
        dq_out = dq_candidate.output[0] if dq_candidate.output else inp_tensor
        for c in consumers_map.get(dq_out, []):
            if c.name != consumer_node_name:
                shared.append(c.name)

        chain = {
            "input_index": idx,
            "input_tensor": inp_tensor,
            "dq_node": dq_candidate,
            "q_node": q_node,
            "upstream_tensor": upstream,
            "dq_output": dq_out,
            "intermediates": intermediates,
            "shared_consumers": shared,
        }
        chains.append(chain)

        if verbose:
            inter_names = [n.name for n in intermediates]
            inter_str = f" via {inter_names}" if inter_names else ""
            shared_str = f" (共享: {shared})" if shared else ""
            print(f"  输入[{idx}] ✓ Q={q_node.name} →{inter_str}→ DQ={dq_candidate.name} → '{inp_tensor}'{shared_str}")

    return chains

# ============================================================
#  图修改
# ============================================================

def remove_chains_and_reconnect(model, chains, consumer_node_name, force_shared=False, verbose=False):
    """
    根据找到的 Q/DQ 链执行删除和重连。
    【优化点】1. 使用对象引用而非 node.name 删除防重名; 2. 增加 force_shared 校验; 3. 清理 Initializer
    """
    graph = model.graph

    # 检查共享情况
    for chain in chains:
        if chain["shared_consumers"] and not force_shared:
            shared_list = ", ".join(chain["shared_consumers"])
            raise ValueError(
                f"安全拦截: DQ '{chain['dq_node'].name}' 被其他节点共享: [{shared_list}]。"
                f"移除它会导致这些节点也被迫回退FP32。如确需执行，请添加 --force-shared 参数。"
            )

    # 使用对象引用收集需要删除的节点，避免 node.name 重复或为空导致误删
    nodes_to_remove_obj = set()
    reconnect_map = {}  

    for chain in chains:
        nodes_to_remove_obj.add(id(chain["q_node"]))
        nodes_to_remove_obj.add(id(chain["dq_node"]))
        for mid in chain["intermediates"]:
            nodes_to_remove_obj.add(id(mid))
        reconnect_map[chain["dq_output"]] = chain["upstream_tensor"]

    # Step 1: 全局替换所有下游引用
    total_replaced = 0
    for node in graph.node:
        if id(node) in nodes_to_remove_obj:
            continue
        for i, inp in enumerate(node.input):
            if inp in reconnect_map:
                node.input[i] = reconnect_map[inp]
                total_replaced += 1

    # 替换 graph output 引用 (需保持 ValueInfoProto 对象不变，只改内部 name 可能导致外部依赖失效，此处保留原逻辑但加日志警告)
    for out in graph.output:
        if out.name in reconnect_map:
            old_name = out.name
            out.name = reconnect_map[out.name]
            print(f"  ⚠️ 警告: 模型输出张量 '{old_name}' 被替换为 '{out.name}', 可能影响外部推理接口!")

    print(f"  全局替换了 {total_replaced} 处 tensor 引用")
    for dq_out, up in reconnect_map.items():
        print(f"    '{dq_out}' → '{up}'")

    # Step 2: 删除节点 (基于对象引用过滤)
    keep = [n for n in graph.node if id(n) not in nodes_to_remove_obj]
    del graph.node[:]
    graph.node.extend(keep)
    print(f"  删除了 {len(nodes_to_remove_obj)} 个节点对象")

    # Step 3: 清理 value_info 和 initializer
    removed_tensors = set()
    for chain in chains:
        for t in chain["q_node"].output: removed_tensors.add(t)
        for t in chain["dq_node"].output: removed_tensors.add(t)
        for t in chain["dq_node"].input:
            if t != chain["upstream_tensor"]: removed_tensors.add(t)
        for mid in chain["intermediates"]:
            for t in mid.output: removed_tensors.add(t)
            for t in mid.input:
                if t != chain["upstream_tensor"]: removed_tensors.add(t)

    # 清理 value_info
    vi_before = len(graph.value_info)
    vi_keep = [v for v in graph.value_info if v.name not in removed_tensors]
    del graph.value_info[:]
    graph.value_info.extend(vi_keep)
    print(f"  清理了 {vi_before - len(vi_keep)} 个孤立 value_info 条目")

    # 【新增】清理 Initializer (权重/常量)，解决模型虚胖问题
    used_tensors = set()
    for node in graph.node:
        for inp in node.input: used_tensors.add(inp)
        for out in node.output: used_tensors.add(out)
    for out in graph.output: used_tensors.add(out.name)

    init_before = len(graph.initializer)
    init_keep = [init for init in graph.initializer if init.name in used_tensors]
    del graph.initializer[:]
    graph.initializer.extend(init_keep)
    print(f"  清理了 {init_before - len(init_keep)} 个孤立 Initializer (权重/常量)")


# ============================================================
#  验证
# ============================================================

def verify_with_ort(model_path):
    import onnxruntime as ort
    opt = ort.SessionOptions()
    opt.log_severity_level = 3
    try:
        sess = ort.InferenceSession(model_path, sess_options=opt, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        outs = sess.get_outputs()
        out_summary = ", ".join(f"{o.name}{list(o.shape)}" for o in outs)
        print(f"  ✅ ORT 加载成功 | 输入: {inp.name}{list(inp.shape)} | 输出: {out_summary}")
        del sess
        return True
    except Exception as e:
        print(f"  ❌ ORT 加载失败: {str(e)}")
        return False

def check_onnx_model(model_path):
    import onnx
    try:
        m = onnx.load(model_path)
        onnx.checker.check_model(m)
        print("  ✅ ONNX checker 通过")
        return True
    except Exception as e:
        print(f"  ⚠️  ONNX checker 警告: {str(e)}")
        return False

# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="删除指定算子节点上游 Q/DQ 对 (增强稳健版 v3)")
    # 【优化点】移除了危险的默认值，改为必填
    parser.add_argument("--model", type=str, required=True, help="输入 INT8 ONNX 路径")
    parser.add_argument("--node", type=str, required=True, help="目标算子节点名, 如 Conv_250")
    parser.add_argument("--output", "--out", type=str, required=True, help="输出模型路径")
    parser.add_argument("--no-verify", action="store_true", help="跳过 ORT 验证")
    parser.add_argument("--force-shared", action="store_true", help="遇到共享 DQ 时强制连带移除(默认中断保护)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    import onnx

    print(f"[INFO] 加载模型: {args.model}")
    model = onnx.load(args.model)
    print(f"[INFO] 图节点数: {len(model.graph.node)}")

    print(f"\n[STEP 1] 查找 {args.node} 上游的所有 Q/DQ 链...")
    chains = find_all_qdq_chains_for_node(model, args.node, verbose=args.verbose)

    if not chains:
        print("\n[ABORT] 未找到任何 Q/DQ 链! 请检查:")
        print(f"  1. 节点名 '{args.node}' 是否正确 (用 Netron 确认)")
        print("  2. 该节点是否确实被量化 (可能本身已是 FP16)")
        sys.exit(1)

    print(f"\n  找到 {len(chains)} 条 Q/DQ 链:")
    for i, c in enumerate(chains):
        inter_str = f" +{len(c['intermediates'])}中间节点" if c['intermediates'] else ""
        shared_str = f" (共享{len(c['shared_consumers'])}个消费者)" if c['shared_consumers'] else ""
        print(f"    [{i}] 输入{c['input_index']}: Q={c['q_node'].name}{inter_str} → DQ={c['dq_node'].name}{shared_str}")

    print(f"\n[STEP 2] 删除 Q/DQ 并重连...")
    try:
        remove_chains_and_reconnect(model, chains, args.node, force_shared=args.force_shared, verbose=args.verbose)
    except ValueError as e:
        print(f"\n[ABORT] {str(e)}")
        sys.exit(1)

    print(f"\n[STEP 3] 保存到: {args.output}")
    onnx.save(model, args.output)
    print(f"  文件大小: {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")

    if not args.no_verify:
        print(f"\n[STEP 4] 模型验证...")
        check_onnx_model(args.output)
        ok = verify_with_ort(args.output)
        if not ok:
            print("\n[ABORT] 验证失败! 输出文件可能不可用.")
            sys.exit(1)

    print(f"\n[DONE] ✅ 成功移除 {args.node} 的 {len(chains)} 条 Q/DQ 链")
    for c in chains:
        print(f"       '{c['upstream_tensor']}' → {args.node} (输入{c['input_index']})")

if __name__ == "__main__":
    main()
