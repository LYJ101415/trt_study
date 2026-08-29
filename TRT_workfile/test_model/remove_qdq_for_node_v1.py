#!/usr/bin/env python3
"""
remove_qdq_for_node.py ：
删除指定算子节点上游的 Q/DQ 对, 支持:
  ✅ 标准 Q→DQ→Consumer
  ✅ 共享 DQ (一个 DQ 输出被多个算子消费)
  ✅ Q/DQ 间插入 Reshape/Transpose/Squeeze 等中间节点
  ✅ 目标算子多输入各自有独立 Q/DQ 链
用法:
    python remove_qdq_for_node.py \
        --model yolov8_int8.onnx \
        --node Conv_250 \
        --output yolov8_int8_fixed.onnx
    # 跳过 ORT 验证 (大模型节省时间)
    python remove_qdq_for_node.py --model m.onnx --node Conv_250 --out o.onnx --no-verify
    # 详细调试日志
    python remove_qdq_for_node.py --model m.onnx --node Conv_250 --out o.onnx --verbose
依赖: pip install onnx onnxruntime
"""

import argparse
import sys
import os
from collections import deque

# ============================================================
#  拓扑分析工具
# ============================================================

def build_tensor_maps(graph):
    """构建 tensor→生产者 / tensor→消费者列表 双向映射,作用：ONNX 图本质上是一个 DAG（有向无环图），
    但原生 graph.node 只是一个无序列表。要高效地"向上追溯"或"向下查找"，必须先建立 O(1) 的哈希索引。
    运行逻辑：遍历所有节点,每个节点的 output → 记入 producer（一个 tensor 只有一个生产者）
    每个节点的 input → 追加到 consumers（一个 tensor 可被多个节点消费）
    为什么需要 consumers？ 后面处理"共享 DQ"时必须知道一个 DQ 的输出还被谁用了。"""
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

def trace_dq_to_q(dq_node, tensor_producer, visited=None): # 反向追溯 Q 节点
    """
    从 DQ 节点反向追溯到 QuantizeLinear。在实际 ONNX 图中，Q 和 DQ 之间经常插入形状变换节点：
    允许中间经过 Reshape/Transpose/Squeeze/Unsqueeze/Cast/Identity 等纯形状/类型变换节点.  
    返回 (q_node, upstream_tensor, intermediate_nodes) 或 (None, None, [])    
    intermediate_nodes: DQ 和 Q 之间的所有中间节点列表 (可能为空)
    """
    if visited is None:
        visited = set()

    # 允许的中间节点类型 (不改变数值, 只改变形状/布局/类型)
    PASSTHROUGH_OPS = {
        "Reshape", "Transpose", "Squeeze", "Unsqueeze",
        "Flatten", "Expand", "Gather", "Slice",
        "Cast", "Identity", "Shape", "Concat",
        "Split", "Pad", "Tile",
    }

    q_input = dq_node.input[0] if dq_node.input else None
    if not q_input:
        return None, None, []

    intermediates = []
    current_tensor = q_input
    max_depth = 10  # 防止无限循环

    for _ in range(max_depth):
        if current_tensor in visited:
            break
        visited.add(current_tensor)

        prod = tensor_producer.get(current_tensor)
        if prod is None:
            # 可能是图输入或初始化器, 不是 Q 节点
            break

        if prod.op_type == "QuantizeLinear":
            upstream = prod.input[0] if prod.input else None
            return prod, upstream, intermediates

        if prod.op_type in PASSTHROUGH_OPS:
            intermediates.append(prod)
            # 继续向上追溯 (取第一个输入)
            current_tensor = prod.input[0] if prod.input else None
            if not current_tensor:
                break
        else:
            # 遇到不支持的算子, 停止追溯
            break

    return None, None, []


def find_all_qdq_chains_for_node(model, consumer_node_name, verbose=False): #  查找目标节点的所有 Q/DQ 链
    """
    查找目标算子所有输入端的 Q/DQ 链.   
    返回 list of dict:
      [{
        "input_index": int,         # 消费者输入索引
        "input_tensor": str,        # 消费者原始输入 tensor 名
        "dq_node": NodeProto,
        "q_node": NodeProto,
        "upstream_tensor": str,     # Q 节点的原始 FP32 输入
        "dq_output": str,           # DQ 的输出 tensor (即消费者原始输入)
        "intermediates": [NodeProto, ...],  # Q/DQ 之间的中间节点
        "shared_consumers": [str, ...],     # 共享同一 DQ 输出的其他消费者名
      }, ...]
    """
    graph = model.graph
    producer, consumers_map = build_tensor_maps(graph)

    # Step 1: 在 graph.node 中找到名为 consumer_node_name 的节点
    consumer = None
    for node in graph.node:
        if node.name == consumer_node_name:
            consumer = node
            break

    if consumer is None:
        print(f"[ERROR] 未找到节点: {consumer_node_name}")
        return []

    # Step 2: 遍历该节点的每一个 input（多输入算子如 Concat/Add 有多个）
    chains = []
    for idx, inp_tensor in enumerate(consumer.input):
        if not inp_tensor:
            continue

        # Step 3: 检查该输入是否直接来自 DQ
        dq_candidate = producer.get(inp_tensor)
        if dq_candidate is None or dq_candidate.op_type != "DequantizeLinear":
            if verbose:
                msg = f"  输入[{idx}] '{inp_tensor}' ← 非 DQ, 跳过"
                print(msg)
            continue

        # Step 4: 调用 trace_dq_to_q() 反向找 QuantizeLinear
        q_node, upstream, intermediates = trace_dq_to_q(
            dq_candidate, producer
        )

        if q_node is None or upstream is None:
            if verbose:
                msg = f"  输入[{idx}] DQ={dq_candidate.name} ← 未找到对应 Q, 跳过"
                print(msg)
            continue

        # Step 5: 检查共享消费者（consumers_map[dq_output]），记录除目标节点外还有哪些节点也消费了这个 DQ 输出
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
        # Step 6: 打包为 chain dict，加入结果列表
        chains.append(chain)

        if verbose:
            inter_names = [n.name for n in intermediates]
            inter_str = f" via {inter_names}" if inter_names else ""
            shared_str = f" (共享: {shared})" if shared else ""
            msg = (f"  输入[{idx}] ✓ Q={q_node.name} →{inter_str}→ "
                   f"DQ={dq_candidate.name} → '{inp_tensor}'{shared_str}")
            print(msg)

    return chains


# ============================================================
#  图修改
# ============================================================

def remove_chains_and_reconnect(model, chains, consumer_node_name, verbose=False): # 删除Q/DQ并重连
    """
    根据找到的 Q/DQ 链执行删除和重连.原地修改 model.   
    处理要点:
      - 共享 DQ: 替换所有消费者引用, 不仅限于目标算子
      - 中间节点: 一并删除 (它们只为 Q/DQ 服务)
      - 多链去重: 同一个 Q/DQ 可能被多条链引用
    """
    graph = model.graph

    # 收集所有需要删除的节点名 (去重)
    nodes_to_remove = set()
    reconnect_map = {}  

    for chain in chains:
        nodes_to_remove.add(chain["q_node"].name)
        nodes_to_remove.add(chain["dq_node"].name)
        for mid in chain["intermediates"]:
            nodes_to_remove.add(mid.name)
        reconnect_map[chain["dq_output"]] = chain["upstream_tensor"]
    # 关键点：不仅替换目标节点的引用，所有消费该 DQ 输出的节点都会被替换。这就是处理"共享 DQ"的核心机制。
    # 这意味着如果你只想让 Conv_250 回退 FP16，但它和一个 Add 共享同一个 DQ，那 Add 也会被迫回退。脚本会打印警告提醒你。

    # Step 1: 全局替换所有下游引用
    total_replaced = 0
    for node in graph.node:
        if node.name in nodes_to_remove:
            continue
        for i, inp in enumerate(node.input):
            if inp in reconnect_map:
                node.input[i] = reconnect_map[inp]
                total_replaced += 1

    # 替换 graph output 引用
    for out in graph.output:
        if out.name in reconnect_map:
            out.name = reconnect_map[out.name]

    repl_msg = f"  全局替换了 {total_replaced} 处 tensor 引用"
    print(repl_msg)
    for dq_out, up in reconnect_map.items():
        detail = f"    '{dq_out}' → '{up}'"
        print(detail)

    # Step 2: 删除节点
    removed_names = sorted(nodes_to_remove)
    keep = [n for n in graph.node if n.name not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(keep)

    rm_msg = f"  删除了 {len(removed_names)} 个节点:"
    print(rm_msg)
    for name in removed_names:
        print(f"    • {name}")

    # Step 3: 清理 value_info
    removed_tensors = set()
    for chain in chains:
        for t in chain["q_node"].output:
            removed_tensors.add(t)
        for t in chain["dq_node"].output:
            removed_tensors.add(t)
        for t in chain["dq_node"].input:
            if t != chain["upstream_tensor"]:
                removed_tensors.add(t)
        for t in chain["q_node"].input:
            # Q 的 scale/zp 等参数不删 (可能被其他 Q 共享)
            pass
        for mid in chain["intermediates"]:
            for t in mid.output:
                removed_tensors.add(t)
            for t in mid.input:
                # 中间节点的输入如果是上游原始 tensor 则保留
                if t != chain["upstream_tensor"]:
                    removed_tensors.add(t)

    vi_before = len(graph.value_info)
    vi_keep = [v for v in graph.value_info if v.name not in removed_tensors]
    del graph.value_info[:]
    graph.value_info.extend(vi_keep)
    vi_cleaned = vi_before - len(vi_keep)

    clean_msg = f"  清理了 {vi_cleaned} 个孤立 value_info 条目"
    print(clean_msg)

    # Step 4: 报告共享情况
    for chain in chains:
        if chain["shared_consumers"]:
            shared_list = ", ".join(chain["shared_consumers"])
            warn = (f"  ⚠️  DQ '{chain['dq_node'].name}' 被其他节点共享: "
                    f"{shared_list}")
            print(warn)
            note = "     这些节点的量化也已一并移除!"
            print(note)

# ============================================================
#  验证
# ============================================================

def verify_with_ort(model_path):
    """ORT CPU 加载验证"""
    import onnxruntime as ort
    opt = ort.SessionOptions()
    opt.log_severity_level = 3
    try:
        sess = ort.InferenceSession(
            model_path, sess_options=opt,
            providers=["CPUExecutionProvider"]
        )
        inp = sess.get_inputs()[0]
        outs = sess.get_outputs()
        out_summary = ", ".join(
            f"{o.name}{list(o.shape)}" for o in outs
        )
        msg = (f"  ✅ ORT 加载成功 | 输入: {inp.name}{list(inp.shape)} | "
               f"输出: {out_summary}")
        print(msg)
        del sess
        return True
    except Exception as e:
        err = str(e)
        print(f"  ❌ ORT 加载失败: {err}")
        return False


def check_onnx_model(model_path):
    """onnx.checker 静态校验"""
    import onnx
    try:
        m = onnx.load(model_path)
        onnx.checker.check_model(m)
        print("  ✅ ONNX checker 通过")
        return True
    except Exception as e:
        err = str(e)
        print(f"  ⚠️  ONNX checker 警告: {err}")
        return False

# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="删除指定算子节点上游 Q/DQ 对 (增强版 v2)"
    )
    parser.add_argument("--model", type=str, default="/root/my_FILE/yolov8_int8_fixed.onnx", help="输入 INT8 ONNX 路径")
    parser.add_argument("--node", type=str, default="Conv_6", help="目标算子节点名, 如 Conv_250")
    parser.add_argument("--output", "--out", type=str, default="yolov8_int8_fixed_1.onnx" , help="输出模型路径") # required=True
    parser.add_argument("--no-verify", action="store_true", help="跳过 ORT 验证")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    import onnx

    # Load
    load_msg = f"[INFO] 加载模型: {args.model}"
    print(load_msg)
    model = onnx.load(args.model)
    node_count = len(model.graph.node)
    count_msg = f"[INFO] 图节点数: {node_count}"
    print(count_msg)

    # Find chains
    find_msg = f"\n[STEP 1] 查找 {args.node} 上游的所有 Q/DQ 链..."
    print(find_msg)
    chains = find_all_qdq_chains_for_node(
        model, args.node, verbose=args.verbose
    )

    if not chains:
        print("\n[ABORT] 未找到任何 Q/DQ 链! 请检查:")
        print(f"  1. 节点名 '{args.node}' 是否正确 (用 Netron 确认)")
        print("  2. 该节点是否确实被量化 (可能本身已是 FP16)")
        print("  3. 加 --verbose 查看详细搜索过程")
        sys.exit(1)

    found_msg = f"\n  找到 {len(chains)} 条 Q/DQ 链:"
    print(found_msg)
    for i, c in enumerate(chains):
        inter_count = len(c["intermediates"])
        shared_count = len(c["shared_consumers"])
        inter_str = f" +{inter_count}中间节点" if inter_count else ""
        shared_str = f" (共享{shared_count}个消费者)" if shared_count else ""
        chain_info = (f"    [{i}] 输入{c['input_index']}: "
                      f"Q={c['q_node'].name}{inter_str} → "
                      f"DQ={c['dq_node'].name}{shared_str}")
        print(chain_info)

    # Remove & reconnect
    print(f"\n[STEP 2] 删除 Q/DQ 并重连...")
    remove_chains_and_reconnect(model, chains, args.node, verbose=args.verbose)

    # Save
    save_msg = f"\n[STEP 3] 保存到: {args.output}"
    print(save_msg)
    onnx.save(model, args.output)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    size_msg = f"  文件大小: {size_mb:.1f} MB"
    print(size_msg)

    # Verify
    if not args.no_verify:
        print(f"\n[STEP 4] 模型验证...")
        check_onnx_model(args.output)
        ok = verify_with_ort(args.output)
        if not ok:
            print("\n[ABORT] 验证失败! 输出文件可能不可用.")
            sys.exit(1)

    done_msg = f"\n[DONE] ✅ 成功移除 {args.node} 的 {len(chains)} 条 Q/DQ 链"
    print(done_msg)
    for c in chains:
        summary = (f"       '{c['upstream_tensor']}' → {args.node}"
                   f" (输入{c['input_index']})")
        print(summary)


if __name__ == "__main__":
    main()

"""
python D:\CV_FILE\my_FILE\TRT_opt\layer_ablation_eval.py \
--int8-model /root/my_FILE/yolov8_int8_1.onnx \
--images-dir /root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val \
--ann-json /root/my_FILE/my_FILE/CV_yolov8/datasets/Data_DeepPCB_YOLO/images/val_coco.json \
--device-id 0 \
--img-size 640

python D:\CV_FILE\my_FILE\TRT_opt\layer_ablation_eval.py \
--model /root/my_FILE/fixed_1.onnx \
--node Conv_3 \
--output fixed.onnx \
--verbose

使用注意事项:
风险	              说明	                      缓解措施
共享 DQ 误伤	其他节点被动回退精度	      看日志中的警告，必要时手动拆分 DQ
中间节点误删	透传节点可能被其他路径复用	   脚本假设中间节点仅服务于该 Q/DQ 链，复杂图需人工确认
精度变化	 移除 Q/DQ 后该层变高精度，但上下游仍是 INT8	   TRT 会自动插入隐式转换，可能有额外开销
Engine 需重建	修改的是 ONNX，不是 .engine	                 改完ONNX 后必须重新 trtexec 构建 Engine
    
这个脚本是 INT8 混合精度部署工作流中不可或缺的一环，配合 Netron 可视化使用效果最佳：先在 Netron 中定位问题节点名，
再用此脚本精准切除 Q/DQ，最后重新构建 Engine 验证精度。
"""