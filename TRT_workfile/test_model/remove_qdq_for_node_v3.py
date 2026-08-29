#!/usr/bin/env python3
"""
remove_qdq_for_node.py (Ultimate v4)
安全移除指定算子上游 Q/DQ 对，支持共享检测、Fan-out 保护、大模型优化。

特性:
  ✅ 对象级节点操作 (防重名/空名误删)
  ✅ 共享 DQ 安全拦截 + force-shared 模式
  ✅ Q 节点 Fan-out 保护 (防止误删被其他分支共享的 Q)
  ✅ 自动清理孤立 Initializer + ValueInfo
  ✅ 自动 Shape Inference 保证元数据一致
  ✅ Dry-Run 预览模式
  ✅ 详细的教学级注释

用法:
    # 预览模式 (不保存)
    python remove_qdq_for_node.py --model m.onnx --node Conv_250 --dry-run -v
    
    # 安全执行
    python remove_qdq_for_node.py --model m.onnx --node Conv_250 --out o.onnx
    
    # 强制处理共享 DQ
    python remove_qdq_for_node.py --model m.onnx --node Conv_250 --out o.onnx --force-shared
"""

import argparse
import sys
import os
from typing import List, Dict, Set, Tuple, Optional
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
#  拓扑分析工具
# ============================================================

def build_tensor_maps(graph):
    """
    构建 tensor→生产者 / tensor→消费者列表 双向映射。
    
    为什么需要这个？
    ONNX graph.node 是无序列表，反向追溯(DQ→Q)或查找共享消费者
    如果每次都遍历全图，复杂度为 O(N²)。建立哈希索引后降为 O(1)。
    """
    producer = {}   # tensor_name → NodeProto
    consumers = {}  # tensor_name → List[NodeProto]
    
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
    
    设计考量:
    - Passthrough 白名单平衡了"灵活性"与"安全性"
    - 包含 Slice/Concat 但限制为单输入追溯，避免跨分支污染
    - max_depth 防止循环引用
    
    Returns:
        (q_node, upstream_fp32_tensor, intermediate_nodes) or (None, None, [])
    """
    if visited is None:
        visited = set()

    # 平衡版白名单: 包含常见形状变换 + 受控的数据重组算子
    PASSTHROUGH_OPS = {
        "Reshape", "Transpose", "Squeeze", "Unsqueeze",
        "Flatten", "Expand", "Cast", "Identity", "Tile",
        "Slice", "Gather", "Pad",  # 这些通常安全，只要沿主输入追溯
    }
    # 注意: Concat/Split 被排除，因为它们的多输入语义复杂，
    # 仅取 input[0] 追溯极易出错。如需支持，应单独实现多分支追溯逻辑。

    q_input = dq_node.input[0] if dq_node.input else None
    if not q_input:
        return None, None, []

    intermediates = []
    current_tensor = q_input
    max_depth = 15

    for _ in range(max_depth):
        if current_tensor in visited:
            break
        visited.add(current_tensor)

        prod = tensor_producer.get(current_tensor)
        if prod is None:
            break  # 到达 Graph Input 或 Initializer

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
    """查找目标节点的所有 Q/DQ 链，并检测共享情况"""
    graph = model.graph
    producer, consumers_map = build_tensor_maps(graph)

    # 定位目标节点
    consumer = next((n for n in graph.node if n.name == consumer_node_name), None)
    if consumer is None:
        logger.error(f"[ERROR] 未找到节点: {consumer_node_name}")
        return []

    chains = []
    for idx, inp_tensor in enumerate(consumer.input):
        if not inp_tensor:
            continue

        dq_candidate = producer.get(inp_tensor)
        if dq_candidate is None or dq_candidate.op_type != "DequantizeLinear":
            if verbose:
                logger.debug(f"  输入[{idx}] '{inp_tensor}' ← 非 DQ, 跳过")
            continue

        q_node, upstream, intermediates = trace_dq_to_q(dq_candidate, producer)
        if q_node is None or upstream is None:
            if verbose:
                logger.debug(f"  输入[{idx}] DQ={dq_candidate.name} ← 未找到对应 Q, 跳过")
            continue

        # 检测共享消费者
        dq_out = dq_candidate.output[0] if dq_candidate.output else inp_tensor
        shared = [
            c.name for c in consumers_map.get(dq_out, [])
            if c.name != consumer_node_name
        ]

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
            logger.info(f"  输入[{idx}] ✓ Q={q_node.name} →{inter_str}→ "
                       f"DQ={dq_candidate.name} → '{inp_tensor}'{shared_str}")

    return chains


# ============================================================
#  图修改 (核心安全逻辑)
# ============================================================

def remove_chains_and_reconnect(model, chains, consumer_node_name, 
                                 force_shared=False, dry_run=False, verbose=False):
    """
    安全删除 Q/DQ 链并重连图。
    
    安全措施:
    1. 对象引用删除 (防重名)
    2. 共享 DQ 拦截
    3. Q 节点 Fan-out 保护
    4. Initializer 清理
    5. Shape Inference
    """
    graph = model.graph
    _, consumers_map = build_tensor_maps(graph)

    # === 安全检查: 共享 DQ ===
    for chain in chains:
        if chain["shared_consumers"] and not force_shared:
            shared_list = ", ".join(chain["shared_consumers"])
            raise ValueError(
                f"安全拦截: DQ '{chain['dq_node'].name}' 被其他节点共享: [{shared_list}]。\n"
                f"移除它会导致这些节点也被迫回退 FP32。\n"
                f"如确需执行，请添加 --force-shared 参数。"
            )

    # === 收集待删除节点 (使用 id 而非 name) ===
    nodes_to_remove_ids: Set[int] = set()
    reconnect_map: Dict[str, str] = {}
    
    # 统计每个 Q 节点被本次 chains 引用的次数
    q_ref_count: Dict[int, int] = {}
    for chain in chains:
        q_id = id(chain["q_node"])
        q_ref_count[q_id] = q_ref_count.get(q_id, 0) + 1

    for chain in chains:
        # DQ 和中间节点总是可以安全删除
        nodes_to_remove_ids.add(id(chain["dq_node"]))
        for mid in chain["intermediates"]:
            nodes_to_remove_ids.add(id(mid))
        
        # Q 节点 Fan-out 保护: 
        # 只有当本次所有 chain 都指向同一个 Q 时才能删除
        # 否则说明 Q 还被其他未处理的 DQ 分支使用
        q_id = id(chain["q_node"])
        q_output = chain["q_node"].output[0] if chain["q_node"].output else None
        
        if q_output:
            total_dq_consumers = len([
                c for c in consumers_map.get(q_output, [])
                if c.op_type == "DequantizeLinear"
            ])
            if q_ref_count[q_id] >= total_dq_consumers:
                nodes_to_remove_ids.add(q_id)
            else:
                logger.warning(
                    f"  ⚠️ Q节点 {chain['q_node'].name} 被 {total_dq_consumers} 个 DQ 共享，"
                    f"本次仅处理 {q_ref_count[q_id]} 个，保留 Q 节点"
                )
        
        reconnect_map[chain["dq_output"]] = chain["upstream_tensor"]

    if dry_run:
        logger.info("\n[DRY-RUN] 预览模式，以下操作不会实际执行:")
        logger.info(f"  将删除 {len(nodes_to_remove_ids)} 个节点")
        logger.info(f"  将重连 {len(reconnect_map)} 条 tensor 引用:")
        for dq_out, up in reconnect_map.items():
            logger.info(f"    '{dq_out}' → '{up}'")
        return

    # === Step 1: 全局替换引用 ===
    total_replaced = 0
    for node in graph.node:
        if id(node) in nodes_to_remove_ids:
            continue
        for i, inp in enumerate(node.input):
            if inp in reconnect_map:
                node.input[i] = reconnect_map[inp]
                total_replaced += 1

    # 替换 graph output 引用
    for out in graph.output:
        if out.name in reconnect_map:
            old_name = out.name
            out.name = reconnect_map[out.name]
            logger.warning(f"  ⚠️ 模型输出 '{old_name}' → '{out.name}', 可能影响外部接口!")

    logger.info(f"  全局替换了 {total_replaced} 处 tensor 引用")

    # === Step 2: 删除节点 (对象级过滤) ===
    keep = [n for n in graph.node if id(n) not in nodes_to_remove_ids]
    removed_count = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)
    logger.info(f"  删除了 {removed_count} 个节点对象")

    # === Step 3: 清理 ValueInfo & Initializer ===
    removed_tensors = set()
    for chain in chains:
        for t in chain["dq_node"].output: removed_tensors.add(t)
        for t in chain["dq_node"].input:
            if t != chain["upstream_tensor"]: removed_tensors.add(t)
        if id(chain["q_node"]) in nodes_to_remove_ids:
            for t in chain["q_node"].output: removed_tensors.add(t)
        for mid in chain["intermediates"]:
            for t in mid.output: removed_tensors.add(t)

    # 清理 value_info
    vi_keep = [v for v in graph.value_info if v.name not in removed_tensors]
    vi_cleaned = len(graph.value_info) - len(vi_keep)
    del graph.value_info[:]
    graph.value_info.extend(vi_keep)

    # 清理 Initializer (只清理确实不再被任何节点使用的)
    used_tensors = set()
    for node in graph.node:
        used_tensors.update(node.input)
        used_tensors.update(node.output)
    for out in graph.output:
        used_tensors.add(out.name)
    for inp in graph.input:
        used_tensors.add(inp.name)

    init_keep = [init for init in graph.initializer if init.name in used_tensors]
    init_cleaned = len(graph.initializer) - len(init_keep)
    del graph.initializer[:]
    graph.initializer.extend(init_keep)

    logger.info(f"  清理: {vi_cleaned} 个 value_info, {init_cleaned} 个 initializer")

    # === Step 4: Shape Inference ===
    try:
        import onnx.shape_inference
        inferred = onnx.shape_inference.infer_shapes(model)
        model.CopyFrom(inferred)
        logger.info("  ✅ Shape Inference 完成")
    except Exception as e:
        logger.warning(f"  ⚠️ Shape Inference 失败 (非致命): {e}")


# ============================================================
#  验证
# ============================================================

def verify_model(model_path):
    """ORT + ONNX Checker 双重验证"""
    import onnx
    try:
        m = onnx.load(model_path)
        onnx.checker.check_model(m)
        logger.info("  ✅ ONNX checker 通过")
    except Exception as e:
        logger.warning(f"  ⚠️ ONNX checker: {e}")

    try:
        import onnxruntime as ort
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        sess = ort.InferenceSession(model_path, sess_options=opt, 
                                     providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        outs = sess.get_outputs()
        out_str = ", ".join(f"{o.name}{list(o.shape)}" for o in outs)
        logger.info(f"  ✅ ORT 加载成功 | 输入: {inp.name}{list(inp.shape)} | 输出: {out_str}")
        del sess
        return True
    except Exception as e:
        logger.error(f"  ❌ ORT 加载失败: {e}")
        return False


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="安全移除指定算子 Q/DQ 对 (Ultimate v4)")
    parser.add_argument("--model", type=str, required=True, help="输入 INT8 ONNX 路径")
    parser.add_argument("--node", type=str, required=True, help="目标算子节点名")
    parser.add_argument("--output", "--out", type=str, default=None, help="输出模型路径 (dry-run 时可省略)")
    parser.add_argument("--force-shared", action="store_true", help="强制处理共享 DQ")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不保存")
    parser.add_argument("--no-verify", action="store_true", help="跳过验证")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.dry_run and not args.output:
        parser.error("--output 在非 dry-run 模式下为必填参数")

    import onnx
    
    logger.info(f"[INFO] 加载模型: {args.model}")
    model = onnx.load(args.model)
    logger.info(f"[INFO] 图节点数: {len(model.graph.node)}")

    logger.info(f"\n[STEP 1] 查找 {args.node} 上游 Q/DQ 链...")
    chains = find_all_qdq_chains_for_node(model, args.node, verbose=args.verbose)

    if not chains:
        logger.error("\n[ABORT] 未找到 Q/DQ 链! 请检查:")
        logger.error(f"  1. 节点名 '{args.node}' 是否正确 (Netron 确认)")
        logger.error("  2. 该节点是否确实被量化")
        logger.error("  3. 加 --verbose 查看搜索细节")
        sys.exit(1)

    logger.info(f"\n  找到 {len(chains)} 条链:")
    for i, c in enumerate(chains):
        inter = f" +{len(c['intermediates'])}mid" if c['intermediates'] else ""
        shared = f" (共享{len(c['shared_consumers'])})" if c['shared_consumers'] else ""
        logger.info(f"    [{i}] 输入{c['input_index']}: Q={c['q_node'].name}{inter} "
                    f"→ DQ={c['dq_node'].name}{shared}")

    logger.info(f"\n[STEP 2] 删除 Q/DQ 并重连...")
    try:
        remove_chains_and_reconnect(
            model, chains, args.node,
            force_shared=args.force_shared,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
    except ValueError as e:
        logger.error(f"\n[ABORT] {e}")
        sys.exit(1)

    if args.dry_run:
        logger.info("\n[DRY-RUN] 预览完成，未保存文件")
        return

    logger.info(f"\n[STEP 3] 保存到: {args.output}")
    onnx.save(model, args.output)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    logger.info(f"  文件大小: {size_mb:.1f} MB")

    if not args.no_verify:
        logger.info(f"\n[STEP 4] 验证...")
        ok = verify_model(args.output)
        if not ok:
            logger.error("\n[ABORT] 验证失败!")
            sys.exit(1)

    logger.info(f"\n[DONE] ✅ 成功移除 {args.node} 的 {len(chains)} 条 Q/DQ 链")


if __name__ == "__main__":
    main()