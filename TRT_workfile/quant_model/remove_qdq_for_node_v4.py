#!/usr/bin/env python3
"""
remove_qdq_for_node_v4.py (工业版)
安全移除指定算子上游的 Q/DQ 对，用于 INT8 量化后对个别敏感层做混合精度回退。

在 v1/v2/v3 基础上的关键改进:
  ✅ [语义修复] 中间节点(Reshape/Transpose等)不再"删除并绕过"，而是保留并重接:
     upstream → Q → [中间节点] → DQ → 消费者
     变为
     upstream ────────────────→ [中间节点] → 消费者
     v1/v2/v3 直接绕过中间节点，经过形状变换时下游张量形状/布局会错。
  ✅ [批量支持] --node 可传多个节点名, --node-re 支持 ModelOpt 风格的正则
     (与 quant_int8 的 nodes_to_exclude 同一套写法, 方便复用排查结论)
  ✅ Q Fan-out 保护: Q 被未处理的 DQ/分支共享时自动保留 Q, 只摘当前链
  ✅ 共享 DQ 拦截: DQ 被非目标节点共享时默认中断, --force-shared 才放行
  ✅ 中间节点共享检测: 中间节点被未处理的 DQ 引用时中断(否则会隐式改变别处精度)
  ✅ 对象级(id)节点删除, 防重名/空名误删
  ✅ 使用集统计的 value_info / Initializer 清理
  ✅ --dry-run 预览 / --list 列出所有可操作节点
  ✅ 原子保存(临时文件+rename), 拒绝覆盖输入文件
  ✅ 数值冒烟验证: 改动前后同输入对比输出 (NaN/inf 检测 + 最大误差报告)
  ✅ Shape Inference 重建元数据

用法:
    # 列出模型中所有带 QDQ 输入的候选节点
    python remove_qdq_for_node_v4.py --model m.onnx --list

    # 预览 (不写文件)
    python remove_qdq_for_node_v4.py --model m.onnx --node /model.1/conv/Conv --dry-run -v

    # 执行 (支持多个节点与正则)
    python remove_qdq_for_node_v4.py --model m.onnx --node /model.1/conv/Conv \
        --node-re "^/model.22/dfl/conv/Conv$" --out m_fixed.onnx

    # 共享 DQ 时强制放行 (接受连带回退)
    python remove_qdq_for_node_v4.py --model m.onnx --node Conv_250 --out o.onnx --force-shared

依赖: onnx, onnxruntime (仅验证用)
"""

import argparse
import logging
import os
import re
import sys
import tempfile

import numpy as np
import onnx

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class RemoveQdqError(Exception):
    """安全拦截类错误 (默认中断, --force-shared 也救不了的除外)"""


# ============================================================
#  拓扑分析
# ============================================================

def build_tensor_maps(graph):
    """tensor→生产者 / tensor→消费者列表 双向索引 (O(1) 查询, 避免全图反复遍历)"""
    producer = {}
    consumers = {}
    for node in graph.node:
        for out in node.output:
            if out:
                producer[out] = node
        for inp in node.input:
            if inp:
                consumers.setdefault(inp, []).append(node)
    return producer, consumers


# 溯源时允许经过的算子。中间节点会被保留并重接(而非删除绕过)，
# 因此经过形状/类型变换是安全的；多输入算子的辅助输入(索引/shape常量)
# 也会原样保留，不受影响。
PASSTHROUGH_OPS = {
    "Reshape", "Transpose", "Squeeze", "Unsqueeze",
    "Flatten", "Expand", "Cast", "Identity", "Tile",
    "Slice", "Gather", "Pad",
}

TRACE_MAX_DEPTH = 15


def trace_dq_to_q(dq_node, producer):
    """
    从 DQ 反向溯源到 QuantizeLinear。

    Returns:
        (q_node, upstream_fp32_tensor, intermediates)
        intermediates: [离DQ最近, ..., 离Q最近] 的中间节点列表 (可能为空)
        找不到 Q 时返回 (None, None, [])
    """
    q_input = dq_node.input[0] if dq_node.input else None
    if not q_input:
        return None, None, []

    intermediates = []
    current = q_input
    visited = set()

    for _ in range(TRACE_MAX_DEPTH):
        if current in visited:
            break
        visited.add(current)

        prod = producer.get(current)
        if prod is None:        # 图输入 / 初始化器
            break
        if prod.op_type == "QuantizeLinear":
            upstream = prod.input[0] if prod.input else None
            return prod, upstream, intermediates
        if prod.op_type in PASSTHROUGH_OPS:
            intermediates.append(prod)   # 先遇到的是离 DQ 近的
            current = prod.input[0] if prod.input else None
            if not current:
                break
        else:
            break   # 遇到普通算子: 该 DQ 不是 Q 的直接下游, 不可整链摘除

    return None, None, []


def collect_target_nodes(graph, names, patterns):
    """按精确名 + 正则收集目标节点 (去重, 保持图中顺序)"""
    targets, target_ids = [], set()
    for node in graph.node:
        hit = node.name in names
        if not hit:
            for pat in patterns:
                if re.search(pat, node.name):
                    hit = True
                    break
        if hit and id(node) not in target_ids:
            target_ids.add(id(node))
            targets.append(node)
    return targets


def find_chains_for_targets(graph, target_nodes, verbose=False):
    """
    为每个目标节点的每个输入找 Q/DQ 链。
    共享消费者检测时, 其他"目标节点"不计入共享 (它们本来也要被处理)。
    """
    producer, consumers_map = build_tensor_maps(graph)
    target_names = {n.name for n in target_nodes}

    chains = []
    for consumer in target_nodes:
        for idx, inp_tensor in enumerate(consumer.input):
            if not inp_tensor:
                continue
            dq = producer.get(inp_tensor)
            if dq is None or dq.op_type != "DequantizeLinear":
                continue

            q_node, upstream, intermediates = trace_dq_to_q(dq, producer)
            if q_node is None or upstream is None:
                if verbose:
                    logger.debug(f"  {consumer.name} 输入[{idx}]: DQ={dq.name} 上游无 Q, 跳过")
                continue

            dq_out = dq.output[0] if dq.output else inp_tensor
            shared = [c.name for c in consumers_map.get(dq_out, [])
                      if c.name not in target_names]

            chains.append({
                "consumer": consumer.name,
                "input_index": idx,
                "dq_node": dq,
                "q_node": q_node,
                "upstream_tensor": upstream,
                "dq_output": dq_out,
                "dq_data_input": dq.input[0],   # DQ 的数据输入 (中间链末端)
                "intermediates": intermediates,
                "shared_consumers": shared,
            })

            if verbose:
                mid_str = f" via {[m.name for m in intermediates]}" if intermediates else ""
                shared_str = f" (共享: {shared})" if shared else ""
                logger.info(f"  {consumer.name} 输入[{idx}] ✓ Q={q_node.name}{mid_str}"
                            f" → DQ={dq.name} → '{inp_tensor}'{shared_str}")
    return chains


# ============================================================
#  计划与校验
# ============================================================

def plan_removal(graph, chains, force_shared=False):
    """
    汇总所有 chain, 做安全校验, 产出删除/重接计划。
    返回 dict:
        delete_node_ids: set[int]
        reconnect: {dq_output: new_tensor}
        rewire_top: {top_intermediate_id: upstream_tensor}
        keep_q: {q_name: reason}
    """
    producer, consumers_map = build_tensor_maps(graph)

    processed_dq_ids = {id(c["dq_node"]) for c in chains}

    # --- 校验 1: 共享 DQ ---
    for c in chains:
        if c["shared_consumers"] and not force_shared:
            raise RemoveQdqError(
                f"安全拦截: DQ '{c['dq_node'].name}' (消费方 {c['consumer']} 输入[{c['input_index']}]) "
                f"被其他节点共享: [{', '.join(c['shared_consumers'])}]。\n"
                f"移除会连带这些节点一起回退精度。确认可接受请加 --force-shared。"
            )

    # --- 校验 2: 中间节点被未处理的 DQ 引用 (改了会隐式影响别处) ---
    for c in chains:
        for mid in c["intermediates"]:
            for out_t in mid.output:
                for cons in consumers_map.get(out_t, []):
                    if cons.op_type == "DequantizeLinear" and id(cons) not in processed_dq_ids:
                        msg = (f"安全拦截: 中间节点 '{mid.name}' 的输出还被未处理的 DQ "
                               f"'{cons.name}' 消费, 摘链会隐式改变该处精度。")
                        if not force_shared:
                            raise RemoveQdqError(msg + " (可用 --force-shared 放行)")
                        logger.warning(f"  ⚠️ --force-shared 放行: {msg}")

    # --- 汇总重接计划, 检测同一中间节点被不同 upstream 冲突重接 ---
    reconnect = {}
    rewire_top = {}
    for c in chains:
        # 无中间节点: Q 被删, 消费者直接读 Q 的 FP32 上游;
        # 有中间节点: 中间链保留, 消费者读 DQ 原来的数据输入 (中间链末端输出)
        new_src = c["dq_data_input"] if c["intermediates"] else c["upstream_tensor"]
        if c["dq_output"] in reconnect and reconnect[c["dq_output"]] != new_src:
            raise RemoveQdqError(f"DQ 输出 '{c['dq_output']}' 出现冲突的重接计划")
        reconnect[c["dq_output"]] = new_src

        if c["intermediates"]:
            top = c["intermediates"][-1]    # 离 Q 最近
            prev = rewire_top.get(id(top))
            if prev is not None and prev != c["upstream_tensor"]:
                raise RemoveQdqError(
                    f"中间节点 '{top.name}' 需要被重接到两个不同的上游 "
                    f"('{prev}' vs '{c['upstream_tensor']}'), 图结构歧义, 无法安全处理。"
                )
            rewire_top[id(top)] = c["upstream_tensor"]
            c["_top_node"] = top

    # --- Q Fan-out 判定: Q 输出的所有消费者是否都在本次处理范围内 ---
    delete_node_ids = set()
    processed_top_ids = set(rewire_top.keys())
    keep_q = {}
    for c in chains:
        q = c["q_node"]
        if id(q) in delete_node_ids:
            continue
        q_out = q.output[0] if q.output else None
        other = []
        if q_out:
            for cons in consumers_map.get(q_out, []):
                if id(cons) not in processed_dq_ids and id(cons) not in processed_top_ids:
                    other.append(cons.name)
        if other:
            keep_q[q.name] = ", ".join(sorted(set(other)))
            logger.warning(f"  ⚠️ Q '{q.name}' 还被其他节点消费 [{keep_q[q.name]}], 保留 Q, 仅摘除其 DQ 链")
        else:
            delete_node_ids.add(id(q))

    for c in chains:
        delete_node_ids.add(id(c["dq_node"]))

    return {
        "delete_node_ids": delete_node_ids,
        "reconnect": reconnect,
        "rewire_top": rewire_top,
        "keep_q": keep_q,
    }


# ============================================================
#  图修改
# ============================================================

def apply_removal(model, chains, plan):
    graph = model.graph
    delete_ids = plan["delete_node_ids"]
    reconnect = plan["reconnect"]

    # Step 1: 重接中间节点顶端输入 (原指向 Q 输出 → 指向 FP32 上游)
    for c in chains:
        top = c.get("_top_node")
        if top is not None:
            top.input[0] = c["upstream_tensor"]

    # Step 2: 全局替换 DQ 输出引用 (节点输入 + 图输出)
    total = 0
    for node in graph.node:
        if id(node) in delete_ids:
            continue
        for i, inp in enumerate(node.input):
            if inp in reconnect:
                node.input[i] = reconnect[inp]
                total += 1
    for out in graph.output:
        if out.name in reconnect:
            logger.warning(f"  ⚠️ 模型输出 '{out.name}' 被重接为 '{reconnect[out.name]}', 外部接口可能受影响!")
            out.name = reconnect[out.name]
    logger.info(f"  全局替换了 {total} 处 tensor 引用")

    # Step 3: 按对象 id 删除节点
    keep = [n for n in graph.node if id(n) not in delete_ids]
    removed = len(graph.node) - len(keep)
    del graph.node[:]
    graph.node.extend(keep)
    logger.info(f"  删除了 {removed} 个节点 (Q/DQ 对)")

    cleanup_unused(graph)


def cleanup_unused(graph):
    """按实际使用集清理孤立的 value_info 与 initializer"""
    used = set()
    for node in graph.node:
        used.update(i for i in node.input if i)
        used.update(o for o in node.output if o)
    used.update(i.name for i in graph.input)
    used.update(o.name for o in graph.output)

    vi_keep = [v for v in graph.value_info if v.name in used]
    vi_cleaned = len(graph.value_info) - len(vi_keep)
    del graph.value_info[:]
    graph.value_info.extend(vi_keep)

    init_keep = [t for t in graph.initializer if t.name in used]
    init_cleaned = len(graph.initializer) - len(init_keep)
    del graph.initializer[:]
    graph.initializer.extend(init_keep)

    logger.info(f"  清理: {vi_cleaned} 个 value_info, {init_cleaned} 个孤立 initializer")


def run_shape_inference(model):
    try:
        inferred = onnx.shape_inference.infer_shapes(model)
        model.CopyFrom(inferred)
        logger.info("  ✅ Shape Inference 完成")
    except Exception as e:
        logger.warning(f"  ⚠️ Shape Inference 失败 (非致命, 可忽略): {e}")


# ============================================================
#  验证
# ============================================================

def verify_load(model_path):
    """checker + ORT 双重加载验证"""
    try:
        onnx.checker.check_model(onnx.load(model_path))
        logger.info("  ✅ ONNX checker 通过")
    except Exception as e:
        logger.warning(f"  ⚠️ ONNX checker: {e}")

    try:
        import onnxruntime as ort
        opt = ort.SessionOptions()
        opt.log_severity_level = 3
        sess = ort.InferenceSession(model_path, sess_options=opt,
                                    providers=["CPUExecutionProvider"])
        info = ", ".join(f"{o.name}{list(o.shape)}" for o in sess.get_outputs())
        logger.info(f"  ✅ ORT 加载成功 | 输入: {sess.get_inputs()[0].name}"
                    f"{list(sess.get_inputs()[0].shape)} | 输出: {info}")
        del sess
        return True
    except Exception as e:
        logger.error(f"  ❌ ORT 加载失败: {e}")
        return False


def numeric_smoke_check(orig_path, new_path, seed=0):
    """
    数值冒烟验证: 同一随机输入分别跑原模型与改动后模型, 检查:
      - 是否能正常推理 (结构未被改坏)
      - 输出无 NaN/inf
      - 报告最大绝对误差 (摘除 QDQ 后该值应明显大于0但幅度有限;
        若出现 NaN/巨大误差说明重接出错)
    """
    import onnxruntime as ort
    opt = ort.SessionOptions()
    opt.log_severity_level = 3

    sess_a = ort.InferenceSession(orig_path, sess_options=opt, providers=["CPUExecutionProvider"])
    sess_b = ort.InferenceSession(new_path, sess_options=opt, providers=["CPUExecutionProvider"])

    inp_meta = sess_a.get_inputs()[0]
    rng = np.random.default_rng(seed)
    dims = [d if isinstance(d, int) and d > 0 else 1 for d in inp_meta.shape]
    feed = {inp_meta.name: rng.standard_normal(dims).astype(np.float32)}

    outs_a = sess_a.run(None, feed)
    outs_b = sess_b.run(None, feed)
    del sess_a, sess_b

    if len(outs_a) != len(outs_b):
        logger.error(f"  ❌ 输出数量不一致: {len(outs_a)} vs {len(outs_b)}")
        return False

    ok = True
    for i, (a, b) in enumerate(zip(outs_a, outs_b)):
        if a.shape != b.shape:
            logger.error(f"  ❌ 输出[{i}] 形状不一致: {a.shape} vs {b.shape}")
            ok = False
            continue
        if not np.isfinite(b).all():
            logger.error(f"  ❌ 输出[{i}] 含 NaN/inf!")
            ok = False
            continue
        diff = float(np.max(np.abs(a - b)))
        scale = float(np.max(np.abs(a))) or 1.0
        logger.info(f"  输出[{i}] 最大绝对误差 {diff:.6g} (相对 {diff / scale:.3g})  "
                    f"[摘除 QDQ 后存在小幅差异属正常]")
    return ok


# ============================================================
#  辅助: 列出候选节点
# ============================================================

def list_candidates(model):
    """列出所有至少有一个 QDQ 输入的节点 (潜在可回退对象)"""
    producer, consumers_map = build_tensor_maps(model.graph)
    rows = []
    for node in model.graph.node:
        qdq_inputs = [t for t in node.input
                      if t and producer.get(t) is not None
                      and producer[t].op_type == "DequantizeLinear"]
        if qdq_inputs:
            shared_total = 0
            for t in qdq_inputs:
                shared_total += max(0, len(consumers_map.get(t, [])) - 1)
            rows.append((node.name, node.op_type, len(qdq_inputs), shared_total))

    if not rows:
        logger.info("模型中没有任何带 QDQ 输入的节点 (可能未量化或已是纯 FP32)")
        return
    logger.info(f"共 {len(rows)} 个节点带 QDQ 输入 (节点名, 类型, QDQ输入数, 被共享的DQ输出数):")
    for name, op, n_qdq, n_shared in rows:
        warn = f"  ⚠️有共享DQ" if n_shared else ""
        logger.info(f"    {name:<45s} {op:<12s} qdq_in={n_qdq}{warn}")


# ============================================================
#  主流程
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="安全移除指定算子上游 Q/DQ 对 (工业版 v4)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("用法:")[-1] if __doc__ else None,
    )
    p.add_argument("--model", type=str, required=True, help="输入 INT8 QDQ ONNX 路径")
    p.add_argument("--node", type=str, action="append", default=[],
                   help="目标节点精确名, 可重复传多个")
    p.add_argument("--node-re", type=str, action="append", default=[],
                   help="目标节点名正则 (ModelOpt nodes_to_exclude 同风格), 可重复")
    p.add_argument("--out", "--output", dest="output", type=str, default=None,
                   help="输出模型路径 (默认 <输入名>_qdq_removed.onnx)")
    p.add_argument("--list", action="store_true", help="列出所有带 QDQ 输入的候选节点后退出")
    p.add_argument("--dry-run", action="store_true", help="仅预览将做的修改, 不写文件")
    p.add_argument("--force-shared", action="store_true", help="共享 DQ/中间节点时强制放行")
    p.add_argument("--no-verify", action="store_true", help="跳过加载与数值验证")
    p.add_argument("--no-numeric", action="store_true", help="跳过数值冒烟对比 (保留加载验证)")
    p.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    if not args.node and not args.node_re and not args.list:
        p.error("必须通过 --node / --node-re 指定目标节点, 或使用 --list")
    return args


def main():
    args = parse_args()
    if not args.list:
        if not args.output:
            stem, ext = os.path.splitext(args.model)
            args.output = f"{stem}_qdq_removed{ext}"
        if os.path.abspath(args.output) == os.path.abspath(args.model):
            logger.error("[ABORT] 输出路径不能与输入相同 (拒绝覆盖原始量化模型)")
            sys.exit(1)

    logger.info(f"[INFO] 加载模型: {args.model}")
    model = onnx.load(args.model)
    logger.info(f"[INFO] 图节点数: {len(model.graph.node)}")

    if args.list:
        list_candidates(model)
        return

    # STEP 1: 定位目标节点
    target_nodes = collect_target_nodes(model.graph, set(args.node), args.node_re)
    if not target_nodes:
        logger.error(f"[ABORT] 未匹配到任何目标节点! --node={args.node} --node-re={args.node_re}\n"
                     f"        可先运行 --list 查看候选节点, 或用 Netron 核对节点名。")
        sys.exit(1)
    logger.info(f"[STEP 1] 匹配到 {len(target_nodes)} 个目标节点: "
                f"{[n.name for n in target_nodes]}")

    # STEP 2: 收集 Q/DQ 链
    logger.info("[STEP 2] 收集目标输入上的 Q/DQ 链...")
    chains = find_chains_for_targets(model.graph, target_nodes, verbose=args.verbose)
    if not chains:
        logger.error("[ABORT] 目标节点上没有任何可摘除的 Q/DQ 链 "
                     "(节点可能本就未被量化, 或上游结构不匹配)。")
        sys.exit(1)

    logger.info(f"  找到 {len(chains)} 条 Q/DQ 链:")
    for i, c in enumerate(chains):
        mid = f" +{len(c['intermediates'])}中间节点" if c["intermediates"] else ""
        shared = f" (共享{len(c['shared_consumers'])})" if c["shared_consumers"] else ""
        logger.info(f"    [{i}] {c['consumer']} 输入[{c['input_index']}]: "
                    f"Q={c['q_node'].name}{mid} → DQ={c['dq_node'].name}{shared}")

    # STEP 3: 计划与安全校验
    logger.info("[STEP 3] 生成修改计划并做安全校验...")
    try:
        plan = plan_removal(model.graph, chains, force_shared=args.force_shared)
    except RemoveQdqError as e:
        logger.error(f"\n[ABORT] {e}")
        sys.exit(1)

    logger.info(f"  计划: 删除 {len(plan['delete_node_ids'])} 个节点, "
                f"重接 {len(plan['reconnect'])} 条引用"
                + (f", 保留 Q: {list(plan['keep_q'])}" if plan["keep_q"] else ""))
    if args.dry_run:
        for dq_out, new_t in plan["reconnect"].items():
            logger.info(f"    重接: '{dq_out}' → '{new_t}'")
        logger.info("[DRY-RUN] 预览完成, 未修改任何文件")
        return

    # STEP 4: 执行图修改
    logger.info("[STEP 4] 执行删除与重接...")
    apply_removal(model, chains, plan)
    run_shape_inference(model)

    # STEP 5: 原子保存 (临时文件 + rename, 避免中途失败留下半截文件)
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".onnx", dir=out_dir)
    os.close(fd)
    try:
        onnx.save(model, tmp_path)
        os.replace(tmp_path, args.output)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    logger.info(f"[STEP 5] 已保存: {args.output} "
                f"({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")

    # STEP 6: 验证
    if not args.no_verify:
        logger.info("[STEP 6] 验证...")
        if not verify_load(args.output):
            logger.error("[ABORT] 加载验证失败! 输出文件不可用, 请勿使用。")
            sys.exit(1)
        if not args.no_numeric:
            try:
                if not numeric_smoke_check(args.model, args.output):
                    logger.error("[ABORT] 数值验证失败! 输出文件不可用, 请勿使用。")
                    sys.exit(1)
            except Exception as e:
                logger.error(f"[ABORT] 数值验证执行出错: {e}")
                sys.exit(1)

    logger.info(f"[DONE] ✅ 已移除 {len(chains)} 条 Q/DQ 链, 涉及节点: "
                f"{sorted({c['consumer'] for c in chains})}")
    logger.info("       注意: 修改的是 ONNX, 下游 TensorRT engine 需要重新构建。")


if __name__ == "__main__":
    main()
