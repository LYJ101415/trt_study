"""
inspect_engine_layers.py — 用 engine inspector 查看 DETAILED 构建的引擎的层结构与精度线索。

注意: TRT 10 的 inspector JSON 没有 "Precision" 字段；
      真实计算精度从两处线索读出:
        1) TacticName 含 i8i8_i8i32 → 该层以 INT8(IMMA) 执行
        2) Name 中融合了 QuantizeLinear/DequantizeLinear → 被量化, 走 INT8
      其余算子(预/后处理等)由 builder 在已启用精度范围内(FP16/FP32)逐层择优。

用法:
    python inspect_engine_layers.py [engine路径] [--json out.json]
"""

import json
import sys
from collections import Counter

import tensorrt as trt


def main(engine_path='/root/my_FILE/models/yolov8_int8_e2e.engine', json_out=None):
    logger = trt.Logger(trt.Logger.ERROR)
    eng = trt.Runtime(logger).deserialize_cuda_engine(open(engine_path, 'rb').read())
    insp = eng.create_engine_inspector()
    data = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))['Layers']

    tactic = Counter()
    qdq_fused = []
    layers_all = []
    for d in data:
        if not isinstance(d, dict):
            continue
        t = d.get('TacticName', '')
        if 'i8i8' in t or 'IMMA' in t:
            tactic['INT8'] += 1
        elif t:
            tactic['非INT8(有tactic)'] += 1
        else:
            tactic['无tactic(shape/常数等辅助层)'] += 1
        if 'QuantizeLinear' in str(d.get('Name', '')) and d.get('LayerType', '').startswith('Cask'):
            qdq_fused.append(d['Name'])
        layers_all.append({k: d[k] for k in ('Name', 'LayerType', 'TacticName') if k in d})

    print(f"总层数: {len(layers_all)}")
    print("\n== 按 Tactic 判断的计算精度分布 ==")
    for k, v in tactic.most_common():
        print(f"  {k}: {v} 层")
    print(f"\n== 融合了 QuantizeLinear 的计算层(= INT8 执行): {len(qdq_fused)} 个 ==")
    for s in qdq_fused[:5]:
        print(" ", s[:90])

    if json_out:
        with open(json_out, 'w') as f:
            json.dump(layers_all, f, ensure_ascii=False, indent=1)
        print(f"\n完整层列表 -> {json_out}")


if __name__ == '__main__':
    args = sys.argv[1:]
    path = args[0] if args and not args[0].startswith('--') else \
        '/root/my_FILE/models/yolov8_int8_e2e.engine'
    out = None
    if '--json' in args:
        out = args[args.index('--json') + 1]
    main(path, out)
