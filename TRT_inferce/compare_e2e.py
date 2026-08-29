"""
compare_e2e.py — 方案⑤（端到端 engine）的 ONNX 图手术正确性对比。

目的：验证把"预处理 + 后处理(NMS)"烧进模型后，端到端模型与原链路输出一致。
两者都用 onnxruntime(CPU) 跑，排除 engine/GPU 因素，只隔离"图手术"本身的正确性。

链路 A（端到端模型 best_PCB_e2e.onnx）:
    原图 uint8 -> [模型内: letterbox+归一化+decode+NMS] -> detections (max_dets,6)

链路 B（参考：原始模型 + CPU 前后处理）:
    原图 -> preprocess_cpu -> best_PCB.onnx -> postprocess_cpu -> 检测框

对比口径: 端到端输出的 padding 行(conf=0)被过滤后，与参考框逐框比较坐标+类别。

用法:
    python compare_e2e.py [image1 image2 ...]
    缺省用 test_images/ 下所有图片。
"""

import sys
import glob
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import preprocess_cpu, postprocess_cpu, letterbox_params

SRC_ONNX = "/root/my_FILE/models/best_PCB.onnx"
E2E_ONNX = "/root/my_FILE/models/best_PCB_e2e.onnx"


def canon(dets):
    """归一化（四舍五入 + 排序）后比较，消除浮点/顺序差异。"""
    return sorted([(round(float(x1), 2), round(float(y1), 2), round(float(x2), 2),
                    round(float(y2), 2), int(cls))
                   for x1, y1, x2, y2, _conf, cls in dets])


def main():
    for p in (SRC_ONNX, E2E_ONNX):
        if not Path(p).exists():
            print(f"[ERR] 未找到 {p}，先运行 e2e/graphsurgeon_e2e.py 生成端到端模型。")
            sys.exit(1)

    ref = ort.InferenceSession(SRC_ONNX, providers=["CPUExecutionProvider"])
    e2e = ort.InferenceSession(E2E_ONNX, providers=["CPUExecutionProvider"])
    print("e2e 输入 :", [(i.name, i.shape) for i in e2e.get_inputs()])
    print("e2e 输出 :", [(o.name, o.shape) for o in e2e.get_outputs()])

    args = sys.argv[1:]
    if args:
        paths = [a for a in args if Path(a).exists()]
    else:
        d = "/root/my_FILE/my_FILE/test_images"
        paths = sorted(glob.glob(f"{d}/*.[jp][pn]g")) if Path(d).exists() else []
    if not paths:
        print("[ERR] 没有可用测试图片。")
        sys.exit(1)

    print(f"\n对比 {len(paths)} 张图\n")
    total = 0
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] 读图失败: {path}")
            continue

        # 链路 B：参考（原模型 + CPU 前后处理）
        blob, scale, pad_x, pad_y = preprocess_cpu(img)
        out0 = ref.run(None, {"images": blob})[0]
        dets_cpu = postprocess_cpu(out0, scale, pad_x, pad_y)

        # 链路 A：端到端模型（原图直出，过滤 padding 行）
        dets_e2e = e2e.run(None, {"image_raw": img[None, ...]})[0]
        dets_e2e = dets_e2e[dets_e2e[:, 4] > 0]

        cc, ce = canon(dets_cpu), canon(dets_e2e)
        match = (len(cc) == len(ce)
                 and all(a == b for a, b in zip(cc, ce)))
        print(f"[{Path(path).name}] 参考 {len(cc)} 框 | e2e {len(ce)} 框 | "
              f"一致={'是' if match else '否'}")
        if not match:
            print("    参考:", cc)
            print("    e2e :", ce)
        total += match

    print(f"\n{'=' * 60}\n正确性: {total}/{len(paths)} 张图一致\n{'=' * 60}")


if __name__ == "__main__":
    main()
