"""
crop_infer_pipeline.py — 两级推理演示：检测 → 最大框裁剪 → 裁剪图再推理。

流程:
    1. yolov8_int8_e2e_dyn.engine 端到端推理整图，得到已 NMS 的检测框
    2. 按框面积选最大的一个，从原图裁出
    3. 裁剪图再送入模型推理（演示用同一个引擎；实际可换成任意其它模型/引擎）
    4. 输出两阶段结果，并保存: 原图+框 / 裁剪图 / 裁剪图+框

注意: 裁剪框可能小于引擎 profile 的 H/W 下限（当前构建配置 min=128），
     小于下限时自动用 letterbox 灰(114) pad 到下限，否则 execute 会失败。

用法:
    python crop_infer_pipeline.py                       # 默认测试图
    python crop_infer_pipeline.py --image xxx.jpg --out result_dir
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "e2e_conf" / "e2e_dyn"))       # E2EDynEngine 所在
from infer_e2e_dyn import E2EDynEngine, draw                  # noqa: E402

DEFAULT_ENGINE = "/root/my_FILE/models/yolov8_int8_e2e_dyn.engine"
DEFAULT_IMAGE = "/root/my_FILE/my_trt_FILE/my_FILE/test_images/00041200.jpg"
CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
PAD_VALUE = 114                                               # 与引擎内 letterbox 填充一致


def detect(engine, img, conf=0.45, iou=0.65):
    """整图推理，返回 (N,6) 检测数组 [x1,y1,x2,y2,conf,cls]（原图坐标）。"""
    out = engine.infer(img[None], iou=iou, conf=conf)
    return out[0][out[0][:, 4] > 0]


def largest_box(dets):
    """按面积选最大的框，返回该行 (6,)；无检测返回 None。"""
    if len(dets) == 0:
        return None
    areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    return dets[int(np.argmax(areas))]


def crop_with_pad(img, box, min_hw):
    """按框裁剪（坐标 clamp 到图内）；若 H/W 小于引擎 profile 下限则 pad 到下限。
    返回 (裁剪图, 实际使用的框)。"""
    h_img, w_img = img.shape[:2]
    x1 = int(max(0, np.floor(box[0])))
    y1 = int(max(0, np.floor(box[1])))
    x2 = int(min(w_img, np.ceil(box[2])))
    y2 = int(min(h_img, np.ceil(box[3])))
    crop = img[y1:y2, x1:x2].copy()
    if crop.size == 0:
        raise RuntimeError(f"裁剪区域为空: box={box}")

    ch, cw = crop.shape[:2]
    pad_h = max(0, min_hw - ch)
    pad_w = max(0, min_hw - cw)
    if pad_h or pad_w:
        # 四周均匀 pad（等效于把目标放在画布中央，周围是 letterbox 灰）
        top, bottom = pad_h // 2, pad_h - pad_h // 2
        left, right = pad_w // 2, pad_w - pad_w // 2
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3)
    return crop, (x1, y1, x2, y2)


def two_stage_infer(engine, img, conf=0.45, iou=0.65, second_conf=0.25):
    """端到端两级推理。
    返回 dict:
        dets        一级全部检测 (N,6)
        largest     最大框 (6,) 或 None
        crop        裁剪(含 pad)后的图
        crop_box    裁剪区域在原图的 (x1,y1,x2,y2)
        dets2       二级（对裁剪图）检测结果 (M,6)，坐标相对裁剪图
    """
    dets = detect(engine, img, conf=conf, iou=iou)
    box = largest_box(dets)
    if box is None:
        return {"dets": dets, "largest": None, "crop": None, "crop_box": None, "dets2": None}

    # 引擎 profile 的 H/W 下限（从引擎读，不硬编码；换构建配置自动适配）
    # get_tensor_profile_shape 返回 (min_dims, opt_dims, max_dims)，[0] 取 min，再 [1]/[2] 取 H/W
    prof_min = engine.engine.get_tensor_profile_shape("image_raw", 0)[0]
    min_hw = max(int(prof_min[1]), int(prof_min[2]))
    crop, crop_box = crop_with_pad(img, box, min_hw)

    # ★ 第二级推理：演示用同一个引擎；实际场景把这里换成另一个模型/引擎即可
    dets2 = detect(engine, crop, conf=second_conf, iou=iou)
    return {"dets": dets, "largest": box, "crop": crop, "crop_box": crop_box, "dets2": dets2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default=DEFAULT_ENGINE)
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--conf", type=float, default=0.45, help="一级(整图)置信度阈值")
    ap.add_argument("--second-conf", type=float, default=0.25, help="二级(裁剪图)置信度阈值")
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--out", default="/tmp/crop_infer_result", help="结果输出目录")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise RuntimeError(f"无法读取图片: {args.image}")
    engine = E2EDynEngine(args.engine)

    # ---- 一级：整图检测 ----
    res = two_stage_infer(engine, img, conf=args.conf, iou=args.iou,
                          second_conf=args.second_conf)
    print(f"== 一级(整图): {len(res['dets'])} 框 ==")
    for x1, y1, x2, y2, c, k in res["dets"]:
        print(f"   {CLASS_NAMES[int(k)]:<10} conf={c:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

    if res["largest"] is None:
        print("无检测，流程结束")
        return

    lb = res["largest"]
    area = (lb[2] - lb[0]) * (lb[3] - lb[1])
    print(f"== 最大框: {CLASS_NAMES[int(lb[5])]} conf={lb[4]:.3f} "
          f"box=({lb[0]:.0f},{lb[1]:.0f},{lb[2]:.0f},{lb[3]:.0f}) 面积={area:.0f}px² ==")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 二级：裁剪图再推理 ----
    print(f"== 二级(裁剪图 {res['crop'].shape[1]}x{res['crop'].shape[0]}): "
          f"{len(res['dets2'])} 框 ==")
    for x1, y1, x2, y2, c, k in res["dets2"]:
        print(f"   {CLASS_NAMES[int(k)]:<10} conf={c:.3f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
              f"(相对裁剪图)")

    # ---- 保存可视化 ----
    vis = img.copy()
    draw(vis, res["dets"])
    x1, y1, x2, y2 = res["crop_box"]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 3)          # 红框标出最大框
    cv2.imwrite(str(out / "stage1_all.jpg"), vis)
    cv2.imwrite(str(out / "crop.jpg"), res["crop"])
    crop_vis = res["crop"].copy()
    draw(crop_vis, res["dets2"])
    cv2.imwrite(str(out / "stage2_crop.jpg"), crop_vis)
    print(f"[done] 结果目录: {out}  (stage1_all.jpg / crop.jpg / stage2_crop.jpg)")


if __name__ == "__main__":
    main()
