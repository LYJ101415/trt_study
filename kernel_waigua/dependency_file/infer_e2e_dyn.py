"""
infer_e2e_dyn.py — 全动态(batch,H,W) e2e INT8 引擎的实际场景推理。

配套模型: graphsurgeon_e2e_conf_iou_dyn.py 产出的动态 ONNX → build_engine_e2e.py 构建引擎
  输入  image_raw (B,H,W,3) uint8 BGR（B/H/W 全动态）+ 运行时阈值
  输出  detections (B,300,6) f32 = [x1,y1,x2,y2,conf,cls]，原图坐标、已 NMS、空行 conf=0

支持三种输入源（--source 自动判别）:
    单张图片 : python infer_e2e_dyn.py --source xxx.jpg
    文件夹   : python infer_e2e_dyn.py --source images_dir/ --batch 8
               （同尺寸图自动归组打包 batch；不同尺寸的图分批处理）
    视频     : python infer_e2e_dyn.py --source video.mp4 --batch 8
               （连续帧天然同尺寸，攒满一个 batch 推一次；末尾不足一 batch 也能跑）

公共参数: --conf 0.45 --iou 0.65 --out 输出路径 --save-txt 保存检测结果
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))   # TRT_inferce/（common.py 所在）
from common import cuda_malloc, device_addr, cuda_memcpy_h2d, cuda_memcpy_d2h  # noqa: E402

MAX_BATCH = 8
MAX_HW = 1920          # 与构建引擎时的 profile 一致，超出前会自动缩放并回映射坐标
MAX_DETS = 300

CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VID_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

# 每个 batch 框的颜色（BGR）
COLORS = [(60, 60, 220), (60, 180, 220), (220, 60, 60),
          (60, 220, 60), (220, 60, 200), (0, 200, 255)]


class E2EDynEngine:
    """全动态 e2e 引擎封装。显存按 profile 上限一次分配复用，换 batch/尺寸不重新 malloc。"""

    def __init__(self, engine_path, logger=None):
        self.logger = logger or trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()

        self._d_raw = cuda_malloc(MAX_BATCH * MAX_HW * MAX_HW * 3)
        self._d_iou = cuda_malloc(4)
        self._d_score = cuda_malloc(4)
        self._d_out = cuda_malloc(MAX_BATCH * MAX_DETS * 6 * 4)

        self.ctx.set_input_shape("iou_thresh", (1,))
        self.ctx.set_input_shape("score_thresh", (1,))
        self.ctx.set_tensor_address("iou_thresh", device_addr(self._d_iou))
        self.ctx.set_tensor_address("score_thresh", device_addr(self._d_score))
        self.ctx.set_tensor_address("detections", device_addr(self._d_out))

        self._h_out = np.empty((MAX_BATCH, MAX_DETS, 6), np.float32)
        self.iou_in = np.zeros(1, np.float32)
        self.score_in = np.zeros(1, np.float32)

    def infer(self, images, iou=0.65, conf=0.45):
        """images: (B,H,W,3) uint8 BGR，H/W 不得超过 MAX_HW。返回 (B,300,6)。"""
        images = np.ascontiguousarray(images)
        b, h, w, _ = images.shape
        assert b <= MAX_BATCH and max(h, w) <= MAX_HW, \
            f"({b},{h},{w}) 超出 profile 上限 batch<={MAX_BATCH}, H/W<={MAX_HW}"
        self.ctx.set_input_shape("image_raw", images.shape)
        self.ctx.set_tensor_address("image_raw", device_addr(self._d_raw))

        self.iou_in[0] = iou
        self.score_in[0] = conf
        cuda_memcpy_h2d(self._d_raw, images.reshape(-1))
        cuda_memcpy_h2d(self._d_iou, self.iou_in)
        cuda_memcpy_h2d(self._d_score, self.score_in)

        if not self.ctx.execute_async_v3(0):
            raise RuntimeError("execute_async_v3 failed")
        cuda_memcpy_d2h(self._h_out[:b], self._d_out)
        return self._h_out[:b].copy()


# ---------------- 工具 ----------------

def fit_image(img):
    """超过 profile 上限时等比缩小，返回 (缩放图, 缩放系数)。检测结果坐标 /scale 回原图。"""
    m = max(img.shape[:2])
    if m <= MAX_HW:
        return img, 1.0
    s = m / MAX_HW
    return cv2.resize(img, (int(round(img.shape[1] / s)), int(round(img.shape[0] / s)))), s


def batch_infer(engine, imgs, iou, conf):
    """imgs: list[np.ndarray(H,W,3)]（可不同尺寸——打包前 pad 到统一画布，pad 用 letterbox 灰 114）。
    返回 [每张图的 (N,6) 数组]（原图各自坐标系）。"""
    pad_val = 114
    ch = max(im.shape[0] for im in imgs)
    cw = max(im.shape[1] for im in imgs)
    canvas = np.full((len(imgs), ch, cw, 3), pad_val, np.uint8)
    offsets = []
    for i, im in enumerate(imgs):
        h, w = im.shape[:2]
        canvas[i, :h, :w] = im            # 左上对齐放置；右侧/下方为 114 灰 = 与 letterbox 填充一致
        offsets.append((0, 0))
    out = engine.infer(canvas, iou=iou, conf=conf)
    res = []
    for i, im in enumerate(imgs):
        d = out[i]
        d = d[d[:, 4] > 0]
        ox, oy = offsets[i]
        d[:, [0, 2]] -= ox
        d[:, [1, 3]] -= oy               # 当前左上对齐，offset=0；预留通用性
        keep = (d[:, 2] > 0) & (d[:, 3] > 0) & (d[:, 0] < im.shape[1]) & (d[:, 1] < im.shape[0])
        res.append(d[keep])
    return res


def draw(img, dets, names=CLASS_NAMES):
    for x1, y1, x2, y2, c, k in dets:
        x1, y1, x2, y2, k = int(x1), int(y1), int(x2), int(y2), int(k)
        color = COLORS[k % len(COLORS)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        tag = f"{names[k] if k < len(names) else k} {c:.2f}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(img, tag, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def save_txt(path, dets, names=CLASS_NAMES):
    """每行: 类别 置信度 x1 y1 x2 y2"""
    with open(path, 'w') as f:
        for x1, y1, x2, y2, c, k in dets:
            f.write(f"{names[int(k)] if int(k) < len(names) else int(k)} "
                    f"{float(c):.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")


# ---------------- 三种输入源 ----------------

def run_images(engine, paths, args, out_dir):
    """单张/文件夹：同尺寸归组 → 组内攒 batch → 推理 → 画框保存。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = {}
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[warn] 无法读取，跳过: {p}")
            continue
        img, scale = fit_image(img)
        groups.setdefault(img.shape[:2], []).append((p, img, scale))

    total, t0 = 0, time.time()
    all_txt = {}
    for (h, w), items in groups.items():
        for i in range(0, len(items), args.batch):
            chunk = items[i:i + args.batch]
            dets_list = batch_infer(engine, [im for _, im, _ in chunk], args.iou, args.conf)
            for (p, im, scale), dets in zip(chunk, dets_list):
                if scale != 1.0:
                    dets[:, :4] /= scale       # 缩放过的图，坐标映射回原始分辨率
                draw(im, dets)
                dst = out_dir / p.name
                cv2.imwrite(str(dst), im)
                if args.save_txt:
                    save_txt(out_dir / (p.stem + '.txt'), dets)
                    all_txt[p.name] = len(dets)
                total += len(dets)
                print(f"{p.name}: {len(dets)} 框  -> {dst}")
    dt = time.time() - t0
    print(f"[done] {len(paths)} 张图, 共 {total} 框, 耗时 {dt:.2f}s "
          f"({len(paths) / dt:.1f} img/s)，结果目录: {out_dir}")
    if args.save_txt and all_txt:
        json.dump(all_txt, open(out_dir / 'summary.json', 'w'), ensure_ascii=False, indent=1)


def run_video(engine, src, args):
    """视频：攒满一个 batch 推一次；结尾不足一 batch 由动态 batch 直接处理。"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    wn = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hn = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = Path(args.out) if args.out else Path(src).with_stem(Path(src).stem + '_result')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (wn, hn))

    buf, scales = [], []
    n_done, n_det, t0 = 0, 0, time.time()

    def flush(frames, scl):
        nonlocal n_done, n_det
        if not frames:
            return
        dets_list = batch_infer(engine, frames, args.iou, args.conf)
        for im, scale, dets in zip(frames, scl, dets_list):
            if scale != 1.0:
                dets[:, :4] /= scale
            draw(im, dets)
            writer.write(cv2.resize(im, (wn, hn)))
            n_det += len(dets)
            n_done += 1
            if n_done % (args.batch * 20) == 0:
                el = time.time() - t0
                print(f"  {n_done}/{total_frames} 帧, {n_det} 框, {n_done / el:.1f} fps")

    ok, frame = cap.read()
    while ok:
        frame, s = fit_image(frame)
        buf.append(frame)
        scales.append(s)
        if len(buf) == args.batch:
            flush(buf, scales)
            buf, scales = [], []
        ok, frame = cap.read()
    flush(buf, scales)                       # ★ 不足一 batch 的尾巴：动态 batch 直接吃下

    cap.release()
    writer.release()
    dt = time.time() - t0
    print(f"[done] {n_done} 帧, 共 {n_det} 框, 耗时 {dt:.2f}s "
          f"(含解码/编码 {n_done / dt:.1f} fps)，输出: {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="/root/my_FILE/datasets/Data_DeepPCB_YOLO/images/test", help="图片路径 / 文件夹 / 视频路径")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_e2e_dyn.engine")
    ap.add_argument("--batch", type=int, default=8, help="文件夹/视频模式的打包 batch（≤profile 上限）")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--out", default=None, help="输出路径（图片模式=结果图路径，文件夹=结果目录，视频=输出视频）")
    ap.add_argument("--save-txt", action="store_true", help="图片/文件夹模式额外保存检测结果 txt")
    args = ap.parse_args()
    assert args.batch <= MAX_BATCH, f"--batch 不能超过 profile 上限 {MAX_BATCH}"

    src = Path(args.source)
    engine = E2EDynEngine(args.engine)

    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not paths:
            raise RuntimeError(f"目录中没有图片: {src}")
        out_dir = Path(args.out) if args.out else src.with_name(src.name + '_result')
        print(f"[folder] {len(paths)} 张图, batch={args.batch}")
        run_images(engine, paths, args, out_dir)
    elif src.suffix.lower() in VID_EXTS:
        print(f"[video] {src}, batch={args.batch}")
        run_video(engine, src, args)
    elif src.suffix.lower() in IMG_EXTS:
        img = cv2.imread(str(src))
        if img is None:
            raise RuntimeError(f"无法读取图片: {src}")
        img_fit, scale = fit_image(img)
        (dets,) = batch_infer(engine, [img_fit], args.iou, args.conf)
        if scale != 1.0:
            dets[:, :4] /= scale
        for x1, y1, x2, y2, c, k in dets:
            name = CLASS_NAMES[int(k)] if int(k) < len(CLASS_NAMES) else int(k)
            print(f"  {name:<10} conf={c:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
        draw(img, dets)
        dst = Path(args.out) if args.out else src.with_stem(src.stem + '_result')
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), img)
        if args.save_txt:
            save_txt(dst.with_suffix('.txt'), dets)
        print(f"[image] {len(dets)} 框 -> {dst}")
    else:
        raise RuntimeError(f"--source 不是图片/文件夹/视频: {src}")


if __name__ == "__main__":
    main()
