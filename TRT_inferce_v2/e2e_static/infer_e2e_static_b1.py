"""
infer_e2e_static_b1.py — batch=1 静态端到端 INT8 引擎 · 全速流水线部署脚本。

与 e2e_dyn/infer_e2e_dyn_b1.py 架构一致（三级流水线 + 锁页内存 + 多槽位环缓冲 + 异步拷贝），
区别在于本脚本面向「静态 640×640 输入」的端到端引擎：

  - 引擎输入固定为 image_raw(1,640,640,3) uint8，因此 resize 不再烧进 engine，而是由
    宿主 CPU 在解码线程里做 letterbox（等比缩放 + 居中灰边到 640×640）；
  - 引擎输出 detections(300,6) 坐标在 640×640 空间；
  - 结果消费线程(IO 线程)用 letterbox 的 scale / pad 把坐标反算回原图，再画框/写盘。

  这样「decode+resize(CPU)」与「GPU 推理」与「坐标还原+画框(CPU)」三级完全重叠。

用法:
    python infer_e2e_static_b1.py --source images_dir/               # 测吞吐
    python infer_e2e_static_b1.py --source images_dir/ --save-img    # 画框存图
    python infer_e2e_static_b1.py --source xxx.jpg --save-img        # 单张
"""

import argparse
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dependency_file"))  # dependency_file/
from common import cuda_malloc, device_addr, cuda_memcpy_d2h            # noqa: E402
from cuda_rt import CudaStream, CudaEvent, PinnedBuffer, async_h2d      # noqa: E402

IMG_SIZE = 640
MAX_DETS = 300
SLOTS = 4                     # 环缓冲槽位数（>2 才能拷贝/执行/后处理三级重叠）
PAD_VALUE = 114

CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
COLORS = [(60, 60, 220), (60, 180, 220), (220, 60, 60),
          (60, 220, 60), (220, 60, 200), (0, 200, 255)]


class E2EStaticEngine:
    """静态 640×640 e2e 引擎的异步封装。所有资源启动时一次分配，运行期零 malloc。"""

    def __init__(self, engine_path, conf=0.45, iou=0.65):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = CudaStream()

        one_img = IMG_SIZE * IMG_SIZE * 3        # 固定输入字节数 640*640*3

        # 环缓冲: 每槽位独立设备输入/输出 + 锁页输入 + 完成事件
        self.d_raw = [cuda_malloc(one_img) for _ in range(SLOTS)]
        self.d_out = [cuda_malloc(MAX_DETS * 6 * 4) for _ in range(SLOTS)]
        self.pin_in = [PinnedBuffer(one_img) for _ in range(SLOTS)]
        self.h_out = [np.empty((MAX_DETS, 6), np.float32) for _ in range(SLOTS)]
        self.ev = [CudaEvent() for _ in range(SLOTS)]
        self._ev_recorded = [False] * SLOTS        # 从未 record 的事件不能 sync(会永久阻塞)
        self._slot = 0

        # 阈值输入: 锁页小缓冲，启动时上传一次（运行期不再动）
        self.pin_iou, self.pin_score = PinnedBuffer(4), PinnedBuffer(4)
        self.pin_iou.view(np.float32)[0] = np.float32(iou)
        self.pin_score.view(np.float32)[0] = np.float32(conf)
        d_iou, d_score = cuda_malloc(4), cuda_malloc(4)
        async_h2d(self.stream, device_addr(d_iou), self.pin_iou.view(np.float32))
        async_h2d(self.stream, device_addr(d_score), self.pin_score.view(np.float32))
        self.ctx.set_tensor_address("iou_thresh", device_addr(d_iou))
        self.ctx.set_tensor_address("score_thresh", device_addr(d_score))

        self.infer_ms = 0.0          # 推理线程内测得的纯 GPU 供给节拍
        self._n_infer = 0

    def submit(self, img: np.ndarray) -> int:
        """提交一张 (640,640,3) uint8，异步执行。返回槽位号；完成由对应事件标记。"""
        slot = self._slot
        self._slot = (self._slot + 1) % SLOTS
        if self._ev_recorded[slot] and not self.ev[slot].query():
            self.ev[slot].sync()     # 该槽位上一轮还没跑完 → 等待（正常流水不会到这）

        src = self.pin_in[slot].view(np.uint8, (IMG_SIZE, IMG_SIZE, 3))
        src[...] = img
        self.ctx.set_tensor_address("image_raw", device_addr(self.d_raw[slot]))
        self.ctx.set_tensor_address("detections", device_addr(self.d_out[slot]))

        t0 = time.perf_counter()
        async_h2d(self.stream, device_addr(self.d_raw[slot]), src)
        if not self.ctx.execute_async_v3(self.stream.handle):
            raise RuntimeError("execute_async_v3 failed")
        self.ev[slot].record(self.stream)
        self._ev_recorded[slot] = True
        self.infer_ms += (time.perf_counter() - t0) * 1000
        self._n_infer += 1
        return slot

    def wait(self, slot: int) -> np.ndarray:
        """等待槽位结果就绪，返回 (300,6) 副本。D2H 用同步拷贝(7.2KB ~5us)。"""
        self.ev[slot].sync()
        cuda_memcpy_d2h(self.h_out[slot], self.d_out[slot])
        return self.h_out[slot].copy()


# ---------------- 预处理 / 后处理坐标还原 ----------------

def letterbox_640(image):
    """CPU 侧 decode + resize：等比缩放 + 居中灰边到 640×640。

    与 pycocotools_test_trt.py 的 preprocess_image 一致（round 取整 + 单边 pad），
    保证推理图与模型标定时的分布一致。返回 (canvas, scale, pad_left, pad_top)。
    """
    h, w = image.shape[:2]
    r = min(IMG_SIZE / h, IMG_SIZE / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = IMG_SIZE - nw, IMG_SIZE - nh
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    canvas = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
    return canvas, r, left, top


def map_back(dets, scale, pad_x, pad_y, orig_w, orig_h):
    """(300,6) → 有效框数组，并把 640×640 空间坐标反 letterbox 回原图坐标。"""
    d = dets[dets[:, 4] > 0]
    if len(d) == 0:
        return d
    d = d.copy()
    d[:, [0, 2]] = (d[:, [0, 2]] - pad_x) / scale
    d[:, [1, 3]] = (d[:, [1, 3]] - pad_y) / scale
    d[:, [0, 2]] = np.clip(d[:, [0, 2]], 0, orig_w)
    d[:, [1, 3]] = np.clip(d[:, [1, 3]], 0, orig_h)
    keep = (d[:, 2] > d[:, 0]) & (d[:, 3] > d[:, 1])
    return d[keep]


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


# ---------------- 三级流水线 ----------------

def decoder_worker(tasks, q_infer, args, errors):
    """生产者: 读图 + letterbox resize。多线程并行解码/缩放。"""
    try:
        for p in iter(tasks.get, None):
            img = cv2.imread(str(p))
            if img is None:
                print(f"[warn] 无法读取，跳过: {p.name}")
                continue
            canvas, scale, pad_x, pad_y = letterbox_640(img)
            q_infer.put((p, img, canvas, scale, pad_x, pad_y))   # 原图 + 640 画布 + 反算参数
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        errors.append(e)
    finally:
        q_infer.put(None)


def infer_worker(engine, q_infer, q_out, args, errors, stats):
    """GPU 线程: 异步提交 + 等待本张完成（下一张的解码/上一张的后处理此时并行）。"""
    try:
        n_none = 0
        while n_none < args.workers:
            item = q_infer.get()
            if item is None:
                n_none += 1
                continue
            p, orig, canvas, scale, pad_x, pad_y = item
            slot = engine.submit(canvas)
            dets = engine.wait(slot)
            q_out.put((p, orig, scale, pad_x, pad_y, dets))
        for _ in range(args.consumers):
            q_out.put(None)
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        errors.append(e)
        q_out.put(None)


def wait_threads(threads, errors):
    """看门狗式 join: 任一线程死亡(errors 非空)立即退出进程，防队列死锁。"""
    while any(t.is_alive() for t in threads):
        if errors:
            traceback.print_exception(type(errors[0]), errors[0], errors[0].__traceback__)
            import os
            os._exit(1)
        time.sleep(0.2)


def consumer_worker(q_out, args, stats, stats_lock, errors):
    """消费者(IO 线程): 用 scale/pad 反算坐标回原图 / 计数 / 可选画框写盘。"""
    try:
        while True:
            item = q_out.get()
            if item is None:
                return
            p, orig, scale, pad_x, pad_y, dets = item
            orig_h, orig_w = orig.shape[:2]
            dets = map_back(dets, scale, pad_x, pad_y, orig_w, orig_h)
            with stats_lock:
                stats['dets'] += len(dets)
                stats['n'] += 1
                n = stats['n']
            if n % 200 == 0:
                print(f"  ... {n} 张, {stats['dets']} 框")
            if args.save_img:
                draw(orig, dets)
                cv2.imwrite(str(args.result_dir / p.name), orig)
            if args.save_txt:
                with open(args.result_dir / (p.stem + '.txt'), 'w') as f:
                    for x1, y1, x2, y2, c, k in dets:
                        f.write(f"{CLASS_NAMES[int(k)] if int(k) < len(CLASS_NAMES) else int(k)} "
                                f"{float(c):.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        errors.append(e)


def run_images(engine, paths, args):
    args.result_dir.mkdir(parents=True, exist_ok=True)
    tasks = queue.Queue()
    q_infer = queue.Queue(maxsize=2 * args.workers)
    q_out = queue.Queue(maxsize=2 * args.consumers)

    for p in paths:
        tasks.put(p)
    for _ in range(args.workers):
        tasks.put(None)                              # 解码退出哨兵

    errors, stats = [], {'n': 0, 'dets': 0}
    stats_lock = threading.Lock()
    threads = [threading.Thread(target=decoder_worker, args=(tasks, q_infer, args, errors))
               for _ in range(args.workers)] + \
              [threading.Thread(target=infer_worker, args=(engine, q_infer, q_out, args, errors, stats))] + \
              [threading.Thread(target=consumer_worker, args=(q_out, args, stats, stats_lock, errors))
               for _ in range(args.consumers)]

    t0 = time.time()
    for t in threads:
        t.daemon = True
        t.start()
    wait_threads(threads, errors)

    if errors:
        raise errors[0]
    dt = time.time() - t0
    gpu_ms = engine.infer_ms / max(engine._n_infer, 1)
    print(f"[done] {stats['n']} 张图, 共 {stats['dets']} 框, 耗时 {dt:.2f}s "
          f"→ 端到端 {stats['n'] / dt:.1f} img/s "
          f"(纯GPU节拍 {gpu_ms:.2f} ms/张, 理论上限 {1000 / gpu_ms:.0f} img/s)")
    print(f"       端到端/GPU 节拍比 {(stats['n'] / dt) / (1000 / gpu_ms) * 100:.0f}% "
          f"(<100% 说明瓶颈在解码或后处理, 可加 --workers)")


def run_single(engine, src, args):
    img = cv2.imread(str(src))
    if img is None:
        raise RuntimeError(f"无法读取图片: {src}")
    orig_h, orig_w = img.shape[:2]
    canvas, scale, pad_x, pad_y = letterbox_640(img)
    t0 = time.perf_counter()
    slot = engine.submit(canvas)
    dets = engine.wait(slot)
    infer_ms = (time.perf_counter() - t0) * 1000
    dets = map_back(dets, scale, pad_x, pad_y, orig_w, orig_h)
    for x1, y1, x2, y2, c, k in dets:
        name = CLASS_NAMES[int(k)] if int(k) < len(CLASS_NAMES) else int(k)
        print(f"  {name:<10} conf={c:.3f} box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
    print(f"[image] {len(dets)} 框, 推理 {infer_ms:.2f} ms")
    if args.save_img or args.save_txt:
        args.result_dir.mkdir(parents=True, exist_ok=True)
        draw(img, dets)
        cv2.imwrite(str(args.result_dir / (src.stem + '_result.jpg')), img)
        if args.save_txt:
            with open(args.result_dir / (src.stem + '.txt'), 'w') as f:
                for x1, y1, x2, y2, c, k in dets:
                    f.write(f"{CLASS_NAMES[int(k)] if int(k) < len(CLASS_NAMES) else int(k)} "
                            f"{float(c):.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
        print(f"       结果 → {args.result_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="/root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test_scaled",
                    help="图片路径 / 文件夹")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_int8_static_e2e.engine")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--out", default="/root/my_FILE/trt_study/TRT_inferce_v2/e2e_static/images_infer", help="结果目录 (默认 <源目录>_result_static_b1)")
    ap.add_argument("--save-txt", action="store_true", help="保存检测 txt（消费线程执行）")
    ap.add_argument("--save-img", action="store_true", help="保存画框结果图（消费线程执行）")
    ap.add_argument("--workers", type=int, default=4, help="解码线程数 (默认 3)")
    ap.add_argument("--consumers", type=int, default=4, help="结果消费线程数 (默认 3)")
    args = ap.parse_args()

    src = Path(args.source)
    if args.out:
        args.result_dir = Path(args.out)
    else:
        args.result_dir = src.with_name(src.name + '_result_static_b1') if src.is_dir() \
            else src.with_stem(src.stem + '_result_static_b1').parent

    engine = E2EStaticEngine(args.engine, conf=args.conf, iou=args.iou)

    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not paths:
            raise RuntimeError(f"目录中没有图片: {src}")
        print(f"[folder] {len(paths)} 张图, batch=1 静态640, 解码线程={args.workers}, 槽位={SLOTS}")
        run_images(engine, paths, args)
    elif src.suffix.lower() in IMG_EXTS:
        run_single(engine, src, args)
    else:
        raise RuntimeError(f"--source 不是图片/文件夹: {src}")


if __name__ == "__main__":
    main()
