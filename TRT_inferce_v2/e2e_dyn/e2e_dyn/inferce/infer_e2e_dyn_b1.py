"""
infer_e2e_dyn_b1.py — batch=1 端到端 INT8 引擎 · 全速流水线部署脚本。

与 infer_e2e_dyn.py 的区别（性能全开版）:
  1. 三级流水线，级间用有界队列衔接，CPU 解码与 GPU 推理完全重叠:
       解码线程池(可配) → GPU 推理线程(异步拷贝+多槽位双缓冲) → 结果消费线程
  2. 全部 H2D/D2H 走 锁页内存 + cudaMemcpyAsync（common.py 的同步拷贝会阻塞 CPU）
  3. 多槽位环缓冲: 第 N 张图在 GPU 上跑时，第 N+1 张的拷贝/第 N-1 张的后处理并行进行
  4. 热路径零打印、默认不写图片不画框（--save-img/--save-txt 显式开启，
     且画框/写盘全部在消费线程做，不占推理时间）
  5. iou/score 阈值只在启动时上传一次（原版每次推理都传）

适用: 单路实时流 / 低延迟单张处理。批量吞吐场景用 infer_e2e_dyn_b8.py。

用法（与 infer_e2e_dyn.py 参数兼容）:
    python infer_e2e_dyn_b1.py --source images_dir/              # 测吞吐
    python infer_e2e_dyn_b1.py --source images_dir/ --save-img   # 需要结果图时
    python infer_e2e_dyn_b1.py --source xxx.jpg --save-txt       # 单张
"""

import argparse
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"dependency_file"))    # dependency_file/
from common import cuda_malloc, device_addr, cuda_memcpy_d2h  # noqa: E402
from cuda_rt import CudaStream, CudaEvent, PinnedBuffer, async_h2d  # noqa: E402

MAX_HW = 1920
MAX_DETS = 300
SLOTS = 4                     # 环缓冲槽位数（>2 才能拷贝/执行/后处理三级重叠）

CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
COLORS = [(60, 60, 220), (60, 180, 220), (220, 60, 60),
          (60, 220, 60), (220, 60, 200), (0, 200, 255)]


class E2EB1Engine:
    """batch=1 e2e 引擎的异步封装。所有资源启动时一次分配，运行期零 malloc。"""

    def __init__(self, engine_path, conf=0.45, iou=0.65):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = CudaStream()

        one_img = MAX_HW * MAX_HW * 3          # 单张最大输入字节数

        # 环缓冲: 每槽位 独立 设备输入/输出 + 锁页输入/输出 + 完成事件
        self.d_raw = [cuda_malloc(one_img) for _ in range(SLOTS)]
        self.d_out = [cuda_malloc(MAX_DETS * 6 * 4) for _ in range(SLOTS)]
        self.pin_in = [PinnedBuffer(one_img) for _ in range(SLOTS)]
        self.h_out = [np.empty((MAX_DETS, 6), np.float32) for _ in range(SLOTS)]
        self.ev = [CudaEvent() for _ in range(SLOTS)]
        self._ev_recorded = [False] * SLOTS            # 从未 record 的事件不能 sync(会永久阻塞)
        self._slot = 0
        self._last_shape = None

        # 阈值输入: 锁页小缓冲，启动时上传一次（运行期不再动）
        self.pin_iou, self.pin_score = PinnedBuffer(4), PinnedBuffer(4)
        self.pin_iou.view(np.float32)[0] = np.float32(iou)
        self.pin_score.view(np.float32)[0] = np.float32(conf)
        self.ctx.set_input_shape("iou_thresh", (1,))
        self.ctx.set_input_shape("score_thresh", (1,))
        async_h2d(self.stream, device_addr(cuda_iou := cuda_malloc(4)), self.pin_iou.view(np.float32))
        async_h2d(self.stream, device_addr(cuda_score := cuda_malloc(4)), self.pin_score.view(np.float32))
        self.ctx.set_tensor_address("iou_thresh", device_addr(cuda_iou))
        self.ctx.set_tensor_address("score_thresh", device_addr(cuda_score))

        self.infer_ms = 0.0          # 推理线程内测得的纯 GPU 供给节拍
        self._n_infer = 0

    def submit(self, img: np.ndarray) -> int:
        """提交一张 (H,W,3) uint8，异步执行。返回槽位号；GPU 完成由对应事件标记。"""
        h, w, c = img.shape
        slot = self._slot
        self._slot = (self._slot + 1) % SLOTS
        if self._ev_recorded[slot] and not self.ev[slot].query():
            self.ev[slot].sync()     # 该槽位上一轮还没跑完 → 等待（正常流水不会到这）

        src = self.pin_in[slot].view(np.uint8)[: h * w * c].reshape(h, w, c)
        src[...] = img

        shape = (1, h, w, 3)
        if shape != self._last_shape:            # 形状没变就不重复设置（省 TRT 校验开销）
            self.ctx.set_input_shape("image_raw", shape)
            self._last_shape = shape
        self.ctx.set_tensor_address("image_raw", device_addr(self.d_raw[slot]))
        self.ctx.set_tensor_address("detections", device_addr(self.d_out[slot]))

        t0 = time.perf_counter()
        async_h2d(self.stream, device_addr(self.d_raw[slot]),
                  self.pin_in[slot].view(np.uint8)[: h * w * 3].reshape(1, h, w, 3))
        if not self.ctx.execute_async_v3(self.stream.handle):
            raise RuntimeError("execute_async_v3 failed")
        self.ev[slot].record(self.stream)
        self._ev_recorded[slot] = True
        self.infer_ms += (time.perf_counter() - t0) * 1000
        self._n_infer += 1
        return slot

    def wait(self, slot: int) -> np.ndarray:
        """等待槽位结果就绪，返回 (300,6) 副本。
        D2H 用同步拷贝: 输出仅 7.2KB (~5us)，本机实测异步 D2H 写锁页缓冲会静默丢数据。"""
        self.ev[slot].sync()
        cuda_memcpy_d2h(self.h_out[slot], self.d_out[slot])
        return self.h_out[slot].copy()


# ---------------- 工具（与 infer_e2e_dyn.py 保持一致） ----------------

def fit_image(img):
    m = max(img.shape[:2])
    if m <= MAX_HW:
        return img, 1.0
    s = m / MAX_HW
    return cv2.resize(img, (int(round(img.shape[1] / s)), int(round(img.shape[0] / s)))), s


def filter_dets(dets, orig_w, orig_h, scale):
    """(300,6) → 有效框数组，并映射回原始分辨率坐标"""
    d = dets[dets[:, 4] > 0]
    if scale != 1.0:
        d[:, :4] /= scale
    keep = (d[:, 2] > 0) & (d[:, 3] > 0) & (d[:, 0] < orig_w) & (d[:, 1] < orig_h)
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
    """生产者: 读图 + 超限缩放。多线程并行解码。"""
    try:
        for p in iter(tasks.get, None):
            img = cv2.imread(str(p))
            if img is None:
                print(f"[warn] 无法读取，跳过: {p.name}")
                continue
            img, scale = fit_image(img)
            q_infer.put((p, img, scale))            # 有界队列: 解码过快自动限流
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        errors.append(e)
    finally:
        q_infer.put(None)


def infer_worker(engine, q_infer, q_out, args, errors, stats):
    """GPU 线程: 异步提交 + 等待本张完成（下一张的解码/上一张的后处理此时都在并行）。"""
    try:
        n_none = 0
        while n_none < args.workers:
            item = q_infer.get()
            if item is None:
                n_none += 1
                continue
            p, img, scale = item
            slot = engine.submit(img)
            dets = engine.wait(slot)
            q_out.put((p, img, scale, dets))        # 有界队列: GPU 跑太快时自动限流
        for _ in range(args.consumers):
            q_out.put(None)
    except Exception as e:                          # noqa: BLE001
        traceback.print_exc()
        errors.append(e)
        q_out.put(None)


def wait_threads(threads, errors):
    """看门狗式 join: 任一线程死亡(errors 非空)立即打印并退出进程, 防止队列死锁卡死。"""
    while any(t.is_alive() for t in threads):
        if errors:
            traceback.print_exception(type(errors[0]), errors[0], errors[0].__traceback__)
            os._exit(1)
        time.sleep(0.2)


def consumer_worker(q_out, args, stats, stats_lock, errors):
    """消费者: 坐标映射 / 计数 / 可选画框写盘。可多线程并行 (draw/imwrite 释放 GIL)。"""
    try:
        while True:
            item = q_out.get()
            if item is None:
                return
            p, img, scale, dets = item
            dets = filter_dets(dets, img.shape[1], img.shape[0], scale)
            with stats_lock:
                stats['dets'] += len(dets)
                stats['n'] += 1
                n = stats['n']
            if n % 200 == 0:
                print(f"  ... {n} 张, {stats['dets']} 框")
            if args.save_img:
                draw(img, dets)
                cv2.imwrite(str(args.result_dir / p.name), img)
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
    tasks, q_infer, q_out = queue.Queue(), queue.Queue(maxsize=2 * args.workers), queue.Queue(maxsize=2 * args.consumers)

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
    orig_w, orig_h = img.shape[1], img.shape[0]
    img, scale = fit_image(img)
    t0 = time.perf_counter()
    slot = engine.submit(img)
    dets = engine.wait(slot)
    infer_ms = (time.perf_counter() - t0) * 1000
    dets = filter_dets(dets, orig_w, orig_h, scale)
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
                    f.write(f"{CLASS_NAMES[int(k)]} {float(c):.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
        print(f"       结果 → {args.result_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="/root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test",
                    help="图片路径 / 文件夹")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_e2e_dyn.engine")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--out", default=None, help="结果目录 (默认 <源目录>_result_b1)")
    ap.add_argument("--save-txt", action="store_true", help="保存检测 txt（消费线程执行）")
    ap.add_argument("--save-img", action="store_true", help="保存画框结果图（消费线程执行）")
    ap.add_argument("--workers", type=int, default=3, help="解码线程数 (默认 3)")
    ap.add_argument("--consumers", type=int, default=3, help="结果消费线程数 (默认 2, save 时调大)")
    args = ap.parse_args()

    src = Path(args.source)
    if args.out:
        args.result_dir = Path(args.out)
    else:
        args.result_dir = src.with_name(src.name + '_result_b1') if src.is_dir() \
            else src.with_stem(src.stem + '_result_b1').parent

    engine = E2EB1Engine(args.engine, conf=args.conf, iou=args.iou)

    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not paths:
            raise RuntimeError(f"目录中没有图片: {src}")
        print(f"[folder] {len(paths)} 张图, batch=1, 解码线程={args.workers}, 槽位={SLOTS}")
        run_images(engine, paths, args)
    elif src.suffix.lower() in IMG_EXTS:
        run_single(engine, src, args)
    else:
        raise RuntimeError(f"--source 不是图片/文件夹: {src} (batch=1 版不支持视频, 请用 b8 版)")


if __name__ == "__main__":
    main()
