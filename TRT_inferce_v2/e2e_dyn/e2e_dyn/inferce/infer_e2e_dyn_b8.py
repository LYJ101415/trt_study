"""
infer_e2e_dyn_b8.py — batch=8 端到端 INT8 引擎 · 四级全速流水线部署脚本。

与 infer_e2e_dyn.py 的区别（性能全开版）:
  1. 四级流水线，级间有界队列衔接，解码/攒批/GPU/后处理四级完全重叠:
       解码线程池 → 攒批线程(同尺寸归组+填锁页画布) → GPU推理线程(异步拷贝+多槽位) → 消费线程
  2. 全部 H2D/D2H 走 锁页内存 + cudaMemcpyAsync；canvas 直接填进锁页缓冲，零中间拷贝
  3. 多槽位环缓冲: 第 N 批在 GPU 上执行时，第 N+1 批拷贝、第 N-1 批后处理并行进行
  4. iou/score 只在启动时上传一次；热路径零打印；写盘全部在消费线程
  5. 攒批是流式的: 图片边解码边归组攒批，同尺寸不足一 batch 的尾巴由动态 batch 直接吃下

GPU 纯推理参考值 (trtexec, 640x640): batch=1 约 2.33 ms/张 → batch=8 约 0.64 ms/张，
batch 化本身有 3.6x 收益；本脚本的任务是让 CPU 端 I/O 不再吃掉这份收益。

用法:
    python infer_e2e_dyn_b8.py --source images_dir/                # 测吞吐
    python infer_e2e_dyn_b8.py --source images_dir/ --save-img     # 需要结果图
    python infer_e2e_dyn_b8.py --source video.mp4 --batch 8        # 视频
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

sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"dependency_file"))   # dependency_file/
from common import cuda_malloc, device_addr, cuda_memcpy_d2h  # noqa: E402
from cuda_rt import CudaStream, CudaEvent, PinnedBuffer, async_h2d  # noqa: E402

MAX_BATCH = 8
MAX_HW = 1920
MAX_DETS = 300
SLOTS = 4                     # 环缓冲槽位数（每槽可容纳一个完整 batch）

CLASS_NAMES = ['open', 'short', 'mousebite', 'spur', 'copper', 'pinhole']
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VID_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
COLORS = [(60, 60, 220), (60, 180, 220), (220, 60, 60),
          (60, 220, 60), (220, 60, 200), (0, 200, 255)]


class SlotPool:
    """槽位资源池: 按需增长的 设备缓冲 + 锁页缓冲 + 事件（换形状时自动扩容）。"""

    def __init__(self, n_slots):
        self._items = []
        for _ in range(n_slots):
            self._items.append({
                "dev_in_bytes": 0, "dev_in": None,
                "pin_in": None,
                # 输出是 (B,300,6)，必须按 MAX_BATCH 分配，否则 batch>1 时越界写显存
                "dev_out": cuda_malloc(MAX_BATCH * MAX_DETS * 6 * 4),
                "h_out": np.empty((MAX_BATCH, MAX_DETS, 6), np.float32),
                "n": 0,
                "ev": CudaEvent(),
                "ev_recorded": False,     # 从未 record 的事件不能 sync(会永久阻塞)
                "drained": threading.Event(),  # GPU线程已把本槽输出拷走(复用槽位前必须等它)
                "used": False,
            })

    def _ensure(self, it, in_bytes):
        if it["dev_in_bytes"] < in_bytes:
            if it["dev_in"] is not None:
                from common import cuda_free
                cuda_free(it["dev_in"])
            it["dev_in"] = cuda_malloc(in_bytes)
            it["pin_in"] = PinnedBuffer(in_bytes)      # 旧锁页块随对象释放(未显式free, 进程退出回收)
            it["dev_in_bytes"] = in_bytes
        return it

    def get(self, idx, in_bytes):
        return self._ensure(self._items[idx], in_bytes)


class E2EB8Engine:
    """batch=8 e2e 引擎异步封装。submit 一次提交一个已填好的锁页画布槽位。"""

    def __init__(self, engine_path, conf=0.45, iou=0.65):
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.stream = CudaStream()
        self.pool = SlotPool(SLOTS)

        self.pin_iou, self.pin_score = PinnedBuffer(4), PinnedBuffer(4)
        self.pin_iou.view(np.float32)[0] = np.float32(iou)
        self.pin_score.view(np.float32)[0] = np.float32(conf)
        self.ctx.set_input_shape("iou_thresh", (1,))
        self.ctx.set_input_shape("score_thresh", (1,))
        d_iou, d_score = cuda_malloc(4), cuda_malloc(4)
        async_h2d(self.stream, device_addr(d_iou), self.pin_iou.view(np.float32))
        async_h2d(self.stream, device_addr(d_score), self.pin_score.view(np.float32))
        self.ctx.set_tensor_address("iou_thresh", device_addr(d_iou))
        self.ctx.set_tensor_address("score_thresh", device_addr(d_score))

        self._slot = 0
        self._last_shape = None
        self.infer_ms = 0.0
        self._n_infer = 0

    def submit(self, canvas: np.ndarray) -> int:
        """canvas: (B,H,W,3) uint8（调用方已填入锁页缓冲则应直接传缓冲视图）。"""
        b, h, w, c = canvas.shape
        it = self.pool.get(self._slot, canvas.nbytes)
        slot = self._slot
        self._slot = (self._slot + 1) % SLOTS
        if it["ev_recorded"] and not it["ev"].query():
            it["ev"].sync()

        shape = (b, h, w, 3)
        if shape != self._last_shape:
            self.ctx.set_input_shape("image_raw", shape)
            self._last_shape = shape
        self.ctx.set_tensor_address("image_raw", device_addr(it["dev_in"]))
        self.ctx.set_tensor_address("detections", device_addr(it["dev_out"]))
        it["n"] = b                                   # 记录本批实际 batch 数

        t0 = time.perf_counter()
        async_h2d(self.stream, device_addr(it["dev_in"]), canvas.reshape(-1).view(np.uint8))
        if not self.ctx.execute_async_v3(self.stream.handle):
            raise RuntimeError("execute_async_v3 failed")
        it["ev"].record(self.stream)
        it["ev_recorded"] = True
        it["used"] = True
        self.infer_ms += (time.perf_counter() - t0) * 1000
        self._n_infer += 1
        return slot

    def wait(self, slot: int) -> np.ndarray:
        """D2H 用同步拷贝: 输出仅 7.2KB (~5us)，本机实测异步 D2H 写锁页缓冲会静默丢数据。"""
        it = self.pool._items[slot]
        it["ev"].sync()
        cuda_memcpy_d2h(it["h_out"], it["dev_out"])
        it["drained"].set()                           # 通知攒批线程: 本槽可以复用了
        return it["h_out"][:it["n"]].copy()           # (n,300,6)，每张图一份 (300,6)


# ---------------- 工具 ----------------

def fit_image(img):
    m = max(img.shape[:2])
    if m <= MAX_HW:
        return img, 1.0
    s = m / MAX_HW
    return cv2.resize(img, (int(round(img.shape[1] / s)), int(round(img.shape[0] / s)))), s


def filter_dets(dets, orig_w, orig_h, scale):
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


# ---------------- 四级流水线 ----------------

def decoder_worker(tasks, q_frame, args, errors):
    """生产者: 读图 + 超限缩放。"""
    try:
        for p in iter(tasks.get, None):
            img = cv2.imread(str(p))
            if img is None:
                print(f"[warn] 无法读取，跳过: {p.name}")
                continue
            img, scale = fit_image(img)
            q_frame.put((p, img, scale))
    except Exception as e:                              # noqa: BLE001
        traceback.print_exc()
        errors.append(e)
    finally:
        q_frame.put(None)


def assembler_worker(q_frame, q_gpu, args, engine, errors, stats):
    """攒批: 同尺寸流式归组 → 填锁页画布槽位 → 提交 GPU 线程队列。
    结束时对每组不足 batch 的尾巴做部分批提交（动态 batch 直接支持）。"""
    try:
        groups = {}                                     # (h,w) -> [(p, img, scale), ...]
        n_none = 0
        while n_none < args.workers:
            item = q_frame.get()
            if item is None:
                n_none += 1
                continue
            p, img, scale = item
            groups.setdefault(img.shape[:2], []).append((p, img, scale))
            g = groups[img.shape[:2]]
            if len(g) >= args.batch:                    # 攒满一批
                _flush_group(engine, q_gpu, g[:args.batch], errors, stats)
                del g[:args.batch]

        for shape in list(groups):                      # 收尾: 各组尾巴
            g = groups[shape]
            while g:                                    # 尾巴可能超过一批(理论上不会, 稳妥处理)
                _flush_group(engine, q_gpu, g[:args.batch], errors, stats)
                del g[:args.batch]
        q_gpu.put(None)
    except Exception as e:                              # noqa: BLE001
        traceback.print_exc()
        errors.append(e)
        q_gpu.put(None)


def _flush_group(engine, q_gpu, items, errors, stats):
    """把一组同尺寸图填进锁页画布并提交。"""
    try:
        b = len(items)
        h, w = items[0][1].shape[:2]
        slot = engine._slot                              # 预取槽位并等待空闲（背压点）
        it = engine.pool.get(slot, b * h * w * 3)
        if it["used"]:
            it["drained"].wait()      # 等上一轮结果被 GPU 线程拷走, 否则会覆盖未读的输出!
            it["drained"].clear()
        if it["ev_recorded"] and not it["ev"].query():
            it["ev"].sync()
        canvas = it["pin_in"].view(np.uint8, (b, h, w, 3))
        canvas[...] = 114                                # letterbox 灰底
        for i, (_, img, _) in enumerate(items):
            ih, iw = img.shape[:2]
            canvas[i, :ih, :iw] = img                    # 左上对齐放置（与原版一致）
        engine.submit(canvas)
        q_gpu.put((slot, items))
    except Exception as e:                              # noqa: BLE001
        traceback.print_exc()
        errors.append(e)
        raise


def gpu_worker(engine, q_gpu, q_out, args, errors, stats):
    """GPU 线程: 等待本批完成并把结果交给消费线程（下一批攒批/上一批后处理并行中）。
    完成节拍(相邻两批 wait 返回的时间差)即流水线 GPU 级的实际供给速度。"""
    try:
        last_t = None
        while True:
            item = q_gpu.get()
            if item is None:
                break
            slot, items = item
            out = engine.wait(slot)                      # (300,6) 副本, 已同步
            now = time.perf_counter()
            if last_t is not None:
                stats['gpu_ms'] += (now - last_t) * 1000
                stats['gpu_n'] += len(items)
            last_t = now
            q_out.put((items, out))
        for _ in range(args.consumers):
            q_out.put(None)
    except Exception as e:                              # noqa: BLE001
        traceback.print_exc()
        errors.append(e)
        q_out.put(None)


def consumer_worker(q_out, args, stats, stats_lock, errors):
    try:
        while True:
            item = q_out.get()
            if item is None:
                return
            items, out = item
            for i, (p, img, scale) in enumerate(items):
                dets = filter_dets(out[i], img.shape[1], img.shape[0], scale)
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
    except Exception as e:                              # noqa: BLE001
        traceback.print_exc()
        errors.append(e)


def wait_threads(threads, errors):
    """看门狗式 join: 任一线程死亡(errors 非空)立即打印并退出进程。
    否则一个消费者崩溃就会让上游线程永久阻塞在队列 put 上(死锁卡死)。"""
    while any(t.is_alive() for t in threads):
        if errors:
            traceback.print_exception(type(errors[0]), errors[0], errors[0].__traceback__)
            os._exit(1)
        time.sleep(0.2)


# ---------------- 视频模式 ----------------

def run_video(engine, src, args):
    """视频: 解码攒批 → 流水线推理 → 写输出视频（帧的画框/编码在消费线程做）。"""
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    wn, hn = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = Path(args.out) if args.out else Path(src).with_stem(Path(src).stem + '_result_b8')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    q_frame, q_gpu, q_out = (queue.Queue(maxsize=4 * args.batch),
                             queue.Queue(maxsize=SLOTS),
                             queue.Queue(maxsize=2 * args.consumers))
    errors, stats = [], {'n': 0, 'dets': 0, 'gpu_ms': 0.0, 'gpu_n': 0}
    stats_lock = threading.Lock()
    stop = threading.Event()

    def video_decoder():
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame, scale = fit_image(frame)
                q_frame.put((None, frame, scale))
        except Exception as e:                          # noqa: BLE001
            errors.append(e)
        finally:
            q_frame.put(None)

    def video_consumer():
        writer = None
        try:
            while True:
                item = q_out.get()
                if item is None:
                    return
                items, out = item
                for i, (_, img, scale) in enumerate(items):
                    dets = filter_dets(out[i], img.shape[1], img.shape[0], scale)
                    stats['dets'] += len(dets)
                    stats['n'] += 1
                    draw(img, dets)
                    if writer is None:
                        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'),
                                                 fps, (wn, hn))
                    writer.write(cv2.resize(img, (wn, hn)))
        except Exception as e:                          # noqa: BLE001
            errors.append(e)
        finally:
            if writer is not None:
                writer.release()
            stop.set()

    threads = [threading.Thread(target=video_decoder),
               threading.Thread(target=assembler_worker, args=(q_frame, q_gpu, args, engine, errors, stats)),
               threading.Thread(target=gpu_worker, args=(engine, q_gpu, q_out, args, errors, stats)),
               threading.Thread(target=video_consumer)]
    t0 = time.time()
    for t in threads:
        t.daemon = True
        t.start()
    wait_threads(threads, errors)
    cap.release()

    if errors:
        raise errors[0]
    dt = time.time() - t0
    print(f"[done] {stats['n']} 帧, 共 {stats['dets']} 框, 耗时 {dt:.2f}s "
          f"(含编解码 {stats['n'] / dt:.1f} fps)，输出: {out_path}")


# ---------------- 主入口 ----------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="/root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test",
                    help="图片文件夹 / 视频路径")
    ap.add_argument("--engine", default="/root/my_FILE/models/yolov8_e2e_dyn.engine")
    ap.add_argument("--batch", type=int, default=8, help="攒批大小 (≤profile 上限)")
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--iou", type=float, default=0.65)
    ap.add_argument("--out", default="/root/my_FILE/infer_images", help="结果目录/输出视频路径")
    ap.add_argument("--save-txt", action="store_true", help="保存检测 txt")
    ap.add_argument("--save-img", action="store_true", help="保存画框结果图")
    ap.add_argument("--workers", type=int, default=3, help="解码线程数 (默认 3)")
    ap.add_argument("--consumers", type=int, default=3, help="结果消费线程数 (默认 2, save 时调大)")
    args = ap.parse_args()
    assert args.batch <= MAX_BATCH, f"--batch 不能超过 profile 上限 {MAX_BATCH}"

    src = Path(args.source)
    if args.out:
        args.result_dir = Path(args.out)
    else:
        args.result_dir = src.with_name(src.name + '_result_b8') if src.is_dir() else out_path_dir(src)

    engine = E2EB8Engine(args.engine, conf=args.conf, iou=args.iou)

    if src.is_dir():
        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not paths:
            raise RuntimeError(f"目录中没有图片: {src}")
        if args.save_img or args.save_txt:
            args.result_dir.mkdir(parents=True, exist_ok=True)   # 缺这句: save时消费线程 FileNotFoundError → 全线死锁
        print(f"[folder] {len(paths)} 张图, batch={args.batch}, 解码线程={args.workers}, 槽位={SLOTS}")
        tasks, q_frame, q_gpu, q_out = (queue.Queue(),
                                        queue.Queue(maxsize=4 * args.batch),
                                        queue.Queue(maxsize=SLOTS),
                                        queue.Queue(maxsize=2 * args.consumers))
        for p in paths:
            tasks.put(p)
        for _ in range(args.workers):
            tasks.put(None)

        errors, stats = [], {'n': 0, 'dets': 0, 'gpu_ms': 0.0, 'gpu_n': 0}
        stats_lock = threading.Lock()
        threads = [threading.Thread(target=decoder_worker, args=(tasks, q_frame, args, errors))
                   for _ in range(args.workers)] + \
                  [threading.Thread(target=assembler_worker, args=(q_frame, q_gpu, args, engine, errors, stats)),
                   threading.Thread(target=gpu_worker, args=(engine, q_gpu, q_out, args, errors, stats))] + \
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
        per_img_gpu = stats['gpu_ms'] / max(stats['gpu_n'], 1)
        print(f"[done] {stats['n']} 张图, 共 {stats['dets']} 框, 耗时 {dt:.2f}s "
              f"→ 端到端 {stats['n'] / dt:.1f} img/s "
              f"(纯GPU节拍 {per_img_gpu:.2f} ms/张, 理论上限 {1000 / per_img_gpu:.0f} img/s)")
        print(f"       端到端/GPU 节拍比 {(stats['n'] / dt) / (1000 / per_img_gpu) * 100:.0f}% "
              f"(<100% 说明瓶颈在解码或后处理, 可加 --workers)")
    elif src.suffix.lower() in VID_EXTS:
        print(f"[video] {src}, batch={args.batch}")
        run_video(engine, src, args)
    else:
        raise RuntimeError(f"--source 不是文件夹/视频: {src} (单张图片请用 b1 版)")


def out_path_dir(src):
    """视频模式无 --out 时不创建结果目录（视频输出路径在 run_video 内决定）"""
    return Path('/tmp') / 'e2e_b8_unused'


if __name__ == "__main__":
    main()
