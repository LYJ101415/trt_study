# cpp_work — 常驻推理的两种方案 + 加固版 main.cpp

本目录针对「图片目录持续有新图到达、需要 7×24 推理」的场景，提供两种常驻方案和一个
健壮性加固后的 C++ 推理程序。

| 文件 | 说明 |
|------|------|
| `main.cpp` | **加固版单发 worker**（跑一次就退）。方案 A 的基础，也单独可用 |
| `run_loop.sh` | **方案 A**：外部调度器（监视目录 → 反复拉起 main.cpp） |
| `main_daemon.cpp` | **方案 B**：内嵌目录监听的常驻 daemon |
| `build.sh` | 编译上面两个 .cpp |
| `README.md` | 本文件 |

---

## 一、两种方案怎么选

| | 方案 A（run_loop.sh） | 方案 B（main_daemon） |
|---|---|---|
| 循环在哪 | 程序外面（shell） | 程序里面（C++） |
| main.cpp 是否改动 | 不用改（用 `main.cpp`） | 换成 `main_daemon.cpp` |
| 引擎是否常热 | 否，每轮重新加载引擎 | 是，只加载一次 |
| 适合 | 图片一批批到、分钟级频率、改动最小 | 图片秒级高频到达、追求零重启开销 |
| 崩溃恢复 | 脚本 while 死循环自动拉起下一轮 | 致命错误 `_Exit(1)`，需 systemd 等 supervisor 重启 |

**共同点**：两者都通过「处理完把图移到 `done/`」来去重，且天然无竞态——处理期间新到的
图留在源目录，下一轮才被扫到。

---

## 二、用法

### 1. 编译

```bash
cd /root/my_FILE/cpp_pipeline/make_cpp/cpp_work
bash build.sh
# 生成 ./infer_e2e_b8_cpp 和 ./infer_e2e_b8_daemon
```

### 2. 方案 A：外部调度

编辑 `run_loop.sh` 顶部配置区（SOURCE / ENGINE / BIN 等），然后：

```bash
# 前台跑（调试）
bash run_loop.sh

# 后台常驻（推荐配合 systemd 或 nohup）
nohup bash run_loop.sh >/dev/null 2>&1 &
```

行为：监视 `SOURCE`，有新图就移入 `.staging/` → 拉起 `infer_e2e_b8_cpp` 处理 →
成功后移入 `done/`，失败退回 `SOURCE` 重试。日志在 `run_loop.log`。

### 3. 方案 B：内嵌监听 daemon

```bash
./infer_e2e_b8_daemon --source 图片目录 --engine xxx.engine --batch 8 \
    --poll-ms 1000 --save-img --out 结果目录
```

行为：进程常驻，轮询（默认 1s）`--source`，发现新图就推理，处理完移入 `source/done/`，
`Ctrl+C` 或 `SIGTERM` 优雅退出。可加 `--no-move-done` 只测不移动。

推荐用 systemd 托管（崩溃自动重启）：

```ini
[Unit]
Description=e2e inference daemon
After=network.target

[Service]
ExecStart=/root/my_FILE/cpp_pipeline/make_cpp/cpp_work/infer_e2e_b8_daemon \
          --source /path/to/images --engine /path/to/engine --batch 8
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 三、相对原 main.cpp 的健壮性/边界加固（均已用 `[改进]` 标注）

1. **尺寸下界**：新增 `MIN_HW=128`，小于 128 的图自动放大，避免触发引擎 `setInputShape`
   越界崩溃（原版只处理 >1920 的缩小）。
2. **致命错误改为 `_Exit(1)` 而非 `exit`**：多线程流水线里某级崩掉，其余线程会永久阻塞在
   队列上造成 join 死锁；`_Exit` 直接终止进程（OS 回收 GPU 内存），并让调度方靠退出码判断失败。
3. **单张图失败只跳过不退出**：解码/缩放异常、病态超大文件（>64MiB 防 OOM）、无法读取，
   均告警后跳过，不影响整批。
4. **类别下标安全取模**：C++ 负数 `%` 仍为负数，原版 `int(d[5])%6` 遇到异常类别号会越界；
   改为 `cls_idx()` 保证落在 `[0,5]`（与 Python 的 `%` 行为对齐）。
5. **消除数据竞争**：消费线程 printf 里原版无锁读 `st->dets`，现改为锁内一并读取。
6. **参数边界校验**：`--batch [1,8]`、`--workers/--consumers >=1`、`--conf/--iou [0,1]`。
7. **落盘路径与错误检查**：开 `--save-*` 未给 `--out` 时自动落到 source 同级结果目录
   （原版会写到当前工作目录）；`ofstream`/`imwrite` 失败显式告警。
8. **setInputShape 返回值检查**：越界会返回 false，原版忽略后继续导致崩溃。
9. **目录遍历/`--source` 容错**：权限异常、符号链接不致命。
10. **daemon 额外**：SIGINT/SIGTERM 优雅停机、可中断睡眠、`--poll-ms` 下限防忙等、
     `--no-move-done` 开关、默认 `source/done`。

---

## 四、需核对项（重要，沿袭自 Python 版，非本次引入）

### 1. 坐标映射方向

`fit_image` 里 `scale = 原图最大边 / 目标边`，`filter_dets` 里对坐标做 `/= scale`。
这套语义是从 `infer_e2e_dyn_b8.py` **原样**继承的（C++ 与 Python 行为一致）。

请务必用真实标注核对两件事：

- 放大/缩小两种情况下，`/= scale` 得到的坐标是否映射回了原图尺度；
- 该 e2e 模型内部是否已经做了 letterbox/resize（ONNX 里确有 `Pad`/`Resize` 节点），
  若是，外部再做一次 letterbox + scale 校正可能冗余甚至方向相反。

> 若核对发现方向反了，正确做法应是把 `filter_dets` 的 `/= scale` 改为 `*= scale`
> （或统一改 `fit_image` 的 scale 定义），但**必须与 Python 版同步改**并重新验证真值，
> 不要只改 C++。

### 2. OpenCV 版本差异

C++ 用系统 OpenCV 4.5d，Python 用 opencv-python 4.10，画框/文字/编码有细微差异，不影响
检测数值，但像素级比对结果图时会不一致。

### 3. 运行时依赖 LD_LIBRARY_PATH

`infer_e2e_b8_cpp` 链接的 `libcudart.so.12` 实际会解析到 pip 的 CUDA 12.8 运行时（靠
LD_LIBRARY_PATH 优先于 RUNPATH）。在干净的 shell（无 LD_LIBRARY_PATH）里跑会回退到系统
12.4，与 TRT 10.8 混用。部署时请确保 `LD_LIBRARY_PATH` 含
`.../tensorrt_libs` 与 `.../nvidia/cuda_runtime/lib`（本机默认已设）。
