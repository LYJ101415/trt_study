#!/bin/bash
# ============================================================================
# run_loop.sh — 方案 A：外部调度器（常驻监视 + 反复拉起单发 worker）
#
# 思路：main.cpp 本身仍是「跑一次就退」的单发 worker；本脚本作为 supervisor 常驻，
#       监视 SOURCE 目录，一旦有新图片就把它们移到 staging 目录、拉起 worker 处理，
#       成功后移到 done/，失败则退回 SOURCE 下次重试。
#
# 关键设计：
#   - 单实例锁（flock）：防止同时起多个本脚本实例并发处理同一批图；
#   - staging 目录去重：处理前先把图移出 SOURCE，处理完再进 done/；
#       中途新到的文件留在 SOURCE，下一轮再处理 —— 天然无竞态、不会重复处理；
#   - 失败退回：worker 非 0 退出（如 CUDA 错误）时，图片退回 SOURCE，等下次重试；
#   - 崩溃自动重启：外层 while 死循环，worker 崩了不影响本脚本继续下一轮；
#   - 可中断等待：优先 inotifywait 事件触发，无 inotifywait 时退化为轮询 sleep。
#
# 用法:  bash run_loop.sh            （后台常驻建议配合 systemd / nohup / supervisor）
# ============================================================================

set -u   # 未定义变量报错（不设 -e，因为要手动处理 worker 的返回码）

# ============================ 配置区（按需修改） ============================
SOURCE="/root/autodl-tmp/datasets/Data_DeepPCB_YOLO/images/test"   # 监视的图片目录
ENGINE="/root/my_FILE/models/yolov8_e2e_dyn.engine"               # 引擎文件
BIN="./infer_e2e_b8_cpp"                                          # 单发 worker（main.cpp 编译产物）
OUT="/root/my_FILE/infer_images"                                  # 结果目录（仅 --save-* 时用）
WORK_DIR="$(cd "$(dirname "$0")" && pwd)"                         # 本脚本目录
STAGE="$SOURCE/.staging"                                          # 处理中的暂存目录
DONE="$SOURCE/done"                                               # 已处理目录
LOCKFILE="$WORK_DIR/run_loop.lock"
LOGFILE="$WORK_DIR/run_loop.log"

BATCH=8
WORKERS=7
CONSUMERS=4
CONF=0.45
IOU=0.65
POLL_SEC=3                                                        # 轮询/等待超时(秒)
SAVE_IMG=0                                                        # 1=保存画框图
SAVE_TXT=0                                                        # 1=保存检测 txt
# ============================================================================

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGFILE"; }

# 单实例锁：用文件描述符 9 持有 flock，脚本存活期间一直占用
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    log "已有实例在运行，本实例退出"
    exit 0
fi

cleanup() {
    log "收到退出信号，清理并退出"
    rm -f "$LOCKFILE"
    exit 0
}
trap cleanup INT TERM

# 等待新图片：优先 inotifywait 事件触发，否则退化为轮询 sleep
wait_for_work() {
    if command -v inotifywait >/dev/null 2>&1; then
        # close_write=文件写完关闭；moved_to=原子改名进目录（生产者常用"写临时文件再 mv"）
        inotifywait -q -t "$POLL_SEC" -e close_write -e moved_to "$SOURCE" 2>/dev/null
    else
        sleep "$POLL_SEC"
    fi
}

# 收集 SOURCE 下所有图片文件（不递归；用 find -exec 处理含空格文件名）
count_images() {
    find "$SOURCE" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
        2>/dev/null | wc -l
}

log "===== run_loop 启动：监听 $SOURCE (间隔 ${POLL_SEC}s) ====="

mkdir -p "$STAGE" "$DONE" "$(dirname "$LOGFILE")"

while true; do
    n=$(count_images)
    if [ "$n" -eq 0 ]; then
        wait_for_work
        continue
    fi

    log "发现 ${n} 张新图，移入 staging 并处理"

    # 把 SOURCE 里的图原子移到 staging（新到的文件不会进来，下一轮再处理）
    find "$SOURCE" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
        -exec mv {} "$STAGE"/ \; 2>/dev/null

    # 组装 worker 参数
    args=(--source "$STAGE" --engine "$ENGINE" --batch "$BATCH" \
          --workers "$WORKERS" --consumers "$CONSUMERS" --conf "$CONF" --iou "$IOU")
    [ "$SAVE_IMG" = 1 ] && args+=(--save-img)
    [ "$SAVE_TXT" = 1 ] && args+=(--save-txt)
    [ -n "$OUT" ] && args+=(--out "$OUT")

    # 拉起单发 worker（每次全新进程，进程级资源由 OS 回收，天然无累积）
    "$BIN" "${args[@]}"
    rc=$?

    if [ $rc -eq 0 ]; then
        log "本轮处理成功，${n} 张移入 done/"
        find "$STAGE" -maxdepth 1 -type f -exec mv {} "$DONE"/ \; 2>/dev/null
    else
        log "本轮 worker 退出码=$rc（失败），图片退回 SOURCE 待重试"
        find "$STAGE" -maxdepth 1 -type f -exec mv {} "$SOURCE"/ \; 2>/dev/null
        # 失败后稍等再重试，避免死循环高频重试
        sleep "$POLL_SEC"
    fi
done
