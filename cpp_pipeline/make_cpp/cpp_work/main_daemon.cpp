/*
 * main_daemon.cpp — e2e INT8 引擎 C++ 常驻推理（方案 B：内嵌目录监听）
 *
 * 定位：常驻 daemon —— 进程不退出，自己轮询图片目录，发现新图就推理，推理完移到 done/，
 *       继续等下一批。引擎只加载一次、显存/锁页内存常热，适合高频、秒级到达的图片流。
 *
 * 与 main.cpp 共用同一套四级流水线（解码线程池 → 攒批 → GPU → 消费），只是把 main()
 * 换成了「轮询 + 分批处理 + 移走已处理」的外层循环，并加了信号优雅停机。
 *
 * 健壮性要点（与 main.cpp 一致，见各处 [改进]/[注意]/[需核对] 标注）：
 *   - 单张图失败只跳过、不退出；致命 CUDA 错误才 _Exit(1)，交给 supervisor(systemd) 重启；
 *   - SIGINT/SIGTERM 优雅停机：处理完当前这一轮再退出；
 *   - 移走已处理文件，天然去重，且中途到达的新文件只会在下一轮被扫到（无竞态）；
 *   - 病态超大文件/尺寸越界/参数越界均已拦截。
 *
 * 用法:
 *   ./infer_e2e_b8_daemon --source 图片目录 --engine xxx.engine [--batch 8]
 *       [--workers 3] [--consumers 2] [--poll-ms 1000] [--done done子目录]
 *       [--no-move-done] [--save-txt] [--save-img] [--conf 0.45] [--iou 0.65] [--out 结果目录]
 *
 * 编译: bash build.sh
 */

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>      // [改进] 信号处理，优雅停机
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <cstring>

namespace fs = std::filesystem;

// [改进] 优雅停机标志：信号处理函数里只能安全操作 volatile sig_atomic_t，
//        不能用 std::atomic<bool>（不保证 async-signal-safe）。
static volatile std::sig_atomic_t g_stop = 0;
static void handle_signal(int) { g_stop = 1; }

#define FATAL(msg) do {                                                     \
    fprintf(stderr, "[FATAL] %s @%s:%d\n", (msg), __FILE__, __LINE__);      \
    fflush(stderr);                                                         \
    std::_Exit(1);                                                          \
} while (0)

#define CUDA_CHECK(x) do {                                                  \
    cudaError_t e_ = (x);                                                   \
    if (e_ != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s @%s:%d: %s\n", #x, __FILE__,         \
                __LINE__, cudaGetErrorString(e_));                          \
        fflush(stderr);                                                     \
        std::_Exit(1);                                                      \
    }                                                                       \
} while (0)

// -------------------------- 运行时常量 --------------------------
static constexpr int   MAX_BATCH = 8;
static constexpr int   MIN_HW    = 128;    // 引擎最小动态维
static constexpr int   MAX_HW    = 1920;
static constexpr int   MAX_DETS  = 300;
static constexpr int   SLOTS     = 4;
static constexpr long long MAX_FILE_BYTES = 64LL << 20;

static const char*     IN_NAME   = "image_raw";
static const char*     OUT_NAME  = "detections";

static const char*     CLASS_NAMES[] = {"open", "short", "mousebite",
                                        "spur", "copper", "pinhole"};
static constexpr int   NUM_CLASSES = 6;

// ---------------- TRT Logger ----------------
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static TrtLogger g_logger;

// ---------------- 有界阻塞队列 ----------------
template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t cap) : cap_(cap) {}
    void push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return;
        q_.push(std::move(v));
        lk.unlock();
        cv_.notify_one();
    }
    bool pop(T& v) {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;
        v = std::move(q_.front());
        q_.pop();
        lk.unlock();
        cv_.notify_one();
        return true;
    }
    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_.notify_all();
    }
private:
    std::queue<T> q_;
    size_t cap_;
    bool closed_ = false;
    std::mutex m_;
    std::condition_variable cv_;
};

// ---------------- 计数信号量 ----------------
class Semaphore {
public:
    explicit Semaphore(int cnt) : cnt_(cnt) {}
    void acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return cnt_ > 0; });
        --cnt_;
    }
    void release() {
        { std::lock_guard<std::mutex> lk(m_); ++cnt_; }
        cv_.notify_one();
    }
private:
    int cnt_;
    std::mutex m_;
    std::condition_variable cv_;
};

// ---------------- 数据结构 ----------------
struct Frame {
    std::string name;
    cv::Mat img;
    float scale = 1.0f;
};

struct Batch {
    int slot = 0;
    int n = 0, h = 0, w = 0;
    std::vector<Frame> frames;
};

struct Results {
    std::vector<Frame> frames;
    std::vector<float> dets;
};

struct Stats {
    std::mutex m;
    long n = 0, dets = 0;
};

struct Args {
    fs::path source, engine, out_dir, done_dir;
    int batch = 8, workers = 7, consumers = 4;
    int poll_ms = 1000;
    float conf = 0.45f, iou = 0.65f;
    bool save_txt = false, save_img = false;
    bool move_done = true;
};

// ---------------- 槽位资源池 ----------------
struct Slot {
    uint8_t* pin_in = nullptr;
    void*    dev_in = nullptr;
    size_t   cap_bytes = 0;
};
static void ensure_slot(Slot& s, size_t bytes) {
    if (s.cap_bytes >= bytes) return;
    if (s.pin_in) { CUDA_CHECK(cudaFreeHost(s.pin_in)); CUDA_CHECK(cudaFree(s.dev_in)); }
    CUDA_CHECK(cudaMallocHost(&s.pin_in, bytes));
    CUDA_CHECK(cudaMalloc(&s.dev_in, bytes));
    s.cap_bytes = bytes;
}

// ---------------- 工具 ----------------
static cv::Mat fit_image(cv::Mat img, float& scale) {
    scale = 1.0f;
    if (img.empty() || img.rows <= 0 || img.cols <= 0) return img;
    int m = std::max(img.rows, img.cols);
    int target = m;
    if      (m < MIN_HW)  target = MIN_HW;
    else if (m > MAX_HW)  target = MAX_HW;
    if (target == m) return img;
    float s = float(m) / float(target);
    cv::resize(img, img, cv::Size(int(img.cols / s + 0.5), int(img.rows / s + 0.5)));
    scale = s;
    return img;
}

static int cls_idx(float c) {
    int k = int(c);
    k %= NUM_CLASSES;
    if (k < 0) k += NUM_CLASSES;
    return k;
}

static void filter_dets(const float* d300x6, int orig_w, int orig_h, float scale,
                        std::vector<std::array<float, 6>>& out) {
    for (int i = 0; i < MAX_DETS; ++i) {
        const float* d = d300x6 + i * 6;
        if (d[4] <= 0.f) continue;
        float x1 = d[0], y1 = d[1], x2 = d[2], y2 = d[3];
        if (scale != 1.0f) { x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale; }
        if (x2 <= 0 || y2 <= 0 || x1 >= orig_w || y1 >= orig_h) continue;
        out.push_back({x1, y1, x2, y2, d[4], d[5]});
    }
}

static const cv::Scalar COLORS[] = {{60, 60, 220}, {60, 180, 220}, {220, 60, 60},
                                    {60, 220, 60}, {220, 60, 200}, {0, 200, 255}};
static cv::Mat draw_dets(cv::Mat img, const std::vector<std::array<float, 6>>& dets) {
    for (const auto& d : dets) {
        int k = cls_idx(d[5]);
        const cv::Scalar& color = COLORS[k];
        cv::rectangle(img, cv::Point(int(d[0]), int(d[1])),
                      cv::Point(int(d[2]), int(d[3])), color, 2);
        char tag[64];
        snprintf(tag, sizeof(tag), "%s %.2f", CLASS_NAMES[k], d[4]);
        int base = 0;
        cv::Size ts = cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);
        cv::Point org(int(d[0]), std::max(0, int(d[1]) - ts.height - 6));
        cv::rectangle(img, org, org + cv::Point(ts.width, ts.height + 4), color, -1);
        cv::putText(img, tag, {org.x, org.y + ts.height},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
    }
    return img;
}

// ---------------- 引擎封装 ----------------
class Engine {
public:
    Engine(const std::string& path, float conf, float iou) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "无法打开引擎: %s\n", path.c_str()); std::_Exit(1); }
        f.seekg(0, std::ios::end);
        size_t sz = f.tellg();
        f.seekg(0);
        std::vector<char> blob(sz);
        f.read(blob.data(), sz);

        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), sz));
        if (!engine_) { fprintf(stderr, "引擎反序列化失败: %s\n", path.c_str()); std::_Exit(1); }
        ctx_.reset(engine_->createExecutionContext());
        if (!ctx_) { fprintf(stderr, "创建执行上下文失败\n"); std::_Exit(1); }
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_, 0));

        CUDA_CHECK(cudaMalloc(&dev_out_, MAX_BATCH * MAX_DETS * 6 * 4));
        out_host_.resize(MAX_BATCH * MAX_DETS * 6);
        CUDA_CHECK(cudaMalloc(&dev_iou_, 4));
        CUDA_CHECK(cudaMalloc(&dev_score_, 4));
        set_scalar("iou_thresh", dev_iou_, iou);
        set_scalar("score_thresh", dev_score_, conf);
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    ~Engine() {
        for (auto& kv : slots_) {
            if (kv.second.pin_in) { cudaFreeHost(kv.second.pin_in); cudaFree(kv.second.dev_in); }
        }
        cudaFree(dev_out_); cudaFree(dev_iou_); cudaFree(dev_score_);
        if (ev_) cudaEventDestroy(ev_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    Slot& get_slot(int idx) { return slots_[idx]; }
    const float* out_host() const { return out_host_.data(); }

    void submit(Batch& b) {
        Slot& s = slots_[b.slot];
        ensure_slot(s, size_t(b.n) * b.h * b.w * 3);
        if (!(b.n == last_n_ && b.h == last_h_ && b.w == last_w_)) {
            if (!ctx_->setInputShape(IN_NAME, nvinfer1::Dims4{b.n, b.h, b.w, 3}))
                FATAL("setInputShape 失败：输入尺寸超出引擎 profile 范围");
            last_n_ = b.n; last_h_ = b.h; last_w_ = b.w;
        }
        ctx_->setTensorAddress(IN_NAME, s.dev_in);
        ctx_->setTensorAddress(OUT_NAME, dev_out_);
        CUDA_CHECK(cudaMemcpyAsync(s.dev_in, s.pin_in, size_t(b.n) * b.h * b.w * 3,
                                   cudaMemcpyHostToDevice, stream_));
        if (!ctx_->enqueueV3(stream_)) {
            FATAL("enqueueV3 推理失败");
        }
        CUDA_CHECK(cudaEventRecord(ev_, stream_));
    }

    void wait() {
        CUDA_CHECK(cudaEventSynchronize(ev_));
        CUDA_CHECK(cudaMemcpy(out_host_.data(), dev_out_,
                              out_host_.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

private:
    void set_scalar(const char* name, void* dev, float v) {
        nvinfer1::Dims d1{};
        d1.nbDims = 1; d1.d[0] = 1;
        ctx_->setInputShape(name, d1);
        ctx_->setTensorAddress(name, dev);
        CUDA_CHECK(cudaMemcpyAsync(dev, &v, 4, cudaMemcpyHostToDevice, stream_));
    }

    std::unique_ptr<nvinfer1::IRuntime>          runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t  ev_     = nullptr;
    void*  dev_out_   = nullptr;
    void*  dev_iou_   = nullptr;
    void*  dev_score_ = nullptr;
    std::vector<float> out_host_;
    std::unordered_map<int, Slot> slots_;
    int  last_n_ = -1, last_h_ = -1, last_w_ = -1;
};

// ---------------- 流水线各级（与 main.cpp 相同）----------------
static void decoder_worker(BoundedQueue<std::string>& q_path,
                           BoundedQueue<Frame>& q_frame) {
    std::string path;
    while (q_path.pop(path)) {
        try {
            if (fs::file_size(path) > MAX_FILE_BYTES) {
                fprintf(stderr, "[warn] 文件过大跳过: %s\n", path.c_str());
                continue;
            }
            cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
            if (img.empty()) { fprintf(stderr, "[warn] 无法读取: %s\n", path.c_str()); continue; }
            Frame fr;
            fr.name = fs::path(path).filename().string();
            fr.img = fit_image(std::move(img), fr.scale);
            if (fr.img.empty()) { fprintf(stderr, "[warn] 空图跳过: %s\n", path.c_str()); continue; }
            q_frame.push(std::move(fr));
        } catch (const std::exception& e) {
            fprintf(stderr, "[warn] 解码异常跳过 %s: %s\n", path.c_str(), e.what());
        }
    }
    q_frame.push(Frame{});
}

static void assembler_worker(BoundedQueue<Frame>& q_frame, BoundedQueue<Batch>& q_gpu,
                             Engine& eng, Semaphore& slots_free, int batch, int workers) {
    std::vector<Frame> group;
    int gh = 0, gw = 0;
    int done = 0;
    auto flush = [&](std::vector<Frame>& g) {
        if (g.empty()) return;
        static thread_local int next_slot = 0;
        Batch b;
        b.slot = next_slot;
        next_slot = (next_slot + 1) % SLOTS;
        b.n = int(g.size());
        b.h = g[0].img.rows;
        b.w = g[0].img.cols;
        b.frames = std::move(g);
        g.clear();
        slots_free.acquire();
        Slot& s = eng.get_slot(b.slot);
        ensure_slot(s, size_t(b.n) * b.h * b.w * 3);
        cv::Mat canvas0(b.h, b.w, CV_8UC3, s.pin_in);
        canvas0.setTo(cv::Scalar(114, 114, 114));
        for (int i = 0; i < b.n; ++i) {
            cv::Mat dst(b.h, b.w, CV_8UC3, s.pin_in + size_t(i) * b.h * b.w * 3);
            const cv::Mat& im = b.frames[i].img;
            im.copyTo(dst(cv::Rect(0, 0, im.cols, im.rows)));
        }
        q_gpu.push(std::move(b));
    };

    Frame fr;
    while (done < workers && q_frame.pop(fr)) {
        if (fr.name.empty()) { ++done; continue; }
        if (!group.empty() && (fr.img.rows != gh || fr.img.cols != gw))
            flush(group);
        if (group.empty()) { gh = fr.img.rows; gw = fr.img.cols; }
        group.push_back(std::move(fr));
        if (int(group.size()) >= batch) flush(group);
    }
    flush(group);
    q_gpu.close();
}

static void gpu_worker(Engine& eng, BoundedQueue<Batch>& q_gpu,
                       BoundedQueue<Results>& q_out, Semaphore& slots_free) {
    Batch b;
    while (q_gpu.pop(b)) {
        eng.submit(b);
        eng.wait();
        Results r;
        r.frames = std::move(b.frames);
        r.dets.assign(eng.out_host(), eng.out_host() + size_t(b.n) * MAX_DETS * 6);
        q_out.push(std::move(r));
        slots_free.release();
    }
    q_out.close();
}

static void consumer_worker(BoundedQueue<Results>& q_out, Args* a, Stats* st) {
    Results r;
    while (q_out.pop(r)) {
        for (size_t i = 0; i < r.frames.size(); ++i) {
            const Frame& fr = r.frames[i];
            std::vector<std::array<float, 6>> dets;
            filter_dets(r.dets.data() + i * MAX_DETS * 6, fr.img.cols, fr.img.rows,
                        fr.scale, dets);
            long n, ndets;
            {
                std::lock_guard<std::mutex> lk(st->m);
                st->n += 1;
                st->dets += long(dets.size());
                n = st->n;
                ndets = st->dets;
            }
            if (n % 200 == 0) { printf("  ... %ld 张, %ld 框\n", n, ndets); fflush(stdout); }
            if (a->save_txt || a->save_img) {
                fs::path base = a->out_dir / fs::path(fr.name).stem();
                if (a->save_txt) {
                    std::ofstream f(base.string() + ".txt");
                    if (!f.is_open()) { fprintf(stderr, "[warn] 无法写 txt: %s.txt\n", base.c_str()); }
                    else {
                        for (auto& d : dets)
                            f << CLASS_NAMES[cls_idx(d[5])] << " " << d[4] << " "
                              << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << "\n";
                    }
                }
                if (a->save_img) {
                    cv::Mat img = draw_dets(fr.img.clone(), dets);
                    if (!cv::imwrite(base.string() + ".jpg", img))
                        fprintf(stderr, "[warn] 无法写图: %s.jpg\n", base.c_str());
                }
            }
        }
    }
}

// [改进] 收集目录内图片（排序），容错目录遍历。
static std::vector<std::string> gather_images(const fs::path& source) {
    std::vector<std::string> paths;
    try {
        for (auto& p : fs::directory_iterator(source)) {
            if (!p.is_regular_file()) continue;
            std::string e = p.path().extension().string();
            std::transform(e.begin(), e.end(), e.begin(), ::tolower);
            if (e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp")
                paths.push_back(p.path().string());
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "[warn] 遍历目录失败 %s: %s\n", source.c_str(), e.what());
    }
    std::sort(paths.begin(), paths.end());
    return paths;
}

// [改进] 跑一轮：对给定图片列表走完四级流水线，返回本轮处理张数。
//        引擎/槽位在调用方（daemon 主循环）只建一次，这里每轮只是重建线程。
static void run_round(Engine& eng, const std::vector<std::string>& paths,
                      const Args& a, Stats& stats) {
    Semaphore slots_free(SLOTS);
    BoundedQueue<std::string> q_path(1024);
    BoundedQueue<Frame>       q_frame(64);
    BoundedQueue<Batch>       q_gpu(SLOTS);
    BoundedQueue<Results>     q_out(2 * a.consumers);

    for (auto& p : paths) q_path.push(p);
    q_path.close();

    std::vector<std::thread> pool;
    for (int i = 0; i < a.workers; ++i)
        pool.emplace_back(decoder_worker, std::ref(q_path), std::ref(q_frame));
    pool.emplace_back(assembler_worker, std::ref(q_frame), std::ref(q_gpu),
                      std::ref(eng), std::ref(slots_free), a.batch, a.workers);
    pool.emplace_back(gpu_worker, std::ref(eng), std::ref(q_gpu),
                      std::ref(q_out), std::ref(slots_free));
    for (int i = 0; i < a.consumers; ++i)
        pool.emplace_back(consumer_worker, std::ref(q_out), const_cast<Args*>(&a), &stats);

    for (auto& t : pool) t.join();
}

// ---------------- 主流程 ----------------
int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--source")    a.source  = next();
        else if (k == "--engine")    a.engine  = next();
        else if (k == "--out")       a.out_dir = next();
        else if (k == "--done")      a.done_dir = next();
        else if (k == "--batch")     a.batch = std::stoi(next());
        else if (k == "--workers")   a.workers = std::stoi(next());
        else if (k == "--consumers") a.consumers = std::stoi(next());
        else if (k == "--poll-ms")   a.poll_ms = std::stoi(next());
        else if (k == "--conf")      a.conf = std::stof(next());
        else if (k == "--iou")       a.iou = std::stof(next());
        else if (k == "--save-txt")  a.save_txt = true;
        else if (k == "--save-img")  a.save_img = true;
        else if (k == "--no-move-done") a.move_done = false;
        else { fprintf(stderr, "未知参数: %s\n", k.c_str()); return 1; }
    }
    if (a.source.empty() || a.engine.empty()) {
        fprintf(stderr, "用法: %s --source 图片目录 --engine xxx.engine [--batch 8] ...\n", argv[0]);
        return 1;
    }

    if (a.batch < 1 || a.batch > MAX_BATCH) { fprintf(stderr, "--batch 必须在 [1,%d]\n", MAX_BATCH); return 1; }
    if (a.workers < 1)  { fprintf(stderr, "--workers 必须 >= 1\n"); return 1; }
    if (a.consumers < 1){ fprintf(stderr, "--consumers 必须 >= 1\n"); return 1; }
    if (a.poll_ms < 100) a.poll_ms = 100;                    // [改进] 轮询间隔下限，防止忙等
    if (a.conf < 0.f || a.conf > 1.f) { fprintf(stderr, "--conf 必须在 [0,1]\n"); return 1; }
    if (a.iou  < 0.f || a.iou  > 1.f) { fprintf(stderr, "--iou 必须在 [0,1]\n"); return 1; }

    // 默认 done 目录：<source>/done
    if (a.move_done && a.done_dir.empty())
        a.done_dir = a.source / "done";
    // [注意] done 目录必须位于 source 之外或非图片扩展名，否则会被 gather_images 反复扫到。
    if (a.move_done) fs::create_directories(a.done_dir);
    if ((a.save_txt || a.save_img) && a.out_dir.empty())
        a.out_dir = a.source.parent_path() / (a.source.filename().string() + "_result");
    if ((a.save_txt || a.save_img) && !a.out_dir.empty())
        fs::create_directories(a.out_dir);

    // 引擎只建一次，常驻热复用
    Engine eng(a.engine.string(), a.conf, a.iou);
    Stats stats;

    // [改进] 优雅停机：SIGINT/SIGTERM 置位 g_stop，处理完当前轮后退出。
    std::signal(SIGINT,  handle_signal);
    std::signal(SIGTERM, handle_signal);

    printf("[daemon] 监听目录: %s (poll=%dms, batch=%d, done=%s)\n",
           a.source.c_str(), a.poll_ms, a.batch,
           a.move_done ? a.done_dir.c_str() : "(不移动)");

    long rounds = 0;
    while (!g_stop) {
        std::vector<std::string> paths = gather_images(a.source);
        if (paths.empty()) {
            // [改进] 可中断睡眠：分小段睡，收到信号能及时醒来退出。
            for (int slept = 0; slept < a.poll_ms && !g_stop; slept += 100)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        ++rounds;
        printf("[round %ld] 发现 %zu 张新图\n", rounds, paths.size());
        fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();

        run_round(eng, paths, a, stats);   // [注意] 若此轮内发生致命 CUDA 错误会 _Exit(1)

        double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("[round %ld] 完成 %zu 张, 本轮 %.2fs (累计 %ld 张 / %ld 框)\n",
               rounds, paths.size(), dt, stats.n, stats.dets);
        fflush(stdout);

        // [改进] 处理完（含跳过的坏图）统一移到 done/，天然去重且不丢新到文件。
        if (a.move_done) {
            for (auto& p : paths) {
                try {
                    fs::rename(p, a.done_dir / fs::path(p).filename());
                } catch (const std::exception& e) {
                    fprintf(stderr, "[warn] 移动失败 %s: %s\n", p.c_str(), e.what());
                }
            }
        }
    }

    printf("[daemon] 收到停机信号，退出。累计 %ld 张图 / %ld 框 / %ld 轮\n",
           stats.n, stats.dets, rounds);
    return 0;
}
