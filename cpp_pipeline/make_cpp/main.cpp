/*
 * main.cpp — e2e INT8 引擎 C++ 四级流水线推理 (batch=8)
 *
 * 与 Python 版 (infer_e2e_dyn_b8.py) 同构, 消除 Python GIL/解释器调度开销:
 *   解码线程池 → 攒批线程(填锁页画布) → GPU线程(异步H2D + enqueueV3) → 消费线程
 *   槽位复用由 counting semaphore 控制(攒批 acquire, GPU线程拷完输出 release)
 *
 * 用法:
 *   ./infer_e2e_b8_cpp --source 图片目录 --engine xxx.engine [--batch 8]
 *       [--workers 3] [--consumers 2] [--save-txt] [--save-img] [--conf 0.45] [--iou 0.65]
 *       [--out 结果目录]
 *
 * 编译: bash build.sh
 */

// TensorRT 相关头文件 - 提供推理引擎接口
#include <NvInfer.h>           // TensorRT 核心接口（引擎、运行时、执行上下文）
#include <cuda_runtime.h>     // CUDA 运行时 API（内存管理、流、事件）

// OpenCV 相关头文件 - 图像处理
#include <opencv2/core.hpp>     // OpenCV 核心数据结构（Mat、Point等）
#include <opencv2/imgproc.hpp>   // 图像处理函数（缩放、颜色空间转换等）
#include <opencv2/imgcodecs.hpp> // 图像编解码（imread、imwrite等）

// C++ 标准库头文件
#include <algorithm>    // 算法（sort、max、min、transform等）
#include <array>        // 固定大小数组（ COLORS 数组）
#include <atomic>       // 原子操作（线程安全计数器）
#include <chrono>       // 时间处理（性能计时）
#include <condition_variable> // 条件变量（线程同步）
#include <filesystem>   // 文件系统操作（路径处理、目录遍历）
#include <fstream>      // 文件流（读取引擎文件、写入检测结果）
#include <mutex>        // 互斥锁（线程同步）
#include <queue>        // 队列（线程间通信）
#include <string>       // 字符串处理
#include <thread>       // 线程（多线程并行处理）
#include <unordered_map> // 哈希映射（槽位管理）
#include <vector>       // 动态数组（存储检测结果、帧等）
#include <cstdio>       // C标准输入输出（printf、fprintf）
#include <cstring>      // C字符串处理（snprintf）

// 使用 std::filesystem 命名空间的别名简化路径操作
namespace fs = std::filesystem;

/*
 * CUDA错误检查宏 - 在每个CUDA API调用后检查错误
 * 如果CUDA操作失败，打印详细的错误信息并退出程序
 * 参数 x: 要检查的CUDA API调用
 */
#define CUDA_CHECK(x)                                                        \
    do {                                                                     \
        cudaError_t e_ = (x);                                                \
        if (e_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s @%s:%d: %s\n", #x, __FILE__,      \
                    __LINE__, cudaGetErrorString(e_));                       \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// -------------------------- 运行时常量配置 --------------------------
// 最大批处理大小 - 受限于TensorRT引擎配置和GPU内存
static constexpr int   MAX_BATCH = 8;
// 图像最大高度/宽度 - 用于防止OOM，大图会被缩放
static constexpr int   MAX_HW    = 1920;
// 单张图像最大检测框数 - 受限于模型输出维度
static constexpr int   MAX_DETS  = 300;
// 槽位数量 - 用于批处理缓冲区复用
static constexpr int   SLOTS     = 4;

// -------------------------- 模型输入输出名称 --------------------------
// 模型输入张量名称 - 接收原始图像数据
static const char*     IN_NAME   = "image_raw";
// 模型输出张量名称 - 输出检测结果
static const char*     OUT_NAME  = "detections";

// -------------------------- 分类类别名称 --------------------------
// PCB缺陷检测的6个类别名称
static const char*     CLASS_NAMES[] = {"open", "short", "mousebite",
                                        "spur", "copper", "pinhole"};

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
    bool pop(T& v) {   // 队列关闭且取空后返回 false
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

// ---------------- 计数信号量 (槽位回收) ----------------
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
    cv::Mat img;          // 已 fit 过的图 (BGR, uint8)
    float scale = 1.0f;   // fit 缩放系数, 用于坐标映射回原图
};

struct Batch {
    int slot = 0;
    int n = 0, h = 0, w = 0;
    std::vector<Frame> frames;
};

struct Results {
    std::vector<Frame> frames;
    std::vector<float> dets;   // n * MAX_DETS * 6, 每帧 300x6 连续
};

struct Stats {
    std::mutex m;
    long n = 0, dets = 0;
};

struct Args {
    fs::path source, engine, out_dir;
    int batch = 8, workers = 7, consumers = 4;
    float conf = 0.45f, iou = 0.65f;
    bool save_txt = false, save_img = false;
};

// ---------------- 槽位资源池 ----------------
struct Slot {
    uint8_t* pin_in = nullptr;   // 锁页输入画布
    void*    dev_in = nullptr;   // 设备输入
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
    int m = std::max(img.rows, img.cols);
    if (m <= MAX_HW) return img;
    float s = float(m) / MAX_HW;
    cv::resize(img, img, cv::Size(int(img.cols / s + 0.5), int(img.rows / s + 0.5)));
    scale = s;
    return img;
}

static void filter_dets(const float* d300x6, int orig_w, int orig_h, float scale,
                        std::vector<std::array<float, 6>>& out) {
    for (int i = 0; i < MAX_DETS; ++i) {
        const float* d = d300x6 + i * 6;
        if (d[4] <= 0.f) continue;                       // 空行
        float x1 = d[0], y1 = d[1], x2 = d[2], y2 = d[3];
        if (scale != 1.0f) { x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale; }
        if (x2 <= 0 || y2 <= 0 || x1 >= orig_w || y1 >= orig_h) continue;
        out.push_back({x1, y1, x2, y2, d[4], d[5]});
    }
}

// ---------------- 标注绘制 (与 Python 版同样式) ----------------
static const cv::Scalar COLORS[] = {{60, 60, 220}, {60, 180, 220}, {220, 60, 60},
                                    {60, 220, 60}, {220, 60, 200}, {0, 200, 255}};
static cv::Mat draw_dets(cv::Mat img, const std::vector<std::array<float, 6>>& dets) {
    for (const auto& d : dets) {
        int k = int(d[5]) % 6;
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
        if (!f) { fprintf(stderr, "无法打开引擎: %s\n", path.c_str()); exit(1); }
        f.seekg(0, std::ios::end);
        size_t sz = f.tellg();
        f.seekg(0);
        std::vector<char> blob(sz);
        f.read(blob.data(), sz);

        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), sz));
        ctx_.reset(engine_->createExecutionContext());
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_, 0));

        // 输出与阈值: 一次分配/上传, 地址固定
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
            ctx_->setInputShape(IN_NAME, nvinfer1::Dims4{b.n, b.h, b.w, 3});
            last_n_ = b.n; last_h_ = b.h; last_w_ = b.w;
        }
        ctx_->setTensorAddress(IN_NAME, s.dev_in);
        ctx_->setTensorAddress(OUT_NAME, dev_out_);
        CUDA_CHECK(cudaMemcpyAsync(s.dev_in, s.pin_in, size_t(b.n) * b.h * b.w * 3,
                                   cudaMemcpyHostToDevice, stream_));
        if (!ctx_->enqueueV3(stream_)) {
            fprintf(stderr, "enqueueV3 failed\n");
            exit(1);
        }
        CUDA_CHECK(cudaEventRecord(ev_, stream_));
    }

    // 等待本批完成并拷回 (输出 ~57KB, 同步拷贝开销可忽略)
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

// ---------------- 流水线各级 ----------------
static void decoder_worker(BoundedQueue<std::string>& q_path,
                           BoundedQueue<Frame>& q_frame) {
    std::string path;
    while (q_path.pop(path)) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) { fprintf(stderr, "[warn] 无法读取: %s\n", path.c_str()); continue; }
        Frame fr;
        fr.name = fs::path(path).filename().string();
        fr.img = fit_image(std::move(img), fr.scale);
        q_frame.push(std::move(fr));
    }
    q_frame.push(Frame{});   // 退出哨兵: 空名帧, 攒批线程收齐 workers 个才算流结束
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
        slots_free.acquire();                        // 等槽位空闲(输出已被拷走)
        Slot& s = eng.get_slot(b.slot);
        ensure_slot(s, size_t(b.n) * b.h * b.w * 3);
        cv::Mat canvas0(b.h, b.w, CV_8UC3, s.pin_in);
        canvas0.setTo(cv::Scalar(114, 114, 114));    // letterbox 灰底
        for (int i = 0; i < b.n; ++i) {
            cv::Mat dst(b.h, b.w, CV_8UC3, s.pin_in + size_t(i) * b.h * b.w * 3);
            const cv::Mat& im = b.frames[i].img;
            im.copyTo(dst(cv::Rect(0, 0, im.cols, im.rows)));   // 左上对齐
        }
        q_gpu.push(std::move(b));
    };

    Frame fr;
    while (done < workers && q_frame.pop(fr)) {
        if (fr.name.empty()) { ++done; continue; }   // 哨兵
        if (!group.empty() && (fr.img.rows != gh || fr.img.cols != gw))
            flush(group);                            // 换尺寸: 先交出当前组
        if (group.empty()) { gh = fr.img.rows; gw = fr.img.cols; }
        group.push_back(std::move(fr));
        if (int(group.size()) >= batch) flush(group);
    }
    flush(group);                                    // 尾巴 (不足 batch 由动态 shape 吃下)
    q_gpu.close();
}

static void gpu_worker(Engine& eng, BoundedQueue<Batch>& q_gpu,
                       BoundedQueue<Results>& q_out, Semaphore& slots_free) {
    Batch b;
    while (q_gpu.pop(b)) {
        eng.submit(b);
        eng.wait();                                  // GPU 完成 + 结果已拷回主机
        Results r;
        r.frames = std::move(b.frames);
        r.dets.assign(eng.out_host(), eng.out_host() + size_t(b.n) * MAX_DETS * 6);
        q_out.push(std::move(r));
        slots_free.release();                        // 通知攒批线程: 该槽可复用
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
            long done;
            {   // 锁只保护计数器! 若罩住下面的写盘, 多消费线程会在 I/O 上完全串行
                std::lock_guard<std::mutex> lk(st->m);
                st->n += 1;
                st->dets += long(dets.size());
                done = st->n;
            }
            if (done % 200 == 0) printf("  ... %ld 张, %ld 框\n", done, st->dets);
            if (a->save_txt || a->save_img) {
                fs::path base = a->out_dir / fs::path(fr.name).stem();
                if (a->save_txt) {
                    std::ofstream f(base.string() + ".txt");
                    for (auto& d : dets)
                        f << CLASS_NAMES[int(d[5]) % 6] << " " << d[4] << " "
                          << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << "\n";
                }
                if (a->save_img) {
                    cv::Mat img = draw_dets(fr.img.clone(), dets);
                    cv::imwrite(base.string() + ".jpg", img);   // 与原图同名, 存到 --out 目录
                }
            }
        }
    }
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
        else if (k == "--batch")     a.batch = std::stoi(next());
        else if (k == "--workers")   a.workers = std::stoi(next());
        else if (k == "--consumers") a.consumers = std::stoi(next());
        else if (k == "--conf")      a.conf = std::stof(next());
        else if (k == "--iou")       a.iou = std::stof(next());
        else if (k == "--save-txt")  a.save_txt = true;
        else if (k == "--save-img")  a.save_img = true;
        else { fprintf(stderr, "未知参数: %s\n", k.c_str()); return 1; }
    }
    if (a.source.empty() || a.engine.empty()) {
        fprintf(stderr, "用法: %s --source 图片目录 --engine xxx.engine [--batch 8] ...\n", argv[0]);
        return 1;
    }
    a.batch = std::min(a.batch, MAX_BATCH);

    std::vector<std::string> paths;
    for (auto& p : fs::directory_iterator(a.source)) {
        if (!p.is_regular_file()) continue;
        std::string e = p.path().extension().string();
        std::transform(e.begin(), e.end(), e.begin(), ::tolower);
        if (e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp")
            paths.push_back(p.path().string());
    }
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) { fprintf(stderr, "目录中没有图片: %s\n", a.source.c_str()); return 1; }

    if ((a.save_txt || a.save_img) && !a.out_dir.empty())
        fs::create_directories(a.out_dir);

    Engine eng(a.engine.string(), a.conf, a.iou);
    Semaphore slots_free(SLOTS);
    BoundedQueue<std::string> q_path(1024);
    BoundedQueue<Frame>       q_frame(64);
    BoundedQueue<Batch>       q_gpu(SLOTS);
    BoundedQueue<Results>     q_out(2 * a.consumers);
    Stats stats;

    for (auto& p : paths) q_path.push(p);
    q_path.close();

    printf("[INFO] %zu 张图, batch=%d, workers=%d, consumers=%d\n",
           paths.size(), a.batch, a.workers, a.consumers);
    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> pool;
    for (int i = 0; i < a.workers; ++i)
        pool.emplace_back(decoder_worker, std::ref(q_path), std::ref(q_frame));
    pool.emplace_back(assembler_worker, std::ref(q_frame), std::ref(q_gpu),
                      std::ref(eng), std::ref(slots_free), a.batch, a.workers);
    pool.emplace_back(gpu_worker, std::ref(eng), std::ref(q_gpu),
                      std::ref(q_out), std::ref(slots_free));
    for (int i = 0; i < a.consumers; ++i)
        pool.emplace_back(consumer_worker, std::ref(q_out), &a, &stats);

    for (auto& t : pool) t.join();                   // 队列按序 close, join 不会死锁
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    printf("[done] %ld 张图, 共 %ld 框, 耗时 %.2fs → 端到端 %.1f img/s\n",
           stats.n, stats.dets, dt, stats.n / dt);
    return 0;
}
