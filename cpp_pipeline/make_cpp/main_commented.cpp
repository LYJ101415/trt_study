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
/**
 * TensorRT 日志记录器类
 * 继承自 nvinfer1::ILogger，用于捕获和记录 TensorRT 的日志信息
 * 只记录 WARNING 及以上级别的日志
 */
class TrtLogger : public nvinfer1::ILogger {
public:
    /**
     * 日志记录函数
     * @param s 日志严重级别（INFO、WARNING、ERROR等）
     * @param msg 日志消息字符串
     * @note 只记录 WARNING 及以上级别的日志到 stderr
     */
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) fprintf(stderr, "[TRT] %s\n", msg);
    }
};

// 全局 TensorRT 日志记录器实例
static TrtLogger g_logger;

// ---------------- 有界阻塞队列 ----------------
/**
 * 线程安全的有界阻塞队列模板类
 * 基于 mutex 和 condition_variable 实现生产者-消费者模式
 * 当队列满时生产者阻塞，当队列空时消费者阻塞
 */
template <typename T>
class BoundedQueue {
public:
    // 构造函数 - 初始化队列容量
    explicit BoundedQueue(size_t cap) : cap_(cap) {}

    /**
     * 向队列推入元素（生产者）
     * 如果队列已满则阻塞直到有空间或队列关闭
     * @param v 要推入的元素值（会被移动而不是复制）
     */
    void push(T v) {
        std::unique_lock<std::mutex> lk(m_);
        // 等待队列不满且未关闭
        cv_.wait(lk, [&] { return q_.size() < cap_ || closed_; });
        if (closed_) return;  // 如果队列已关闭，直接返回
        q_.push(std::move(v));  // 移动语义避免拷贝
        lk.unlock();
        cv_.notify_one();     // 通知可能的等待消费者
    }

    /**
     * 从队列弹出元素（消费者）
     * 如果队列已空且关闭则返回 false
     * @param v 用于存储弹出元素的引用
     * @return 队列未空且未关闭时返回 true，否则返回 false
     */
    bool pop(T& v) {   // 队列关闭且取空后返回 false
        std::unique_lock<std::mutex> lk(m_);
        // 等待队列不空或队列关闭
        cv_.wait(lk, [&] { return !q_.empty() || closed_; });
        if (q_.empty()) return false;  // 队列已空
        v = std::move(q_.front());     // 移动元素
        q_.pop();
        lk.unlock();
        cv_.notify_one();     // 通知可能的等待生产者
        return true;
    }

    /**
     * 关闭队列
     * 通知所有等待的线程，此后 push 操作将被忽略
     */
    void close() {
        std::lock_guard<std::mutex> lk(m_);
        closed_ = true;
        cv_.notify_all();     // 唤醒所有等待的线程
    }
private:
    std::queue<T> q_;           // 底层队列
    size_t cap_;                // 队列容量
    bool closed_ = false;       // 队列是否已关闭
    std::mutex m_;              // 互斥锁保护队列
    std::condition_variable cv_; // 条件变量用于线程同步
};

// ---------------- 计数信号量 (槽位回收) ----------------
/**
 * 计数信号量类 - 用于槽位资源管理
 * 继承自简单的计数器，通过条件变量实现阻塞等待
 * 用于实现批处理槽位的复用控制
 */
class Semaphore {
public:
    /**
     * 构造函数 - 初始化信号量计数值
     * @param cnt 初始计数值
     */
    explicit Semaphore(int cnt) : cnt_(cnt) {}

    /**
     * 获取信号量（阻塞等待）
     * 如果计数为0则等待，直到有信号量被释放
     */
    void acquire() {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return cnt_ > 0; });  // 等待计数>0
        --cnt_;  // 计数减1
    }

    /**
     * 释放信号量
     * 增加计数并通知等待的线程
     */
    void release() {
        { std::lock_guard<std::mutex> lk(m_); ++cnt_; }  // 计数加1
        cv_.notify_one();  // 唤醒一个等待线程
    }
private:
    int cnt_;                  // 计数器
    std::mutex m_;            // 互斥锁
    std::condition_variable cv_; // 条件变量
};

// ---------------- 数据结构 ----------------
/**
 * 单帧图像数据结构
 * 包含图像数据、文件名和缩放信息
 */
struct Frame {
    std::string name;         // 图像文件名
    cv::Mat img;             // 已 fit 过的图像数据（BGR格式，uint8）
    float scale = 1.0f;       // fit 操作的缩放系数，用于坐标映射回原图
};

/**
 * 批处理数据结构
 * 将多个帧组织成一个批次，准备送入GPU推理
 */
struct Batch {
    int slot = 0;                      // 使用的槽位索引（0~SLOTS-1）
    int n = 0, h = 0, w = 0;           // 批次信息：帧数、高度、宽度
    std::vector<Frame> frames;         // 该批次包含的所有帧
};

/**
 * 推理结果数据结构
 * 存储一批图像的检测结果
 */
struct Results {
    std::vector<Frame> frames;         // 原始帧信息
    std::vector<float> dets;          // 检测结果数据，格式为 n * MAX_DETS * 6 连续数组
};

/**
 * 统计信息结构
 * 用于记录处理进度和性能指标
 */
struct Stats {
    std::mutex m;                    // 互斥锁保护统计数据
    long n = 0, dets = 0;            // 处理的图像数和检测框总数
};

/**
 * 命令行参数结构
 * 存储程序运行的所有配置参数
 */
struct Args {
    fs::path source;                 // 输入图像目录路径
    fs::path engine;                 // TensorRT引擎文件路径
    fs::path out_dir;                // 输出目录路径
    int batch = 8;                   // 批处理大小
    int workers = 7;                 // 解码线程数
    int consumers = 4;               // 结果消费线程数
    float conf = 0.45f;              // 置信度阈值
    float iou = 0.65f;               // NMS IOU阈值
    bool save_txt = false;           // 是否保存txt格式检测结果
    bool save_img = false;           // 是否保存带标注的图像
};

// ---------------- 槽位资源池 ----------------
/**
 * 槽位资源结构
 * 每个槽位包含一个锁页内存（用于CPU-GPU数据传输）和对应的GPU内存
 */
struct Slot {
    uint8_t* pin_in = nullptr;   // 锁页输入画布（CPU端，可被GPU访问）
    void*    dev_in = nullptr;   // 设备内存（GPU端）
    size_t   cap_bytes = 0;      // 当前分配的内存容量（字节）
};

/**
 * 确保槽位有足够的内存空间
 * 如果当前容量不足则重新分配
 * @param s 槽位引用
 * @param bytes 需要的字节数
 */
static void ensure_slot(Slot& s, size_t bytes) {
    if (s.cap_bytes >= bytes) return;  // 容量足够，无需重新分配
    if (s.pin_in) {                    // 如果已有内存，先释放
        CUDA_CHECK(cudaFreeHost(s.pin_in));
        CUDA_CHECK(cudaFree(s.dev_in));
    }
    // 分配锁页内存（CPU可访问，GPU可直接访问）
    CUDA_CHECK(cudaMallocHost(&s.pin_in, bytes));
    // 分配GPU设备内存
    CUDA_CHECK(cudaMalloc(&s.dev_in, bytes));
    s.cap_bytes = bytes;  // 更新容量
}

// ---------------- 工具 ----------------
/**
 * 图像适配函数 - 将图像缩放到合适大小
 * 如果图像尺寸超过 MAX_HW 则进行缩放，保持长宽比
 * @param img 输入图像（会被修改）
 * @param scale 输出参数，返回缩放比例
 * @return 适配后的图像（原图或缩放后的图像）
 */
static cv::Mat fit_image(cv::Mat img, float& scale) {
    scale = 1.0f;  // 初始化缩放比例为1
    int m = std::max(img.rows, img.cols);  // 获取图像的最大边长
    if (m <= MAX_HW) return img;  // 如果尺寸足够，直接返回原图

    // 计算缩放比例，使最大边长不超过 MAX_HW
    float s = float(m) / MAX_HW;
    // 缩放图像，保持长宽比，四舍五入取整
    cv::resize(img, img, cv::Size(int(img.cols / s + 0.5), int(img.rows / s + 0.5)));
    scale = s;  // 返回实际的缩放比例
    return img;
}

/**
 * 过滤检测结果并映射到原图坐标
 * 将模型输出的检测框坐标映射回原图尺寸，并过滤无效检测结果
 * @param d300x6 模型输出检测数组（300x6，每行格式：x1,y1,x2,y2,score,class）
 * @param orig_w 原图像宽度
 * @param orig_h 原图像高度
 * @param scale 缩放比例（来自fit_image）
 * @param out 输出检测结果容器
 */
static void filter_dets(const float* d300x6, int orig_w, int orig_h, float scale,
                        std::vector<std::array<float, 6>>& out) {
    for (int i = 0; i < MAX_DETS; ++i) {
        const float* d = d300x6 + i * 6;  // 指向当前检测框

        // 空行判断：如果置信度为0，表示这是填充位，跳过
        if (d[4] <= 0.f) continue;

        // 提取坐标和类别
        float x1 = d[0], y1 = d[1], x2 = d[2], y2 = d[3];
        // 如果图像被缩放过，将坐标映射回原图
        if (scale != 1.0f) {
            x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale;
        }

        // 边界检查：确保检测框在图像范围内
        if (x2 <= 0 || y2 <= 0 || x1 >= orig_w || y1 >= orig_h) continue;

        // 保存有效的检测结果：x1,y1,x2,y2,score,class
        out.push_back({x1, y1, x2, y2, d[4], d[5]});
    }
}

// ---------------- 标注绘制 (与 Python 版同样式) ----------------
/**
 * 颜色数组 - 为6个类别定义不同的颜色
 * 颜色格式为 BGR（OpenCV默认）
 */
static const cv::Scalar COLORS[] = {{60, 60, 220}, {60, 180, 220}, {220, 60, 60},
                                    {60, 220, 60}, {220, 60, 200}, {0, 200, 255}};

/**
 * 在图像上绘制检测结果
 * 为每个检测框绘制矩形边界框和类别标签
 * @param img 要标注的图像（会被修改）
 * @param dets 检测结果列表
 * @return 标注后的图像
 */
static cv::Mat draw_dets(cv::Mat img, const std::vector<std::array<float, 6>>& dets) {
    for (const auto& d : dets) {
        // 计算类别索引（对6取模确保在有效范围内）
        int k = int(d[5]) % 6;
        const cv::Scalar& color = COLORS[k];  // 获取对应颜色

        // 绘制检测框边界
        cv::rectangle(img, cv::Point(int(d[0]), int(d[1])),
                      cv::Point(int(d[2]), int(d[3])), color, 2);

        // 创建类别标签文本
        char tag[64];
        snprintf(tag, sizeof(tag), "%s %.2f", CLASS_NAMES[k], d[4]);

        // 计算文本大小
        int base = 0;
        cv::Size ts = cv::getTextSize(tag, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base);

        // 计算文本位置（在检测框上方）
        cv::Point org(int(d[0]), std::max(0, int(d[1]) - ts.height - 6));

        // 绘制文本背景矩形
        cv::rectangle(img, org, org + cv::Point(ts.width, ts.height + 4), color, -1);

        // 绘制文本
        cv::putText(img, tag, {org.x, org.y + ts.height},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, {255, 255, 255}, 1);
    }
    return img;
}

// ---------------- 引擎封装 ----------------
/**
 * TensorRT 引擎封装类
 * 负责加载引擎文件、管理GPU资源、执行推理等操作
 */
class Engine {
public:
    /**
     * 构造函数 - 加载TensorRT引擎并初始化资源
     * @param path 引擎文件路径
     * @param conf 置信度阈值
     * @param iou NMS的IOU阈值
     */
    Engine(const std::string& path, float conf, float iou) {
        // 1. 读取引擎文件
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            fprintf(stderr, "无法打开引擎: %s\n", path.c_str());
            exit(1);
        }
        f.seekg(0, std::ios::end);
        size_t sz = f.tellg();  // 获取文件大小
        f.seekg(0);
        std::vector<char> blob(sz);  // 创建缓冲区
        f.read(blob.data(), sz);     // 读取整个文件

        // 2. 创建TensorRT组件
        runtime_.reset(nvinfer1::createInferRuntime(g_logger));
        engine_.reset(runtime_->deserializeCudaEngine(blob.data(), sz));
        ctx_.reset(engine_->createExecutionContext());

        // 3. 创建CUDA流和事件
        CUDA_CHECK(cudaStreamCreate(&stream_));
        CUDA_CHECK(cudaEventCreateWithFlags(&ev_, 0));

        // 4. 分配输出缓冲区和参数
        // 分配检测结果输出缓冲区（MAX_BATCH * MAX_DETS * 6 * float大小）
        CUDA_CHECK(cudaMalloc(&dev_out_, MAX_BATCH * MAX_DETS * 6 * 4));
        // 分配主机端缓冲区（用于存储从GPU拷回的结果）
        out_host_.resize(MAX_BATCH * MAX_DETS * 6);

        // 5. 分配阈值参数缓冲区并设置初始值
        CUDA_CHECK(cudaMalloc(&dev_iou_, 4));    // IOU阈值
        CUDA_CHECK(cudaMalloc(&dev_score_, 4));  // 置信度阈值
        set_scalar("iou_thresh", dev_iou_, iou);
        set_scalar("score_thresh", dev_score_, conf);

        // 同步CUDA流确保所有设置完成
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    /**
     * 析构函数 - 释放所有GPU资源和内存
     */
    ~Engine() {
        // 释放槽位资源
        for (auto& kv : slots_) {
            if (kv.second.pin_in) {
                cudaFreeHost(kv.second.pin_in);
                cudaFree(kv.second.dev_in);
            }
        }
        // 释放输出缓冲区
        cudaFree(dev_out_); cudaFree(dev_iou_); cudaFree(dev_score_);
        // 释放CUDA事件和流
        if (ev_) cudaEventDestroy(ev_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    /**
     * 获取指定索引的槽位
     * @param idx 槽位索引（0~SLOTS-1）
     * @return 槽位引用
     */
    Slot& get_slot(int idx) { return slots_[idx]; }

    /**
     * 获取主机端输出缓冲区指针
     * @return 指向检测结果的const指针
     */
    const float* out_host() const { return out_host_.data(); }

    /**
     * 提交批次到GPU进行推理
     * @param b 批次数据
     */
    void submit(Batch& b) {
        Slot& s = slots_[b.slot];  // 获取对应的槽位
        // 确保槽位有足够的内存
        ensure_slot(s, size_t(b.n) * b.h * b.w * 3);

        // 如果批次尺寸发生变化，更新输入形状
        if (!(b.n == last_n_ && b.h == last_h_ && b.w == last_w_)) {
            ctx_->setInputShape(IN_NAME, nvinfer1::Dims4{b.n, b.h, b.w, 3});
            last_n_ = b.n; last_h_ = b.h; last_w_ = b.w;
        }

        // 设置输入输出张量的内存地址
        ctx_->setTensorAddress(IN_NAME, s.dev_in);
        ctx_->setTensorAddress(OUT_NAME, dev_out_);

        // 异步拷贝数据到GPU（H2D - Host to Device）
        CUDA_CHECK(cudaMemcpyAsync(s.dev_in, s.pin_in, size_t(b.n) * b.h * b.w * 3,
                                   cudaMemcpyHostToDevice, stream_));

        // 执行推理（V3版本支持动态形状）
        if (!ctx_->enqueueV3(stream_)) {
            fprintf(stderr, "enqueueV3 failed\n");
            exit(1);
        }

        // 记录事件用于同步
        CUDA_CHECK(cudaEventRecord(ev_, stream_));
    }

    /**
     * 等待GPU推理完成并拷贝结果到主机
     * @note 输出数据约57KB，同步拷贝开销可忽略
     */
    void wait() {
        // 等待CUDA事件（即推理完成）
        CUDA_CHECK(cudaEventSynchronize(ev_));
        // 从GPU拷贝结果到主机
        CUDA_CHECK(cudaMemcpy(out_host_.data(), dev_out_,
                              out_host_.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

private:
    /**
     * 设置标量参数到GPU
     * @param name 参数名称（TensorRT中定义的标量输入）
     * @param dev GPU内存地址
     * @param v 参数值
     */
    void set_scalar(const char* name, void* dev, float v) {
        nvinfer1::Dims d1{};  // 创建1维形状
        d1.nbDims = 1; d1.d[0] = 1;
        ctx_->setInputShape(name, d1);  // 设置形状
        ctx_->setTensorAddress(name, dev);  // 设置地址
        // 异步拷贝值到GPU
        CUDA_CHECK(cudaMemcpyAsync(dev, &v, 4, cudaMemcpyHostToDevice, stream_));
    }

    // TensorRT核心组件（使用智能指针自动管理生命周期）
    std::unique_ptr<nvinfer1::IRuntime>          runtime_;  // 运行时对象
    std::unique_ptr<nvinfer1::ICudaEngine>       engine_;  // 引擎对象
    std::unique_ptr<nvinfer1::IExecutionContext> ctx_;     // 执行上下文

    // CUDA资源
    cudaStream_t stream_ = nullptr;  // CUDA流
    cudaEvent_t  ev_     = nullptr;  // CUDA事件

    // 输出缓冲区
    void*  dev_out_   = nullptr;     // GPU输出缓冲区
    void*  dev_iou_   = nullptr;     // GPU IOU阈值参数
    void*  dev_score_ = nullptr;     // GPU置信度阈值参数

    // 主机端输出缓冲区
    std::vector<float> out_host_;    // 存储从GPU拷回的结果

    // 槽位资源池
    std::unordered_map<int, Slot> slots_;  // 4个槽位的资源

    // 上次使用的批次尺寸（用于动态形状优化）
    int  last_n_ = -1, last_h_ = -1, last_w_ = -1;
};

// ---------------- 流水线各级 ----------------
/**
 * 解码线程工作函数
 * 从文件队列读取图像路径，解码图像并适配尺寸，推送到帧队列
 * @param q_path 输入的文件路径队列
 * @param q_frame 输出的帧队列
 */
static void decoder_worker(BoundedQueue<std::string>& q_path,
                           BoundedQueue<Frame>& q_frame) {
    std::string path;
    // 循环处理直到收到退出信号
    while (q_path.pop(path)) {
        // 使用OpenCV读取图像文件
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            fprintf(stderr, "[warn] 无法读取: %s\n", path.c_str());
            continue;
        }

        // 创建帧结构
        Frame fr;
        fr.name = fs::path(path).filename().string();  // 提取文件名
        // 适配图像尺寸并获取缩放比例
        fr.img = fit_image(std::move(img), fr.scale);
        // 将帧推送到帧队列
        q_frame.push(std::move(fr));
    }

    // 发送退出哨兵：空帧名表示解码线程结束
    // 收集齐所有workers的哨兵后，攒批线程才知道结束
    q_frame.push(Frame{});
}

/**
 * 攒批线程工作函数
 * 从帧队列收集图像，组成批次，填充到锁页内存，提交到GPU队列
 * @param q_frame 输入的帧队列
 * @param q_gpu 输出的GPU批处理队列
 * @param eng TensorRT引擎引用
 * @param slots_free 槽位信号量
 * @param batch 目标批大小
 * @param workers 解码线程总数（用于判断结束）
 */
static void assembler_worker(BoundedQueue<Frame>& q_frame, BoundedQueue<Batch>& q_gpu,
                             Engine& eng, Semaphore& slots_free, int batch, int workers) {
    std::vector<Frame> group;        // 当前收集的帧组
    int gh = 0, gw = 0;              // 当前帧组的尺寸（高度、宽度）
    int done = 0;                    // 收到的退出哨兵计数

    // 定义刷新函数：将当前帧组提交为一个批次
    auto flush = [&](std::vector<Frame>& g) {
        if (g.empty()) return;  // 空组不处理

        // 轮询使用槽位（SLOTS个槽位循环使用）
        static thread_local int next_slot = 0;
        Batch b;
        b.slot = next_slot;
        next_slot = (next_slot + 1) % SLOTS;  // 循环使用槽位

        // 设置批次信息
        b.n = int(g.size());
        b.h = g[0].img.rows;
        b.w = g[0].img.cols;
        b.frames = std::move(g);  // 移动帧组
        g.clear();

        // 等待槽位空闲（GPU已完成该槽位的处理）
        slots_free.acquire();

        // 获取槽位并确保内存足够
        Slot& s = eng.get_slot(b.slot);
        ensure_slot(s, size_t(b.n) * b.h * b.w * 3);

        // 在锁页内存中创建画布并填充letterbox灰底
        cv::Mat canvas0(b.h, b.w, CV_8UC3, s.pin_in);
        canvas0.setTo(cv::Scalar(114, 114, 114));    // letterbox 灰底

        // 将每张图像拷贝到画布的相应位置
        for (int i = 0; i < b.n; ++i) {
            // 计算当前帧在批处理中的位置
            cv::Mat dst(b.h, b.w, CV_8UC3, s.pin_in + size_t(i) * b.h * b.w * 3);
            // 将图像拷贝到左上角（letterbox模式下保持左对齐）
            const cv::Mat& im = b.frames[i].img;
            im.copyTo(dst(cv::Rect(0, 0, im.cols, im.rows)));
        }

        // 将批次推送到GPU队列
        q_gpu.push(std::move(b));
    };

    Frame fr;
    // 主循环：收集帧直到收到所有退出哨兵
    while (done < workers && q_frame.pop(fr)) {
        if (fr.name.empty()) {
            ++done;           // 收到退出哨兵，计数加1
            continue;          // 跳过哨兵帧
        }

        // 如果当前帧组不为空且尺寸变化，先刷新当前组
        if (!group.empty() && (fr.img.rows != gh || fr.img.cols != gw))
            flush(group);                            // 换尺寸: 先交出当前组

        // 如果组为空，记录新尺寸
        if (group.empty()) { gh = fr.img.rows; gw = fr.img.cols; }

        // 将帧加入当前组
        group.push_back(std::move(fr));

        // 如果达到目标批次大小，立即刷新
        if (int(group.size()) >= batch) flush(group);
    }

    // 处理剩余的帧（不足一个批次）
    flush(group);                                    // 尾巴 (不足 batch 由动态 shape 吃下)
    q_gpu.close();  // 关闭GPU队列，通知GPU线程结束
}

/**
 * GPU线程工作函数
 * 从GPU队列获取批次，执行推理，等待结果，推送到结果队列
 * @param eng TensorRT引擎
 * @param q_gpu 输入的批处理队列
 * @param q_out 输出的结果队列
 * @param slots_free 槽位信号量
 */
static void gpu_worker(Engine& eng, BoundedQueue<Batch>& q_gpu,
                       BoundedQueue<Results>& q_out, Semaphore& slots_free) {
    Batch b;
    // 循环处理批次直到队列为空
    while (q_gpu.pop(b)) {
        // 1. 提交批次到GPU
        eng.submit(b);
        // 2. 等待GPU推理完成并拷贝结果
        eng.wait();

        // 3. 构建结果结构
        Results r;
        r.frames = std::move(b.frames);  // 移动帧信息
        // 从引擎获取检测结果并拷贝到结果结构
        r.dets.assign(eng.out_host(), eng.out_host() + size_t(b.n) * MAX_DETS * 6);

        // 4. 将结果推送到输出队列
        q_out.push(std::move(r));

        // 5. 释放槽位信号量，通知攒批线程该槽可复用
        slots_free.release();
    }
    q_out.close();  // 关闭结果队列，通知消费线程结束
}

/**
 * 结果消费线程工作函数
 * 从结果队列获取检测结果，过滤坐标，保存文件或显示结果
 * @param q_out 输入的结果队列
 * @param a 命令行参数（包含保存配置）
 * @param st 统计信息（用于进度记录）
 */
static void consumer_worker(BoundedQueue<Results>& q_out, Args* a, Stats* st) {
    Results r;
    // 循环处理结果直到队列为空
    while (q_out.pop(r)) {
        // 处理批次中的每一帧
        for (size_t i = 0; i < r.frames.size(); ++i) {
            const Frame& fr = r.frames[i];

            // 过滤检测结果并映射到原图坐标
            std::vector<std::array<float, 6>> dets;
            filter_dets(r.dets.data() + i * MAX_DETS * 6, fr.img.cols, fr.img.rows,
                        fr.scale, dets);

            // 更新统计信息（线程安全）
            long done;
            {   // 锁只保护计数器! 若罩住下面的写盘, 多消费线程会在 I/O 上完全串行
                std::lock_guard<std::mutex> lk(st->m);
                st->n += 1;              // 处理的图像数+1
                st->dets += long(dets.size());  // 检测框总数+当前框数
                done = st->n;           // 获取当前已处理的图像总数
            }

            // 每200张图像打印一次进度
            if (done % 200 == 0) printf("  ... %ld 张, %ld 框\n", done, st->dets);

            // 如果需要保存结果
            if (a->save_txt || a->save_img) {
                // 构建输出文件路径（去除扩展名）
                fs::path base = a->out_dir / fs::path(fr.name).stem();

                // 保存txt格式检测结果
                if (a->save_txt) {
                    std::ofstream f(base.string() + ".txt");
                    for (auto& d : dets)
                        // 格式：类别 置信度 x1 y1 x2 y2
                        f << CLASS_NAMES[int(d[5]) % 6] << " " << d[4] << " "
                          << d[0] << " " << d[1] << " " << d[2] << " " << d[3] << "\n";
                }

                // 保存带标注的图像
                if (a->save_img) {
                    cv::Mat img = draw_dets(fr.img.clone(), dets);
                    // 保存为jpg格式，与原图同名
                    cv::imwrite(base.string() + ".jpg", img);
                }
            }
        }
    }
}

// ---------------- 主流程 ----------------
/**
 * 主函数 - 程序入口点
 * 解析命令行参数，初始化流水线，启动多线程处理
 */
int main(int argc, char** argv) {
    Args a;
    // 1. 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        // 获取下一个参数的辅助函数
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };

        if      (k == "--source")    a.source  = next();      // 输入目录
        else if (k == "--engine")    a.engine  = next();      // 引擎文件
        else if (k == "--out")       a.out_dir = next();      // 输出目录
        else if (k == "--batch")     a.batch = std::stoi(next());  // 批大小
        else if (k == "--workers")   a.workers = std::stoi(next()); // 解码线程数
        else if (k == "--consumers") a.consumers = std::stoi(next()); // 消费线程数
        else if (k == "--conf")      a.conf = std::stof(next());    // 置信度阈值
        else if (k == "--iou")       a.iou = std::stof(next());     // IOU阈值
        else if (k == "--save-txt")  a.save_txt = true;      // 保存txt
        else if (k == "--save-img")  a.save_img = true;      // 保存图像
        else {
            fprintf(stderr, "未知参数: %s\n", k.c_str());
            return 1;
        }
    }

    // 2. 验证必需参数
    if (a.source.empty() || a.engine.empty()) {
        fprintf(stderr, "用法: %s --source 图片目录 --engine xxx.engine [--batch 8] ...\n", argv[0]);
        return 1;
    }
    // 确保批大小不超过最大值
    a.batch = std::min(a.batch, MAX_BATCH);

    // 3. 收集输入图像路径
    std::vector<std::string> paths;
    // 遍历输入目录，查找图像文件
    for (auto& p : fs::directory_iterator(a.source)) {
        if (!p.is_regular_file()) continue;  // 跳过非文件
        std::string e = p.path().extension().string();
        // 转换为小写进行比较
        std::transform(e.begin(), e.end(), e.begin(), ::tolower);
        // 检查是否为支持的图像格式
        if (e == ".jpg" || e == ".jpeg" || e == ".png" || e == ".bmp")
            paths.push_back(p.path().string());
    }
    // 按文件名排序确保处理顺序一致
    std::sort(paths.begin(), paths.end());
    if (paths.empty()) {
        fprintf(stderr, "目录中没有图片: %s\n", a.source.c_str());
        return 1;
    }

    // 4. 创建输出目录（如果需要保存结果）
    if ((a.save_txt || a.save_img) && !a.out_dir.empty())
        fs::create_directories(a.out_dir);

    // 5. 初始化流水线组件
    Engine eng(a.engine.string(), a.conf, a.iou);  // 创建TensorRT引擎
    Semaphore slots_free(SLOTS);                    // 槽位信号量（4个槽位）

    // 创建各级队列（设置合适的容量）
    BoundedQueue<std::string> q_path(1024);    // 文件路径队列
    BoundedQueue<Frame>       q_frame(64);      // 帧队列
    BoundedQueue<Batch>       q_gpu(SLOTS);     // GPU批处理队列
    BoundedQueue<Results>     q_out(2 * a.consumers);  // 结果队列
    Stats stats;                                 // 统计信息

    // 6. 将所有图像路径推送到文件队列
    for (auto& p : paths) q_path.push(p);
    q_path.close();  // 关闭文件队列，通知解码线程结束

    // 7. 打印初始信息
    printf("[INFO] %zu 张图, batch=%d, workers=%d, consumers=%d\n",
           paths.size(), a.batch, a.workers, a.consumers);

    // 开始计时
    auto t0 = std::chrono::steady_clock::now();

    // 8. 创建并启动所有工作线程
    std::vector<std::thread> pool;

    // 启动解码线程池
    for (int i = 0; i < a.workers; ++i)
        pool.emplace_back(decoder_worker, std::ref(q_path), std::ref(q_frame));

    // 启动攒批线程（1个）
    pool.emplace_back(assembler_worker, std::ref(q_frame), std::ref(q_gpu),
                      std::ref(eng), std::ref(slots_free), a.batch, a.workers);

    // 启动GPU线程（1个）
    pool.emplace_back(gpu_worker, std::ref(eng), std::ref(q_gpu),
                      std::ref(q_out), std::ref(slots_free));

    // 启动结果消费线程
    for (int i = 0; i < a.consumers; ++i)
        pool.emplace_back(consumer_worker, std::ref(q_out), &a, &stats);

    // 9. 等待所有线程完成
    // 队列按序 close，join 不会死锁
    for (auto& t : pool) t.join();

    // 计算总耗时
    double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // 10. 打印最终统计信息
    printf("[done] %ld 张图, 共 %ld 框, 耗时 %.2fs → 端到端 %.1f img/s\n",
           stats.n, stats.dets, dt, stats.n / dt);

    return 0;
}