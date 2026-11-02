#pragma once
// pipeline.h — 图片推理流水线编排。
//
// 多级异步：
//   解码线程 × N (CPU) : 遍历文件夹 → jpeg_decode_pinned 直写锁页内存 → CPU 算 letterbox
//   GPU 线程 × 1      : 三流流水（见下）→ 事件链衔接
//   保存线程 × N (CPU) : 等 event → 打印检测结果 → jpeg 编码写盘 → 归还槽
//
// GPU 三流（num_streams=3 时真正并发；num_streams=1 时三条流为同一对象，退化为单流串行）：
//   Stream_Pre   : H2D + NPP letterbox      （处理第 i 帧）
//   Stream_Infer : TRT enqueueV3            （处理第 i-1 帧）
//   Stream_Post  : 画框 + NPP 恢复 + D2H     （处理第 i-2 帧）
// 跨流依赖用 cudaStreamWaitEvent + 每槽 ev_pre/ev_infer/ev_done 衔接，
// 实现 pre(i+1) ∥ infer(i) ∥ post(i-1) 的流水重叠。

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "common.h"
#include "config.h"
#include "npp_ops.h"
#include "tensorrt_e2e.h"

// 单帧槽：锁页内存(直写解码 + D2H 目标) + 显存 + letterbox 参数随槽迁移。
struct Slot {
    unsigned char* h_src = nullptr;  size_t h_src_cap = 0;   // 解码 BGR, W*H*3
    unsigned char* h_res = nullptr;  size_t h_res_cap = 0;   // 结果 BGR, W*H*3
    float*         h_dets = nullptr;                         // 检测结果 300*6 (pinned)
    unsigned char* d_src = nullptr;  size_t d_src_cap = 0;
    unsigned char* d_letterbox = nullptr;                    // 640*640*3 固定
    unsigned char* d_res = nullptr;  size_t d_res_cap = 0;
    float*         d_dets = nullptr;                         // 300*6 固定
    int w = 0, h = 0;
    Letterbox lb;
    std::string path, name;
    cudaEvent_t ev_pre = nullptr, ev_infer = nullptr, ev_done = nullptr;
};

class Pipeline {
public:
    explicit Pipeline(const PipelineOptions& opt);
    ~Pipeline();

    // 运行整条流水线，返回成功处理的图片数；初始化失败返回 -1。
    int run();

private:
    void loader_thread();
    void gpu_thread();
    void saver_thread();

    PipelineOptions opt_;

    std::vector<std::string> files_;
    std::vector<Slot> slots_;
    std::atomic<size_t> file_cursor_{0};   // 多个解码线程共享的文件游标

    cudaStream_t stream_pre_ = nullptr, stream_infer_ = nullptr, stream_post_ = nullptr;
    NppOps npp_;
    TensorRTE2E* trt_ = nullptr;

    std::mutex mtx_;
    std::condition_variable cv_free_, cv_ready_, cv_save_;
    std::deque<int> free_q_, ready_q_, save_q_;
    bool done_loading_ = false, done_gpu_ = false;
    int saved_ = 0;
};
