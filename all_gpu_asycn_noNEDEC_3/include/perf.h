#pragma once
// perf.h — 通用计时工具（CPU 毫秒计时 + GPU cudaEvent 毫秒计时）。
//
// 用途：对流水线各阶段做「单线程/单流」的孤立耗时测试（bench/bench_stages.cpp 复用），
// 也便于后续在任意 CUDA 工程里测某段 kernel/拷贝/CPU 函数的平均耗时。

#include <chrono>
#include <cuda_runtime.h>

// CPU 计时器（毫秒级，steady_clock）。
class CpuTimer {
public:
    void start() { t0_ = std::chrono::steady_clock::now(); }
    // 自上次 start() 到现在的耗时（毫秒）。
    double ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0_).count();
    }

private:
    std::chrono::steady_clock::time_point t0_;
};

// GPU 计时器（基于 cudaEvent，可测异步 kernel/拷贝的真实执行时间，毫秒）。
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&s_);
        cudaEventCreate(&e_);
    }
    ~GpuTimer() {
        cudaEventDestroy(s_);
        cudaEventDestroy(e_);
    }
    GpuTimer(const GpuTimer&) = delete;
    GpuTimer& operator=(const GpuTimer&) = delete;

    // 在流 st 上记录起点。
    void start(cudaStream_t st) { cudaEventRecord(s_, st); }
    // 在流 st 上记录终点并同步，返回这段区间耗时（毫秒）。
    float stop_ms(cudaStream_t st) {
        cudaEventRecord(e_, st);
        cudaEventSynchronize(e_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, s_, e_);
        return ms;
    }

private:
    cudaEvent_t s_, e_;
};
