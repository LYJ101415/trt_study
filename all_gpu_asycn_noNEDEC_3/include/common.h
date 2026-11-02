#pragma once
// common.h — 公共工具：CUDA / NPP 错误检查、日志、letterbox 参数计算。
//
// 设计说明：本项目剥离 NVDEC/NVENC，仅保留「图片 → NPP 预处理 → e2e engine → NPP 后处理」链路。
// 因此这里只有计算/访存相关的公共设施，不含任何视频编解码依赖。

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <cuda_runtime.h>
#include <npp.h>

// ---------------- 错误检查 ----------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "[CUDA] %s:%d %s failed: %s\n", __FILE__,      \
                         __LINE__, #call, cudaGetErrorString(_e));              \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define NPP_CHECK(call)                                                         \
    do {                                                                        \
        NppStatus _s = (call);                                                  \
        if (_s != NPP_SUCCESS) {                                                \
            std::fprintf(stderr, "[NPP] %s:%d %s failed: NppStatus=%d\n",       \
                         __FILE__, __LINE__, #call, (int)_s);                   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---------------- 日志 ----------------

#define LOGI(fmt, ...) std::fprintf(stdout, "[info] " fmt "\n", ##__VA_ARGS__)
#define LOGW(fmt, ...) std::fprintf(stderr, "[warn] " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) std::fprintf(stderr, "[error] " fmt "\n", ##__VA_ARGS__)

// ---------------- letterbox 参数（纯 CPU 计算，与 common.py 的 letterbox_params 对齐） ----------------
// scale = min(TARGET/h, TARGET/w)；nw/nh 为缩放后尺寸（截断取整）；
// pad_x/pad_y 为居中填充量。该结果随 slot 迁移，供 NPP 预处理与后处理共同使用。
struct Letterbox {
    int src_w = 0, src_h = 0;   // 原图宽高
    float scale = 1.0f;         // 缩放因子
    int nw = 0, nh = 0;         // 缩放后尺寸（进入 640 画布的内容区）
    int pad_x = 0, pad_y = 0;   // 居中填充量
};

inline Letterbox compute_letterbox(int w, int h, int target = 640) {
    Letterbox lb;
    lb.src_w = w;
    lb.src_h = h;
    lb.scale = std::min((float)target / (float)h, (float)target / (float)w);
    lb.nw = (int)(w * lb.scale);
    lb.nh = (int)(h * lb.scale);
    if (lb.nw < 1) lb.nw = 1;   // 极端细长图防 0
    if (lb.nh < 1) lb.nh = 1;
    lb.pad_x = (target - lb.nw) / 2;
    lb.pad_y = (target - lb.nh) / 2;
    return lb;
}

// ---------------- 按需增长显存/锁页内存 ----------------

inline void grow_device(unsigned char** ptr, size_t* cap, size_t need) {
    if (*cap >= need) return;
    if (*ptr) cudaFree(*ptr);
    CUDA_CHECK(cudaMalloc(ptr, need));
    *cap = need;
}

inline void grow_pinned(unsigned char** ptr, size_t* cap, size_t need) {
    if (*cap >= need) return;
    if (*ptr) cudaFreeHost(*ptr);
    CUDA_CHECK(cudaMallocHost(ptr, need));
    *cap = need;
}
