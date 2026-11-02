// bench_stages.cpp — 各阶段「单线程/单流」耗时测试（通用）。
//
// 目的：独立测量流水线每个阶段的平均耗时，定位瓶颈（解码/各 GPU 阶段/编码）。
// 复用正式流水线的组件（jpeg_loader / npp_ops / tensorrt_e2e / draw），结果与真实运行一致。
// 用法：./bench_stages [--images <dir>] [--engine <path>] [--iters <n>] [--conf f] [--iou f]

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "common.h"
#include "draw.h"
#include "jpeg_loader.h"
#include "npp_ops.h"
#include "perf.h"
#include "tensorrt_e2e.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::string engine = "/root/my_FILE/models/yolov8_int8_static_e2e.engine";
    std::string images = "/root/autodl-tmp/Data_DeepPCB_YOLO/images/test_scaled";
    int iters = 200;
    float conf = 0.45f, iou = 0.65f;

    for (int i = 1; i < argc; ++i) {
        auto next = [&]() -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (!std::strcmp(argv[i], "--engine")) engine = next();
        else if (!std::strcmp(argv[i], "--images")) images = next();
        else if (!std::strcmp(argv[i], "--iters")) iters = std::atoi(next());
        else if (!std::strcmp(argv[i], "--conf")) conf = (float)std::atof(next());
        else if (!std::strcmp(argv[i], "--iou")) iou = (float)std::atof(next());
    }

    std::vector<std::string> files;
    for (const auto& e : fs::directory_iterator(images)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg") files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) { std::printf("目录无 JPEG 图片\n"); return 1; }
    const int n = (int)std::min<size_t>(files.size(), (size_t)iters);

    constexpr int TARGET = 640;
    constexpr int MAXDETS = 300;

    // —— 采样图（取第 0 张）——
    unsigned char* h_src = nullptr;
    size_t cap = 0;
    int w = 0, h = 0;
    if (!jpeg_decode_pinned(files[0], &h_src, &cap, &w, &h)) { std::printf("解码失败\n"); return 1; }

    // —— CPU 解码（复用锁页，单线程）——
    {
        CpuTimer t;
        t.start();
        for (int i = 0; i < n; ++i) jpeg_decode_pinned(files[i], &h_src, &cap, &w, &h);
        std::printf("CPU 解码(jpeg→锁页)    %8.3f ms/img\n", t.ms() / n);
    }
    // 恢复采样图（保证 GPU 各阶段用第 0 张、尺寸一致）
    jpeg_decode_pinned(files[0], &h_src, &cap, &w, &h);

    // —— 显存/锁页缓冲 ——
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t st = nullptr;
    CUDA_CHECK(cudaStreamCreate(&st));
    unsigned char *d_src = nullptr, *d_letterbox = nullptr, *d_res = nullptr;
    float* d_dets = nullptr;
    unsigned char* h_res = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, (size_t)w * h * 3));
    CUDA_CHECK(cudaMalloc(&d_letterbox, (size_t)TARGET * TARGET * 3));
    CUDA_CHECK(cudaMalloc(&d_res, (size_t)w * h * 3));
    CUDA_CHECK(cudaMalloc(&d_dets, (size_t)MAXDETS * 6 * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_res, (size_t)w * h * 3));

    const Letterbox lb = compute_letterbox(w, h, TARGET);
    NppOps npp(TARGET, TARGET, 114);
    npp.setup(st, st);                      // 单流测试：pre==post 同一流
    TensorRTE2E trt(engine, conf, iou, MAXDETS);
    trt.set_stream(st);
    if (!trt.ok()) { std::printf("engine 加载失败\n"); return 1; }

    // 预热（把整条链跑热，避免首帧编译/分配干扰）
    for (int i = 0; i < 20; ++i) {
        cudaMemcpyAsync(d_src, h_src, (size_t)w * h * 3, cudaMemcpyHostToDevice, st);
        npp.letterbox(d_src, w, h, d_letterbox, lb.scale, lb.pad_x, lb.pad_y, lb.nw, lb.nh);
        trt.infer(d_letterbox, d_dets);
        draw_boxes_bgr(d_letterbox, TARGET, TARGET, TARGET * 3, d_dets, MAXDETS, 2, st);
        npp.restore(d_letterbox, d_res, w, h, lb.pad_x, lb.pad_y, lb.nw, lb.nh);
        cudaMemcpyAsync(h_res, d_res, (size_t)w * h * 3, cudaMemcpyDeviceToHost, st);
    }
    CUDA_CHECK(cudaStreamSynchronize(st));

    const size_t bytes = (size_t)w * h * 3;
    GpuTimer gt;

    gt.start(st);
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(d_src, h_src, bytes, cudaMemcpyHostToDevice, st);
    std::printf("H2D(锁页→显存)         %8.4f ms/img\n", gt.stop_ms(st) / n);

    gt.start(st);
    for (int i = 0; i < n; ++i)
        npp.letterbox(d_src, w, h, d_letterbox, lb.scale, lb.pad_x, lb.pad_y, lb.nw, lb.nh);
    std::printf("NPP letterbox resize    %8.4f ms/img\n", gt.stop_ms(st) / n);

    gt.start(st);
    for (int i = 0; i < n; ++i) trt.infer(d_letterbox, d_dets);
    std::printf("TRT 推理(e2e)           %8.4f ms/img\n", gt.stop_ms(st) / n);

    gt.start(st);
    for (int i = 0; i < n; ++i)
        draw_boxes_bgr(d_letterbox, TARGET, TARGET, TARGET * 3, d_dets, MAXDETS, 2, st);
    std::printf("画框 kernel             %8.4f ms/img\n", gt.stop_ms(st) / n);

    gt.start(st);
    for (int i = 0; i < n; ++i)
        npp.restore(d_letterbox, d_res, w, h, lb.pad_x, lb.pad_y, lb.nw, lb.nh);
    std::printf("NPP 恢复原尺寸          %8.4f ms/img\n", gt.stop_ms(st) / n);

    gt.start(st);
    for (int i = 0; i < n; ++i) cudaMemcpyAsync(h_res, d_res, bytes, cudaMemcpyDeviceToHost, st);
    std::printf("D2H(显存→锁页)         %8.4f ms/img\n", gt.stop_ms(st) / n);

    // —— CPU 编码（单线程）——
    {
        CpuTimer t;
        t.start();
        for (int i = 0; i < n; ++i) jpeg_encode_bgr("/tmp/_bench_stages.jpg", h_res, w, h);
        std::printf("CPU 编码(BGR→jpeg)     %8.3f ms/img\n", t.ms() / n);
    }

    std::printf("(样本: %s, %dx%d, iters=%d)\n", fs::path(files[0]).filename().c_str(), w, h, n);

    cudaFreeHost(h_res);
    cudaFreeHost(h_src);
    cudaFree(d_src);
    cudaFree(d_letterbox);
    cudaFree(d_res);
    cudaFree(d_dets);
    cudaStreamDestroy(st);
    return 0;
}
