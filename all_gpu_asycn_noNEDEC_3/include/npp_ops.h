#pragma once
// npp_ops.h — NPP 预处理（letterbox resize）与后处理（恢复原图尺寸）。
//
// 多流支持：NPP 经典 API 用全局流（nppSetStream），无法同时绑定两条流；
// 故此处改用「NppStreamContext」流上下文 API（_Ctx 变体），预处理走 ctx_pre、
// 后处理走 ctx_post，二者可与推理流并发。

#include <cuda_runtime.h>
#include <npp.h>

class NppOps {
public:
    NppOps(int target_w = 640, int target_h = 640, unsigned char pad = 114);
    ~NppOps();

    // 建立两条流的 NPP 上下文（多流：预处理/后处理各用一条流）。
    // 单流模式下 pre == post，退化为串行。
    void setup(cudaStream_t stream_pre, cudaStream_t stream_post);

    // 预处理：d_src(原图 BGR, W×H) → d_dst(640×640 letterbox BGR)，走 ctx_pre。
    void letterbox(const unsigned char* d_src, int src_w, int src_h, unsigned char* d_dst,
                   float scale, int pad_x, int pad_y, int nw, int nh);

    // 后处理：d_canvas(640×640，已画框) → d_dst(原图 W×H)，裁剪内容区并缩放回原尺寸，走 ctx_post。
    void restore(const unsigned char* d_canvas, unsigned char* d_dst, int out_w, int out_h,
                 int pad_x, int pad_y, int nw, int nh);

private:
    int tw_, th_;
    unsigned char pad_;
    NppStreamContext ctx_pre_{}, ctx_post_{};
};
