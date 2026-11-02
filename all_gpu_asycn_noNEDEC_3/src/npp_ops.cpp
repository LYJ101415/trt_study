#include "npp_ops.h"

#include "common.h"

NppOps::NppOps(int target_w, int target_h, unsigned char pad)
    : tw_(target_w), th_(target_h), pad_(pad) {}

NppOps::~NppOps() = default;

void NppOps::setup(cudaStream_t stream_pre, cudaStream_t stream_post) {
    // 建立流上下文：先绑定全局流再捕获上下文（nppGetStreamContext 取当前全局流）。
    NPP_CHECK(nppSetStream(stream_pre));
    NPP_CHECK(nppGetStreamContext(&ctx_pre_));
    if (stream_post != stream_pre) {
        NPP_CHECK(nppSetStream(stream_post));
        NPP_CHECK(nppGetStreamContext(&ctx_post_));
    } else {
        ctx_post_ = ctx_pre_;
    }
}

void NppOps::letterbox(const unsigned char* d_src, int src_w, int src_h,
                       unsigned char* d_dst, float scale, int pad_x, int pad_y,
                       int nw, int nh) {
    (void)scale;
    NppiSize src_size{src_w, src_h};
    NppiSize dst_size{tw_, th_};
    const int dst_step = tw_ * 3;

    // ① 灰底填充整幅画布
    Npp8u pad_val[3] = {pad_, pad_, pad_};
    NPP_CHECK(nppiSet_8u_C3R_Ctx(pad_val, d_dst, dst_step, dst_size, ctx_pre_));

    // ② 等比缩放进画布内容区（letterbox）
    NppiRect src_roi{0, 0, src_w, src_h};
    NppiRect dst_roi{pad_x, pad_y, nw, nh};
    NPP_CHECK(nppiResize_8u_C3R_Ctx(d_src, src_w * 3, src_size, src_roi,
                                    d_dst, dst_step, dst_size, dst_roi,
                                    NPPI_INTER_LINEAR, ctx_pre_));
}

void NppOps::restore(const unsigned char* d_canvas, unsigned char* d_dst,
                     int out_w, int out_h, int pad_x, int pad_y, int nw, int nh) {
    NppiSize src_size{tw_, th_};
    NppiSize dst_size{out_w, out_h};
    NppiRect src_roi{pad_x, pad_y, nw, nh};   // 只取内容区（去掉 letterbox 灰边）
    NppiRect dst_roi{0, 0, out_w, out_h};
    NPP_CHECK(nppiResize_8u_C3R_Ctx(d_canvas, tw_ * 3, src_size, src_roi,
                                    d_dst, out_w * 3, dst_size, dst_roi,
                                    NPPI_INTER_LINEAR, ctx_post_));
}
