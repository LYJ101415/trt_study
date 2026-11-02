// draw.cu — 在 BGR888 图像上绘制检测框（NPP 无画框原语，此为唯一自定义 kernel）。
#include "draw.h"

#include <cuda_runtime.h>

#define DMIN(a, b) ((a) < (b) ? (a) : (b))
#define DMAX(a, b) ((a) > (b) ? (a) : (b))

__device__ __forceinline__ void put_pixel(unsigned char* img, int pitch, int x, int y,
                                          int w, int h, unsigned char b, unsigned char g,
                                          unsigned char r) {
    if (x < 0 || x >= w || y < 0 || y >= h) return;
    unsigned char* p = img + (size_t)y * pitch + (size_t)x * 3;
    p[0] = b;
    p[1] = g;
    p[2] = r;
}

__device__ __forceinline__ void hline(unsigned char* img, int pitch, int x0, int x1, int y,
                                      int w, int h, unsigned char b, unsigned char g,
                                      unsigned char r) {
    if (y < 0 || y >= h) return;
    int xs = DMAX(0, DMIN(x0, x1));
    int xe = DMIN(w - 1, DMAX(x0, x1));
    for (int x = xs; x <= xe; ++x) put_pixel(img, pitch, x, y, w, h, b, g, r);
}

__device__ __forceinline__ void vline(unsigned char* img, int pitch, int y0, int y1, int x,
                                      int w, int h, unsigned char b, unsigned char g,
                                      unsigned char r) {
    if (x < 0 || x >= w) return;
    int ys = DMAX(0, DMIN(y0, y1));
    int ye = DMIN(h - 1, DMAX(y0, y1));
    for (int y = ys; y <= ye; ++y) put_pixel(img, pitch, x, y, w, h, b, g, r);
}

__global__ void draw_boxes_bgr_kernel(unsigned char* img, int w, int h, int pitch,
                                      const float* dets, int max_dets, int thickness) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= max_dets) return;

    const float* d = dets + (size_t)i * 6;
    if (d[4] <= 0.0f) return;   // conf<=0 为填充行

    int x1 = (int)(d[0] + 0.5f);
    int y1 = (int)(d[1] + 0.5f);
    int x2 = (int)(d[2] + 0.5f);
    int y2 = (int)(d[3] + 0.5f);
    int cls = (int)d[5];
    if (cls < 0) cls = 0;

    // 6 类调色板（BGR）
    unsigned char pal[6][3] = {{0, 255, 0},     // 绿
                               {0, 0, 255},     // 红
                               {255, 0, 0},     // 蓝
                               {0, 255, 255},   // 黄
                               {255, 255, 0},   // 青
                               {255, 0, 255}};  // 品红
    unsigned char b = pal[cls % 6][0], g = pal[cls % 6][1], r = pal[cls % 6][2];

    for (int t = 0; t < thickness; ++t) {
        hline(img, pitch, x1, x2, y1 + t, w, h, b, g, r);
        hline(img, pitch, x1, x2, y2 - t, w, h, b, g, r);
        vline(img, pitch, y1, y2, x1 + t, w, h, b, g, r);
        vline(img, pitch, y1, y2, x2 - t, w, h, b, g, r);
    }
}

void draw_boxes_bgr(unsigned char* d_img, int w, int h, int pitch,
                    const float* d_dets, int max_dets, int thickness,
                    cudaStream_t stream) {
    const int threads = 128;
    const int blocks = (max_dets + threads - 1) / threads;
    draw_boxes_bgr_kernel<<<blocks, threads, 0, stream>>>(d_img, w, h, pitch, d_dets,
                                                          max_dets, thickness);
}
