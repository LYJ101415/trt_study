#pragma once
// draw.h — 检测框绘制（自定义 CUDA kernel）。
//
// 说明：NPP 不提供画框原语（nppiDrawBox 不存在），因此画框为唯一保留的自定义 kernel
// （与原项目 ARCHITECTURE.md §3.3 结论一致）。画框在 640×640 letterbox 画布上就地完成，
// 之后由 NPP(nppiResize) 恢复原图尺寸。NPP 覆盖的是坐标恢复与尺寸缩放。

#include <cuda_runtime.h>

// 在 BGR888 图像上绘制检测框。dets 为 max_dets×6 的 float 数组 (x1,y1,x2,y2,conf,cls)，
// conf<=0 的行视为填充行跳过。坐标在画布(w×h)空间内，越界自动裁剪。
void draw_boxes_bgr(unsigned char* d_img, int w, int h, int pitch,
                    const float* d_dets, int max_dets, int thickness,
                    cudaStream_t stream);
