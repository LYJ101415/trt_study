#pragma once
// tensorrt_e2e.h — 端到端 TensorRT engine 封装。
//
// engine I/O（已实测）：
//   输入  image_raw     (1,640,640,3) UINT8   BGR 序（engine 内部再做 BGR→RGB 与 /255）
//   输入  iou_thresh    (1,) FLOAT   NMS 阈值
//   输入  score_thresh  (1,) FLOAT   置信度阈值
//   输出  detections    (300,6) FLOAT (x1,y1,x2,y2,conf,cls)，已 NMS，坐标在 640 空间
//
// 预处理/后处理算子已烧进 engine，故本模块只负责内存绑定 + enqueueV3。

#include <string>

#include <NvInfer.h>
#include <cuda_runtime.h>

class TensorRTE2E {
public:
    TensorRTE2E(const std::string& engine_path, float conf, float iou, int max_dets = 300);
    ~TensorRTE2E();

    bool ok() const { return ctx_ != nullptr; }
    void set_stream(cudaStream_t stream) { stream_ = stream; }
    int max_dets() const { return max_dets_; }

    // 一次推理：d_img(640×640×3 BGR uint8) → d_dets(max_dets×6 float)。
    // 调用方负责保证 d_img/d_dets 生命周期覆盖 enqueueV3 的异步执行。
    bool infer(const unsigned char* d_img, float* d_dets);

private:
    int max_dets_;
    cudaStream_t stream_ = nullptr;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* ctx_ = nullptr;
    float *d_iou_ = nullptr, *d_score_ = nullptr;
};
