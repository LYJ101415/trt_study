#include "tensorrt_e2e.h"

#include <cstdio>
#include <fstream>
#include <vector>

#include "common.h"

namespace {
class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity s, const char* msg) noexcept override {
        if (s <= Severity::kWARNING) std::fprintf(stderr, "[TRT] %s\n", msg);
    }
};
TrtLogger g_trt_logger;
}  // namespace

TensorRTE2E::TensorRTE2E(const std::string& engine_path, float conf, float iou, int max_dets)
    : max_dets_(max_dets) {
    std::ifstream f(engine_path, std::ios::binary);
    if (!f) {
        LOGE("无法打开 engine: %s", engine_path.c_str());
        return;
    }
    std::vector<char> data((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());

    runtime_ = nvinfer1::createInferRuntime(g_trt_logger);
    if (!runtime_) { LOGE("createInferRuntime 失败"); return; }
    engine_ = runtime_->deserializeCudaEngine(data.data(), data.size());
    if (!engine_) { LOGE("deserializeCudaEngine 失败"); return; }
    ctx_ = engine_->createExecutionContext();
    if (!ctx_) { LOGE("createExecutionContext 失败"); return; }

    // 打印实际 I/O（部署校验）
    LOGI("TensorRT engine I/O (%d tensors):", engine_->getNbIOTensors());
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* n = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(n);
        auto dims = engine_->getTensorShape(n);
        LOGI("  [%s] %s dims=[%d,%d,%d,%d]",
             mode == nvinfer1::TensorIOMode::kINPUT ? "IN " : "OUT", n,
             (int)dims.d[0], dims.nbDims > 1 ? (int)dims.d[1] : 0,
             dims.nbDims > 2 ? (int)dims.d[2] : 0, dims.nbDims > 3 ? (int)dims.d[3] : 0);
    }

    // 阈值标量只需分配一次并绑定，推理循环内不变。
    CUDA_CHECK(cudaMalloc(&d_iou_, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_score_, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_iou_, &iou, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_score_, &conf, sizeof(float), cudaMemcpyHostToDevice));
    if (!ctx_->setTensorAddress("iou_thresh", d_iou_) ||
        !ctx_->setTensorAddress("score_thresh", d_score_)) {
        LOGE("setTensorAddress(iou/score) 失败");
        return;
    }
}

TensorRTE2E::~TensorRTE2E() {
    if (d_iou_) cudaFree(d_iou_);
    if (d_score_) cudaFree(d_score_);
    if (ctx_) delete ctx_;
    if (engine_) delete engine_;
    if (runtime_) delete runtime_;
}

bool TensorRTE2E::infer(const unsigned char* d_img, float* d_dets) {
    if (!ctx_) return false;
    // image_raw / detections 每槽不同，每次绑定对应地址。
    if (!ctx_->setTensorAddress("image_raw", (void*)d_img)) {
        LOGE("setTensorAddress(image_raw) 失败");
        return false;
    }
    if (!ctx_->setTensorAddress("detections", (void*)d_dets)) {
        LOGE("setTensorAddress(detections) 失败");
        return false;
    }
    return ctx_->enqueueV3(stream_);
}
