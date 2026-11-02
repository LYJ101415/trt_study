#include "pipeline.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <thread>

#include "draw.h"
#include "jpeg_loader.h"

namespace fs = std::filesystem;

namespace {
constexpr int kTarget = 640;      // engine 固定输入尺寸
constexpr int kMaxDets = 300;     // engine 固定输出框数

bool has_display() {
    const char* d = std::getenv("DISPLAY");
    const char* wl = std::getenv("WAYLAND_DISPLAY");
    return (d && *d) || (wl && *wl);
}
}  // namespace

Pipeline::Pipeline(const PipelineOptions& opt) : opt_(opt), npp_(kTarget, kTarget, 114) {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaSetDevice(dev));

    // 槽深度兜底：解码×N + 3级GPU + 保存×N，至少能同时容纳，否则 CPU 线程互相等槽。
    const int min_buffers = opt_.num_decode_threads + opt_.num_save_threads + 3;
    if (opt_.num_buffers < min_buffers) {
        LOGW("num_buffers=%d 偏小（解码%d + 保存%d + 3级GPU = %d），已自动调大到 %d",
             opt_.num_buffers, opt_.num_decode_threads, opt_.num_save_threads,
             min_buffers, min_buffers);
        opt_.num_buffers = min_buffers;
    }

    // 建立三条流；num_streams==1 时三者指向同一流对象（退化为单流串行）。
    CUDA_CHECK(cudaStreamCreate(&stream_pre_));
    if (opt_.num_streams == 1) {
        stream_infer_ = stream_pre_;
        stream_post_ = stream_pre_;
    } else {
        CUDA_CHECK(cudaStreamCreate(&stream_infer_));
        CUDA_CHECK(cudaStreamCreate(&stream_post_));
    }
    npp_.setup(stream_pre_, stream_post_);

    trt_ = new TensorRTE2E(opt_.engine_path, opt_.conf, opt_.iou, kMaxDets);
    if (trt_->ok()) trt_->set_stream(stream_infer_);

    slots_.resize(opt_.num_buffers);
    for (size_t i = 0; i < slots_.size(); ++i) {
        Slot& s = slots_[i];
        CUDA_CHECK(cudaMalloc(&s.d_letterbox, (size_t)kTarget * kTarget * 3));
        CUDA_CHECK(cudaMalloc(&s.d_dets, (size_t)kMaxDets * 6 * sizeof(float)));
        CUDA_CHECK(cudaMallocHost(&s.h_dets, (size_t)kMaxDets * 6 * sizeof(float)));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.ev_pre, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.ev_infer, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.ev_done, cudaEventDisableTiming));
        free_q_.push_back((int)i);
    }
}

Pipeline::~Pipeline() {
    for (auto& s : slots_) {
        if (s.h_src) cudaFreeHost(s.h_src);
        if (s.h_res) cudaFreeHost(s.h_res);
        if (s.h_dets) cudaFreeHost(s.h_dets);
        if (s.d_src) cudaFree(s.d_src);
        if (s.d_letterbox) cudaFree(s.d_letterbox);
        if (s.d_res) cudaFree(s.d_res);
        if (s.d_dets) cudaFree(s.d_dets);
        if (s.ev_pre) cudaEventDestroy(s.ev_pre);
        if (s.ev_infer) cudaEventDestroy(s.ev_infer);
        if (s.ev_done) cudaEventDestroy(s.ev_done);
    }
    delete trt_;
    if (stream_pre_) cudaStreamDestroy(stream_pre_);
    if (stream_infer_ && stream_infer_ != stream_pre_) cudaStreamDestroy(stream_infer_);
    if (stream_post_ && stream_post_ != stream_pre_) cudaStreamDestroy(stream_post_);
}

void Pipeline::loader_thread() {
    while (true) {
        size_t fi = file_cursor_.fetch_add(1, std::memory_order_relaxed);
        if (fi >= files_.size()) break;
        const std::string& p = files_[fi];

        int idx;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_free_.wait(lk, [&] { return !free_q_.empty(); });
            idx = free_q_.front();
            free_q_.pop_front();
        }
        Slot& s = slots_[idx];
        s.path = p;
        s.name = fs::path(p).stem().string() + ".jpg";

        int w = 0, h = 0;
        if (!jpeg_decode_pinned(p, &s.h_src, &s.h_src_cap, &w, &h)) {
            LOGE("解码失败: %s", p.c_str());
            std::lock_guard<std::mutex> lk(mtx_);
            free_q_.push_back(idx);
            cv_free_.notify_one();
            continue;
        }
        s.w = w;
        s.h = h;
        s.lb = compute_letterbox(w, h, kTarget);   // scale/pad 在 CPU 算，随槽迁移

        {
            std::lock_guard<std::mutex> lk(mtx_);
            ready_q_.push_back(idx);
        }
        cv_ready_.notify_one();
    }
}

void Pipeline::gpu_thread() {
    while (true) {
        int idx;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_ready_.wait(lk, [&] { return !ready_q_.empty() || done_loading_; });
            if (ready_q_.empty()) {
                if (done_loading_) break;
                continue;
            }
            idx = ready_q_.front();
            ready_q_.pop_front();
        }

        Slot& s = slots_[idx];
        const size_t bytes = (size_t)s.w * s.h * 3;
        grow_device(&s.d_src, &s.d_src_cap, bytes);
        grow_device(&s.d_res, &s.d_res_cap, bytes);
        grow_pinned(&s.h_res, &s.h_res_cap, bytes);

        // —— Stage A：预处理（H2D + NPP letterbox）on Stream_Pre ——
        CUDA_CHECK(cudaMemcpyAsync(s.d_src, s.h_src, bytes, cudaMemcpyHostToDevice, stream_pre_));
        npp_.letterbox(s.d_src, s.w, s.h, s.d_letterbox, s.lb.scale, s.lb.pad_x,
                       s.lb.pad_y, s.lb.nw, s.lb.nh);
        CUDA_CHECK(cudaEventRecord(s.ev_pre, stream_pre_));

        // —— Stage B：推理 on Stream_Infer ——
        CUDA_CHECK(cudaStreamWaitEvent(stream_infer_, s.ev_pre, 0));
        if (!trt_->infer(s.d_letterbox, s.d_dets)) {
            LOGE("推理失败: %s", s.path.c_str());
        }
        CUDA_CHECK(cudaEventRecord(s.ev_infer, stream_infer_));

        // —— Stage C：后处理（画框 + NPP 恢复 + D2H）on Stream_Post ——
        CUDA_CHECK(cudaStreamWaitEvent(stream_post_, s.ev_infer, 0));
        draw_boxes_bgr(s.d_letterbox, kTarget, kTarget, kTarget * 3, s.d_dets,
                       kMaxDets, opt_.draw_thickness, stream_post_);
        npp_.restore(s.d_letterbox, s.d_res, s.w, s.h, s.lb.pad_x, s.lb.pad_y,
                     s.lb.nw, s.lb.nh);
        CUDA_CHECK(cudaMemcpyAsync(s.h_dets, s.d_dets,
                                   (size_t)kMaxDets * 6 * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_post_));
        CUDA_CHECK(cudaMemcpyAsync(s.h_res, s.d_res, bytes, cudaMemcpyDeviceToHost, stream_post_));
        CUDA_CHECK(cudaEventRecord(s.ev_done, stream_post_));

        {
            std::lock_guard<std::mutex> lk(mtx_);
            save_q_.push_back(idx);
        }
        cv_save_.notify_one();
    }
}

void Pipeline::saver_thread() {
    while (true) {
        int idx;
        {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_save_.wait(lk, [&] { return !save_q_.empty() || done_gpu_; });
            if (save_q_.empty()) {
                if (done_gpu_) break;
                continue;
            }
            idx = save_q_.front();
            save_q_.pop_front();
        }

        Slot& s = slots_[idx];
        CUDA_CHECK(cudaEventSynchronize(s.ev_done));   // 等 D2H 完成，锁页内存才可读

        if (opt_.print_dets) {
            int n = 0;
            for (int i = 0; i < kMaxDets; ++i)
                if (s.h_dets[i * 6 + 4] > 0.0f) ++n;
            LOGI("[%s] %d objects", s.name.c_str(), n);
            for (int i = 0; i < kMaxDets; ++i) {
                const float* d = &s.h_dets[i * 6];
                if (d[4] <= 0.0f) continue;
                LOGI("  cls=%d conf=%.3f box=[%.0f,%.0f,%.0f,%.0f]", (int)d[5], d[4],
                     d[0], d[1], d[2], d[3]);
            }
        }

        const std::string out = opt_.output_dir + "/" + s.name;
        if (jpeg_encode_bgr(out, s.h_res, s.w, s.h)) {
            std::lock_guard<std::mutex> lk(mtx_);
            ++saved_;
        } else {
            LOGE("保存失败: %s", out.c_str());
        }

        {
            std::lock_guard<std::mutex> lk(mtx_);
            free_q_.push_back(idx);
        }
        cv_free_.notify_one();
    }
}

int Pipeline::run() {
    if (!trt_ || !trt_->ok()) {
        LOGE("engine 加载失败，无法运行");
        return -1;
    }

    for (const auto& e : fs::directory_iterator(opt_.images_dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".jpg" || ext == ".jpeg") files_.push_back(e.path().string());
    }
    std::sort(files_.begin(), files_.end());
    if (files_.empty()) {
        LOGW("目录无 JPEG 图片: %s", opt_.images_dir.c_str());
        return 0;
    }
    LOGI("待处理图片 %zu 张 | buffers=%d streams=%d decode=%d save=%d conf=%.2f iou=%.2f",
         files_.size(), opt_.num_buffers, opt_.num_streams,
         opt_.num_decode_threads, opt_.num_save_threads, opt_.conf, opt_.iou);

    if (has_display()) {
        LOGI("检测到显示器，预留 GPU 直写显示入口（本构建未链接 EGL/GL，回退保存文件夹）");
    } else {
        LOGI("无显示器 → 结果保存到文件夹: %s", opt_.output_dir.c_str());
    }
    fs::create_directories(opt_.output_dir);

    auto t0 = std::chrono::steady_clock::now();

    std::vector<std::thread> loaders, savers;
    loaders.reserve(opt_.num_decode_threads);
    savers.reserve(opt_.num_save_threads);
    for (int i = 0; i < opt_.num_decode_threads; ++i)
        loaders.emplace_back(&Pipeline::loader_thread, this);
    std::thread gpu(&Pipeline::gpu_thread, this);
    for (int i = 0; i < opt_.num_save_threads; ++i)
        savers.emplace_back(&Pipeline::saver_thread, this);

    // 按依赖顺序收尾：先等解码线程全部结束 → 通知 GPU 无新图 → 等 GPU 结束 → 通知保存线程退出。
    for (auto& t : loaders) t.join();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        done_loading_ = true;
    }
    cv_ready_.notify_all();

    gpu.join();
    {
        std::lock_guard<std::mutex> lk(mtx_);
        done_gpu_ = true;
    }
    cv_save_.notify_all();

    for (auto& t : savers) t.join();

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    LOGI("完成 %d/%zu 张, 用时 %.2f s, 吞吐 %.1f img/s", saved_, files_.size(), sec,
         sec > 0 ? saved_ / sec : 0.0);
    return saved_;
}
