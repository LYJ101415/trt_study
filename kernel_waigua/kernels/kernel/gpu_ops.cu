/*
 * gpu_ops.cu — 融合 GPU 预处理 / 后处理 kernel（零 PyTorch / 零 OpenCV-CUDA）。
 *
 * 目标：把 CPU 上的 BGR->RGB + normalize（预处理）以及 decode + 阈值 + NMS（后处理）
 * 整体搬到 GPU；letterbox resize 改在 CPU 读取图片时完成（cv2.resize + 灰度 pad），
 * 宿主只做：
 *      cv2.imread(解码) + CPU letterbox -> 一次 H2D(640×640 uint8) -> execute_v2 -> 一次小 D2H(最终框)
 *
 * 输出语义与 server_trtapi.py 的 CPU 参考实现逐位对齐：
 *   预处理:  scale = min(640/H, 640/W); nh=int(H*scale); pad_h=(640-nh)//2 (同理 w)
 *   后处理:  output0 (1,10,8400) -> [cx,cy,w,h | s0..s5] -> xyxy -> 反 letterbox -> NMS
 *
 * 构建: 见 compile.sh（nvcc -> libgpu_ops.so），Python 侧经 gpu_ops.py 用 ctypes 调用。
 */

#include <cuda_runtime.h>    // CUDA 运行时API：kernel启动、内存拷贝、stream管理等
#include <thrust/sort.h>     // Thrust排序：sort_by_key
#include <thrust/sequence.h> // Thrust序列生成：sequence (生成 0,1,2,...)
#include <thrust/device_ptr.h> // Thrust设备指针包装
#include <thrust/execution_policy.h>  // thrust::device 执行策略，指定在GPU上执行
// Thrust 是 CUDA 自带的 STL-like 模板库，这里只用到了 sort_by_key 和 sequence，避免手写 GPU 排序。

// ---------------------------------------------------------------------------
// 预处理Kernel：BGR->RGB + uint8->float ÷255 + HWC->CHW（只做归一化 + 转通道）。
// ★ Resize/letterbox 已移到 CPU（读取图片时用 cv2.resize + 灰度 pad 完成），
//   这里只对 640×640 的 letterbox 结果做轻量格式转换，供模型输入。
// 一个线程写一个输出像素 (x,y)，一次性读 3 通道复用坐标计算。
// ---------------------------------------------------------------------------
__global__ void normalize_transpose_kernel(
        const unsigned char* __restrict__ src,   // 源图指针，BGR 格式，HWC 布局，尺寸 size × size（已在 CPU 完成 letterbox）
                                                 // __restrict__ 告诉编译器此指针不与其他指针别名，允许更激进的优化
        float* __restrict__ dst,   // 目标显存指针，RGB 格式，CHW 布局，3 × size × size
        int size)                  // 目标正方形边长，通常 640
{
    // ---- 计算当前线程负责的输出像素坐标 ----
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 输出图的列坐标 [0, size)
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 输出图的行坐标 [0, size)
    if (x >= size || y >= size) return;  // 超出目标图范围的线程直接退出（grid 可能比 size 大）

    const int wh = size * size;               // 单通道平面大小 = 640×640 = 409600，用于 CHW 索引计算
    int src_idx = (y * size + x) * 3;         // 源图 HWC 布局中该像素 BGR 三通道的起始下标

    // BGR→RGB 通道映射 + ÷255 归一化 + HWC→CHW 转置，一次完成（乘倒数比除法快）。
    dst[0 * wh + y * size + x] = src[src_idx + 2] * (1.0f / 255.0f);  // R 通道 <- B
    dst[1 * wh + y * size + x] = src[src_idx + 1] * (1.0f / 255.0f);  // G 通道 <- G
    dst[2 * wh + y * size + x] = src[src_idx + 0] * (1.0f / 255.0f);  // B 通道 <- R
}

// ---------------------------------------------------------------------------
// 后处理 1/3：decode + 阈值，把通过阈值的框压缩到 dets[]（原子计数）。
// output0 布局 (1,10,N) 连续 -> out0[c*N + i]。
// 同时把坐标反 letterbox 映射回原图尺寸，CPU 侧无需再算。
// ---------------------------------------------------------------------------
__global__ void decode_kernel(
        const float* __restrict__ out0, // YOLOv8输出tensor，布局 (1, 4+n_classes, N)，连续存储
                                         // out0[ch * N + i] 表示第i个anchor的第ch个值
        int N,                // anchor总数 = 8400 
        int n_classes,        // 类别数，如6
        float conf_thresh,    // 置信度阈值，如0.25
        float scale,          // letterbox缩放因子，如0.8
        int pad_x, int pad_y,  // letterbox填充偏移
        float* __restrict__ dets,  // 输出：候选框数组 [max_dets × 6]，格式 x1,y1,x2,y2,conf,cls
        float* __restrict__ confs,   // 输出：对应置信度数组 [max_dets]，供后续排序用
        int* __restrict__ count,    // 原子计数器，记录已通过阈值的框数量
        int max_dets   // 最大允许检测框数 = 300
        )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 当前线程处理的anchor索引
    if (i >= N) return;    // 超出anchor总数的线程退出

    // ---- 读取当前anchor的bbox参数（cxcywh格式）----
    float cx = out0[0 * N + i];  // center_x
    float cy = out0[1 * N + i];  // center_y
    float w  = out0[2 * N + i];  // width
    float h  = out0[3 * N + i];  // height

    // ---- 找最大类别分数及其类别ID ----
    float best = out0[4 * N + i]; // 初始化为第0类的分数
    int cls = 0;   // 初始类别=0
    for (int c = 1; c < n_classes; ++c) {
        float s = out0[(4 + c) * N + i];  // 第c类分数
        if (s > best) { best = s; cls = c; }  // 更新最大值和对应类别
    }
    // ---- 置信度过滤 ----
    if (best <= conf_thresh) return;  // 严格大于才保留（与Python参考实现一致）
                                      // 未通过的线程直接退出，不产生任何写入

    // ---- 坐标解码 + 反letterbox还原到原图坐标系 ----
    // cxcywh → xyxy，同时减去pad再除以scale
    float x1 = (cx - w * 0.5f - pad_x) / scale; // 左边界
    float y1 = (cy - h * 0.5f - pad_y) / scale; // 上边界
    float x2 = (cx + w * 0.5f - pad_x) / scale; // 右边界
    float y2 = (cy + h * 0.5f - pad_y) / scale; // 下边界
    // ★ 注意：坐标还原在GPU上完成，CPU拿到的是原图坐标，无需二次计算

    // ---- 原子写入紧凑数组 ----
    int idx = atomicAdd(count, 1); // 原子自增，获取当前框的写入位置
    if (idx >= max_dets) return;   // 超过上限则丢弃（但仍会递增count，外部需clamp）

    // 写入6个float到dets数组,格式 x1,y1,x2,y2,conf,cls
    dets[idx * 6 + 0] = x1;
    dets[idx * 6 + 1] = y1;
    dets[idx * 6 + 2] = x2;
    dets[idx * 6 + 3] = y2;
    dets[idx * 6 + 4] = best;
    dets[idx * 6 + 5] = (float)cls;
    // 单独写入confs数组（供thrust sort_by_key使用）
    confs[idx] = best;
}

// ---------------------------------------------------------------------------
// 后处理 3/3：贪心 NMS（单线程，O(n^2)，n 经阈值后通常 < 100，开销可忽略）。
// 输入 dets[] 需已按 conf 降序排列（order[] 是 argsort）。结果顺序写回 final[]。
// ---------------------------------------------------------------------------
__global__ void nms_greedy_kernel(
        const float* __restrict__ dets,     // 候选框数组 [6*n]，x1,y1,x2,y2,conf,cls
        const int* __restrict__ order,      // argsort结果 [n]，order[i]是第i高置信度框在dets中的原始索引
        int n,                // 候选框数量（已经过conf过滤）
        float iou_thresh,     // NMS IoU阈值，如0.45
        float* __restrict__ final,       // 输出：NMS后的最终框 [6*max_dets]
        int* __restrict__ final_count)   // 输出：最终有效框数量
{
    // ★ 只让一个线程执行（block=1, thread=0），其余线程全部返回
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int cnt = 0;   // 已保留框计数
    // ---- 外层循环：按置信度降序遍历每个候选框 ----
    for (int i = 0; i < n; ++i) {
        int bi = order[i];  // 通过排序索引获取dets中的实际位置

        // 读取当前框坐标和面积
        float x1i = dets[bi * 6 + 0], y1i = dets[bi * 6 + 1];
        float x2i = dets[bi * 6 + 2], y2i = dets[bi * 6 + 3];
        float ai = (x2i - x1i) * (y2i - y1i); // 当前框面积
        bool keep = true;

        // ---- 内层循环：与所有已保留框计算IoU ----
        for (int j = 0; j < cnt; ++j) {
            float x1j = final[j * 6 + 0], y1j = final[j * 6 + 1];
            float x2j = final[j * 6 + 2], y2j = final[j * 6 + 3];

            // 计算交集矩形坐标和面积
            float xx1 = fmaxf(x1i, x1j); // 交集左边界 = max(两个左边界)
            float yy1 = fmaxf(y1i, y1j); // 交集上边界
            float xx2 = fminf(x2i, x2j); // 交集右边界 = min(两个右边界)
            float yy2 = fminf(y2i, y2j); // 交集下边界
            float iw = fmaxf(0.f, xx2 - xx1);  // 交集宽度，无交集时为0
            float ih = fmaxf(0.f, yy2 - yy1);  // 交集高度
            float inter = iw * ih;     // 交集面积

            float aj = (x2j - x1j) * (y2j - y1j); // 已保留框面积
            float iou = inter / (ai + aj - inter + 1e-9f); // IoU = 交集 / 并集，+eps防除零
            if (iou > iou_thresh) { keep = false; break; }  // 与任一已保留框IoU超标 → 抑制
        }
        // ---- 保留则写入final数组 ----
        if (keep) {
            #pragma unroll  // 6次拷贝展开为6条赋值指令
            for (int k = 0; k < 6; ++k) final[cnt * 6 + k] = dets[bi * 6 + k];
            cnt++;
        }
    }
    *final_count = cnt;  // 写回最终有效框数量
}

// ---------------------------------------------------------------------------
// extern "C" 入口函数（供 Python ctypes 调用）
// ---------------------------------------------------------------------------

extern "C" void gpu_preprocess(
        const unsigned char* d_src, // 对应 gpu_ops.py 函数签名中的 ctypes.c_void_p（640×640 BGR uint8，已由 CPU 完成 letterbox）
        float* d_dst,               // 对应 ctypes.c_void_p（模型输入缓冲，3×640×640 float CHW）
        int size,                   // 对应 ctypes.c_int（边长 640）
        cudaStream_t stream         // 对应 gpu_ops.py 函数签名中的 ctypes.c_void_p
        )
{
    // 配置2D线程块：32×32 = 1024线程/block（GPU上限），每个线程处理1个像素
    dim3 block(32, 32);
    // 计算grid尺寸，向上取整确保覆盖整个 size×size
    dim3 grid((size + block.x - 1) / block.x,  // x方向block数 = ceil(640/32) = 20
              (size + block.y - 1) / block.y); // y方向block数 = ceil(640/32) = 20

    // 启动kernel，共享内存=0，指定stream实现异步执行
    normalize_transpose_kernel<<<grid, block, 0, stream>>>(d_src, d_dst, size);
} // 总线程数 = 20×20×1024 = 409600 = 640×640，恰好每个像素一个线程

/*
 * 完整后处理：decode -> thrust 按 conf 降序排序 -> 贪心 NMS -> 返回最终框数量。
 * 返回前把 final_count 拷回宿主（内部有一次 stream 同步，量级 ~微秒）。
 */
extern "C" int gpu_postprocess(
        const float* d_out0, int N, int n_classes, // 模型输出 + 元信息
        float conf_thresh, float iou_thresh,       // 阈值
        float scale, int pad_x, int pad_y,         // 坐标还原参数
        float* d_dets, float* d_confs, int* d_count, // scratch buffers
        int* d_order, float* d_final, int* d_final_count,  // 更多scratch + 输出
        int max_dets, cudaStream_t stream)  // 上限 + stream
{
    // ======== 步骤0: 清零计数器 ========
    cudaMemsetAsync(d_count, 0, sizeof(int), stream);
    // 异步清零，不阻塞host；在stream上与后续kernel保持顺序

    // ======== 步骤1: Decode + Conf Filter ========
    int threads = 256;  // 每block 256线程，每个线程处理1个候选框
    int blocks = (N + threads - 1) / threads; // ceil(8400/256) = 33 blocks
    decode_kernel<<<blocks, threads, 0, stream>>>(  // 8400线程并行decode
        d_out0, N, n_classes, conf_thresh, scale, pad_x, pad_y,
        d_dets, d_confs, d_count, max_dets);

    // ======== 步骤2: 获取候选框数量（★ 唯一一次中间同步点）========
    int n;
    cudaMemcpyAsync(&n, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);  // ★ 必须等待：后续逻辑依赖n的值
    // 这次同步不可避免——Thrust排序需要知道确切长度
    // 耗时 ~几μs（仅传4字节），是整个后处理中唯一的host等待点

    if (n == 0) return 0;  // 无检测结果，提前返回
    if (n > max_dets) n = max_dets;  // clamp到上限（decode中atomicAdd可能超限）

    // ======== 步骤3: Argsort by Confidence（降序）========
    thrust::sequence(thrust::device, d_order, d_order + n);   // 生成 order = [0, 1, 2, ..., n-1]
    thrust::sort_by_key(thrust::device, d_confs, d_confs + n, d_order,
                        thrust::greater<float>());                    // 按 conf 降序 argsort
    // 以 confs 为 key 降序排序，order 作为 value 跟随重排;排序后：confs[0] >= confs[1] >= ... >= confs[n-1]
    // order[i] = 原始dets中第i高置信度框的索引;★ 这步在GPU上执行（thrust::device策略），但由host API发起

    // ======== 步骤4: Greedy NMS ========
    nms_greedy_kernel<<<1, 1, 0, stream>>>(d_dets, d_order, n, iou_thresh,
                                           d_final, d_final_count); 
    // 单线程kernel，在同一个stream上排队，等排序完成后自动开始

    // ======== 步骤5: 获取最终框数量并返回 ========
    int out;
    cudaMemcpyAsync(&out, d_final_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // ★ 第二次同步：需要返回值给Python
    return out; // 返回给Python的n，用于控制D2H拷贝范围
}
