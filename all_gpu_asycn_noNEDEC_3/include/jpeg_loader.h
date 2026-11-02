#pragma once
// jpeg_loader.h — JPEG 解码/编码，解码输出【直接写入锁页内存】。
//
// 关键点（对应需求「文件夹读取图片直接分配进入锁页内存」）：
//   先 jpeg_read_header 拿到宽高 → 锁页内存不足时 cudaMallocHost 增长 →
//   jpeg_read_scanlines 逐行把 BGR 像素直接写进锁页内存。
//   全程没有「普通可分页内存 → 锁页内存」的二次拷贝，H2D 可异步 DMA。
//   *buf/*cap 传入槽的既有锁页缓冲，跨图复用，避免每次 cudaMallocHost。

#include <string>

// 解码 JPEG 到 BGR888，输出写入 *buf 指向的锁页内存（不足时自动增长）。
// 成功返回 true，*w/*h 为宽高。
bool jpeg_decode_pinned(const std::string& path, unsigned char** buf, size_t* cap,
                        int* w, int* h);

// 将 BGR888 缓冲（锁页或普通内存均可）编码为 JPEG 文件。quality 1..100。
bool jpeg_encode_bgr(const std::string& path, const unsigned char* data, int w, int h,
                     int quality = 90);
