#include "jpeg_loader.h"

#include <cstdio>
#include <cstring>

#include <jpeglib.h>

#include "common.h"

// 解码 JPEG 到 BGR888，直接写入锁页内存（cudaMallocHost，按需增长）。
bool jpeg_decode_pinned(const std::string& path, unsigned char** buf, size_t* cap,
                        int* w, int* h) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, fp);

    bool ok = false;
    if (jpeg_read_header(&cinfo, TRUE) != JPEG_HEADER_OK) goto done;

    *w = cinfo.image_width;
    *h = cinfo.image_height;

    // 关键：拿到尺寸后直接（增长）锁页内存，解码结果直写进去，无中间可分页缓冲。
    grow_pinned(buf, cap, (size_t)(*w) * (*h) * 3);

    // 解码输出 BGR（与 engine 输入一致：engine 内部会再做 BGR→RGB）。
    cinfo.out_color_space = JCS_EXT_BGR;
    jpeg_start_decompress(&cinfo);

    {
        const int row_stride = (*w) * 3;
        while (cinfo.output_scanline < cinfo.output_height) {
            unsigned char* row = *buf + (size_t)cinfo.output_scanline * row_stride;
            jpeg_read_scanlines(&cinfo, &row, 1);
        }
    }
    jpeg_finish_decompress(&cinfo);
    ok = true;

done:
    jpeg_destroy_decompress(&cinfo);
    std::fclose(fp);
    return ok;
}

// 编码 BGR888 为 JPEG 文件。
bool jpeg_encode_bgr(const std::string& path, const unsigned char* data, int w, int h,
                     int quality) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_EXT_BGR;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);

    {
        const int row_stride = w * 3;
        while (cinfo.next_scanline < cinfo.image_height) {
            const unsigned char* row = data + (size_t)cinfo.next_scanline * row_stride;
            jpeg_write_scanlines(&cinfo, (JSAMPROW*)&row, 1);
        }
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(fp);
    return true;
}
