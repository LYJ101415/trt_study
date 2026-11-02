#include "config.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace {

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string lower(std::string s) {
    for (char& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

bool parse_bool(const std::string& v) {
    std::string s = lower(trim(v));
    return s == "1" || s == "true" || s == "yes" || s == "on";
}

}  // namespace

std::map<std::string, std::string> load_config_file(const std::string& path) {
    std::map<std::string, std::string> kv;
    std::ifstream f(path);
    if (!f) return kv;

    std::string line;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';') continue;   // 注释
        if (t[0] == '[') continue;                                // [section]

        size_t pos = t.find_first_of("=:");
        if (pos == std::string::npos) continue;
        std::string key = lower(trim(t.substr(0, pos)));
        std::string val = trim(t.substr(pos + 1));
        if (!key.empty()) kv[key] = val;
    }
    return kv;
}

void apply_kv(PipelineOptions& o, const std::string& key, const std::string& value) {
    if (key == "engine_path") o.engine_path = value;
    else if (key == "images_dir") o.images_dir = value;
    else if (key == "output_dir") o.output_dir = value;
    else if (key == "conf") o.conf = (float)std::atof(value.c_str());
    else if (key == "iou") o.iou = (float)std::atof(value.c_str());
    else if (key == "num_buffers") o.num_buffers = std::atoi(value.c_str());
    else if (key == "num_streams") o.num_streams = std::atoi(value.c_str());
    else if (key == "num_decode_threads") o.num_decode_threads = std::atoi(value.c_str());
    else if (key == "num_save_threads") o.num_save_threads = std::atoi(value.c_str());
    else if (key == "draw_thickness") o.draw_thickness = std::atoi(value.c_str());
    else if (key == "print_dets") o.print_dets = parse_bool(value);
}

bool apply_cli(PipelineOptions& opt, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "[error] 缺少 %s 的值\n", name); return ""; }
            return argv[++i];
        };
        const char* a = argv[i];
        if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) return false;
        else if (!std::strcmp(a, "--config")) {
            // 已在 main 单独提取；这里连同其值一起跳过，避免误解析
            if (i + 1 < argc) ++i;
        } else if (!std::strcmp(a, "--engine")) { const char* v = next("--engine"); if (*v) apply_kv(opt, "engine_path", v); }
        else if (!std::strcmp(a, "--images")) { const char* v = next("--images"); if (*v) apply_kv(opt, "images_dir", v); }
        else if (!std::strcmp(a, "--output")) { const char* v = next("--output"); if (*v) apply_kv(opt, "output_dir", v); }
        else if (!std::strcmp(a, "--conf")) { const char* v = next("--conf"); if (*v) apply_kv(opt, "conf", v); }
        else if (!std::strcmp(a, "--iou")) { const char* v = next("--iou"); if (*v) apply_kv(opt, "iou", v); }
        else if (!std::strcmp(a, "--num-buffers")) { const char* v = next("--num-buffers"); if (*v) apply_kv(opt, "num_buffers", v); }
        else if (!std::strcmp(a, "--num-streams")) { const char* v = next("--num-streams"); if (*v) apply_kv(opt, "num_streams", v); }
        else if (!std::strcmp(a, "--num-decode-threads")) { const char* v = next("--num-decode-threads"); if (*v) apply_kv(opt, "num_decode_threads", v); }
        else if (!std::strcmp(a, "--num-save-threads")) { const char* v = next("--num-save-threads"); if (*v) apply_kv(opt, "num_save_threads", v); }
        else if (!std::strcmp(a, "--thickness")) { const char* v = next("--thickness"); if (*v) apply_kv(opt, "draw_thickness", v); }
        else if (!std::strcmp(a, "--print-dets")) apply_kv(opt, "print_dets", "true");
        else { std::fprintf(stderr, "[error] 未知参数: %s\n", a); return false; }
    }
    return true;
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
        "用法: %s [--config <file>] [选项]\n"
        "  --config <file>          配置文件 (INI 格式，命令行可覆盖其值)\n"
        "  --engine <path>          engine 路径 (默认见 config.h)\n"
        "  --images <dir>           输入图片目录\n"
        "  --output <dir>           结果输出目录 (默认 ./images_results)\n"
        "  --conf <float>           置信度阈值 (默认 0.45)\n"
        "  --iou <float>            NMS IoU 阈值 (默认 0.65)\n"
        "  --num-buffers <n>        流水线槽深度 (默认 12)\n"
        "  --num-streams <1|3>      GPU 流数：1 单流 / 3 预处理·推理·后处理 (默认 3)\n"
        "  --num-decode-threads <n> 解码线程数 (默认 3)\n"
        "  --num-save-threads <n>   保存线程数 (默认 3)\n"
        "  --thickness <n>          画框线宽 (默认 2)\n"
        "  --print-dets             逐图打印检测框\n",
        argv0);
}
