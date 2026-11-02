#pragma once
// config.h — 配置项与解析。
//
// 参数优先级：命令行 > 配置文件 > 代码默认值。
// 代码内保留全部默认值，配置文件与命令行均可覆盖。

#include <map>
#include <string>

struct PipelineOptions {
    // —— 模型 / 数据路径 ——
    std::string engine_path = "/root/my_FILE/models/yolov8_int8_static_e2e.engine";
    std::string images_dir  = "/root/autodl-tmp/Data_DeepPCB_YOLO/images/test";
    std::string output_dir  = "./images_results";

    // —— 推理阈值 ——
    float conf = 0.45f;
    float iou  = 0.65f;

    // —— 流水线 ——
    int num_buffers        = 12;   // in-flight 槽深度（需容纳 解码×N + 3级GPU + 保存×N）
    int num_streams        = 3;    // 1=单流  3=预处理/推理/后处理 三流异步
    int num_decode_threads = 3;    // 解码线程数（CPU：libjpeg 解码）
    int num_save_threads   = 3;    // 保存线程数（CPU：libjpeg 编码 + 写盘）
    int draw_thickness     = 2;    // 画框线宽
    bool print_dets        = false;
};

// 解析 INI 风格配置文件（`key = value` 或 `key : value`，支持 # 注释与 [section]）。
// 键名统一转小写；返回键值表（文件不存在/打不开返回空表）。
std::map<std::string, std::string> load_config_file(const std::string& path);

// 将单个 key=value 应用到 options（未知键静默忽略）。
void apply_kv(PipelineOptions& opt, const std::string& key, const std::string& value);

// 解析命令行参数并覆盖 options（不含 --config 本身，main 里先单独取出）。
// 返回 false 表示需要打印帮助并退出。
bool apply_cli(PipelineOptions& opt, int argc, char** argv);

// 打印用法。
void print_usage(const char* argv0);
