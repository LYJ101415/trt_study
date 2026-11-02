// main.cpp — 剥离 NVDEC/NVENC 的图片推理入口（配置化 + 三流异步）。
//
// 参数优先级：命令行 > 配置文件(--config) > 代码默认值。
// 链路：文件夹图片 → libjpeg 直写锁页内存 → 异步 H2D → NPP letterbox resize
//       → e2e engine(已含归一化+NMS) → 画框 → NPP 恢复原尺寸 → 异步 D2H → 存图。
#include <cstring>
#include <string>

#include "config.h"
#include "pipeline.h"

int main(int argc, char** argv) {
    PipelineOptions opt;   // 代码内默认值

    // ① 先扫出 --config，加载配置文件（覆盖默认值）
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--config") && i + 1 < argc) {
            config_path = argv[i + 1];
            break;
        }
    }
    if (!config_path.empty()) {
        auto kv = load_config_file(config_path);
        if (kv.empty()) {
            std::fprintf(stderr, "[warn] 配置文件不存在或为空: %s\n", config_path.c_str());
        } else {
            for (const auto& [k, v] : kv) apply_kv(opt, k, v);
            std::fprintf(stdout, "[info] 已加载配置: %s (%zu 项)\n", config_path.c_str(), kv.size());
        }
    }

    // ② 命令行覆盖（优先级最高）
    if (!apply_cli(opt, argc, argv)) {
        print_usage(argv[0]);
        return 1;
    }

    Pipeline pipeline(opt);
    int n = pipeline.run();
    return n < 0 ? 1 : 0;
}
