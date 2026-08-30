# Deterministic Engine Building In TensorRT

**NOTE: This example requires TensorRT 8.7 or newer.**

## Introduction

During engine building, TensorRT runs and times several kernels in order to select
the most optimal ones. Since kernel timings may vary slightly from run to run, this
process is inherently non-deterministic.

In many cases, deterministic engine builds may be desirable. One way of achieving this
is to use a timing cache to ensure the same kernels are picked each time.

## Running The Example

1. Build an engine and save a timing cache:

    ```bash
    polygraphy convert identity.onnx \
        --save-timing-cache timing.cache \
        -o 0.engine
    ```

2. Use the timing cache for another engine build:

    ```bash
    polygraphy convert identity.onnx \
        --load-timing-cache timing.cache --error-on-timing-cache-miss \
        -o 1.engine
    ```

    We specify `--error-on-timing-cache-miss` so that we can be sure that the new engine
    used the entries from the timing cache for each layer.

3. Verify that the engines are exactly the same:

    <!-- Polygraphy Test: Ignore Start -->
    ```bash
    diff <(polygraphy inspect model 0.engine --show layers attrs) <(polygraphy inspect model 1.engine --show layers attrs)
    ```
    <!-- Polygraphy Test: Ignore End -->
===============================================================================
这份文件是 NVIDIA TensorRT 的一个示例说明文档，标题为 “Deterministic Engine Building In TensorRT”（在 TensorRT 中构建确定性推理引擎）。其核心目标是解决 TensorRT 引擎构建过程中的非确定性问题，并提供一种通过 timing cache（计时缓存） 实现可复现、确定性引擎构建的方法。

下面是对该文件的逐段详细解析：

一、背景说明（Introduction）；解析：
TensorRT 引擎构建机制：在将 ONNX 或其他格式模型转换为 TensorRT 引擎（.engine 文件）时，TensorRT 会尝试多种不同的 CUDA kernel 实现（例如卷积的不同算法、精度组合等），并对它们进行实际运行和计时。
非确定性来源：由于 GPU 调度、内存带宽波动、系统负载等因素，同一 kernel 在不同运行中的执行时间可能略有差异。这会导致 TensorRT 在不同构建中选择不同的“最优” kernel，从而生成结构或性能不同的引擎。
后果：这种非确定性会影响实验可复现性、部署一致性（例如 A/B 测试中两个“相同”模型行为不一致）等场景。

解析：需求：用户希望每次构建出完全相同的 TensorRT 引擎（即确定性构建）。
解决方案：使用 timing cache（计时缓存） —— 一个记录了各层 kernel 计时结果的二进制文件。后续构建时直接读取该缓存，跳过实际计时，强制使用缓存中记录的 kernel 选择。

二、操作步骤（Running The Example）；整个流程分为三步：
步骤 1：首次构建引擎并保存 timing cache
polygraphy convert identity.onnx \
    --save-timing-cache timing.cache \
    -o 0.engine
使用 polygraphy（NVIDIA 提供的命令行工具，用于简化 TensorRT 操作）将 identity.onnx 转换为 TensorRT 引擎。
--save-timing-cache timing.cache：在构建过程中，将每一层所测试的 kernel 及其计时结果保存到 timing.cache 文件中。
输出引擎为 0.engine。
这一步是非确定性的（因为实际运行了 kernel 计时），但结果被缓存下来。

步骤 2：使用 timing cache 构建第二个引擎
polygraphy convert identity.onnx \
    --load-timing-cache timing.cache --error-on-timing-cache-miss \
    -o 1.engine
--load-timing-cache timing.cache：加载之前保存的计时缓存。
--error-on-timing-cache-miss：关键参数。如果某一层在缓存中找不到对应的计时记录（例如模型结构变化、硬件不同等），则立即报错退出，而不是回退到重新计时。
输出引擎为 1.engine。
此步骤确保：所有层都严格使用缓存中的 kernel 选择，不会引入新的非确定性。

步骤 3：验证两个引擎是否完全相同
diff <(polygraphy inspect model 0.engine --show layers attrs) <(polygraphy inspect model 1.engine --show layers attrs)
使用 polygraphy inspect model 查看引擎的内部结构（包括各层类型、属性、使用的算法等）。
--show layers attrs：显示每层的详细属性。
通过 diff 比较两个引擎的文本表示。
预期结果：无任何输出（即两个引擎完全一致）。
注释 <!-- Polygraphy Test: Ignore Start --> 表明这是自动化测试的一部分，在 CI/CD 中会被忽略或特殊处理。

三、关键概念总结
概念	                          说明
非确定性引擎构建	因 kernel 计时波动导致每次构建可能选择不同实现
Timing Cache	   二进制缓存文件，记录各层 kernel 的计时与选择结果
--error-on-timing-cache-miss	强制要求缓存覆盖所有层，确保完全复用，避免隐式回退
Polygraphy	       NVIDIA 开源工具，简化 TensorRT 模型转换、分析、调试

四、适用场景
科研实验：需要确保模型部署前后行为一致。
生产部署：多台机器上构建的引擎必须完全相同（例如边缘设备集群）。
CI/CD 流水线：自动化构建需保证可复现性。
性能回归测试：确认优化未意外改变 kernel 选择策略。
五、注意事项
硬件一致性：timing cache 通常不能跨不同 GPU 架构使用（如从 A100 到 T4），因为可用 kernel 不同。
TensorRT 版本：文档注明需 TensorRT 8.7+，旧版本可能不支持完整 timing cache 功能。
模型变更：若 ONNX 模型结构修改，旧缓存将失效（触发 --error-on-timing-cache-miss 报错）。
首次构建仍非确定：只有后续使用缓存的构建才是确定性的。若需完全确定性，首次构建也应在受控环境中进行。
六、延伸建议
在生产环境中，可将 timing.cache 纳入版本控制，与模型文件一同管理。
结合 trtexec 或 Python API 也可实现类似功能，但 Polygraphy 提供了更简洁的 CLI 接口。
若需跨平台确定性，还需考虑其他因素（如 cuDNN 算法选择、随机数种子等），但 timing cache 是 TensorRT 层面最关键的一步。
结论：该示例展示了如何通过 timing cache + 严格缓存命中检查，在 TensorRT 中实现可复现、确定性的推理引擎构建，是工程实践中保障一致性的推荐做法。
-------------------------------------------------------------------------------
在实际工程中，TensorRT 的确定性引擎构建（Deterministic Engine Building） 并非仅仅是“学术需求”，而是部署一致性、模型版本管理、A/B 测试、边缘设备批量烧录等场景的核心保障。下面通过一个 真实工业级应用场景 详细说明其应用方式。
实际应用场景：自动驾驶感知模块的批量部署
背景：某自动驾驶公司使用 TensorRT 部署其 3D 目标检测模型（如 PointPillars 或 CenterPoint） 到车载计算平台（如 NVIDIA DRIVE Orin）。

全国部署 10,000+ 台车辆，每台车配备相同的 Orin 芯片。
要求：所有车辆运行完全相同的推理行为（输出 bbox 坐标、类别概率等必须逐帧一致），否则会导致：A/B 测试失效（无法判断性能差异来自模型还是引擎）；安全审计失败（某辆车误检而其他车正常 → 难以复现）；OTA 升级后行为漂移（用户投诉“升级后刹车更敏感”）
传统做法的问题（无 timing cache）
开发团队在 CI/CD 流水线中每次构建 TensorRT 引擎：
trtexec --onnx=model.onnx --saveEngine=model.trt
但发现：同一模型在两台 Orin 设备上构建出的 model.trt SHA256 哈希值不同；
推理时，对同一帧点云输入，输出置信度相差 0.001~0.01；在极端 case 下，一台车检测到障碍物，另一台漏检。根本原因：TensorRT 在构建时因 kernel 计时微小波动选择了不同算法（如 CUDNN_CONVOLUTION_FWD_ALGO_1 vs _3）。

 引入 Deterministic Engine Building 的解决方案
步骤 1：在受控环境中生成权威 timing cache：在 CI/CD 流水线中的专用构建机（与车载 Orin 同型号 GPU + 相同驱动/TensorRT 版本）上执行：
# 构建一次，并保存 timing cache
polygraphy convert model.onnx \
    --fp16 \
    --workspace 4096 \
    --save-timing-cache model.timing.cache \
    -o model_deterministic.trt

这台构建机需：独占 GPU（避免其他进程干扰计时）；关闭 CPU/GPU 频率动态调整（nvidia-smi -lgc 1310 锁频）；使用 Docker 固化环境（CUDA 12.2 + TensorRT 8.7.3）；
生成的 model.timing.cache 被提交到 Git LFS 或 Artifactory，与模型 ONNX 文件绑定。

步骤 2：所有后续构建强制使用 timing cache：
无论是本地调试、测试服务器、还是产线烧录脚本，都使用：
polygraphy convert model.onnx \
    --fp16 \
    --workspace 4096 \
    --load-timing-cache model.timing.cache \
    --error-on-timing-cache-miss \  # 关键！防止回退
    -o model_release.trt

若模型结构变更（如新增层），此命令会 立即报错，迫使开发者重新生成 cache。
若硬件不匹配（如在 A100 上构建 Orin 模型），也会报错，避免错误部署。

步骤 3：验证引擎一致性（自动化测试）：在 CI 中加入校验步骤：
# 重新构建一次
polygraphy convert ... --load-timing-cache model.timing.cache -o test.trt

# 比较二进制内容（或更严格的 layer-wise 比较）
if ! cmp -s model_deterministic.trt test.trt; then
    echo "ERROR: Non-deterministic build detected!"
    exit 1
fi

或者使用 Polygraphy 的语义比较（忽略无关元数据）：
polygraphy run model.onnx --trt --load-timing-cache=model.timing.cache
polygraphy run test.trt --onnx-data=model.onnx  # 确保输出一致

步骤 4：产线烧录与 OTA 升级
车辆出厂前：烧录脚本使用上述命令生成 model_release.trt，并校验 SHA256 与 CI 输出一致。
OTA 升级包：包含 model.onnx + model.timing.cache，车载系统在升级时本地重建引擎（而非直接下发 .trt 文件），因为：.trt 文件可能与当前驱动不兼容
但 timing cache + ONNX 可保证重建结果与 CI 完全一致
结果：10,000 台车运行字节级相同的推理逻辑。

 其他典型应用场景
场景	                    应用方式
金融风控模型	 确保 A/B 测试组使用完全相同的推理结果，避免“随机波动”干扰策略评估
医疗影像 AI	     法规要求模型行为可复现，任何输出差异需有明确原因（不能是构建随机性）
手机端 NLP 模型	 在高通/NPU 上通过类似机制（如 SNPE 的 DLR cache）实现跨设备一致
大模型推理服务	  多实例部署时，确保所有 GPU worker 加载的 TRT-LLM 引擎行为一致

 注意事项（实战经验）：不要直接分发 .engine 文件，→ 应分发 ONNX + timing cache，让目标设备本地构建，兼顾兼容性与确定性。timing cache 不跨 TensorRT 版本；

→ TensorRT 8.7 的 cache 不能用于 8.6 或 9.0。需在版本升级时重新生成。
首次构建仍需“黄金环境”

→ 虽然后续构建确定，但第一次的 cache 必须在干净、锁频、无干扰环境下生成。
结合 set_tactic_sources() 更彻底。
config = builder.create_builder_config()
config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS_LT))  # 仅允许特定 tactic
可进一步限制 kernel 搜索空间，减少非确定性来源。
 总结
在实际工程中，timing cache 不是“可选项”，而是生产级 TensorRT 部署的标配。它解决了：
跨设备行为一致性；构建可复现性；安全合规审计；高效 A/B 测试。
通过将 --save-timing-cache + --load-timing-cache --error-on-timing-cache-miss 纳入 CI/CD 和部署流程，团队可以像管理代码一样管理推理引擎的确定性，这是现代 MLOps 的重要一环。

 当部署的设备与构建引擎的设备一致时，直接使用首次构建的.engine模型文件即可，为什么要使用timing cache呢？是因为如果部署设备不同，就需要重新构建引擎，这时使用timing cache就可以保证模型与与首次构建的.engine模型文件行为一致，不然tensorrt会根据不同的硬件、设备运行动态情况而选择不同的kernel，导致模型行为不一致！！！




