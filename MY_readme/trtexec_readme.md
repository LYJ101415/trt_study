TensorRT 示例程序（Samples）
目录
1. “Hello World” 入门示例
表格
示例名称	语言	模型格式	描述
sampleOnnxMNIST	C++	ONNX	使用 ONNX 格式的 TensorRT “Hello World” 入门示例
network_api_pytorch_mnist	Python	INetwork	使用 PyTorch 和 TensorRT Network API 的 “Hello World” 入门示例
2. TensorRT API 进阶示例
表格
示例名称	语言	模型格式	描述
sampleCudla	C++	INetwork	使用 CuDLA API 运行 TensorRT 引擎（仅限 aarch64 架构）
sampleDynamicReshape	C++	ONNX	在 TensorRT 中使用动态形状进行数字识别
sampleEditableTimingCache	C++	INetwork	使用可编辑的时间缓存创建确定性构建
sampleNamedDimensions	C++	ONNX	处理命名输入维度
sampleNonZeroPlugin	C++	INetwork	添加输出形状依赖于数据的插件
sampleIOFormats	C++	ONNX	指定 TensorRT 的输入/输出格式
sampleProgressMonitor	C++	ONNX	进度监控 API 的使用
trtexec	C++	所有格式	TensorRT 命令行封装工具：trtexec
engine_refit_onnx_bidaf	Python	ONNX	通过解析器重新拟合从 ONNX 模型构建的引擎
introductory_parser_samples	Python	ONNX	使用 TensorRT 解析器导入模型的入门介绍
onnx_packnet	Python	ONNX	使用自定义层对 ONNX 模型进行 TensorRT 推理
simpleProgressMonitor	Python	ONNX	进度监控 API 的使用
python_plugin	Python	INetwork/ONNX	基于 Python 的 TRT 插件
non_zero_plugin	Python	INetwork/ONNX	针对 NonZero 算子的 Python TRT 插件
sample_plugin_v2_to_v3_migration	Python	INetwork	将自定义插件从 IPluginV2DynamicExt 迁移到 IPluginV3
cute_dsl_plugin	Python	INetwork	使用 CuteDSL 内核实现的 RMSNorm Python TRT 插件
attention_mdtrt	Python	ONNX	使用 MPI 和 NCCL 进行多设备注意力推理
3. 应用级示例
表格
示例名称	语言	模型格式	描述
detectron2	Python	ONNX	在 TensorRT 中支持 Detectron2 Mask R-CNN R50-FPN 3x 模型
4. 安全认证示例
表格
示例名称	语言	模型格式	描述
sampleSafeMNIST	C++	ONNX	为 MNIST 构建安全认证引擎
sampleSafePluginV3	C++	ONNX	在安全引擎中使用支持安全认证的插件
trtSafeExec	C++	ONNX	带有安全选项的 TensorRT 命令行封装工具
准备示例数据
许多示例需要 TensorRT 示例数据包。如果尚未挂载到 /usr/src/tensorrt/data（NVIDIA NGC 容器默认路径），请下载并解压：
从 TensorRT GitHub Releases 下载示例数据。
解压并配置数据：
unzip tensorrt_sample_data_xxx.zip
mkdir -p /usr/src/tensorrt/data
cp -r tensorrt_sample_data_*/* /usr/src/tensorrt/data/
export TRT_DATADIR=/usr/src/tensorrt/data
解压后，数据目录结构应如下所示：
$TRT_DATADIR/
├── int8_api/
├── mnist/
└── resnet50/
📝 详细解析与核心含义
这份文档是 TensorRT 官方示例程序的导航索引。它不仅仅是代码列表，更是学习 TensorRT 的最佳路线图。以下是深度解析：
1. 文档的核心作用
学习路径指引：将示例按难度和用途分级（入门 → API 进阶 → 实际应用 → 安全认证），帮助开发者循序渐进地掌握 TensorRT。
功能验证基准：每个 sample 都是经过验证的“标准答案”，当你自己的代码出问题时，可以对照 sample 排查。
新特性演示：TensorRT 10.x 的新功能（如 IPluginV3、CuteDSL、多设备推理）都通过特定 sample 第一时间展示用法。
2. 四大分类的深度含义
🟢 "Hello World" 入门示例
含义：最小可运行示例，跑通完整流程。
sampleOnnxMNIST：C++ 开发者的起点。演示了 ONNX Parser → Builder → Engine → Execution 的标准四步流程。
network_api_pytorch_mnist：Python 开发者的起点。注意：它用的是 INetworkDefinition API 逐层构建网络，而非 ONNX 解析器。这适用于无法导出 ONNX 或需要精细控制网络结构的场景。
🔵 TensorRT API 进阶示例（最重要）
含义：覆盖 TensorRT 核心能力的“工具箱”，按需查阅。
表格
关键能力	对应 Sample	实际意义
动态 Shape	sampleDynamicReshape, sampleNamedDimensions	解决生产环境中 batch size / 分辨率变化的问题
自定义插件	sampleNonZeroPlugin, python_plugin, cute_dsl_plugin	当 ONNX 不支持某算子时，如何自己写 Plugin 接入
插件版本迁移	sample_plugin_v2_to_v3_migration	⚠️ TRT 10.x 重点：V2 插件已废弃，此示例教你迁移到 V3
构建优化	sampleEditableTimingCache	实现确定性构建 + 加速编译，避免每次重新搜索最优 kernel
I/O 控制	sampleIOFormats	指定 FP16/INT8 输入输出、CHW/HWC 布局，减少前后处理拷贝
多卡推理	attention_mdtrt	大模型场景下，用 MPI+NCCL 实现跨 GPU 注意力计算
命令行工具	trtexec	你之前问的工具，这里展示了它的源码实现
🟡 应用级示例
含义：端到端的真实模型部署案例。
detectron2：Mask R-CNN 是出了名的难部署（包含 ROIAlign、动态 shape 等）。这个 sample 展示了如何处理复杂检测模型的完整转换链路，比简单分类模型更有参考价值。
🔴 安全认证示例
含义：面向汽车/医疗等需要 ISO 26262 / IEC 61508 认证的行业。
Safety Engine：不是普通引擎，而是带有运行时错误检测、内存保护的安全引擎。
trtSafeExec：trtexec 的安全版，用于验证安全引擎的性能和正确性。
💡 如果你不做车规/医疗项目，这部分可以跳过。
3. 关于“准备示例数据”的注意事项
表格
要点	说明
NGC 容器用户	数据已预置在 /usr/src/tensorrt/data，无需手动下载
非容器用户	必须手动下载，否则大部分 sample 会报 file not found
环境变量	TRT_DATADIR 是关键，sample 代码通过它定位数据文件
数据内容	mnist/（手写数字）、resnet50/（分类模型）、int8_api/（INT8 校准数据）
💡 给你的建议
结合你之前的问题（trtexec 大小、源码位置），建议的学习顺序：
先用现成的 trtexec 二进制跑通你的模型
阅读 samples/trtexec/README.md 理解参数含义
如果需要定制，看 sampleOnnxMNIST 理解 C++ API 基础
如果遇到动态 shape / 自定义算子问题，直接跳到对应的 API sample
不要试图读完所有 sample，把它们当作字典，按需查阅