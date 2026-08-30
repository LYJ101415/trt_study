# Inspecting Input Data


## Introduction

The `inspect data` subtool can display information about input data generated
by a data loader.


## Running The Example
1. Generate some input data by running inference:

    ```bash
    polygraphy run identity.onnx --onnxrt --save-inputs inputs.json
    ```

2. Inspect the input data:

    ```bash
    polygraphy inspect data inputs.json --show-values
    ```

    This will display something like:

    ```
    [I] ==== Data (1 iterations) ====

        x [dtype=float32, shape=(1, 1, 2, 2)] | Stats: mean=0.35995, std-dev=0.25784, var=0.066482, median=0.35968, min=0.00011437 at (0, 0, 1, 0), max=0.72032 at (0, 0, 0, 1), avg-magnitude=0.35995, p90=0.62933, p95=0.62933, p99=0.71123
            [[[[4.17021990e-01 7.20324516e-01]
               [1.14374816e-04 3.02332580e-01]]]]
    ```
================================================================================
这份文档展示了 如何使用 Polygraphy 的 inspect data 工具来分析模型推理所使用的输入数据（input data）。虽然表面上与“Inspecting Inference Outputs”类似，但其核心价值在于 确保输入的一致性、可复现性和合理性，这是调试精度问题和验证模型行为的基础。

下面从 核心目的、流程机制、输出解析、与输出检查的区别、高级用法、工程意义 六个维度进行深度解析。
一、核心目的：为什么需要 Inspect Input Data？
关键问题场景：结果不可复现;第一次运行 OK，第二次失败 → 可能是随机输入不同;跨后端比较失效
ONNX Runtime vs TensorRT 结果不同 → 若输入不一致，比较无意义;边界条件测试缺失;模型在 [0,1] 输入下正常，但在 [-1000, 1000] 下崩溃;数据预处理错误;图像未归一化（应 [0,1] 却为 [0,255]）
 inspect data inputs.json 能立即回答：“模型到底吃了什么数据？”

二、完整流程机制
步骤 1：生成输入数据
polygraphy run identity.onnx --onnxrt --save-inputs inputs.json
Polygraphy 自动生成 符合模型输入规格的随机数据
类型：float32（默认）
范围：[0, 1)（可通过 --val-range 修改）
形状：从 ONNX 模型中读取（如 (1,1,2,2)）
将输入保存为 inputs.json（Polygraphy 自定义格式，含 dtype/shape）

步骤 2：检查输入内容
polygraphy inspect data inputs.json --show-values
解析 JSON 中的 NumPy 数组;显示 统计信息 + 具体数值
 典型工作流：生成输入 → 保存 → 多次复用 → 检查一致性

三、输出结构深度解析（与输出检查高度相似但语义不同）
示例输出：
[I] ==== Data (1 iterations) ====

    x [dtype=float32, shape=(1, 1, 2, 2)] | Stats: mean=0.35995, std-dev=0.25784, ... 
        [[[[4.17021990e-01 7.20324516e-01]
           [1.14374816e-04 3.02332580e-01]]]]
关键字段解读：
字段	            含义	                   调试价值
x	          输入张量名称（来自 ONNX 模型）	确认是否为目标输入
dtype/shape	  数据类型与形状	               验证是否匹配模型要求
mean/std-dev  均值/标准差	                判断数据分布是否合理（如图像应在 [0,1]）
min/max at (...)	极值及其位置	        定位异常输入（如负值、超大值）
具体数值	       实际输入值	            用于手动计算验证

 注意：此例中输入与之前“输出检查”的数值完全相同，因为 identity.onnx 是恒等映射！

四、与 “Inspecting Inference Outputs” 的本质区别
维度	Inspect Inputs	         Inspect Outputs
数据来源	--save-inputs	    --save-outputs
核心问题	“模型吃了什么？”	      “模型吐了什么？”
调试焦点	数据预处理、随机性、范围	算子实现、精度损失、数值稳定性
典型异常	负像素值、形状错误	        NaN、全零输出、溢出
复现关键	固定输入是跨后端比较的前提	 输出差异需结合输入分析

 黄金法则：任何输出差异分析，必须建立在相同输入的基础上！
→ 因此 inspect inputs 是 inspect outputs 的先决条件。

五、高级用法与实战技巧
1. 固定输入以确保可复现性
# 生成并保存输入
polygraphy run model.onnx --save-inputs fixed_inputs.json

# 多次使用相同输入
polygraphy run model.onnx --load-inputs fixed_inputs.json --trt --save-outputs trt_out.json
polygraphy run model.onnx --load-inputs fixed_inputs.json --onnxrt --save-outputs ort_out.json

# 检查输入是否一致
polygraphy inspect data fixed_inputs.json --show-values

2. 自定义输入范围（测试边界条件）
# 生成 [-1, 1] 范围的输入
polygraphy run model.onnx --val-range -1,1 --save-inputs test_inputs.json
polygraphy inspect data test_inputs.json --show-values

3. 检查真实数据（非随机）
# 假设你有预处理好的 inputs.npz
polygraphy inspect data inputs.npz --show-values
支持 .npz, .pkl, .json 等多种格式

4. 多输入模型检查
# 输出示例（两个输入）
x [dtype=float32, shape=(1,3,224,224)] | ...
mask [dtype=bool, shape=(1,224,224)] | ...
确保所有输入张量均符合预期

六、工程意义：为什么这是部署必备技能？
场景 1：提交 Bug 报告,向 NVIDIA 报告 TRT 问题时，必须提供输入数据
inputs.json + model.onnx = 完整可复现用例
场景 2：CI/CD 流水线
# 伪代码
- run: polygraphy run model.onnx --save-inputs ci_inputs.json
- run: polygraphy inspect data ci_inputs.json --check "mean > 0 and max < 1.0"
自动验证输入数据合规性
场景 3：客户模型支持;客户说“模型在你们引擎上结果不对”;要求其提供 inputs.json，快速复现问题
场景 4：教学与文档
在模型文档中附上 inputs.json 示例;用户可直接复现 README 中的结果
七、注意事项
注意点	                   说明
随机种子未固定	            默认每次 --save-inputs 生成不同数据，需配合 --seed
大输入慎用 --show-values	如 4K 图像会输出巨量文本
动态形状支持有限	       随机输入基于 ONNX 中的静态形状（需先 surgeon sanitize）
数据格式兼容性	          确保 inputs.json 由 Polygraphy 生成（或符合其 schema）

 总结：inspect data inputs.json 的核心价值
维度	    说明
定位	    模型推理的“食材检查员”
核心作用	确保 输入可知、可控、可复现
不可替代性	没有它，所有输出分析都可能是“空中楼阁”
最佳实践	任何精度调试的第一步：先 inspect inputs！

 一句话精髓：“垃圾进，垃圾出”（Garbage In, Garbage Out）——用 polygraphy inspect data 确保你喂给模型的不是垃圾。因此，尽管命令简单，但它是构建 可靠、可复现、可调试 的 AI 推理流水线的基石。无论是研究、开发还是生产环境，都应将其纳入标准工作流。
---------------------------------------------------------------------------------
 实战示例：用真实图像测试 ResNet50
步骤 1：准备输入
# preprocess.py
import numpy as np
from PIL import Image

def load_data():
    img = Image.open("dog.jpg").resize((224, 224))
    img = np.array(img).astype(np.float32)
    img = img / 255.0  # 归一化到 [0,1]
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]  # ImageNet 标准化
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # HWC → CHW → NCHW
    
    return {"input": img}  # 假设 ONNX 输入名为 "input"
步骤 2：运行并检查
# 用真实图像运行 ONNX Runtime
polygraphy run resnet50.onnx --onnxrt \
    --load-inputs preprocess.py \
    --save-outputs ort_out.json

# 用相同输入运行 TensorRT
polygraphy run resnet50.onnx --trt \
    --load-inputs preprocess.py \
    --save-outputs trt_out.json

# 检查输入是否合理
polygraphy inspect data preprocess.py --show-values

# 比较两个后端输出差异
polygraphy compare ort_out.json trt_out.json

 六、高级技巧
技巧 1：批量输入;在 .npz 或 Python 脚本中提供 batch > 1 的数据：
# batch_size=4
return {"input": np.random.rand(4, 3, 224, 224).astype(np.float32)}
技巧 2：多输入模型
return {
    "input_1": np.random.rand(1, 3, 224, 224),
    "input_2": np.array([1, 2, 3], dtype=np.int32)
}
技巧 3：验证输入合规性
# 检查输入是否在 [0,1] 范围内
polygraphy inspect data real_inputs.npz | grep "min\|max"
✅ 总结
问题	               答案
能否用真实数据？	完全可以，且强烈推荐用于调试
支持哪些格式？	   .npz（首选）、.pkl、.json、Python 脚本、目录
关键要求是什么？	张量名称和形状必须与 ONNX 模型输入匹配
如何验证输入？	    polygraphy inspect data your_inputs --show-values
最佳实践？	       永远用真实/典型数据进行最终验证，随机数据仅用于初步测试