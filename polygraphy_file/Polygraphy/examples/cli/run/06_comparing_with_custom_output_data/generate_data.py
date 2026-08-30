"""以下代码是 Polygraphy 中用于生成并保存“自定义输入-输出对”以供后续精度验证的标准脚本，特别适用于 Identity 模型（输出 = 输入） 的场景。
它展示了如何构造符合 Polygraphy 格式要求的输入数据和 RunResults 输出对象，并将其序列化为 JSON 文件。"""

"""
Generates input and output data for an identity model and saves it to disk.
"""
import numpy as np
from polygraphy.comparator import RunResults # 导入 RunResults —— Polygraphy 用于表示推理结果的核心数据结构
from polygraphy.json import save_json

"""定义输入张量形状（batch=1, channel=1, height=2, width=2）
 此 shape 需与 ONNX 模型的输入要求一致"""
INPUT_SHAPE = (1, 1, 2, 2)


# We'll generate arbitrary input data and then "compute" the expected output data before saving both to disk.
# In order for Polygraphy to load the input and output data, they must be in the following format:
#   - Input Data: List[Dict[str, np.ndarray]] (A list of feed_dicts)
#   - Output Data: RunResults
"""以上注释说明：- 输入数据格式：List[feed_dict]，每个 feed_dict 是 {"input_name": np.ndarray}
- 输出数据格式：必须是 RunResults 对象（内部也是 List[Dict] 的封装）。输入数据格式要求：
List[Dict[str, np.ndarray]]
为什么是 List？ → 支持多组输入（例如 100 张测试图像）
为什么是 Dict？ → 支持多输入模型（如 {"input1": ..., "input2": ...}）
 单样本输入：[{"x": arr}]
 多样本输入：[{"x": arr1}, {"x": arr2}, ...]"""

# Generate arbitrary input data compatible with the model.
#
# TIP: We could have alternatively used a generator as in `run` example 05 (05_comparing_with_custom_input_data).
#   In that case, we would simply provide this script to `--data-loader-script` instead of saving the inputs here
#   and then using `--load-inputs`.
"""创建一个 feed_dict： - key "x"：必须与 ONNX 模型的输入名一致
 - value：全1的 float32 张量，shape=(1,1,2,2)"""
input_data = {"x": np.ones(shape=INPUT_SHAPE, dtype=np.float32)}

# NOTE: Input data must be in a list (to support multiple sets of inputs), so we create one before saving it.
#   The `description` argument is optional:
"""将 input_data 包装成列表 [input_data] → 符合 List[feed_dict] 要求
# 保存为 custom_inputs.json，添加描述便于识别"""
save_json([input_data], "custom_inputs.json", description="custom input data")


# "Compute" the outputs based on the input data. Since this is an identity model, we can just copy the inputs.
"""对于 Identity 模型，输出 = 输入 - key "y"：必须与 ONNX 模型的输出名一致
 - value：直接复用 input_data["x"]"""
output_data = {"y": input_data["x"]}

# To save output data, we can create a RunResults object:
"""初始化一个空的 RunResults 对象; RunResults 本质是一个字典：{runner_name: List[IterationResult]}"""
custom_outputs = RunResults()

# The `add()` helper function allows us to easily add entries.
#
# NOTE: As with input data, output data must be in a list, so we create one before saving it.
#
# TIP: Alternatively, we can manually add entries using an approach like:
#   runner_name = "custom_runner"
#   custom_outputs[runner_name] = [IterationResult(output_data, runner_name=runner_name), ...]
#
# TIP: To store outputs from multiple different implementations, you can specify different `runner_name`s to `add()`.
#   If `runner_name` is omitted, a default is used.
"""使用 add() 方法添加输出数据： - 第一个参数：List[feed_dict]（此处只有1个样本）
 - runner_name：标识该输出来源（如 "pytorch", "golden", "custom_runner"） → 后续对比时会显示此名称"""
custom_outputs.add([output_data], runner_name="custom_runner")
"""将 RunResults 对象序列化为 custom_outputs.json; 此文件可被 `polygraphy run --load-outputs` 直接加载"""
custom_outputs.save("custom_outputs.json")

"""
三、实际应用示例
场景：验证 identity.onnx 的 TensorRT 引擎是否正确
Step 1: 保存代码为 generate_data.py 并运行
python3 generate_data.py
# 生成 custom_inputs.json 和 custom_outputs.json
Step 2: 用 Polygraphy 验证 TRT 引擎
polygraphy run identity.onnx --trt \
    --load-inputs custom_inputs.json \
    --load-outputs custom_outputs.json
预期输出（成功）：

Comparing Output: 'y'
    Runner: trt-runner        | Stats: mean=1.000, min=1.000, max=1.000
    Runner: custom_runner     | Stats: mean=1.000, min=1.000, max=1.000
    Absolute Difference | Max: 0.000, Mean: 0.000, Median: 0.000
 PASSED | All outputs are within tolerance.
若 TRT 引擎有 bug（如输出全零），则会报错：

 FAILED | Absolute difference exceeds tolerance!
    Max absolute error: 1.000

 四、扩展应用场景
场景 1：用 PyTorch 生成黄金输出
# 替换 output_data 生成逻辑
import torch
model = torch.load("model.pth").eval()
with torch.no_grad():
    torch_out = model(torch.from_numpy(input_data["x"])).numpy()
output_data = {"output": torch_out}  # 注意输出名匹配
场景 2：多 runner 对比（TRT vs Golden vs ONNXRT）
polygraphy run model.onnx --trt --onnxrt \
    --load-inputs inputs.json \
    --load-outputs golden_outputs.json  # golden 作为第三个 runner
场景 3：批量生成 100 个测试样本
input_list = []
output_list = []
for i in range(100):
    inp = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    input_list.append({"x": inp})
    output_list.append({"y": inp})  # Identity

save_json(input_list, "100_inputs.json")
RunResults().add(output_list, "golden").save("100_outputs.json")

 五、注意事项
张量名称必须严格匹配
输入名 "x" → 必须与 ONNX 模型 graph.input[0].name 一致
输出名 "y" → 必须与 ONNX 模型 graph.output[0].name 一致
可用 polygraphy inspect model identity.onnx 查看
dtype 和 shape 必须合法
若模型要求 float32，不能传 float64
动态 shape 模型需确保输入在 TRT profile 范围内
RunResults.add() 的参数是 List
即使只有1个样本，也必须写 [output_data]，而非 output_data
 六、总结：这份代码的工程价值
特性	       说明
标准化格式	    严格遵循 Polygraphy 的输入/输出规范，确保兼容性
解耦生成与验证	数据生成（Python）与精度验证（CLI）分离，便于自动化
CI/CD 友好	   JSON 文件可纳入 Git，实现回归测试
调试高效	   快速验证引擎是否“数学正确”，无需依赖其他框架

 最佳实践流程：
1. 用权威工具生成 golden_outputs.json
2. 用 polygraphy run ... --load-outputs golden_outputs.json 验证部署模型
3. 将该流程集成到 CI，防止精度回退
这种模式是 工业级模型部署验证的基石，广泛应用于自动驾驶、医疗影像、推荐系统等领域
"""
