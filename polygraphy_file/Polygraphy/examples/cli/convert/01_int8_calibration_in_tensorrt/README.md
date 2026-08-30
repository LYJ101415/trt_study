# Int8 Calibration In TensorRT


## Introduction

In [API example 04](../../../api/04_int8_calibration_in_tensorrt/), we saw how we can leverage
Polygraphy's included calibrator to easily run int8 calibration with TensorRT.

But what if we wanted to do the same thing on the command-line?

To do this, we need a way to supply custom input data to our command-line tools.
Polygraphy provides multiple ways to do so, which are detailed [here](../../../../how-to/use_custom_input_data.md).

In this example, we'll use a data loader script by defining a `load_data` function in a Python
script called `data_loader.py` and then use `polygraphy convert` to build the TensorRT engine.

*TIP: We can use a similar approach with `polygraphy run` to build and run the engine.*

## Running The Example

1. Convert the model, using the custom data loader script to supply calibration data,
    saving a calibration cache for future use:

    ```bash
    polygraphy convert identity.onnx --int8 \
        --data-loader-script ./data_loader.py \
        --calibration-cache identity_calib.cache \
        -o identity.engine
    ```

2. **[Optional]** Rebuild the engine using the cache to skip calibration:

    ```bash
    polygraphy convert identity.onnx --int8 \
        --calibration-cache identity_calib.cache \
        -o identity.engine
    ```

    Since the calibration cache is already populated, calibration will be skipped.
    Hence, we do *not* need to supply input data.


3. **[Optional]** Use the data loader directly from the API example.

    The method outlined here is so flexible that we can even use the data loader we defined in the API example!
    We just need to specify the function name since the example does not call it `load_data`:

    ```bash
    polygraphy convert identity.onnx --int8 \
        --data-loader-script ../../../api/04_int8_calibration_in_tensorrt/example.py:calib_data \
        -o identity.engine
    ```
====================================================================
这份文档详细介绍了 如何在命令行（CLI）中使用 Polygraphy 工具对 TensorRT 引擎进行 INT8 量化校准（Calibration），核心亮点是 通过自定义 Python 数据加载脚本提供校准数据，从而实现与 API 编程方式同等的灵活性。下面我将从背景动机、技术原理、命令详解、工作流设计、实际价值五个维度进行深度解析。

一、背景动机：为什么需要 CLI 校准？在 API 示例 04 中，我们通过编写 Python 脚本调用 polygraphy.backend.trt.Calibrator 实现 INT8 校准。
但许多用户更习惯命令行操作（如快速验证、CI/CD 集成、非开发人员使用），因此 Polygraphy 提供了 polygraphy convert 命令直接支持校准。

核心目标：让命令行用户也能轻松提供自定义校准数据，无需写完整 Python 脚本。

二、关键技术：自定义数据加载器（Data Loader Script）
1. data_loader.py 的约定
Polygraphy 要求数据加载脚本必须包含一个名为 load_data() 的函数（或通过:function_name 指定），该函数需满足：返回类型：可迭代对象（如 generator, list）
元素格式：Dict[str, np.ndarray]（即 feed_dict，键为输入名，值为 NumPy 数组）
示例 data_loader.py：
# data_loader.py
import numpy as np

def load_data():
    for _ in range(100):  # 提供 100 个校准样本
        yield {
            "x": np.ones((1, 1, 2, 2), dtype=np.float32)  # 输入名"x"，shape=(1,1,2,2)
        }
为什么用 generator？避免一次性加载大量数据到内存（尤其校准集较大时）。

2. 灵活指定函数名；若脚本中函数不叫 load_data（如 API 示例中的 calib_data），可通过 script_path:function_name 语法指定：
--data-loader-script ../../../api/04.../example.py:calib_data
优势：直接复用已有 API 示例代码，无需复制粘贴！

三、核心命令详解：polygraphy convert；基础命令（带校准）
polygraphy convert identity.onnx --int8 \
    --data-loader-script ./data_loader.py \
    --calibration-cache identity_calib.cache \
    -o identity.engine

参数	                  作用	                必要性
--int8	                启用 INT8 量化	        必须
--data-loader-script	指定校准数据来源	     首次校准时必须
--calibration-cache	    保存/加载校准缓存文件	 强烈建议（加速后续构建）
-o	                    输出 TensorRT 引擎路径	 必须

重用校准缓存（跳过校准）：
polygraphy convert identity.onnx --int8 \
    --calibration-cache identity_calib.cache \
    -o identity.engine
关键点：当 identity_calib.cache 已存在且非空时，Polygraphy 自动跳过校准步骤
优势：引擎重建速度提升 10x+（校准通常最耗时）；确保多次构建的量化参数一致（避免精度波动）。校准缓存是什么？一个二进制文件，存储每个激活层的 min/max 值（用于计算 scale factor）。TensorRT 官方格式，可跨设备复用（同架构 GPU）。

四：完整工作流设计；步骤分解：
首次构建：
执行 polygraphy convert ... --data-loader-script ... --calibration-cache cache
Polygraphy 内部：加载 ONNX 模型；创建 Calibrator 并注入 data_loader.py 的数据
运行校准（约 100~500 次推理）；保存 cache 文件；生成最终 INT8 引擎
后续构建：
执行 polygraphy convert ... --calibration-cache cache
直接读取 cache 中的量化参数；跳过校准，直接构建引擎

五、实际应用场景与价值
场景 1：快速验证模型 INT8 精度
# 1. 用少量数据校准（快速）
echo "def load_data(): yield {'input': np.random.rand(1,3,224,224).astype(np.float32)}" > quick_calib.py

# 2. 构建引擎
polygraphy convert model.onnx --int8 --data-loader-script quick_calib.py -o model_int8.engine

# 3. 对比 FP16 vs INT8 精度
polygraphy run model.onnx --trt --precision fp16 --inputs calib_data.json
polygraphy run model_int8.engine --inputs calib_data.json

场景 2：CI/CD 自动化校准
# .gitlab-ci.yml
build-int8-engine:
  script:
    - python generate_calib_data.py  # 生成校准集
    - polygraphy convert model.onnx --int8 --data-loader-script calib_loader.py --calibration-cache model.cache -o model.engine
    - polygraphy run model.engine --validate-with model.onnx  # 精度验证

场景 3：复用现有数据管道；若已有 PyTorch DataLoader：
# torch_to_numpy.py
from my_dataset import get_dataloader
import numpy as np

def load_data():
    dataloader = get_dataloader(batch_size=1)
    for batch in dataloader:
        yield {"input": batch[0].cpu().numpy()}  # 转为 NumPy

直接用于校准：
polygraphy convert model.onnx --int8 --data-loader-script torch_to_numpy.py -o engine
六、高级技巧与注意事项
技巧 1：校准数据质量决定 INT8 精度
数量：通常 100~500 样本足够（覆盖激活值分布）
代表性：需与真实推理数据同分布（避免校准集过拟合）
预处理：必须与训练/推理时一致（如归一化、resize）
注意事项：
问题	             解决方案
校准后精度暴跌	    检查校准数据是否 representative；尝试增加样本量
cache 文件无效	    确保 ONNX 模型未修改（节点名/结构变化会使 cache 失效）
自定义算子校准失败	 需在 TensorRT 中注册 plugin，并确保校准数据兼容

调试校准过程：
# 启用详细日志
polygraphy convert model.onnx --int8 --data-loader-script loader.py --verbose
日志会显示：校准进度、每层的 min/max 值、cache 保存路径等

七、与其他工具对比
方法	                优点	           缺点
TensorRT C++ API	   最高性能	         开发复杂，需编译
trtexec	               官方 CLI 工具	 校准数据需转为 .pgm 等特定格式
Polygraphy CLI	  Python 数据加载 + ONNX 原生支持	依赖 Python 环境
Polygraphy API	  最灵活	             需编写完整脚本

结论：对于 Python 生态用户，polygraphy convert 是 INT8 校准的最佳平衡点——兼顾易用性与灵活性。

八、总结：核心价值与最佳实践
核心价值：无缝衔接 Python 数据生态：直接用 NumPy/PyTorch 生成校准数据
校准缓存机制：加速迭代开发，保证量化一致性；零代码 CLI 操作：降低 TensorRT 使用门槛
最佳实践：始终保存 --calibration-cache；校准数据 ≥ 100 样本，且覆盖真实场景；
校准后必做精度验证（用 polygraphy run --validate-with）
复用 API 示例代码：通过 script.py:function_name 语法

一句话总结：“polygraphy convert --int8 让 INT8 量化从‘专家专属’变为‘人人可用’。”
