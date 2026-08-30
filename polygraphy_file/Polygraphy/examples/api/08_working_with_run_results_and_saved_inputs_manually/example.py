"""以下代码是 Polygraphy（NVIDIA 的推理调试工具库）的一个典型示例，
用于手动加载并解析通过 polygraphy run 命令保存下来的推理输入（inputs）和输出结果（outputs）。下
一、整体目标：从 JSON 文件中加载之前保存的推理输入和多引擎输出结果，并以人类可读方式打印出来，便于调试或分析。
这是模型部署中常见的“事后分析”场景：你已经运行过一次推理（比如对比 ONNX Runtime 和 TensorRT），现在想离线查看具体数值。"""

"""
This script demonstrates how to use the `load_json` and `RunResults` APIs to load
and manipulate inference inputs and outputs respectively.
"""

from polygraphy.comparator import RunResults # Polygraphy 内置的专用数据结构，用于存储多个 runner（推理引擎）在多次迭代中的输出。
from polygraphy.json import load_json # Polygraphy 提供的安全 JSON 加载器，能正确处理其中嵌入的 base64 编码的 NumPy 数组。注意：普通 json.load() 无法解析这类文件！


def main():
    # Use the `load_json` API to load inputs from file.
    #
    # NOTE: The `save_json` and `load_json` standalone helpers should be used only with non-Polygraphy objects.
    # Polygraphy objects that support serialization include `save` and `load` methods.
    """步骤 1：加载、打印输入（Inputs）:文件来源：由 polygraphy run ... --save-inputs inputs.json 生成。
    数据类型：List[Dict[str, np.ndarray]];外层 List：支持多次推理迭代（例如跑 5 次不同随机输入）
    内层 Dict：每个推理的 feed_dict，key 是输入张量名（如 "X"），value 是 np.ndarray。典型结构示例：
    inputs = [
        {"X": np.array([[0.1, -0.5, 2.3]], dtype=np.float32)}   # 第1次推理的输入
        # 可能还有更多...
    ]"""
    inputs = load_json("inputs.json")

    # Inputs are stored as a `List[Dict[str, np.ndarray]]`, i.e. a list of feed_dicts,
    # where each feed_dict maps input names to NumPy arrays.
    #
    # TIP: In the typical case, we'll only have one iteration, so we'll only look at the first item.
    # If you need to access inputs from multiple iterations, you can do something like this instead:
    #
    #    for feed_dict in inputs:
    #        for name, array in feed_dict.items():
    #            ... # Do something with the inputs here
    #
    [feed_dict] = inputs # 这是 Python 的解包语法，等价于 feed_dict = inputs[0]；前提：inputs 列表长度必须为 1，否则会抛出 ValueError。
                    # 使用 [x] = list 是一种显式断言：强调“这里必须且只能有一个元素”，增强代码可读性和安全性。
    for name, array in feed_dict.items():
        print(f"Input: '{name}' | Values:\n{array}")

    # Use the `RunResults.load` API to load results from file.
    #
    # TIP: You can provide either a file path or a file-like object here.
    """步骤2，加载打印输出结果（RunResults：输入用 load_json()（因为不是 Polygraphy 对象）
    输出用 RunResults.load()（因为 RunResults 是 Polygraphy 自定义类，有专用序列化方法）"""
    results = RunResults.load("outputs.json")

    # The `RunResults` object is structured like a `Dict[str, List[IterationResult]]``,
    # mapping runner names to inference outputs from one or more iterations.
    # An `IterationResult` behaves just like a `Dict[str, np.ndarray]` mapping output names
    # to NumPy arrays.
    #
    # TIP: In the typical case, we'll only have one iteration, so we can unpack it
    # directly in the loop. If you need to access outputs from multiple iterations,
    # you can do something like this instead:
    #
    #    for runner_name, iters in results.items():
    #        for outputs in iters:
    #             ... # Do something with the outputs here
    #
    for runner_name, [outputs] in results.items():
        print(f"\nProcessing outputs for runner: {runner_name}")
        # Now you can read or modify the outputs for each runner.
        # For the sake of this example, we'll just print them:
        for name, array in outputs.items():
            print(f"Output: '{name}' | Values:\n{array}")


if __name__ == "__main__":
    main()
