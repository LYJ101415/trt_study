"""
打开onnx推理和trt推理的保存的json文件，对比两个runner的输出误差
"""

import numpy as np
from polygraphy.comparator import RunResults, Comparator

def compare_runner_outputs(json_path, rtol=1e-3, atol=1e-3):
    """
    自动加载 Polygraphy 结果并对比不同 Runner 的输出误差
    """
    # 1. 加载推理结果
    results = RunResults.load(json_path)
    
    # 2. 提取各个 Runner 的第一次迭代结果 (Iteration 0)
    # results 是一个字典，key 为 runner 名称，value 为 IterationResult 列表
    runner_names = list(results.keys())
    if len(runner_names) < 2:
        print("[Error] 结果文件中至少需要包含两个 Runner 的输出才能进行对比。")
        return

    # 假设第一个是 TRT，第二个是 ONNXRT（或者根据实际名称提取）
    trt_name = [name for name in runner_names if 'trt' in name][0]
    onnxrt_name = [name for name in runner_names if 'onnxrt' in name][0]

    trt_outputs = results[trt_name][0]      # 取第一次迭代
    onnxrt_outputs = results[onnxrt_name][0] # 取第一次迭代

    # 3. 遍历输出张量进行对比
    for output_name in trt_outputs.keys():
        print(f"\n{'='*50}")
        print(f"Comparing Output: '{output_name}'")
        
        trt_array = trt_outputs[output_name]
        onnxrt_array = onnxrt_outputs[output_name]

        # 基础检查：Shape 和 Dtype 是否一致
        if trt_array.shape != onnxrt_array.shape:
            print(f"  [FAILED] Shape mismatch: TRT {trt_array.shape} vs ONNXRT {onnxrt_array.shape}")
            continue

        # 4. 计算误差
        abs_diff = np.abs(trt_array - onnxrt_array)
        max_abs_error = np.max(abs_diff)
        mean_abs_error = np.mean(abs_diff)

        # 计算相对误差 (加上极小值防止除零)
        rel_diff = abs_diff / (np.abs(onnxrt_array) + 1e-8)
        max_rel_error = np.max(rel_diff)

        # 5. 使用 Polygraphy 的容差标准判断是否通过
        is_close = np.allclose(trt_array, onnxrt_array, rtol=rtol, atol=atol)
        status = "[PASSED]" if is_close else "[FAILED]"

        # 6. 打印统计信息
        print(f"  Status: {status} (Tolerance: rtol={rtol}, atol={atol})")
        print(f"  Max Absolute Error:  {max_abs_error:.6f}") # 最大绝对误差
        print(f"  Mean Absolute Error: {mean_abs_error:.6f}") # 平均绝对误差
        print(f"  Max Relative Error:  {max_rel_error:.6e}") # 最大相对误差

if __name__ == "__main__":
    # 替换为你的实际 JSON 文件路径
    compare_runner_outputs("/root/autodl-tmp/inference_results_batch.json", rtol=1e-3, atol=1e-3)