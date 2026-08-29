"""
查找经过onnx推理和trt推理后保存的.json文件，输出结果中最大误差的坐标，并打印出来
"""

import numpy as np
from polygraphy.comparator import RunResults

def debug_max_error(json_path, output_name="output0"):
    # 1. 加载推理结果
    results = RunResults.load(json_path)
    runner_names = list(results.keys())
    
    # 自动识别 TRT 和 ONNXRT runner
    trt_name = next((name for name in runner_names if 'trt' in name), None)
    onnxrt_name = next((name for name in runner_names if 'onnxrt' in name), None)
    
    if not trt_name or not onnxrt_name:
        print("[Error] 无法在结果中找到 TRT 或 ONNXRT runner。")
        return

    # 获取第一次迭代的结果
    trt_array = results[trt_name][0][output_name]
    onnxrt_array = results[onnxrt_name][0][output_name]

    # 2. 计算绝对误差并定位最大误差的坐标
    abs_diff = np.abs(trt_array - onnxrt_array)
    # np.unravel_index 会将一维的最大值索引转换回多维坐标 (如 batch, class, anchor)
    max_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)
    
    max_abs_err = abs_diff[max_idx]
    trt_val = trt_array[max_idx]
    onnxrt_val = onnxrt_array[max_idx]

    # 3. 打印核心诊断信息
    print(f"\n{'='*60}")
    print(f"🔍 最大误差定位分析 (Output: '{output_name}')")
    print(f"{'='*60}")
    print(f"  最大绝对误差坐标 (Index): {max_idx}")
    print(f"  最大绝对误差值:           {max_abs_err:.6f}")
    print(f"  TRT 在该位置的值:         {trt_val:.6f}")
    print(f"  ONNXRT 在该位置的值:      {onnxrt_val:.6f}")
    
    # 4. 检查是否涉及极小值 (相对误差爆炸的常见原因)
    if abs(onnxrt_val) < 1e-4:
        print(f"\n  ⚠️  警告: ONNXRT 基准值接近于 0，这很可能是导致相对误差爆炸的原因！")

    # 5. 打印该坐标周围的上下文 (帮助判断是孤立异常还是区域异常)
    print(f"\n  📊 误差分布上下文 (以最大误差点为中心的 3x3 区域):")
    # 注意：这里仅展示前两个维度，如果维度较高会自动截断
    try:
        # 构建切片对象，提取中心点周围的数据
        slices = []
        for dim, idx in enumerate(max_idx):
            start = max(0, idx - 1)
            end = min(abs_diff.shape[dim], idx + 2)
            slices.append(slice(start, end))
        
        context_errors = abs_diff[tuple(slices)]
        print(context_errors)
    except Exception as e:
        print(f"  无法提取上下文: {e}")

if __name__ == "__main__":
    # 替换为你的实际 JSON 文件路径
    debug_max_error("/root/autodl-tmp/inference_results_batch.json")