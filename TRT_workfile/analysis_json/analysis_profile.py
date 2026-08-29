"""
对于构建好的引擎，我们可以采用D:\CV_FILE\my_trt_FILE\TRT_workfile\trt_Profiling\trtexec_Benchmark.md中的方法进行性能分析，
保存的Json文件为profile.json，通过以下代码进行解析，并输出总性能分析结果。
"""

import json
import sys

filepath = "profile.json"

with open(filepath) as f:
    data = json.load(f)

# 分离 summary 和 layer 条目
summary = next((d for d in data if "count" in d), None)
layers = [d for d in data if "percentage" in d and d["percentage"] > 0]

# 基础统计
count = summary["count"] if summary else 1
total_ms = sum(l.get("timeMs", l.get("averageMs", 0)) for l in layers)
avg_ms = total_ms / count if count > 0 else 0
fps = 1000.0 / avg_ms if avg_ms > 0 else 0

print(f"📊 Profiling Summary")
print(f"   Iterations:    {count}")
print(f"   Total Time:    {total_ms:.3f} ms")
print(f"   Avg Latency:   {avg_ms:.3f} ms/iter")
print(f"   Throughput:    {fps:.1f} FPS")

# Top20 瓶颈层（来自你验证过的排序逻辑）
sorted_layers = sorted(layers, key=lambda x: -x["percentage"])[:20]
print(f"\n🔥 Top20 Layers by Percentage:")
print(f"{'Pct':>7}\t{'Avg':>8}\tName")
print("-" * 70)
for l in sorted_layers:
    pct = l["percentage"]
    avg_us = int(l["averageMs"] * 1000)
    name = l["name"]
    print(f"{pct:6.2f}%\t{avg_us:>5}μs\t{name}")