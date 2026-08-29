# ========== 基础延迟/吞吐测试 ==========
trtexec --loadEngine=/root/my_FILE/yolov8_int8_fixed_1.engine \
        --shapes=images:1x3x640x640 \
        --iterations=100 \
        --warmUp=500 \
        --duration=10 \
        --dumpProfile \
        --separateProfileRun \
        --exportTimes=timing_2.json \
        --exportProfile=profile_2.json

# timing.json文件的解读：
    字段	                      含义	                                  数值范围
startEnqMs / endEnqMs	    CPU 端调用 enqueueV3() 的起止时刻（ms）	            ~0.4 ms
startH2dMs / endH2dMs	    Host→Device 数据传输（输入图片拷入显存）起止时刻	      ~0.42 ms
startComputeMs / endComputeMs	GPU 上模型推理计算的起止时刻	                     ~0.47 ms
startD2hMs / endD2hMs	    Device→Host 数据传输（结果拷回内存）起止时刻	     ~0.03 ms
h2dMs	                    H2D 耗时 = endH2dMs - startH2dMs	                  0.42 ms
computeMs	        纯 GPU 推理耗时 = endComputeMs - startComputeMs	           0.47 ms
d2hMs	                D2H 耗时 = endD2hMs - startD2hMs	                  0.03 ms
latencyMs	        端到端总延迟 ≈ h2d + compute + d2h	                  ~0.93 ms

# ===============profile.json文件的解读：=================================
字段	           含义
count	        profiling 期间的总迭代次数（8961 次）
name	        层名称（可能是单层，也可能是 TRT 融合后的算子组）
timeMs	        该层在所有迭代中的累计耗时（ms）
averageMs	单次迭代平均耗时 = timeMs / count
medianMs	单次迭代中位数耗时（抗异常值干扰）
percentage	该层占所有层总耗时的百分比

# 总结profile.json文件中的Top-20 最耗时层
python3 -c "
import json
data = json.load(open('profile.json'))
layers = sorted([d for d in data if 'percentage' in d and d['percentage'] > 0], key=lambda x: -x['percentage'])[:20]
for l in layers:
    print(f\"{l['percentage']:6.2f}%\t{int(l['averageMs']*1000):>4}μs\t{l['name']}\")
"

# 按类型聚合（Conv / PWN / Quantize / Concat 等）
python3 -c "
import json
from collections import defaultdict

data = json.load(open('profile.json'))

groups = defaultdict(lambda: {'total_pct': 0.0, 'layers': 0})
for d in data:
    if d.get('timeMs', 0) <= 0 or 'name' not in d:
        continue
    name = d['name']
    if 'Conv' in name:
        t = 'Conv'
    elif 'PWN' in name:
        t = 'PointWise'
    elif 'Quantize' in name:
        t = 'Quantize'
    elif 'Concat' in name or 'copy' in name:
        t = 'MemoryOp'
    else:
        t = 'Other'
    groups[t]['total_pct'] += d.get('percentage', 0)
    groups[t]['layers'] += 1

print(f'{\"Type\":<12} {\"Total%\":>8} {\"Layers\":>8}')
print('-' * 32)
for t, v in sorted(groups.items(), key=lambda x: -x[1]['total_pct']):
    print(f'{t:<12} {v[\"total_pct\"]:>7.2f}% {v[\"layers\"]:>7d}')
"
输出的聚合解读：
类型	   占比	     层数	                                含义与判断
Conv	 68.93%	    64	   ✅ 正常且健康。YOLOv8 是 CNN 架构，卷积占绝对主导说明 INT8 量化后计算密集**                            型算子仍是瓶颈，GPU Tensor Core 被充分利用
MemoryOp  15.77%    28	  ⚠️ 偏高。包含 Concat、copy、Reshape 等纯内存搬运操作。理想情况下应 <10%。**                           这表示有大量数据重排/拼接无法被融合进计算 kernel
PointWise  6.88%    12	   ✅ 正常。Sigmoid/Mul/Add 等逐元素操作已被 TRT 融合为 PWN kernel，占比小说**                           明融合效果好
Other	  5.57%	    7	    包含 Resize、Softmax、Reduce 等特殊算子，占比小，无需关注
Quantize  2.85%	    4	   ✅ 极低。仅 4 个量化节点（输入量化 + 少量动态量化），说明大部分权重已静态量**                          化，运行时量化开销可忽略
--------------------------------------------------------------------------------------------
@@@@ 注：MemoryOp 层的搜索关键词是Concat 或 copy（见上面的命令），它的算子本质是纯内存搬运 kernel，不做算术运算，数值大可采用启用 --useCudaGraph，将 copy + compute 编排为固定 graph，消除调度抖动，同时降低 MemoryOp 占比；增大 batch size
@@@ 验证使用--useCudaGraph的成效：
# 不用 CUDA Graph
trtexec --loadEngine=/root/my_FILE/yolov8_int8.engine --iterations=200 2>&1 \
  | grep -E "(Latency:|Enqueue Time:|Throughput:|bound by)" \
  | tee benchmark_result_no.txt
# 用 CUDA Graph
trtexec --loadEngine=/root/my_FILE/yolov8_int8_fixed_1.engine --iterations=200 --useCudaGraph 2>&1 \
  | grep -E "(Latency:|Enqueue Time:|Throughput:|bound by)" \
  | tee benchmark_result_2.txt

验证时输出的各字段含义与作用：
Throughput（吞吐量）：每秒完成的推理次数（queries per second）
Latency（端到端延迟）：从调用 enqueueV2() 到拿到完整结果的总墙钟时间，包含：Enqueue + H2D + GPU Compute + D2H + 同步等待。
Enqueue Time（CPU 提交开销）：CPU 向 GPU 提交 kernel 指令的耗时
H2D Latency（Host → Device 传输）：输入数据从 CPU 内存拷贝到 GPU 显存的时间
D2H Latency（Device → Host 传输）：输出结果从 GPU 拷回 CPU 的时间
Warning（瓶颈提示）：TRT 自动判断的性能瓶颈

-----------------------------------------------------------------------------------------------
# 找出 median 与 average 偏差 >10% 的不稳定层
python3 -c "
import json
data = json.load(open('profile.json'))
results = []
for d in data:
    avg = d.get('averageMs', 0)
    med = d.get('medianMs', 0)
    if avg > 0 and abs(avg - med) / avg > 0.1:
        results.append(d)
if not results:
    print('No unstable layers found (all within 10% deviation)')
else:
    print(f'{\"Avg(μs)\":>10} {\"Med(μs)\":>10} {\"Dev%\":>8}  Layer')
    print('-' * 70)
    for r in sorted(results, key=lambda x: -abs(x['averageMs']-x['medianMs'])/x['averageMs']):
        dev = abs(r['averageMs'] - r['medianMs']) / r['averageMs'] * 100
        print(f'{r[\"averageMs\"]*1000:>9.1f} {r[\"medianMs\"]*1000:>9.1f} {dev:>7.1f}%  {r[\"name\"]}')
"
===========================================================================================

# ==============用trtexec创建引擎=======================================
1. 用fp32模型量化（采用了校准缓存），创建引擎并使用参数--profilingVerbosity=detailed可以后续打印详细层信息。
   采用这个直接构建的精度损失太多，应该采用先量化为插入Q/DQ节点的onnx模型，再创建trt引擎的方法。
trtexec --onnx=yolov8.onnx \
        --saveEngine=/root/my_FILE/yolov8_int8.engine \
        --int8 --fp16 \
        --calib=/root/my_FILE/data_loader.py \
        --profilingVerbosity=detailed

2. 使用量化后的插入Q/DQ节点的onnx模型创建引擎并采用参数--profilingVerbosity=detailed。
trtexec --onnx=/root/my_FILE/yolov8_int8.onnx \
        --saveEngine=/root/my_FILE/yolov8_int8_trtexec.engine \
        --int8 --fp16 \
        --profilingVerbosity=detailed
==================================================================================

# =======怎么用trtexec打印输出层结构，数据类型，算子等信息=========================
1. 如果在构造引擎时加上参数“--profilingVerbosity=detailed”，则在模型输出层结构时会输出详细信息，
如果不加，则只能打印出层的组成算子（大kernel的算子组成）：
第一个的输出文件比较正规：
trtexec --loadEngine=/root/my_FILE/yolov8_int8.engine \
        --exportLayerInfo=/root/my_FILE/layer_info_2.json

trtexec --loadEngine=/root/my_FILE/yolov8_int8_fixed_1.engine \
        --dumpLayerInfo > /root/my_FILE/layer_info_2.json
=========================================================================

# ========== 指定输入 shape（动态 batch）==========
trtexec --loadEngine=model.engine \
        --shapes=input:1x3x640x640 \
        --benchmark \
        --iterations=200

# ========== FP16 vs INT8 对比（需分别构建引擎后测试）==========
trtexec --loadEngine=model_fp16.engine --benchmark --dumpProfile
trtexec --loadEngine=model_int8.engine --benchmark --dumpProfile

# ========== GPU 利用率联动采集 ==========
# 终端1: 启动 nsys
nsys profile --trace=cuda,nvtx -o perf_report trtexec --loadEngine=model.engine --benchmark --iterations=500
# 生成 perf_report.qdrep，用 Nsight Systems GUI 打开分析


# 直接查看原始输出（不看过滤结果）
trtexec --loadEngine=/root/my_FILE/yolov8_int8_trtexec.engine --dumpLayerInfo 2>&1 | head -100

trtexec --loadEngine=/root/my_FILE/yolov8_int8.engine --dumpLayerInfo 2>&1 | grep -oP 'dtype=\w+' | sort | uniq -c

