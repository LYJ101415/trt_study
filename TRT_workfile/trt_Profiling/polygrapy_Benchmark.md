# ========== 基础 benchmark ==========
polygraphy benchmark model.engine \
    --model-type=engine \
    --iterations=100 \
    --warmup=500 \
    --save-outputs=baseline.json

# ========== 多后端性能对比（TRT vs ONNX RT vs PyTorch）==========
polygraphy benchmark model.onnx \
    --trt --onnxrt --torch \
    --atol=1e-3 --rtol=1e-3 \
    --validate \
    --save-results=comparison.json

# ========== 批量 shape 扫描 ==========
polygraphy benchmark model.engine \
    --model-type=engine \
    --model-inputs input:[1,3,640,640] \
    --model-inputs input:[4,3,640,640] \
    --model-inputs input:[8,3,640,640] \
    --save-results=batch_scan.json

# ========== Profiling + 可视化 ==========
polygraphy profile model.engine \
    --model-type=engine \
    --save-profile=profile.json

# 将 profile 转为可读表格
polygraphy inspect profile profile.json --show layers --top 20

# ========== 集成 nsys ==========
nsys profile --trace=cuda,nvtx -o poly_perf \
    polygraphy benchmark model.engine --model-type=engine --iterations=500

# ================层信息=============
polygraphy inspect model /root/my_FILE/yolov8_int8.engine --show layers attrs