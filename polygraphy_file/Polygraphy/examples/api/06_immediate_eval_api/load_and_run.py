"""以下代码展示了如何 从磁盘加载已序列化的 TensorRT 引擎（.engine 文件）并执行推理，
是模型部署阶段的核心操作。"""
"""
This script uses Polygraphy's immediately evaluated functional APIs
to load the TensorRT engine built by `build_and_run.py` and run inference.
"""
import numpy as np
from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import TrtRunner, engine_from_bytes

"""bytes_from_path(path)：一个通用工具函数，从指定路径读取文件内容为字节流（bytes）。
不限于 TensorRT，也可用于 ONNX、权重文件等。等价于原生代码：
with open("identity.engine", "rb") as f:
    engine_bytes = f.read()"""
def main():
    """下句代码执行流程：bytes_from_path("identity.engine")→ 读取 "identity.engine" 文件内容为 bytes 对象。
    engine_from_bytes(...)→ 调用 TensorRT 的 runtime.deserialize_cuda_engine(engine_bytes)
    → 返回可执行的 ICudaEngine 实例。为什么需要序列化？
    TensorRT 引擎构建过程耗时（可能数秒至数分钟），且依赖特定 GPU 架构。预编译 + 序列化 可实现：
    快速启动：部署时直接加载，无需在线构建；跨设备兼容：同一 .engine 文件可在相同架构 GPU 上运行
    版本控制：将引擎作为制品（artifact）纳入 CI/CD 流程"""
    engine = engine_from_bytes(bytes_from_path("identity.engine"))

    with TrtRunner(engine) as runner:
        inp_data = np.ones((1, 1, 2, 2), dtype=np.float32)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        outputs = runner.infer(feed_dict={"x": inp_data})
        """验证结果并输出成功信息"""
        assert np.array_equal(outputs["output"], inp_data)  # It's an identity model!

        print("Inference succeeded!")


if __name__ == "__main__":
    main()

"""
实际应用场景示例
场景 1：Web 服务部署（FastAPI 示例）
from fastapi import FastAPI
import uvicorn

app = FastAPI()
runner = None

@app.on_event("startup")
async def load_engine():
    global runner
    engine = engine_from_bytes(bytes_from_path("model.engine"))
    runner = TrtRunner(engine)
    runner.__enter__()  # 手动初始化

@app.post("/predict")
async def predict(data: list):
    input_array = np.array(data, dtype=np.float32).reshape(1, 3, 224, 224)
    output = runner.infer({"input": input_array})
    return {"prediction": output["output"].tolist()}

@app.on_event("shutdown")
async def unload_engine():
    if runner:
        runner.__exit__(None, None, None)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
优势：引擎在服务启动时加载一次;每次请求直接推理，延迟极低

场景 2：批量推理（注意缓冲区复用问题）
# 错误做法：结果会被覆盖！
results = []
for batch in data_batches:
    out = runner.infer({"x": batch})
    results.append(out["output"])  # 所有元素指向同一内存！

# 正确做法：深拷贝
results = []
for batch in data_batches:
    out = runner.infer({"x": batch})
    results.append(out["output"].copy())  # 或 np.copy()

场景 3：多模型流水线
# 加载多个引擎
encoder_engine = engine_from_bytes(bytes_from_path("encoder.engine"))
decoder_engine = engine_from_bytes(bytes_from_path("decoder.engine"))

with TrtRunner(encoder_engine) as enc_runner, \
     TrtRunner(decoder_engine) as dec_runner:

    latent = enc_runner.infer({"input": image})["latent"]
    output = dec_runner.infer({"latent": latent})["output"]

安全与性能注意事项:
问题	                   解决方案
输入 shape 不匹配	使用 polygraphy inspect model identity.engine 确认输入 shape
FP16 精度损失	    若需高精度，构建引擎时禁用 fp16=True
多线程安全	        每个线程应持有独立的 TrtRunner 实例（因 IExecutionContext 非线程安全）
GPU 内存不足	    监控 nvidia-smi，必要时减小 batch size

总结：该脚本的核心价值
特性	           说明
最小化部署代码	仅需 10 行核心代码即可加载引擎并推理
自动内存管理	TrtRunner 上下文管理器确保资源安全释放
生产就绪	    直接适用于 Web 服务、边缘设备、批处理系统
无缝衔接构建阶段	与 build_and_run.py 形成完整 MLOps 闭环

最佳实践：
构建阶段：用 Functional API 生成 .engine 文件
部署阶段：用此脚本加载引擎，实现 “零构建开销”推理
监控阶段：结合 polygraphy inspect 验证引擎正确性
"""
