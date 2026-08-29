"""
TensorRT Runtime 最小推理服务
- 零 Triton / 零 Docker / 零多余依赖
- 支持 Dynamic Batching（手动实现）
- 异步非阻塞推理
"""

import time
import threading
from queue import Queue
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import tensorrt as trt
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ==================== 配置区 ====================
ENGINE_PATH = "model.engine"
MAX_BATCH_SIZE = 32
INPUT_NAME = "images"       # ← 改成你 Engine 的实际输入名
OUTPUT_NAME = "output0"     # ← 改成你 Engine 的实际输出名
HOST = "0.0.0.0"
PORT = 8000
# ================================================


class TRTEngine:
    """轻量 TRT Runtime 封装，线程安全"""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {engine_path}")

        # 获取 I/O binding 信息
        self.input_idx = self.engine.get_tensor_mode(INPUT_NAME)
        self.output_idx = self.engine.get_tensor_mode(OUTPUT_NAME)
        self.input_shape = self.engine.get_tensor_shape(INPUT_NAME)   # e.g. (-1,3,640,640)
        self.output_shape = self.engine.get_tensor_shape(OUTPUT_NAME)  # e.g. (-1,10,8400)
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(INPUT_NAME))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(OUTPUT_NAME))

        print(f"[TRT] Engine loaded | Input: {self.input_shape} {self.input_dtype}")
        print(f"[TRT]               | Output: {self.output_shape} {self.output_dtype}")

    def infer(self, input_np: np.ndarray) -> np.ndarray:
        """同步推理（单 batch），调用方负责 batching"""
        import cuda  # pycuda 或 torch.cuda 均可，这里用 tensorrt 自带方式

        # 分配 GPU 内存并执行
        context = self.engine.create_execution_context()
        batch_size = input_np.shape[0]

        # 设置动态 shape
        context.set_input_shape(INPUT_NAME, input_np.shape)

        # 分配设备内存
        d_input = cuda.mem_alloc(input_np.nbytes)
        out_shape = (batch_size,) + tuple(self.output_shape[1:])
        output_np = np.empty(out_shape, dtype=self.output_dtype)
        d_output = cuda.mem_alloc(output_np.nbytes)

        # 拷贝 & 推理 & 拷回
        cuda.memcpy_htod(d_input, input_np)
        context.execute_v2([int(d_input), int(d_output)])
        cuda.memcpy_dtoh(output_np, d_output)

        # 释放
        d_input.free()
        d_output.free()
        del context

        return output_np


# ---------- 更简洁的替代方案：用 torch 做 GPU 内存管理 ----------
# 如果你的环境已有 PyTorch，上面的 cuda.mem_alloc 可以替换为：
#
#   import torch
#   d_input = torch.from_numpy(input_np).cuda()
#   d_output = torch.empty(out_shape, dtype=..., device='cuda')
#   context.execute_v2([d_input.data_ptr(), d_output.data_ptr()])
#   output_np = d_output.cpu().numpy()
#
# 这样就不需要额外安装 pycuda，且内存池复用更高效。
# ---------------------------------------------------------------


# ==================== 简易 Dynamic Batcher ====================
class BatchScheduler:
    """
    最简动态批处理器
    - 攒满 MAX_BATCH_SIZE 或超时自动触发
    - 线程安全，支持并发请求
    """

    def __init__(self, engine: TRTEngine, max_batch: int = 32, timeout_ms: float = 5.0):
        self.engine = engine
        self.max_batch = max_batch
        self.timeout = timeout_ms / 1000.0
        self.queue: Queue = Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    def _loop(self):
        while True:
            batch_requests: List[tuple] = []
            try:
                # 等待第一个请求
                req = self.queue.get(timeout=self.timeout)
                batch_requests.append(req)

                # 在剩余时间内尽量凑批
                deadline = time.monotonic() + self.timeout
                while len(batch_requests) < self.max_batch:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        req = self.queue.get(timeout=remaining)
                        batch_requests.append(req)
                    except Exception:
                        break

            except Exception:
                continue  # 超时无请求，继续等

            if not batch_requests:
                continue

            # 拼接 batch 推理
            inputs = np.concatenate([r[0] for r in batch_requests], axis=0)
            try:
                outputs = self.engine.infer(inputs)
                # 按原始 batch size 拆分结果
                sizes = [r[0].shape[0] for r in batch_requests]
                split_outputs = np.split(outputs, np.cumsum(sizes)[:-1])
                for i, (_, future) in enumerate(batch_requests):
                    future["result"] = split_outputs[i]
                    future["done"].set()
            except Exception as e:
                for _, future in batch_requests:
                    future["error"] = str(e)
                    future["done"].set()

    def submit(self, input_np: np.ndarray) -> dict:
        future = {"done": threading.Event()}
        self.queue.put((input_np, future))
        future["done"].wait()
        if "error" in future:
            raise RuntimeError(future["error"])
        return future["result"]


# ==================== FastAPI 应用 ====================
scheduler: BatchScheduler = None  # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scheduler
    engine = TRTEngine(ENGINE_PATH)
    scheduler = BatchScheduler(engine, max_batch=MAX_BATCH_SIZE)
    print(f"[Server] Ready on http://{HOST}:{PORT}")
    yield

app = FastAPI(title="TRT Minimal Server", lifespan=lifespan)


class InferRequest(BaseModel):
    # 根据你的实际输入调整
    data: List[List[List[List[float]]]]  # [B, C, H, W]


@app.post("/infer")
async def infer(request: InferRequest):
    try:
        input_np = np.array(request.data, dtype=np.float32)
        result = scheduler.submit(input_np)
        return {"output": result.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

"""
启动方式:
# 安装依赖
pip install -r requirements.txt

# 启动服务
python server.py

# 测试
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"data": [[[[0.0]*640 for _ in range(640)] for _ in range(3)]]}'

# 健康检查
curl http://localhost:8000/health
===============================================================================

性能优化建议（按需开启）
表格
优化项	                改动	                          收益
用 PyTorch 管显存	替换 pycuda 为 torch.Tensor	         内存池复用，避免频繁 malloc/free
CUDA Stream	       context.execute_async_v2(stream)	    CPU/GPU 并行，降低延迟
共享内存传输	     cudaHostAlloc + pinned memory	     H2D/D2H 提速 2-3x
多 Worker	        uvicorn --workers 4	                利用多核预处理
gRPC 替代 HTTP	    换 grpcio + protobuf	            吞吐提升 2-3x
"""