"""
cuda_rt.py — CUDA Runtime 扩展封装：Stream / 异步拷贝 / 事件 / 锁页内存。

common.py 只提供同步 cudaMemcpy；本文件补充流水线部署需要的异步原语:
  - CudaStream:  流（拷贝与 kernel 执行在流内按序，可与 CPU 并行）
  - async_h2d / async_d2h:  异步内存拷贝（源/目标须为锁页内存）
  - CudaEvent:   事件（标记流进度，供消费者等待某批次完成）
  - PinnedBuffer: cudaHostAlloc 分配的锁页内存（DMA 直达，比 pageable 快且支持异步）

与 common.py 一致地使用 ctypes 直连 libcudart，无第三方依赖。
"""

import ctypes

import numpy as np

_cudart = ctypes.CDLL("libcudart.so")


def _fn(name, argtypes, restype=ctypes.c_int):
    f = getattr(_cudart, name)
    f.argtypes = argtypes
    f.restype = restype
    return f


_c_stream_create = _fn("cudaStreamCreate", [ctypes.POINTER(ctypes.c_void_p)])
_c_stream_destroy = _fn("cudaStreamDestroy", [ctypes.c_void_p])
_c_stream_sync = _fn("cudaStreamSynchronize", [ctypes.c_void_p])
_c_stream_query = _fn("cudaStreamQuery", [ctypes.c_void_p])
_c_memcpy_async = _fn("cudaMemcpyAsync", [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p])
_c_event_create = _fn("cudaEventCreateWithFlags", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint])
_c_event_destroy = _fn("cudaEventDestroy", [ctypes.c_void_p])
_c_event_record = _fn("cudaEventRecord", [ctypes.c_void_p, ctypes.c_void_p])
_c_event_query = _fn("cudaEventQuery", [ctypes.c_void_p])
_c_event_sync = _fn("cudaEventSynchronize", [ctypes.c_void_p])
_c_host_alloc = _fn("cudaHostAlloc", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint])
_c_host_free = _fn("cudaFreeHost", [ctypes.c_void_p])

H2D, D2H = 1, 2   # cudaMemcpyKind


def _check(err):
    if err != 0:
        raise RuntimeError(f"CUDA runtime error: {err}")


class CudaStream:
    """CUDA 流。with 语句可用，析构时自动销毁。"""

    def __init__(self):
        ptr = ctypes.c_void_p()
        _check(_c_stream_create(ctypes.byref(ptr)))
        self._ptr = ptr

    @property
    def handle(self) -> int:
        """可传给 trt IExecutionContext.execute_async_v3(stream) 的流句柄"""
        return self._ptr.value

    def sync(self):
        _check(_c_stream_sync(self._ptr))

    def query(self) -> bool:
        """True = 流中所有工作已完成"""
        return _c_stream_query(self._ptr) == 0

    def destroy(self):
        if self._ptr is not None:
            _check(_c_stream_destroy(self._ptr))
            self._ptr = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.destroy()


class CudaEvent:
    """CUDA 事件：record 后可 query / sync 等待该点之前的流工作全部完成。"""

    def __init__(self):
        ptr = ctypes.c_void_p()
        _check(_c_event_create(ctypes.byref(ptr), 0))   # 0 = 默认阻塞式同步
        self._ptr = ptr

    def record(self, stream: CudaStream):
        _check(_c_event_record(self._ptr, stream._ptr))

    def query(self) -> bool:
        return _c_event_query(self._ptr) == 0

    def sync(self):
        _check(_c_event_sync(self._ptr))

    def destroy(self):
        if self._ptr is not None:
            _check(_c_event_destroy(self._ptr))
            self._ptr = None


class PinnedBuffer:
    """
    锁页(host pinned)缓冲区，按 uint8 一维分配，用 view() 按需解释成任意 dtype/shape。
    cudaHostAlloc 的内存天然 page-locked，可直接做异步拷贝源/目标。
    """

    def __init__(self, nbytes: int):
        ptr = ctypes.c_void_p()
        _check(_c_host_alloc(ctypes.byref(ptr), nbytes, 0))
        self._ptr = ptr
        self._nbytes = nbytes
        buf = (ctypes.c_char * nbytes).from_address(ptr.value)
        self.arr = np.frombuffer(buf, dtype=np.uint8)   # 可写一维视图

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def dev_addr_src(self) -> int:
        """内存首地址（做 cudaMemset / 指针运算用）"""
        return self._ptr.value

    def view(self, dtype, shape=None) -> np.ndarray:
        """把缓冲区前若干字节解释为指定 dtype/shape 的可写数组"""
        v = self.arr.view(dtype)          # 整块重解释（一维连续，切片再整形安全）
        if shape is not None:
            return v[:int(np.prod(shape))].reshape(shape)
        return v

    def destroy(self):
        if self._ptr is not None:
            _check(_c_host_free(self._ptr))
            self._ptr = None


def async_h2d(stream: CudaStream, dst_dev: int, src: np.ndarray):
    """异步 Host→Device。src 必须是锁页内存（PinnedBuffer 的 view）且 C 连续。"""
    assert src.flags["C_CONTIGUOUS"], "src must be C-contiguous"
    _check(_c_memcpy_async(dst_dev, src.ctypes.data, src.nbytes, H2D, stream._ptr))


def async_d2h(stream: CudaStream, dst: np.ndarray, src_dev: int):
    """异步 Device→Host。dst 必须是锁页内存（PinnedBuffer 的 view）且 C 连续。"""
    assert dst.flags["C_CONTIGUOUS"], "dst must be C-contiguous"
    _check(_c_memcpy_async(src_dev, dst.ctypes.data, dst.nbytes, D2H, stream._ptr))
