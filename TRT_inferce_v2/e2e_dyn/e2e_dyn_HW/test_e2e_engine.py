"""test_e2e_engine.py — 验证 INT8 e2e engine：动态 uint8 图 + 运行时阈值，真实推理出框。"""
import sys
from pathlib import Path

import numpy as np
import cv2
import tensorrt as trt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # TRT_inferce/
from common import cuda_malloc, device_addr, cuda_memcpy_h2d, cuda_memcpy_d2h  # noqa: E402


def main(engine_path='/root/my_FILE/models/yolov8_int8_e2e.engine',
         img_path='/root/my_FILE/my_trt_FILE/datasets/Data_DeepPCB_YOLO/images/test/00041200.jpg'):
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    print("engine I/O:")
    in_names, out_names = [], []
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            in_names.append(n)
            print(f"  IN  {n}: dtype={engine.get_tensor_dtype(n)}")
        else:
            out_names.append(n)
            print(f"  OUT {n}: dtype={engine.get_tensor_dtype(n)}")

    img = cv2.imread(img_path)
    scale = 640.0 / max(img.shape[:2])
    raw = np.ascontiguousarray(cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)))[None])

    ctx.set_input_shape('image_raw', raw.shape)

    host = {}

    # 分配显存并绑定地址
    dev = {}
    for i in range(engine.num_io_tensors):
        n = engine.get_tensor_name(i)
        shp = tuple(ctx.get_tensor_shape(n))
        dt = trt.nptype(engine.get_tensor_dtype(n))
        nb = max(int(np.prod(shp)) * np.dtype(dt).itemsize, 4)
        dev[n] = cuda_malloc(nb)
        ctx.set_tensor_address(n, device_addr(dev[n]))
        host[n] = np.empty(shp, dtype=dt) if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT else None

    host['image_raw'] = raw
    iou_in = np.array([0.65], np.float32)
    score_in = np.array([0.45], np.float32)

    for n, val in [('image_raw', host['image_raw']), ('iou_thresh', iou_in), ('score_thresh', score_in)]:
        cuda_memcpy_h2d(dev[n], val)

    ok = ctx.execute_async_v3(0)  # 默认 stream
    print("execute:", "OK" if ok else "FAILED")

    for n in out_names:
        cuda_memcpy_d2h(host[n], dev[n])
        dets = host[n]
        valid = dets[dets[:, 4] > 0]
        print(f"{n}: {len(valid)} 框 (score_thresh=0.45)")
        for row in valid[:5]:
            print(f"  box={np.round(row[:4], 1)}, conf={row[4]:.3f}, cls={int(row[5])}")


if __name__ == '__main__':
    main(*sys.argv[1:])
