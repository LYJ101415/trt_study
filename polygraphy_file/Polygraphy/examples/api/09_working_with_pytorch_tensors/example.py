"""以下代码展示了如何在 Polygraphy + TensorRT 中直接使用 PyTorch 张量进行 INT8 量化校准与推理，
并充分利用 GPU 内存避免冗余拷贝。"""
"""
This script demonstrates how to use PyTorch tensors with the TensorRT runner and calibrator.
"""

import torch

from polygraphy.backend.trt import (
    Calibrator,
    CreateConfig,
    TrtRunner,
    engine_from_network,
    network_from_onnx_path,
)

# If your PyTorch installation has GPU support, then we'll allocate the tensors
# directly in GPU memory. This will mean that the calibrator and runner can skip the
# host-to-device copy we would otherwise incur with NumPy arrays.
""" 1.自动检测是否有 CUDA GPU，决定张量分配在 CPU 还是 GPU。若 PyTorch 支持 GPU，
则所有 tensor 直接在 GPU 上创建 → 后续校准/推理可跳过 Host-to-Device 拷贝！"""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

"""2.校准数据生成器（支持 PyTorch Tensor）,返回一个 generator，每次 yield 一个 feed_dict。
注意：这里直接使用 torch.Tensor，并指定 device=DEVICE（GPU 优先）。
Polygraphy 的 Calibrator 能直接消费这些 GPU tensor，无需转为 NumPy！"""
def calib_data():
    for _ in range(4): # 校准通常只需少量样本（如 100~500），这里简化为 4 个
        yield {"x": torch.ones((1, 1, 2, 2), dtype=torch.float32, device=DEVICE)}


def main():
    """ 3.INT8 校准器直接接收 PyTorch Tensors,Calibrator(data_loader=...) 接收一个可迭代对象（如 generator/list），
    其中每个元素是 {"input_name": torch.Tensor}。优势：若 tensor 在 GPU 上，校准过程全程不经过 CPU！"""
    calibrator = Calibrator(data_loader=calib_data())

    """4：构建 INT8 TensorRT 引擎: - 从 ONNX 文件加载网络结构;
    - 启用 int8 量化，并传入上述校准器;- 引擎构建时会自动运行校准数据，生成 scale/activation 统计信息"""
    engine = engine_from_network(
        network_from_onnx_path("identity.onnx"),  # 一个“输出=输入”的简单模型
        config=CreateConfig(int8=True, calibrator=calibrator),
    )
    """5：使用 TrtRunner 进行推理（支持 PyTorch 输入/输出）"""
    with TrtRunner(engine) as runner:
        """ 创建推理输入：直接使用 GPU tensor（若 DEVICE="cuda"）"""
        inp_data = torch.ones((1, 1, 2, 2), dtype=torch.float32, device=DEVICE)

        # NOTE: The runner owns the output buffers and is free to reuse them between `infer()` calls.
        # Thus, if you want to store results from multiple inferences, you should use `copy.deepcopy()`.
        #
        # When you provide PyTorch tensors in the feed_dict, the runner will try to use
        # PyTorch tensors for the outputs. Specifically:
        # - If the `copy_outputs_to_host` argument to `infer()` is set to `True` (the default),
        #       it will return PyTorch tensors in CPU memory.
        # - If `copy_outputs_to_host` is `False`, it will return:
        #       - PyTorch tensors in GPU memory if you have a GPU-enabled PyTorch installation.
        #       - Polygraphy `DeviceView`s otherwise.
        """6：关键行为 —— copy_outputs_to_host 默认为 True: runner.infer() 的默认行为：
           - 输入是 PyTorch tensor → 输出也尝试返回 PyTorch tensor; - BUT: 默认会将输出从 GPU 拷贝回 CPU（copy_outputs_to_host=True）
           - 所以 outputs["y"] 是 CPU 上的 torch.Tensor"""
        outputs = runner.infer({"x": inp_data})

        # `copy_outputs_to_host` defaults to True, so the outputs should be PyTorch
        # tensors in CPU memory.
        """验证输出类型和位置"""
        assert isinstance(outputs["y"], torch.Tensor) #  是 PyTorch 张量
        assert outputs["y"].device.type == "cpu" #  在 CPU 上（因默认拷贝）

        """验证功能正确性（identity 模型：输出应等于输入）"""
        assert torch.equal(outputs["y"], inp_data.to("cpu")) # 数值一致。It's an identity model!


if __name__ == "__main__":
    main()
