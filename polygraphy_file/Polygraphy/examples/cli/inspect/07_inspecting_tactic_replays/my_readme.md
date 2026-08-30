检查策略重放文件
简介
inspect tactics 子工具可以显示由 Polygraphy 生成的 TensorRT 策略重放文件的相关信息。
运行示例:
生成策略重放文件：
polygraphy run model.onnx --trt --save-tactics replay.json

检查策略重放：
polygraphy inspect tactics replay.json

这将显示类似如下的内容：
[I] Layer: ONNXTRT_Broadcast
        Algorithm: (Implementation: 2147483661, Tactic: 0) | Inputs: (('DataType.FLOAT'),) | Outputs: (('DataType.FLOAT'),)
    Layer: node_of_z
        Algorithm: (Implementation: 2147483651, Tactic: 1) | Inputs: (('DataType.FLOAT'), ('DataType.FLOAT')) | Outputs: (('DataType.FLOAT'),)