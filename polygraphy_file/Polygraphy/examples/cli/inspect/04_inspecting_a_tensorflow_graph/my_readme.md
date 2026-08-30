检查 TensorFlow 图
简介
inspect model 子工具可以显示 TensorFlow 图。
运行示例:
检查 TensorFlow 冻结模型：
polygraphy inspect model identity.pb --model-type=frozen

这将显示类似如下的内容：
[I] ==== TensorFlow Graph ====
    ---- 1 Graph Inputs ----
    {Input:0 [dtype=float32, shape=(1, 15, 25, 30)]}

    ---- 1 Graph Outputs ----
    {Identity_2:0 [dtype=float32, shape=(1, 15, 25, 30)]}

    ---- 4 Nodes ----