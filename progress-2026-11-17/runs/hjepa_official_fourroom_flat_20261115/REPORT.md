# FourRoom 官方任务 · seed42 平面 LeWM

作者发布的 50 项任务全部完成。**19/50 项在 210 环境步内成功，成功率 38.0%。** 19 项成功任务首次到达平均为 167.58 步，最早 142 步，最晚 187 步。作者 Figure 6 的 LeWM 参考值为 40.7% ± 6.8%，统计口径是训练种子 42、43、44 的均值和标准误。 [作者结果及统计口径](https://github.com/kevinghst/H-JEPA/blob/main/docs/RESULTS.md#depth-comparison-figure-6)

原生评估器汇总的逐项成功及步数与 50 个原始记录、CSV 逐项表一致。有效记录共 1,100 次求解，每项 22 次，每次 200 个候选。AdamW 学习率 0.1，执行原作者 30–90 次停机规则。RTX 4060 Laptop GPU 上每项评估耗时中位数 256.63 秒，包含并发运行期间的 GPU 共享时间。 [作者配置](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/config/eval/fourroom_flat.yaml) · [求解器](https://github.com/kevinghst/H-JEPA/blob/main/stable_worldmodel/solver/gd.py)

本次以三进程运行彼此独立的单任务 chunk，对第 `i` 项沿用作者的规划器 seed `42 + i`。发布轨迹没有逐步环境种子字段，作者评估器保留默认环境随机性，汇总 `seeds` 为 `null`。读取了官方发布包中评估所需的首尾 RGB 和全部非图像张量，各项保留原 76 帧长度与官方顺序。数值核验覆盖全部端点、任务元数据和发布 normalizer。CUDA 运行采用本机 PyTorch 2.8.0/cu128、torchvision 0.23.0/cu128 与作者要求的 Gymnasium 1.2.3。 [完整执行协议](PROTOCOL.md) · 输入核验（原工程资料）

- [逐项成功及步数](tasks.csv)
- 作者评估器汇总（原工程资料）
- 实际评估配置（原工程资料）
- 1,100 次逐次求解统计（原工程资料）
- 三个队列日志（原工程资料） · 队列 1（原工程资料） · 队列 2（原工程资料）
