# PRBD 单轨学习与跨动作预测 · 2026-11-20

[PRBD 单文件技术交接报告](PRBD_TECHNICAL_HANDOFF.md)收录 P0–P3、G 系列前置核验及 G1–G3 的完整信息流、发布版核心代码片段、实验中间量和逐动作结果分解。

本版发布 TwoRoom 两个连续阶段的实现、检查点和逐起点数据。P3b 完成 G1 数值检验、G2 零动作检验和 G3 单轨训练。G3 预先指定的多时域训练方案未通过完整写入的验收条件。G4 独立历史与 P4 闭环均未运行。[P0–P3 报告](runs/prbd_evidence_trajectory_20261119/REPORT.md) · [P3b 技术报告](runs/prbd_consistency_20261120/REPORT.md)

## G3 开环实测

62 条有效单轨用于训练，两臂各做 62 次更新。冻结预测器在 48 个新开发起点实际采集的轨迹上接受评价，42 个起点的三帧观察、近期命令和物理位置可以精确配对，其中 39 个起点有非零速度与动作交互。下表报告这 39 个起点的聚合有向恢复率。它度量写入带来的预测修订沿真实交互方向恢复了多少，分母为各起点真实交互向量平方范数之和。[协议](runs/prbd_consistency_20261120/PROTOCOL.md) · [逐起点 CSV](runs/prbd_consistency_20261120/paired_reset_results.csv)

| 训练和写入对照 | H1 | H4 |
|---|---:|---:|
| 多时域主方案，仅首次探测相对均匀 | +0.0574% | +0.0518% |
| 多时域主方案，完整写入相对均匀 | −0.0650% | −0.0284% |
| 多时域主方案，完整写入相对仅省略首次探测 | +0.0582% | +0.0526% |
| 单步训练对照，完整写入相对均匀 | +0.0713% | +0.0883% |

首次探测后再沿反向命令返回，主方案给后续正交动作的聚合修订转负。完整写入的 H1 与 H4 各有 35/39 和 34/39 个起点的交互误差下降，两个聚合有向恢复率的 95% 配对 bootstrap 区间均跨零。起点 2026112128 的 H4 有向分子在返回动作前后从 +0.100394 变为 −2.653770，其他 38 个起点在完整写入后合计为 +2.010235。42 个配对起点共 168 次静止到达均逐值保留写入前的支持。[主方案完整结果](runs/prbd_consistency_20261120/g3_episode_dual_measure.md) · [单步对照](runs/prbd_consistency_20261120/g3_episode_h1_measure.md) · [返回动作归因](runs/prbd_consistency_20261120/g3_influence.md)

## 核心代码和数据

| 内容 | 入口 |
|---|---|
| 连续观测似然、真实写入及独立积分复核 | [概率实现](research/p1a/prbd.py) · [数值复核](runs/prbd_consistency_20261120/G1_G3_AUDIT.md) |
| 原生动作响应差分、零动作保持 | [预测器](research/p1a/prbd_native.py) · [容量与静止检验](runs/prbd_consistency_20261120/g2.md) |
| 整回合原型边缘似然与多步训练 | [训练目标](research/p1a/prbd_public_fit.py) · [两臂拟合](research/p1a/prbd_consistency_fit.py) |
| 同窗配对评测、逐真实动作贡献 | [测量代码](research/p1a/prbd_consistency_measure.py) · [作用分解](research/p1a/prbd_consistency_influence.py) |
| 输入、权重、输出 | [P3b 轨迹、检查点、逐起点 JSON 与张量](runs/prbd_consistency_20261120/) · [P0–P3 数据](runs/prbd_evidence_trajectory_20261119/) |

CSV 含两臂、42 个配对起点、四个跨度和三种写入对照，共 1,008 行；未匹配的 6 个起点及原因保存在测量 JSON。复算原生前向需要作者发布的 LeWM 源码和 TwoRoom 宿主权重。此前的 PushT、PushObj 宿主核查保存在 11 月 19 日快照。[复算命令与外部资产](research/CODE_REPRODUCTION.md) · [11 月 19 日记录](../progress-2026-11-19/README.md)
