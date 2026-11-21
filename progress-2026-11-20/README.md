# PRBD 单轨学习与跨动作预测 · 2026-11-20 至 11-21

## 11 月 21 日 · Phase 2 修订复验

在同一 TwoRoom 面板上分别测量了两件事：给定真速度时分支能预测到什么程度，以及只从实际观察形成支持时，后续跨动作预测获得多少收益。两组训练各完成 2,048 次更新。无标签模型的原型 0 在 H1–H4 与冻结宿主逐值相等。[完整技术报告](runs/prbd_phase2_20261121/repair_REPORT.md) · [训练与验收协议](runs/prbd_phase2_20261121/repair_PROTOCOL.md)

| 方法 | 慢→快后 MSE | 快→慢后 MSE | H4 跨动作交互 SSE |
|---|---:|---:|---:|
| 给定真速度训练并选分支的容量对照 | 0.02494 | 0.02688 | 29.17 |
| 无标签修订，同响应均匀支持 | 0.06449 | 0.06242 | 71.60 |
| 无标签修订，真实写入 | 0.06410 | 0.06231 | 72.46 |
| AdaJEPA | 0.04986 | 0.03528 | 62.39 |

34 个同窗根上，完整写入的 H4 有向恢复为 +1.37%，16 个根的交互 SSE 改善。156 次静止到达中，写入 TV≤0.01 有 0 次，TV>0.5 有 11 次。Phase 2 的五项预定验收中两项通过，Phase 3 尚未启动。[逐根评分](runs/prbd_phase2_20261121/repair_static_physical_nonzero_roots.json) · [逐次反转](runs/prbd_phase2_20261121/repair_switch_full_events.json) · [验收结果](runs/prbd_phase2_20261121/repair_admission.json)

修订的机制、两次梯度审计和复算入口见[机制 §52](research/PRBD_PHASE1_PHASE2_MECHANISM.md) · [旧检查点审计](runs/prbd_phase2_20261121/repair_audit__ORIGINAL_CHECKPOINT_AUDIT.md) · [正则量级审计](runs/prbd_phase2_20261121/repair_audit__REPAIR_REGULARIZER_AUDIT.md) · [代码与复算说明](research/CODE_REPRODUCTION.md#phase-2-repair).

## 11 月 21 日 · 原 Phase 1 与 Phase 2

| 阶段 | 实际测量 | 报告与数据 |
|---|---|---|
| Phase 1 · 给定真速度的表达容量 | 44 个独立有效根，A / B 的 H4 latent 有向恢复率为 31.98% / 45.17%，同一冻结坐标读出的物理恢复率为 27.38% / 24.39%。 | [容量与观测核报告](runs/prbd_phase1_20261121/REPORT.md) |
| Phase 2 · 无标签静止与跨动作预测 | 单轨训练 2,048 次更新。156 次真实静止到达精确保持 0 次，写入 TV≤0.01 共 68 次。34 个独立非零交互根上，完整写入相对均匀支持的 H4 latent 有向恢复率为 −0.0733%，交互 SSE 增加 0.185%。 | [完整技术报告](runs/prbd_phase2_20261121/REPORT.md) · [逐根结果](runs/prbd_phase2_20261121/static_physical_nonzero_roots.json) |
| Phase 2 · 回合内速度反转 | 另 48 个独立重置，两向各 96 次反转。PRBD 在慢→快 / 快→慢分别恢复 67 / 56 次，同响应均匀对照为 68 / 55 次。恢复指连续两个可辨动作均预报正确。 | [全部方法结果](runs/prbd_phase2_20261121/switch_results.json) · [逐反转事件](runs/prbd_phase2_20261121/switch_full_events.json) · [四种基线](runs/prbd_phase2_20261121/baselines/REPORT.md) |

Phase 2 的公开材料包括官方模拟器轨迹及编码、原型权重、每步训练记录、各方法签发的预测、静态逐根误差和双向切换事件。归档预测可直接用于复算；重新训练及签发需要官方 LeWM 宿主。此次独立测试停在 Phase 2。[机制定义](research/PRBD_PHASE1_PHASE2_MECHANISM.md) · [核心实现](research/p1a/prbd_phase2.py) · [实验协议](runs/prbd_phase2_20261121/PROTOCOL.md) · [训练过程](runs/prbd_phase2_20261121/A/updates.jsonl) · [复算说明](research/CODE_REPRODUCTION.md)

## 11 月 20 日记录

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
