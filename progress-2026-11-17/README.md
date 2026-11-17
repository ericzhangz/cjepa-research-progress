# cJEPA 研究进展 · 2026-11-17

本版收录 PushT 和 FourRoom 的近期核心实现、实验协议和结果数据。两条线都在检验同一件事：真实交互形成的持续支持，能不能通过 JEPA 的原生动作预测，改善后续候选动作的选择。

| 任务 | 读哪份报告 | 目前能确认什么 |
| --- | --- | --- |
| PushT：全视野 H4 | [独立底座实验](runs/fullview_pusht_base_20261115/REPORT.md) · [固定候选的历史差归因](runs/native_history_rollout_attribution_20261115/REPORT.md) | 64 个留出场景中，单模式微调把 H4 自由递推的位置 RMSE 从 25.78 降到 15.83 px。在另一个固定候选实验里，隐历史的 H4 跨动作作用仍弱。 |
| PushT：可逆制动 | [最近一次闭环小试](runs/native_full_depth_task_geometry_20261115/REPORT.md) · [后续固定候选测试](runs/native_full_depth_paired_effect_20261115/REPORT.md) | 闭环中持续支持和均匀支持均为 0/4 成功。随后在 16 个目标未完成的场景中，真实写入与均匀支持选出相同动作。 |
| FourRoom：官方静态参照 | [H-JEPA 官方 50 任务的平铺 LeWM 复现](runs/hjepa_official_fourroom_flat_20261115/REPORT.md) | 官方配置的单种子本地结果为 19/50。这个参照与自建的可逆门锁任务分别报告。 |
| FourRoom：自建可逆门锁 | [§47 累加难例训练](runs/native_topology_accumulated_20261116/REPORT.md) · [§48 固定候选溯源](runs/native_topology_objective_trace_20261116/REPORT.md) | §47 在其独立重采池中把首选假进房从 13/64 降到 0/64。§48 小试保住 Cand 380 的墙界修复，但未恢复 Cand 231 的合法绕行，连续反转也有退步。**H5 未过线；当前修订未跑模型 H6 或闭环。** |

- **PushT 源码**：[全视野训练](research/p1a/fullview_pusht_base.py) · [自由递推归因](research/p1a/native_history_rollout_attribution.py) · [可逆闭环小试](research/p1a/native_full_depth_task_geometry.py)
- **FourRoom 源码**：[原生条件预测](research/p1a/native_topology_condition.py) · [配对数据](research/p1a/native_topology_paired_data.py) · [离线训练](research/p1a/native_topology_prefix_fit.py) · [坐标观测似然](research/p1a/native_topology_xy_evidence.py) · [固定候选复算](runs/native_topology_objective_trace_20261116/fixed_eval.py)
- **跨实验阅读**：[机制与训练推导节选](research/CORE_REGULATION_MECHANISM.md) · [全视野底座技术报告](research/FULLVIEW_RESTART_REPORT_20261115.md) · [FourRoom 纯物理 H6 设计检验](runs/native_topology_live_20261116/physical_h6/REPORT.md)

各实验目录里的 JSON、CSV 保留所选逐场景结果、候选诊断或训练日志。源码摘自原工程；完整复现还需要原环境、预训练权重、数据和未收录的中间产物。旧快照在仓库中继续保留。
