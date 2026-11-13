# cJEPA 研究交接 · 2026-11-13

这批资料记录 PushT LeWM 的单单元预测调节机制。研究暂停于两项固定检查点对照之后。用同一个物理后果目标训练动作相关性已写入推导，等待实验。评审可从报告读到运行入口，再按场景、物理世界和候选动作追查数值。 [外审报告](research/CJEPA_PROGRESS_REVIEW_20261113.md) · [实验设计与代码执行流程](research/EXPERIMENT_DESIGN.md) · [机制推导 §16](research/CORE_REGULATION_MECHANISM.md)

| 要读的内容 | 入口 |
| --- | --- |
| 原生计算中如何形成三种响应、更新真实信念 | [LeWM 末块单元](research_sources_2026-10-22/lewm/module.py#L353) · [JEPA 预测与规划接口](research_sources_2026-10-22/lewm/jepa.py#L62) |
| 真实观测、训练损失和自由推演 | [发出与写入](research/p1a/native_response_mixture_pilot.py#L108) · [候选推演](research/p1a/native_response_mixture_pilot.py#L136) · [训练](research/p1a/native_response_mixture_pilot.py#L219) |
| 两级监督、同检查点保留干预 | [直接监督报告](runs/native_response_mixture_direct_20261113/REPORT.md) · [物理监督报告](runs/native_response_mixture_physical_20261113/REPORT.md) · [历史开发逐根证据](runs/native_response_mixture_physical_20261113/development_evidence.csv) · [保留逐场景误差](runs/native_response_mixture_physical_20261113/retention_by_scene.csv) |
| 步 20 查询、步 30 的 H2 合并与展开 | [评估代码](research/p1a/native_response_mixture_rollout.py) · [报告](runs/native_response_mixture_rollout_20261113/REPORT.md) · [逐候选预测](runs/native_response_mixture_rollout_20261113/candidate_predictions.csv) · [匹配世界分差](runs/native_response_mixture_rollout_20261113/matched_candidate_pairs.csv) |
| 退离、停留与返回 | [物理采集](research/p1a/native_response_mixture_lifecycle_data.py) · [评估代码](research/p1a/native_response_mixture_lifecycle.py) · [报告](runs/native_response_mixture_lifecycle_20261113/REPORT.md) · [逐候选物理记录与预测](runs/native_response_mixture_lifecycle_20261113/candidate_predictions.csv) · [信念时序](runs/native_response_mixture_lifecycle_20261113/belief_trajectory.csv) |

CSV 来自已保存的实验产物，保留逐批损失、三模式似然及信念、预测和真实位置、候选比较分差，以及回放中的接触与执行标记。字段、统计单位和完整复跑所需资产见实验设计。 [数据说明](research/EXPERIMENT_DESIGN.md#公开数值记录) · [LeWM 上游](https://github.com/lucas-maes/le-wm) · [上游 MIT 许可](research_sources_2026-10-22/lewm/LICENSE)
