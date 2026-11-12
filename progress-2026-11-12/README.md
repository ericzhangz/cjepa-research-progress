# CJepa 研究进展 · 2026-11-12

本目录补充上一批交接之后的核心材料。建议先读[研究提案](research/RESEARCH_PROPOSAL.md)，再从下面的结果和代码追到对应计算。研究目标是在真实任务执行期间，让视觉—行动经历持续调节 LeWM 原生预测，并检验这种调节是否改善后续候选动作的真实后果预测。

## 代码与结果

| 阶段 | 已经测到什么 | 入口 |
| --- | --- | --- |
| FourRoom 源请求条件化反馈 | 原生 attention 能读取已经兑现的状态。步 20 刷新后的中心化位移误差比旧状态平均降低 1.70%，只有 15/32 个起点改善，聚合变化集中在一个起点 | [运行态实现](research/p1a/native_response.py) · [实验入口](research/p1a/query_feedback_source_branches.py) · [结果](runs/query_feedback_20261112/intermediates/source_branches/REPORT.md) · [逐起点配对差](runs/query_feedback_20261112/intermediates/source_branches/root_pairs.csv) |
| PushT 原生四事实联合项 | 两个上下文各执行两种动作，实际差分消去两类加性偏差。训练后独立候选的中心化位移误差平均改善约 0.0062 px，场景配对区间跨零 | [推导](research/methodology/NATIVE_JOINT_CYCLE_20261112.md) · [实现](research/p1a/native_joint_cycle.py) · [结果](runs/native_joint_cycle_20261112/REPORT.md) · [逐候选数据](runs/native_joint_cycle_20261112/pusht_trained/requests.csv) |
| PushT 推手到物体的位移传递 | 原生预测头可定向改变物体位移读出。使用真实推手位移逐根事后拟合，H2 仍有 12.06 px 的物体位移差残差 | [推导](research/methodology/NATIVE_CONTACT_TRANSFER_20261112.md) · [实验入口](research/p1a/contact_transfer_stage.py) · [结果](runs/native_contact_transfer_20261112/REPORT.md) · [原生中间量](runs/native_contact_transfer_20261112/native_responses.csv) |
| PushT 接触面与旋转位移锥 | 严格单面条件下可约束质心方向。只用视觉状态时，H2 相对保持动作的后果差平均改善 0.058 px，区间 [−0.023, 0.148]。接触方向已归档 | [推导](research/methodology/NATIVE_CONTACT_FACE_20261112.md) · [原生实验入口](research/p1a/contact_cone_native.py) · [结果](runs/native_contact_face_20261112/REPORT.md) · [逐分支物理量](runs/native_contact_face_20261112/cone_physical_queries.csv) · [严格条件统计](runs/native_contact_face_20261112/isolated/cone_native.json) |

## 阅读和复跑范围

`research/p1a/` 保留选定的研究源码及共享数据入口。`runs/` 保存对应的 Markdown、JSON 和标量 CSV，包括逐起点或逐候选结果、训练摘要及计时样本。此目录用于阅读方法和分析已记录结果。完整复跑还依赖原项目中的宿主权重、校准资产、模拟器和分支档案。FourRoom 的 `history_match.py`、`readout_feedback.py` 与 `p1a_reference.py` 可在仓库[上一批代码](../research/p1a/)查看。PushT 脚本还会调用原项目的 `oic_step1/run_native_pusht_host.py` 与冻结模型加载器。 [LeWM 上游代码](https://github.com/lucas-maes/le-wm) · [LeWM 权重与数据](https://huggingface.co/collections/quentinll/lewm)

本次提交包含文本源码和精选数值记录。原报告提到的权重、图像、视频、张量数组及部分大型逐例表仍在原实验目录。短跨度结果支持的范围写在各报告中。跨多次真实交互的长期预测收益和底层读写关系仍待建立。
