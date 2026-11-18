# cJEPA 研究进展 · 2026-11-18

本目录由 `progress-2026-11-17` 整体改名并更新，保留原有 PushT／FourRoom 方法、代码和结果记录，补齐 11 月 17–18 日 Stage A 与 Stage C 的当前实现、技术报告和关键 JSON／CSV。最新机制原稿完整到 **§49.12**，不再仅保留 §17–48 的节选。

## 当前应从哪里读

1. [研究提案](research/RESEARCH_PROPOSAL.md)：长期能力、三时间轴及方法构造纪律；较早的阶段判断保留原日期。
2. [研究目标](research/RESEARCH_GOAL.md)：当前结果与执行边界。
3. [完整核心机制稿](research/CORE_REGULATION_MECHANISM.md)：包含历史正反例和 §49.11–49.12。
4. [Stage A 技术报告](runs/stage_a_20261117/REPORT.md)：原门位 FourRoom 资格检验、官方宿主和匹配基线。
5. [Stage C Step 2 最新收尾报告](runs/stage_c_bridge_source_20261118/REPORT.md)：H1 真实到达和 H12 菜单采用同一无标签排列后，哪些能力成立、哪些仍失败。
6. [历史接口与代码摘录](research/FOURROOM_CODE_TRACE_EXPORT_20261116.md)：11 月 16 日汇集的旧接口；当前 Stage C 代码以下列入口为准。
7. [历轮方法与数据索引](METHOD_ITERATIONS.md)：保留 PushT、移动门洞和原门位的不同证据口径。

## 最新结果及边界

| 阶段 | 已测结果 | 仍未成立 |
| --- | --- | --- |
| Stage A：官方原门位 FourRoom | 全开物理与冻结原生 LeWM 均为 24/24；固定真实履历上比较 Frozen、AdaJEPA／OpTTA 作者适配和等参数 GRU。 | 固定 ABA/BAB 的双历史路线方向未被这些基线稳定恢复；这是固定候选／真实履历功能测试，不是完整通用基准的复现。 |
| Stage C：静止背景 | Oracle-k 末两块共同训练首块遗憾 2.81；双历史名义路线可学习。 | 两方案无标签训练的 384 候选真实写入遗憾仍为 75.52。 |
| Step 1：同履历八方案覆盖 | Cand 115 的 H12 自由窗减实际窗由 73.04 降到 4.28 px；固定面板写入遗憾降到 5.02。 | 不同方位偏好的解释编号相反，不能把局部对齐视为全局一致。 |
| Step 2：贯通桥及同一排列 | 两端共享排列的目标和对照已实现。 | 单独续训时每阶段首探仍仅 4/8，真实写入遗憾 39.23。 |
| Step 2 收尾：官方重训＋H1/H12 同排列 | 每阶段首探 8/8；真解释首块遗憾 **2.89**，接近同面板真 RGB 的 **2.81**；Cand 115 受阻 H12 自由窗差 1.15 px。 | 西北回撤重复产生反向证据，正确支持由约 **0.985 降至 0.339**；真实写入遗憾 **19.63**、双历史 **6/12**、无关改选 **0/48**。Step 2 整体未通过，随机背景与新闭环未启动。 |

最新的 24 个保留点评价来自 4 个物理根、8 段历史，各点 384 候选；重复几何根不计为 24 个独立场景。真解释 2.89 是直接选择正确分支的诊断，不等于真实写入的决策收益。

## 当前核心代码

- [原生条件计算](research/p1a/stage_c_native_check.py) · [同一解释的证据、支持与多步预测](research/p1a/stage_c_paired.py)
- [实际前缀和八方案／贯通桥数据](research/p1a/stage_c_data.py) · [H1/H12 联合排列训练](research/p1a/stage_c_paired_fit.py) · [逐到达和固定候选复播](research/p1a/stage_c_paired_replay.py)
- [官方原门位环境与规划入口](research/p1a/stage_a_fourroom.py) · [基线与 GRU](research/p1a/stage_a_baselines.py) · [同履历对照](research/p1a/stage_a_replay.py)

## 原始数据入口

最新运行目录保存 [训练与验收汇总](runs/stage_c_bridge_source_20261118/paired_unlabeled_summary.json)、[逐到达证据和支持](runs/stage_c_bridge_source_20261118/arrival_trace.json)、[同候选 H12 物理溯源](runs/stage_c_bridge_source_20261118/fixed_regret_trace.json)、[拟合端身份检查](runs/stage_c_bridge_source_20261118/fit_bridge_diagnostic.json) 与 [全局编号冻结](runs/stage_c_bridge_source_20261118/fit_alignment.json)。

本包更新已有文本、源码和关键数值数据。预训练权重、训练检查点、NPZ／PT 张量和约 10–12 MB 的完整逐候选复播仍由原工程提供；保留汇总、逐到达与固定候选诊断以界定可核查范围。未收录资料的文档链接标为“原工程资料，未随包同步”，不将此包称为完整独立复现。
