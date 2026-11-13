# CJepa 研究进展与代码交接

**2026-11-13 最新交接** · PushT LeWM 单单元预测调节、离线监督、H2 展开及退离与返回回放。 [源码、实验设计与关键数据](progress-2026-11-13/) · [外审报告](progress-2026-11-13/research/CJEPA_PROGRESS_REVIEW_20261113.md)

资料截止 2026-11-08 的阶段梳理。先读本页，再从最新结果追到代码。研究目标是在预训练动作条件 JEPA 的原生计算上形成可随经历维持和修订、支持多个未来跨度的预测内容。当前证据最明确的是同观测配对中的历史信息及其单步读取。持续多时域运算仍待构造，实验处于暂停状态。

[阶段梳理](research/STAGE_SYNTHESIS_20261108.md) · [现行计划](research/CURRENT_RESEARCH_PLAN.md) · [方法推导](research/METHOD_CONSTRUCTION.md) · [近期证据对照](research/methodology/RECENT_MECHANISM_EVIDENCE_20261108.md)

## 最近工作与代码对应关系

| 阶段 | 实际计算与观察 | 入口 |
|---|---|---|
| 有序有限差分 | 以原生动作调用形成差分坐标。固定低阶并令尾项为零可精确复现短前缀，FourRoom 的长跨度误差却随阶数扩大，真实观察换锚还需重新形成坐标。该具体构造已退役。 | [实现](research/p1a/mechanism_stage.py) · [推导](research/methodology/INNOVATION_CONSTRUCTION_WORKING.md) · [实验](runs/method_mechanism_20261107/RESULTS.md) |
| 完整零空间回边 | 保存原生读出矩阵零空间中的 GELU 激活分量，在下一次 GELU 前回灌并更新状态。真实观察覆盖当前 latent，保留回边状态。FourRoom 四个开发留出根的平均 MSE 为 .074429，原生单帧为 .072638。 | [实现](research/p1a/readout_feedback.py) · [实验](runs/readout_feedback_20261107/RESULTS.md) |
| 低秩回边 | 以同一个正交基写入和回灌。独立延迟执行器上正确历史比清空或交换好，等容量无约束基同样有效。FourRoom 固定 16 维状态得到 .072505，清空 .072531，原生三帧 .071073。 | [FourRoom 实现](research/p1a/readout_feedback.py) · [执行器实现](research/p1a/lowrank_actuator.py) · [实验](runs/readout_lowrank_20261108/RESULTS.md) |
| 同观测、同动作配对 | 原档中的 3,855 对像素相同且后缀相同的轨迹未因 ego 历史改变未来。受控干扰物对照则有真实方向差，旧激活与旧低秩状态均携带区别，旧回边几乎没有在输出使用它。 | [原档配对](research/p1a/history_archive_pairs.py) · [受控配对](research/p1a/history_match.py) · [实验](runs/history_match_20261108/RESULTS.md) |
| 解耦切线读取 | 复用旧写入基，从上一真实区间的原生激活取得一次状态。用真实未来端点拟合独立读取。单步预测有收益，递推和多时域能力仍待构造。 | [同一脚本的 `route`](research/p1a/history_match.py) · [端点指标](runs/history_match_20261108/route_anchored.json) |

## 最新实验的计算与结果

冻结宿主的一步预测为 \(F(z,a)=V\sigma(q_\theta(z,a))+b\)。\(\sigma\) 为 GELU，\(V\) 将 2048 维激活读到 192 维。旧递推在 GELU 前加入反馈并从新的激活重新写入状态。最新读取继承旧实验训练得到的 16 维基 \(B\)，从上一已执行区间的原生激活取 \(m_t=B^\top\sigma(q_\theta(z_{t-1},a_{t-1}))\)，以拟合对的统计量得到 \(\tilde m_t\)。

\[
\hat z_{t+1}=F(z_t,a_t)+V D_\sigma(q_\theta(z_t,a_t))Q R\tilde m_t .
\]

\(Q\) 张成 \(V\) 的行空间，\(R\) 是本轮新拟合的 192×16 矩阵。代码直接计算 GELU 的一阶响应项，输出一次真实观察之后的预测。最新读取没有调用旧状态递推，也没有生成后继 \(m\)。

受控实验在 reset 设置两条不同的真实历史，使干扰物在第 10 个环境步抵达相同位置，当前图像逐像素相同。两侧此后使用相同动作和随机过程。96 对分为 72 个拟合对和 24 个留出对，每个模型步对应 5 个环境步。首步无传送的留出为 21 对，前两步均无传送为 18 对。后者用于可读性诊断，下表则是全部 24 个留出对的一步结果。

| 模型 | 单条端点 MSE | 配对差异 MSE |
|---|---:|---:|
| 原生单帧 | .007008 | .002348 |
| 同方向对角切线读取 | .006909 | .002350 |
| 同新增参数量的直接 latent 读取 | .004000 | .001714 |
| 解耦切线读取 | .003813 | .001538 |
| 解耦切线读取，交换匹配历史 | .004320 | .003569 |

先前仅拟合配对差异时，差异 MSE 降到 .000418，单条端点 MSE 却升到 .665600。端点拟合同时约束共同误差与两侧差异。约 94% 的端点总改善来自共同误差降低，交换历史的结果另行支持正确历史的作用。相比同参数的直接 latent 读取，切线读取的额外差距较小。所有配对使用固定后续动作，主要恢复干扰物运动方向。这组数据用于机制开发，独立动作、常规分布、持续递推和长期用途仍待研究。

[原档负对照](runs/history_match_20261108/archive_pairs.json) · [信息可读性与旧回边](runs/history_match_20261108/diagnosis.json) · [仅差异拟合](runs/history_match_20261108/route.json) · [端点拟合](runs/history_match_20261108/route_anchored.json)

## 实验记录

JSON 中是汇总指标、分组统计与设置，CSV 是逐步骤或逐跨度的标量误差。各轮的样本和度量不同，差分实验比较对原生 rollout 的近似，回边和读取实验主要比较真实未来编码。

| 目录 | 文本记录 |
|---|---|
| `runs/method_mechanism_20261107/complete_state_trace/` | [解析系统逐步误差](runs/method_mechanism_20261107/complete_state_trace/analytic_steps.csv)、[原生预测逐跨度误差](runs/method_mechanism_20261107/complete_state_trace/native_steps.csv)、[真实观察换锚误差](runs/method_mechanism_20261107/complete_state_trace/observation_steps.csv) |
| `runs/readout_feedback_20261107/` | [FourRoom 指标](runs/readout_feedback_20261107/metrics.json)、[延迟执行器记录](runs/readout_feedback_20261107/actuator.json) |
| `runs/readout_lowrank_20261108/` | [原生接口记录](runs/readout_lowrank_20261108/native_interface.json)、[独立执行器样本指标](runs/readout_lowrank_20261108/actuator/metrics.json)、[FourRoom 指标](runs/readout_lowrank_20261108/fourroom/metrics.json) |
| `runs/history_match_20261108/` | `archive_pairs.json`、`diagnosis.json`、`route.json`、`route_anchored.json`。上节给出各文件的用途。 |

## 代码入口与运行条件

[宿主适配](research/p1a/host_adapter.py) 负责冻结模型、图像预处理和动作归一化。[原生滚动参照](research/p1a/p1a_reference.py) 负责单帧与三帧等时间窗口。[环境采集](research/p1a/collect_branches.py) 提供 FourRoom 注册与原有分支入口。

解析差分的 `--stage analytic` 只需 NumPy。低秩延迟执行器需要 NumPy 和 PyTorch，可在 CPU 上运行。它们会向所指定的新目录写入输出，包括运行时生成的数组。

```powershell
python research/p1a/mechanism_stage.py --stage analytic --out new_runs/analytic
python research/p1a/lowrank_actuator.py --out new_runs/actuator
```

FourRoom 路径需要外部的原工程资产和环境，以下是阅读代码或在原环境复跑时的输入关系。

| 入口 | 所需输入 |
|---|---|
| `history_match.py collect` | 可导入 H-JEPA 的 `scripts.generate_fourroom_eval_tasks`、`stable_worldmodel` 和环境依赖。`--roots` 指定开发根记录，脚本用首行的 `world_config`，输出 `pairs.npz`。 |
| `history_match.py diagnose` | 该轮 `pairs.npz`，上一轮低秩 `trace.npz`，CUDA 和冻结宿主。输出 `diagnosis.npz` 及小指标文件。 |
| `history_match.py route --fit-endpoints` | 同轮 `pairs.npz` 和 `diagnosis.npz`，上一轮低秩 `trace.npz` 内的写入基，以及冻结宿主。`--fit-endpoints` 对应表中的最终端点拟合。 |
| `history_archive_pairs.py` | `--source` 下的原 P1A `physical.npz` 和 `roots.jsonl`，以及 `--observations` 下的逐根渲染记录。 |
| `readout_feedback.py` | 冻结 FourRoom checkpoint 与已有开发分支记录。`--low-rank` 选择低秩版。 |
| `mechanism_stage.py --stage native` | 冻结 FourRoom checkpoint 与 `--development` 指定的开发分支记录。 |

`FrozenHost` 的运行依赖包含 CUDA、兼容的 PyTorch/torchvision 与 scikit-learn、H-JEPA 模块、`fourroom_lewm_object.ckpt`、`normalizer.pt` 和原工程的 `probe_heads.pt`。包内提供可阅读的代码和既有标量结果，FourRoom 原始数据与模型资产由原工程提供。文档中标为“原工程资料”的链接说明该记录的来源。

## 五篇论文阅读入口

[五篇方法总笔记](research/METHODOLOGY_READING.md) 将每篇的研究问题、核心选择、数学关系、代码证据和对当前项目的启发对应起来。

| 论文 | 分篇笔记 | 公式与实现 |
|---|---|---|
| LeHami | [方法推导](research/methodology/LEHAMI_METHOD_DERIVATION.md) | [代码路径](research/methodology/LEHAMI_CODE_PATH.md) |
| AdaJEPA | [方法推导](research/methodology/ADAJEPA_METHOD_DERIVATION.md) | [代码路径](research/methodology/ADAJEPA_CODE_PATH.md) |
| ID-WM | [方法推导](research/methodology/IDWM_METHOD_DERIVATION.md) | [论文算法与公开实现范围](research/methodology/IDWM_CODE_PATH.md) |
| H-JEPA | [方法推导](research/methodology/HJEPA_METHOD_DERIVATION.md) | [代码路径](research/methodology/HJEPA_CODE_PATH.md) |
| RWM | [方法推导](research/methodology/RWM_METHOD_DERIVATION.md) | [公式、附录及公开实现范围](research/methodology/RWM_METHOD_DERIVATION.md#6-公开实现范围与当前项目的关系) |

[LeHami 与 AdaJEPA 阶段回看](research/methodology/LEHAMI_ADA_STAGE_REVISIT_20261108.md) · [H-JEPA、ID-WM、RWM 阶段回看](research/methodology/HJEPA_IDWM_RWM_STAGE_REVISIT_20261108.md)
