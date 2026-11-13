# PushT LeWM 实验设计与实际代码执行

研究目标是在固定部署参数下，利用跨真实观察的状态调节 LeWM 原生动作预测。当前实验检验同一条链路上的四个环节，即从真实经历区分模式、用模式预测候选动作、跨无区分力观察保留证据、以及沿假想动作递推。研究暂停于固定检查点的退离与返回对照。 [机制推导](CORE_REGULATION_MECHANISM.md) · [进程报告](CJEPA_PROGRESS_REVIEW_20261113.md)

## 一个真实根怎样走完发出、写入和预测

以历史开发场景 102022 的世界 3/2 为例。两世界在步 10 有相同的三帧公开前缀和实际命令，步 15 的小块观测相反，步 30 的三帧窗口再次相同。以下时点按环境步计，每 5 步构成一个动作块。 [数据读取](p1a/native_response_mixture_pilot.py#L39) · [逐候选分差](../runs/native_response_mixture_rollout_20261113/matched_candidate_pairs.csv)

| 时点 | 实际代码的数据和状态 |
| --- | --- |
| 步 10 发出 | `issue_event()` 取步 0、5、10 的三个 192 维公开 latent 和对齐的三个动作块。最后的动作条件是即将执行的 10→15 动作块。`predict_response(factual=True)` 返回 `IssuedResponse`，包含三模式各自的 192 维预测、发出时的 log 权重和恢复率 logit。两世界共用同一先验。 |
| 步 15 到达 | 3×3 掩膜内的实际 RGB 决定是否看到推手，判据为至少一个像素的 B−R 大于 115。每个预测模式先经固定线性坐标读出，乘 96/512 转到图像坐标。观测 logit 是传感半径 `15*96/512` 减去预测推手到掩膜内最近像素的距离。对实际二值事件计算三个 Bernoulli log 似然。`condition()` 对旧 log 权重加 log 似然后归一化，再执行恢复转移。该转移输出是步 15 写入状态。 |
| 步 20、25、30 | `subsequent_roots()` 用已到达的三帧公开窗口及实际动作推进公共状态。公开条带在这些时点不提供新的区分事件。`write` 逐次向固定先验恢复，`skip` 保留未条件化的先验，`carry` 直接沿用步 15 完成当次恢复后的写入状态。 |
| 步 30 查询四个候选 | `future_inputs()` 取步 20、25、30 的公开 latent 和已执行的 20→25、25→30 两个动作块。`forecasts()` 按候选复制真实窗口与当前公共信念，把候选的首块放在第三个动作位置，预测 H1。每个候选仅推进自己的假想信念。 |
| H2 自由推演 | 标准路径删除最老帧、追加 H1 的混合 latent，把动作窗口前移一块，再执行同一预测器。完整展开对三种 H1 后继分别构造第二步窗口，按首步信念和原模式转移加权三乘三的响应。固定候选比较读取最终预测和模拟器独立执行的实际结果。 |

在该场景的世界 3/2，候选 1 减候选 2 的 H1 真值分差为 −4.757 / +2.519。两世界省略臂都给出 +0.115。原生写入预测为 −0.455 / +0.432，保留干预为 −3.739 / +2.299。这组实测使用 `forecasts()`。`get_cost()` 具备传入信念并将其复制至候选的接口，闭环任务结果仍待测。 [发出与写入代码](p1a/native_response_mixture_pilot.py#L108) · [候选自由推演](p1a/native_response_mixture_pilot.py#L136) · [展开](p1a/native_response_mixture_rollout.py#L17) · [规划接口](../research_sources_2026-10-22/lewm/jepa.py#L170)

## 离线训练和对照

96 个拟合场景训练，48 个历史开发场景评价。每个场景由四个实际模拟世界构成。合法配对要求同朝向、不同推手侧向、相同公开前缀和相反的步 15 事件。候选固定为直推、正侧转向、负侧转向和保持。H1 与 H2 分别为五步、十步。完整图像编码和物体物理终点进入离线标定、训练目标与评分，部署输入为公开图像和掩膜、已执行动作及合法候选。 [采集设计](../JEPA_OIC_Codex_Handoff_2026-11-03/native_content/prepare_events.py) · [候选定义](../oic_step1/prepare_visual_contact.py#L18)

| 路径 | 样本、监督和梯度 |
| --- | --- |
| 首轮 64 更新 | 事件边缘 NLL 和步 10/15/30 自由 H1/H2 latent MSE。梯度进入最后原生条件块、预测头及两行读出。 |
| 成对监督旁支 64 更新 | 从首轮恢复，用曲率方向初始化分叉行，增加步 30 H1 的真实成对中心化 latent 差损失。分叉行只接收该成对项的梯度。 |
| 直接监督 256 更新 | 从首轮恢复并将分叉行置零。源监督拟合完整视野真实后继给出的两端 logit 差。用实际成对 latent 差教候选响应，离线参考信念和保留因子视为固定系数。分叉行接收两项直接监督的梯度。 |
| 物理监督 256 更新 | 从直接监督的检查点继续。保留源监督，候选监督改为实际写入信念下的步 30 H1 中心化物体位置误差。信念停止梯度，固定物体坐标读出，候选物理损失的梯度经原生响应回传。原事件及未来项继续作用于其余可训练路径。 |

四轮均使用 AdamW、学习率 $10^{-4}$、每批 4 个场景、种子 1611、梯度范数上限 1。编码器、动作编码器、此前 predictor 块和 BatchNorm 运行统计固定。原生写入、跳过似然和保持步 15 信念的比较共享同一检查点与候选。H2 的 `merged`、`product`、`joint` 共享 H1，分别考察均值窗口、逐窗口独立加权和按原模式转移的联合展开。退离与返回沿原生 PushT 的固定动作，把 16 个旧拟合起点的轨迹延长到步 60，并在步 30/45/60 采集同样四个候选。 [损失实现](p1a/native_response_mixture_pilot.py#L219) · [训练入口](p1a/native_response_mixture_pilot.py#L559) · [回放采集](p1a/native_response_mixture_lifecycle_data.py)

§16.21 拟用同一个物理候选损失，分别利用实际步 30/45/60 的候选终点监督相关、暂时无关、重新相关三种情况。停留时仍保留已有证据，约束它在当前候选物体后果上的作用。该训练仍处于推导阶段。 [§16.21](CORE_REGULATION_MECHANISM.md#1621-把证据存续与候选作用的相关程度分开)

## 公开数值记录

场景是统计单位。同一场景的世界和候选共享生成条件。`world` 和 `candidate` 从 0 起编号。位置是 PushT 环境坐标。`horizon` 按动作块计。候选分差为候选 A 减 B，真值在两个世界符号相反时，两边均排对才计一次正确反转。中心化误差按相同根、跨度和方法，在四候选上减去平均误差，再计算二维平方范数。`state` 区分三条事实信念路径，`rollout` 区分假想展开方式。

| 数据 | 内容和来源 |
| --- | --- |
| `training_steps.csv` | 每次优化更新的原始损失及相应的分叉行梯度范数。来自各轮原始 `intermediates.npz` 的 `loss` 字段。 [物理监督训练记录](../runs/native_response_mixture_physical_20261113/training_steps.csv) |
| `development_evidence.csv` / `fit_evidence.csv` | 场景×世界的实际事件、三模式的发出概率、对数似然、logit、后验及步 15/30 信念。`event_reset_probability` 存 sigmoid 后的恢复概率。拟合组的可见事件按相同 RGB 与掩膜规则从原始档案提取。 [开发组](../runs/native_response_mixture_physical_20261113/development_evidence.csv) · [拟合组](../runs/native_response_mixture_physical_20261113/fit_evidence.csv) |
| `retention_by_scene.csv` | 各场景×跨度×保留条件的中心化和绝对位置 MSE。来自保留干预逐场景数组。 [逐场景结果](../runs/native_response_mixture_physical_20261113/retention_by_scene.csv) |
| 展开 CSV | 包含步 20/30 逐候选预测与真值、latent MSE、真实根信念、配对分差、H2 窗口非线性项及模式关联项。预测坐标由存档 latent 经当次固定坐标读出导出。 [候选](../runs/native_response_mixture_rollout_20261113/candidate_predictions.csv) · [信念](../runs/native_response_mixture_rollout_20261113/root_beliefs.csv) · [匹配分差](../runs/native_response_mixture_rollout_20261113/matched_candidate_pairs.csv) · [分解](../runs/native_response_mixture_rollout_20261113/merge_terms_by_scene.csv) |
| 回放 CSV | 步 30/45/60 的物体起点与实际终点、预测坐标、中心化预测变化、有效执行和接触步数，步 15 至 60 的原生信念，以及 H1 模式直径。 [候选与物理](../runs/native_response_mixture_lifecycle_20261113/candidate_predictions.csv) · [信念时序](../runs/native_response_mixture_lifecycle_20261113/belief_trajectory.csv) · [模式直径](../runs/native_response_mixture_lifecycle_20261113/h1_mode_diameters.csv) · [匹配世界](../runs/native_response_mixture_lifecycle_20261113/matched_worlds.csv) |

## 原环境复跑所需资产

已发布的 JSON 与 CSV 支持复算报告主要数值，无需加载模型。重新前向需要原环境中的 PyTorch、torchvision、transformers、einops 和本地 PushT LeWM `model/weights.pt`，以及相应的阶段 `pilot.pt`。训练和 H2 评估需要原 `train_11033026.npz`、`confirmation_11034026.npz` 及两组缓存的公开、完整和未来 latent。重新生成物理分支需要原 PushT 模拟器环境与物理档案。权重、完整观测及 latent 数组留在实验原目录。 [模型配置](../research_sources_2026-10-22/lewm_assets/model/config.json) · [动作归一化](../oic_step1/runs/pusht_continuation_2026-10-31/action_normalization.json) · [上游模型与数据](https://huggingface.co/collections/quentinll/lewm)
