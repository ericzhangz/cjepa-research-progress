# AdaJEPA：已执行转移怎样更新同一个世界模型

本文对应 [AdaJEPA v1 正文及附录](https://arxiv.org/html/2606.32026v1) 与 [官方仓库 agentic-learning-ai-lab/adajepa](https://github.com/agentic-learning-ai-lab/adajepa)。2026-11-07 读取公开 main 分支，通过 GitHub 插件取得源码；本轮没有下载发布权重、没有调用训练/规划，也没有用另写的相似实现代替官方代码。来源副本位于 AdaJEPA 来源目录（原工程资料）。

本文将三层事实分开：论文的抽象算法、公开代码中存在的具体实现、需要发布 checkpoint 相邻 hydra.yaml 才能确定的实例配置。论文主模型为 ResNet 全局表示，仓库通用 train.yaml 却默认选择 DINO；直接运行通用配置并不等于复现论文主模型。

## 1. 核心机制只增加在两个位置

官方 README 明确列出两个主要新文件：

- [planning/adajepa.py](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/adajepa.py)：AdaJEPATrainer，选择可更新参数、组织真实轨迹片段、计算预测损失、做梯度更新。
- [planning/adajepa_mpc.py](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/adajepa_mpc.py)：AdaJEPAMPCPlanner，继承已有 MPC，在环境反馈 hook 中调用 trainer。

它没有引入一个额外的预测器，也没有将适配结果与原 predictor 通过另一个融合网络拼起来。原生 VWorldModel.predict 所引用的 predictor 参数已经改变，下一次同样的 predict 调用就产生新映射。[论文式 (4)–(5)](https://arxiv.org/html/2606.32026v1#S3.E4)

实际调用链为：

~~~text
plan.planning_main
  ├─ 读取发布模型 hydra.yaml、checkpoint → load_model → VWorldModel
  └─ PlanWorkspace
       ├─ 把环境步转换成模型步
       └─ AdaJEPAMPCPlanner.plan
            └─ 每个 episode 独立：
                 trainer.reset，清空 buffer
                 → MPCPlanner.plan
                    → GDPlanner/CEMPlanner.plan
                    → 执行动作前缀，得到真实反馈
                    → _post_env_feedback
                       → _extract_adajepa_data
                       → buffer选择/合并
                       → trainer.finetune
                    → 下一轮同一个 world model 重新规划
~~~

BasePlanner 构造时先将整个 wm 设成 eval 并冻结参数；finetune 时再临时开启指定参数的 requires_grad。这样动作优化阶段只优化动作，模型适配阶段才优化选定模型参数。[base_planner.py:6](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/base_planner.py#L6)

## 2. 式 (1) 中被简写的真实张量

论文 [式 (1)](https://arxiv.org/html/2606.32026v1#S3.E1) 写成 z=E(o)，u=A(a)，预测器 f(z,u)。实际宿主将视觉、proprioception 和动作编码装入同一个 predictor token 张量：

- VWorldModel.encode_obs：视觉帧 B×T×3×H×W 展平为 BT 张图送 encoder，再恢复 B×T×P×D_v；proprio 经独立 embedding。
- VWorldModel.encode_act：一段真实低层动作串经 ProprioceptiveEmbedding，其实现是 kernel_size=1 的 Conv1d 和 LayerNorm。
- concat_dim=1 时，在每个视觉 patch 的通道轴追加 proprio 与动作；concat_dim=0 时，追加两个 token。
- VWorldModel.predict 将 B×T×P×D 展平为 B×(TP)×D，调用 ViTPredictor，再恢复时间维。[visual_world_model.py:140](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/models/visual_world_model.py#L140)

在论文主 ResNet 配置对应的代码实例中，SmallResNet 输出一个 384 维全局 token；源文件的 [scratch_resnet.yaml](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/conf/encoder/scratch_resnet.yaml) 显式将 dim 设成 384。若配合 train.yaml 的 proprio/action embedding 各 10 维、各 repeat 1，concat_dim=1，那么：

\[
z_{b,t,1,:}
=
[\,z^{visual}_{384};\,z^{proprio}_{10};\,z^{action}_{10}\,]
\in\mathbb R^{404}.
\]

所以“384 维视觉表征”不能直接写成“预测器处理 384 维全部状态”。预测损失排除动作部分，保留视觉和 proprio，即该实例是 394 个通道。DINO patch 版本的 P 则通常不是 1。这些数字仍须服从具体发布模型 hydra.yaml；本轮没有读取权重包里的配置，不能将通用 YAML 组合冒称每个发布权重的精确结构。

[SmallResNet](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/models/encoder/resnet.py#L101) 是五个降采样 residual block，通道 3→16→32→64→128→256，随后 flatten、Linear(256·7·7,512)、GELU、Linear(512,384)、LayerNorm。encoder 的“最后一层”在适配代码中指最后一个子模块 projection，即整个上述投影序列，不是最后一个卷积层，也不只是最后一组 LayerNorm。

[ViTPredictor](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/models/vit.py#L107) 加可学习位置 embedding，经过 pre-norm attention/FFN 残差层，最后 LayerNorm。公开 predictor YAML 配置 depth=6、heads=16、mlp_dim=2048、dropout=0.1。mask 允许同一帧的所有 patch 互相可见，并禁止看到以后帧的 token。因此输出第 j 个位置预测下一个时刻时，没有读取输入窗口中 j 之后的未来观察。

实现中的 mask 不是 buffer，构造时直接 .to('cuda')。完整公开 ViTPredictor 不能未经改动就宣称 CPU 可运行；此处描述它的部署要求，不另写 CPU 替代品冒充官方实现。

## 3. 时间单位：25/5 与 5/1 是同一配置的两层表示

[官方 PushObj 规划配置](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/conf/adajepa_plan_gd_pushobj.yaml) 写：

- sub_planner.horizon=25；
- n_taken_actions=5；
- num_start_frames=1；
- 最大 20 次 MPC；
- recent5、一次适配更新。

但是 [PlanWorkspace:203](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/plan.py#L203) 将 horizon 和 n_taken_actions 分别整除发布模型的 frameskip。在 frameskip=5 时，内部实际为：

\[
H_{\mathrm{model}}=25/5=5,\quad
R_{\mathrm{model}}=5/5=1.
\]

一个模型动作是五个连续环境动作的拼接，不是将同一个动作重复五遍。[TrajSlicerDataset:122](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/datasets/traj_dset.py#L122) 使用 (n f) d→n (f d)；PushT/PushObj 原始动作二维，因此每个模型动作的原始向量有 10 维。

起点只放一个真实观察。模型训练可用 num_hist=3，不等于规划起点有三帧历史；rollout 会从一个已知 token 开始，在生成中逐渐累积到最近三帧上下文。每轮重规划也用刚到达的单帧作为新起点。[mpc.py:147](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/mpc.py#L147)

## 4. 式 (3)：原有规划是怎样使用模型的

[VWorldModel.rollout](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/models/visual_world_model.py#L410) 的核心是：

1. 把真实起点与第一个动作编码为 z。
2. 对最近 num_hist 个 token 调 predict。
3. 只取最后一个预测状态。
4. 将其中预测出来的 action 通道替换为候选动作的下一项，再接回上下文。
5. 重复到最后一步，再预测最终状态。

动作在 candidate 序列中已给定，模型不靠自己预测 action 再使用它。只有动作通道被替换，视觉与 proprio 使用自由预测值。每个 horizon 模型步仍需要一次 Transformer predictor 调用；不存在一步直达任意时间的读出。

[GDPlanner.plan](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/gd.py#L98) 每次规划先以当前 encoder 重编码目标观察，目标编码位于 no_grad 内；随后把候选动作设成 requires_grad=True，100 次 Adam 动作优化，每次完整 rollout。默认学习率 0.1、zero 初值、无动作噪声；GD 类还默认启用 cosine learning-rate scheduler。模型参数在动作优化时冻结，但动作梯度穿过 action embedding、predictor 和自回归链。

[CEMPlanner.plan](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/cem.py#L65) 则采样候选动作，no_grad rollout，按目标代价选 elites，再更新动作均值/标准差。它也在每次 replan 重新编码目标，不跨模型更新缓存旧 z_goal。

公开配置的 objective.mode=staged，不是始终相同的全时域加权和。实际 [objectives.py](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/objectives.py#L6) 是：

\[
c_{terminal}
=\operatorname{MSE}(z^v_H,z^v_g)
+\alpha\operatorname{MSE}(z^p_H,z^p_g).
\]

当 MPC 轮数 step 小于 rollout 长度减一时用 terminal；以后使用随预测位置指数增大的权重，base=2，归一化后再做时间 mean。主配置 \(\alpha=1\)。这解释了论文式 (3) 的抽象目标在发布实现中如何落地，不能把所有实际规划代价都等同于终点 latent 欧氏距离。

MPC.reuse_actions=True 复用上一轮未执行的动作作为优化初值。这是**动作 warm start**；未来状态仍重新 rollout，不是保存 Transformer 中间计算后增量修订。

## 5. 式 (4)：适配的样本究竟是什么

### 5.1 环境反馈如何切段

_post_env_feedback 先调用 [_extract_adajepa_data](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/adajepa_mpc.py#L155)。若本轮执行 T 个模型动作、frameskip=f：

\[
start=\text{iter}\cdot T f;
\quad
o^{segment}
=e\_obs[:,start:start+Tf+1:f].
\]

返回 T+1 个真实观察和 T 个已执行模型动作。默认 T=1，因而每段有两个模型时间点的观察、一个五环境步动作块。这里不从假想候选 rollout 取训练标签。

实际模拟器包装的 MPC 每轮把累计已执行前缀交给 evaluator，由同一初态和 seed 重放 env.rollout 后取最新段。这是公开评估代码组织闭环反馈的方式，不是一个持续持有模拟器状态、只 step 新动作的流式 API。它没有把候选动作的真实后果反馈给规划器；但总运行成本包含累计前缀的模拟和额外模型评估，不能全算成 AdaJEPA 更新成本。[mpc.py:122](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/mpc.py#L122)、[evaluator.py:127](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/evaluator.py#L127)

### 5.2 recent5 不是五个互不相关 transition batch

recent 模式保留最近五个执行片段。finetune 默认 merge=True，_merge_segments 将连续片段拼成一段，丢弃每个后续片段重复的首帧。默认五段因此变成六个观察、五个模型动作。

_prepare_segment 在最后加一个零动作，使 encode 看到相同长度的观察和动作。它只是让最后目标帧的 token 形状完整：最后帧不进入 source，且动作维不参与 loss，因此零动作不是虚构的训练 transition。[adajepa.py:184](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/adajepa.py#L184)

### 5.3 真正的 loss 是滑动窗口、每个位置的一步预测

[_prediction_loss](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/adajepa.py#L206) 接收 z∈R^(B×(T+1)×P×D)，设 w=min(num_hist,T)，对 t=0…T−w：

\[
z^{src}=z_{t:t+w},\quad
z^{tgt}=z_{t+1:t+w+1},\quad
\hat z=f_\theta(z^{src}).
\]

每个窗口所有输出位置都做一步预测误差，排除动作通道，然后对窗口取平均：

\[
\mathcal L_{\rm code}
=\frac{1}{T-w+1}
\sum_{t=0}^{T-w}
\operatorname{MSE}
\left(
\operatorname{obs}(f_\theta(z_{t:t+w})),
\operatorname{obs}(\operatorname{sg}z_{t+1:t+w+1})
\right).
\]

这与论文式 (4) 的单转移求和思想一致，但其具体实现具有上下文和重叠窗口。它**不是自由展开多步适配**：每个 source 窗口都来自已到达的真实观察编码。

在 recent5、T=5、num_hist=3 时，恰好有三个 source 窗口 [0,1,2]、[1,2,3]、[2,3,4]。同一真实转移会在不同上下文位置重复出现，五个转移的出现次数为 1、2、3、2、1。因此不能把这段实现简写成“五条等权、完全独立 transition MSE”。早期 buffer 不满时 w 从 1 逐步增加到 3。

hardN 是不同设置：保留历史高误差片段，片段不连续，merge=False。默认每段 T=1 时，它们分别以单帧上下文计算 loss；这与 recentN 的连续长窗口同时改变了样本选择和上下文组织。代码用上次得到的分数选择保留对象，新来的未打分段以正无穷暂保留；适配后再刷新分数。[adajepa_mpc.py:164](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/adajepa_mpc.py#L164)

## 6. 式 (5)：参数、梯度、buffers、optimizer 分别在哪里

### 6.1 实际更新范围

默认 [_select_predictor_params](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/adajepa.py#L119) 选 transformer.layers[-1] 和 transformer.norm。可学习 pos_embedding 不在默认更新列表；其余 Transformer blocks 冻结。

默认 _select_encoder_params 对 SmallResNet 选最后一个 named_child，即 projection。对有 base_model 的 DINO wrapper，只选非 backbone 参数或指定 projector，并令 base_model.eval()。不是将所有视觉骨干统一解冻。

action/proprio encoder 不在 AdaJEPA 的 optimizer 参数组中，BasePlanner 已将它们冻结。正文式 (5) 允许抽象 \(\Omega\subseteq\{\theta,\phi,\psi\}\)，不等于公开默认更新了这三组所有参数。

### 6.2 source/target 梯度

finetune_encoder=True 时，每个梯度 step 都重新 encode buffer，source 不 detach；target 是否 detach 读取 wm.stop_grad，默认 true：

\[
r=f_\theta(E_\phi(o),A_\psi(a))-\operatorname{sg}E_\phi(o'),
\]

\[
\nabla_\theta L=J_\theta f^\top r,\qquad
\nabla_\phi L=J_\phi E(o)^\top J_z f^\top r.
\]

这是为解释代码写的简式，实际 MSE 还含均值比例、proprio 和所有窗口求和。target 分支无该步梯度，但下一步重编码目标时其数值可变化，不能称为永久固定的 latent target。

finetune_encoder=False 时，先在 no_grad 下把 buffer 编码一次，后续多次适配重复使用这些 embeddings；source 和 target 均 detach，只更新 predictor。在线更新不调用 VWorldModel.forward，因此不会自动启用离线的 temporal straightening、VCReg、decoder loss；它只调用 _prediction_loss。

### 6.3 “只更新 head”并不等于骨干所有状态不动

默认代码对 SmallResNet 调整个 encoder.train()。虽然骨干权重 requires_grad=False，其 residual blocks 中的 BatchNorm running_mean、running_var 等 buffers 仍会随真实观察 batch 更新。_take_snapshot 明确保存 encoder/predictor buffers，reset 在新 episode 前恢复它们。

所以其持续状态至少包括：

\[
\text{已更新的选定权重}
+\text{训练模式中更新的buffers}
+\text{原始观察/动作buffer}.
\]

这并非本文额外添加的模块，而是公开代码已经发生的行为。DINO 的 base_model 被单独设为 eval，故这点不能不加区分推广到全部 encoder。

### 6.4 Adam 的状态不跨 MPC 轮保存

_make_optimizer 在每次 finetune 内部重新创建，默认 Adam，predictor lr=5e-4、encoder lr=1e-5。一次 finetune 内若 steps>1，Adam 状态在这些 steps 之间保留；下一次 MPC hook 则新建 optimizer。持续积累的是模型参数，而不是跨轮延续的 Adam 一阶/二阶矩。

这使论文式 (5) 的“梯度更新”在实现上有非常具体的含义：默认每轮一次新 Adam 的第一步，不能直接说成持续 Adam optimizer，或严格等价于固定学习率 SGD。

训练完成后 finetune 将 encoder/predictor 切回 eval，再关闭选定参数梯度，供下一次动作优化使用。记录的 step_losses 是 optimizer.step 前算出的 loss；默认唯一数值不自动代表更新后在同一 buffer 上的误差。[adajepa.py:58](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/planning/adajepa.py#L58)

## 7. 离线训练和在线适配没有共用同一段 loss 前向

公开仓库保留 train.py，其 init_models 构造 VWorldModel、encoder、predictor、action/proprio embeddings；训练循环调用 model(obs,act)，然后分别 step 各 optimizer。[train.py:555](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/train.py#L555)

离线 [VWorldModel.forward](https://github.com/agentic-learning-ai-lab/adajepa/blob/main/models/visual_world_model.py#L297) 取前 num_hist 帧为 source、偏移 num_pred=1 的帧为 target。基础误差仍只监督视觉/proprio。可选 temporal straightening 和 encoder VCReg 在这里叠加；decoder 若存在，以 detached latent 学重建，不将其重建梯度传回 encoder/predictor。

在线则由上节的 AdaJEPATrainer 重新组织长片段滑窗，仅用预测 MSE。这就是“复用原生预测任务作为在线更新信号”的实际含义，不能理解成重新执行整套离线训练 objective。

官方 README 建议需要自行预训练时使用 Temporal Straightening 上游仓库。当前代码中通用 train.yaml 仍有 DINO、batch32、straighten=False 等默认；论文附录报告的主训练设置、不同发布 checkpoint 的选择应从对应配置读出，不能拿通用默认覆盖论文报告。

## 8. 哪些论文选项在这里可以直接调用，哪些不能

| 选项 | 公开实现 |
|---|---|
| 默认最后 predictor block+encoder head | 有，明确选择参数 |
| predictor 全量更新 | last_layer_only=False |
| encoder冻结 | finetune_encoder=False |
| 全encoder参数或非DINO子模块选取 | encoder_last_layer_only=False；DINO仍限制非backbone |
| recentN / hardN /全部buffer | 有；false表示保留全部，并不是关闭buffer |
| 每k轮适配、每轮多步 | finetune_every、steps |
| Adam/AdamW/SGD | _make_optimizer 可选；不混同各自默认weight decay |
| 论文附录的 LoRA rank8/alpha16 | 本轮所读公开 trainer 与配置未提供对应路径 |
| 任意指定“第一层”“首尾两层” | 当前 trainer 无对应显式选择开关，不能假装已复现附录所有行 |
| 发布权重与evaluation data | README给Google Drive入口；本轮未下载 |
| 官方完整CPU小实验 | ViTPredictor构造包含CUDA绑定，不能直接承诺 |

这里“未提供”限定于本轮读到的官方文件与默认分支，不主张作者没有运行过这些附录实验。

## 9. 实际成本来自什么，节省的又是什么

假设一次规划对 H 个模型步、N 个候选或 I 次 GD 迭代反复查询，原预测成本仍随 H 增长。AdaJEPA没有缩短 rollout 的串行深度。在线适配增加 buffer 编码、滑窗 predictor 前向、反向传播和 optimizer step。

默认 recent5 满时，H_hist=3，可概括为每次适配：

\[
C_{ada}
\approx C_E(6\ {\rm frames})
+3C_P(3\ {\rm frames})
+C_{\rm backward}
+C_{\rm optimizer}.
\]

这里三个窗口是三个实际 wm.predict 调用，每次包含三个时刻；不是三个独立样本并行 batch 调用。encoder冻结时，可在同一次 finetune 的多个优化 steps 间缓存 z；encoder更新时每 step 重编码。

方法贡献不是“比冻结 MPC 更省计算”，而是用少量额外更新，使原有行动后果映射适应当前分布。论文附录的低额外延迟是指定硬件与配置的经验结果，不是免重算性质，也不能归因于持续多时域中间态。MPC action warm start、encoder冻结时的编码复用、online parameter adaptation 是三件不同的事。

## 10. 对上轮理解的具体修订

1. 25 horizon、5 execution 是外部环境步；frameskip=5 后内部是 5/1 模型步。
2. recent5 默认合并为一条轨迹，用重叠上下文做单步误差；不是五个互不相干的 transition。
3. 默认适配不仅改变 head权重，还可能改变骨干 BN buffers；每个 episode reset恢复这些状态。
4. Adam 每轮新建，不是跨轮保留 optimizer memory。
5. 目标图像每次 replan 按当前 encoder 重编码，上轮对此仅给了数学上的实现要求，现在已有 GD/CEM 代码落点。
6. 在线停止梯度由 wm.stop_grad 控制；默认 true，但不应把通用 trainer 的全部配置说成无条件detach。
7. LoRA、第一层等论文附录选项，不能因为论文写过就宣称已在当前公开 trainer 中找到。
8. “持续适配共享映射”没有解决我们要构造的“复用完成的预测计算而服务多个 horizon”。可以学习其单一学习信号和原生路径内改动的风格，不能把它现成的参数适配改名成新持续层。

## 11. 最小机制实验建议：只看一次更新怎样传播到下一次预测

使用一个正式发布的 world model 实例和一段真实记录中的六个模型时间点，直接实例化官方 AdaJEPATrainer。无需环境、CEM、GD或目标成功率：仅调用 reset→finetune，随后用同一批固定 source/action 做 predict，比较更新前后的预测映射。

记录三个对象即可：

- 适配前后，同一真实转移及尚未用于适配的后继转移的输出变化；区分训练片段拟合与更新外传播。
- 选定参数变化和 BatchNorm buffers 变化分别对应哪些输出变化；以官方已有 finetune_encoder 开关构成解释性对照。
- 当前真实观察与目标图在更新后的 encoder 中怎样共同移动；并记录重编码与梯度更新所花的运算。

仍需使用真正官方 VWorldModel / trainer 与发布配置，不造一个近似小网络然后称为 AdaJEPA 复现。由于本轮没有下载其权重和数据，这只是可执行路径明确的实验建议，尚未运行。若只实例化随机参数，也必须称作更新算子的机制演示，不能据此声称适配能改善物理预测。

从这条代码链得到的研究原则是：**先确定真实经历究竟改变哪个原生数学对象，再定义它怎样影响以后的全部查询。** 对我们的冻结宿主方案，这个对象还不能偷换成可训练模型参数；需要另一个同样具体、同时具有更新语义与跨时域运算意义的对象。

