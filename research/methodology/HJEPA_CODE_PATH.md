# H-JEPA：公式、训练代码与执行路径

本文按实际调用路径解释 H-JEPA，不启动训练、规划评价或新的模型实现。重点是哪些原生算子可以单独调用，以及这些算子的时间和坐标含义。

## 1. 阅读来源与适用范围

论文是 [H-JEPA v2（2026-11-01）](https://arxiv.org/html/2610.06805v2)，重点对应正文 §2、附录 C、E、K、L；原有完整论文推导见 [HJEPA_METHOD_DERIVATION.md](HJEPA_METHOD_DERIVATION.md)。论文给出的官方仓库是 [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA)。

下面逐函数、逐配置的叙述以本地实际阅读的源码为准：

`C:/Users/ASUS/AppData/Local/Temp/cjepa-p0a-ff1d5adbd6ae448e993c31d8bf9c4668/upstream/`

其中 `h_jepa/` 放模型、训练与层次规划，`stable_worldmodel/` 放数据集、梯度求解器和真实执行 policy。本文另读取了远端 `main` 的 `h_jepa/models/jepa.py` 的 encode/predict/rollout 段，以及官方仓库说明；这些入口的语义与本文对应。**没有把本地整棵目录声明为最新仓库的逐文件副本，也没有以远端当前配置替代本地 checkpoint 保存的配置。** 复用时应由所加载对象的模块和保存配置决定具体维度。

关键本地入口：

| 路径 | 作用 |
|---|---|
| models/hjepa.py（原工程资料） | 底向上的训练编码、随机裁剪、层间连接 |
| models/jepa.py（原工程资料） | 状态/动作窗口、单层 encode/predict/rollout/criterion |
| hjepa_utils.py（原工程资料） | 层级 loss、目标是否 detach、模型构造 |
| seq_encoder.py（原工程资料） | 动作块顺序编码 |
| hierarchical_solver.py（原工程资料） | 上层计划、下层子目标、跨层代价 |
| dataset.py（原工程资料） | 原始环境动作到五步动作块 |
| policy.py（原工程资料） | 计划动作拆块、反归一化、逐步执行 |

## 2. 一张图看完整数据流

```text
训练 episode：真实图像 o_i、真实动作 u_i
  │ 每 5 帧取一次图像；保留期间全部 5 个动作
  ▼
L1：z¹_t = E1(o_5t)                b¹_t = Embed1(concat(norm(u_5t:5t+5)))
  │ 点式 E2，每隔 2 个 L1 时间取一个状态    │ 每两个 b¹ 做有序块编码
  ▼                                      ▼
L2：z²_t = E2(z¹_2t)              v²_t = A2(b¹_2t:2t+2), b²_t = Embed2(v²_t)
  │ 同样构造 L3                           │
  ▼                                      ▼
各层独立 Fℓ(zℓ 的真实上下文, bℓ) → 下一真实编码的预测
各层 prediction MSE + SIGReg；所有层联合反传

规划：当前图像和目标图像 → E1 → E2 → E3
  F3 自回归 + 优化 4 维宏动作 → 第一个 z³ 子目标
  F2 自回归 + E3 投影代价 → 第一个 z² 子目标
  F1 自回归 + E2 投影代价 → 10 维动作块
  拆成五个 2 维动作；执行前两个块；真实观察到达后重新规划
```

这张图中 `v` 是规划器直接搜索的紧凑宏动作；`b` 是送进 predictor 的条件向量。官方代码中的 `action_0` / `action_1` / `action_2` 通常指 `b`，不是 `v`。这是最容易混淆的接口。

## 3. 时间单位和窗口究竟是什么

正文 Eq. (1) 的两条路径有不同的窗口：

\[
z_t^\ell=E_\ell(z_{ts:ts+w}^{\ell-1}),\qquad
b_t^\ell=\mathrm{Embed}_\ell\big(A_\ell(b_{ts+w-1:(t+1)s+w-1}^{\ell-1})\big).
\]

这里将论文的动作编码拆成了代码中的 `action_encoder` 和 `action_embed`。`JEPA._state_windows` 从 `ts` 开始取 `w` 个状态；`_action_windows` 从**该状态窗口最后一帧** `ts+w−1` 开始取 `s` 个动作。动作并非从窗口第一帧起算。

例如 `s=2,w=3`：上层状态 0 总结下层状态 `[0,1,2]`，上层状态 1 总结 `[2,3,4]`，两者间动作取 `[2,3]`。这是代码支持的几何；论文所有模型的 `w=1`，FourRoom 则始终是 `[z_0] --[a_0,a_1]--> [z_2]`。不应把一般窗口能力描述为已发表实验使用了跨帧状态汇聚。

FourRoom 默认配置：

| 模型层 | 相邻状态的原始环境间隔 | 状态编码 | 规划动作宽度 | predictor 条件宽度 | 训练上下文 |
|---|---:|---|---:|---:|---:|
| L1 | 5 | ViT-Tiny + projector | 10 = 5×2 | 192 | 3 |
| L2 | 10 | 点式 192→384→192 MLP | 4 | 192 | 3 |
| L3 | 20 | 点式 192→384→192 MLP | 4 | 192 | 3 |
| L4（对应四层配置） | 40 | 点式 MLP | 4 | 192 | 3 |

`stable_worldmodel/data/dataset.py` 的加载顺序是：

1. 读取完整原始动作序列；只有非动作列按 `frameskip=5` 抽取。
2. 动作 reshape 为 `[T,5,2]`，逐原始动作坐标按训练数据均值/标准差归一化。
3. 再 reshape 为 `[T,10]`。因此第十维动作块保留五个动作的顺序，**不是一个动作重复五次**。`PlanConfig.action_block` 的注释用了 repeated，但实际存储和执行是拼接后再拆开。
4. 给最顶层保留四个状态所需的 L1 长序列。三层时 L1/L2/L3 长度依次为 13/7/4；观测覆盖环境时间 0 到 60，loader 预留 65 个原始位置以容纳末尾动作块。论文最小观测跨度 61 与 loader span 65 是不同量。
5. `HJEPA.encode_hierarchical` 先形成所有层，再各取 `history_size+rollout_n=4` 的裁剪。默认稀疏 L1 编码先反推这些裁剪依赖哪些原始帧，只对这些帧运行 ViT，再放回全长张量。动作仍完整编码，因为上层需要完整动作块。

上层动作最后一个窗口可能越过序列尾部，代码补零并传 `action_mask`；训练的一步目标只使用存在后继状态的前三个动作。这个末尾补零不能解释成“训练过任意长度宏步”。

## 4. 原生 predictor 的实际计算

`JEPA.predict(emb, act_emb)` 仅调用 `ProjectedPredictor`。FourRoom 的路径为：

\[
[B,T,192]\ \xrightarrow{+p_{0:T}}\
4\ \mathrm{ConditionalBlocks}\ \xrightarrow{\mathrm{LayerNorm}}\
\mathrm{MLPProjector}\ \longrightarrow [B,T,192].
\]

每个 ConditionalBlock 的动作条件生成六组向量：

\[
(\delta_a,\gamma_a,g_a,\delta_f,\gamma_f,g_f)
=\operatorname{Linear}(\operatorname{SiLU}(b)),
\]
\[
x' = x+g_a\odot\operatorname{Attn}((1+\gamma_a)\odot\operatorname{LN}(x)+\delta_a),
\]
\[
x''=x'+g_f\odot\operatorname{MLP}((1+\gamma_f)\odot\operatorname{LN}(x')+\delta_f).
\]

`Attention.forward` 默认 `causal=True`；8 heads、每头 64、MLP width 1024。每层收到同位置动作 embedding；不是另外插入动作 token。最终 predictor projector 为 `192→2048→192`，中间 BatchNorm/GELU，默认不做最终单位球归一化。`Transformer` 的最终 LN 发生在该 projector 之前，**不能把其 hidden 当作已经输出的 JEPA latent**。

位置表按 `history_size=3` 建立，因此 FourRoom ARPredictor 直接接受 `T=1,2,3`；没有任意长上下文接口。`predict` 输出 T 个 one-step successor，不是 T 个逐层自回归未来。训练时三个位置分别预测真实上下文的三个下一步；只有调用 `rollout` 才会把自身输出接回输入。

上层点式编码器类名是 `ResidualLatentMLP`，但所读 `forward` 是 input LN→Linear→GELU→Linear→可选 final LN，**没有 `x + f(x)` 的残差相加**。FourRoom `final_ln=false`。应按执行式描述，不能按类名推断机制。

## 5. 动作抽象：真正的块接口

FourRoom `level2/level3.action_encoder` 未指定 `type`，`create_world_model` 默认选 `SequenceEncoder`，不是 `SequenceMLPEncoder`。

输入是 `[B,T,s,192]`。每个动作 token 先线性投影至 64，加入学习到的块内位置，prepend 一个 CLS；通过单层、2 heads、feedforward width 128 的 Transformer；取 CLS 经 `64→64→4` head 和最终 LN 得到 `[B,T,4]`。块内注意力不因果，但输入全部是**给定的候选动作**，不是未来视觉观察。然后普通 `Embedder(4→192)` 生成 predictor 的条件。

结构能处理 `L≤max_chunk` 的不同输入长度。FourRoom 上层 `max_chunk=3`，训练转移的实际长度固定为 2。**张量形状允许 L=1 或 3，并不赋予 F2 一步对应 5 或 15 环境步的含义。** F2 训练目标始终是 10 环境步之后的上层状态；编码器没有显式 horizon 参数。

DROID 是不同实例：上层动作 encoder 选择固定 stride=3 的 `SequenceMLPEncoder`，flatten 三个 7 维动作，hidden 64，输出 8 维并 final LN；`action_embed=Identity`，12 层的 `CausalTransformerPredictor` 在内部处理 8 维动作条件。FourRoom 与 DROID 不能共享一句“动作维度总是 192”的接口说明。

## 6. loss、detach 与归一化

`hjepa_forward` 调用 `encode_hierarchical` 后，对每层执行默认路径：

```python
teacher_pred = jepa.predict(emb[:, :3], act_emb[:, :3])
prediction_loss = mse(teacher_pred, emb[:, 1:4])
sigreg_loss = sigreg(emb.flatten(2).transpose(0, 1))
```

训练 MSE 对 batch、时间、维度取平均。所有层的加权 loss 相加。

| 机制 | FourRoom 发布配置 | 可选/其他配置 | 实际位置 |
|---|---|---|---|
| 目标 stop-gradient | 不启用 | `wm.detach_pred_target` 可启用 | `hjepa_utils.py:105` |
| 上层到下层 E 的梯度 | 连通 | 冻结式分阶段训练另属配置/流程 | `models/hjepa.py:257` 的连接没有 detach |
| 默认预测训练 | teacher-forced 一步 | `rollout_n>1` 加自由 rollout loss | `hjepa_utils.py:77` |
| 带状反复预测 | 未启用 | `wm.nsteps` 调用 `parallel_unroll` | `models/jepa.py:362` |
| 状态 SIGReg | H-JEPA 每层 0.18；flat LeWM 0.72 | DROID L1/L2 为 0.08/0.16 | 对应 train yaml |
| 动作 SIGReg | 不启用 | DROID L2 为 0.005 | `hjepa_utils.py:153` |
| IDM | 不启用 | DROID L1/L2 权重 100/50 | `hjepa_utils.py:161` |

三个不同的 detach 不可混为一谈：

- 可选 `detach_pred_target` 只截断 prediction target 分支；默认目标编码仍反传。
- IDM 把**动作目标** detach，而状态对 `(z_t,z_{t+1})` 保留梯度；DROID 的 action_embed 是 Identity，因此动作目标确实是归一化原动作/上层紧凑宏动作。通用训练函数传给 IDM 的是 `act_emb`，不能仅凭 loss 的 docstring 在所有宿主上称其为 raw action。
- `parallel_unroll` 的每一轮将上一轮预测 `.detach()` 后再作为下一轮输入，且重新注入 frame 0；这是可选训练支路，不是默认持续工作态，也不是通过所有迭代反传的 RNN。

SIGReg 实际输入 `[T,B,D]`，沿 B 算经验特征函数、再平均 T/随机投影。默认 1024 个随机单位方向、17 个积分节点、区间 `[0,3]`。代码没有将时间和 batch 合并成一个样本轴；论文正文 Eq. (3) 邻近文字的简写应按附录 K 和执行实现展开。DDP 分支同步 cos/sin 均值并调整样本数；这是一种分布约束，不能解释成直接保留某种物理内容。

附录 L 所讨论的背景型慢特征塌缩，对代码的含义也很具体：仅有预测 MSE+边际正态约束，仍可能让跨 episode 不同但 episode 内不变的背景承载大部分方差。DROID 的 IDM 要求状态对读出动作，从而添加转移动作相关约束；FourRoom 配置没有这项。原生 predictor 输出没有追加 L2 normalize，probe 的标准化和 planner 的平方距离不能回填为训练时已有的物理约束。

## 7. 从原生 rollout 到层次计划

### 7.1 单层 rollout 的时间合同

`JEPA.rollout(info, action_sequence, history_size=1)` 接受：

- `action_sequence`: `[B,S,T,A]`，S 是候选数；L1 的 A=10，上层 A=4。
- `info['pixels']`: `[B,S,C,3,H,W]`，C 表示初始真实上下文长度。
- 可选 `info['embed_0']`: `[B,S,C,D]`，给出时不再视觉编码，但 `pixels` 仍被读取以确定 C。
- 输出 `predicted_embed_0`: `[B,S,T+1,D]`，包含初始 C 个状态。

若要从 C 个真实状态预测 K 个未来，则动作长度是 `T=C−1+K`：前 C−1 个动作连接已有真实状态，后 K 个动作是未来候选。官方规划 C=1，所以 T=K，运行 K 次 predictor。

每次只截取最后 `history_size` 个 latent/action embedding，默认值为 1。即使保存的训练配置是 history=3，官方 `LevelCostModel.rollout` 也没有覆盖该默认值。训练首位置在 causal mask 下只看到自己，因而论文将 C=1 视作训练包含的上下文模式；这不等于训练和规划上下文完全相同。

rollout 在每一步把**累积动作前缀**重新送进 tokenwise `action_embed`，再截取最近一步/窗口；预测器调用数 O(K)，动作 Embedder 的总 token 处理数可达 `1+2+…+K`。缓存动作 embedding 是普通实现优化，不是 H-JEPA 的时间抽象机制。

### 7.2 顶向下求解

`HierarchicalSolver.solve` 每轮先 `_encode_all_levels` 重新编码真实当前图像和目标图像。该过程 no-grad，逐层只留最新一个状态。`w>1` 时函数可以重复首帧补齐窗口，但论文模型 `w=1`。

顶层直接在其自身 latent 内最小化到真实目标的距离。得到的预测序列仍含 context，所以下层取索引 1 开始的第一个预测作为子目标，不取索引 0 的当前状态。

下层对其候选动作自由 rollout 后，经**上层 encoder**把预测状态投到上层坐标，才与子目标比较：

\[
\min_{v^\ell}\ \big\|E_{\ell+1}(\hat z^\ell_s)-\hat z^{\ell+1,*}_1\big\|^2.
\]

它没有 decoder 将上层 latent 直接还原成下层 latent，也没有承诺上层宏动作一定可逆。下层优化过程承担找到可执行原动作的职责。FourRoom 关闭宏动作一致性代价，所以下层主要追随**状态子目标**，不要求下层动作经 A2 后等于上层被优化出的宏动作。Push-T/Cube 才开启 weight 0.2 的该项。

`LevelCostModel._encode_states_up` 在正常层间子目标时用上层 stride=2，因而只比较对齐的粗时间点；`*_project` 或跳过上层时可以设置 `dense=True`，用 stride=1 仅改变**目标代价投影采样**。它不把 F2 的物理预测步长改成 1。

当上层下降到 horizon=1，该层求解被跳过。FourRoom `horizon_one_goal_cost_space=upper` 保留最高被跳过层的真实目标表示，下层预测可连续投影至那里。这使“使用更抽象的目标代价”和“执行宏步预测”在代码上可以分开调用。

### 7.3 代价和真实执行

`JEPA.criterion` 先移除 context，再取最后 `cost_last_n` 个预测，对每维平方误差求和（不是训练 loss 的全维平均）。FourRoom flat 默认取末尾两步，上层末尾一步。下层按 stride 投到上层后可能只剩一个对齐未来，此时即使 cost_last_n=2 也只会取现有的一步。`intermediate_cost_weight` 只在多个被评分时间点时生效。

梯度求解器只把候选动作张量交给 AdamW optimizer；`model.eval()` 控制 dropout/BN，但本身不等于所有权重 `requires_grad=False`。在小型固定宿主调用中可以关闭参数梯度，同时保留对 action 输入求导的能力。优化动作不是更新模型参数。

上层紧凑宏动作没有原始物理边界：`JEPA` 在训练模式下每 batch 保存时间索引 0 的宏动作到最长 4000 的队列；规划按队列逐维均值/标准差初始化，并按 2%–98% 分位裁剪。论文说保留最近训练宏动作，代码的具体采样是每条序列的第一个窗口。该队列是搜索先验，不是在线吸收真实经历的工作记忆。

默认 FourRoom 三层 horizon 是 `(2,2,6)`，候选数 `(50,200,200)`，最多 90 次优化、最少 30 次后可因停滞提前结束。solver 返回最优候选后还执行一次原生 rollout，以向下层传状态子目标。

`WorldModelPolicy` 只接收最终 L1 动作 `[B,H,10]`，取前两个块 reshape 成 `[B,10,2]`，逐环境步反归一化并执行；buffer 耗尽后从新真实图像重新求解。规划器没有显式把上一轮的完整 latent trajectory、Transformer hidden、KV cache 或宏步状态作为下一轮 predictor 的持久输入。`GradientSolver.init_action` 支持手动传初始动作，但这条默认层次路径没有传前轮计划。

## 8. 最低成本的原生算子调用

以下是接口示例，不是已经运行的实验。假设 `model` 是同一个已加载、已设 eval 的 Three-level FourRoom H-JEPA 对象；`z0` 来自它自己的 L1 encoder，`a10` 已按该对象训练统计归一化。

### 8.1 两个细步与一个原生粗步

```python
# z0: [B,1,192]; a10: [B,2,10]，两个完整五环境步动作块。
l1, l2 = model.get_level(1), model.get_level(2)
b1 = l1.action_embed(a10)                 # [B,2,192]
z1 = l1.predict(z0, b1[:, :1])           # [B,1,192]
z2_fine = l1.predict(z1, b1[:, 1:2])     # [B,1,192]

# FourRoom w=1，因此可直接点式编码；无随机裁剪、无未来图像。
z0_upper = l2.encoder(z0[:, 0])[:, None] # [B,1,192]
v2 = l2.action_encoder(b1[:, None])      # [B,1,4]，块内 L=2
b2 = l2.action_embed(v2)                # [B,1,192]
z2_upper = l2.predict(z0_upper, b2)      # [B,1,192]
z2_fine_projected = l2.encoder(z2_fine[:, 0])[:, None]
```

可比较 `z2_upper` 与 `z2_fine_projected`；两者都在 L2 坐标，对应 10 个原始环境步。该比较不需 planner、不需新训练，也不需真实未来输入预测分支。需要真实终点时，单独作为结果对照编码。

在固定动作序列上递归对 `b2` 做相同二步汇聚即可得到 L3 条件。注意传给 A3 的是 **Embed2(v2)**，不是 4 维 v2。所有 H 都应先用明确的原始环境时间单位表达。

### 8.2 已编码起点的官方 rollout

```python
# pixel0: [B,1,3,224,224]，已做宿主图像归一化。
# z0: [B,1,192]；actions: [B,S,H,10]。
info = {
    'pixels': pixel0[:, None],          # [B,1,1,3,224,224]
    'embed_0': z0[:, None],             # [B,1,1,192]
}
pred = l1.rollout(info, actions, history_size=1)['predicted_embed_0']
# pred: [B,S,H+1,192]，第 0 项是输入状态。
```

不要为了取原生宏步而调用 `encode_hierarchical`：它承担训练随机裁剪和样本窗口准备，不能当成保持全部时间点的在线 encoding API。需要纯编码时分别调用相应 level.encoder 或不带动作的 `level.encode(..., chunk_temporal_inputs=True)`。

### 8.3 可以直接回答接口含义的小型调用

这些是给后续调用者的建议，本文未执行：

1. **动作顺序**：固定 z0，对同一两个动作块交换顺序；记录 A2 的输出、两细步输出、一个粗步输出。问题是顺序怎样进入原生宏动作，不需要启动行动搜索。
2. **上下文**：选取同一终止时刻，比较 `F1(z_t,a_t)` 与 `F1(z_{t-2:t+1},a_{t-2:t+1})[:,−1:]`。三个状态必须均已真实到达；只比较同一条下一步预测。T=3 不能被误记为三个未来预测步。
3. **相同物理跨度**：10 环境步用 L1 两次或 L2 一次；20 环境步用 L1 四次、L2 两次或 L3 一次。先投到同一个合法层坐标，再描述原生算子间的差别；不能对不同层直接算 latent 距离。
4. **抽象代价与抽象动力学分离**：同一 L1 rollout 可以只过 E2/E3 形成目标距离，完全不调用 F2/F3；因此代价变化不能自动归因于宏步动力学。

当前如果仍只加载 **flat FourRoom LeWM checkpoint**，只有 L1。没有可直接调用的已训练 E2/A2/F2；换进独立训练的 H-JEPA 上层也不合法，因为其下层编码坐标来自另一组联合训练参数。要研究该 flat 宿主内部的持续多时域方法，H-JEPA 提供的是方法参照和另一个已有模型，不是无需训练即可插上的宏步部件。

## 9. 对本项目方法构造最有用的边界

H-JEPA 已有的计算事实是：**独立训练粗时间动力学 + 学习每层状态/动作抽象 + 用上层坐标约束下层行动**。长期预测的计算节约来自大步长上层预测和缩短下层行动搜索，不来自复用原生 F1 的历史 hidden 或把已经执行的 F1 计算编译成宏步。

没有在上述主路径中发现以下机制：同一个 predictor 被一个 horizon 参数改成任意步长；上层一步严格等于下层多步组合；只靠当前已有 native computation 自动生成新宏步；真实观察到达后在持久的中间工作态上续算。不同步长的监督通过各层真实编码连接，并不包含显式的跨层动力学交换/接续等式。

因此，若本项目只是给一个 memory token 再配一个动作块 encoder，训练其输出较远未来，那仍非常接近“递归记忆 + 已有宏步预测”。真正需构造的差异不能由 H-JEPA 的代码接口代答：**什么可执行的中间计算被保留，怎样在真实观察变化后继续使用，以及宏步怎样从原生预测计算本身产生。** 本文提供现成算子的精确边界，不把这些尚未实现的性质算入已有方法。

