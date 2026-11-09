# LeHami：论文公式如何成为实际的预测运算

本文对应 [LeHami v1 正文及附录](https://arxiv.org/html/2610.15267v1) 和作者的 [rominay/LeHami 官方仓库](https://github.com/rominay/LeHami)。源码阅读日期为 2026-11-07，读取的是公开 main 分支；没有下载论文 checkpoint，没有训练或运行本文描述的完整实验。源码副本位于 LeHami 来源目录（原工程资料）。本文只记录读到的代码行为，不把论文公式、类的默认参数、论文实验配置混为一件事。

## 1. 从哪条路径开始读

论文主结果对应的是动作摆实验 a0q。入口由三个文件共同决定：

- [实验配置 a0q](https://github.com/rominay/LeHami/blob/main/lehami/training/configs/experiment/phase2_lehami_sg_sep_conv_w1_swingup_exp079_a0q.yaml)：q 维数 32，逐帧 encoder，SIGReg 权重 0.25，训练 35 epochs，batch 32。
- [模型配置](https://github.com/rominay/LeHami/blob/main/lehami/training/configs/model/lehami_sg_sep_sreg_q_ms_conv.yaml)：直接相空间、可分离 Hamiltonian、可学习 momentum scale、三层宽 1024 的势能/动能网络。
- [train.py](https://github.com/rominay/LeHami/blob/main/lehami/training/train.py#L234)：数据 → registry builder → stable-pretraining Module → Lightning Trainer / Manager。

调用关系是：

~~~text
train.main
  ├─ _build_dataset
  ├─ REGISTRYcfg.model.name（原工程资料）
  │    └─ _build_lehami_sg_sep_sreg_joint_ms_conv
  │         ├─ ConvEncoder + MLP projector
  │         ├─ Embedder(action)
  │         └─ HamiltonianPredictor
  │              └─ SeparableHamiltonianFlow
  │                   └─ SeparableHamiltonianNet.K / V
  └─ spt.Module(forward=builder返回的_forward, optim=AdamW)
       └─ _forward：编码完整训练片段 → 构造(q,p) → 自由展开8步 → loss
~~~

重要之处是 builder 返回了**自定义训练前向闭包**。这里并不是调用一个通用 JEPA.forward，就自然得到论文的 SG momentum、八步自由展开和 phase-space loss。模型容器复用了上游 JEPA；论文方法由 builder 的闭包和新 predictor 共同实现。builders.py:821（原工程资料）

上游依赖也须分开：JEPA、MLP、Embedder、SIGReg 来自外部 LeWorldModel；HDF5Dataset 与训练调度来自 stable-worldmodel / stable-pretraining。作者仓库没有内嵌这些完整依赖，本轮没有安装它们或宣称跑过其内部训练调度。所需新动力学算子则完整公开在 ham_flow.py，文件自身只需常见 Python 数值与 PyTorch 依赖。

## 2. 式 (1)：状态并不是两个自由的 memory token

论文 [式 (1)](https://arxiv.org/html/2610.15267v1#S2.E1) 的 q 是观测编码，p 是 q 轨迹的时间导数。代码把这个定义落在两个位置：

1. [ConvEncoder](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/builders.py#L36) 依次使用通道 3→32→64→128→192，每块卷积、BatchNorm、GELU、下采样，最终全局池化。
2. builder 使用上游 MLP projector 将 192 维 encoder 输出映成 32 维 q；p 再由 q 的固定线性差分生成，并非第二个任意学习的 head。

主配置一条训练片段的形状为：

| 量 | 形状 | 含义 |
|---|---|---|
| pixels | B×14×3×64×64 | 包含起点过去帧、8 步目标、差分目标尾部帧 |
| encoder 输出 | B×14×192 | 逐帧视觉特征 |
| z，即论文 q | B×14×32 | projector 后的坐标 |
| p_ref | B×14×32 | 居中 SG 为主的导数序列 |
| 起点 q、p | 各 B×32 | anchor 为片段第 3 帧 |
| 一次 predictor 输入 | B×1×64 | concat(q,p)，没有跨时间 attention |
| action embedding | B×1×64 | 原始转矩 B×1×1 经 Embedder |
| predictor 输出 | B×1×64 | 下一步 q、p |

这里 history_size=1 表示每次 flow 消费一个**已构造好的相空间状态**，不表示只需要一个真实图像。构造起点 p 要四个真实 q，即帧 0、1、2、3。

[derivative.py](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/derivative.py#L32) 用 NumPy 计算固定最小二乘多项式导数系数，运行时转成对应 device / dtype 的 tensor，再使用 einsum。对均匀时间步，其两个内部 stencil 可写为：

\[
p^{causal}_t=
\frac{9q_{t-3}-17q_{t-2}-13q_{t-1}+21q_t}{20\Delta t},
\quad
p^{centered}_t=
\frac{-2q_{t-2}-q_{t-1}+q_{t+1}+2q_{t+2}}{10\Delta t}.
\]

p_causal_last 用于在线可得到的起点；p_interp_series 在训练目标的内部时刻用 centered stencil。最前两帧的 backward difference、最后两帧的 causal fallback 用于让整段数组有定义；主配置的预测目标取在 centered stencil 有效区域，不靠这些边缘值训练未来目标。

这使 p 具有明确经历含义：同一个 q_t 可以因过去的 q 不同而带有不同 p_t。进入自由预测后，p 由动力学继续推进，不能重新用真未来观察算 p 再喂给线上路径。

## 3. 式 (3)–(4)：保证来自 K/V 的依赖关系，而非函数名称

论文 [式 (3)](https://arxiv.org/html/2610.15267v1#S2.E3) 对应 SeparableHamiltonianNet。其 K 网络只看 p，V 网络只看 concat(q,u)：

\[
K:\mathbb R^{32}\to\mathbb R,\qquad
V:\mathbb R^{32+64}\to\mathbb R.
\]

各自 MLP 为 Linear→Tanh→Linear→Tanh→Linear，hidden 1024，输出层没有 bias。这个标量网络与常规 predictor 的向量输出不同：它输出势函数，再对输入求导，形成状态变化。动作不直接追加成任意残差力，而是进入 V(q,u)。[ham_flow.py:199](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/ham_flow.py#L199)

scale 参数是 log_p_scale∈R^32，实际 s=exp(log_p_scale)>0。K 在 p/s 上求值。论文 [式 (4)](https://arxiv.org/html/2610.15267v1#S2.E4) 在 [SeparableHamiltonianFlow.forward:362](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/ham_flow.py#L362) 被直接实现：

\[
\begin{aligned}
g_1&=\partial_q V(q,u), &
p_{1/2}&=p-\tfrac h2 g_1,\\
g_2&=\partial_{p_{1/2}}K(p_{1/2}/s),&
q'&=q+h g_2,\\
g_3&=\partial_{q'}V(q',u),&
p'&=p_{1/2}-\tfrac h2 g_3.
\end{aligned}
\]

每次 forward 是三次标量网络求值、三次 autograd.grad：V(q,u)，K(p_half/s)，V(q_next,u)。求 K 对 p_half 的导数时，autograd 自然包含 1/s 的链式项，不能在公式之外再手动乘一遍 scale。

调用 HamiltonianPredictor.forward 时，输入 B×T×64 先 flatten 成 BT×64，切成 q/p，flow 后再拼回原形。主训练 T=1；wrapper 支持 T>1 时，是对各位置独立调用相同 flow，不意味着这些位置之间有 attention 或历史归纳。[ham_flow.py:542](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/ham_flow.py#L542)

附录 [C.1](https://arxiv.org/html/2610.15267v1#A3.SS1) 的两类 shear 在这条路径里成立，是因为 K 不依赖 q、V 不依赖 p，且 Tanh 网络可微；每次 kick 的 Hessian 是对称矩阵。因此固定外部动作下，雅可比满足 MᵀJM=J，det(M)=1。它不是通过给 determinant 加 loss 来学出来的。

## 4. 梯度图中容易被摘要遗漏的两件事

**第一，推理也要对状态求导。** forward 开始记录外层 torch.is_grad_enabled()，内部强制 torch.enable_grad()。即使 CEM 以 no_grad 包住调用，仍须计算 ∂V/∂q、∂K/∂p，才能产生下一状态。外层 no_grad 时 create_graph=False，仅不为后续学习保留高阶计算图。

训练时 create_graph=True，后续 loss.backward() 会通过这些导数再对网络参数与状态求导，包含混合二阶导数。_ensure_grad 在输入已有 requires_grad 时返回原输入，没有切断 encoder→q/p→flow 的图；只有无梯度的普通输入才被 detach 后设为可求导叶子。[ham_flow.py:53](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/ham_flow.py#L53)

**第二，主路径内部固定用 fp32。** flow 显式关闭 autocast 并执行 q.float()、p.float()、u.float()。外部训练配置是 bf16-mixed，不等于 Hamiltonian 导数在 bf16 中算；也不能把模型改成 double 后就宣称运行的是 float64 官方 flow。输出最后转回原 q/p dtype。小型算子实验应保留这一事实。

还应区分 Module.eval()、no_grad、inference_mode。eval() 本身不关闭 autograd；这个 forward 刻意在 no_grad 中重新开启局部求导，并不能由此推断它支持所有 inference_mode 用法。

## 5. 式 (5)：实际八步学习循环以及未来目标的位置

主训练闭包在 [builders.py:881](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/builders.py#L881) 中完成：

1. 用同一 encoder / projector 编码 14 帧，得到 z。
2. q_pred=z[:,3]；p_pred=p_causal_last(z[:,:4])。
3. 第 j 次循环读取动作 a_(3+j)，而目标索引是 4+j。
4. predictor 仅接收上一轮预测 q_pred、p_pred 和该动作。
5. 八步后，分别平均 q MSE 和 p MSE。
6. 真实编码 z 在索引 2…11 上做 q-only SIGReg。

因此原论文 [式 (5)](https://arxiv.org/html/2610.15267v1#S2.E5) 的主配置实际为：

\[
\mathcal L=
\frac18\sum_{j=1}^{8}
\left[
\operatorname{MSE}(\hat q_{3+j},q_{3+j})
+
\operatorname{MSE}(\hat p_{3+j},p^{centered}_{3+j})
\right]
+
0.25\,\operatorname{SIGReg}(\{q_2,\ldots,q_{11}\}).
\]

PyTorch mse_loss 默认对 batch 和坐标维取平均；上述 MSE 不是未归一化的坐标平方和。训练还对时间 horizon 取平均。

**代码中的 q/p target 没有 detach，也没有 EMA target encoder。** 未来 q 目标和由它们构造的 p_ref 都向 encoder / projector 传回梯度，起点也传回梯度。这里只有日志中的 v.detach() 和无梯度统计，不可把它们误读成学习目标 stop-gradient。这是对上轮“联合学习”说法的具体化，也禁止直接套用其他 JEPA 的 target 机制。

未来帧仅在离线编码目标/损失中出现。自由 rollout 第 j 步不重新读取 z[:,3+j]；下一步来自刚生成的 qp_next。这是多步训练，不是每步 teacher forcing。

_action_emb 的 W=1 路径读取 batch action[:,3+j:4+j]。W>1 可选 temporal encoder 则使用 offset=(W−1)−lag，将 latent 时间映射回实际动作时间；这里 lag 是 encoder 的属性，不能固定写成 W−1。[builders.py:414](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/builders.py#L414)

训练循环外层使用 AdamW，lr=5e-5，weight_decay=1e-3；配置 LinearWarmupCosineAnnealingLR，gradient_clip_val=1。main 配置仅给出这些 scheduler 名称，没有显式展开库内部的全部 warmup 默认值，本文不补写未知值。输入为 ImageNet 归一化后 64×64 图像，不只是像素除以 255。[train.py:55](https://github.com/rominay/LeHami/blob/main/lehami/training/train.py#L55)

## 6. 推理、观察刷新、时间步：哪些能力是真的

run_eval.load_model 从 checkpoint 相邻的 config.yaml 重建 registry 模型。_build_anchor 使用最后四个可用 q 的 causal SG 构造状态；主规划代码 anchor_from_angles 复用这一函数，避免用 history_size=1 错当成一帧即可构造速度。[run_eval.py:606](https://github.com/rominay/LeHami/blob/main/lehami/experiments/src/run_eval.py#L606)

[latent_rollout](https://github.com/rominay/LeHami/blob/main/lehami/experiments/src/plan_pendulum_swingup.py#L263) 对候选动作逐步循环，每步经原 action_encoder 再调一次 HamiltonianPredictor。CEM 外层 no_grad，返回 terminal qp 或逐步列表，latent_cost 比较原生 q/p：

\[
c=\|q-q_g\|_2^2+\lambda_p\|p-p_g\|_2^2 .
\]

规划这里是平方和，与训练 MSE 不同；两个 q/p block 同维时，统一的维数因子不改变候选排序。running cost 是逐步求和。[latent_cost:608](https://github.com/rominay/LeHami/blob/main/lehami/experiments/src/plan_pendulum_swingup.py#L608)

在真实观察到来后的下一次规划，代码重新用已观察帧构造新的 q/p 起点，然后重新预测候选未来。它没有学习一个跨所有 replan 永不重置的额外 memory，也没有保存上一轮预测树用于增量重算。

dt 的主值是 1.0，表示训练 frame 的时间单位；实际摆数据帧间隔为 0.1 秒。可选 dt_ref 机制用 h=dt_episode/dt_ref，同时将差分 p 除以 h；这段代码存在，不等于论文主 a0q 配置启用了混合采样率训练。缩小积分步可在同一学得的生成函数上细分轨迹；主代码仍每个子步各调用一次 flow。论文的 sub-frame re-stepping 不能推出任意大步跳转准确，也不能推出长 horizon 推理少于 O(H) 次 flow。

## 7. 默认配置和可选实现必须分开

| 选择 | 论文主 a0q | 仓库还提供什么 |
|---|---|---|
| 表示 | 直接(q,p)，总64维 | z→qp→z 的线性投影桥接模式 |
| Hamiltonian | K(p/s)+V(q,u) | joint H(q,p,u)，以及 Euler / 非可分离显式更新 |
| K 的偶对称性 | even_kinetic=False | symmetrise、square、quadratic |
| encoder 时间窗 | W=1 | temporal W>1，多种 temporal mix |
| SIGReg | 只作用 q | joint(q,p)、q/p split |
| 时间步 | h=1、fd_dt=1 | dt_ref+每样本h |
| 视觉骨干 | 从头训练ConvEncoder | frozen DINOv2 另配 adapter / projector |

HamiltonianPredictor 类自身默认 phase_space=False、separable=False、hidden_dim=256，**不等于论文默认**。论文实验配置将它们覆盖成 true、true、1024。z→qp→z 桥接不继承直接相空间上的体积结论；joint H 的显式函数也不能因为名叫 Störmer–Verlet 就自动获得可分离 leapfrog 的证明。[wrapper 构造器](https://github.com/rominay/LeHami/blob/main/lehami/models/lehami/ham_flow.py#L492)

even_kinetic=False 下，固定动作的“先 h、再 −h”的代数逆，和翻转 p 后正向运行的物理时间反演，是不同性质。后者要 K 的偶对称性等额外条件。主默认不能继承可选 quadratic / symmetrise 分支的全部结论。

## 8. 相对上轮笔记需要补上的具体认识

- 主方法训练目标的 encoder 梯度未被切断。不能把它描述成仅在冻结坐标中拟合一个新动力学器。
- 32 是 q 的维数，phase-space 总维数 64；原始 ConvEncoder 输出仍是 192。三个维数不能互换。
- 代码中的 h=1 是 frame 单位，论文物理采样时间 0.1 秒不是直接作为主 flow 的 dt。
- 这项工作改造了预测器本身，并重新联合学习表示。它没有展示“给现有冻结 FourRoom LeWM 附加一个工作态，同时保持原 predictor 不变”的完成方案。
- 每个预测步的成本是两次 V、一次 K 和三次输入导数；参数少或单步 FLOPs 较低，不等于复用已完成的历史计算。
- 附录的几何证明约束模型映射，不保证已编码 q/p 对真实动力学的准确性。随机初始化的 flow 也满足结构性质，这恰好说明该性质与学会世界的能力应分开研究。

## 9. 一个最小机制实验建议

先直接 import 官方 ham_flow.py，通过 HamiltonianPredictor(phase_space=True, separable=True, learnable_scale=True) 调用同一 flow；将 q/p 各缩成 2 维，hidden 缩成 32，仅用于看清算子。明确这是**官方类的小尺寸实例**，不是论文训练 checkpoint，也不是新方法。

固定一小批 q/p 和外部动作，观察一次、重复多次、再反向步长时的状态传播；同时记录 qp 雅可比的 symplectic residual、determinant、q/p 分块变化及实际计算量。这样可以直接看到“一次结构选择怎样约束每一次前向”，无须视觉 encoder、训练、CEM 或任务成功率。

官方实现内部 fp32，因此前向须保留 fp32；若用 float64 汇总矩阵量，须单独注明。经 wrapper 传显式步长时采用每样本形状 [B]，与官方主训练路径一致。这个实验只能解释结构约束，不回答模型是否学到物理状态，也不提供我们的持续多时域创新。下一项构造须有同样明确的单一运算及其能力来源，不能把它的体积保证改名借用。

