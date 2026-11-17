# 路径 A：D-JEPA 原文核对与 FourRoom 决策间隔的可训练条件

2026-11-16 · 核对原文、作者代码与本项目计算图；完成算式与[现有配对模型零更新梯度检查](../../runs/native_topology_paired_20261115/writein_attribution_20261116/decision_gradient/REPORT.md)，未训练、未改变规划或运行闭环。

**D-JEPA 的 87.89% 是 PushT 的 225/256，不是 FourRoom。** 这是关系选择与受限预测更新的校准组合结果；同协议的纯关系选择为 223/256，受限预测更新独用为 204/256。论文确实训练决策相关成对间隔，但配对的是同根候选池中的成功和失败动作，没有对称隐藏模式训练。本项目可借鉴用真实动作后果保护部署决策的间隔。直接对硬分房分数套排序损失，会给部分错房间预测错误方向的梯度。

## 原始来源与 87.89% 的口径

用户提供的 Google 链接经内部 Vertex 重定向最终指向 [natemurthy/world-models 综述](https://github.com/natemurthy/world-models#ref-liu-et-al-2026)。综述的 D-JEPA 行写的也是 **PushT 87.9%**，并指向 Shuaijun Liu 等八位作者的 [arXiv:2609.24749v1](https://arxiv.org/abs/2609.24749v1)（2026-09-21）《D-JEPA: A Decision-Aligned Latent World Model》。核对了[作者站](https://nebulis-lab.com/D-JEPA/)、[正文 §3–6](https://arxiv.org/html/2609.24749v1#S3)、[附录 A–D](https://arxiv.org/html/2609.24749v1#A1)、[附录 H](https://arxiv.org/html/2609.24749v1#A8)和作者站链接的 [NEBULIS-Lab/D-JEPA](https://github.com/NEBULIS-Lab/D-JEPA) 源码和协议。DINO-WM 是对照和可选额外预测来源；Dreamer 是另一模型族。原文、作者代码发布清单和[训练数据卡](https://huggingface.co/datasets/Shuaijun/D-JEPA-Dataset)都没有这项 FourRoom 结果。

- **对象与分母。** PushT 的 256 个确认起点与拟合、校准及开发身份分离；按预先规定的成功/失败结构和较小的预训练排序边界间隔纳入。各方法共享恢复状态、目标、可选动作池和执行时间。[候选保留代码](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/src/djepa/candidates/controls.py#L123)从 256 条提案中去掉示范 ID 0，按 TD-JEPA 原生代价保留前 16 条，再按扰动家族覆盖补 47 条。动作一旦选定，执行整条 25 控制序列（250 个模拟器积分步，2.5 秒），以最终环境状态判成功。[协议与输入](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/docs/PROTOCOLS.md#populations) · [物理执行接口](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/src/djepa/native/pusht.py#L191)
- **对应行。** [正文表 1](https://arxiv.org/html/2609.24749v1#S4.T1)和[附录表 8、9](https://arxiv.org/html/2609.24749v1#A3.T9)报告组合 225/256、LeWM 214/256，配对为 19 次改善、8 次退步，差 4.30 个百分点，配对区间 [+0.39,+8.20]。[正文表 6](https://arxiv.org/html/2609.24749v1#S4.T6)的 PushT 原生距离精确实现是**同一个关系选择**的 223/256、秩及动作回收 100%；组合结果是不同部署行。
- **适用范围。** 该结果衡量已筛过的固定池中一次整序列选择，未检验隐藏门锁证据的持续更新、只执行首块后的反复重规划、官方 FourRoom 50 任务或自建可逆任务。项目已有静态平面 LeWM 19/50 是自己的参照，两者分母及任务分布均不可比。

## 论文的计算和作者实现

**关系选择（§4.1、式 1–2，附录 A.2、B.1 式 6–7）。** 对每条候选 \(i\)，LeWM 与 TD-JEPA 分别提供 192 维终点目标差 \(d_i^m=\operatorname{LN}(\hat z^m_{i,H}-z_g^m)\)，以及同池固定候选 ID 解平局的原生代价秩 \(r_i^m=(\operatorname{rank}c_i^m-1)/(K-1)\)。将两描述和两秩拼成 386 维 token；\(b_i=\alpha r_i^L+(1-\alpha)r_i^T\)。共享 386→64 编码、两层 4 头无候选位置编码的 Transformer，输出经 64→8→1 零初始化残差头：

\[
\delta_i=\epsilon\tanh(W_{\rm up}\tanh(W_{\rm down}h_i)),\qquad s_i=b_i+\delta_i,\qquad \epsilon=0.2.
\]

最低分更优。令 \(p_i=\operatorname{softmax}_i(-s/T)\)，\(y_i\) 是真实执行的二元成功标签，\(\mathcal T_k\) 是当前对齐分数最低的 \(k\) 个候选：

\[
\mathcal L_R=-\log\sum_{i:y_i=1}p_i+
\lambda_{\rm local}\underset{\substack{i,j\in\mathcal T_k\\ y_i=1,\;y_j=0}}{\operatorname{mean}}
\operatorname{softplus}\!\left(\frac{\mu+s_i-s_j}{T}\right)
+\lambda_{\rm trust}K^{-1}\sum_i\delta_i^2 .
\]

PushT 为 \(K=63,\ k=16,\ T=0.05,\ \mu=0.02,\ \lambda_{\rm local}=0.25,\ \lambda_{\rm trust}=0.1\)。子池只有一个类别时用全池正负对，全池训练行缺任一类别则报错。\(\mu\) 是归一化分数的软间隔。代码以 detached 的当前分数选 top 16，从仍可导的分数取值构成正负笛卡尔积；训练 shortlist 的排序直接用 PyTorch argsort，固定候选 ID 则明确用于原生秩和部署选择。部署 gate 在 \(b_{i_b}-s_{i_s}>\tau_R\) 时取关系首选，否则取基线首选。作者报告 \(\alpha=0.42,\ \tau_R=-0.0316252634\)；精确实现 gate 比较前将优势取到四位小数，独立关系 profile 不取整。残差只能改变基线间隔 \(\leq2\epsilon\) 的排序，这条界不适用于整个校准组合。[关系前向与损失](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/src/djepa/models/relational.py#L67) · [门控说明](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/docs/CHECKPOINTS.md#gates-and-composition) · [原文间隔界](https://arxiv.org/html/2609.24749v1#A8.SS2)

**受限预测更新（§4.2、附录 A.2、式 8）。** 此支把排序梯度送进 TD-JEPA 的最后预测块和 projection，维持 eval 模式。分数 \(c_i^P\) 是学生预测终点对目标 latent 的 MSE。完整候选仍做成功概率质量；用固定参考分数和 ID 选头部子集（缺一类时扩大），正负对仍用 \(\operatorname{softplus}((\mu+c^P_i-c^P_j)/T)\)。作者代码把教师对学生的非头部 softmax KL、头部**同类**的温标归一分数差 Smooth L1 放在 rank preservation 中；latent preservation 是层归一化后学生与冻结教师的**终点** MSE，不能据附录 A.2 的概述推称每一步都有教师监督：

\[
\mathcal L_P=\mathcal L_{\rm success}
             +0.5\,\mathcal L_{\rm boundary}
             +0.25\,\mathcal L_{\rm rank}
             +1.0\,\mathcal L_{\rm latent}.
\]

独用时读学生原生终点代价。组合以关系动作为默认，用校准优势接纳预测更新的提案（阈值 0.0065339543）；作者公开缓存实现要求**用适配后未来重新送入关系头**得到提案分数，不能直接拿原始适配后 MSE 与关系分数比较。四源版本另加 DINO-WM 与 JEPA-WM 秩，不是表 1 的双源组合的必要组成。[梯度范围和损失](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/src/djepa/models/plasticity.py#L116) · [组合接口](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/src/djepa/evaluation/inference.py#L52)

**原生距离读出（§4.3、式 3–4，附录 B.3、D.2、H.4）。** PushT 的精确实现按关系门控后的严格秩 \(\pi_i\)，沿原目标差方向改写 TD-JEPA 终点 \(\tilde z_{i,H}=z_g+\frac{\pi_i}{K+1}u_i\)（\(\|u_i\|_{\rm RMS}=1\)，零差用固定方向），前四步保留。原生 MSE 因而为 \((\pi_i/(K+1))^2\)。代码核对全秩和 argmin；合包仍含两个预测器和关系头。Reacher 则用 \(\beta_{it}=0.1\tanh f(r_i^L,r_i^T,r_i^R,q,t/H)\)，沿同动作同帧的 \(z^L_{it}-z^T_{it}\) 归一方向做有符号更新。训练项为按成功、物理成本和 ID 排出的首选交叉熵、两读出的全对 softplus 排序及轨迹保持 MSE。Reacher 的 120/128 是关系选择的执行结果，时间传输只另报表示诊断。[PushT 前向与检查](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/src/djepa/models/exact_realization.py#L177) · [Reacher 损失](https://github.com/NEBULIS-Lab/D-JEPA/blob/main/src/djepa/objectives/transport_objective.py#L59) · [作者模型卡](https://huggingface.co/Shuaijun/D-JEPA)

## 路径 A 在本项目怎样成立

**要纠正的量。** [§42 的归因](../../runs/native_topology_paired_20261115/writein_attribution_20261116/REPORT.md)显示，真模式仍无法修复主要的共同代价错误，模式作用的跨动作幅度及方向也不稳。保留现有同根四槽、两模式、H1–H5 真 latent、真实前缀一步、gate BCE 和部署，只问：同根同池中，已实际重播的两条不同首块候选，预测的物理终点能否保住真实代价次序和真实模式换序？沿用 §42 指定的候选构造：共同受阻/合法绕行对和两模式真优选见证同时保留，重合去重，同一对完整动作在两模式均执行。物理无严格顺序、完整首块相同或在该模式的实际首块运动相同的，不算首块见证。共同对在两模式同向；条件换序须同一对在两模式都严格且反向。候选 ID 或模型分数不定义物理序。训练只用原拟合布局与可见地图，生命周期固定池留作检查。

**直接套排序式的问题。** 当前 [forecast 与读出](../p1a/native_topology_condition.py#L79)、[fit](../p1a/native_topology_prefix_fit.py#L72) 和 [TorchVisibleMapCost / ForecastCost](../p1a/native_topology_mpc.py#L36) 实现：

\[
\hat z_{a,m,1:H}=F_\theta(o_{\le t},u_{\rm past},a,m),\quad
\hat x_{a,m}=R(\hat z_{a,m,H}),\quad
\widehat J_{a,m}=G(\hat x_{a,m}).
\]

NumPy \(G\) 会离开计算图；Torch 版只能对当前房间选中的路径距离求导。室号由 \(\hat x_x,\hat x_y\ge112\) 定离散索引；路线 min 也只对当前取值的分支传梯度。房内 \(G_r(x)=\min_d(\|x-v_{r,d}\|_2+\ell_{r,d})\) 是 1-Lipschitz，但跨中线时 \(G\) 可跳变。使用 \(\operatorname{softplus}((\mu+G(\hat x_{\rm good})-G(\hat x_{\rm bad}))/T)\) 既可能因为虚报入房而给出合意排序，也可能把已经错房的预测推得更深入错误房间。例：同一示例布局 2026111600、东南目标 \((142,140)\)，固定 \(y=127\)，在 \(x=111.999\) 与 \(112.001\) 两点的原硬代价分别为 55.9531、32.6946。

另一个门槛是既有 [first-effect gate](../p1a/native_full_depth_conditioning.py#L161)：\(\operatorname{ReLU}(2\sigma(\mathrm{logit})-1)\) 在严格负 logit 处给模式专有残差零梯度。正确的坐标梯度未必能到实现条件差的参数。训练须沿现有 eval 模式、compact=False 的预测图，用固定读出权重传输输入梯度；读出调用不能放进 no_grad/inference_mode 或转 NumPy。保留 factual BCE；不强制曝光，不引入可学习软分房。

**由硬代价推出的一个配对监督。** 这是本项目自己的训练候选。对于真严格有序、两真终点都不在线上的同根同模式对 \((a,b,m)\)，设 \(y_a,y_b\) 为物理 H5 终点，\(q_i=G(y_i)\)，\(a\) 为低成本者，\(\gamma=q_b-q_a>0\)。令 \(C=112,\ W=2C\)，\(e_i=\|\hat x_i-y_i\|_2\)，每个坐标的真实侧别 \(\sigma_{ij}=\operatorname{sign}(y_{ij}-C)\) 和净空 \(d_{ij}=|y_{ij}-C|\)。用同一个配对约束的平方违反量：

\[
\mathcal L_{a,b,m}=\frac{
\sum_{i\in\{a,b\}}\sum_{j\in\{x,y\}}
 [d_{ij}/2-\sigma_{ij}(\hat x_{ij}-C)]_+^2
 +[e_a+e_b-\gamma/2]_+^2}{W^2}.
\]

第一项保留同侧半个真实净空，错房时对垂直中线的坐标直接有纠正梯度；第二项限制两端误差吃掉的真实间隔。半量来自要保留的自然余量，宽 \(W\) 来自地图，不引入论文的温度或秩阈值。若损失为零，两预测都在各自**真实**房间，按房内 1-Lipschitz 有：

\[
G(\hat x_b)-G(\hat x_a)\ \ge\ q_b-q_a-e_b-e_a\ \ge\ \gamma/2.
\]

真坐标恰落在中线的对保留原 latent 目标并单列，不宣称正净空。此条件充分而不必要；有些排序正确的预测仍会受物理约束，训练未必能使项归零。

在原 fit 的同一次 forecast 输出上，仅取 H5 的两模式终点接固定读出。物理 \(q,\sigma,d\) 只用于离线监督，绝不进入预测、支持或部署打分。每根指定见证取均值，再按两类见证等权汇总，并报告缺见证的数量及原因。若之后获授权做同预算训练对照，两臂共用完全相同的四槽数据和初始化；采用无量纲配对项系数 1，先记录其与原目标的梯度量级，不扫系数。比较 H1–H5 的传播时不把 H5 的绕行优劣硬套到早期跨度。

对同一原始候选顺序，若两模式真排序反转且两对均满足条件，则预测 \(g_0=\widehat J_{b,0}-\widehat J_{a,0}\) 和 \(g_1=\widehat J_{b,1}-\widehat J_{a,1}\) 也反号；跨动作条件差 \(g_1-g_0\) 方向正确，幅度至少为真实差的一半。它不保证实际支持恰好越过 \(g_0+p(g_1-g_0)=0\) 的切换点，也不保证池内第三候选、后缀或长程收益。D-JEPA 的径向写秩只保证原生 latent MSE 的顺序；这里固定 \(R\) 和硬 \(G\) 无此恒等式，终点改写还会改变用于解释实际到达的同一 latent。因此保留原预测和事实读出，不移植关系重排或径向写秩。

## 已做检查及下一最小台阶

1. **算式检查已做，参数未更新。** 从项目原文件在内存调用现有地图打分。固定上述布局，合成真终点 \((103.5,127)\)、\((87,85)\)，预测 \((112.1,127)\)、\((142.1,140)\)。真实代价 57.4138、103.3286，预测硬代价 32.6038、0.1000。softplus 排序对真实较优计划的预测 \(x\) 导数为负，最小化会把它再推入错误房间；上式在同一点的导数为 +0.0027043，最小化会推回左侧。按真净空及真间隔给定扰动半径，固定种子的 1,024 对扰动都保持各自真房间，最小硬间隔 31.7671 大于保证值 22.9574。它验证梯度反例与计算界，不是模型性能。
2. **真实冻结计算图已检查，零优化步。** 原训练档案四槽 384 对中有 60 对严格换序，其中 46 对有一条全程静默动作；没有两条动作都静默的换序见证，也没有恰落中线例。固定四例中复现了直接排序的错房方向；物理间隔项能传到预测器及有曝光的条件分支，关闭处的 gate 梯度和全程静默候选的反对称 Jacobian 均精确为零。参数微扰与自动微分方向相同，未改任何权重。[完整结果](../../runs/native_topology_paired_20261115/writein_attribution_20261116/decision_gradient/REPORT.md)
3. **若以后授权训练，先做冻结同池验收。** 以相同来源、预算、根、候选和事实比较同序/换序硬间隔、虚报进目标房和 H1–H5 解码误差；换序同时报两模式各自间隔、\(\Delta\widehat B\) 和 \(\Delta\widehat D\) 的方向与幅度、支持切换点。用实际 post 对 omit / uniform 的首块遗憾判断写入增量；给真模式只作归因。回归合法保持、首次证据写入及静默期支持。配对损失、几何均值或真模式单项好转不能单独证明长程收益。

本次核验来源和训练形式，只给出可导的物理中间约束。成对排序、间隔界和同侧约束是训练工具，不作为 cJEPA 的原创机制，也不构成后续执行授权。
