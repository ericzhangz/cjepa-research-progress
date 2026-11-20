# PRBD 技术交接报告 · P0–P3 与 G0–G3

**记录截止 2026-11-20。** PRBD（Native Predictive Response Belief Dynamics）研究一个可以直接测量的问题。智能体在 TwoRoom 中先沿一个方向移动，看到自己实际走了多远，再回到起点。当两段历史的最近三帧画面和近期命令完全相同，早先的经历还能否让模型更准确地预测另一个方向的移动？本报告中“真实”指作者模拟器对已执行命令产生的画面或位置。“真实写入”指模型用已经到达的画面编码更新四个原型的概率。预测期间参数固定。

**一次写入的对象和用途。** 冻结的 LeWM 根据三帧画面编码和动作块预测下一个画面编码。原型 0 沿用宿主计算，另外三个原型分别在同一预测器的动作条件上施加学得的偏移。模型先对实际发出的命令给出四个后继预测。下一帧到达后，哪一条预测对该观察给出较高的似然，其相对支持就增加。比较未来的正交移动和停止时，两条候选共用这份支持，各自保留四条分支预测。慢速和快速分别对应模拟器参数 2.5 和 7.5；训练未指定哪个原型应当对应哪种速度。

**最新开发集读数。** 在 39 个同窗且确有物理速度差的起点上，预指定的 H1/H4 联合训练组只吸收首次探测时，H4 跨动作有向恢复率相对均匀支持为 **+0.05179%**。再吸收返回和两次静止观察后，同一对照变为 **−0.02843%**。两次静止写入都精确保留支持，变化发生在返回观察。若两边都吸收返回与静止观察，仅一边额外吸收首次探测，则该次探测的 H4 增量为 **+0.05263%**。这三个数衡量的是模型对真实动作差的预测修订，全都使用同一批起点与候选。长程规划与任务成功率尚无测量。

当前归档的阶段状态为 **P0–P3 已执行，G1 和 G2 通过，G3 两臂均完成训练和开发测量，预指定主臂未通过 G3**。P3 后的回合训练阶段名为 P3b，其中 G1 核查观测密度，G2 核查原生动作响应，G3 核查训练后的跨动作连续写入。P 与 G 两代实验使用不同的数据、观测法则和训练目标。本文分别给出各自的分母及对照。现存 PRBD 的 G 系列验收记录从 G1 开始编号；本文将 G1 前已有的数学、宿主和数据核验归入 **G0（前置记录）**，该栏没有独立的 G0 运行、配置或通过判定。

| 阶段 | 实际固定了什么 | 实际读什么量 |
|---|---|---|
| G0 前置记录 | 指定的四个解析响应；作者宿主、动作归一化和发布数据 | 等响应赔率、跨动作读出、零条件原生输出、可辨速度样本数 |
| P0 | 标量双原型均值和真实观察 | 原始与有界协方差的似然比、连续积分与有限节点的差 |
| P1 | 固定四原型、源观察和未来候选 | 只改证据模型后，两种速度的后验分离度和实际读出 |
| P2 | 固定未来候选的原型响应族 | 实际权重作用方向、可达投影、最近可达向量 |
| P3 | 同一批单轨、初值和 30 次更新 | 两种几何 × 两种训练目标；首次、连续及跳过首次的差 |
| G1 | 连续尺度模型、固定均匀中心、预定积分误差限 | 成对似然比、绝对密度、后验、梯度、真实读出符号 |
| G2 | 原生零动作差分、同窗、恒等输运 | 静止保持、带真速度的宿主容量、从实际经历取得的传递 |
| G3 | 回合内恒定速度、同一原型贯穿回合；预定 H1 对照和 H1/H4 主臂 | 另一方向的 H1–H4 修订、绝对端点误差、多次写入及逐起点贡献 |

**表中术语。** “根”是一次分支比较的模拟器起点。同窗要求在发起未来查询前，配对历史给模型的最近三帧和近期动作相同。“H1/H4”分别指向前预测 1/4 个模型动作块，每块含 5 条原始命令。“首次”只用探测到达，“完整”用探测、返回及两次静止到达，“省略首次”只用后三次到达。“均匀”对四个原型各给 1/4 支持。TV 是两组概率逐项绝对差之和的一半，log BF 是两原型对同一观察的自然对数似然差，单位 nat。两者描述证据与支持。动作方向用与真值的内积判定，交互 SSE 对动作差的误差取平方后求和，端点 MSE 对终点编码各坐标的平方误差取平均。

## 1. 从环境的一步到模型的一次签发

模型时间用零基块下标 $t$。每块包含五个二维原始命令，环境逐个执行 `env.step`。$z_t$ 是该块开始前的真实图像编码，$a_t$ 是整块命令的原生编码，第五个 primitive 后取得 $z_{t+1}$。遇到终止或截断立即停，仅已完成的块及相应观察可进入训练。给定 $z_{t-2:t}$ 和 $a_{t-2:t}$，predictor 的最后一个输出预测 $z_{t+1}$。从 $t=2$ 才有三个真实状态可用；含 $T$ 个完整块的合格回合有 $T-2$ 个后继目标。

| 对象 | 形状与精度 | 生产者 → 消费者 |
|---|---|---|
| RGB，原始命令 | `uint8[224,224,3]`，`float32[5,2]` | 环境 → 冻结编码器；位置、速度和碰撞保存在独立诊断档 |
| 一个真实前缀 | 状态及命令分别为 `[B,3,192]`，原生前向 FP32 | 编码档 → `NativePrototypes.forward` |
| $f_k$ 与 $w$ | 一次签发 `[K,192]`，$K=4$；支持 `[4]` | 四个原生后继 → 观测模型；概率运算 FP64 |
| $L,B,x$ | 共享噪声 Cholesky `[192,192]`、白化响应差 `[192,4]`、实际新息 `[192]` | `issue` 签发前两项相关几何；后继到达后 `assimilate` 或 `assimilate_continuous` 计算 $x$ |
| 候选路径 | `[batch,K,H,192]` | `rollout` 为各原型推进窗口；公开支持对同一候选的整族后果加权 |
| 同窗评价 | `[速度,候选,原型,H,192] = [2,2,4,4,192]` | 评价器读移动、停止两条路径；真实未来只供计算误差和事后容量 |

RGB 从通道末维变为 `NCHW`，除以 255，再按均值 $(.485,.456,.406)$、标准差 $(.229,.224,.225)$ 归一化；补齐最后一个图像批次使编码批量形状固定，随后只保留有效输出。五个真实动作按发布数据的逐轴均值和样本标准差归一化，展平为 10 维交给冻结的动作编码器。P3b 逐块固定 `[1,1,10]` 的调用形状，因此原始全零命令在不同批次得到逐值一致的 192 维编码。零命令编码本身不要求是零向量。

固定噪声来自另一批默认速度 5 的轨迹。32 个标定回合中 31 个提供 413 个残差 $z_{t+1}-F_0(W_t,a_t)$。`LedoitWolf.fit` 估计中心化协方差 $R$，收缩系数 0.05117024，$\operatorname{tr}R=8.500108$，保存 $L=\mathrm{chol}(R)$。P1–P3 的真实四原型测量及 G 阶段沿用该标定；P0 标量例另取单位噪声。对真实回合，拟合函数读取编码、命令、长度和记录编号。物理速度供环境生成；配对和位置供离线评分，另在 G2 单列一个读取真速度的参照。

### 1.1 原型在预测器内的位置

两代实现都冻结宿主。原型 0 直接取 `host.predict(...)[:, -1]`，原型 1–3 各有六个 192 维参数向量，共 3,456 个可学习偏移，初始为固定随机种子的 0.01 倍标准正态。每一层只改变最新 token 的动作条件；状态仍走原有 block、归一化和预测头。下面和后文的 Python 均为发布版的连续原文片段，包含原缩进，行号按本版源码。

源码 `prbd_native.py` 第 39–49 行。

```python
def native_conditioned(host, window, controls, offsets):
    """Apply each LeWM block's offset to the latest token's action condition."""
    predictor = host.predictor
    transformer = predictor.transformer
    state = transformer.input_proj(predictor.dropout(window + predictor.pos_embedding[:, :window.size(1)]))
    condition = transformer.cond_proj(controls)
    for block, offset in zip(transformer.layers, offsets, strict=True):
        altered = torch.cat((condition[:, :-1], condition[:, -1:] + offset), dim=1)
        state = block(state, altered)
    output = transformer.output_proj(transformer.norm(state))
    return host.pred_proj(output.flatten(0, 1)).unflatten(0, window.shape[:2])
```

P 阶段取非零原型的上述条件响应。G2/G3 另减去同窗口参考动作下的条件差，详见 §5。个别原型的下一枚 token 由窗口、动作与其参数决定；公共支持 $w$ 改变各条路径的混合读出，不直接作为 token 输入。

源码 `prbd_native.py` 第 133–146 行。

```python
    def rollout(self, window, controls):
        """Free-roll each prototype's own window; return [batch, K, H, D]."""
        count, length = self.offsets.size(0) + 1, window.size(1)
        horizon = controls.size(1) - length + 1
        if horizon < 1:
            raise ValueError("Controls must include the past L-1 and at least one future action")
        windows = window[:, None].expand(-1, count, -1, -1)
        forecast = []
        for step in range(horizon):
            command = controls[:, None, step:step + length].expand(-1, count, -1, -1)
            predicted = self._step(windows, command)
            forecast.append(predicted)
            windows = torch.cat((windows[:, :, 1:], predicted[:, :, None]), dim=2)
        return torch.stack(forecast, dim=2)
```

这里同一 $k$ 贯穿整条假想路径。实际收到后继时，各原型可从刷新后的公共真实窗口再签发；候选内部只回填该分支自己的 `predicted`。一次自由预测没有写公共状态。沿路径预测移动一步后再给零命令，先前已经分开的窗口仍可能生成不同后果。

### 1.2 实际观察怎样改变另一动作的预测

签发时先检查支持非负、概率和及转移矩阵各行是否合法，计算 $w^-=T^\top w$。本报告的两种速度在每一回合内恒定，$T=I$。从各响应的公共中心 $\bar f$ 构造 $b_k=L^{-1}(f_k-\bar f)$，令 $B=[b_0,\dots,b_3]$。P 阶段取先验加权中心，G 阶段取固定均匀中心。真实后继到达后才计算 $x=L^{-1}(z_{t+1}-\bar f)$ 和各分支密度 $\ell_k$。写入与未来加权读出分别为

$$w_k^+=\frac{w_k^-\ell_k}{\sum_j w_j^-\ell_j},\qquad
\hat Y_\beta=\sum_k w_k^+Y_{\beta,k}.$$

对相同的未来窗口和动作程序 $\beta$，只交换写入前后支持时，实际修订恰为 $\sum_k(w_k^+-w_k^-)Y_{\beta,k}=\mathrm{Cov}_{w^-}(\ell,Y_\beta)/\mathbb E_{w^-}\ell$。因此两原型等似然时其相对赔率保持；若它们对未来某动作给出相同后果，仅在这两者之间重分配支持对该后果无影响。这两个性质是概率运算及固定响应族的恒等关系，不指定学出的 $Y$ 与物理方向是否一致。

**评分口径。** 记 $d=(z_{\rm 慢,动}-z_{\rm 慢,停})-(z_{\rm 快,动}-z_{\rm 快,停})$，四项在相同跨度读取真实编码。以预测替换四项得 $\hat d$。从策略 A 换为 B 时，有向恢复率为 $100\sum_i(\hat d_{B,i}-\hat d_{A,i})^\top d_i/\sum_i\|d_i\|^2$，先按根求和再取比值。交互误差改善为 $\sum_i(\|\hat d_{A,i}-d_i\|^2-\|\hat d_{B,i}-d_i\|^2)$。报告交互 SSE 的相对改善时，上式乘以 100，再除以比较前策略 A 的总平方误差 $\sum_i\|\hat d_{A,i}-d_i\|^2$。有向恢复率和相对误差改善各有自己的分母。端点 MSE 对 192 个潜坐标取平均，配对汇总时再平均两种速度与两条候选。物理位移另用环境坐标报告。

## 2. G0（前置记录）与 P0 的数值构造

**解析响应。** 二维例中，四个原型对水平、垂直命令的位移增益依次为 $(1,1),(1,3),(2,1),(2,3)$，先验为 $(1/2,1/4,1/6,1/12)$。从原点发出 $(1,0)$，实际到达 $(2,0)$ 后支持为 $(0.451738,0.225869,0.214929,0.107464)$。两个等响应组内的赔率均保持 2∶1。从该到达点对同一斜推 $(1,1)$ 重算，未来横坐标从 3.250000 变为 3.322393，纵坐标保持 1.666667。固定同一观察及四个未来值，仅将未来原型的对应顺序换为 $[2,3,0,1]$，横向修订即反号。此例直接指定响应，不含原型学习。

**宿主记录。** 发布的 TwoRoom 首条轨迹、PushT 的 16 条发布轨迹都做过零条件 H1–H5 核验，与原预测的最大差为 0。TwoRoom 未训练原型写入后，下一动作编码平方误差和从 6.57610 变为 6.58742；PushT 的每坐标 MSE 平均增加约 $1.20\times10^{-8}$，0/16 条改善。PushObj 的独立 ViTPredictor 在八个 I 形起点验证零条件逐值等价和非零条件可激活。后面的速度训练及配对恢复率只来自 TwoRoom。发布的 TwoRoom 数据共 10,000 回合；5,304 个回合中可排除碰撞且命令足以反推速度的 42,875 个转移均与默认速度 5 吻合。速度 2.5、7.5 的数据通过作者环境原有的速度选项另行采集。

**P0 双原型。** 固定白化空间均值 $-v,+v$、均匀先验、单位观测噪声，观察恰达 $+v$ 或 $-v$。原法给定尺度的协方差为 $I+\xi BB^\top$，有界法改为 $I+\xi P$，其中 $P=B(I+B^\top B)^{-1}B^\top$。二者分支均值相同。有界几何的谱增益是 $\sigma^2/(1+\sigma^2)$；独立随机尺度 $\xi$ 的取值上界仍是无穷。

固定 $\xi$ 时，正观察支持正确分支的原版 log BF 为 $2v^2/(1+2\xi v^2)$，随分离增大趋于 $1/\xi$。对 $\xi\sim\mathrm{InvGamma}(3/2,3/2)$ 连续积分，原版后验极限为 $25/34=0.735294$，32 点积分的极限为 0.735274。原 32 点规则先取 Gauss–Legendre 节点，再经逆分布变换得到尺度和非负质量。测量如下。

| $v$ | 1 | 5 | 10 |
|---|---:|---:|---:|
| 原版 32 点，正确分支后验 | 0.653362 | 0.730577 | 0.734085 |
| 有界 32 点，正确分支后验 | 0.740945 | 0.998676 | 0.999919 |

在 $v=10{,}000$ 的有界例中，两种积分显示的后验均舍入到 1，但 32 点 log BF 为 1,978,730.464，连续参考为 37.046 nat。这里检验的是同一套分支均值、不同尺度积分下的尾部密度。P0 的解析极限以这一对称标量设置为条件。

## 3. P1–P3 的实轨、固定响应和训练变更

P 阶段拟合输入为独立的 32 个重置种子，慢、快速度各 16 个，采用发布档案中的真实动作序列。30 条有效轨迹共提供 418 个预测起点。开发集的 16 个种子各实跑两种速度和两个候选。15 个走完全程，11 个在查询时满足三帧与近期动作相同，其中 9 个在未来移动上产生非零速度差，另 2 个因边界约束为零。P 阶段主表按 **11 个同窗根** 汇总，方向正负计数另以非零根为分母。在同窗程序中，首次探测造成的位移分别为 3.125 和 9.375；查询窗口刷新后比较正交移动与停止。

P 阶段的默认调用是 `NativePrototypes(null_control=None)`、`issue(reference="prior")` 和 `assimilate(..., geometry=...)`。原版 Q32 令 $G=B^\top B$、$r_k=x-b_k$，每个尺度解四阶系统 $(I+\xi G)$。高斯对数密度所需的两项是 $\log\det(I+\xi G)$ 和 $\|r_k\|^2-\xi (B^\top r_k)^\top(I+\xi G)^{-1}(B^\top r_k)$。沿 32 个节点按质量做 logsumexp 得到每个原型的边缘密度；公共项保留在绝对 log density 中，计算相对支持时可消去。在有界对照中均值、白化矩阵和混合步骤相同，条件协方差换为 $I+\xi P$。G 阶段的均匀中心、薄 SVD 与零动作差分分别在 §4、§5 说明。

### 3.1 P1 和 P2 共用同一份预测缓存

**P1** 冻结原型、噪声标定和全部源/未来预测，只改变源观察给原型的似然。旧法用响应矩阵放缩噪声，有界法取 §2 的 $P$，共享高斯对照直接用同一个 $R$ 为各中心评分。每种方法先输出两份后验，再让相同的未来响应消费它们。

| 11 根的实测量 | 原版 | 有界 | 共享高斯 |
|---|---:|---:|---:|
| 慢速与快速的源后验 TV 平均值 | 0.107774 | 0.514395 | 0.999962 |
| 在源动作读出的速度差恢复率 | +0.4609% | +7.0151% | +9.3046% |
| 同一后验在未来正交动作上的 H1 恢复率 | −0.0824% | −0.3452% | −0.7212% |
| 同一后验在未来正交动作上的 H4 恢复率 | −0.3281% | −0.5403% | −2.1002% |

**P2** 保留上述未来族。对同窗根，令 $Y_k=\Phi_k(\mathrm{动})-\Phi_k(\mathrm{停})$，则 $\hat d=(w_{\rm 慢}-w_{\rm 快})^\top Y$。两种速度的未来原型族相同，旧似然下有 8/9 个非零根的 H4 修订反向。归档另做两个允许使用未来真值的离线计算。

| 离线问题 | 权重约束和读数 |
|---|---|
| 固定 $Y$ 能得到的最大投影 | 慢速权重集中于 $Y_k^\top d$ 最大的分支，快速权重集中于最小的分支。所得 H1/H4 有向投影上限为 6.22%/20.61%。 |
| 离真实交互向量最近的可达向量 | 对两份非负、和为一的权重最小化 $\|(w_s-w_f)^\top Y-d\|^2$，移动、停止共用该速度的权重。H4 相对零预测的最优 SSE 改善为 26.53%，数值最优性界宽低于 $5\times10^{-8}$ 个百分点。强制同一对权重跨 H1–H4 共用，四跨度合计最多改善 19.81%。 |

例：种子 2026111974 在 H4 的四个 $Y_k^\top d$ 为 $(-99.358,-112.148,-103.702,-60.708)$。最大投影的慢/快索引为 3/1；共享高斯后验近乎选中 1/2，实际有向分子为 −8.446。固定同一真值和四个 $Y_k$，旧似然与有界似然的分子分别为 −2.292 和 +0.222。P2 的两个事后最优量分别度量有向投影和全向量平方误差，均未用来写部署权重。

对于查询窗口不同的两条轨迹，P2 还把交互向量精确分成两项。令两种速度的族为 $Y_s,Y_f$、支持为 $w_s,w_f$，则支持项为 $(w_s-w_f)^\top(Y_s+Y_f)/2$，窗口项为 $(w_s+w_f)^\top(Y_s-Y_f)/2$。同窗时第二项为零。以下是保存逐根结果之前的恒等式检查。

源码 `prbd_evidence_attribution.py` 第 192–205 行。

```python
        midpoint = (families[0] + families[1]) / 2
        mid_scores = midpoint @ truth[hi]
        range_dot = float(mid_scores.max() - mid_scores.min())
        for mi, method in enumerate(METHODS):
            weight = supports[mi]
            # Both independently executed programs must use the same retained support.
            prediction = (weight[0, 0] @ branches[0, 0] - weight[0, 1] @ branches[0, 1]
                          - weight[1, 0] @ branches[1, 0] + weight[1, 1] @ branches[1, 1])
            delta = weight[0, 0] - weight[1, 0]
            support_part = delta @ midpoint
            window_part = ((weight[0, 0] + weight[1, 0]) / 2) @ (families[0] - families[1])
            error = float((prediction - support_part - window_part).abs().max())
            if error > 1e-11:
                raise ValueError("Support/input midpoint identity failed")
```

### 3.2 P3 在哪里混合原型，哪些观察参与梯度

P3 的 `trajectory_paths` 为每个实际起点 $t$ 取 $H_t=\min(4,T-t)$。以每个原型自己的窗口沿随后实际执行过的命令自由递推。单个根内，条件于原型，各跨度观察密度相乘；整段只混合一次。下一实际根的初始支持只取上一个根的**第一个**真实到达。全部根的损失之和除以目标步数。公式为

$$\mathcal L_{\rm P3}=-\frac{\sum_t \log\sum_k w_{t,k}\exp(\sum_{h=1}^{H_t}\log\ell_{t,h,k})}{\sum_t H_t}\tag{D}$$

目标区间相互重叠。418 个根产生 1,492 个评分目标步，没有增加独立轨迹。以下两段分别是根内混合与根间传递；`posterior` 赋值保留计算图，未来目标可以通过损失训练过去的取证响应，但部署写入只读取已经到达的观察。

源码 `prbd_public_fit.py` 第 50–57 行。

```python
    identity = torch.eye(paths.size(0), dtype=torch.float64, device=paths.device)
    signed = [issue(responses, weights, identity, noise)
              for responses in paths.unbind(dim=1)]
    arrivals = [assimilate(issued, target, nodes, masses, geometry=geometry)
                for issued, target in zip(signed, observations, strict=True)]
    branch_joint = torch.stack([arrival.log_likelihood for arrival in arrivals]).sum(0)
    joint = torch.logsumexp(_log_mass(signed[0].prior) + branch_joint, dim=0)
    return joint, arrivals[0]
```

源码 `prbd_public_fit.py` 第 98–108 行。

```python
    paths = trajectory_paths(model, z, a, horizon)
    posterior = transition.new_full((len(transition),), 1 / len(transition))
    losses = []
    for root, path in enumerate(paths):
        actual = z[root + 3:root + 3 + path.size(1)]
        log_joint, first_arrival = trajectory_log_predictive(
            path, posterior, actual, noise, nodes, masses, geometry=geometry)
        losses.append(-log_joint)
        posterior = first_arrival.posterior
    targets = sum(path.size(1) for path in paths)
    loss = torch.stack(losses).sum() / targets
```

四组共享相同冻结宿主、初始条件、$R$、Q32 积分及 Adam 学习率 0.001。每个有效回合更新一次，共 30 次。一步拟合用时约 2–3 秒，式 (D) 约 12–13 秒。`original_h1` 的拟合偏移逐值等于先前存档，NLL 最大绝对差为 0。

| 训练配置 | 源恢复 | 首次写入 H1 | 首次写入 H4 | 完整写入 H4 | 完整减省略首次 H4 |
|---|---:|---:|---:|---:|---:|
| 原版噪声、一步 | +0.4609% | −0.0824% | −0.3281% | +0.0651% | −0.1974% |
| 有界噪声、一步 | +7.2580% | −0.2502% | −0.1819% | −2.2695% | −0.8337% |
| 原版噪声、式 (D) | +0.4574% | −0.0299% | −0.0727% | −0.1270% | +0.1060% |
| 有界噪声、式 (D) | +7.5539% | +0.6337% | +3.1315% | −5.3210% | −1.5839% |

四组 H4 首次写入的共同分母为 1084.874257 latent²，分子依次为 −3.559884、−1.973335、−0.788770、+33.972565。最后一组有 3 个根方向正、6 个负、2 个物理零；逐根留一的聚合范围为 −0.3747% 到 +4.5202%。该组的 H4 移动端点 MSE 从均匀的 0.23604 变为首次写入的 0.24547，停止端点则从 0.09471 变为 0.10510。

P3 结束后保留了两个输入不变的复算。① 在同一检查点的 22 次源观察上，Q32 与连续参考的最大权重索引全部相同，后验 TV 中位数 0.20481、最大 0.69880。仅替换首次后验、固定同一未来，H4 有向恢复从 +3.13147% 变为 +0.18677%。② 在 11 根 × 2 种速度 × 2 个静止动作上，实际位置和编码不变，44 次到达却都改变支持，TV 中位数 0.18091，范围 0.02063–0.67693。前述 418 个已评分训练块中，完整五命令全零的块数为 0。

## 4. G1 将训练、写入和评价统一到同一密度

G1 实际使用 P0 的有界几何、自由度 3 的连续尺度模型，以及固定均匀响应中心。这样对相同 $(W,a,k)$，分支的条件观察分布独立于当前支持。主入口必须显式调用 `issue(..., reference="uniform")` 与 `assimilate_continuous`；同一代码文件保留 P 阶段默认的先验中心和 Q32 路径。

源码 `prbd.py` 第 150–158 行。

```python
    differences = responses - responses[0]
    if reference not in ("prior", "uniform"):
        raise ValueError("reference must be 'prior' or 'uniform'")
    mean_difference = (prior @ differences if reference == "prior"
                       else differences.mean(0))
    center = responses[0] + mean_difference
    basis = torch.linalg.solve_triangular(
        noise_cholesky, (differences - mean_difference).T, upper=False
    )
```

给定尺度，$x\mid k,\xi\sim\mathcal N(b_k,I+\xi P)$。白化回原空间时，每个绝对 log density 再减去 $\sum_i\log L_{ii}$。训练消费绝对密度；求后验时使用相对分支 0 的 log 比值避免减去两个含巨大公共残差的数。零先验分支的对数质量为负无穷。

实现取 $\log\xi\in[-6,40]$，间距 0.25，185 个节点。节点质量含逆 Gamma 密度、变量替换和梯形端点权重，再归一化。185 节点给出连续逆 Gamma 混合的固定求积近似。该有限混合在观测空间有归一化密度，对连续法则的似然比与梯度误差只在表列范围内核验。对 $B$ 做薄 SVD，奇异值 $\sigma$ 映射为 $\lambda=\sigma^2/(1+\sigma^2)$，各节点在对应方向的方差为 $1+\xi\lambda$。响应张成空间之外，所有原型共用同一高斯残差。实现显式计算解析一阶 Gaussian score，再按节点责任加权回传，绕开重复奇异值处的奇异向量导数；该核只提供一阶导。

源码 `prbd.py` 第 181–192 行。

```python
        vectors, singular, right = torch.linalg.svd(basis, full_matrices=False)
        raw = singular.square()
        eigenvalues = raw / (1 + raw)
        variances = 1 + scales[:, None] * eigenvalues[None]
        coordinates = vectors.T @ (x[:, None] - basis)
        orthogonal = x - vectors @ (vectors.T @ x)
        node_logs = -.5 * (variances.log().sum(-1)[:, None]
                           + (coordinates[None].square() / variances[:, :, None]).sum(1))
        log_joint = _log_mass(masses)[:, None] + node_logs
        reduced = torch.logsumexp(log_joint, dim=0)
        responsibilities = (log_joint - reduced[None]).exp()
        absolute = reduced - .5 * (d * math.log(2 * math.pi) + orthogonal.square().sum())
```

源码 `prbd.py` 第 270–282 行。

```python
    x = torch.linalg.solve_triangular(
        issued.noise_cholesky, (observation - issued.center)[:, None], upper=False)[:, 0]
    absolute, relative = _ContinuousBoundedDensity.apply(issued.basis, x, scales, masses)
    log_likelihood = absolute + issued.log_jacobian
    log_joint = _log_mass(issued.prior) + relative
    posterior = log_joint.softmax(0)
    if torch.equal(relative, relative[0].expand_as(relative)):
        posterior = issued.prior.detach() + (posterior - posterior.detach())
    common = log_likelihood[0]
    return PRBDResult(
        posterior=posterior, log_likelihood=log_likelihood,
        log_predictive=torch.logsumexp(_log_mass(issued.prior) + log_likelihood, 0),
        relative_log_likelihood=relative, common_log_density=common)
```

在 `reference="uniform"` 和单位转移下，签发时也逐值保留输入公共支持。上段代码的 `detach` 组合使等似然时前向值精确等于 prior，同时保留普通贝叶斯计算的一阶梯度。新旧密度用了相同的 `log_likelihood` 字段表示各分支绝对 log 密度，但新核的 `common_log_density` 具体取分支 0 的绝对 log 密度。

| 校验范围 | 对照和观测量 |
|---|---|
| 预设容差 | 成对 log BF 与绝对分支 log density 误差各 ≤ $10^{-4}$ nat；后验 TV ≤ $10^{-5}$；梯度 L2 误差 ≤ $10^{-4}(1+\|g_{\rm ref}\|)$；逐根和聚合方向与自适应积分一致 |
| 22 次不同源到达 × 2 个旧检查点 | 最大成对 log BF 误差 $8.14\times10^{-13}$ nat，最大后验 TV $1.63\times10^{-13}$，归一化原生参数梯度误差最大 $4.20\times10^{-8}$ |
| 新初值和两组 G3 终值 | 扫描 14,232 个一步或自由多步的密度因子，按预定规则抽取 107 个因子（54 个非均匀前缀），其中 46 个核对完整原生参数梯度。最大 log BF 误差 $5.76\times10^{-13}$ nat，后验 TV $1.04\times10^{-13}$，归一化原生梯度误差 $5.69\times10^{-7}$ |
| 两个终值的实际时间线 | 对 42 个匹配根的首次和返回观察，用独立自适应积分重写支持、读取完全相同的未来族。三组对照在 H1/H4 的每个根及聚合恢复符号均保持 |

## 5. G2 在原生预测中约束零命令的条件效应

TwoRoom 的速度参数乘在动作位移上。同一实际窗口下，若五个原始命令全为零，不同速度不会使物理位置改变。G2 保留实际三帧和前两块动作，只将最新动作替换为经原生编码的 $a_\varnothing$。记 $D_u(W,a)=G(W,a;u)-G(W,a;0)$，实际签发为

$$F_u(W,a)=F_0(W,a)+D_u(W,a)-D_u(W,a_\varnothing).$$

原型 0 直接走冻结预测；对其他三个原型，保留各自当前窗口，在相同调用批形下计算基础预测，再加上两个动作条件差的差。`conditioned` 在下段入口处是 $G(W,a;u)$。$a=a_\varnothing$ 时校正逐值为零。若不同原型在更早的自由步骤已有不同窗口，随后输入零命令仍可能得到不同的 $F_0$；下面的严格等价针对**同窗**。本设置每个模型步为 8 次 predictor 批调用，按原型计 16 次分支计算。

源码 `prbd_native.py` 第 105–125 行。

```python
        if self.null_control is not None:
            # The reference changes only the latest command, retaining each
            # branch's own state history and every historical action.
            reference_controls = torch.cat((other_controls[:, :-1],
                self.null_control.expand(len(other_controls), 1, -1)), dim=1)
            zero_offsets = torch.zeros_like(offsets)
            action_zero = native_conditioned(self.host, other_windows, other_controls,
                                             zero_offsets)[:, -1]
            reference_response = native_conditioned(self.host, other_windows, reference_controls,
                                                    offsets)[:, -1]
            reference_zero = native_conditioned(self.host, other_windows, reference_controls,
                                                zero_offsets)[:, -1]
            delta_action = conditioned - action_zero.reshape(batch, count - 1, dimension)
            delta_reference = (reference_response - reference_zero).reshape(batch, count - 1, dimension)
            # Match the nominal call's [B,L,D] shape for every frozen branch.
            # Combining corrections before adding F0 also avoids FP32 drift
            # when the latest action is exactly its raw-zero reference.
            base = torch.stack([self.host.predict(windows[:, k], controls[:, k])[:, -1]
                                for k in range(1, count)], dim=1)
            conditioned = base + (delta_action - delta_reference)
        return torch.cat((nominal[:, None], conditioned), dim=1)
```

**旧面板实测。** 同一批 44 次不同的实际静止到达在初值和旧 `bounded_d` 两个检查点上分别核验，共 88 个检查点–到达组合。各分支均值和似然逐值相同，非均匀支持亦逐值保持。另在 9 个无碰撞、缩放动作未饱和的起点，在归一化前将原始命令乘以真速度/5，再交给冻结 LeWM。此处读取真速度仅为离线宿主容量参照。固定原型的“有利重分配”读取真实未来，单独列出。

| 读出方式 | H1 有向恢复 | H4 有向恢复 | H1/H4 方向为正 |
|---|---:|---:|---:|
| 冻结宿主，真速度缩放原始命令 | 97.6700% | 95.0690% | 9/9，9/9 |
| 初值原型，给定有利支持 | 0.30843% | 0.19180% | — |
| 初值原型，只凭真实源观察 | +0.00214% | +0.00134% | 9/9，9/9 |
| 旧训练原型，给定有利支持 | 7.48261% | 3.49129% | — |
| 旧训练原型，只凭真实源观察 | +0.53402% | +0.17903% | 2/9，4/9 |


## 6. G3 一条真实回合怎样进入损失

**学习数据。** 新训练种子 2026112000–2026112063，各重置只在一个速度下执行一条预定命令序列。令 $p,q$ 是沿两个正交方向的五步动作块，$-\mathrm{rev}(p)$ 同时反向和倒序。以原始程序数组保留训练和开发的差异。

源码 `prbd_consistency_data.py` 第 17–34 行。

```python
def program(split, index, candidate=0):
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
    along = directions[index % 4]
    across = np.array([-along[1], along[0]], dtype=np.float32)
    if ((index // 4) % 2 == 1) != (split == "confirm"):
        across = -across
    amplitude = [.1875, .25, .3125, .375][index // 8 % 4]
    waveform = [.5, 1, .5, 1.5, 1.5] if split == "confirm" else [.5, 1, 1.5, 1, .5]
    gains = np.asarray(waveform, dtype=np.float32)[:, None] * amplitude
    p, q = gains * along, gains * across
    zero = np.zeros_like(p)
    if split == "train":
        blocks = [zero, zero, p, q, zero, -q[::-1], -p[::-1], zero,
                  q, p, zero, -p[::-1], -q[::-1], zero, p, q]
    else:
        blocks = [zero, zero, p, -p[::-1], zero, zero,
                  q if candidate == 0 else zero, zero, zero, zero]
    return np.stack(blocks)
```

64 条训练轨迹共执行 4,962 个 primitive，完成 992 块，其中 372 块为零。种子 2026112009 和 2026112030 在第一块完成前终止，其余 62 条各完成 16 块，各有 14 个可评分后继，共 868 个。物理速度在训练重置时平衡分配。拟合档仅传入实际 $z,a$、种子、有效长度、统一的零命令编码。

**两个训练臂的共同部分。** 每条轨迹从 $z_0,z_1,z_2$ 起算。分区 $S_1$ 将 $z_3,\ldots,z_T$ 分成单步；$S_4$ 把同一集合切成连续不重叠的最多四步。每段开始时刷新为该时刻的真实三帧，段内所有预测先从各原型自己的路径产生。设 $\ell^{(S)}_{tk}$ 为对应预测对真实后继的密度，则一个分区的回合评分为

$$S(z,a)=\log\sum_k \tfrac14\exp\!\left(\sum_{t=2}^{T-1}\log\ell^{(S)}_{tk}\right).$$

实现只在整回合结束后混合一次。在 16 块回合里，H4 分区的目标长度是 $4,4,4,2$；新段起点换实际窗，不更换 $k$。当条件固定且分支密度独立于先验时，H1 回合 log 边缘似然和按因果时间依次进行贝叶斯预测的 log 密度之和相等。G3 训练直接计算前者，训练目标内公共状态写入次数为零。

源码 `prbd_public_fit.py` 第 158–177 行。

```python
    uniform = transition.new_full((count,), 1 / count)
    factors, coverage, block_lengths, roots = [], [], [], []
    for root, path in partition_paths(model, z, a, horizon):
        if path.size(0) != count:
            raise ValueError("Native path and transition prototype counts disagree")
        roots.append(root)
        block_lengths.append(path.size(1))
        for step, response in enumerate(path.unbind(1), start=1):
            target = root + step
            signed = issue(response, uniform, transition, noise, reference="uniform")
            arrival = assimilate_continuous(signed, z[target], nodes, masses)
            factors.append(arrival.log_likelihood)
            coverage.append(target)
    if coverage != list(range(3, len(z))):
        raise RuntimeError("A partition must score each usable actual target exactly once")
    branch_joint = torch.stack(factors).sum(0)
    joint = torch.logsumexp(_log_mass(uniform) + branch_joint, 0)
    return joint, {"roots": roots, "block_lengths": block_lengths,
                   "target_indices": coverage, "scored_target_steps": len(coverage),
                   "prototype_episode_log_likelihood": branch_joint.detach().tolist()}
```

**对照与主臂。** `episode_h1` 用 $-S_1/(T-2)$，预指定主臂 `episode_dual` 用 $-(S_1+S_4)/(2(T-2))$。每个分区内各目标只出现一次；两个分区分别定义归一化密度，二者对数的等权平均作为复合训练准则。P3 中允许不同根有不同支持的重叠后缀目标没有在此处调用。

源码 `prbd_public_fit.py` 第 194–204 行。

```python
    partitions, scores = {}, []
    for horizon in ((1, 4) if dual else (1,)):
        log_joint, details = partition_episode_log_predictive(
            model, z, a, noise, transition, nodes, masses, horizon=horizon)
        partitions[f"H{horizon}"] = details
        scores.append(log_joint / details["scored_target_steps"])
    loss = -torch.stack(scores).mean()
    return loss, {"transitions": len(a) - 2, "scored_target_steps": len(a) - 2,
                  "density_factor_evaluations": sum(
                      part["scored_target_steps"] for part in partitions.values()),
                  "public_belief_writes_in_objective": 0, "partitions": partitions}
```

两臂从相同偏移出发，按同一回合顺序做相同的 62 次 Adam 更新，每次对该回合反向传播，只更新 `model.offsets`；学习率固定 0.001。H1 共计 868 个条件密度因子，dual 共计 1,736 个。两线程 CPU 上拟合分别用时 19.03 和 71.82 秒。下表给出第 1、31、62 次的原记录。NLL 是当前回合的每目标负对数**密度**，可以为负；各行对应不同回合。

| 更新序号、重置种子 | H1 NLL | dual NLL | dual 三个可学习原型的梯度 L2 |
|---|---:|---:|---|
| 1 · 2026112000 | −98.22996 | −87.58916 | 60.069 / 46.658 / 86.681 |
| 31 · 2026112032 | −71.38047 | −42.84503 | 154.672 / 42.303 / 130.116 |
| 62 · 2026112063 | −35.95146 | −34.20852 | 77.796 / 21.748 / 97.136 |

拟合后相对初值的三个原型偏移 L2，H1 为 0.24696 / 0.22102 / 0.31879，dual 为 0.28081 / 0.25481 / 0.30410。结构校验中 H1 回合式与顺序式的损失绝对差 $1.11\times10^{-16}$，梯度最大绝对差 $1.39\times10^{-16}$；H4 与独立标量展开的评分差为 0，dual 有限差分梯度最大误差 $3.69\times10^{-12}$。训练后两臂各自验证了原型 0 与 $F_0$ 相等、从共同根发出的四步零命令各原型路径相同、非均匀支持保持。

### 6.1 开发时间线、支持对照与取样单位

48 个新种子 2026112100–2026112147，各实际执行 2 种速度 × 2 条后续，共 192 条轨迹、9,450 个 primitive。配对条件包括完整终点、真实 RGB 和潜表示窗口相同、近期动作相同、跨速度同程序命令相同、候选间真实前缀重放一致、物理根一致，以及静止命令的编码逐值等于参考。42 个根满足条件，其中 39 个有非零物理交互、3 个为零；未匹配的 6 根保留在归档。主表与验收按 39 根计算。物理坐标用于配对资格检查、物理非零分层和评分。拟合、原生前向与支持更新的形参均不含坐标。

| 零基动作块 | 已知输入及实际到达 | 对应计算 |
|---|---|---|
| $t=0,1$ | 两块零命令，到达 $z_1,z_2$ | 建立三帧实际窗 |
| $t=2$ | 发出轴向探测 $p$，到达 $z_3$ | 四原型先预测，之后更新支持；记录首次写入 |
| $t=3$ | 发出 $-\mathrm{rev}(p)$，到达 $z_4$ | 用真实返回更新支持，回合速度维持 |
| $t=4,5$ | 两块零命令，到达 $z_5,z_6$ | 逐数核对四个均值、似然及公共支持 |
| $t=6$ | 同一 $z_4,z_5,z_6$，比较正交移动 $q$ 和停止 | 从同窗签发；随后 $t=7,8,9$ 均为零命令 |
| H1–H4 | 实际目标 $z_7,\ldots,z_{10}$ | 只参与评分；候选的 imagined token 从未反写公共支持 |

五种支持定义分别为 `nominal`（只取索引 0）、`uniform`（各 1/4）、`single_source`（只保留 $z_3$ 更新）、`all_arrivals`（使用 $z_3,\ldots,z_6$）、`omit_source`（只跳过 $z_3$，同样吸收 $z_4,\ldots,z_6$）。`forecast_prefix` 的状态形参截止到 $z_6$。循环变量 `speed` 选取一条事实轨迹；它的数值不作为模型特征。以下分别给出时间更新与未来读出。

源码 `prbd_consistency_measure.py` 第 69–81 行。

```python
        z, a = z_prefix[speed,0], actions[speed,0]
        state = {"all_arrivals":uniform.clone(),"omit_source":uniform.clone()}
        speed_trace = []
        for t in range(2,6):
            means = model(z[None,t-2:t+1],a[None,t-2:t+1])[0]
            event = {"action_t":t,"arrival_index":t+1,"weights":{}}
            for chronology in ("all_arrivals","omit_source"):
                before = state[chronology]
                signed = issue(means,before,transition,noise,reference="uniform")
                arrival = assimilate_continuous(signed,z[t+1],nodes,masses)
                skipped = chronology=="omit_source" and t==2
                after = before if skipped else arrival.posterior
                state[chronology] = after
```

源码 `prbd_consistency_measure.py` 第 98–113 行。

```python
            if t==2:
                supports["single_source"][speed] = state["all_arrivals"]
            speed_trace.append(event)
        for name in ("all_arrivals","omit_source"):
            supports[name][speed] = state[name]
        supports["nominal"][speed],supports["uniform"][speed] = nominal,uniform
        paths = [model.rollout(z_prefix[speed,p,4:7][None],actions[speed,p,4:10][None])[0]
                 for p in (0,1)]
        forecasts.append(torch.stack(paths))
        traces.append(speed_trace)
    paths = torch.stack(forecasts).double()
    common = paths[:,:,0]
    predictions = {name:common+torch.einsum("sk,spkhd->sphd",weight,paths-common[:,:,None])
                   for name,weight in supports.items()}
    return {"supports":supports,"future_responses":paths,"predictions":predictions,
            "actual_write_traces":traces,"no_op_checks":noops}
```

`paths` 的维度依次为速度、后续候选、原型、跨度和潜坐标。用名义路径作公共项来求加权和，避免各分支全相同时仅因浮点求和产生预测差。`score_case` 在预测结束后另读真实未来 $z_7,\ldots,z_{10}$，计算本报告 §1.2 定义的 $d$、$\hat d$、交互 SSE 和绝对端点 MSE。

### 6.2 阶段结果和原定验收

下表是同一 39 根在各跨度的聚合有向恢复率。每列均为百分比，正负号按慢速减快速、移动减停止的同一口径。

| 训练臂和支持之差 | H1 | H2 | H3 | H4 |
|---|---:|---:|---:|---:|
| dual 首次 − 均匀 | +0.05740% | +0.05002% | +0.05070% | +0.05179% |
| dual 完整 − 均匀 | −0.06500% | −0.03066% | −0.02917% | −0.02843% |
| dual 完整 − 省略首次 | +0.05817% | +0.05096% | +0.05160% | +0.05263% |
| H1 首次 − 均匀 | +0.03883% | +0.04139% | +0.04162% | +0.04189% |
| H1 完整 − 均匀 | +0.07126% | +0.08711% | +0.08754% | +0.08826% |
| H1 完整 − 省略首次 | +0.03772% | +0.03975% | +0.03995% | +0.04020% |

| dual 的两端跨度 | H1 / H4 交互 SSE 相对改善 | H1 / H4 逐根交互误差改善数 | H1、H4 有向恢复率的 95% 区间（%） |
|---|---|---|---|
| 首次 − 均匀 | +0.11464% / +0.10353% | 36/39，36/39 | [0.0334, 0.0839]；[0.0310, 0.0784] |
| 完整 − 均匀 | −0.13641% / −0.05886% | 35/39，34/39 | [−0.4128, 0.1145]；[−0.3038, 0.1198] |
| 完整 − 省略首次 | +0.1165% / +0.1052% | 36/39，36/39 | [0.0355, 0.0818]；[0.0329, 0.0776] |

bootstrap 以重置种子为配对单位，有放回重采样 4,000 次，随机种子 20261120，每次重算聚合分子与分母。就绝对移动端点 MSE 而言，主臂的均匀 / 首次 / 完整 / 省略在 H1 分别为 0.147405 / 0.147314 / 0.147457 / 0.147549，在 H4 为 0.180243 / 0.180164 / 0.180278 / 0.180358。各支持下的停止端点 MSE 完全相同，H1 为 0.057087，H4 为 0.095141。42 个同窗根、两种速度、两次静止共 168 个不同的实际静止到达；沿完整与省略两种计算时间线进行 336 次检查，公共支持最大变化为 0。

预定判据要求在 H1、H4 上，首次相对均匀、完整相对均匀均同时具有正的聚合有向修订、正的聚合交互误差改善、过半起点改善。置信区间及端点 MSE 独立报告。主臂完整写入的两个聚合量在两个跨度均小于零，因此 `episode_dual.gate.passed = false`。`episode_h1.gate.passed = true` 对应预定对照。G4 独立新历史与普通递归基线、P4 回合内速度反转及规划成功率均无本轮实测数据。

## 7. 一次真实写入的完整账目

开发种子 2026112128 是归档逐根分析中对 dual 完整写入交互 SSE 贡献绝对值最大的根。该根保留在上表。五个原始命令的探测使慢/快轨迹分别向左 4.21875 / 12.65625 个地图单位；反向块分别向右移动同一距离，精确回到 $(99.2294693,86.9803238)$。两种后续程序的前缀重放及两种速度下共 40 个相关 primitive 的碰撞标记均为假，各 primitive 的实际位移与无碰撞计算最大差为 0。随后的正交移动分别向下 4.21875 / 12.65625。

| 实际速度 | $z_3$ 后的支持，按 $k=0,1,2,3$ | $z_4$ 的分支 log BF，相对原型 0（nat） | $z_4$ 后的支持，$z_5,z_6$ 均保持 |
|---|---|---|---|
| 2.5 | [0.25364, 0.24919, 0.24963, 0.24754] | [0, −0.870923, +2.141343, −1.646090] | [0.10025, 0.04122, 0.83967, 0.01886] |
| 7.5 | [0.22429, 0.26099, 0.26830, 0.24643] | [0, −0.076097, −0.032097, −0.238433] | [0.24375, 0.26286, 0.28238, 0.21100] |

对两行分别用 `softmax(log(旧支持)+logBF)` 即可复得右列（表值已舍入）。取 $q=w_{\rm 慢}-w_{\rm 快}$，$c_k=(Y_k-\bar Y)^\top d/\|d\|$。由于 $\sum q_k=0$，本根的有向分子为 $N=\|d\|q^\top c$。H4 的 $\|d\|=14.223283$，$c=(0.054680,-0.037826,-0.253999,0.237144)$。实际返回观察之后，两种速度的支持 TV 从 0.030472 增至 0.557288，$q$ 与 $c$ 的余弦从 +0.543 变为 −0.816。该根 H4 的 $N$ 从首次写入的 +0.100394 变为完整写入的 −2.653770 latent²。这里的因果时间线与同一未来族均已固定。

| dual 主臂的有向分子（latent²） | 首次写入 | 返回写入带来的增量 | 两次静止增量 | 完整写入 | 完整减省略首次 |
|---|---:|---:|---:|---:|---:|
| 全部 39 根，H1 | +1.299218 | −2.770472 | 0 | −1.471254 | +1.316720 |
| 全部 39 根，H4 | +1.172267 | −1.815801 | 0 | −0.643535 | +1.191189 |

两行各自的真值能量分母为 2263.384856 latent²，因此重现 §6.2 中的百分比。H4 其余 38 根的完整分子之和为 +2.010235。逐一留出一个根时，只有留出 2026112128 会把主臂两个跨度的聚合完整恢复翻为正值。这里记录的是支持、固定未来族与最终误差之间可复算的关系。原型编号没有物理速度标签；测量没有把优化过程归因为单一物理原因。

## 8. 哪些能力已经测到，哪些尚无结果

| 读者关心的问题 | 已有测量覆盖的范围 |
|---|---|
| 当前动作不能区别原型时，公共支持能保持多久？ | 概率公式保证等似然原型间的相对赔率保持。G2/G3 对同窗零命令构造了四原型完全相同的响应。G3 开发集有 168 次不同的实际静止到达，完整与省略两条时间线共检查 336 次，最大支持变化为 0。这一测量覆盖返回后的两块零命令。 |
| 一次有信息的真实观察能否改变另一个动作的预测？ | G3 预指定主臂在 39 个物理非零的同窗起点上，首次写入相对均匀支持的聚合有向恢复在 H1、H4 均为正。对同一未来，仅改变支持即可重算该修订。§7 给出一个起点的概率、未来分支投影及二者的内积。 |
| 连续吸收观察的净结果是什么？ | 同一主臂完整写入相对均匀的有向恢复与交互 SSE 改善在 H1、H4 均为负。H4 有 34/39 个起点的交互 SSE 改善，但聚合 SSE 改善为 −0.05886%。前者逐起点计数，后者合计各起点的误差量后再求比。H1 单步训练对照按同一规则通过；主臂保留其原定身份与未通过记录。 |
| 这些数据能回答长程任务成功率吗？ | 训练和开发中的速度各自在回合内固定。开发协议实际执行预先给定的探测、返回、静止、正交移动或停止，设定了两种未来候选。G4 独立历史确认、普通递归对照及 P4 回合内物理条件反转、在线规划耗时和任务成功率均无本阶段结果。PushT 与 PushObj 的记录限于 §2 列出的宿主核验。 |

**实现与复算边界。** 以上源码片段保留了实验实际调用的计算。拟合与测量使用 Python 3.10.15、PyTorch 2.1.1+cu121 和两个 CPU 线程。重新执行完整原生前向需要作者的 LeWM 模型实现、TwoRoom 配置与宿主权重、动作统计，以及该次运行的编码轨迹和原型检查点。若仅核算 §7 的概率变化与聚合恢复率，所需的旧支持、相对 log 似然、分子和分母已列在该节。
