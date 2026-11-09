# ID-WM：论文接口与官方代码的公开边界

阅读日期：2026-11-07。论文版本：*Hierarchical World Models with Implicit Dynamics*，arXiv:2610.42146v1。本文承接 [方法推导笔记](IDWM_METHOD_DERIVATION.md)，专门区分论文算法、实际公开文件与尚不可读取的实现。

**目前无法完成 ID-WM 的官方训练—规划—环境执行调用链阅读，因为官方仓库尚未发布实现。** 这不是源码阅读后的实现结论；是公开源本身的范围。下面不补造文件名、函数名、配置默认值或张量排列。

## 1. 实际打开的官方来源

作者主页的论文条目将 code 指向 `gaoyuezhou/id_wm`。已通过 GitHub 连接器读取该仓库的根目录、README、assets 目录、分支列表与 release 列表。根目录只有 `README.md` 和 `assets/`；assets 中只有 `diverse_plans.png`、`teaser.gif`；当前分支列表只有 `main`，release 列表为空。README 明示“Code will be released soon. Stay tuned!”。项目页也将代码标为 coming soon。[作者主页](https://gaoyuezhou.github.io/) · [官方仓库](https://github.com/gaoyuezhou/id_wm) · [README](https://github.com/gaoyuezhou/id_wm/blob/main/README.md) · [项目页](https://id-wm.github.io/)

没有公开训练脚本、DDIM 实现、high/low-level planner、CEM、数据集读取器、环境执行入口、配置文件或 checkpoint 下载说明。仓库中的图像与动画是展示材料；它们不是可执行示例，也不能据此声称拿到了完整论文代码。

论文采用 DINO-WM 风格的低层世界模型，不等于现有 DINO-WM 仓库就是 ID-WM 的实现。尤其高低层坐标转换、规划嵌套和执行频率属于 ID-WM 自己的组合接口，不能由组件仓库替代。

## 2. 公式到实现：能确定的接口与缺失的位置

以下符号是用于解释论文的接口记号，不是已经发现的 Python API：B 为 batch，c 为低层历史长度，d_s 为所处表示空间维度，d_a 为低层动作维度；t 是环境采样时间，j 是扩散索引。高层和低层的 d_s 不应预先视为相等。

| 原文位置 | 论文明确描述的运算 | 可由数学式确定的张量关系 | 官方文件／函数 |
| --- | --- | --- | --- |
| [§3.1，模型定义](https://arxiv.org/html/2610.42146v1#S3.SS1) | 高层根据当前状态生成 Δ 跨度未来；低层根据历史和动作预测下一状态 | 高层条件 s_t，目标 s_{t+Δ}；低层历史至少含状态与对应动作 | 未公开 |
| [式 (1)](https://arxiv.org/html/2610.42146v1#S3.E1) | 对干净未来状态加噪 | x_0、x_j、ε 同形；状态向量可记 B×d_s，视觉 latent 保留其空间结构 | noise scheduler 与 dataset 配对均未公开 |
| [式 (2)](https://arxiv.org/html/2610.42146v1#S3.E2) | 条件噪声预测损失 | ε_θ(x_j,j,s_t) 与 ε 同形；j 是扩散步，不是环境时间 | denoiser、loss reduction、训练入口均未公开 |
| [式 (3)](https://arxiv.org/html/2610.42146v1#S3.E3) | 确定性 DDIM 将初始噪声 z 变成邻接未来状态 | z 与被生成状态同形；给定 s_t、z、固定日程得到一个确定端点 | DDIM timesteps、scheduler 调用、精度与随机数路径未公开 |
| [§3.1.2，式 (4)](https://arxiv.org/html/2610.42146v1#S3.SS1.SSS2) | Transformer 低层前向预测；以真实下一状态监督 | 抽象输入为 B×c×d_s 的历史与相应动作；输出下一状态。具体 token 组织不能从公式推得 | action encoder、predictor、teacher-forcing 切片均未公开 |
| [式 (5)–(6)](https://arxiv.org/html/2610.42146v1#S3.E5) | 高层搜索噪声序列，最小化终点距离与噪声幅度惩罚 | N 个候选可记 N×H_hi×shape(z)，逐跳调用高层模型 | high-level planner、regularizer reduction 均未公开 |
| [§3.2](https://arxiv.org/html/2610.42146v1#S3.SS2) | 针对每个子目标，用低层前向模型搜索动作 | 候选动作可抽象记 N×H_lo×d_a；历史如何复制、滚动、截断未知 | low-level planner、动作边界、warm start 未公开 |
| [附录 §6.5，CEM 算法](https://arxiv.org/html/2610.42146v1#S6.SS5) | 采样候选、前向评分、选 elite、更新搜索分布 | N 为样本数，K_elite 为入选数；不能把它们与扩散 K 混用 | 分布更新、方差下限、batch 与 episode 轴未公开 |
| [§3.2，层次执行算法](https://arxiv.org/html/2610.42146v1#S3.SS2) | 高层产生子目标链，低层求动作并执行，取得真实状态，再继续 | 真实观察替换执行后的状态；不能把预测子目标当作已到达观察 | env.step、终止处理、实际刷新频率未公开 |

这张表中的 B×c×d_s 等表示数学接口，而非实际 tensor layout。不能据此写成“作者代码中某函数接收这个 shape”。

## 3. 真实时间、扩散时间和动作时间

论文主设置把环境序列按 frameskip=5 处理，高层 Δ=5 对应 25 个原始环境步。因此至少存在三种不同索引：原始环境步、低层采样步，以及扩散去噪步。论文描述了这些尺度，但没有公开数据读取器来确定动作块的具体端点约定。[附录 §6.1](https://arxiv.org/html/2610.42146v1#S6.SS1)、[§6.7](https://arxiv.org/html/2610.42146v1#S6.SS7)

可以将低层状态暂记为 S_t=O_{5t}，高层监督暂记为 (S_t,S_{t+5})，用于解释时间含义；这是时间尺度的重建，不能冒充作者代码。以下实现问题仍无官方答案：

- 一个低层动作 token 是 5 个连续动作的拼接、重复动作，还是其他编码；动作落在两帧之间的确切切片是什么。
- 历史长度 c 的实际含义是否包含当前帧，序列起点如何补齐，teacher forcing 与 rollout 是否使用相同上下文窗口。
- Δ 是否真正作为网络输入；正文接口写有 Δ，但具体噪声网络写为 ε_θ(x_j,j,s_t)，主实验为固定 Δ。不能据此宣布已实现一个可任意查询时域的模型。
- 执行一段低层计划时，是逐 primitive action 刷新，还是执行完整动作块后刷新；论文层次算法与具体 wrapper 调用之间尚无源码连接。

扩散索引 j 不改变真实环境时间。运行 50 个 DDIM 步不是预测 50 个环境步。

## 4. 状态坐标：最不能直接猜接的接口

状态向量实验与视觉实验需要分开看。PushT 的状态增强明确涉及 agent 位置、速度、T 形物体位置和角度的 sin/cos；视觉高层使用 VAE latent，论文给出 3×64×64 图像对应 4×8×8 latent 的设置。低层定量实验采用 DINO-WM 相关表示。[附录 §6.3](https://arxiv.org/html/2610.42146v1#S6.SS3)、[§6.11](https://arxiv.org/html/2610.42146v1#S6.SS11)

因此，视觉 high-level output 能否直接用作 low-level goal，不能仅凭两者都称为 latent 来回答。需要源码明确：是否先解码再编码、是否共用 encoder、是否存在另一种规划配置，以及 high/low cost 分别在哪个坐标中计算。当前公开资料不足以把任何一个方案写成作者实现。

这也意味着不能把 ID-WM 高层 VAE latent 直接接到当前 FourRoom LeWM 的 192 维向量上，并称为复用官方方法。其形状、坐标、训练目标和距离含义都尚未给出共同接口。

## 5. 训练梯度与规划搜索分别做什么

论文算法层面，高层训练对噪声预测参数求梯度，低层训练对下一状态预测参数求梯度。规划中的两层 CEM 则调整候选采样分布，通过前向值比较选取 elite；它不是沿 DDIM 轨迹对噪声做梯度下降，也不是在线更新世界模型权重。[§3.1–3.2](https://arxiv.org/html/2610.42146v1#S3)、[附录 §6.5](https://arxiv.org/html/2610.42146v1#S6.SS5)

“低层 inverse dynamics”在这里指用前向世界模型求解动作逆问题。它不能改写成作者训练了一个单独的逆动力学网络。相反，DDIM 的确定性作用是令高层噪声成为可重复评价的决策变量；确定性本身不提供物理可达性的硬约束。

具体哪些参数被冻结、encoder 是否在训练中更新、何处调用 detach/no_grad、优化器 parameter groups 如何构造，均无源码可读。论文公式不能替代这些代码事实。另一个需要保留的区别：式 (4) 写的是未平方 L2 范数，不能擅自报告为代码中的 MSE。

## 6. 论文表格配置不等于代码默认值

下面仅摘录论文配置的数值接口，用于将来对应源码；本轮没有发现任何 config 文件。[附录 §6.7](https://arxiv.org/html/2610.42146v1#S6.SS7)、[§6.11](https://arxiv.org/html/2610.42146v1#S6.SS11)

| 设置 | 论文报告值 |
| --- | --- |
| frameskip / 高层 Δ | 5 / 5 |
| H_hi / H_lo | 5 / 5 |
| state DDPM / DDIM 步数 | 50 / 50 |
| visual DDPM / DDIM 步数 | 250 / 25 |
| CEM 候选数 / elite / 优化迭代 | 300 / 30 / 30 |
| 最大重新规划次数 / variance scale | 15 / 1 |
| 噪声正则权重 α / 阈值 τ / 幂 p | {0,2} / 2 / 1 |
| high-level embedding / down dimensions / kernel / groups | 56 / [256] / 3 / 8 |
| high-level learning rate / weight decay / warmup | 1e-4 / 0.1 / 200 |
| low-level history：PushT / Wall / Maze | 3 / 1 / 3 |
| predictor / action encoder learning rate | 5e-4 / 5e-4 |
| low-level depth / heads / MLP dimension | 6 / 8 / 64 |
| dropout / embedding dropout / pooling | 0 / 0 / mean |

α 的集合尤其不能转换成一个所谓统一默认值。状态、视觉、环境和消融设置也不能拼成作者实际使用过的一份总配置。

## 7. 对已有笔记的界限说明

已有 `IDWM_METHOD_DERIVATION.md` 是论文方法推导，未提供官方代码调用链。本轮没有发现可以纠正其中某个实际 Python 函数描述的源码，因为它没有声称读过这些函数。必须继续保留其中的两个限定：高层噪声网络是否显式接收可变 Δ 未知；视觉高低层坐标桥接未知。

进一步明确一点：将某一步动作解释为 frameskip 对应的动作块，是时间尺度抽象，不是已经读到的拼接实现。训练图中的梯度路径、DDIM scheduler 参数、CEM 方差更新与真实执行的逐步切片，在正式方法实现报告里必须保持未定。

本轮没有下载一个占位目录来制造“已取得代码”的外观，也没有把其他论文的实现重新标为 ID-WM。无法完成的具体部分是官方函数级与真实执行级映射；论文算法脉络的阅读已经完成。

## 8. 可实施的最小机制步骤及其身份

**目前不能提出一个能立刻调用 ID-WM 官方函数的实验命令。** 对 ID-WM，最小有意义的源码实验要等 denoiser、checkpoint 和状态桥接公开：固定一个真实起点、一个高层噪声和一个采样日程，保存 DDIM 端点；让一个低层 CEM 实现该端点，只执行这一段动作，并保存执行前后真实观察。它直接展示“噪声决策→子目标坐标→实际动作”的机制链，不需要完整 benchmark 或训练。若自行写高层 diffusion 替代，就应明确是独立实现，不能称为官方调用结果。

对当前动作条件 JEPA，能立即落实、又不依赖未发布 ID-WM 的最小构造步骤，是把**跨时域 operator 的返回值改成可继续调用原生 predictor 的完整状态接口**，而非只返回终点 latent。具体使用同一个原生 predictor，固定动作前缀 A 和后续动作 B：细步路径先计算完整续接状态 W_A；拟议跨步路径返回同形 W̃_A；两者交给完全相同的后续 predictor 调用，目标是替换前缀计算后仍可继续处理 B 和下一次真实观察。这里的新增对象只能是前缀算子的替代，不应另建子目标搜索器或 inverse solver。

这条建议是我们的方法接口选择，不是 ID-WM 的算法复现。ID-WM 对它的启发仅在于：先决定高层变量承担什么计算职责，再让模块、训练目标和执行路径围绕同一个职责展开。它不证明我们的紧凑工作态存在，也不证明一个 memory token 加一致性 loss 就构成新的方法。下一步真正要构造的是替代算子的状态闭合方式；无需借未发布的官方代码继续扩张系统。
