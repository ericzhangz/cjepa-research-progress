# 结构升级 B：FourRoom 层级宿主能接入什么

2026-11-16。本文核对发表数据、公开实现与已有本地资产，回答两件事：静态 FourRoom 的高成功率是否验证了预测不穿墙；官方层级模型是否已包含我们需要的持续信念与按需分支。两项都未由公开结果验证。不过，完整的 HWM3 是一种可取得的多尺度宿主，值得用同根物理轨迹与现有平面路径直接比较。

## 公开结果的口径

| FourRoom Distractors 配置 | 成功率 | 改动 |
|---|---:|---|
| 作者 LeWM flat | 40.7 ± 6.8% | 单层预测，原生 latent 代价 |
| 作者 H-JEPA3 | 96.0 ± 1.1% | 联训的三层预测、分层 latent、顶向下规划 |
| 作者 HWM3 | 99.3 ± 0.7% | 联训的三层预测、同一 latent 空间、同一顶向下规划 |
| T2G dense | 81.3 ± 2.9% | 冻结 flat LeWM，全部未来预测状态的 T2G 距离取均值 |
| T2G last-2 | 78.7 ± 2.9% | 同一冻结模型与距离函数，只取最后两步；时域不足时取可用步 |
| 本地 LeWM flat seed 42 | 19/50 = 38.0% | 已完成官方 50 个任务的单模型种子运行 |

前五行是作者发表均值 ± 标准误，模型/规划种子为 42、43、44，每种子评估同一 50 个固定起终点，FourRoom 的预算为 210 个环境步。T2G 的 81.3 与 78.7 来自同一版本、同一张表的不同聚合；本地 P0c 记录（原工程资料） 中的 78.7 也是文献值。本地单种子 38.0 不可当作三种子基线；本轮没有得到本地 HWM、H-JEPA 或 T2G 成功率。[H-JEPA 官方结果](https://github.com/kevinghst/H-JEPA/blob/main/docs/RESULTS.md#depth-comparison-figure-6) · [H-JEPA 论文 v2 §4.1、附录 E/H](https://arxiv.org/html/2610.06805v2#S4.SS1) · [T2G 论文 v1 表 1–2、§3](https://arxiv.org/html/2611.05847v1#S3) · [本地协议](../../runs/hjepa_official_fourroom_flat_20261115/PROTOCOL.md) · [本地结果](../../runs/hjepa_official_fourroom_flat_20261115/REPORT.md)。

H-JEPA3 还有一个可用的分解：固定其已训练的 L1 动力学与平面规划，原生 L1 成功率为 43.3%，改用 L2/L3 投影代价为 65.3%/71.3%；完整层级为 96.0%。前几项把代价空间的作用单独测出来；完整层级还改变了时间分解、各层时域以及候选分配，不能把差值解释为某个预测器单独修掉的碰撞错误。HWM3 的结果进一步说明 FourRoom 可从时间分解受益，即使上层状态编码是恒等映射；三种子统计不支持把两个层级模型宣称为普遍优劣顺序。[H-JEPA 官方代价对照](https://github.com/kevinghst/H-JEPA/blob/main/docs/RESULTS.md#cost-ladder-tables-1-and-9) · [论文 §4.2](https://arxiv.org/html/2610.06805v2#S4.SS2)。

## 静态到达与预测碰撞是两项测量

官方 FourRoom 的任务起终点相隔两间房、记录端点相隔 75 个环境步。公开实现的 `close_door_prob: 0.5` 由 reset 使用：要么不关门，要么从四扇门中选一扇；静态任务加载器也支持从初始字段恢复关门状态。关门同时影响图像的 wall/door mask 和碰撞，回合步进不切换该状态。官方代码的成功事件是：执行的动作由模拟器碰撞处理后，自身位置在预算内进入目标的 16 像素半径。论文附录 B 对 reset 的简述与发布代码的抽样方式不完全相同，这里由代码具体说明部署环境。[三层评估配置](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/config/eval/fourroom_l3.yaml) · [渲染及门掩码](https://github.com/kevinghst/H-JEPA/blob/main/stable_worldmodel/envs/four_room/env.py#L390-L509) · [关门抽样](https://github.com/kevinghst/H-JEPA/blob/main/stable_worldmodel/envs/four_room/env.py#L774-L794) · [真实步进及到达](https://github.com/kevinghst/H-JEPA/blob/main/stable_worldmodel/envs/four_room/env.py#L971-L996) · [任务加载与成功汇总](https://github.com/kevinghst/H-JEPA/blob/main/stable_worldmodel/world.py)。

成功率说明给定模型、代价、求解与真实环境组成的闭环常能到达。即使某些候选 latent 在门前错误地跳到另一房间，物理环境也会挡住实际动作；规划还会从新观测再算，可以改走合法路线。较粗的两帧端点跨房本身也不等于穿墙，中间可以经过开放门。要诊断模型穿墙，必须从同根比较**同一串原始动作**下的实际轨迹、房间和预测：把实测受阻而预测进房，与实际可从开门通过、只是在粗时间上跨房的轨迹分开。作者的到达表、专家轨迹代价单调性和精选执行图没有统计这类错房间；它们也没有给出该错误必然存在的结论。[H-JEPA 附录 E、I](https://arxiv.org/html/2610.06805v2) · [T2G §3.3](https://arxiv.org/html/2611.05847v1#S3.SS3)。

我们的 `ReversibleFourRoom` 只在物理碰撞函数内部设置锁门，渲染时恢复开门；模式可在同一回合中反转。布局坐标也与发布配置不同，研究接口保留实际到达的三帧和其间真动作。于是两个模式可有同一合法 RGB 历史、同一目标与动作，却在有区分力的门前动作上产生不同实际后果。以官方单帧规划为例，确定性且不接收解释状态的 $F(z,a)$ 对两个相同输入只有一个输出；它不能同时给出两条不同的物理后果。增加时间层、改目标距离，或只读取更多仍相同的帧，都不会消除这个条件缺口。这个不可区分例不表示官方模型在锁门任务必败：保守路线、实际受阻后的重规划、或窗口已含区分事实，仍可能使它到达。是否能跨无信息间隔保留并使用事实，要在我们的任务上另外检验。[任务物理实现](../p1a/physical_topology_gate.py) · [配对实验](../../runs/native_topology_paired_20261115/REPORT.md) · [固定候选写入分解](../../runs/native_topology_paired_20261115/writein_attribution_20261116/REPORT.md)。

## 真正的层间接口

- **时间和动作。** 每层训练接收三状态因果上下文，监督真实上下文的一步后继；FourRoom 的 L1/L2/L3 一步分别为 5/10/20 个原始环境步。L1 将五个依序的二维原始动作经原归一化拼为 10 维。L2/L3 的 stride 是 2、状态编码窗口是 1；状态编码逐点应用，未池化过去三个状态。每个上层的 `SequenceEncoder` 读取两个下层已嵌入的动作 token（各 192 维），用带位置的单层 Transformer/CLS 汇聚成 4 维宏动作，再由动作 embedder 投给 predictor。输入槽支持别的长度不等于学会了别的物理时间尺度。H-JEPA3 的 L2/L3 各有独立 latent MLP（192→384→192）与 SIGReg；HWM3 上层 E 为恒等且无上层 SIGReg，但保留独立粗步 predictor/宏动作 encoder。两层级配置都联训 L1，HWM 空间恒等只在该 checkpoint 的各层之间成立。[原文式 (1)–(3) 与附录 C](https://arxiv.org/html/2610.06805v2#S2.SS1) · [H-JEPA3 配置](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/config/train/fourroom_hjepa_l3.yaml) · [HWM3 配置](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/config/train/fourroom_hwm_l3.yaml) · [层训练数据流](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/models/hjepa.py#L239-L270) · [动作编码](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/models/encoders/seq_encoder.py#L30)。
- **子目标和执行。** 官方规划每轮重新编码当前真实画面与目标，每个 predictor 的 rollout 默认只看最近一状态。L3 优化到全局目标的距离，L2/L1 依次追踪上层第一个预测状态，将自身 rollout 按 stride 投到同一时间、同一上层坐标比较；通常一个上层子目标对应两个下层步。FourRoom 的跨层动作匹配权重为零；4 维宏动作训练队列产生搜索先验，不是实际历史，也不保证优化出的宏动作有对应的下层原始命令。L1 才输出五步命令块；每轮最多执行两个块再从真实图像重算。评估初始层级时域由低到高为 (2,2,6)，候选数为 (50,200,200)；上层时域降到 1 时跳过上层求解，FourRoom 可保留投影后的目标成本。论文的训练上下文、上层时间尺度及发布代码的宏动作队列都不能当作跨真实轮次的持续状态。[原文式 (4)–(6)](https://arxiv.org/html/2610.06805v2#S2.SS2) · [solver 子目标与层跳过](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/hierarchical_solver.py#L118-L214) · [planner rollout/projection](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/hierarchical_solver.py#L372-L485) · [单层 rollout](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/models/jepa.py#L376)。
- **T2G 接点。** 式 (9)、(12) 学的是 $d(z,z_g)=\alpha\max(10^{-6},\|f(z)-f(z_g)\|_2)$；同一离线轨迹的锚、近、远三个冻结图像 latent 提供 margin 排序。$d$ 对称，没有绝对步数标定，没有同一根跨动作、跨隐锁条件的物理顺序监督。FourRoom 冻结编码器和 predictor，训练指标时只调用编码器；规划时保留原预测轨迹，然后选择 terminal 或 dense（可截取 last-n）评分。故原 T2G 可以在匹配的 flat LeWM 上作评分器对照，既不会更改同一动作的预测，也不应把其 MLP 直接用在 HWM 或 H-JEPA 的 192 维 latent 上。[式 (9)–(14) 与附录 A](https://arxiv.org/html/2611.05847v1#S2.SS3) · [距离与损失](https://github.com/yrluestc/T2G-JEPA/blob/main/T2G-JEPA/hjepa/temporal_metric.py) · [冻结训练](https://github.com/yrluestc/T2G-JEPA/blob/main/T2G-JEPA/hjepa/train_temporal_metric.py#L74-L145) · [rollout/代价](https://github.com/yrluestc/T2G-JEPA/blob/main/T2G-JEPA/hjepa/models/jepa.py#L379-L505) · [FourRoom 配置](https://github.com/yrluestc/T2G-JEPA/blob/main/T2G-JEPA/hjepa/config/eval/fourroom_flat_metric.yaml)。

## 可取的资产和不能拼接的边界

官方 [模型文件清单](https://huggingface.co/api/models/jepa-world-models/h-jepa/tree/main/fourroom?recursive=true) 列出 `fourroom_hwm_l3`、`fourroom_hjepa_l3` 各 seed42/43/44 的完整 `*_object.ckpt` 与配套 `normalizer.pt`，单个 checkpoint 分别为 85,933,967 与 87,128,555 字节。两者共用 `fourroom_l3` 配置，使用 [作者的固定任务发布](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/main/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt)。本地此次检查到的是临时目录下已经运行过的官方 flat seed42 权重、2,239 字节的 normalizer、官方源码树及端点评估缓存；开始核对时，在这些检查范围及本地 HF hub 目录未找到两组完整层级权重。端点缓存保留发布任务的首尾 RGB 与非图像字段，不是完整中间视频或层级训练集。T2G [发布文件](https://github.com/yrluestc/T2G-JEPA/blob/main/ckpts/fourroom/fourroom_lewm/seed42/temporal_metric.pt) 是 735,434 字节的 seed42 指标，作者明确依原始 LeWM 训练；文件未绑定其使用的确切本地编码器坐标。H-JEPA3 和 T2G 未下载或运行。

本轮下载了 **HWM3 seed42** 的配套 normalizer 和 checkpoint 的首部及尾部，使用完整的对象元信息在 CPU 上做形状核对（FakeTensorMode，不填补缺失权重做数值预测）。配套 normalizer 的 action、xy、distractor0_xy、min_distractor_dist 所有统计值与本地 flat 数值一致。三层实际对象的 predictor 为 ProjectedPredictor/ARPredictor，位置表均为 [1,3,192]；L1 的 ProjectedEncoder/ViTModel 用归一化原始动作块 10→192，L2/L3 的 ProjectedEncoder 外壳内状态编码与投影均为 Identity，动作 encoder 为 SequenceEncoder，192→4、实际 stride 2、max_chunk 3、动作队列形状 [4000,4]，4 维宏动作再嵌入为 192。三层实际保存的相邻 stride 是 (1,2,2)，状态窗口都是 1，L1 的 5 环境步来自外部数据动作打包。

首部还含 HWM3 的真实 ViT CLS token（192 个参数）。与 flat seed42 同名参数的数值差 RMS 为 0.02930、最大绝对差 0.07207，直接否定了两者使用同一 L1 参数的假设；这不是该观察上的 latent 实测差。HWM 同一 checkpoint 的 L2/L3 的状态编码逐点精确恒等，是执行路径中确切的坐标关系；各层预测器的输出未被要求精确相同。官方完整下载中断，小区间下载的实测速率约 12–22 KB/s。本轮未取得完整 checkpoint，因而没有用 HWM 真实权重编码首帧或运行 rollout。

本地 HWM 工作目录为 `AppData/Local/Temp/cjepa-hwm3-interface-20261116`。其中 normalizer.pt（原工程资料） 是完整的配套统计；fourroom_hwm_l3_incomplete_download.bin（原工程资料） 是完整文件下载中断后的 16,502 字节残片；official_head.bin（原工程资料） 和 mirror_tail_90k.bin（原工程资料） 保存用于元数据检查的首尾范围。元数据脚本 metadata_only.py（原工程资料） 只用于核对模块与形状、比较下载范围内实际存在的 CLS 参数，生成的 `partial_metadata_only.ckpt` 不能用于数值前向。

在已有官方 FourRoom 环境下，复查元数据及参数的 PowerShell 命令如下：

```powershell
$assetRoot = Join-Path $env:TEMP 'cjepa-hjepa-official-20261115'
$officialHost = Join-Path $env:TEMP 'cjepa-p0a-ff1d5adbd6ae448e993c31d8bf9c4668'
$dependencies = Join-Path $env:TEMP 'cjepa-fullview-rooms-20261115'
$source = Join-Path $officialHost 'upstream'
$py = 'anaconda3/envs/neuro-veomni-import/python.exe'
$env:HJEPA_HOME = Join-Path $env:TEMP 'cjepa-hwm3-interface-20261116'
$env:PYTHONPATH = "$assetRoot/gymnasium-1.2.3;$dependencies/four-transformers;$source/h_jepa;$source;$env:TEMP/cjepa-physical-topology-gate-deps;$dependencies/neuro-deps;$env:TEMP/cjepa_p1a_deps;$officialHost/probe_deps;$env:TEMP/cjepa-pusht-018-restored-h_l5t_2i"
& $py (Join-Path $env:HJEPA_HOME 'metadata_only.py')
```

单根 CPU 接口检查（原工程资料） 需要 85,933,967 字节的完整发布权重，读取同目录的 `normalizer.pt`。它在加载模型前拒绝已知的下载残片和按发布长度拼成的元数据文件。完整权重可用后，沿用上面的环境运行：

```powershell
& $py 'research/p1a/native_hwm_interface_probe.py' `
  --checkpoint (Join-Path $env:HJEPA_HOME 'fourroom_hwm_l3_object.ckpt') `
  --flat-checkpoint (Join-Path $officialHost 'fourroom_lewm_object.ckpt') `
  --endpoints (Join-Path $assetRoot 'assets/fourroom_official_endpoints.pt')
```

该脚本只在官方任务 0 的首帧上编码，再用该任务前 20 条真实动作检查 L1/L2/L3 的 4/2/1 步 rollout 及原生 `predict`，输出同帧 HWM 与 flat 的 latent 差。即使调用通过，也只验证接口，不是隐藏锁移植、碰撞预测验收或官方成功率复现。

HWM3 和 H-JEPA3 的下层架构与平面宿主近似相同，但作者用独立训练，且上层损失反传到下层编码；同 seed、维数与输入图像并不能建立坐标同一。应以各 checkpoint 中 **同一组** encoder、动作嵌入/宏动作编码、预测器、训练动作先验和归一文件为一套加载。现平面模型训练得出的条件偏置、门控、latent 似然尺度和物理读出也依赖原宿主参数或坐标，必须重新绑定、用真轨迹校准并验证；原来宽度相同也不足以复用其数值。保留现有平面宿主时没有官方可直接插上的已训练宏步模块，需从现宿主重新拟合层间接口，此分支不能引用 99.3% 作为预期结果。[H-JEPA 附录 C.1](https://arxiv.org/html/2610.06805v2) · 本地时间及模块合同（原工程资料）。

## 推荐的最小迁移与验收

1. **选完整 HWM3 seed42 作为结构宿主候选。** 依据是共享状态坐标让当前物理读出和层间实现问题可直接定义，时间分解本身有静态证据；不是按 99.3 与 96.0 的数值挑模型。先冻结该宿主，以它自己的 L1 平面 rollout 作为严格控制，和同 checkpoint 的 L2/L3 比较；现 flat seed42 保留参照。第一步在已固定的真实根、目标和实际动作上，用原 action encoders 做 10/20 环境步聚合，比较粗步预测、递归细步和真实碰撞、房间及末端位置。在物理上同为绕行优先的共同案例和真正随模式换序的案例分别看候选差；无条件宿主探针可以检验是否改善共同错误，不能替代隐锁测试。H-JEPA3 可作为需要分离上层表示几何时的对照，T2G 是现平面坐标上的终端评分对照。
2. **把当前持续支持接到同一组粗细预测，而非只调整最后评分。** 复用真实三帧 L1 窗口与预发预测的写入方式；一次事实到达才修订公共支持 $p_t(m)$。上层以从同一真实起点对齐的窗口起步；没有已到达的粗历史时，不从未来观测补齐。对每个保留的解释 $m$，具体命令 $u$ 与宏动作 $A_\ell(u)$ 在同一物理条件下进入 $F_{\ell,m}$；递归细步与原生粗步应在相同物理时刻与目标坐标上与真结果一致。即对两个细步，检查

   $$
   \widehat z^{\ell+1}_{m,1}
   = F_{\ell+1,m}(z^{\ell+1}_{m,0},A_{\ell+1}(b_0^\ell,b_1^\ell)),
   \qquad
   \widetilde z^{\ell+1}_{m,1}=E_{\ell+1}(\widehat z^\ell_{m,2}).
   $$

   这两条预测分别对齐相同的真实未来作比较；二者彼此接近不能排除共同预测错。候选上层子目标保持 $m$、物理时刻与实参对应，到下层仍由同一 $m$ 细化。最终候选在各解释下必须对应**同一个能执行的首块**，再按公共支持合成代价。固定开放式候选时，各解释应评同一条完整原始动作序列；若允许未来根据新观测调整后续动作，比较的应是同一因果行动规则，不能让后续命令直接读取未观测的真模式。若上层自由优化的宏动作没有可实现的细动作见证，子目标不能当作已可达。区分下层可行性和上层打分，避免把两模式平均后的假想中间状态传回细层。
3. **先验证模式已知，再验证事实作用和节省。** 当前提案下先保留同预算、同候选的已知模式对照，检验共同受阻与合法绕行是否改善，以及模式相关换序的方向和幅度是否保留；真实间隔和起点相同才比较层级时域。粗步内部若可能因门前后果而分歧，不能仅凭一个粗步端点重合就认为其内部可共享。沿用按需分歧的原则：仅共享经原生计算确认尚未出现条件依赖的公共段；若粗层无法确认，应经原细层检验或按解释分别求值，仍保留持久支持。用密集逐模式评估作同权重对照，核对同候选的物理读出、行动排序与预发证据后再报告共享获益。反转调节随后用既有真实交互复播检验 true / uniform / 旧支持 / 实际写入；未来真实切换时刻不可作为输入，也不把假想前向写回事实。当前固定库中真模式也不能修复共同排序、跨动作的条件差仍不稳定，层级必须分别通过这两个约束才有继续做实际闭环比较的理由。提案 §§1–5（原工程资料） · [共同项与条件项分解](../../runs/native_topology_paired_20261115/writein_attribution_20261116/REPORT.md)。

本轮复核范围：H-JEPA v2 §2、§3、§4.1–4.2 及附录 B/C/D/E/H/I 的相关段落；T2G v1 §2.3–2.4、§3、附录 A、表 1–3；上述发布配置、源码入口和模型列表；本地 H-JEPA 推导（原工程资料）、执行路径（原工程资料）、T2G 阅读（原工程资料）、[发布任务运行协议](../../runs/hjepa_official_fourroom_flat_20261115/PROTOCOL.md) 及隐藏锁代码与配对报告。本文件交付来源核对、HWM3 的发布对象元信息与一个实际底层参数的数值对照，以及迁移条件。本轮没有完整权重前向、训练、调整超参数、复跑 50 项或启动新闭环。
