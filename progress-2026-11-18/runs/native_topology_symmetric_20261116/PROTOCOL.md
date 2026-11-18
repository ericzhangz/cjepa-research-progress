# 双门对称监督、固定坐标约束及真实到达打分

本轮只回答三个问题：双门都有监督后，现有原生条件计算能否同时学对两处门锁的后果；解除硬门控并补上物理坐标约束后，同一候选内的虚假捷径和条件反号是否减轻；固定预测后，观测打分是否能减轻中性步误翻，而不损失真反转或首动作收益。沿用现有 FourRoom 和 FullDepthConditioner。本阶段复用已使用过的开发复查布局，不能作为最终泛化结果。

两组均从 `native_topology_prefix_20261115/prefix/conditioned.pt` 继续。固定既有 32 个拟合布局、每根 4 条 H5 计划、两模式后继、相同真实前缀与抽样索引、同一随机种子、160 个 AdamW 更新。Arm 1 保留原门控、H1–H5 latent MSE、真实窗一步 latent MSE 及两项门控 BCE；Arm 2 保持第一个共享原生块，将后续的条件暴露固定为 1，训练及推理均如此，无门控 BCE。原 latent 两项保持系数；两条物理路径各加同一冻结 XY 读出对真实块末坐标的 MSE。坐标项除以训练根上的固定读出灵敏度 $G=\mathbb E[\|J_{\rm readout}(z)\|_F^2]/2$。在一阶近似下，单位各向同性 latent 误差对每坐标造成的平均平方误差为 $G$，故这一换算无需根据评估效果调权。坐标由实际执行提供，只作为离线标签。编码器及坐标读出冻结。

同一组合里的上门和下门监督必须分别来自两模式下实际执行同一动作；单条候选因开锁而受益，不等于同根上、下两条候选确实换序。报告每方向的覆盖、严格物理换序和共同受阻/合法移动。这一轮允许 Arm 2 同时改门控与训练目标：若两指标都改善，只能说组合修订对这批样本有效、现有参数化能表达实测到的差异；若仍失败，也不能据此证明全局码结构表达不了。仅在同权重推理时切换 Arm 1 门控，能读出门控的推理净作用，不能替代单因子训练对照。

评价固定原有候选和开发布局；原模型真模式首选是否假进目标房定义 A/B，组别不随本轮模型重选。报告每臂自己的假进目标房计数、合法位移、物理真模式和均匀的首块实际遗憾，以及固定候选对 $\Delta D$ 的同向、反号、为零和相对幅度。另保留刚反转的 13 个既有有害固定候选对及查询 42、26、11，原定两候选的分项代价与同池重选分开报告。刚显露模式的历史条件与两模式都合法的物理根也分开，门洞内反转源不是合格的两模式公共决策根。

似然单独从固定预测评估。用冻结读出把已发两模式预测和到达编码都映射到 2D XY；只在与评价隔离的校准布局上，由匹配实际模式的误差定一个共享二维协方差。两个模式使用同一协方差，原支持转移、动作候选和预测轨迹保持相同。比较原始极端中性事件、真正无效的原始步、中途分开再合流、真实反转两方向、连续支持和首块遗憾。中性标签来自离线物理，仅用于评估；部署更新不能用它过滤。去除数值尖峰本身不足以证明写入可靠，也不足以开放 H6 闭环。H6 必须另满足预定物理可达性和模式决策价值，以及同池真模式和真实写入的决策前提。

## 执行入口

在项目根下使用 [原 H5 数据的 Python 与依赖设置](../native_topology_prefix_20261115/DATA.md#运行)。预报与证据分别保留在 `arm1/`、`arm2/` 与 `evidence_*`；读已有训练日志或复算汇总不需要重训。两项支持评分的重算例见 [EVIDENCE.md](EVIDENCE.md)。

```text
python -m research.p1a.native_topology_symmetric_data
python -m research.p1a.native_topology_prefix_fit --arms arm1 arm2
python -m research.p1a.native_topology_symmetric_eval --forecast arm1
python -m research.p1a.native_topology_symmetric_eval --forecast arm2
python -m research.p1a.native_topology_symmetric_eval --forecast arm1_inference_r1
python -m research.p1a.native_topology_symmetric_eval --summary --include-inference-control
```
