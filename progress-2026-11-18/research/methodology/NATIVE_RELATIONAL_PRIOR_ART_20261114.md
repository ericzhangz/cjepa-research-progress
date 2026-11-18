# 原生响应关系的直接技术参照

2026-11-14。设最后一块的原生激活为 \(v=G(u)\)，候选在进入原生 \(W_2\)、动作门与预测头之前，用 \(T_k(v)\) 代替 \(v\)。各解释共享关系方向与阈值 \((B,\beta)\)，离线学习各自的非负强度 \(\lambda_k\)，部署时固定。

\[
T_k(v)=\arg\min_q\left\{\tfrac12\|q-v\|^2+\tfrac12\sum_e\lambda_{ke}[B_e q-\beta_e]_+^2\right\}.
\]

## 算子等价

- **最直接的原式是无摩擦柔顺接触。** Castro、Permenter 与 Han 的 SAP §III.C 式（24）在无摩擦时只保留非负法向冲量，式（23）的局部投影为 \([\hat v_{c,e}-J_e x]_+/R_{n,e}\)。于是速度目标恰为 \(\tfrac12\|x-x^*\|_A^2+\tfrac12\sum_e R_{n,e}^{-1}[\hat v_{c,e}-J_e x]_+^2\)。取 \(q=A^{1/2}x\)、\(v=A^{1/2}x^*\)、\(B=-JA^{-1/2}\)、\(\beta=-\hat v_c\)、\(\lambda_e=R_{n,e}^{-1}\) 即得本式。原文式（22）的 \(JA^{-1}J^\top\) 已承担多关系间的同时耦合。 [SAP §III.A–C，式（19）、（22）–（24）](https://arxiv.org/pdf/2110.10107#page=5)
- **OptNet、LCS 与 monDEQ 可精确复现。** 令 \(S=\operatorname{diag}(\sqrt{\lambda_k})\)、\(M=SB\)、\(s=S(Bv-\beta)\)。给 \(Mq-S\beta\) 加非负松弛变量 \(r\)，将目标写成 \(\tfrac12\|q-v\|^2+\tfrac12\|r\|^2\)，约束 \(r\ge Mq-S\beta\)、\(r\ge0\)，就是 OptNet 的 QP。最优条件给出 \(q=v-M^\top\alpha\) 与 \(0\le\alpha\perp(I+MM^\top)\alpha-s\ge0\)，即对称正定 LCP，亦可写成 ReLU 平衡式 \(\alpha=[s-MM^\top\alpha]_+\)，对应 monDEQ 的对称特例。单行或正交行时分解为已有的单侧平方惩罚近端映射，即移位、反向后的 PReLU。 [OptNet 式（1）与 §3](https://proceedings.mlr.press/v70/amos17a/amos17a.pdf) · [Learning LCS 式（1）](https://proceedings.mlr.press/v168/jin22a/jin22a.pdf#page=3) · [monDEQ 式（2）与定理 1](https://arxiv.org/pdf/2006.08591#page=3) · [Combettes–Pesquet 例 2.7](https://pcombet.math.ncsu.edu/svva5.pdf#page=4)

## 事实到未来的最近参照

Perez、Such 与 Karaletsos 已在同一动作条件动力学中用环境隐变量解释实际转移与未来。式（1）沿序列共用一个环境隐变量。式（3）和（5）以新到达的实际下一状态似然递推后验，部署时固定动力学网络。算法 1 在每条粒子轨迹开始时从当前后验取隐变量，以 TS∞ 递推候选动作序列，再用实际执行的后果更新后验。cJEPA 按模型设定的似然精确重加权有限解释，并逐解释展开。真实后果更新公共后验、同一解释贯穿多步，以及在各步预测后按支持汇总，都可由已有隐变量预测直接解释。 [Perez 等式（1）、（3）](https://arxiv.org/pdf/1812.03399#page=2) · [式（5）与 §2.2](https://arxiv.org/pdf/1812.03399#page=3) · [算法 1 第 9–10、14–17 行](https://arxiv.org/pdf/1812.03399#page=4) · [PETS §5.1 TS∞](https://proceedings.neurips.cc/paper_files/paper/2018/file/3de568f8597b94bda53149c7d7f5958c-Paper.pdf#page=6)

事实和每条未来共享解空间中的超平面 \(B_e q=\beta_e\)，解释只改变关系强度。相互作用由 \(MM^\top\) 的非对角项决定，改变一条关系可以改变另一条的响应。若 \(Bd=0\)，有 \(T_k(v+d)=T_k(v)+d\)，所以该方向的激活变化各解释一律原样传递。参考动作使用完整预测，关系差异经过原生读出及观测模型仍可区分时，真实参考后果就能修订各解释的权重。待验证的是共享关系的约束能否让实际动作上的证据更可靠地传递到其他动作及较远未来。已有算子加已有后验和 rollout 已足以解释当前计算图，写在 JEPA 的哪个端口本身无法建立计算贡献。该等价已经说明此处没有独立于已有模型的核心计算主张；不能由后续涨分把等价运算改判为原创。后续§19改为构造响应及其解释原像共同产生的求值方式，区别须先在数学上说明。

## 直接对照

1. **独立关系的精确收缩。** 保留同一 \((B,\beta,\lambda_k)\)、观测似然、原生接口和逐解释 rollout。只在求 \(\alpha\) 时将 \(MM^\top\) 替换为其对角，得到 \(\tilde\alpha_e=[s_e]_+/(1+\lVert M_e\rVert^2)\)，仍以 \(v-M^\top\tilde\alpha\) 读出。两者在只有一条关系或行彼此正交时精确相同，有耦合时可定位实际观测的解释权重及独立真实候选后果是否因相互作用而改进。
2. **普通条件化的有限隐变量模型。** 以参数量匹配的原生激活残差 \(v+R_\theta(v,k)\) 替代关系求解。两者使用相同的宿主可训练范围、真实信息、观测似然、有限后验、分支递推、训练监督和可比计算预算。该比较考察共享关系结构能否改善同一事实向其他动作及后缀的迁移。
