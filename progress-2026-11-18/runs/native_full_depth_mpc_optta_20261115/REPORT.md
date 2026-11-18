# 可逆 PushT 的官方源码在线调参对照

复用 OpTTA 作者发布的 LeWM `OpTTAAdapter` 原文件。AdaJEPA 对照采用同一份作者实现的 `rho=0` 分支，OpTTA 采用 `rho=1`。两臂各更新 2,591,552 个参数，覆盖末 Transformer 块、输出归一化和预测投影。编码器固定。其余设置沿用作者 PushT 配置，每次更新一步 Adam，学习率 `5e-5`，保留最近五个实际执行段。OpTTA 的计划项使用上一轮 CEM 最后一个样本群的全部 16 条候选，按作者实现用梯度范数缩放，随已完成更新次数递增。将调度的总重规划次数从作者的 30 改为本试验的 7。 [作者适配器](https://github.com/akshaymete/optta/blob/main/lewm/code/lewm_optta/adapter.py) · [作者配置](https://github.com/akshaymete/optta/blob/main/lewm/code/lewm_optta/config.yaml) · [作者执行顺序](https://github.com/akshaymete/optta/blob/main/lewm/code/lewm_optta/policy.py)

本试验沿用相同官方 LeWM 检查点、可逆环境、四个种子、动作归一化和 CEM 设置。各臂每轮 16 条样本、3 次迭代、4 个精英、2 个模型步预测，执行一个模型步。桥接代码将实际 RGB 后继和已执行命令提供给原适配器，以本试验三帧原生窗口和终点 latent 目标代价计算上一轮的计划项。源码自带的单帧及分阶段目标代价由这一桥接替换，以便各臂使用同一个规划目标。每个 episode 恢复权重。环境真实制动状态只记录于评估结果，从未提供给适配器或代价函数。 桥接入口（原工程资料，未随包同步） · [共同 CEM](../../research/p1a/native_reversible_pusht.py) · 源码及许可证（原工程资料，未随包同步）

| 方法 | 成功场景 | 平均最高覆盖率 | 平均末步覆盖率 | 平均评分回调耗时 |
| --- | ---: | ---: | ---: | ---: |
| 冻结 LeWM | 0/4 | 0.7881 | 0.7815 | 18.99 ms |
| 持续支持 | 0/4 | 0.8847 | 0.8847 | 19.98 ms |
| EvidenceGRU | 1/4 | 0.8729 | 0.8729 | 23.48 ms |
| AdaJEPA（作者 LeWM 适配版） | 1/4 | 0.8612 | 0.8490 | 41.53 ms |
| OpTTA | 0/4 | 0.8577 | 0.8473 | 44.26 ms |

四个场景开始时，两种在线方法的首个实际动作和规划代价均与冻结 LeWM 逐值相等。每个 episode 各有 21 次评分回调和 6 次实际后继更新。OpTTA 首次更新按作者的递增规则将乐观项权重置零，两臂随后执行的动作也逐值相同。之后的更新均记录正的乐观项权重。每个 episode 的 336 条 CEM 候选预算相同，OpTTA 的梯度额外对已录下的 96 条候选重算前向，没有增加采样。运行中 AdaJEPA、OpTTA 单次适配耗时中位数分别为 85.21、90.50 ms。评分回调包含新到达处理及实际发生的适配，耗时来自同一设备上的独立运行，受 GPU 热身和进程状态影响。 逐场景数据（原工程资料，未随包同步） · [冻结和持续支持原数据](../native_full_depth_reversible_20261115/closed_loop.json) · EvidenceGRU 原数据（原工程资料，未随包同步）

本轮是相同 CEM 采样预算下的四种子闭环比较。持续支持的平均覆盖率最高，AdaJEPA 和 EvidenceGRU 各有一个成功场景。任务成功数和本轮样本规模不足以判断长时域方法的优劣。
