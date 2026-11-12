# Query feedback：固定离线学习阶段

`source` 模式下，两臂均完成固定 60 epoch / 240 次 Adam 更新。训练参数为首 attention QKV 的原始权重，W_O 冻结，共 294,912 个原生参数，模型始终 eval。前缀 6 项和后缀 4 项等权。开发数据仅用于初始和固定最终快照的评价。

数据为此前使用过的 50 条 P0b 专家轨迹：0–29 拟合，30–49 为本次拟合未使用的开发留出；它们不是新测试集。统计单位为 episode。

| 开发对照（正值为 full 改善） | H | 平均 MSE 改善 | 中位数 | 改善 episode / 20 |
|---|---:|---:|---:|---:|
| feedback_full_vs_native_finetune | 1 | -0.00043987 | -0.00007066 | 9 / 20 |
| feedback_full_vs_native_finetune | 2 | -0.00058140 | -0.00025416 | 7 / 20 |
| feedback_full_vs_native_finetune | 4 | -0.00151693 | -0.00187830 | 6 / 20 |
| feedback_full_vs_native_finetune | 8 | -0.00383856 | -0.00381757 | 5 / 20 |
| feedback_full_vs_same_parameters_cleared | 1 | -0.00059990 | +0.00009264 | 11 / 20 |
| feedback_full_vs_same_parameters_cleared | 2 | -0.00082309 | -0.00001518 | 9 / 20 |
| feedback_full_vs_same_parameters_cleared | 4 | -0.00147667 | -0.00125797 | 7 / 20 |
| feedback_full_vs_same_parameters_cleared | 8 | -0.00354907 | -0.00395265 | 5 / 20 |
| feedback_full_vs_same_parameters_shuffled | 1 | -0.00064188 | +0.00009109 | 10 / 20 |
| feedback_full_vs_same_parameters_shuffled | 2 | -0.00087325 | +0.00028891 | 11 / 20 |
| feedback_full_vs_same_parameters_shuffled | 4 | -0.00207762 | -0.00031501 | 8 / 20 |
| feedback_full_vs_same_parameters_shuffled | 8 | -0.00397625 | -0.00446272 | 5 / 20 |

| 训练臂 | 预测调用 | 反馈特征调用 | 源状态保留 | 训练耗时（秒） |
|---|---:|---:|---:|---:|
| query_feedback | 3360 | 2880 | 1440 | 59.967 |
| native_finetune | 3360 | 0 | 0 | 47.404 |

full 与同参数 cleared 的差异说明已训练模型对当前记忆的依赖；full 与 native_finetune 的差异比较整条训练和状态路径。shuffled 只说明经历对应关系的敏感性，错误状态造成损害本身不证明正常状态有益。这些对照均不能证明 query 地址或 r+delta 写入优于一般记忆方法。

本阶段没有同起点的未见动作分支、额外真实刷新后的续算或最终任务评价，不能据此声称跨动作迁移、候选排序、长期稳定性或任务成功。

逐 episode 原始预测、六项前缀误差、各 horizon 误差及训练曲线在两个 arm 子目录；初末选定参数和参数差独立保存。完整分布见 comparisons.json 与 episode_comparisons.csv。上表记录实际调用次数与训练耗时。反馈臂的预测调用还包含记忆读取。

反馈臂的初末 outputs.npz 同时保存每个 episode 在前缀结束后的六条 query/value 记录。
