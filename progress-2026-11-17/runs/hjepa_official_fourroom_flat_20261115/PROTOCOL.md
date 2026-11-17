# FourRoom 官方 50 项任务上的 seed42 LeWM 平面规划

使用作者的 `fourroom_flat` 配置、`fourroom/fourroom_lewm/seed42/fourroom_lewm_object.ckpt` 和同目录的 `normalizer.pt`。本地 normalizer 的全部统计字段与重新下载的发布资产数值相同。任务来自作者发布的 `fourroom_tp35_cross2_goal75_n50_d1.pt`，共 50 个原顺序固定起点和目标，目标记录距起点 75 步，固定两次房间跨越及一个活动干扰者。 [运行说明](https://github.com/kevinghst/H-JEPA#3-reproducing-push-t-ogbench-cube-visual-antmaze-fourroom) · [模型与归一文件](https://huggingface.co/jepa-world-models/h-jepa/tree/main/fourroom/fourroom_lewm/seed42) · [任务文件](https://huggingface.co/datasets/jepa-world-models/h-jepa/blob/main/eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt)

每个闭环预算 210 个环境步，模型动作块长度 5，每次应用至多两块。原生平面求解器初始规划时域 23，随后按作者代码递减。每次优化 200 个候选，AdamW 学习率 0.1，依作者求解器在 30–90 次迭代内停机。目标代价取末两步的原生 latent 平方误差。每项记录完整执行预算内首次环境成功步与原生求解统计。 [作者配置](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/config/eval/fourroom_flat.yaml) · [梯度求解器](https://github.com/kevinghst/H-JEPA/blob/main/stable_worldmodel/solver/gd.py)

作者评估器默认每个 chunk 运行一项，按全局位置 `i` 设置规划器 seed 为 `42 + i`。本地用三个进程处理独立任务，在每次调用原生评估器时传入对应的官方任务和等价 seed，保持作者的随机流定义与一项一环境。发布轨迹的逐步字段没有 `seed`，作者评估器输出的环境 seed 是 `null`。发布元数据中的任务生成 seed 与规划器 seed 分开，评估程序的环境随机事件沿用其默认 reset 语义。 [作者评估入口](https://github.com/kevinghst/H-JEPA/blob/main/h_jepa/planning_eval.py) · [真实环境执行与成功判定](https://github.com/kevinghst/H-JEPA/blob/main/stable_worldmodel/world.py)

由于大文件 CDN 传输较慢，从官方发布文件下载 ZIP 数据目录、各任务的非图像张量以及首尾两个 RGB 帧的原始字节区间。作者 `World.evaluate_from_dataset` 为每个张量仅使用首尾时间下标，`pixels` 保留原始的 76 帧形状以维持长度元数据。启动单项时直接还原所需张量与端点，批量读取时使用具备同样字节布局的本地存档。该本地存档适用于本评估。中间 RGB 帧未读取，完整轨迹研究应使用作者的完整发布包。 [任务字段提取](https://github.com/kevinghst/H-JEPA/blob/main/stable_worldmodel/world.py)

运行环境为 Windows、RTX 4060 Laptop GPU、Python 3.11.15、PyTorch 2.8.0/cu128、torchvision 0.23.0/cu128、Gymnasium 1.2.3。Gymnasium 与作者规定版本相同，PyTorch 和 torchvision 使用本机已有的 CUDA 构建。作者 MATH attention 和原生浮点梯度求解路径均实际执行。 [作者依赖声明](https://github.com/kevinghst/H-JEPA/blob/main/pyproject.toml)

逐任务运行日志见 `task_NNN.log`，第 0 项的首次运行日志为 `probe.log`。每项原生结果在 `chunks/tasks_NNN-MMM.json`。三个 `worker_N.log` 记录运行顺序和完成时间。`summarize.py` 验证原生求解配置并导出逐任务表。作者参考值 40.7% ± 6.8% 是三个模型种子的均值和标准误。 [作者结果](https://github.com/kevinghst/H-JEPA/blob/main/docs/RESULTS.md#depth-comparison-figure-6)
