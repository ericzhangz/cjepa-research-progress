"""Render Phase 2 factual report from independent score artifacts and training logs."""
import json

from .prbd_phase2 import OUT
from .prbd_paths import output_path


def read(name):
    return json.loads((OUT/name).read_text(encoding="utf-8"))


NAMES = {"frozen":"Frozen LeWM","gru":"Plain GRU","adajepa":"AdaJEPA","optta":"OpTTA",
         "uniform":"同预测、均匀支持","first":"仅首次","omit_first":"省略首次","full":"完整写入"}


def report():
    s, sw = read("static_results.json"), read("switch_results.json")
    fit, checks = read("A/result.json"), read("A/final_checks.json")
    h = s["physical_nonzero_panel"]["horizon"]
    n = s["physical_nonzero"]
    pre = s["probes"]["initial"]["matched"]
    w = s["probes"]["trained"]["matched"]
    guards = {
        "真实静止写入保持":w["statistics"]["write"]["p95"]<=.01 and w["over_half"]==0,
        "H1/H4 首次及完整写入改善跨动作预测":all(h[i][arm]["directed_percent"]>0 and
            h[i][arm]["sse_gain_percent"]>0 and 2*h[i][arm]["improved"]>n
            for i in ("1","4") for arm in ("first","full")),
        "两个反转方向分别至少半数恢复":all(sw["full"]["summary"][d]["opportunities"]>0 and
            sw["full"]["summary"][d]["recovered"]*2>=sw["full"]["summary"][d]["opportunities"]
            for d in ("to_slow","to_fast")),
        "相对四对照在 H4 交互及两方向切换后误差有可靠优势":all(
            h["4"][m]["prbd_gain_over_method"]["ci95"][0]>0 and
            all(sw[m]["summary"][d]["prbd_gain_over_method"]["ci95"][0]>0 for d in ("to_slow","to_fast"))
            for m in ("frozen","gru","adajepa","optta"))}
    verdict="全部通过" if all(guards.values()) else "未全部通过"
    intro = [
        "# PRBD Phase 2 · 无标签真实写入、跨动作预测与回合内反转",
        "", f"TwoRoom 独立检验{verdict}。静态部分使用 {n} 个同窗且具有真实交互差的重置，切换部分另用 48 个重置。训练完成 {fit['updates']:,} 次更新，使用最终检查点。两部分沿预定动作测量预测与写入，任务成功率留给后续闭环。",
        "", "## 机制与训练", "",
        "执行一次真实命令前，冻结宿主与三个原生条件偏移分别签发后继，合计 3,456 个学习参数。先把上次支持乘对称转移矩阵（对角 14/15，非对角 1/45），再用真实新帧和 G1 185 节点有界子空间密度作贝叶斯更新。四个响应的均匀中心与冻结的 Ledoit–Wolf 尺度决定评分几何。每个候选复用同一份支持，各原型自由续推各自的原生窗口。对一个 H1–H4 预测窗，使用第一次输运后的权重和固定原型近似。遇到下一次真实观察再输运、取证。",
        "", "192 个新训练重置中有 186 条至少提供一个合法三帧读根，合计 5,522 个训练读根。轨迹包含四向运动、正交运动、反向返回、静止与连续随机命令。每回合预设两次变速，发生时刻在两个区间独立抽取，提前终止按实际长度保留。训练只接收真实 latent、已执行命令、有效长度与采样用的种子编号；编号不输入网络。每条轨迹内递推的支持保留梯度。每个中途读根对该次实际动作后缀的 H1 至 H4 混合联合负对数似然按跨度和维数取平均。反事实速度、坐标及切换时刻只供数据生成和离线评分。",
        "", "每个真实读根另加 0.001 × 三个非名义分支相对宿主的原生平方偏离之和 / 192。Adam，batch 4，初始学习率 0.0003，在第 1,024 和 1,536 步各乘 0.1。冻结尺度来自另 32 个历史校准种子。训练输入上的重复 FP32 宿主前向给出白化最大两两距离死区，低于阈值仅跳过观测写入。静止或零命令没有任何特例。",
        "",f"同一训练全集的每维多步 NLL 从 {fit['initial']['nll_per_dimension']:.6f} 到 {fit['final']['nll_per_dimension']:.6f}。最小偏离项乘权后由 {fit['initial']['weighted_min_deviation']:.3g} 到 {fit['final']['weighted_min_deviation']:.3g}。末态复核名义分支 H1–H4 与宿主逐值相等。终点训练检查中，选定后根损失对较早实际写入的梯度范数为 {checks['full_prefix_source_to_later_suffix_gradient_L2']:.6g}。该量验证梯度路径是否贯通。",
        "", "## 两次真实静止到达", "",
        "以下主表在 39 个严格同窗根、两个速度上各计两次真实静止，候选共享的前缀只计一次。写入 TV 为后验与本步输运后先验的总变差距离。近保持定义为最多移动一个百分点的概率质量。另核对关闭数值死区时同一签发的写入。",
        "", "| 权重 | 到达数 | 精确保持 | TV≤0.01 | 写入 TV 均值 / P95 / 最大 | TV>0.5 | 死区触发 |","|---|---:|---:|---:|---|---:|---:|"]
    for label, stats in (("无标签初始化",pre),("最终",w)):
        v=stats["statistics"]["write"]
        intro.append(f"| {label} | {stats['n']} | {stats['exact']} | {stats['near']} | {v['mean']:.5f} / {v['p95']:.5f} / {v['maximum']:.5f} | {stats['over_half']} | {stats['deadzone']} |")
    all_w=s["probes"]["trained"]["all"]
    intro+=["",f"扩到全部 48 个重置中的合法真实静止到达，TV≤0.01 共 {all_w['near']}/{all_w['n']} 次，"
            +f"P95 为 {all_w['statistics']['write']['p95']:.5f}，最大为 {all_w['statistics']['write']['maximum']:.5f}。"]
    stat=w["statistics"]
    two=w["two_quiet_actual_vs_transport_only"]
    worst=w["strongest_write"]
    intro += ["",f"白化数值阈值为 {checks['deadzone']:.8g}。最终模型的单步先验输运 TV 均值为 {stat['transport']['mean']:.5f}，观察与输运合计为 {stat['total']['mean']:.5f}。禁用数值死区后，同响应写入 TV 的均值 / 最大为 {stat['no_deadzone']['mean']:.5f} / {stat['no_deadzone']['maximum']:.5f}。将同一似然施于均匀先验，平均 TV 为 {stat['at_uniform']['mean']:.5f}。两次静止后的实际支持与只输运两次的支持距离为均值 {two['mean']:.5f}，最大 {two['max']:.5f}。最大单次写入发生在种子 {worst['seed']}、速度 {worst['speed']}、块 {worst['transition']}，白化分支分离为 {worst['separation']:.4f}。",
        "",f"在这些同窗根的真实静止后继上，名义分支支持有 {w['nominal_mass_increased']}/{w['n']} 次增加，平均增量为 {w['mean_nominal_mass_change']:+.4f}。各分支取得最高似然的次数按原型 0–3 依次为 {w['highest_likelihood_by_branch']}。原型编号只表示模型内部响应。",
        "", "物体在快慢两种速度下都静止，只约束两个物理后继相同。它没有规定冻结宿主已经准确预报这一帧的视觉 latent。原生平方偏离项在分支重合处梯度为零，真实后继的似然梯度仍可能因宿主预测残差而推动分支分开。终点检查取预先固定的前四条训练轨迹、120 个读根，加权偏离项与 NLL 对同一偏移张量的梯度范数分别为 "
        +f"{read('A/final_gradient_decomposition.json')['gradient_L2']['weighted_min_dev']:.3g} 与 {read('A/final_gradient_decomposition.json')['gradient_L2']['NLL']:.3g}。该局部测量没有将静止漂移唯一归因到某项损失。",
        "", "Phase 1 的 156/156 来自有真速度监督的容量模型、闭式 Student-t 与此前 39 个开发根。本轮采用无标签学习、G1、另一组种子和非恒等输运。上表首行与末行共享本轮协议。对正则项的单独因果贡献需要等预算去掉该项的训练对照。",
        "", "## 同一窗口的跨动作、跨时域读出", "",
        f"48 个新静态重置，{s['matched']} 个满足完整轨迹、相同 RGB/latent、相同物理位置与同一候选菜单，{n} 个还有非零物理交互差。每个根有慢/快两段历史和移动/停止两个未来。先算同一速度的移动减停止，再算慢减快得到真实向量 δ。模型变化沿 δ 的总投影除以真实 δ 的总平方范数给出有向恢复率。交互 SSE 改善以均匀支持的总 SSE 为分母，正值表示改善。以下数值按根配对，训练与独立根分离。",
        "", "| 支持 | H | 有向恢复 % | 交互 SSE 改善 % | 改善根 | 移动端点 MSE | 停止端点 MSE |","|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ("uniform","first","full","omit_first"):
        for i in range(1,5):
            row=h[str(i)][arm]
            a,b=row["mse_move_stop"]
            intro.append(f"| {NAMES[arm]} | {i} | {row['directed_percent']:+.3f} | {row['sse_gain_percent']:+.3f} | {row['improved']}/{n} | {a:.6f} | {b:.6f} |")
    intro += ["", "首次只吸收探测后继，完整再吸收返回与两次静止，省略首次只吸收后三次。各臂时钟一致。完整减省略首次在 H1/H4 的有向增量分别为 "
        +f"{h['1']['full_vs_omit']['directed_percent']:+.3f}% / {h['4']['full_vs_omit']['directed_percent']:+.3f}%。",
        "", "同一预测经既有 TwoRoom 坐标读出后，可与模拟器实际位置交互差作独立比较。读出是先前用作者训练库拟合的冻结 OLS，独立检查的坐标 RMSE 为 4.58 地图单位。分数包含读出自身误差。下表按 H1 到 H4 报告有向恢复百分数。",
        "", "| 写入 | H1 | H2 | H3 | H4 |","|---|---:|---:|---:|---:|"]
    for arm in ("first","full","omit_first"):
        xy=s["physical_xy_panel"]["horizon"]
        intro.append("| "+NAMES[arm]+" | "+" | ".join(f"{xy[str(i)][arm]['directed_percent']:+.3f}" for i in range(1,5))+" |")
    intro += [
        "", "## 实际条件反转与对照", "",
        "另 48 个重置各自运行两条相反起始速度的实际轨迹，均走完 32 块，每条在第 10 和 20 块开始前变速。每个实际动作执行之前保存一步预测，评分时比较它与同一物理起点、相同命令下两个速度对应的真实后继。3072 个实际后继均与对应的反事实重演逐值一致。可辨步处连续两次更接近当前真实速度算一次持续恢复。从切换算起记录这一对中的首次/确认延迟，中间实际经过的静止步也占时间。表中延迟只对已恢复事件取中位数，未恢复单列。重复事件的区间估计按重置重采样。",
        "", "| 方法 | H4 交互 SSE/根 | 慢→快 恢复 / 有机会 | 首次 / 确认延迟 | 切换后 MSE | 快→慢 恢复 / 有机会 | 首次 / 确认延迟 | 切换后 MSE |","|---|---:|---:|---|---:|---:|---|---:|"]
    for arm in ("frozen","gru","adajepa","optta","uniform","full"):
        row=[NAMES[arm],f"{h['4'][arm]['interaction_sse_mean']:.4f}"]
        for d in ("to_fast","to_slow"):
            v=sw[arm]["summary"][d]
            row.extend([f"{v['recovered']}/{v['opportunities']}",
                        f"{v['median_lag']} / {v['median_confirmation']}" if v['recovered'] else "—",
                        f"{v['mse']:.6f}"])
        intro.append("| "+" | ".join(row)+" |")
    intro+=["", "PRBD 在切换前最后两个可辨动作均预测正确的子集，慢→快 / 快→慢分别恢复 "
        +"、".join(f"{sw['full']['summary']['pre_correct_'+d]['recovered']}/{sw['full']['summary']['pre_correct_'+d]['opportunities']}" for d in ("to_fast","to_slow"))
        +"。主表包括切换前已错的情况。切换后偶然连对两次与因写入而改对有不同含义，同分支均匀臂提供配对参照。",
        "", "Frozen、PRBD 和在线适配均从同一冻结编码表示出发。Plain GRU 用同一批无标签单轨、读根、动作和 2,048 次更新，隐藏宽度 192，参数量 369,984，从头拟合。AdaJEPA/OpTTA 使用发布的 LeWM 在线适配器和合法实际前缀，每条连续轨迹独立初始化，切换处保留适配状态。OpTTA 的优化项使用当时公开的重置目标及执行前计算的影子 CEM 池（16 个候选 × 3 轮，H4），实际执行命令仍由既定脚本给出。训练参数量、适配开销与目标访问按方法披露；这些是同批实际到达上的比较。",
        "", "### PRBD 相对对照的配对差", "",
        "正值表示 PRBD 的误差更小。区间以独立重置为单位做 4,000 次配对 bootstrap。静态列为平均每根交互 SSE 差，切换列为每根切换段每维 latent MSE 差。",
        "", "| 对照 | H4 静态 | 慢→快 | 快→慢 |","|---|---:|---:|---:|"]
    for arm in ("frozen","gru","adajepa","optta","uniform"):
        info=[h["4"][arm]["prbd_gain_over_method"]]+[sw[arm]["summary"][d]["prbd_gain_over_method"] for d in ("to_fast","to_slow")]
        intro.append("| "+NAMES[arm]+" | "+" | ".join(f"{a['mean']:+.5f} [{a['ci95'][0]:+.5f}, {a['ci95'][1]:+.5f}]" for a in info)+" |")
    intro += ["", "### 计算口径", "",
        f"PRBD 最终训练进程共 {fit['seconds']:.1f} 秒，包含训练内核及全库监控。Plain GRU 同预算训练实耗 {read('baselines/gru_training.json')['training_seconds']:.1f} 秒。PRBD 使用 QR 加小矩阵 SVD 计算同一个 G1 密度；替换前后的密度及梯度已对原核复核。早期等价实现的计时运行单列归档，独立结果只来自本次最终权重。",
        "", "| 对照 | 真实到达梯度更新数 | 影子池次数 | 影子规划秒 | 在线适配秒 |","|---|---:|---:|---:|---:|"]
    for arm in ("frozen","gru","adajepa","optta"):
        rows=read(f"baselines/timing_{arm}.json")["episodes"]
        intro.append(f"| {NAMES[arm]} | {sum(r['updates'] for r in rows)} | {sum(r['shadow_solves'] for r in rows)} | "
                     +f"{sum(r['shadow_seconds'] for r in rows):.2f} | {sum(r['adaptation_seconds'] for r in rows):.2f} |")
    intro += ["", "各方法共享部分 GPU 运行时段。计时描述本机这次作业。这里的 AdaJEPA 使用作者 LeWM 适配代码中的零 optimism 分支，OpTTA 使用同一代码中的完整分支；两者每次可适配 2,591,552 个参数。两表检验给定实际序列上的预测收益，不能据此认定官方规划成功率。"]
    intro += ["", "## 阶段判定", ""]
    intro += [f"- {k}　{'通过' if v else '未通过'}。" for k,v in guards.items()]
    intro += ["", "判定连同条件窗内模型的近似、静止漂移和两方向的恢复情况一起解释。静态同窗的向量恢复量刻画预测，闭环任务成功率需要在社区环境另测。",
        "", "### 可复算数据", "",
        "本目录的 static_results.json 含所有匹配根、非零物理差子集、冻结坐标读出和静止写入分布。static_physical_nonzero_roots.json 保留逐根逐跨度分子及 SSE。switch_results.json 和 switch_*_events.json 保留各方法两个切换方向、可辨机会、切换前已正确子集与逐事件延迟。A/updates.jsonl 及 A/initial_checks.json、A/final_checks.json 记录全训练过程和梯度检查。baselines/ 下包含各对照签发的预测。"]
    output_path(OUT/"REPORT.md").write_text("\n".join(intro)+"\n",encoding="utf-8")
    output_path(OUT/"admission.json").write_text(json.dumps(guards,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    report()
