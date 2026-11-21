"""Render the repair's fixed-panel report from the final saved scores."""
import json

from .prbd_phase2 import OUT
from .prbd_paths import REPAIR


def read(directory, filename):
    return json.loads((directory / filename).read_text(encoding="utf-8"))


def report():
    static = read(REPAIR, "static_results.json")
    h = static["physical_nonzero_panel"]["horizon"]
    sw = read(REPAIR, "switch_results.json")
    geometry = read(REPAIR, "response_geometry.json")
    guards = read(REPAIR, "admission.json")
    old_static = read(OUT, "static_results.json")
    old_h = old_static["physical_nonzero_panel"]["horizon"]
    old_sw = read(OUT, "switch_results.json")
    oracle = read(REPAIR/"oracle", "result.json")
    audit = read(REPAIR/"audit", "original_checkpoint_audit.json")
    fit = read(REPAIR/"unlabeled", "result.json")
    names = {"frozen":"Frozen LeWM", "gru":"Plain GRU", "adajepa":"AdaJEPA", "optta":"OpTTA",
             "uniform":"同响应、均匀支持", "full":"PRBD 完整写入", "first":"仅首次写入", "omit_first":"省略首次"}
    lines = ["# PRBD Phase 2 修订 · TwoRoom 同面板复验", "",
        "同一起点、真实命令、候选和冻结表示上，分别检验已知当前速度时分支能否预测其后果，以及只看到真实后继时支持能否选用这些分支。训练使用 192 条单轨，其中 186 条至少有一个合法读根。评测沿用上轮留出的 39 个同窗重置和另一组 48 个切换重置。39 个同窗根中有 34 个存在非零真实动作交互。面板已在上一轮使用。",
        "", "本轮未达到 Phase 3 准入。单独用真速度训练并指定分支时，两个切换方向的端点 MSE 均低于 AdaJEPA。无标签修订也降低了端点 MSE，但同权重的均匀支持几乎取得相同结果。完整写入的静止漂移增大，H4 跨动作交互 SSE 高于均匀臂。",
        "", "## 旧目标的梯度归因", "",
        "全训练集每维混合目标从 +0.074273 降到 −0.269839。下面的干预把前四条训练轨迹的签发均值固定，替换各自签发位置的共享协方差，然后重新计算实际前缀的后验及后缀的混合损失。两行均使用相同的 456 个后缀目标。",
        "", "| 固定的均值 | 初态协方差 | 终态协方差 | 白化单位协方差 |", "|---|---:|---:|---:|"]
    for key, title in (("initial", "初态"), ("final", "终态")):
        swaps = audit["same_frozen_means_covariance_swap"][key]
        vals = [swaps[k]["mixture_prefix_nll_per_dimension"] for k in ("initial", "final", "P_zero")]
        lines.append(f"| {title} | " + " | ".join(f"{v:.6f}" for v in vals) + " |")
    gradients = audit["checkpoints"]["final"]["native_gradients"]
    lines += ["", f"终态这批轨迹的原型参数 NLL 梯度范数为 1.56835。其中均值路径范数为 {gradients['mean']['L2']:.5f}，协方差路径为 {gradients['covariance']['L2']:.5f}。两条路径是向量，分量可能相消。谱值伸缩和子空间旋转都有贡献；此干预测量固定检查点的依赖关系。原来报告的 5.219 是最大静止写入事件上的白化分支距离，该事件 TV 为 0.6794。",
        "", "## 规格与容量", "",
        "原型 0 直接调用冻结的 LeWM。其余三个原型仍通过原生条件偏移区分，共享从宿主复制的末 Transformer 块、末归一化和预测头，一共训练 2,595,008 个参数。AdaJEPA 的适配范围为 2,591,552 个参数。逐分支独立递推 H1–H4，测试期参数固定。",
        "", "在真实前缀，先用同一转移矩阵输运，再用原 185 节点 G1 更新支持。训练只固定协方差谱值的梯度，保留均值与子空间方向导数。在每个未来窗口，原型能量是各步固定白化平方预测误差之和乘 1/2，按当时的支持作混合，最后除以跨度和维数。每个实际根另用相同白化尺度约束非名义分支对宿主的平方偏离，按分支和维数平均，系数固定为 1。真实写入和后续读根之间保留梯度。",
        "", f"Oracle 使用同一训练轨迹、同一结构及 2,048 次更新，训练时可见速度，优化原始 latent 均方误差。评测先签发全族，再以当前真速度选出对应分支。无标签臂独立从已发布宿主初始化，以固定白化预测目标完成 {fit['updates']:,} 次更新；两者均取终值。Oracle 是给定真速度时已实现的性能。",
        "", "| 两个切换方向的已签发 H1 | 慢→快 MSE | 快→慢 MSE | H4 跨动作交互 SSE |",
        "|---|---:|---:|---:|"]
    oh = oracle["static"]["physical_nonzero"]["horizon"]["4"]
    for m, name in (("oracle","真速度选分支"),("uniform","Oracle 分支均匀"),("adajepa","AdaJEPA")):
        o = oracle["switch"]
        lines.append(f"| {name} | {o['to_fast'][m]['raw_endpoint_mse']:.6f} | "
                     f"{o['to_slow'][m]['raw_endpoint_mse']:.6f} | {oh[m]['interaction_SSE']:.4f} |")
    lines += ["", "## 真实静止到达", "",
        "先验输运与实际观测分别计量。下表 TV 比较本步观测后的后验与已输运先验。每个同窗重置、每种速度只对共享前缀计数一次。均匀起写列将同一组实际似然作用于均匀分布，用来辨认已饱和的先验。",
        "", "| 权重 | 精确保持 | TV≤0.01 | 平均 / P95 / 最大 TV | TV>0.5 | 死区触发 | 均匀起写平均 TV |",
        "|---|---:|---:|---|---:|---:|---:|"]
    for label, w in (("旧终值", old_static["probes"]["trained"]["matched"]),
                     ("修订初值", static["probes"]["initial"]["matched"]),
                     ("修订终值", static["probes"]["trained"]["matched"])):
        v = w["statistics"]["write"]
        lines.append(f"| {label} | {w['exact']}/{w['n']} | {w['near']}/{w['n']} | "
                     f"{v['mean']:.5f} / {v['p95']:.5f} / {v['maximum']:.5f} | {w['over_half']} | "
                     f"{w['deadzone']} | {w['statistics']['at_uniform']['mean']:.5f} |")
    w = static["probes"]["trained"]["matched"]
    reg = read(REPAIR/"audit","repair_regularizer_final.json")
    go, gs = (reg["gradient_groups"][k] for k in ("offsets","shared_suffix"))
    zero, active = (reg["H1_resident_deviation"][k] for k in ("zero_executed_command","active_executed_command"))
    lines += ["", f"修订终值的平均写入 TV 中，名义分支净转移的绝对值占 {w['mean_abs_nominal_mass_change']:.5f}，"
              f"三个非名义分支间剩余重分配占 {w['mean_remaining_redistribution_tv']:.5f}。每轮两次静止后的支持与只输运两次的支持相比，"
              f"平均 TV 为 {w['two_quiet_actual_vs_transport_only']['mean']:.5f}。", "",
              f"固定前四条训练轨迹，终态偏移和共享末层的正则／预测梯度范数比分别为 {go['regularizer_to_predictive_norm_ratio']:.3f} 和 {gs['regularizer_to_predictive_norm_ratio']:.3f}，"
              f"梯度余弦分别为 {go['cosine']:.3f} 和 {gs['cosine']:.3f}。{zero['roots']} 个零命令根与 {active['roots']} 个活动命令根的分支对宿主白化距离均值为 {zero['white_branch_L2']['mean']:.3f} 和 {active['white_branch_L2']['mean']:.3f}。"
              "正则已经产生同量级梯度，仍未使实际静止到达的分支重合。该分组只用于事后复算。[正则量级审计](audit/REPAIR_REGULARIZER_AUDIT.md)",
              "", "## 同窗跨动作响应", "",
        "固定 34 个有真实物理差的重置，比较先移动再返回的两种速度历史对同一正交移动与停止命令的影响。以同一权重的均匀臂为参照，分别报告预测交互差沿真实交互差的投影、交互平方误差改善和改善根数。末列为移动及停止各自的绝对端点 MSE。",
        "", "| H | 写入 | 有向恢复 % | 交互 SSE 改善 % | 改善根数 | 移动 / 停止 MSE |",
        "|---:|---|---:|---:|---:|---|"]
    for i in range(1,5):
        for arm in ("first","full","omit_first"):
            r = h[str(i)][arm]
            move, stop = r["mse_move_stop"]
            lines.append(f"| {i} | {names[arm]} | {r['directed_percent']:+.4f} | {r['sse_gain_percent']:+.4f} | "
                         f"{r['improved']}/34 | {move:.5f} / {stop:.5f} |")
    ceiling = geometry["hindsight_simplex_directed_percent"]
    lines += ["", "在每个根、每个跨度分别允许两段历史使用事后最佳支持，可达的有向恢复上界（H1 到 H4）为 "
              + " / ".join(f"{v:.3f}%" for v in ceiling)
              + "。上界使用了真实未来，仅对本次训练出的四个原型有效。实际写入的恢复值由同一分支动作差乘两段历史的权重差重建。",
              "", f"事后强制所有根对慢、快共用同一对原型，H4 的最佳有向恢复为 {geometry['hindsight_common_slow_fast_pair_directed_percent'][3]:.3f}%。"
              f"真实完整写入使两种历史的读出支持相差平均 TV {geometry['actual']['full']['mean_tv_histories']:.3f}。"
              f"将 34 个根的向量串联，预测交互差与实际交互差的余弦只有 {geometry['actual']['full']['global_alignment_cosine'][3]:.3f}。"
              f"均匀臂的交互差在同窗处为零。按真实交互能量归一，写入带来的两倍有向投影为 {2*h['4']['full']['directed_percent']:.3f}%，"
              f"所引入预测差的平方范数为 {geometry['actual']['full']['output_energy_percent'][3]:.3f}%。两者相减给出表中的 H4 SSE 改善。",
              "", "## 实际切换与既有对照", "",
              "每个重置有两条相反起始速度的实际轨迹，每条切换两次。先预测，实际到达后再更新。连续两个可辨动作的预测更接近当前速度对应的真实后继，才计为恢复。延迟以真实执行块计，表中中位数只取已恢复的事件。",
              "", "| 方法 | H4 交互 SSE | 慢→快 恢复 / 机会 | 首次 / 确认延迟 | MSE | 快→慢 恢复 / 机会 | 首次 / 确认延迟 | MSE |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, rows, curves in [("旧 PRBD", old_sw["full"]["summary"], old_h["4"]["full"])] + [
            (names[m],sw[m]["summary"],h["4"][m]) for m in ("frozen","gru","adajepa","optta","uniform","full")]:
        row = [name, f"{curves['interaction_sse_mean']:.4f}"]
        for d in ("to_fast","to_slow"):
            r = rows[d]
            lag = f"{r['median_lag']} / {r['median_confirmation']}" if r["recovered"] else "—"
            row += [f"{r['recovered']}/{r['opportunities']}", lag, f"{r['mse']:.6f}"]
        lines.append("| " + " | ".join(row) + " |")
    pre = sw["full"]["summary"]
    lines += ["", "限定为切换前最后两个可辨动作均正确，慢→快 / 快→慢恢复和机会分别为 "
              + "、".join(f"{pre['pre_correct_'+d]['recovered']}/{pre['pre_correct_'+d]['opportunities']}"
                         for d in ("to_fast","to_slow")) + "。下表为对照误差减完整写入误差，按重置做 4,000 次配对 bootstrap。",
              "", "| 对照 | H4 交互 SSE 差 [95%] | 慢→快 MSE 差 [95%] | 快→慢 MSE 差 [95%] |",
              "|---|---|---|---|"]
    for m in ("frozen","gru","adajepa","optta","uniform"):
        intervals = [h["4"][m]["prbd_gain_over_method"]] + [
            sw[m]["summary"][d]["prbd_gain_over_method"] for d in ("to_fast","to_slow")]
        cells = [f"{x['mean']:+.5f} [{x['ci95'][0]:+.5f}, {x['ci95'][1]:+.5f}]" for x in intervals]
        lines.append("| " + " | ".join([names[m], *cells]) + " |")
    checks = {"quiet":"静止观测增量", "cross_action":"首次及完整写入的 H1/H4 跨动作收益",
              "both_switches":"两个方向各有至少半数恢复", "fast_lag":"慢→快首次恢复中位数至多两块",
              "matched_controls":"对四个对照的 H4 交互及两方向切换误差优势"}
    lines += ["", "## 阶段判定", ""]
    lines.extend(f"- {title}　{'通过' if guards[k] else '未通过'}。" for k,title in checks.items())
    lines += ["", "冻结宿主、GRU 以及两个在线适配器复用上轮同根逐次签发的输出。两次 Phase 2 有多个训练因素同时改变，因此整轮差值不能单独归因给新末层、平方目标或偏离正则。该面板测量给定动作的预测与取证；闭环成功率需要单独测量。"]
    lines += ["", "## 训练过程与复算", "",
              "| 更新 | 全训练集固定预测能量目标 | 白化偏离代价 | 两项之和 |", "|---:|---:|---:|---:|"]
    for step in read(REPAIR/"unlabeled","objective_history.json"):
        lines.append(f"| {step['update']} | {step['nll_per_dimension']:.6f} | {step['weighted_min_deviation']:.6f} | {step['objective']:.6f} |")
    lines += ["", f"无标签训练用时 {fit['seconds']:.1f} 秒。原型 0 在 H1–H4 与宿主逐值相等。评分重用原始实际轨迹和对照签发。"
              "逐根检验、事件延迟及后验重建分别保存在 static_physical_nonzero_roots.json、switch_full_events.json 及 response_geometry.json。"
              "原型容量和旧目标固定权重的梯度分解可分别复算。[真条件容量](oracle/REPORT.md) · [旧检查点审计](audit/ORIGINAL_CHECKPOINT_AUDIT.md)"]
    (REPAIR/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    report()
