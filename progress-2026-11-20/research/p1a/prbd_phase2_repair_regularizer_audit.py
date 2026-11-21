"""Fixed first-four-track predictive/white-regularizer derivative measurement."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .prbd_paths import REPAIR
from .prbd_phase2 import OUT, load_episodes, load_phase2_checkpoint, repair_batch_objective


AUDIT = REPAIR / "audit"


def statistics(values):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not len(values):
        return {"count": 0}
    return {"count": len(values), "mean": float(values.mean()),
            "median": float(values.median()), "p95": float(torch.quantile(values, .95)),
            "maximum": float(values.max())}


def gradient_pair(q, r):
    q2 = sum(float(g.double().square().sum()) for g in q)
    r2 = sum(float(g.double().square().sum()) for g in r)
    dot = sum(float((a.double() * b.double()).sum()) for a, b in zip(q, r, strict=True))
    total2 = q2 + r2 + 2 * dot
    return {"predictive_L2": q2 ** .5, "regularizer_L2": r2 ** .5,
            "sum_L2": max(0., total2) ** .5, "dot_predictive_regularizer": dot,
            "cosine": dot / (q2 * r2) ** .5 if q2 and r2 else None,
            "regularizer_to_predictive_norm_ratio": (r2 / q2) ** .5 if q2 else None,
            "signed_projection_fraction_of_combined_gradient": {
                "predictive": (q2 + dot) / total2 if total2 else None,
                "regularizer": (r2 + dot) / total2 if total2 else None},
            "negative_predictive_gradient_backprop_linearization": {
                "predictive": -q2, "regularizer": -dot},
            "negative_regularizer_gradient_backprop_linearization": {
                "predictive": -dot, "regularizer": -r2},
            "negative_combined_gradient_backprop_linearization": {
                "predictive": -(q2 + dot), "regularizer": -(r2 + dot)}}


def write_comparison_report():
    measurements = {}
    for name in ("initial", "final"):
        path = AUDIT / f"repair_regularizer_{name}.json"
        if path.exists():
            measurements[name] = json.loads(path.read_text(encoding="utf-8"))
    if set(measurements) != {"initial", "final"}:
        return
    a, b = measurements["initial"], measurements["final"]
    lines = ["# 修订训练的白化最小偏离项：固定轨迹诊断", "",
        "只读独立恢复的初态与第 2,048 步终态，使用同一前四条实际训练轨迹 "
        "2026113000–2026113003，120 个实际读根、456 个自由后缀目标。"
        "两次均调用真实 repair_batch_objective，完整保留实际前缀反传。"
        "CPU 两线程，无优化、调参或评测面板访问。", "",
        "Q 是固定白化平方能量的后缀混合项，按合法跨度、根、轨迹归一；"
        "R 是系数 1 的 0.5×非名义分支白化 H1 平方偏离均值。", "",
        "| 同轨训练量 | 初态 | 终态 |", "|---|---:|---:|"]
    for key, label in (("squared_predictive_mixture_energy_per_dimension", "预测 Q"),
                       ("unit_white_regularizer", "白化偏离 R"), ("total", "Q+R")):
        lines.append(f"| {label} | {a['values'][key]:.10g} | {b['values'][key]:.10g} |")
    lines += ["", "梯度范数分组计算；两组使用同一损失和同一前缀。余弦为负表示两个反传方向相反。", "",
              "| 检查点 / 参数组 | ‖gQ‖ | ‖gR‖ | ‖gR‖/‖gQ‖ | cos(gQ,gR) |",
              "|---|---:|---:|---:|---:|"]
    for name, data in measurements.items():
        for group in ("offsets", "shared_suffix", "all_learned"):
            g = data["gradient_groups"][group]
            lines.append(f"| {name} / {group} | {g['predictive_L2']:.7g} | {g['regularizer_L2']:.7g} "
                         f"| {g['regularizer_to_predictive_norm_ratio']:.7g} | {g['cosine']:.7g} |")
    lines += ["", "若沿负的总反传梯度走一个无穷小欧氏步，对 R 的真实局部变化为 "
              "−gR·(gQ+gR)；正号表示这一步局部增加偏离。Q 对应列使用训练配置的反传线性化。", "",
              "| 检查点 / 参数组 | Q 的反传线性化 | R 的方向导数 |", "|---|---:|---:|"]
    for name, data in measurements.items():
        for group in ("offsets", "shared_suffix", "all_learned"):
            direction = data["gradient_groups"][group]["negative_combined_gradient_backprop_linearization"]
            lines.append(f"| {name} / {group} | {direction['predictive']:+.7g} | {direction['regularizer']:+.7g} |")
    lines += ["", "H1 驻留偏离的离线分组只读取原训练轨迹的实际原始动作。"
              "零命令表示当块 5 个原始命令均为零；这不等同于物理上严格静止。"
              "该分组未传入训练函数，也未使用速度或切换标签。", "",
              "| 检查点 / 实际命令 | 根数 | R 根均值 | 分支白化 L2 均值 / P95 |",
              "|---|---:|---:|---:|"]
    for name, data in measurements.items():
        for group in ("zero_executed_command", "active_executed_command"):
            v = data["H1_resident_deviation"][group]
            lines.append(f"| {name} / {group} | {v['roots']} | {v['regularizer_H1']['mean']:.8g} "
                         f"| {v['white_branch_L2']['mean']:.7g} / {v['white_branch_L2']['p95']:.7g} |")
    lines += ["", "数值检查：预测与正则梯度之和复原总梯度，最大分量误差为 "
              f"{a['gradient_sum_max_absolute_error']:.3g} / {b['gradient_sum_max_absolute_error']:.3g}。"
              "独立白化驻留偏离计算复原训练 R。完整逐参数、逐根统计见相邻两个 JSON。", "",
              "这些是固定检查点的局部诊断，不能确定训练变化或最终预测表现的因果来源。"
              "Q 的历史前缀反传采用固定协方差谱值的指定替代导数，故上述 Q 内积并非未修改前向的精确有限差分。"
              "这些欧氏方向也不等于经过 Adam 预条件的实际更新；未据此选择系数或增加训练。", "",
              "复算：`python -m research.p1a.prbd_phase2_repair_regularizer_audit "
              "reproduce-output/prbd_phase2_repair_20261121/unlabeled/{initial,final}.pt`。分别替换为两个检查点文件名。", ""]
    (AUDIT / "REPAIR_REGULARIZER_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def diagnose(checkpoint):
    started = perf_counter()
    model, cal = load_phase2_checkpoint(checkpoint, "cpu")
    if checkpoint.name not in ("initial.pt", "final.pt"):
        raise ValueError("This fixed diagnostic only measures initial and final checkpoints")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    expected_update = 0 if checkpoint.name == "initial.pt" else 2048
    if saved["update"] != expected_update:
        raise ValueError("Checkpoint update differs from the frozen initial/final selection")
    episodes = load_episodes(OUT / "train.pt")[:4]
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    parameters = tuple(p for _, p in named)
    loss, parts = repair_batch_objective(model, episodes, cal["noise_cholesky"],
        transition=cal["transition"], deadzone=cal["deadzone"], return_trace=True)
    predictive, regularizer = parts["nll_per_dimension"], parts["weighted_min_deviation"]
    gq = torch.autograd.grad(predictive, parameters, retain_graph=True)
    gr = torch.autograd.grad(regularizer, parameters, retain_graph=True)
    gt = torch.autograd.grad(loss, parameters)
    error = 0.
    for q, r, total in zip(gq, gr, gt, strict=True):
        assert torch.isfinite(q).all() and torch.isfinite(r).all() and torch.isfinite(total).all()
        error = max(error, float((q + r - total).abs().max()))
        torch.testing.assert_close(q + r, total, atol=3e-5, rtol=3e-4)
    groups = {"offsets": [i for i, (name, _) in enumerate(named) if name == "offsets"],
              "shared_suffix": [i for i, (name, _) in enumerate(named) if name != "offsets"],
              "all_learned": list(range(len(named)))}
    metrics = {name: {"parameter_count": sum(parameters[i].numel() for i in ids),
               **gradient_pair([gq[i] for i in ids], [gr[i] for i in ids])}
               for name, ids in groups.items()}
    # Executed raw actions label these offline summaries only. They never enter
    # the model or the training function and contain no speed or switch target.
    with np.load(OUT / "data_train.npz") as data:
        raw = {int(seed): actions.copy() for seed, actions in zip(data["seed"], data["actions"], strict=True)}
    rows = []
    for (seed, _, _), trace in zip(episodes, parts["trace"], strict=True):
        for event in trace:
            delta = (event["responses"][1:] - event["responses"][:1]).detach().double()
            white = torch.linalg.solve_triangular(cal["noise_cholesky"], delta.T, upper=False).T
            zero_command = bool(np.all(raw[seed][event["root"]] == 0))
            rows.append({"seed": seed, "root": event["root"], "zero_executed_command": zero_command,
                "regularizer_H1": float(.5 * white.square().mean()),
                "white_branch_L2": white.norm(dim=-1).tolist(),
                "raw_branch_L2": delta.norm(dim=-1).tolist(),
                "response_separation": float(event["separation"].detach())})
    subgroups = {}
    for name, selected in (("all_roots", rows),
                           ("zero_executed_command", [r for r in rows if r["zero_executed_command"]]),
                           ("active_executed_command", [r for r in rows if not r["zero_executed_command"]])):
        subgroups[name] = {"roots": len(selected),
            **{field: statistics([r[field] for r in selected]) for field in (
                "regularizer_H1", "white_branch_L2", "raw_branch_L2", "response_separation")}}
    # The training regularizer uses equal episodes and then equal resident H1
    # roots. Check that the independent resident-coordinate calculation agrees.
    reconstructed = sum(sum(r["regularizer_H1"] for r in rows if r["seed"] == seed)
                        / sum(r["seed"] == seed for r in rows) for seed, _, _ in episodes) / len(episodes)
    assert abs(reconstructed - float(regularizer.detach())) < 2e-6
    result = {"checkpoint": str(checkpoint), "update": expected_update, "device": "cpu",
        "scope": "First four original usable actual training tracks; independently restored checkpoint; no optimization or benchmark access",
        "seeds": [r[0] for r in episodes], "read_roots": parts["read_roots"],
        "suffix_targets": parts["suffix_target_steps"], "full_actual_prefix_gradient": True,
        "values": {"total": float(loss.detach()),
                   "squared_predictive_mixture_energy_per_dimension": float(predictive.detach()),
                   "unit_white_regularizer": float(regularizer.detach())},
        "gradient_groups": metrics, "gradient_sum_max_absolute_error": error,
        "H1_resident_deviation": subgroups, "H1_per_root": rows,
        "regularizer_resident_recomposition_absolute_error": abs(reconstructed - float(regularizer.detach())),
        "per_parameter": {name: {"parameter_count": p.numel(),
            **gradient_pair([q], [r])} for (name, p), q, r in zip(named, gq, gr, strict=True)},
        "limitations": ["Checkpoint-local terms and gradients do not identify the cause of the training trajectory or evaluation behavior.",
            "The predictive backward includes the configured frozen-eigenvalue G1 prefix surrogate. Its dot products describe that backpropagation rule, not exact finite differences of the unmodified forward objective.",
            "Reported negative-gradient directions are Euclidean unit-time backprop linearizations, not the actual preconditioned Adam step.",
            "Zero executed command is an offline action classification; it does not assert exact physical stillness or introduce a training gate.",
            "Subgroup summaries weight resident roots equally; training values weight episodes and then each episode's roots equally."],
        "optimizer_updates": 0, "seconds": perf_counter() - started}
    AUDIT.mkdir(parents=True, exist_ok=True)
    destination = AUDIT / f"repair_regularizer_{checkpoint.stem}.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_comparison_report()
    print(json.dumps({"checkpoint": checkpoint.stem, "values": result["values"],
                      "gradient_groups": metrics, "seconds": result["seconds"], "file": str(destination)}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    diagnose(args.checkpoint)


if __name__ == "__main__":
    main()
