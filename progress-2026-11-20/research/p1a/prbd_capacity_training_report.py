"""Render the fixed Phase 1 training records without reading evaluation data."""

import json

from .prbd_capacity_oracle import OUT, write_json
from .prbd_paths import output_path


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = {}
    for arm in ("A", "B"):
        directory = OUT / arm
        history = json.loads((directory / "objective_history.json").read_text(encoding="utf-8"))
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        final = history[-1]
        early = next(row for row in history if row["update"] == 62)
        previous = next(row for row in history if row["update"] == final["update"] - 512)
        records[arm] = {"result": result, "early62": early, "final": final,
            "last512_relative_objective_decrease": (previous["objective"] - final["objective"]) / previous["objective"]}
    write_json(OUT / "training_summary.json", records)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for arm, color in (("A", "#4878a8"), ("B", "#c84c4c")):
        history = json.loads((OUT / arm / "objective_history.json").read_text(encoding="utf-8"))
        updates = [row["update"] for row in history]
        axes[0].plot(updates, [row["objective"] for row in history], color=color, label=arm)
        axes[1].plot(updates, [row["oracle_H1"] for row in history], color=color, label=f"{arm} H1")
        axes[1].plot(updates, [row["oracle_H4"] for row in history], color=color, linestyle="--", label=f"{arm} H4")
    for axis, title in zip(axes, ("Full training objective", "True-condition endpoint MSE")):
        axis.set_title(title)
        axis.set_xlabel("Optimizer updates")
        axis.set_ylabel("Latent MSE")
        axis.set_yscale("log")
        axis.grid(alpha=.2)
        axis.legend(frameon=False)
    figure.savefig(output_path(OUT / "training_objectives.png"), dpi=180)
    plt.close(figure)
    lines = ["# Phase 1 oracle capacity training", "",
        "This is a supervised offline capacity diagnostic on the same 62 usable G3 trajectories. It does not train PRBD belief updates or authorize Phase 2.", "",
        "The collector's fixed RNG(20261120) speed permutation is checked against stored physical metadata. Slow 2.5 maps to branch 1 and fast 7.5 to branch 2. Branch 0 is the literal frozen host. Branch 3 receives both conditions' same true targets with pooled supervision; it has equal total loss weight to the labeled branches together and is retained as an ordinary fourth branch. Oracle readout uses branches 1/2. There is no null-control reference.", "",
        "All 682 actual roots with a complete H4 suffix are used. For every root, H1 and freely rolled H4 endpoint latent MSE receive weights 1/2 each; the true-condition branch and pooled branch 3 also receive weights 1/2 each. No likelihood, noise, XY readout, development outcome, or future beyond the actual training target enters the objective.", "",
        "A trains exactly the original 3×6×192 = 3,456 condition offsets. B retains these and adds a rank-4 map from the latest native action condition to each condition/layer's offset, plus the jointly trained last two native blocks shared by all conditions. B computes F0 + [G(k) − G(0)] and returns literal F0 for branch 0. Its 3,614,848 trainable parameters comprise 3,456 offsets, 14,592 bilinear parameters, and 3,596,800 last-block parameters.", "",
        "Both arms use the same initialization, ordered training data, sampled batches of 16 roots, Adam at 3e−4, and preset 10× learning-rate reductions at updates 2048 and 3072. The predeclared limit is 4096 updates. At or after 3072 updates, a nonnegative full-training-objective decrease of at most 0.1% over 512 updates stops training. The final state is retained; development data do not select checkpoints.", "",
        "| Arm | Updates | Seconds | Early62 objective | Final objective | Oracle H1 | Oracle H4 | Last512 decrease | Converged by preset rule |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm, record in records.items():
        result, final = record["result"], record["final"]
        lines.append(f"| {arm} | {result['updates']} | {result['seconds']:.2f} | {record['early62']['objective']:.8f} | {final['objective']:.8f} | {final['oracle_H1']:.8f} | {final['oracle_H4']:.8f} | {100*record['last512_relative_objective_decrease']:.5f}% | {result['converged']} |")
    lines += ["", "This stopping result describes convergence under the preset optimization schedule. It does not prove a global optimum or an absolute capacity limit. The A/B contrast tests their combined response families; it does not separately identify the contribution of the bilinear map and last-two-block training. Generalization and physical action-effect recovery belong to the separate fixed-checkpoint evaluation.", "",
        "CPU unit checks reproduce A's original unanchored path exactly, confirm B contains A at initialization, verify nominal H1–H4 bitwise equality after changing shared G weights, and establish nonzero pooled-branch gradients from both physical conditions. Training checks verify finite native gradients and the selected-branch API. CUDA is used for training; fixed milestone training trajectories are exported on CPU.", "",
        "The per-arm folders retain the written protocol, every minibatch loss, full-training objective every 128 updates plus update 62, initial/62/final checkpoints, and identical chronological training-root trajectories at those milestones. No hyperparameter or checkpoint search used evaluation scores.", "",
        "![Training objectives](training_objectives.png)", ""]
    output_path(OUT / "TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "TRAINING_REPORT.md", flush=True)


if __name__ == "__main__":
    main()
