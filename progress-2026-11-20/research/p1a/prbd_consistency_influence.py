"""Closed accounting of the frozen G3 development measurements; no fitting."""

from .prbd_paths import CONSISTENCY, output_path

import json
import math
from pathlib import Path

import numpy as np
import torch


OUT = CONSISTENCY
ARMS = ("h1", "dual")
METHODS = ("single_source", "all_arrivals", "omit_source")


def v(value):
    return value.detach().tolist()


def analyze():
    torch.set_num_threads(2)
    reports = {arm: json.loads((OUT / f"g3_episode_{arm}_measure.json").read_text(encoding="utf-8"))
               for arm in ARMS}
    cases = {arm: {row["seed"]: row for row in torch.load(
        OUT / f"g3_episode_{arm}_measure_tensors.pt", weights_only=True)["cases"]}
             for arm in ARMS}
    rows = {arm: {row["seed"]: row for row in reports[arm]["seeds"]
                  if row["horizons"]["1"]["physical_nonzero"]} for arm in ARMS}
    seeds = list(rows["dual"])
    assert seeds == list(rows["h1"]) and len(seeds) == 39
    influential = {h: max(seeds, key=lambda seed: abs(rows["dual"][seed]["horizons"][h]
                   ["comparisons"]["all_vs_uniform"]["squared_error_gain"])) for h in ("1", "4")}
    assert influential["1"] == influential["4"]
    root = influential["1"]
    result = {"inputs": "Existing G3 dev JSON/tensors and recorded raw/physical traces only",
              "new_training_or_model_forward": False, "G4_P4_execution": False,
              "panel": "39 matched physical-nonzero reset seeds; original panel preserved",
              "coverage": {"offered": 48, "matched": 42, "physical_nonzero": 39,
                           "physical_zero": 3, "original_excluded": 6},
              "influential_seed": root, "units": {
                  "numerator": "latent squared units: predicted interaction dot true interaction",
                  "unit_truth_projection": "latent units; projection on unit true-interaction direction",
                  "signed_recovery": "dimensionless numerator / truth squared norm",
                  "interaction_gain": "latent squared units, not divided by 192",
                  "endpoint_mse": "latent squared units per coordinate, averaged over two speeds and two programs",
                  "positions": "environment XY coordinate units"},
              "arms": {}, "crossed_root_full_numerators": {}, "closure_errors": {}}
    largest_closure, largest_score_error = 0., 0.
    for arm in ARMS:
        details = {}
        for seed in seeds:
            row, saved = rows[arm][seed], cases[arm][seed]
            path, target = saved["future_responses"].double(), saved["true_future"].double()
            assert torch.equal(path[0], path[1])
            assert torch.equal(path[:, 1], path[:, 1, :1].expand_as(path[:, 1]))
            truth = target[0, 0] - target[0, 1] - target[1, 0] + target[1, 1]
            item = {"seed": seed, "horizons": {}, "supports": row["supports"]}
            for h in (1, 4):
                key, hi = str(h), h - 1
                y = path[0, 0, :, hi] - path[0, 1, :, hi]
                c = (y - y.mean(0)) @ (truth[hi] / truth[hi].norm())
                stats = {}
                for method in METHODS:
                    weights = saved["supports"][method]
                    q = weights[0] - weights[1]
                    interaction = q @ y
                    numerator = float(interaction @ truth[hi])
                    energy = float(interaction.square().sum())
                    original = row["horizons"][key]["interaction"][method]
                    largest_score_error = max(largest_score_error,
                        abs(numerator - original["prediction_truth_dot"]),
                        abs(energy - original["prediction_squared_norm"]))
                    stats[method] = {"numerator": numerator, "prediction_squared_norm": energy,
                        "interaction_gain_vs_uniform": 2 * numerator - energy,
                        "between_speed_delta_weights": v(q),
                        "between_speed_TV": float(q.abs().sum() / 2),
                        "between_speed_overlap": float(weights.min(0).values.sum()),
                        "unit_truth_projection": float(q @ c),
                        "coefficient_cosine": float((q @ c) / (q.norm() * c.norm()))
                            if q.norm() > 0 and c.norm() > 0 else None}
                chronology = {}
                for kind in ("all_arrivals", "omit_source"):
                    increments = {}
                    for step in range(4):
                        events = [row["actual_write_traces"][s][step] for s in (0, 1)]
                        before = torch.tensor([event["weights"][kind]["before"] for event in events], dtype=torch.float64)
                        after = torch.tensor([event["weights"][kind]["after"] for event in events], dtype=torch.float64)
                        dq = (after[0] - before[0]) - (after[1] - before[1])
                        increment = float((dq @ y) @ truth[hi])
                        increments[str(step + 2)] = increment
                        if step in (2, 3):
                            assert torch.equal(before, after) and increment == 0
                    largest_closure = max(largest_closure,
                        abs(math.fsum(increments.values()) - stats[kind]["numerator"]))
                    chronology[kind] = increments
                full, single = stats["all_arrivals"], stats["single_source"]
                item["horizons"][key] = {"truth_squared_norm": float(truth[hi].square().sum()),
                    "truth_l2": float(truth[hi].norm()), "unit_truth_family_coefficients": v(c),
                    "strategies": stats, "write_numerator_increments": chronology,
                    "full_minus_single_numerator": full["numerator"] - single["numerator"],
                    "full_minus_single_gain": 2 * (full["numerator"] - single["numerator"])
                        - (full["prediction_squared_norm"] - single["prediction_squared_norm"]),
                    "original_endpoint_metrics": row["horizons"][key]["endpoints"],
                    "original_endpoint_comparisons": row["horizons"][key]["comparisons"]}
            details[str(seed)] = item
        aggregate = {}
        for h in ("1", "4"):
            values = [details[str(seed)]["horizons"][h] for seed in seeds]
            denominator = math.fsum(value["truth_squared_norm"] for value in values)
            group = {"truth_squared_norm_sum": denominator, "strategies": {}}
            for method in METHODS:
                numerator = math.fsum(value["strategies"][method]["numerator"] for value in values)
                gain = math.fsum(value["strategies"][method]["interaction_gain_vs_uniform"] for value in values)
                group["strategies"][method] = {"numerator": numerator,
                    "signed_recovery": numerator / denominator, "interaction_gain": gain}
            group["reverse_increment_with_source"] = math.fsum(
                value["write_numerator_increments"]["all_arrivals"]["3"] for value in values)
            group["zero_increment_with_source"] = math.fsum(
                value["write_numerator_increments"]["all_arrivals"][t] for value in values for t in ("4", "5"))
            single, full, omit = (group["strategies"][name] for name in METHODS)
            group["full_minus_single_numerator"] = full["numerator"] - single["numerator"]
            group["full_minus_single_interaction_gain"] = full["interaction_gain"] - single["interaction_gain"]
            group["full_minus_single_prediction_energy"] = 2 * group["full_minus_single_numerator"] - group["full_minus_single_interaction_gain"]
            group["full_minus_omit_numerator"] = full["numerator"] - omit["numerator"]
            root_value = details[str(root)]["horizons"][h]
            original = reports[arm]["panels"]["physical_nonzero"]["horizons"][h]["comparisons"]["all_vs_uniform"]
            root_original = root_value["original_endpoint_comparisons"]["all_vs_uniform"]
            remaining_num = full["numerator"] - root_value["strategies"]["all_arrivals"]["numerator"]
            remaining_den = denominator - root_value["truth_squared_norm"]
            group["full_panel_endpoint_mse_gain"] = original["paired_endpoint_mse_gain_mean"]
            group["leave_influential_root_out"] = {"numerator": remaining_num,
                "signed_recovery": remaining_num / remaining_den,
                "interaction_gain": full["interaction_gain"] - root_value["strategies"]["all_arrivals"]["interaction_gain_vs_uniform"],
                "endpoint_mse_gain": (len(seeds) * original["paired_endpoint_mse_gain_mean"]
                    - root_original["paired_endpoint_mse_gain"]) / (len(seeds) - 1)}
            loo_n = [full["numerator"] - value["strategies"]["all_arrivals"]["numerator"] for value in values]
            loo_gain = [full["interaction_gain"] - value["strategies"]["all_arrivals"]["interaction_gain_vs_uniform"] for value in values]
            group["all_leave_one_out"] = {"positive_numerator_count": sum(value > 0 for value in loo_n),
                "positive_gain_count": sum(value > 0 for value in loo_gain), "number": len(values)}
            aggregate[h] = group
        result["arms"][arm] = {"aggregate": aggregate, "root": details[str(root)], "by_seed": details}
    for h in (1, 4):
        cross = {}
        for weight_arm in ARMS:
            weights = cases[weight_arm][root]["supports"]["all_arrivals"]
            q = weights[0] - weights[1]
            for family_arm in ARMS:
                saved = cases[family_arm][root]
                path, target = saved["future_responses"].double(), saved["true_future"].double()
                truth = target[0, 0, h - 1] - target[0, 1, h - 1] - target[1, 0, h - 1] + target[1, 1, h - 1]
                y = path[0, 0, :, h - 1] - path[0, 1, :, h - 1]
                cross[f"weights_{weight_arm}_family_{family_arm}"] = float((q @ y) @ truth)
        result["crossed_root_full_numerators"][str(h)] = cross
    batch = torch.load(OUT / "dev.pt", weights_only=True)
    index = batch["seed"].tolist().index(root)
    with np.load(OUT / "data_dev.npz", allow_pickle=False) as raw:
        actions = raw["actions"][index].copy()
    with np.load(OUT / "diag_dev.npz", allow_pickle=False) as raw:
        pos, speed = raw["positions"][index].copy(), raw["speed"][index].copy()
        actual = np.diff(pos[:, :, 10:21].astype(np.float64), axis=2)
        expected = np.clip(actions[:, :, 2:4], -1, 1).reshape(2, 2, 10, 2).astype(np.float64) * speed[:, :, None, None]
        result["root_physics"] = {"speeds": speed[:, 0].tolist(),
            "source_raw_primitives": actions[0, 0, 2].tolist(),
            "reverse_raw_primitives": actions[0, 0, 3].tolist(),
            "source_displacement": (pos[:, 0, 15] - pos[:, 0, 10]).tolist(),
            "reverse_displacement": (pos[:, 0, 20] - pos[:, 0, 15]).tolist(),
            "future_displacement": (pos[:, 0, 35] - pos[:, 0, 30]).tolist(),
            "source_root": pos[:, 0, 10].tolist(),
            "reverse_is_exact_primitive_time_reversal": bool(np.array_equal(pos[:, :, 15:21], pos[:, :, 10:16][:, :, ::-1])),
            "root_return_exact": bool(np.array_equal(pos[:, :, 10], pos[:, :, 20])),
            "reverse_raw_is_exact_reversed_negative": bool(np.array_equal(actions[:, :, 3], -actions[:, :, 2, ::-1])),
            "source_reverse_collision_mask_true_count": int(raw["collision"][index, :, :, 2:4].sum()),
            "source_reverse_collision_mask_entries": int(raw["collision"][index, :, :, 2:4].size),
            "terminated_or_truncated": bool(raw["terminated"][index, :, :, 2:4].any() or raw["truncated"][index, :, :, 2:4].any()),
            "max_actual_minus_commanded_step_abs_error": float(np.abs(actual - expected).max())}
    result["root_relative_log_likelihoods"] = {
        arm: [[event["weights"]["all_arrivals"]["relative_log_likelihood"] for event in trace[:2]]
              for trace in rows[arm][root]["actual_write_traces"]] for arm in ARMS}
    result["closure_errors"] = {"max_write_sum_minus_full_numerator_abs": largest_closure,
        "max_recomputed_minus_original_score_abs": largest_score_error}
    assert largest_closure < 1e-10 and largest_score_error < 1e-10
    for arm in ARMS:
        zero = [row for row in reports[arm]["seeds"] if not row["horizons"]["1"]["physical_nonzero"]]
        assert len(zero) == 3
        for h in ("1", "4"):
            for row in zero:
                assert row["horizons"][h]["interaction"]["uniform"]["truth_squared_norm"] == 0
                assert all(row["horizons"][h]["interaction"][method]["prediction_truth_dot"] == 0
                           for method in METHODS)
    result["physical_zero_roots_add_exactly_zero_signed_numerator_and_denominator"] = True
    physics = result["root_physics"]
    assert (physics["reverse_is_exact_primitive_time_reversal"] and physics["root_return_exact"]
            and physics["reverse_raw_is_exact_reversed_negative"]
            and physics["source_reverse_collision_mask_true_count"] == 0
            and not physics["terminated_or_truncated"]
            and physics["max_actual_minus_commanded_step_abs_error"] == 0)
    return result


def render(result):
    lines = ["# G3 fixed-development influence accounting", "",
        "The dual failure is concentrated in reset **2026112128**: its collision-free inverse-action arrival changes the support in a direction that is harmful for that arm's frozen future response family. The two zero-action arrivals contribute exactly zero. This is a post hoc diagnostic of the original 39-root physical-nonzero panel; the full-panel failure remains the result.", "",
        "Let `d` be the actual slow-minus-fast movement-minus-stop interaction, `Y_k` the frozen native movement-minus-stop response, and `q=w_slow-w_fast`. The signed numerator is `N=(qY)·d` in **latent squared units**; signed recovery is `N/||d||²`. Interaction SSE gain is `2N-||qY||²`. Endpoint MSE divides by 192 coordinates and averages the four speed/program cells, so it is a different quantity. Uniform gives exactly zero paired interaction because each arm's entire future family is exactly shared across speeds.", "",
        "## Closed signed-numerator accounting", "",
        "All 39 primary roots; the common `sum ||d||²` is 2263.384856 at H1 and H4. The other three matched roots have exactly zero truth interaction, so all signed-numerator totals also apply to all 42 matched roots. The original six exclusions remain unchanged. Single means t2 only; full means t2+t3+t4+t5; omit skips t2.", "",
        "| Arm | H | Single N | Inverse t3 increment | Zero t4+t5 | Full N | Omit N | Full−omit N |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for h in ("1", "4"):
            row = result["arms"][arm]["aggregate"][h]
            s, f, o = (row["strategies"][name]["numerator"] for name in METHODS)
            lines.append(f"| {arm} | {h} | {s:+.6f} | {row['reverse_increment_with_source']:+.6f} | {row['zero_increment_with_source']:.0f} | {f:+.6f} | {o:+.6f} | {row['full_minus_omit_numerator']:+.6f} |")
    lines += ["", "The first source still helps after the inverse: full−omit is positive for both arms/horizons. Full−single equals the t3 increment exactly; the zero arrivals preserve all public weights bitwise. For dual, full−single SSE gain closes as **−5.540944−0.141115=−5.682060 (H1)** and **−3.631603−0.043897=−3.675500 (H4)**: twice the numerator increment minus the change in squared prediction norm. Across all roots/arms, maximum write-sum closure error is 4.00e-15 and maximum disagreement with original saved scores is 3.45e-14.", "",
        "## Influential reset and leave-one-out sensitivity", "",
        "Reset 2026112128 is the largest absolute dual full-vs-uniform SSE contributor at both horizons. It carries only 202.301775/2263.384856 = 8.94% of the truth-energy denominator.", "",
        "| Arm | H | Root single N | Root inverse ΔN | Root full N | Root omit N | Root full SSE gain |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for h in ("1", "4"):
            r = result["arms"][arm]["root"]["horizons"][h]
            s, f, o = (r["strategies"][name] for name in METHODS)
            lines.append(f"| {arm} | {h} | {s['numerator']:+.6f} | {r['full_minus_single_numerator']:+.6f} | {f['numerator']:+.6f} | {o['numerator']:+.6f} | {f['interaction_gain_vs_uniform']:+.6f} |")
    lines += ["", "| Arm | H | Full N, all 39 → other 38 | SSE gain, all → other 38 | Paired endpoint MSE gain, all → other 38 |",
        "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for h in ("1", "4"):
            r = result["arms"][arm]["aggregate"][h]
            f, loo = r["strategies"]["all_arrivals"], r["leave_influential_root_out"]
            lines.append(f"| {arm} | {h} | {f['numerator']:+.6f} → {loo['numerator']:+.6f} | {f['interaction_gain']:+.6f} → {loo['interaction_gain']:+.6f} | {r['full_panel_endpoint_mse_gain']:+.8f} → {loo['endpoint_mse_gain']:+.8f} |")
    lines += ["", "Only **1/39** dual leave-one-out panels has a positive full numerator and SSE gain at either horizon: the one removing this reset. Dual full signed recovery changes −0.0650% → +0.0937% at H1 and −0.0284% → +0.0975% at H4. Every H1-arm leave-one-out panel stays positive. Removing this reset is sensitivity analysis, not an exclusion rule or a revised gate.", "",
        "## Actual source and inverse; fixed future projection", "",
        "The stored speeds are 2.5 and 7.5. The source moves left by `[-4.21875,0]` / `[-12.65625,0]`; the inverse moves right by exactly the opposite amounts. All 40 stored t2/t3 collision-mask entries are false, termination/truncation are false, and the maximum actual-minus-commanded primitive displacement error is 0. The inverse position sequence exactly time-reverses the source and returns to `[99.2294693,86.9803238]`. The mask means a displacement discrepancy greater than 1e-5; this trace shows no collision-induced rejection/clipping. Future movement is orthogonal downward by `[0,4.21875]` / `[0,12.65625]`. These are environment XY units, not latent units.", "",
        "Dual supports in prototype order `[0,1,2,3]`:", "",
        "| Speed | After source t2 | After inverse t3 (= after both zeros) |",
        "|---|---|---|"]
    supports = result["arms"]["dual"]["root"]["supports"]
    for i, name in enumerate(("2.5", "7.5")):
        format_vector = lambda values: "[" + ", ".join(f"{value:.5f}" for value in values) + "]"
        lines.append(f"| {name} | {format_vector(supports['single_source'][i])} | {format_vector(supports['all_arrivals'][i])} |")
    lines += ["", "At t3, relative log likelihoods versus prototype 0 are `[0,-0.870923,+2.141343,-1.646090]` for speed 2.5 and `[0,-0.076097,-0.032097,-0.238433]` for speed 7.5 (nats). Prototype 2 receives 0.839671 slow-speed support, but its future truth-direction coefficient is negative.", "",
        "Define `c_k=(Y_k-mean_j Y_j)·d/||d||`. These coefficients are in **latent units**; `N=||d|| q·c`, with `||d||=14.223283` at this reset. Centering removes only a common response, since `sum q=0`.", "",
        "| Frozen family | H | c[0], c[1], c[2], c[3] |",
        "|---|---:|---|"]
    for arm in ARMS:
        for h in ("1", "4"):
            c = result["arms"][arm]["root"]["horizons"][h]["unit_truth_family_coefficients"]
            lines.append(f"| {arm} | {h} | " + ", ".join(f"{value:+.6f}" for value in c) + " |")
    lines += ["", "Dual slow/fast TV separation grows **0.030472 → 0.557288** (overlap 0.969528 → 0.442712). The cosine between the support contrast and `c` changes **+0.642 → −0.760** at H1 and **+0.543 → −0.816** at H4. Thus the failure is a wrong projection direction, rather than indistinguishable supports. The H1 arm's full contrast remains positively aligned (cosines +0.082/+0.440).", "",
        "As a coordinate-level diagnostic, swapping the saved full supports and saved families gives:", "",
        "| Support source | Family source | H1 N | H4 N |",
        "|---|---|---:|---:|"]
    for weight_arm in ARMS:
        for family_arm in ARMS:
            key = f"weights_{weight_arm}_family_{family_arm}"
            lines.append(f"| {weight_arm} | {family_arm} | {result['crossed_root_full_numerators']['1'][key]:+.6f} | {result['crossed_root_full_numerators']['4'][key]:+.6f} |")
    lines += ["", "Both crossed combinations are positive; the negative result occurs for the learned dual-support/dual-family combination. This localizes an alignment failure in their combination. Across arms, prototype indices need not identify the same physical explanation, so the swaps do not isolate a physical causal mechanism or prove either component correct.", "",
        "## Absolute endpoints and limits", "",
        "At this root, dual H1 moving endpoint MSE (slow/fast) changes **0.479761/0.519981 → 0.496114/0.517772** from uniform to full; H4 changes **0.717614/0.587554 → 0.732012/0.586135**. The slow-speed deterioration dominates the fast-speed improvement. Stop MSE remains exactly **0.110611 (H1)** / **0.344715 (H4)** for all supports; that shared frozen prediction error cancels out of the interaction. The root's paired full-vs-uniform MSE gains are **−0.00353611/−0.00324490** at H1/H4.", "",
        "The separate [G1 adaptive continuous-reference replay](G1_G3_SIGNED_REFERENCE.md) preserves every matched root's H1/H4 sign and the aggregate signs. For the dual model, maximum informative-arrival log-Bayes-factor error is 4.41e-13 nat and maximum prefix-posterior TV is 9.17e-14. The observed sign failure survives replacing the fixed integration rule with the independent reference.", "",
        "This establishes exact attribution within the saved computation: an actual, physically reversible inverse observation changed support, and its projection on the fixed future family produced the negative term. It does not establish which unlabeled prototype is physically true, why optimization learned this mismatch, or a general failure rate. The influential reset was identified after seeing development results; no training, parameter, data exclusion, G4, or P4 was changed. Detailed vectors, every root, and closure checks are in `g3_influence.json`; regenerate with `python -m research.p1a.prbd_consistency_influence`.", ""]
    return "\n".join(lines)


def main():
    result = analyze()
    (output_path(OUT / "g3_influence.json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_path(OUT / "g3_influence.md")).write_text(render(result), encoding="utf-8")
    print(json.dumps({"seed": result["influential_seed"], "closure_errors": result["closure_errors"],
                      "physics_max_displacement_error": result["root_physics"]["max_actual_minus_commanded_step_abs_error"],
                      "physics_collision_mask_true_count": result["root_physics"]["source_reverse_collision_mask_true_count"]}, indent=2))


if __name__ == "__main__":
    main()
