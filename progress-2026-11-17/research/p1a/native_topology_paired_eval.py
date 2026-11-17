"""Compare fixed and renewed H5 action banks before any closed-loop study."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .native_topology_condition import load_readout, load_trained
from .native_topology_paired_data import clipping
from .native_topology_paired_fit import ARMS, RUN
from .native_topology_prefix_data import query_bank, replay_bank
from .native_topology_prefix_eval import (
    REFERENCE, RUN as PREFIX_RUN, SEEDS, endpoint_report, factual_masks, geometry, load_panels,
    predict_bank, predict_local, progress_report, ranking_report, reduce_ranks, run_arm, scores, selected_measures,
)
from .physical_topology_gate import ACTION_BLOCK, ITERATIONS, SAMPLES, PhysicalRollout, VisibleMapCost, make_scene, room


def first_command_report(actions, predicted_cost, actual_cost):
    rows = []
    for layout, candidates in enumerate(actions):
        for root, plans in enumerate(candidates):
            commands, group = np.unique(plans[:, 0].reshape(len(plans), -1), axis=0, return_inverse=True)
            for mode in (0, 1):
                prediction = predicted_cost[layout, root, :, mode]
                physical = actual_cost[layout, root, :, mode]
                selected = int(prediction.argmin())
                actual_group = np.full(len(commands), np.inf)
                np.minimum.at(actual_group, group, physical)
                winner = group[selected]
                rows.append({
                    "layout": int(SEEDS[layout]), "navigation_root": root, "mode": mode,
                    "candidate": selected, "first_command": plans[selected, 0].tolist(),
                    "distinct_commands": len(commands),
                    "first_command_regret": float(actual_group[winner] - actual_group.min()),
                    "suffix_regret": float(physical[selected] - actual_group[winner]),
                })
    average = {key: float(np.mean([row[key] for row in rows])) for key in ("first_command_regret", "suffix_regret")}
    return {**average, "zero_first_command_regret": sum(row["first_command_regret"] == 0 for row in rows), "rows": rows}


def renew_bank(bank, proposal_model, decode):
    renewed = {key: bank[key].copy() for key in ("seed", "root_z", "root_xy", "root_action", "action", "future_xy")}
    parent_xy = np.zeros_like(bank["parent_xy"])
    parent_cost = np.zeros_like(bank["parent_cost"])
    for i, seed in enumerate(bank["seed"]):
        scene = make_scene(int(seed))
        physical = PhysicalRollout(scene)
        for root, goal in ((2, scene.southeast), (3, scene.northwest)):
            actions, xy, cost, _ = query_bank(
                scene, VisibleMapCost(scene, goal), root, bank["root_z"][i, root],
                bank["root_action"][i, root], bank["action"][i, root, :4], proposal_model, decode,
            )
            renewed["action"][i, root] = actions
            renewed["future_xy"][i, root] = replay_bank(physical, bank["root_xy"][i, root, -1], actions)
            parent_xy[i, root], parent_cost[i, root] = xy, cost
    return renewed, parent_xy, parent_cost


def nav_result(bank, estimated_xy, estimated_cost):
    y = bank["future_xy"][..., -1, :]
    physical_cost = scores(bank, y)
    p, t = estimated_cost[:, 2:, :, :, -1], physical_cost[:, 2:, :, :, -1]
    goals = np.broadcast_to(np.array([2, 0]), bank["root_xy"][:, 2:].shape[:2])
    ranking = ranking_report(p, t, estimated_xy[:, 2:, :, :, -1], y[:, 2:, :, :, -1], goals)
    return {"plans": reduce_ranks(ranking), "first_commands": first_command_report(bank["action"][:, 2:], p, t)}


def live_gate_report(bank, proposal_xy, proposal_cost):
    """Evaluate only this checkpoint's four complete CEM rounds at fixed H5 roots."""
    actions = bank["action"][:, 2:, 4:]
    traces = bank["future_xy"][:, 2:, 4:]
    xy, cost = proposal_xy[:, 2:, 4:], proposal_cost[:, 2:, 4:]
    assert actions.shape[:3] == (16, 2, ITERATIONS * SAMPLES)
    assert np.isfinite(xy).all() and np.isfinite(cost).all()
    rows, blocked_rows = [], []
    for i, seed in enumerate(bank["seed"]):
        scene = make_scene(int(seed))
        for j, goal in enumerate((scene.southeast, scene.northwest)):
            root = "northwest" if j == 0 else "southeast"
            start = bank["root_xy"][i, j + 2, -1]
            raw, paths = actions[i, j], traces[i, j]
            terminal = paths[..., -1, -1, :]
            predicted, learned = xy[i, j], cost[i, j]
            score = VisibleMapCost(scene, goal)
            np.testing.assert_array_equal(score(predicted), learned)
            truth = score(terminal)
            group, ids = np.unique(raw[:, 0].reshape(len(raw), -1), axis=0, return_inverse=True)
            same_throughout = (paths[:, 0] == paths[:, 1]).all(axis=(1, 2, 3))
            clipped = clipping(start, raw, paths)
            blocked = same_throughout & clipped[:, 0, -ACTION_BLOCK:].any(axis=1)
            # Use each plan's true H4 predecessor in both projections, as in the
            # historical fixed-bank geometry. Do not compare checkpoints on different actual origins.
            direction = raw[:, -1].sum(axis=1)
            direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
            before = paths[..., -2, -1, :]
            p = ((predicted - before) * direction[:, None]).sum(axis=-1)
            t = ((terminal - before) * direction[:, None]).sum(axis=-1)
            blocked_rows.append((int(seed), p[blocked].astype(np.float64), t[blocked].astype(np.float64)))
            predicted_goal = room(predicted) == room(goal)
            physical_goal = room(terminal) == room(goal)
            for mode in (0, 1):
                selected = int(learned[:, mode].argmin())
                group_min = np.full(len(group), np.inf)
                np.minimum.at(group_min, ids, truth[:, mode])
                first, suffix = group_min[ids[selected]], truth[selected, mode] - group_min[ids[selected]]
                rows.append(dict(seed=int(seed), root=root, mode=mode, selected_candidate=selected,
                                 predicted_goal_room=bool(predicted_goal[selected, mode]),
                                 physical_goal_room=bool(physical_goal[selected, mode]),
                                 physical_goal_available=bool(physical_goal[:, mode].any()),
                                 false_goal_room=bool(predicted_goal[selected, mode] and not physical_goal[selected, mode]),
                                 physical_best_candidate=int(truth[:, mode].argmin()),
                                 first_block_regret=float(first - group_min.min()), suffix_regret=float(suffix)))

    # Cluster by scene: both navigation roots and both mode predictions share the same layout.
    seed_index = np.searchsorted(bank["seed"], [r[0] for r in blocked_rows])
    counts, p_sum, t_sum = [np.zeros(len(bank["seed"])) for _ in range(3)]
    for k, (_, p, t) in zip(seed_index, blocked_rows, strict=True):
        counts[k] += p.size
        p_sum[k] += p.sum()
        t_sum[k] += t.sum()
    common = {"plans": int(counts.sum()) // 2, "historical_physical_px": 3.04,
              "max_absolute_bias_px": 1., "pass": False,
              "layouts": [dict(seed=int(seed), plans=int(n) // 2,
                               forecast_minus_physical_px=float((p - t) / n) if n else None)
                          for seed, n, p, t in zip(bank["seed"], counts, p_sum, t_sum, strict=True)]}
    if counts.any():
        draw = np.random.default_rng(11752026).integers(0, len(counts), (10000, len(counts)))
        n = counts[draw].sum(axis=1)
        keep = n > 0
        gap = (p_sum - t_sum)[draw[keep]].sum(axis=1) / n[keep]
        bias = float((p_sum - t_sum).sum() / counts.sum())
        common |= dict(forecast_px=float(p_sum.sum() / counts.sum()), physical_px=float(t_sum.sum() / counts.sum()),
                       forecast_minus_physical_px=bias,
                       gap_layout_bootstrap_95ci_px=np.quantile(gap, [.025, .975]).tolist())
        common["pass"] = abs(bias) <= common["max_absolute_bias_px"]
    assert len(rows) == 64
    false = sum(row["false_goal_room"] for row in rows)
    strata = {}
    for j, root in enumerate(("northwest", "southeast")):
        strata[root] = {}
        for mode in (0, 1):
            p, t = [np.concatenate([row[k][:, mode] for row in blocked_rows[j::2]]) for k in (1, 2)]
            group = [row for row in rows if row["root"] == root and row["mode"] == mode]
            strata[root][mode] = dict(queries=len(group), false_goal_room=sum(row["false_goal_room"] for row in group),
                                      common_blocked_plans=len(p), forecast_px=float(p.mean()) if len(p) else None,
                                      physical_px=float(t.mean()) if len(t) else None)
    return {"queries": len(rows), "candidates_per_root": ITERATIONS * SAMPLES,
            "selected_false_goal_room": false, "false_goal_room_pass": false <= 6,
            "common_final_blocked_projection": common, "by_direction_and_mode": strata, "rows": rows}


def run_live_gate(args):
    _, bank = load_panels(PREFIX_RUN)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_trained(args.checkpoint, "conditioned", device)
    decode = load_readout(device, REFERENCE)
    args.output.mkdir(parents=True, exist_ok=False)
    live, xy, cost = renew_bank(bank, model, decode)
    result = live_gate_report(live, xy, cost)
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["archived_bank_live_start_slot"] = 4
    nav = {key: value[:, 2:] if key != "seed" else value for key, value in live.items()}
    np.savez_compressed(args.output / "live_bank.npz", **nav, proposal_xy=xy[:, 2:], proposal_cost=cost[:, 2:])
    with (args.output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(args.output / "summary.json", flush=True)


def save(filename, value):
    with (RUN / filename).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def gate_intervention(device, small, decode, masks):
    model = load_trained(RUN / "paired", "conditioned", device)
    free, exposure, root_c = predict_bank(*model, decode, small, force=1.0)
    truth = small["future_xy"][..., -1, :]
    with np.load(RUN / "evaluation_paired_arrays.npz", allow_pickle=False) as stored:
        actual, local, ordinary = (stored[name] for name in ("C", "L", "F"))
    arms = {}
    for name, xy in (("ordinary", ordinary), ("forced_one", free)):
        geo = geometry(small, xy, root_c)
        unique = masks["unique_slot"]
        arms[name] = {
            "visible_collision_branch_count": int((geo["visible_wall_depth"][unique] > 0).any(axis=-1).sum()),
            "true_mode_collision_branch_count": int((geo["true_mode_wall_depth"][unique] > 0).any(axis=-1).sum()),
            "wrong_room_branch_count": int(geo["wrong_room"][unique].any(axis=-1).sum()),
            "endpoints": endpoint_report(actual, local, xy, truth, unique[..., None, None]),
            "progress": progress_report(actual, local, xy, truth, masks, {
                "Y": scores(small, truth), "C": scores(small, actual),
                "L": scores(small, local), "F": scores(small, xy),
            }),
        }
    np.savez_compressed(RUN / "gate_forced_arrays.npz", F=free, exposure=exposure, bank_index=small["bank_index"])
    save("gate_intervention.json", arms)


def evidence_slices(device, small, decode, scale, masks):
    valid, effect = masks["eligible"], masks["end_change"]
    result = {}
    for arm in ("prefix", *ARMS):
        source = (PREFIX_RUN if arm == "prefix" else RUN) / arm
        model = load_trained(source, "conditioned", device)
        terms = {}
        _, _, exposure, gaps = predict_local(*model, decode, scale, small, evidence_terms=terms)
        midpoint, signal = terms["midpoint_projection_nat"], terms["true_gap_projection_nat"]
        estimated = np.stack((signal - midpoint, signal + midpoint), axis=-1)
        np.testing.assert_allclose(estimated[valid], gaps[valid], rtol=2e-4, atol=0.1)
        result[arm] = {}
        for name, group in (("block_end_equal", valid & ~effect), ("block_end_distinct", valid & effect)):
            rows = []
            small_gap = terms["prediction_gap_px"] < 2
            large_residual = terms["midpoint_residual_px"] >= 10
            for small_value in (True, False):
                for residual_value in (False, True):
                    cell = group & (small_gap == small_value) & (large_residual == residual_value)
                    magnitude = np.abs(midpoint[cell])
                    rows.append({
                        "decoded_gap_below_2px": small_value, "decoded_residual_at_least_10px": residual_value,
                        "n": int(cell.sum()), "nonzero_evidence": int((gaps[cell] != 0).any(axis=-1).sum()),
                        "wrong_on_either_arrival": int((gaps[cell] < 0).any(axis=-1).sum()) if name == "block_end_distinct" else None,
                        "max_abs_midpoint_projection_nat": float(magnitude.max()) if len(magnitude) else None,
                    })
            result[arm][name] = rows
        np.savez_compressed(RUN / f"evidence_projection_{arm}.npz", **terms, signed_gaps=gaps,
                            exposure=exposure, seed=small["seed"], bank_index=small["bank_index"], **masks)
    save("evidence_slices.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--gate", action="store_true", help="Only rerun the short pool with mode exposure forced to one")
    diagnostic.add_argument("--evidence", action="store_true", help="Record the components of the fixed-scale evidence")
    diagnostic.add_argument("--checkpoint", type=Path, help="Renew live H5 CEM with this conditioned.pt directory")
    parser.add_argument("--output", type=Path, help="New result directory, required with --checkpoint")
    parser.add_argument("--device", default="cuda", help="Device for checkpoint live evaluation")
    options = parser.parse_args()
    if options.checkpoint:
        if options.output is None:
            parser.error("--checkpoint requires --output")
        run_live_gate(options)
        return
    torch.set_num_threads(1)
    device = torch.device("cuda")
    small, bank = load_panels(PREFIX_RUN)
    decode = load_readout(device, REFERENCE)
    scale = torch.load(REFERENCE / "likelihood_scale.pt", map_location=device, weights_only=True)
    masks = factual_masks(small)
    if options.gate:
        gate_intervention(device, small, decode, masks)
        return
    if options.evidence:
        evidence_slices(device, small, decode, scale, masks)
        return
    masks["bank_common_corner"] = bank["common_corner"]
    truth_cost = scores(bank, bank["future_xy"][..., -1, :])
    goal_room = np.broadcast_to(np.array([2, 0, 2, 0]), bank["root_xy"].shape[:2])
    fixed = {}
    for arm in ("prefix", *ARMS):
        if arm != "prefix":
            print(f"Fixed bank: {arm}", flush=True)
            report, per_layout = run_arm(RUN, arm, device, decode, scale, small, bank, masks, truth_cost, goal_room)
            fixed[arm] = {"all": report, "layout_measures": [selected_measures(row) for row in per_layout]}
        array_run = PREFIX_RUN if arm == "prefix" else RUN
        with np.load(array_run / f"evaluation_{arm}_arrays.npz", allow_pickle=False) as stored:
            fixed.setdefault(arm, {})["navigation"] = nav_result(bank, stored["F_bank"], stored["F_bank_cost"])
    save("fixed_evaluation.json", fixed)

    # Each proposal bank uses the same roots, scalar cost, 4 x 96 budget and RNG seeds.
    # It remains fixed while the two predictors independently score it.
    cross = {}
    for proposer in ARMS:
        print(f"Renewed bank from {proposer}", flush=True)
        model = load_trained(RUN / proposer, "conditioned", device)
        live_bank, proposed_xy, proposed_cost = renew_bank(bank, model, decode)
        del model
        np.savez_compressed(RUN / f"bank_{proposer}.npz", **live_bank, proposal_xy=proposed_xy, proposal_cost=proposed_cost)
        cross[proposer] = {}
        for scorer in ARMS:
            model = load_trained(RUN / scorer, "conditioned", device)
            xy, _, _ = predict_bank(*model, decode, live_bank)
            del model
            cost = scores(live_bank, xy)
            if scorer == proposer:
                np.testing.assert_allclose(cost[:, 2:, :, :, -1], proposed_cost[:, 2:], rtol=2e-4, atol=2e-4)
            # Compare the 384 live proposals. The public route anchors are common diagnostics.
            nav_bank = {**live_bank, "action": live_bank["action"][:, :, 4:], "future_xy": live_bank["future_xy"][:, :, 4:]}
            cross[proposer][scorer] = nav_result(nav_bank, xy[:, :, 4:], cost[:, :, 4:])
            np.savez_compressed(RUN / f"pred_{proposer}_scored_{scorer}.npz", xy=xy[:, 2:], cost=cost[:, 2:])
    save("renewed_evaluation.json", cross)


if __name__ == "__main__":
    main()
