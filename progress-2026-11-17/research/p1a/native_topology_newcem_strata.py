"""Stratify archived paired/paired 384-candidate CEM results on CPU."""

import csv
import json
from pathlib import Path

import numpy as np
from stable_worldmodel.envs.four_room.env import FourRoomEnv

from .native_topology_writein_gate import geometry_flags, raw_stock
from .physical_topology_gate import (
    DOOR_SIZE, HALF_WALL, LOCK, RADIUS, SPEED, PhysicalRollout, VisibleMapCost, make_scene, room,
)


RUN = Path(__file__).resolve().parents[2] / "runs/native_topology_paired_20261115"
OUTPUT = RUN / "writein_attribution_20261116/decision_root_followup"
WALL_FLAGS = ("static_occupancy", "static_skip", "lock_only", "invalid_static_prior", "border_occupancy")
POLICIES = ("learned_uniform", "learned_true", "physical_B_learned_D_true",
            "learned_B_physical_D_true", "physical_uniform", "physical_true")


def read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return dict(source)


def delta_rows(row, label, a, b, learned_cost, actual_cost, learned_xy, actual_xy):
    for name, cost, xy in (("learned", learned_cost, learned_xy), ("actual", actual_cost, actual_xy)):
        row[f"{label}_delta_B_{name}"] = float(cost[a].mean() - cost[b].mean())
        d_a, d_b = np.diff(cost[[a, b]], axis=1).squeeze(axis=1)
        row[f"{label}_delta_D_{name}"] = float(d_a - d_b)
        gap = xy[a, 1, -1] - xy[a, 0, -1] - xy[b, 1, -1] + xy[b, 0, -1]
        row[f"{label}_delta_D_x_{name}"], row[f"{label}_delta_D_y_{name}"] = map(float, gap)


def summarize(rows):
    summary = {"n": len(rows), "n_layouts": len({row["seed"] for row in rows})}
    if not rows:
        return summary
    a = lambda key: np.array([row[key] for row in rows])
    summary["choices"] = {
        name: {**{f"mean_{metric}": float(a(f"{name}_{metric}").mean())
                  for metric in ("first_block_regret", "suffix_regret", "plan_regret", "actual_cost")},
               "actual_goal_room": int(a(f"{name}_actual_goal_room").sum())}
        for name in POLICIES
    }
    for name in ("learned_true", "physical_true"):
        same = a(f"{name}_first_block") == a(f"{name.split('_')[0]}_uniform_first_block")
        summary[f"{name}_uniform_comparison"] = {"changed_first_block": int((~same).sum())}
        for metric in ("first_block_regret", "plan_regret"):
            gain = a(f"{name.split('_')[0]}_uniform_{metric}") - a(f"{name}_{metric}")
            summary[f"{name}_uniform_comparison"][metric] = {
                "mean_uniform_minus_true": float(gain.mean()), "improved": int((gain > 0).sum()),
                "worse": int((gain < 0).sum()), "tied": int((gain == 0).sum()),
            }
    for name in ("selected_same_plan_both_false", "selected_same_plan_both_actual_miss", "selected_xy_H5_actual_modes_equal",
                 "selected_all_raw_actual_modes_equal", "selected_both_modes_static_geometry", "selected_forecast_goal_room",
                 "selected_actual_goal_room", "selected_false_target_other_mode", "selected_static_geometry_any_h", *WALL_FLAGS):
        summary[name] = int(a(name).sum())
    for name in ("selected_forecast_cost", "selected_actual_cost", "selected_terminal_actual_rank",
                 "selected_endpoint_error", "selected_h5_midpoint_error", "selected_h5_D_xy_error"):
        summary[f"mean_{name}"] = float(a(name).mean())
    for prefix in ("uniform_vs_physical_true", "learned_true_vs_physical_true"):
        eligible = a(prefix + "_physical_first_gap") > 0
        part = {"strict_physical_first_gap": int(eligible.sum())}
        for metric in ("B", "D_cost", "D_xy"):
            if metric == "D_xy":
                predicted, actual = (np.stack([a(f"{prefix}_delta_D_{axis}_{arm}")[eligible] for axis in ("x", "y")], axis=1)
                                     for arm in ("learned", "actual"))
            else:
                name = "B" if metric == "B" else "D"
                predicted, actual = (a(f"{prefix}_delta_{name}_{arm}")[eligible, None] for arm in ("learned", "actual"))
            pnorm, anorm = (np.linalg.norm(v, axis=1) for v in (predicted, actual))
            active = anorm > 0
            dot = (predicted * actual).sum(axis=1)
            part[metric] = {"physical_nonzero": int(active.sum()), "aligned": int((dot > 0).sum()),
                            "reversed": int((dot < 0).sum()), "zero_prediction": int((active & (pnorm == 0)).sum()),
                            "orthogonal_nonzero": int((active & (pnorm > 0) & (dot == 0)).sum()),
                            "spurious_on_physical_zero": int((~active & (pnorm > 0)).sum()),
                            "median_physical_nonzero_magnitude": float(np.median(anorm[active])) if active.any() else None,
                            "median_amplitude_ratio_nonzero_truth": float(np.median(pnorm[active] / anorm[active])) if active.any() else None}
        d_aligned = a(prefix + "_delta_D_learned") * a(prefix + "_delta_D_actual") > 0
        b_reversed = a(prefix + "_delta_B_learned") * a(prefix + "_delta_B_actual") < 0
        delta = a(prefix + "_delta_B_learned") + (a("true_mode") - .5) * a(prefix + "_delta_D_learned")
        jointly = eligible & d_aligned & b_reversed
        part["D_aligned_B_reversed"] = int(jointly.sum())
        part["D_aligned_B_reversed_and_learned_true_prefers_higher_regret"] = int((jointly & (delta < 0)).sum())
        summary[prefix] = part
    return summary


def main():
    bank = read_npz(RUN / "bank_paired.npz")
    pred = read_npz(RUN / "pred_paired_scored_paired.npz")
    fixed = read_npz(RUN / "evaluation_paired_arrays.npz")
    reference = read_npz(RUN.parent / "native_topology_prefix_20261115/data_eval_bank.npz")
    lifecycle = read_npz(RUN / "writein_fixed_h5_banks.npz")
    historical = json.loads((RUN / "renewed_evaluation.json").read_text(encoding="utf-8"))["paired"]["paired"]
    for key in ("seed", "root_z", "root_xy", "root_action"):
        np.testing.assert_array_equal(bank[key], reference[key])
    np.testing.assert_array_equal(bank["seed"], fixed["seed"])
    np.testing.assert_allclose(fixed["root_C"][:, 2:], reference["decoded_current"][:, 2:], rtol=0, atol=2e-5)
    assert not np.intersect1d(bank["seed"], lifecycle["seed"]).size
    np.testing.assert_array_equal(bank["root_action"][:, 2:], 0)
    np.testing.assert_allclose(pred["cost"][..., -1], bank["proposal_cost"][:, 2:], rtol=2e-4, atol=2e-4)

    rows = []
    for i, seed in enumerate(bank["seed"]):
        scene = make_scene(int(seed))
        physical = PhysicalRollout(scene)
        env = FourRoomEnv(agent_speed=SPEED, agent_radius=RADIUS, wall_thickness=2 * HALF_WALL,
                          door_size=[DOOR_SIZE] * 4, door_positions=scene.door_positions,
                          close_door_prob=0.0, terminate_on_goal=False)
        env.reset(seed=scene.seed, options={"state": scene.northwest, "target_state": scene.southeast})
        assert env.closed_door is None and env.door_positions == scene.door_positions
        for root, root_name, start, goal in ((0, "northwest_to_southeast", scene.northwest, scene.southeast),
                                             (1, "southeast_to_northwest", scene.southeast, scene.northwest)):
            j = root + 2
            np.testing.assert_array_equal(bank["root_xy"][i, j], np.broadcast_to(start, (3, 2)))
            action = bank["action"][i, j, 4:]
            forecast_xy = pred["xy"][i, root, 4:]
            physical_raw = bank["future_xy"][i, j, 4:]
            physical_xy = physical_raw[..., -1, :]
            score = VisibleMapCost(scene, goal)
            np.testing.assert_array_equal(pred["cost"][i, root, 4:], score(forecast_xy))
            lc = pred["cost"][i, root, 4:, :, -1].astype(np.float64)
            tc = score(physical_xy[..., -1, :]).astype(np.float64)
            _, group = np.unique(action[:, 0].reshape(384, -1), axis=0, return_inverse=True)
            target = int(room(goal))
            reaches = room(physical_xy[..., -1, :]) == target
            predicts = room(forecast_xy[..., -1, :]) == target
            for mode in range(2):
                scores = (lc.mean(axis=1), lc[:, mode], tc.mean(axis=1) + (mode - .5) * np.diff(lc, axis=1)[:, 0],
                          lc.mean(axis=1) + (mode - .5) * np.diff(tc, axis=1)[:, 0], tc.mean(axis=1), tc[:, mode])
                candidates = [int(s.argmin()) for s in scores]
                u, c, _, _, _, best = candidates
                gt = tc[:, mode]
                min_by_group = np.full(group.max() + 1, np.inf)
                np.minimum.at(min_by_group, group, gt)
                false = predicts[c] & ~reaches[c]
                same_plan = physical_raw[c]
                record = dict(seed=int(seed), root_index=j, root=root_name, true_mode=mode,
                              root_x=float(start[0]), root_y=float(start[1]), goal_x=float(goal[0]), goal_y=float(goal[1]),
                              stratum="B" if false[mode] else "A", actual_post_support=None,
                              selected_same_plan_both_false=bool(false.all()), selected_false_target_other_mode=bool(false[1-mode]),
                              selected_same_plan_both_actual_miss=bool((~reaches[c]).all()),
                              selected_xy_H5_actual_modes_equal=bool(np.array_equal(physical_xy[c, 0, -1], physical_xy[c, 1, -1])),
                              selected_all_raw_actual_modes_equal=bool(np.array_equal(*same_plan)),
                              selected_forecast_goal_room=bool(predicts[c, mode]), selected_actual_goal_room=bool(reaches[c, mode]),
                              selected_forecast_cost=float(lc[c, mode]), selected_actual_cost=float(gt[c]),
                              selected_terminal_actual_rank=int(1 + (gt < gt[c]).sum()),
                              selected_endpoint_error=float(np.linalg.norm(forecast_xy[c, mode, -1] - physical_xy[c, mode, -1])),
                              selected_h5_midpoint_error=float(np.linalg.norm(forecast_xy[c, :, -1].mean(0) - physical_xy[c, :, -1].mean(0))),
                              selected_h5_D_xy_error=float(np.linalg.norm(np.diff(forecast_xy[c, :, -1], axis=0) - np.diff(physical_xy[c, :, -1], axis=0))))
                static_geometry = []
                for m in (mode, 1-mode):
                    before, factual = fixed["root_C"][i, j], start
                    per_block = []
                    for h, (command, end) in enumerate(zip(action[c], forecast_xy[c, m], strict=True)):
                        flags, *_ = geometry_flags(env, physical, before, end, command, m)
                        per_block.append(flags)
                        path, _ = raw_stock(env, physical, factual, command, LOCK[m])
                        np.testing.assert_array_equal(path, same_plan[m, h])
                        before, factual = end, path[-1]
                    if m == mode:
                        for k in WALL_FLAGS:
                            record[k] = any(block[k] for block in per_block)
                            record[f"{k}_blocks"] = " ".join(f"H{h+1}" for h, block in enumerate(per_block) if block[k])
                    static_geometry.append(any(block["static_occupancy"] or block["static_skip"] for block in per_block))
                    for name, xy, cost in (("learned", forecast_xy, lc), ("actual", physical_xy, tc)):
                        x, y = xy[c, m, -1].tolist()
                        record.update({f"selected_{name}_x_m{m}": x, f"selected_{name}_y_m{m}": y,
                                       f"selected_{name}_cost_m{m}": float(cost[c, m])})
                record["selected_static_geometry_any_h"] = static_geometry[0]
                record["selected_both_modes_static_geometry"] = all(static_geometry)
                for name, k in zip(POLICIES, candidates, strict=True):
                    first = min_by_group[group[k]] - gt[best]
                    suffix = gt[k] - min_by_group[group[k]]
                    record.update({f"{name}_slot": k + 4, f"{name}_first_block": int(group[k]),
                                   f"{name}_first_block_regret": float(first), f"{name}_suffix_regret": float(suffix),
                                   f"{name}_plan_regret": float(gt[k] - gt[best]), f"{name}_actual_cost": float(gt[k]),
                                   f"{name}_actual_goal_room": bool(reaches[k, mode])})
                    assert first + suffix == gt[k] - gt[best]
                for name, a in (("uniform_vs_physical_true", u), ("learned_true_vs_physical_true", c)):
                    record[f"{name}_physical_first_gap"] = float(min_by_group[group[a]] - gt[best])
                    delta_rows(record, name, a, best, lc, tc, forecast_xy.astype(np.float64), physical_xy.astype(np.float64))
                    for branch, costs in (("learned", lc), ("actual", tc)):
                        np.testing.assert_allclose(record[f"{name}_delta_B_{branch}"] + (mode - .5) * record[f"{name}_delta_D_{branch}"],
                                                   costs[a, mode] - costs[best, mode], rtol=0, atol=1e-10)
                saved = historical["first_commands"]["rows"][len(rows)]
                assert (seed, root, mode, c) == (saved["layout"], saved["navigation_root"], saved["mode"], saved["candidate"])
                np.testing.assert_array_equal(action[c, 0], np.array(saved["first_command"], dtype=np.float32))
                assert abs(record["learned_true_first_block_regret"] - saved["first_command_regret"]) < 1e-5
                assert abs(record["learned_true_suffix_regret"] - saved["suffix_regret"]) < 1e-5
                rows.append(record)

    count_b = sum(v["stratum"] == "B" for v in rows)
    assert len(rows) == 64 and count_b == historical["plans"]["selected_false_goal_room"] == 32
    summary = {"metadata": {"seed": bank["seed"].tolist(), "candidate_archive_slots": [4, 387],
                            "candidate_count": 384, "actual_post_support": None,
                            "lifecycle_seed_overlap": [], "A": "no false target", "B": "false target"},
               "all": summarize(rows)}
    for s in ("A", "B"):
        subset = [row for row in rows if row["stratum"] == s]
        summary[s] = summarize(subset)
        summary[s]["by_root_mode"] = {f"{root}/mode_{m}": summarize([r for r in subset if r["root"] == root and r["true_mode"] == m])
                                     for root in ("northwest_to_southeast", "southeast_to_northwest") for m in (0, 1)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "newcem_rows.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "newcem_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
