"""Measure the physical decision value of knowing the mode after a real PushT reversal."""

import argparse
import json
from pathlib import Path

import numpy as np

from native_reversible_pusht import (
    ACTION_BLOCK,
    HOLD,
    PUSH,
    RETREAT,
    ROOT,
    ReversiblePushT,
    Scene,
    candidate_actions,
    make_scene,
    replay,
)


OUTPUT = ROOT / "runs/native_reversible_delivery_20261115"
SCENES = range(55810, 55826)
GOAL_DISTANCE = 180.0
COST_TIE = 0.01
FLOAT_TOLERANCE = 1e-12
PHASES = ("initial_push", "retreat", "hold_1", "hold_2", "hold_3", "return", "recontact", "reversal_probe")
PREFIX = (PUSH, RETREAT, HOLD, HOLD, HOLD, PUSH, PUSH, PUSH)
CANDIDATES = ("hold", "quarter_push", "half_push", "three_quarter_push", "push", "rising_push", "falling_push")
POLICIES = ("known_mode", "uniform_modes", "stale_mode", "always_open", "always_braked")


def motion_state(env):
    return np.array([
        (*body.position, *body.velocity, body.angle, body.angular_velocity, *body.force, body.torque)
        for body in (env.agent, env.block)
    ])


def execute_path(env, block):
    frames, states, coverages, brake = [], [], [], []
    start_brake = env.brake_substeps

    def record(frame):
        frames.append(frame)
        states.append(env.get_obs())
        coverages.append(env._get_coverage())
        brake.append(env.brake_substeps - start_brake)

    env.execute(block, observe_step=record)
    return np.stack(frames), np.stack(states), np.asarray(coverages), np.asarray(brake)


def collect_scene(seed, blocks, live, fork):
    base = make_scene(seed)
    scene = Scene(base.reset, base.reset[2:].copy() + [GOAL_DISTANCE, 0.0, 0.0])
    old_mode = 1 - seed % 2
    mode = 1 - old_mode
    modes = np.array([old_mode] * (len(PREFIX) - 1) + [mode])
    history = list(zip(modes, PREFIX))

    first_rgb = live.start(scene, old_mode)
    goal_rgb = live.goal_image(scene)
    first_state, first_coverage = live.get_obs(), live._get_coverage()
    prefix = [execute_path(live, block) for block in PREFIX[:-1]]
    switch_rgb, switch_motion = live.observe(), motion_state(live)
    live.set_regime(mode)
    np.testing.assert_array_equal(live.observe(), switch_rgb)
    np.testing.assert_array_equal(motion_state(live), switch_motion)

    prefix.append(execute_path(live, PREFIX[-1]))
    actual_rgb, actual_state, actual_coverage, actual_brake = map(np.concatenate, zip(*prefix))
    root_rgb, root_motion = live.observe(), motion_state(live)

    replay(fork, scene, history[:-1], old_mode)
    np.testing.assert_array_equal(fork.observe(), switch_rgb)
    np.testing.assert_array_equal(motion_state(fork), switch_motion)
    stale_probe = execute_path(fork, PREFIX[-1])

    future = []
    for future_mode in (0, 1):
        rollouts = []
        for block in blocks:
            replay(fork, scene, history, future_mode)
            np.testing.assert_array_equal(fork.observe(), root_rgb)
            np.testing.assert_array_equal(motion_state(fork), root_motion)
            rollouts.append(execute_path(fork, block))
        future.append([np.stack(values) for values in zip(*rollouts)])
    future_rgb, future_state, future_coverage, future_brake = map(np.stack, zip(*future))
    return {
        "reset": scene.reset, "goal_pose": scene.goal, "goal_rgb": goal_rgb, "actual_mode": modes,
        "actual_rgb": np.concatenate((first_rgb[None], actual_rgb)),
        "actual_state": np.concatenate((first_state[None], actual_state)),
        "actual_coverage": np.concatenate(([first_coverage], actual_coverage)),
        "actual_brake_substeps_within_block": actual_brake,
        "switch_motion": switch_motion, "candidate_root_motion": root_motion,
        "stale_probe_rgb": stale_probe[0], "stale_probe_state": stale_probe[1],
        "stale_probe_coverage": stale_probe[2], "stale_probe_brake_substeps": stale_probe[3],
        "candidate_rgb": future_rgb, "candidate_state": future_state,
        "candidate_coverage": future_coverage, "candidate_brake_substeps": future_brake,
    }


def action_comparison(coverage, mode, threshold):
    cost = 1.0 - coverage[..., -1]
    oracle_minimum = cost.min(axis=1)
    regret_by_mode = np.maximum(cost - oracle_minimum[:, None], 0.0)
    actual_regret = regret_by_mode[mode]
    mean_regret = regret_by_mode.mean(axis=0)
    scores = (cost[mode], cost.mean(axis=0), cost[1 - mode], cost[0], cost[1])
    policies = {}
    for name, score in zip(POLICIES, scores):
        # The ordered menu resolves numerical ties; the interval explicitly keeps all near ties.
        index = int(np.flatnonzero(score <= score.min() + FLOAT_TOLERANCE)[0])
        near = score <= score.min() + COST_TIE + FLOAT_TOLERANCE
        policies[name] = {
            "choice": CANDIDATES[index], "near_optimal": list(np.asarray(CANDIDATES)[near]),
            "actual_regret": float(actual_regret[index]),
            "actual_regret_interval": [float(actual_regret[near].min()), float(actual_regret[near].max())],
            "mode_average_regret": float(mean_regret[index]),
            "mode_average_regret_interval": [float(mean_regret[near].min()), float(mean_regret[near].max())],
            "actual_terminal_success": bool(coverage[mode, index, -1] > threshold),
            "actual_any_step_success": bool(coverage[mode, index].max() > threshold),
        }
    left, right = np.triu_indices(len(CANDIDATES), 1)
    gaps = cost[:, left] - cost[:, right]
    pair_reversal = (gaps[0] * gaps[1] < 0) & (np.abs(gaps) > COST_TIE + FLOAT_TOLERANCE).all(axis=0)
    near_by_mode = cost <= oracle_minimum[:, None] + COST_TIE + FLOAT_TOLERANCE
    return {
        "open_and_braked_near_optimal_disjoint": not bool((near_by_mode[0] & near_by_mode[1]).any()),
        "strict_pair_reversals": [[CANDIDATES[i], CANDIDATES[j]]
                                  for i, j in zip(left[pair_reversal], right[pair_reversal])],
        "terminal_coverage_by_mode": coverage[..., -1].tolist(),
        "policies": policies,
    }


def describe_gain(values):
    values = np.asarray(values)
    indices = np.random.default_rng(55899).integers(len(values), size=(10000, len(values)))
    bootstrap = values[indices].mean(axis=1)
    return {
        "scenes": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
        "min": float(values.min()), "max": float(values.max()),
        "scene_bootstrap_95_percent_interval": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "scenes_above_one_coverage_point": int(np.count_nonzero(values > COST_TIE + FLOAT_TOLERANCE)),
    }


def summarize(data, threshold):
    rows = []
    probe_begin = 1 + (len(PREFIX) - 1) * ACTION_BLOCK
    for row, seed in enumerate(SCENES):
        coverage = data["actual_coverage"][row]
        old_mode, mode = data["actual_mode"][row, [0, -1]].tolist()
        visible_gap = np.any(data["actual_rgb"][row, probe_begin:] != data["stale_probe_rgb"][row], axis=(1, 2, 3))
        x_gap = data["actual_state"][row, probe_begin:, 2] - data["stale_probe_state"][row, :, 2]
        record = action_comparison(data["candidate_coverage"][row], mode, threshold)
        record.update({
            "scene": seed, "old_mode": old_mode, "mode": mode,
            "coverage_at_switch": float(coverage[probe_begin - 1]), "coverage_at_decision": float(coverage[-1]),
            "max_coverage_before_switch": float(coverage[:probe_begin].max()),
            "max_coverage_during_probe": float(coverage[probe_begin:].max()),
            "reached_before_switch": bool((coverage[:probe_begin] > threshold).any()),
            "reached_during_probe": bool((coverage[probe_begin:] > threshold).any()),
            "probe_differs_visually_by_raw_step": visible_gap.tolist(),
            "actual_minus_stale_probe_block_x_by_raw_step": x_gap.tolist(),
        })
        rows.append(record)

    def cohort(records):
        return {
            "scenes": len(records),
            "reached_before_switch": sum(row["reached_before_switch"] for row in records),
            "reached_during_probe": sum(row["reached_during_probe"] for row in records),
            "probe_rgb_differs": sum(any(row["probe_differs_visually_by_raw_step"]) for row in records),
            "disjoint_near_optimal_sets": sum(row["open_and_braked_near_optimal_disjoint"] for row in records),
            "scenes_with_strict_pair_reversal": sum(bool(row["strict_pair_reversals"]) for row in records),
            "uniform_mode_average_regret": describe_gain([row["policies"]["uniform_modes"]["mode_average_regret"]
                                                           for row in records]),
            "policies": {
                name: {
                    "actual_regret": describe_gain([row["policies"][name]["actual_regret"] for row in records]),
                    "scenes_where_every_near_tie_has_over_one_point_regret": sum(
                        row["policies"][name]["actual_regret_interval"][0] > COST_TIE + FLOAT_TOLERANCE
                        for row in records
                    ),
                    "actual_terminal_successes": sum(row["policies"][name]["actual_terminal_success"] for row in records),
                    "actual_any_step_successes": sum(row["policies"][name]["actual_any_step_success"] for row in records),
                } for name in POLICIES
            },
        }

    return {
        "phases": PHASES, "goal_distance": GOAL_DISTANCE, "candidate_names": CANDIDATES,
        "raw_steps_per_candidate": ACTION_BLOCK, "goals_fixed_before_prefix": True,
        "coverage_tie": COST_TIE, "success_threshold_strict": threshold,
        "verified_same_root_candidate_rollouts": len(rows) * 2 * len(CANDIDATES),
        "all_scenes": cohort(rows), "braked_to_open": cohort([row for row in rows if row["mode"] == 0]),
        "open_to_braked": cohort([row for row in rows if row["mode"] == 1]), "per_scene": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    output = parser.parse_args().output
    native = candidate_actions()
    blocks = np.stack((HOLD, PUSH / 4, PUSH / 2, PUSH * 3 / 4, PUSH, native[1, 0], native[2, 0]))
    live, fork = ReversiblePushT(), ReversiblePushT()
    try:
        scenes = [collect_scene(seed, blocks, live, fork) for seed in SCENES]
        threshold = live.success_threshold
    finally:
        live.close()
        fork.close()
    data = {key: np.stack([row[key] for row in scenes]) for key in scenes[0]}
    result = summarize(data, threshold)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "physical_paths.npz", scene_id=np.array(SCENES), phase=np.array(PHASES),
                        prefix_action=np.stack(PREFIX), candidate=np.array(CANDIDATES), candidate_action=blocks,
                        candidate_mode=np.arange(2), **data)
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {result['verified_same_root_candidate_rollouts']} same-root candidate runs in {output}")


if __name__ == "__main__":
    main()
