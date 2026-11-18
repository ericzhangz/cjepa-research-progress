"""Qualify the fixed full-view PushT fallback after the official FourRoom gate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from .native_reversible_pusht import (
    ACTION_BLOCK, HOLD, ORACLE_CLOSED_LOOP_CEM, PUSH, RETREAT, ROOT,
    ReversiblePushT, Scene, cem, native_cost, replay,
)


OUTPUT = ROOT / "runs/stage_a_20261117/pusht"
ARCHIVE = ROOT / "runs/native_reversible_delivery_20261115/physical_paths.npz"
CHECKPOINT = ROOT / "runs/fullview_pusht_base_20261115/world_model.pt"
SEEDS = tuple(range(55810, 55826))
PREFIX = (PUSH, RETREAT, HOLD, HOLD, HOLD)
QUIET_START = 2 * ACTION_BLOCK
QUIET_END = len(PREFIX) * ACTION_BLOCK
MAX_REPLANS = 12
MIN_SUCCESS = 0.70
MODEL_PARTS = ("action_encoder", "predictor", "pred_proj")


def fixtures():
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["scene_id"], SEEDS)
        np.testing.assert_array_equal(archive["actual_mode"][:, 0], 1 - np.asarray(SEEDS) % 2)
        np.testing.assert_array_equal(archive["prefix_action"][:len(PREFIX)], np.stack(PREFIX))
        return [(seed, Scene(reset, goal), int(mode)) for seed, reset, goal, mode in zip(
            SEEDS, archive["reset"], archive["goal_pose"], archive["actual_mode"][:, 0], strict=True,
        )]


def motion(env):
    return np.asarray([
        (*body.position, *body.velocity, body.angle, body.angular_velocity, *body.force, body.torque)
        for body in (env.agent, env.block)
    ])


def execute_path(env, block):
    rgb, dynamics, coverage, brake = [], [], [], []
    start_brake = env.brake_substeps

    def observe(frame):
        rgb.append(frame)
        dynamics.append(motion(env))
        coverage.append(env._get_coverage())
        brake.append(env.brake_substeps - start_brake)

    env.execute(block, observe_step=observe)
    return tuple(np.asarray(field) for field in (rgb, dynamics, coverage, brake))


def assert_root(env, rgb, dynamics):
    np.testing.assert_array_equal(env.observe(), rgb)
    np.testing.assert_array_equal(motion(env), dynamics)


def paired_probe(env, fork, scene, history, mode):
    root_rgb, root_motion = env.observe(), motion(env)
    env.set_regime(mode)
    assert_root(env, root_rgb, root_motion)
    replay(fork, scene, history, 1 - mode)
    assert_root(fork, root_rgb, root_motion)
    actual, opposite = execute_path(env, PUSH), execute_path(fork, PUSH)
    history.append((mode, PUSH))
    return actual, opposite


def probe_summary(actual, opposite, mode):
    # The T is body 1 and its x coordinate is field 0.
    x_gap = actual[1][:, 1, 0] - opposite[1][:, 1, 0]
    rgb_diff = np.any(actual[0] != opposite[0], axis=(1, 2, 3))
    return {
        "actual_mode": mode, "actual_minus_opposite_x": x_gap.tolist(),
        "raw_rgb_diff": rgb_diff.tolist(),
        "reveals_mode": bool(rgb_diff[-1] and (1 - 2 * mode) * x_gap[-1] > 0),
    }


def branch_coverage(fork, scene, history, mode, plan):
    replay(fork, scene, history, mode)
    endpoints = []
    for block in plan:
        for command in block:
            fork._step_relative(command)
        endpoints.append(fork._get_coverage())
    return np.asarray(endpoints)


def physical_cost(fork, scene, history, modes):
    def cost(plans):
        return np.asarray([
            1.0 - np.mean([branch_coverage(fork, scene, history, mode, plan)[-1] for mode in modes])
            for plan in plans
        ])

    return cost


def decision(fork, scene, history, mode, seed, rgb, dynamics):
    for current in (0, 1):
        replay(fork, scene, history, current)
        assert_root(fork, rgb, dynamics)

    bank, outcomes = [], []

    def uniform_cost(plans):
        paired = np.asarray([[branch_coverage(fork, scene, history, current, plan) for current in (0, 1)]
                             for plan in plans])
        bank.append(plans.copy())
        outcomes.append(paired)
        return 1.0 - paired[:, :, -1].mean(axis=1)

    selected, _ = cem(uniform_cost, np.random.default_rng(seed), **ORACLE_CLOSED_LOOP_CEM)
    actions, coverage = np.concatenate(bank), np.concatenate(outcomes)
    true_indices = np.argmin(1.0 - coverage[:, :, -1], axis=0)
    uniform_index = int(np.argmin(1.0 - coverage[:, :, -1].mean(axis=1)))
    np.testing.assert_array_equal(selected, actions[uniform_index])
    choices = {}
    for name, index in (("true", int(true_indices[mode])), ("uniform", uniform_index)):
        choices[name] = {"index": index, "actions": actions[index].tolist(),
                         "coverage_by_mode": coverage[index].tolist()}
    gains = coverage[true_indices, (0, 1)] - coverage[uniform_index]
    return {
        "actual_mode": mode, "choices": choices, "true_indices_by_mode": true_indices.tolist(),
        "distinct_true_first_actions_by_mode": bool(np.any(actions[true_indices[0], 0] != actions[true_indices[1], 0])),
        "true_minus_uniform_coverage": gains[mode].tolist(),
        "true_minus_uniform_coverage_by_mode": gains.tolist(),
    }, {"actions": actions, "coverage_by_mode": coverage}


def lifecycle(seed, scene, mode, env, fork):
    history = []
    initial_rgb = env.start(scene, mode)
    initial_motion = motion(env)
    initial_coverage = env._get_coverage()
    probe, probe_opposite = paired_probe(env, fork, scene, history, mode)
    retreat = execute_path(env, RETREAT)
    history.append((mode, RETREAT))

    replay(fork, scene, history, 1 - mode)
    assert_root(fork, env.observe(), motion(env))
    holds, holds_opposite = [], []
    for hold in PREFIX[2:]:
        holds.append(execute_path(env, hold))
        holds_opposite.append(execute_path(fork, hold))
        history.append((mode, hold))
    quiet_rgb = np.concatenate([block[0] for block in holds])
    quiet_motion = np.concatenate([block[1] for block in holds])
    matching = np.all(quiet_rgb == np.concatenate([block[0] for block in holds_opposite]), axis=(1, 2, 3))
    matching &= np.all(quiet_motion == np.concatenate([block[1] for block in holds_opposite]), axis=(1, 2))

    arrived, arrived_candidates = decision(fork, scene, history, mode, seed + len(history), env.observe(), motion(env))
    reprobe, reprobe_opposite = paired_probe(env, fork, scene, history, 1 - mode)
    reversed_decision, reversed_candidates = decision(
        fork, scene, history, 1 - mode, seed + len(history), env.observe(), motion(env),
    )
    data = {
        "rgb": np.concatenate((initial_rgb[None], probe[0], retreat[0], quiet_rgb, reprobe[0])),
        "motion": np.concatenate((initial_motion[None], probe[1], retreat[1], quiet_motion, reprobe[1])),
        "coverage": np.concatenate(([initial_coverage], *(block[2] for block in (probe, retreat, *holds, reprobe)))),
        "brake_substeps_within_block": np.concatenate([block[3] for block in (probe, retreat, *holds, reprobe)]),
        "goal_rgb": env.goal_image(scene),
        "opposite_initial_probe_rgb": probe_opposite[0], "opposite_reverse_probe_rgb": reprobe_opposite[0],
        "candidate_actions": np.stack((arrived_candidates["actions"], reversed_candidates["actions"])),
        "candidate_coverage_by_mode": np.stack((arrived_candidates["coverage_by_mode"],
                                                reversed_candidates["coverage_by_mode"])),
    }
    return {
        "initial_probe": probe_summary(probe, probe_opposite, mode),
        "neutral_raw_steps": [bool(value) for value in matching],
        "after_neutral": arrived, "reverse_probe": probe_summary(reprobe, reprobe_opposite, 1 - mode),
        "after_reverse": reversed_decision,
    }, data


def known_open(seed, scene, env, *, fork=None, model=None, normalize=None, reference=None):
    frames = [env.start(scene, 0)]
    goal = env.goal_image(scene)
    history = []
    for block in PREFIX:
        frames.append(env.execute(block))
        history.append((0, block))
    root = {"rgb_window": np.stack(frames[-3:]), "motion": motion(env), "goal_rgb": goal}
    if reference is not None:
        for field, values in reference.items():
            np.testing.assert_array_equal(root[field], values)

    outcomes = []
    for replan in range(MAX_REPLANS):
        if env._get_coverage() > env.success_threshold:
            break
        if model is None:
            cost = physical_cost(fork, scene, history, (0,))
        else:
            past = [block for _, block in history[-2:]]
            cost = native_cost(model, frames[-3:], goal, past, normalize=normalize)
        plan, value = cem(cost, np.random.default_rng(seed + len(PREFIX) + replan), **ORACLE_CLOSED_LOOP_CEM)
        pixels, _, coverage, _ = execute_path(env, plan[0])
        frames.append(pixels[-1])
        history.append((0, plan[0].copy()))
        outcomes.append({"action": plan[0].tolist(), "planned_cost": value, "raw_coverage": coverage.tolist()})
        if coverage.max() > env.success_threshold:
            break

    rates = [value for step in outcomes for value in step["raw_coverage"]]
    end = float(env._get_coverage())
    best = max([end, *rates])
    return {"success": best > env.success_threshold, "best_coverage": best,
            "final_coverage": end, "replans": outcomes}, root


def run_physical(fixture):
    seed, scene, mode = fixture
    env, fork = ReversiblePushT(), ReversiblePushT()
    try:
        row, facts = lifecycle(seed, scene, mode, env, fork)
        control, root = known_open(seed, scene, env, fork=fork)
    finally:
        env.close()
        fork.close()
    print(f"PushT physical seed {seed}: known-open success {control['success']}", flush=True)
    return row, facts, control, root


def run_native(scenes, physical, device):
    import torch
    from .native_response_mixture_regulation import _action_moments, _native_model

    torch.set_num_threads(3)
    model = _native_model(device)
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    if set(checkpoint) != set(MODEL_PARTS):
        raise ValueError("Unexpected modules in the fixed full-view PushT checkpoint")
    for name in MODEL_PARTS:
        getattr(model, name).load_state_dict(checkpoint[name], strict=True)

    def normalize(actions):
        mean, std = _action_moments(actions)
        return (actions - mean) / std

    output = []
    env = ReversiblePushT()
    try:
        for (seed, scene, _), physical_result in zip(scenes, physical, strict=True):
            score, _ = known_open(seed, scene, env, model=model, normalize=normalize, reference=physical_result[-1])
            output.append(score)
            print(f"PushT native seed {seed}: known-open success {score['success']}", flush=True)
    finally:
        env.close()
    return output


def summarize(scenes, physical, native):
    rows = {}
    for (seed, _, _), (lifecycle_row, _, exact, _), learned in zip(scenes, physical, native, strict=True):
        rows[str(seed)] = lifecycle_row | {"known_open_exact": exact, "known_open_native": learned}
    records = list(rows.values())
    rates = {name: float(np.mean([row[name]["success"] for row in records]))
             for name in ("known_open_exact", "known_open_native")}
    physical_results = {}
    for phase in ("after_neutral", "after_reverse"):
        gains = np.asarray([row[phase]["true_minus_uniform_coverage"] for row in records])
        modes = np.asarray([row["initial_probe"]["actual_mode"] for row in records])
        physical_results[phase] = {
            "mean_true_minus_uniform_coverage_h1_h2": gains.mean(0).tolist(),
            "brake_to_open_mean_h1_h2": gains[modes == 1].mean(0).tolist(),
            "open_to_brake_mean_h1_h2": gains[modes == 0].mean(0).tolist(),
            "distinct_true_first_actions_by_mode_scenes": sum(row[phase]["distinct_true_first_actions_by_mode"]
                                                               for row in records),
        }
    gates = {name: rate >= MIN_SUCCESS for name, rate in rates.items()}
    gates["hidden_lifecycle"] = all(
        row["initial_probe"]["reveals_mode"] and row["reverse_probe"]["reveals_mode"]
        and len(row["neutral_raw_steps"]) >= 10 and all(row["neutral_raw_steps"]) for row in records
    )
    gates["physical_decision_value"] = all(
        row["mean_true_minus_uniform_coverage_h1_h2"][-1] > 1e-12 for row in physical_results.values()
    )
    return {
        "seeds": list(SEEDS), "cem": ORACLE_CLOSED_LOOP_CEM, "max_open_replans": MAX_REPLANS,
        "fixture_archive": str(ARCHIVE), "native_checkpoint": str(CHECKPOINT),
        "min_known_open_success_rate": MIN_SUCCESS, "known_open_success_rates": rates,
        "neutral_first_and_last_raw_step": [QUIET_START + 1, QUIET_END],
        "decision_roots_raw_steps": [QUIET_END, QUIET_END + ACTION_BLOCK],
        "first_reprobe_raw_step": QUIET_END + 1,
        "candidate_proposals": "uniform physical CEM with the same two-mode rollouts ranked by both rules",
        "physical_decision_value": physical_results, "gates": gates, "gate_passed": all(gates.values()),
        "scenes": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    scenes = fixtures()
    with ProcessPoolExecutor(max_workers=4) as pool:
        physical = list(pool.map(run_physical, scenes))
    facts = [entry[1] for entry in physical]
    np.savez_compressed(args.output / "physical_paths.npz", scene_id=np.asarray(SEEDS),
                        action=np.stack((*PREFIX, PUSH)), reset=np.stack([scene.reset for _, scene, _ in scenes]),
                        goal_pose=np.stack([scene.goal for _, scene, _ in scenes]),
                        **{name: np.stack([fact[name] for fact in facts]) for name in facts[0]})
    native = run_native(scenes, physical, args.device)
    summary = summarize(scenes, physical, native)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"PushT Stage A gate passed: {summary['gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
