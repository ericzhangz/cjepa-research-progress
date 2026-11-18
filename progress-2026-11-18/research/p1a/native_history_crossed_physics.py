"""Execute the fixed physical parent–destination pairing of §37.4."""

import json

import numpy as np

from .native_history_partial_arrivals import HOLD, HORIZONS, PAIRS, image_x
from .native_reversible_pusht import ReversiblePushT, candidate_actions, make_scene, replay


OUTPUT = PAIRS.parent / "native_history_crossed_physics_20261115"
SCENES = np.arange(56000, 56016)
NON_HOLD = [0, 1, 2, 4, 5]


def collect():
    with np.load(PAIRS / "fresh_switches.npz", allow_pickle=False) as reference:
        commands = reference["actions"][0, :8]
        phases = reference["phase"][:8]
        common = reference["mode"][:6]
        if phases[-2:].tolist() != ["recontact", "reverse_open"]:
            raise ValueError("Expect recontact and the following probe from the fixed replay")
        np.testing.assert_array_equal(reference["actions"][:, :8], np.broadcast_to(commands, (16, *commands.shape)))
    plans = candidate_actions()
    if np.any(plans[HOLD]):
        raise ValueError("The reference action must be hold")
    head = len(SCENES), 2, 2
    initial = np.empty((len(SCENES), 96, 96, 3), dtype=np.uint8)
    observed_x = np.empty((*head, len(commands)))
    root_xy = np.empty((*head, 5))
    future = np.empty((*head, *plans.shape[:2], 5))
    coverage = np.empty((*head, *plans.shape[:2]))
    env = ReversiblePushT()
    try:
        for row, seed in enumerate(SCENES):
            scene = make_scene(int(seed))
            initial[row] = env.start(scene, int(common[0]))
            for parent, current in np.ndindex((2, 2)):
                np.testing.assert_array_equal(env.start(scene, int(common[0])), initial[row])
                modes = [*map(int, common), parent, current]
                history = list(zip(modes, commands))
                images = []
                for mode, command in history:
                    env.set_regime(mode)
                    images.append(env.execute(command))
                observed_x[row, parent, current] = image_x(np.stack(images))
                root_image = images[-1]
                root_state = env.get_obs().copy()
                root_xy[row, parent, current] = root_state
                for action, plan in enumerate(plans):
                    replay(env, scene, history, current)
                    np.testing.assert_array_equal(env.observe(), root_image)
                    np.testing.assert_array_equal(env.get_obs(), root_state)
                    for tick, command in enumerate(plan):
                        env.execute(command)
                        future[row, parent, current, action, tick] = env.get_obs()
                        coverage[row, parent, current, action, tick] = env._get_coverage()
    finally:
        env.close()
    np.testing.assert_array_equal(observed_x[:, 0, :, :6], observed_x[:, 1, :, :6])
    return {"scene_ids": SCENES, "commands": commands, "common_modes": common, "initial_rgb": initial,
            "observed_x": observed_x, "root_state": root_xy, "plans": plans, "future_state": future,
            "future_coverage": coverage}


def analyze(panel):
    source = np.diff(panel["observed_x"], axis=1)
    xy = panel["future_state"][..., HORIZONS, 2:4]
    gap = (xy[:, 1] - xy[:, 0])
    hold = gap[:, :, HOLD]
    active = gap[:, :, NON_HOLD] - gap[:, :, HOLD:HOLD + 1]
    effect = np.linalg.norm(active, axis=-1)
    coverage = panel["future_coverage"][..., HORIZONS]
    mean_choices = coverage.mean(axis=1).argmax(axis=2)
    mean_physical = np.take_along_axis(coverage, mean_choices[:, None, :, None], axis=3).squeeze(3)
    regret = (coverage.max(axis=3) - mean_physical) * 100
    return {
        "scene_ids": panel["scene_ids"].tolist(),
        "replayed_candidate_roots": int(np.prod(coverage.shape[:-1])),
        "source_visible_gap_rms_by_mode_and_arrival": np.sqrt(np.mean(source[:, 0] ** 2, axis=0)).tolist(),
        "hold_xy_gap_rms_by_mode_and_horizon": np.sqrt(np.mean(np.sum(hold ** 2, axis=-1), axis=0)).tolist(),
        "relative_xy_gap_rms_by_mode_and_horizon": np.sqrt(np.mean(effect ** 2, axis=(0, 2))).tolist(),
        "roots_with_visible_gap_above_one": (np.abs(source[:, 0, :, -1]) > 1).sum(axis=0).tolist(),
        "roots_with_both_source_and_any_relative_gap_above_one": (
            (np.abs(source[:, 0, :, -1, None]) > 1) & (effect.max(axis=2) > 1)
        ).sum(axis=0).tolist(),
        "mean_uniform_parent_policy_regret_points": regret.mean(axis=(0, 1)).tolist(),
        "roots_where_parent_changes_best_action_count": (
            coverage[:, 0].argmax(axis=2) != coverage[:, 1].argmax(axis=2)
        ).sum(axis=0).tolist(),
    }


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    panel = collect()
    summary = analyze(panel)
    with (OUTPUT / "physical.npz").open("xb") as stream:
        np.savez_compressed(stream, **panel)
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    print(json.dumps(summary), flush=True)
