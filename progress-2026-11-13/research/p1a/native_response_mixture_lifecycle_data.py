"""Collect the fixed §16.20 PushT retreat and return without model calls."""

from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HANDOFF = ROOT / "JEPA_OIC_Codex_Handoff_2026-11-03"
SOURCE = HANDOFF / "runs/train_11033026.npz"
OUTPUT = ROOT / "runs/native_response_mixture_lifecycle_20261113/physics.npz"
OBSERVATION_STEPS = np.arange(0, 61, 5)
QUERY_STEPS = np.array([30, 45, 60])
HORIZON_STEPS = np.array([5, 10])
SCENES = 16
WORLDS = CANDIDATES = 4
FRAME = (96, 96, 3)

sys.path.insert(0, str(HANDOFF / "native_content"))
from prepare_events import BACKGROUND, environment, public_masks, scene, simulate


TRANSITION_FIELDS = (
    ("absolute_control_targets", "controls", (2,), np.float64),
    ("actual_step", "was_executed", (), bool),
    ("contacts", "contacts", (), np.int32),
    ("contact_points", "contact_points", (), np.int32),
)
STATE_FIELDS = (
    ("physical_state", "state", (10,), np.float64),
    ("physical_coverage", "coverage", (), np.float64),
    ("terminated", "terminated", (), bool),
)


def collect():
    factual_shape = (SCENES, WORLDS)
    branch_shape = (SCENES, WORLDS, len(QUERY_STEPS), CANDIDATES)
    panel = dict(
        observation_steps=OBSERVATION_STEPS, query_steps=QUERY_STEPS, horizon_steps=HORIZON_STEPS,
        obs_rgb=np.empty((*factual_shape, len(OBSERVATION_STEPS), *FRAME), np.uint8),
        full_rgb=np.empty((*factual_shape, len(OBSERVATION_STEPS), *FRAME), np.uint8),
        future_rgb=np.empty((*branch_shape, len(HORIZON_STEPS), *FRAME), np.uint8),
        mask=np.empty((SCENES, len(OBSERVATION_STEPS), *FRAME[:2]), bool),
        patch=np.empty((SCENES, 9, 2), np.int64),
        arrived_pusher=np.empty(factual_shape, bool),
        executed_actions=np.empty((SCENES, 12, 5, 2)),
    )
    for name, _, shape, dtype in TRANSITION_FIELDS:
        panel[name] = np.empty((*factual_shape, 60, *shape), dtype)
        panel[f"query_{name}"] = np.empty((*branch_shape, 10, *shape), dtype)
    for name, _, shape, dtype in STATE_FIELDS:
        panel[name] = np.empty((*factual_shape, 61, *shape), dtype)
        query_name = "query_future_terminated" if name == "terminated" else f"query_{name}"
        panel[query_name] = np.empty((*branch_shape, 11, *shape), dtype)

    env = environment()
    try:
        with np.load(SOURCE, allow_pickle=False) as source:
            rows = np.argsort(source["scene_ids"])[:SCENES]
            panel["scene_ids"] = source["scene_ids"][rows]
            np.testing.assert_array_equal(panel["scene_ids"], np.arange(101000, 101000 + SCENES))
            query_anchor = np.flatnonzero(source["query_steps"] == QUERY_STEPS[0]).item()
            panel["query_actions"] = source["query_actions"][rows]
            panel["reset_state"] = source["reset_state"][rows]
            panel["psi"] = source["psi"][rows]
            original_fields = (
                "full_rgb", "obs_rgb", "physical_state", "physical_coverage", "terminated",
                "absolute_control_targets", "actual_step", "contacts", "contact_points",
                "query_physical_state", "query_physical_coverage", "query_future_terminated",
                "query_absolute_control_targets", "query_actual_step", "query_contacts",
                "query_contact_points", "future_rgb",
            )
            original = {name: source[name][rows] for name in original_fields}
            original_masks = source["mask"][rows]
            original_commands = source["executed_actions"][rows]

        for s, angle in enumerate(panel["psi"]):
            reset, _, candidates, cameras = scene(angle)
            np.testing.assert_array_equal(reset, panel["reset_state"][s])
            np.testing.assert_array_equal(candidates, panel["query_actions"][s])
            early_mask = public_masks(cameras, "minimal")
            np.testing.assert_array_equal(early_mask, original_masks[s])
            mask = np.concatenate((early_mask, np.repeat(early_mask[-1:], 6, axis=0)))
            panel["mask"][s] = mask
            y, x = np.nonzero(early_mask[3])
            panel["patch"][s] = np.stack((x, y), axis=1)
            forward = candidates[0, 0]
            hold = np.zeros_like(forward)
            continuation = np.stack((-forward, -forward, hold, forward, forward, hold))
            commands = np.concatenate((original_commands[s], continuation))
            panel["executed_actions"][s] = commands

            for w, world in enumerate(reset):
                row = (s, w)
                factual = simulate(env, world, commands)
                panel["full_rgb"][row] = factual["rgb"]
                visible = np.where(mask[..., None], factual["rgb"], BACKGROUND)
                panel["obs_rgb"][row] = visible
                frame = visible[3].astype(np.int16)
                panel["arrived_pusher"][row] = ((frame[..., 2] - frame[..., 0] > 115) & mask[3]).any()
                for name, key, _, _ in (*TRANSITION_FIELDS, *STATE_FIELDS):
                    panel[name][row] = factual[key]
                for name in original_fields[:9]:
                    earlier = original[name][row]
                    compare(panel[name][row][:len(earlier)], earlier, f"archive {row} {name}")

                for q, step in enumerate(QUERY_STEPS):
                    for a, candidate in enumerate(candidates):
                        fork = (s, w, q, a)
                        branch = simulate(env, world, np.concatenate((commands[:step // 5], candidate)))
                        for _, key, _, _ in TRANSITION_FIELDS:
                            compare(branch[key][:step], factual[key][:step], f"fork {fork} {key}")
                        for _, key, _, _ in STATE_FIELDS:
                            compare(branch[key][:step + 1], factual[key][:step + 1], f"fork {fork} {key}")
                        compare(branch["rgb"][:step // 5 + 1], factual["rgb"][:step // 5 + 1], f"fork {fork} RGB")

                        query_fields = ("future_rgb",)
                        panel["future_rgb"][fork] = branch["rgb"][-len(HORIZON_STEPS):]
                        for name, key, _, _ in (*TRANSITION_FIELDS, *STATE_FIELDS):
                            query_name = "query_future_terminated" if name == "terminated" else f"query_{name}"
                            panel[query_name][fork] = branch[key][step:]
                            query_fields += (query_name,)
                        if step == QUERY_STEPS[0]:
                            for name in query_fields:
                                compare(panel[name][fork], original[name][s, w, query_anchor, a],
                                        f"archive fork {fork} {name}")

            print(f"Replayed scene {panel['scene_ids'][s]} ({s + 1}/{SCENES})", flush=True)
    finally:
        env.close()
    return panel


def compare(actual, expected, label):
    if "coverage" in label:
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, err_msg=label)
    else:
        np.testing.assert_array_equal(actual, expected, err_msg=label)


if __name__ == "__main__":
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    data = collect()
    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("xb") as file:
        np.savez_compressed(file, **data)
