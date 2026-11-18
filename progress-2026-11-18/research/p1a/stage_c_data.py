"""Collect causal latent contrasts for the zero-training Stage C-0 audit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .native_topology_condition import load_host
from .stage_a_fit_gru import encode_frames
from .stage_a_fourroom import (
    ACTION_BLOCK, CENTER, DOORS, HORIZON, LOCKS, OfficialFourRoom,
    OpenPhysics, Task, fixed_tasks, route_controls, write_json,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/stage_c_20261117"
SPLITS = {"fit": (2026112600, 48, 16), "check": (2026112800, 24, 8)}


def execute(env, physics, start, goal, seed, lock, warmup, controls):
    env.ego_lock = lock
    env.reset(seed=seed, options={"state": start, "target_state": goal})
    initial = env.xy
    before = [env.execute(block) for block in warmup]
    rgb, xy = [env.render()], [env.xy]
    distractors = [env.distractor_positions[0].numpy().copy()]
    paths = []
    for block in controls:
        paths.append(env.execute(block))
        rgb.append(env.render())
        xy.append(env.xy)
        distractors.append(env.distractor_positions[0].numpy().copy())
    raw_xy = np.concatenate(before + paths)
    program = np.concatenate((warmup, controls), axis=0)
    expected = physics.rollout(initial, program[None], lock)[0]
    np.testing.assert_allclose(raw_xy, expected, atol=1e-5, rtol=0)
    return SimpleNamespace(rgb=np.stack(rgb), xy=np.stack(xy),
                           distractors=np.stack(distractors), raw_xy=raw_xy)


def matches(left, right, rows):
    np.testing.assert_array_equal(left.rgb[:rows], right.rgb[:rows])
    np.testing.assert_array_equal(left.xy[:rows], right.xy[:rows])
    np.testing.assert_array_equal(left.distractors, right.distractors)


def encode(encoder, rgb):
    rgb = np.asarray(rgb)
    z = encode_frames(encoder, rgb.reshape(-1, *rgb.shape[-3:])).numpy()
    return z.reshape(*rgb.shape[:-3], z.shape[-1])


def command(axis, direction):
    block = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    block[:, axis] = direction
    return block


def collect_probes(env, physics, encoder, seed, count):
    """Use each actual lock query as its own distractor-control measurement."""
    frames = {name: [] for name in ("lock", "lock_alt", "action", "open")}
    arrays = {name: [] for name in ("lock_a", "action_a", "open_a", "open_raw_path",
                                    "lock_xy", "action_xy", "open_xy", "open_static_wall",
                                    "bg_distractor_xy")}
    rows = []
    for i in range(count):
        door = ("top", "bottom")[i % 2]
        sign = (1, -1)[(i // 2) % 2]
        tangent = DOORS[door] + (-7.5, 0., 7.5)[(i // 4) % 3]
        start = np.array((CENTER - sign * (9.5 + ((i // 4) % 3)), tangent), dtype=np.float32)
        goal = np.array((CENTER + sign * 48., tangent), dtype=np.float32)
        forward = command(0, sign)
        lateral = command(1, (-1, 1)[(i // 4) % 2])
        past = np.stack((-forward, forward))
        controls = np.concatenate((past, forward[None]), axis=0)
        alternate = np.concatenate((past, lateral[None]), axis=0)
        # Retain a range of real distractor clocks while keeping the recorded window
        # physically identical. The zero commands are actual unrecorded prefix steps.
        warmup = np.zeros(((i // 4) % 8, ACTION_BLOCK, 2), dtype=np.float32)
        primary = tuple(execute(env, physics, start, goal, seed+i, lock, warmup, controls)
                        for lock in LOCKS)
        changed_bg = tuple(execute(env, physics, start, goal, seed+64+i, lock, warmup, controls)
                           for lock in LOCKS)
        moved = execute(env, physics, start, goal, seed+i, LOCKS[0], warmup, alternate)
        matches(*primary, rows=3)
        matches(*changed_bg, rows=3)
        matches(primary[0], moved, rows=3)
        for original, changed in zip(primary, changed_bg, strict=True):
            np.testing.assert_array_equal(original.raw_xy, changed.raw_xy)
            if np.array_equal(original.distractors, changed.distractors):
                raise AssertionError("Background replay failed to produce independent distractor motion")
        if np.array_equal(primary[0].xy[-1], primary[1].xy[-1]):
            raise AssertionError("The issued H1 probe must separate the two actual lock conditions")
        if np.array_equal(primary[0].xy[-1], moved.xy[-1]):
            raise AssertionError("The two actual H1 commands must cause different ego outcomes")

        axis = (i // 12) % 2
        open_static_wall = bool((i // 4) % 2)
        if open_static_wall:
            open_start = np.empty(2, dtype=np.float32)
            open_start[axis] = start[0]
            open_start[1-axis] = (83.5, 140.5)[i % 2]
            open_goal = open_start.copy()
            open_goal[axis] = goal[0]
            open_forward = command(axis, sign)
            open_controls = np.stack((-open_forward, open_forward, open_forward))
        else:
            open_start, open_goal, open_controls = start, goal, controls
        opened = execute(env, physics, open_start, open_goal, seed+i, None, warmup, open_controls)

        frames["lock"].append(np.stack([trace.rgb for trace in primary]))
        frames["lock_alt"].append(np.stack([trace.rgb for trace in changed_bg]))
        frames["action"].append(moved.rgb[-1])
        frames["open"].append(opened.rgb)
        arrays["lock_a"].append(controls)
        arrays["action_a"].append(np.stack((controls, alternate)))
        arrays["open_a"].append(open_controls)
        arrays["lock_xy"].append(np.stack([trace.xy for trace in primary]))
        arrays["action_xy"].append(np.stack((primary[0].xy, moved.xy)))
        arrays["open_xy"].append(opened.xy)
        arrays["open_raw_path"].append(opened.raw_xy[-ACTION_BLOCK:])
        arrays["open_static_wall"].append(open_static_wall)
        arrays["bg_distractor_xy"].append(np.stack((primary[0].distractors, changed_bg[0].distractors)))
        rows.append({"row": i, "seed": seed+i, "alternate_distractor_seed": seed+64+i,
                     "door": door, "direction": sign, "warmup_blocks": len(warmup),
                     "open_static_wall": open_static_wall, "open_axis": axis if open_static_wall else 0})

    arrays = {name: np.asarray(values) for name, values in arrays.items()}
    lock_z = encode(encoder, np.stack(frames["lock"]))
    lock_alt_z = encode(encoder, np.stack(frames["lock_alt"]))
    moved_z = lock_z[:, 0].copy()
    moved_z[:, -1] = encode(encoder, np.stack(frames["action"]))
    arrays.update(lock_z=lock_z, lock_alt_z=lock_alt_z, open_z=encode(encoder, np.stack(frames["open"])),
                  bg_z=np.stack((lock_z[:, 0], lock_alt_z[:, 0]), axis=1),
                  bg_a=arrays["lock_a"].copy(), bg_xy=np.repeat(arrays["lock_xy"][:, :1], 2, axis=1),
                  action_z=np.stack((lock_z[:, 0], moved_z), axis=1))
    return arrays, rows


def collect_bundles(env, physics, encoder, seed, count):
    """Run both lawful H12 proposals from each of two matched physical roots."""
    tasks = tuple(task for task in fixed_tasks() if task.family == "two_doors")
    arrays = {name: [] for name in ("bundle_z", "bundle_root_z", "bundle_xy",
                                    "bundle_root_xy", "bundle_a", "bundle_past")}
    rows = []
    for i in range(count):
        template = tasks[i % len(tasks)]
        start = np.array(template.start, dtype=np.float32)
        goal = np.array(template.goal, dtype=np.float32)
        start[1] += (-2.5, 2.5)[i % 2]
        task = Task(template.name, template.family, tuple(start.tolist()), template.goal, seed+i, ())
        out = command(0, -0.4 if start[0] < CENTER else 0.4)
        past = np.stack((out, -out))
        proposals = np.stack([route_controls(task, start, direction, HORIZON) for direction in (1, -1)])
        traces = [[execute(env, physics, start, goal, seed+i, lock, past[:0],
                           np.concatenate((past, proposal), axis=0))
                   for proposal in proposals] for lock in LOCKS]
        first = traces[0][0]
        for trace in (traces[0][1], *traces[1]):
            matches(first, trace, rows=3)
        np.testing.assert_allclose(first.xy[2], start, atol=1e-5, rtol=0)
        rgb = np.stack([[trace.rgb[3:] for trace in pair] for pair in traces])
        xy = np.stack([[trace.xy[3:] for trace in pair] for pair in traces])
        x = np.concatenate((first.rgb[:3], rgb.reshape(-1, *rgb.shape[-3:])), axis=0)
        z = encode(encoder, x)
        if not np.all(np.linalg.norm(xy[0] - xy[1], axis=-1).max(axis=1) > 0):
            raise AssertionError("Each closed-door condition must affect a lawful H12 route")
        if not np.all(np.linalg.norm(xy[:, 0] - xy[:, 1], axis=-1).max(axis=1) > 0):
            raise AssertionError("The two route proposals must differ within each true condition")
        arrays["bundle_z"].append(z[3:].reshape(2, 2, HORIZON, z.shape[-1]))
        arrays["bundle_root_z"].append(z[:3])
        arrays["bundle_xy"].append(xy)
        arrays["bundle_root_xy"].append(first.xy[:3])
        arrays["bundle_a"].append(proposals)
        arrays["bundle_past"].append(past)
        rows.append({"row": i, "seed": seed+i, "template": template.name,
                     "start": start.tolist(), "goal": goal.tolist()})
    return {name: np.stack(values) for name, values in arrays.items()}, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    encoder = SimpleNamespace(
        host=load_host(device), device=device,
        mean=torch.tensor((.485, .456, .406), device=device)[:, None, None],
        std=torch.tensor((.229, .224, .225), device=device)[:, None, None],
    )
    env = OfficialFourRoom()
    physics = OpenPhysics(env)
    arrays, cases = {}, {}
    try:
        for split, (seed, count, bundle_count) in SPLITS.items():
            probes, probe_rows = collect_probes(env, physics, encoder, seed, count)
            print(f"{split}: collected {count} aligned H1 probe cases", flush=True)
            bundles, bundle_rows = collect_bundles(env, physics, encoder, seed+128, bundle_count)
            print(f"{split}: collected {bundle_count} matched H12 bundles", flush=True)
            arrays.update({f"{split}_{key}": value for key, value in (probes | bundles).items()})
            cases[split] = {"probes": probe_rows, "bundles": bundle_rows}
    finally:
        env.close()
    args.output.mkdir(parents=True, exist_ok=True)
    temp = args.output / "calibration.tmp.npz"
    np.savez_compressed(temp, **arrays)
    os.replace(temp, args.output / "calibration.npz")
    write_json(args.output / "calibration_cases.json", {
        "phases": cases, "latent_model": "Official frozen FourRoom LeWM",
        "lock_order_for_physical_diagnostics": list(LOCKS),
        "route_order": [1, -1], "horizon_blocks": HORIZON, "raw_actions_per_block": ACTION_BLOCK,
        "shape": {key: list(value.shape) for key, value in arrays.items()},
        "contracts": [
            "All latent targets are frozen official encodings of executed simulator RGB.",
            "H1 index 0..2 is the common factual prefix. Index 3 is the actual arrival.",
            "Each bg query replays lock world 0 with an independent distractor random stream.",
            "lock_alt_z includes the same alternate stream for both lock worlds.",
            "Lock and action branches share byte-identical RGB prefixes and the full distractor clock.",
            "Every block is checked against author scalar collisions and exact batched OpenPhysics.",
            "H12 bundle latents contain 12 separately rendered actual arrivals, excluding their root.",
            "Physical and seed metadata are for audit only, outside the model forward inputs.",
        ],
    })
    print(f"Wrote {args.output / 'calibration.npz'}", flush=True)


if __name__ == "__main__":
    main()
