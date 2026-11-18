"""Paired physical labels and factual full-RGB histories for the Gate 0 task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from physical_topology_gate import (
    ACTION_BLOCK, CENTER, PhysicalRollout, ReversibleFourRoom, VisibleMapCost,
    block, make_scene, move_to, probe_and_dormancy, route_seed, shared_cem,
)


SPLITS = {
    "train": tuple(range(2026111600, 2026111696)),
    "holdout": tuple(range(2026111800, 2026111832)),
}
HORIZON = 4
ROOT_NAMES = ("top_contact", "bottom_contact", "northwest_navigation", "southeast_navigation")
LIFE_EVENTS = {"first_probe": 1, "reentry_probe": 2, "reversed_probe": 3}


def actual_common_windows(scene, env):
    """Every root has two already executed blocks that cannot read either lock."""
    positions, actions = [], []
    for from_left, anchor, goal, door in (
        (True, scene.northwest, scene.southeast, "top"),
        (False, scene.southeast, scene.northwest, "bottom"),
    ):
        mode_windows = []
        for mode in (0, 1):
            env.begin(anchor, goal, mode)
            move_to(env, (80 if from_left else 144, scene.door_positions[door]), [], "contact")
            control = block((.8 if from_left else -.8, 0))
            mode_windows.append(np.stack((env.xy, env.execute(control)[-1], env.execute(control)[-1])))
        np.testing.assert_array_equal(*mode_windows)
        positions.append(mode_windows[0])
        actions.append(np.stack((control, control)))
    for anchor, goal in ((scene.northwest, scene.southeast), (scene.southeast, scene.northwest)):
        mode_windows = []
        for mode in (0, 1):
            env.begin(anchor, goal, mode)
            away = -1 if anchor[0] < CENTER else 1
            move_to(env, anchor + (15 * away, 0), [], "free_out")
            move_to(env, anchor, [], "free_back")
            zero = block((0, 0))
            mode_windows.append(np.stack((env.xy, env.execute(zero)[-1], env.execute(zero)[-1])))
        np.testing.assert_array_equal(*mode_windows)
        positions.append(mode_windows[0])
        actions.append(np.stack((zero, zero)))
    return np.stack(positions), np.stack(actions)


def candidates(scene, physical, start, root):
    if root < 2:
        sign = 1 if root == 0 else -1
        rng = np.random.default_rng(np.random.SeedSequence((scene.seed, root, 551)))
        suffix = rng.uniform(-.35, .35, (3, 2)).astype(np.float32)
        # Identical suffix in candidates 0/1 isolates persistence of the first action.
        direct = np.vstack(((sign, 0), suffix))
        retreat = np.vstack(((-sign, 0), suffix))
        angled = np.vstack(((sign, rng.choice((-.55, .55))), (sign, 0), suffix[1:]))
        quiet = np.vstack(((0, .6), (-sign * .5, 0), suffix[1:]))
        controls = np.stack((direct, retreat, angled, quiet)).astype(np.float32)
    else:
        goal = scene.southeast if root == 2 else scene.northwest
        rng = np.random.default_rng(np.random.SeedSequence((scene.seed, root, 552)))
        plans, costs = shared_cem(scene, physical, VisibleMapCost(scene, goal), start, HORIZON, rng)
        controls = np.stack((
            route_seed(scene, start, goal, 1, HORIZON),
            route_seed(scene, start, goal, -1, HORIZON),
            plans[np.argmin(costs[0])],
            plans[np.argmin(costs[1])],
        ))
    return np.repeat(controls[:, :, None], ACTION_BLOCK, axis=2)


def both_mode_successors(physical, start, commands):
    xy = np.broadcast_to(start, (2, 1, 2))
    endpoints = []
    for control in commands:
        xy = physical.advance(xy, control)
        endpoints.append(xy[:, 0])
    return np.stack(endpoints, axis=1)


def lifecycle(scene, env, physical, initial_mode):
    """Only actual arrivals form the timeline consumed by the later belief updater."""
    anchor, goal = scene.northwest, scene.southeast
    env.begin(anchor, goal, initial_mode)
    timeline = []
    probe_and_dormancy(env, anchor, timeline, "first")
    probe_and_dormancy(env, anchor, timeline, "reentry")
    switch_index = len(timeline)
    np.testing.assert_array_equal(env.xy, anchor)
    env.set_mode(1 - initial_mode)
    probe_and_dormancy(env, anchor, timeline, "reversed")

    commands = np.asarray([tick["raw_actions"] for tick in timeline], dtype=np.float32)
    start = np.asarray([tick["start"] for tick in timeline], dtype=np.float32)
    trace = np.asarray([tick["raw_xy"] for tick in timeline], dtype=np.float32)
    np.testing.assert_array_equal(start[1:], trace[:-1, -1])
    mode = np.full(len(timeline), initial_mode, dtype=np.int8)
    mode[switch_index:] = 1 - initial_mode
    other = np.stack([both_mode_successors(physical, s, a) for s, a in zip(start, commands)])
    np.testing.assert_array_equal(trace, other[np.arange(len(timeline)), mode])

    event = np.asarray([LIFE_EVENTS.get(tick["phase"], 0) for tick in timeline], dtype=np.int8)
    assert (event == 1).sum() == (event == 2).sum() == (event == 3).sum() == 1
    informative = (other[:, 0, -1] != other[:, 1, -1]).any(axis=-1)
    assert informative[event > 0].all()
    return dict(
        life_action=commands, life_xy=np.vstack((start[0], trace[:, -1])),
        life_raw_xy=trace, life_cf_xy=other, life_mode=mode, life_event=event,
        life_informative=informative, life_valid=np.ones(len(timeline), dtype=bool),
        life_switch_index=np.asarray(switch_index, dtype=np.int16),
    )


def collect_scene(seed):
    scene = make_scene(seed)
    env = ReversibleFourRoom(scene)
    physical = PhysicalRollout(scene)
    root_xy, root_action = actual_common_windows(scene, env)
    actions, futures = [], []
    for root, window in enumerate(root_xy):
        goal = scene.southeast if root in (0, 2) else scene.northwest
        plans = candidates(scene, physical, window[-1], root)
        actions.append(plans)
        per_candidate = []
        for plan in plans:
            per_mode = []
            for mode in (0, 1):
                # Offline branching alone resets at the recorded, physically reached root.
                env.begin(window[-1], goal, mode)
                per_mode.append(np.stack([env.execute(command) for command in plan]))
            per_candidate.append(np.stack(per_mode))
        futures.append(np.stack(per_candidate))
    future_xy = np.stack(futures)
    assert np.all(np.linalg.norm(future_xy[:2, 0, 0, 0, -1] - future_xy[:2, 0, 1, 0, -1], axis=-1) > 0)
    histories = tuple(lifecycle(scene, env, physical, mode) for mode in (0, 1))
    env.close()
    return {
        "seed": np.asarray(seed, dtype=np.int64),
        "root_xy": root_xy, "root_action": root_action,
        "action": np.stack(actions), "future_xy": future_xy,
    }, histories


def physical_split(seeds):
    scenes, histories = zip(*(collect_scene(seed) for seed in seeds))
    data = {name: np.stack([scene[name] for scene in scenes]) for name in scenes[0]}
    n_ticks = max(len(episode["life_mode"]) for pair in histories for episode in pair)
    for name, values in histories[0][0].items():
        fill = -1 if name in ("life_mode", "life_event") else 0
        if name == "life_switch_index":
            data[name] = np.asarray([[episode[name] for episode in pair] for pair in histories])
            continue
        padded = np.full((len(seeds), 2, n_ticks + (name == "life_xy"), *values.shape[1:]),
                         fill, dtype=values.dtype)
        for i, pair in enumerate(histories):
            for j, episode in enumerate(pair):
                padded[i, j, :len(episode[name])] = episode[name]
        data[name] = padded
    return data


def gate_targets(data):
    """Find onset globally and one-block divergences from each identical local start."""
    future = data["future_xy"]
    roots, actions = data["root_xy"], data["action"]
    changed = (future[:, :, :, 0] != future[:, :, :, 1]).any(axis=-1)
    changed = changed.reshape(*changed.shape[:-2], HORIZON * ACTION_BLOCK)
    onset = np.where(changed.any(axis=-1), changed.argmax(axis=-1), -1).astype(np.int8)
    local = np.full(future.shape[:-2], -1, dtype=np.int8)
    for i, seed in enumerate(data["seed"]):
        physics = PhysicalRollout(make_scene(int(seed)))
        for h in range(HORIZON):
            source = (future[i, :, :, :, h - 1, -1] if h else np.broadcast_to(
                roots[i, :, None, None, -1], future.shape[1:4] + (2,),
            ))
            commands = np.broadcast_to(
                actions[i, :, :, None, h], future.shape[1:4] + (ACTION_BLOCK, 2),
            ).reshape(-1, ACTION_BLOCK, 2)
            source = source.reshape(-1, 2)
            xy = np.broadcast_to(source, (2, *source.shape))
            for raw in range(ACTION_BLOCK):
                xy = physics.advance(xy, commands[:, raw])
                states = xy.reshape(2, *future.shape[1:4], 2)
                np.testing.assert_array_equal(
                    np.stack([states[m, :, :, m] for m in (0, 1)], axis=2), future[i, :, :, :, h, raw],
                )
                target = local[i, :, :, :, h]
                new = (states[0] != states[1]).any(axis=-1) & (target == -1)
                target[new] = raw
    return dict(seed=data["seed"], first_branch_divergence_raw=onset, first_local_divergence_raw=local)


@torch.inference_mode()
def encode_split(data, model, device, batch_size):
    means = torch.tensor((.485, .456, .406), device=device)[None, :, None, None]
    scales = torch.tensor((.229, .224, .225), device=device)[None, :, None, None]
    root_z = np.empty((*data["root_xy"].shape[:-1], 192), dtype=np.float32)
    future_z = np.empty((*data["future_xy"].shape[:-2], 192), dtype=np.float32)
    life_z = np.zeros((*data["life_xy"].shape[:-1], 192), dtype=np.float32)
    life_cf_z = np.zeros((*data["life_cf_xy"].shape[:-2], 192), dtype=np.float32)
    total_frames = 0

    for i, seed in enumerate(data["seed"]):
        scene = make_scene(int(seed))
        env = ReversibleFourRoom(scene)
        env.begin(scene.northwest, scene.southeast, 0)
        active = data["life_valid"][i]
        observations = np.column_stack((np.ones(2, dtype=bool), active))
        groups = (
            data["root_xy"][i], data["future_xy"][i, ..., -1, :],
            data["life_xy"][i][observations], data["life_cf_xy"][i][active][..., -1, :],
        )
        locations, index = np.unique(
            np.concatenate([group.reshape(-1, 2) for group in groups]),
            axis=0, return_inverse=True,
        )
        unique_z = []
        for first in range(0, len(locations), batch_size):
            # These positions all come from completed physical transitions. Fixed
            # geometry and the unmarked target make RGB a function of the position.
            points = torch.from_numpy(locations[first:first + batch_size])
            rgb = torch.stack([env._render_frame(agent_pos=point) for point in points])
            pixels = (rgb.to(device, dtype=torch.float32) / 255 - means) / scales
            encoded = model.encode({"pixels": pixels[:, None]})["embed_0"][:, 0]
            unique_z.append(encoded.float().cpu().numpy())
        env.close()
        z = np.concatenate(unique_z)[index]
        cuts = np.cumsum([group.size // 2 for group in groups])
        root, future, factual, counterfactual = np.split(z, cuts[:-1])
        root_z[i] = root.reshape(root_z[i].shape)
        future_z[i] = future.reshape(future_z[i].shape)
        life_z[i][observations] = factual
        life_cf_z[i][active] = counterfactual.reshape(-1, 2, 192)
        np.testing.assert_array_equal(life_z[i][:, 1:][active], life_cf_z[i][active][
            np.arange(active.sum()), data["life_mode"][i][active],
        ])
        total_frames += len(locations)
        if (i + 1) % 8 == 0:
            print(f"Encoded {i + 1}/{len(data['seed'])} layouts, {total_frames} unique RGB frames", flush=True)
    result = dict(data, root_z=root_z, future_z=future_z, life_z=life_z, life_cf_z=life_cf_z)
    return result, total_frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/native_topology_condition_20261115"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--split", choices=tuple(SPLITS))
    parser.add_argument("--phase", choices=("physical", "gate", "encode"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    if args.phase == "encode":
        if args.checkpoint is None:
            parser.error("--checkpoint is required to encode actual RGB")
        model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get_level(1)
        model.interpolate_pos_encoding = True
        model.to(args.device).eval().requires_grad_(False)

    status = {}
    for name, seeds in SPLITS.items():
        if args.split is not None and name != args.split:
            continue
        path = args.output / f"data_{name}_physical.npz"
        if args.phase == "physical":
            data = physical_split(seeds)
            np.savez_compressed(path, **data)
        elif args.phase == "gate":
            with np.load(path) as physical:
                targets = gate_targets(physical)
            np.savez_compressed(args.output / f"data_{name}_gate.npz", **targets)
            status[name] = {key: list(values.shape) for key, values in targets.items()}
            print(name, status[name], flush=True)
            continue
        else:
            with np.load(path) as physical:
                data, frames = encode_split(dict(physical), model, args.device, args.batch)
            np.savez_compressed(args.output / f"data_{name}.npz", **data)
            status.setdefault(name, {})["unique_encoded_rgb"] = frames
        status.setdefault(name, {})["shapes"] = {key: list(values.shape) for key, values in data.items()}
        status[name]["lifecycle_ticks"] = int(data["life_valid"].sum())
        status[name]["lifecycle_informative_ticks"] = int(data["life_informative"].sum())
        status[name]["mode_separation_rms_xy_per_horizon"] = np.sqrt(
            np.square(data["future_xy"][:, :, :, 0, :, -1] -
                      data["future_xy"][:, :, :, 1, :, -1]).sum(axis=-1).mean(axis=(0, 1, 2))
        ).tolist()
        print(name, status[name], flush=True)
    suffix = args.phase if args.split is None else f"{args.phase}_{args.split}"
    (args.output / f"data_{suffix}_summary.json").write_text(
        json.dumps({"root_names": ROOT_NAMES, "life_events": LIFE_EVENTS, "splits": status}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
