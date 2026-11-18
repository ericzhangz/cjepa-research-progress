"""Freeze and label same-root H5 training/evaluation banks prior to fitting."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from topology_training_data import ROOT_NAMES, actual_common_windows, candidates

from .native_topology_condition import ROOT, RUN, forecast, load_readout, load_trained
from .physical_topology_gate import (
    ACTION_BLOCK, CENTER, ITERATIONS, SAMPLES, PhysicalRollout, ReversibleFourRoom,
    VisibleMapCost, cem_population, make_scene, room, route_seed,
)


OUTPUT = ROOT / "runs/native_topology_prefix_20261115"
TRAIN_SEEDS = tuple(range(2026111600, 2026111632))
EVAL_SEEDS = tuple(range(2026112000, 2026112016))
HORIZON = 5
ROLES = ("original", "retained_original", "common_corner", "block_end_separated",
         "corner_fallback", "separated_fallback")


def original_h5(scene, physical, start, goal, root):
    old = candidates(scene, physical, start, root)
    plans = np.concatenate((old, old[:, -1:]), axis=1)
    if root >= 2:
        for route, direction in enumerate((1, -1)):
            extended = np.repeat(route_seed(scene, start, goal, direction, HORIZON)[:, None],
                                 ACTION_BLOCK, axis=1)
            np.testing.assert_array_equal(extended[:-1], old[route])
            plans[route, -1] = extended[-1]
    return plans


@torch.inference_mode()
def encode_groups(env, host, groups, batch, known=None):
    """Render the exact stock RGB at reached positions, with scene-fixed geometry."""
    inputs = (known[0], *groups) if known is not None else groups
    locations, indices = np.unique(np.concatenate([group.reshape(-1, 2) for group in inputs]),
                                  axis=0, return_inverse=True)
    available = np.ones(len(locations), dtype=bool)
    values = np.empty((len(locations), 192), dtype=np.float32)
    if known is not None:
        prefix = known[0].size // 2
        values[indices[:prefix]] = known[1].reshape(-1, 192)
        available[indices[:prefix]] = False
        indices = indices[prefix:]
    device = host.action_mean.device
    means = torch.tensor((.485, .456, .406), device=device)[None, :, None, None]
    scales = torch.tensor((.229, .224, .225), device=device)[None, :, None, None]
    missing = np.flatnonzero(available)
    for start in range(0, len(missing), batch):
        rows = missing[start:start + batch]
        points = torch.from_numpy(locations[rows])
        rgb = torch.stack([env._render_frame(agent_pos=point) for point in points])
        pixels = (rgb.to(device, dtype=torch.float32) / 255 - means) / scales
        values[rows] = host.encode({"pixels": pixels[:, None]})["emb"][:, 0].float().cpu().numpy()
    cuts = np.cumsum([group.size // 2 for group in groups[:-1]])
    result = [value.reshape(*group.shape[:-1], 192)
              for value, group in zip(np.split(values[indices], cuts), groups)]
    return result, len(missing)


@torch.inference_mode()
def query_bank(scene, score, root, root_z, root_action, old, model, decode):
    host, conditioner, gate = model
    device = host.action_mean.device
    history = torch.as_tensor(root_z, device=device)[None]
    past = torch.as_tensor(root_action, device=device)[None]
    decoded_current = decode(history[0, -1]).cpu().numpy()
    endpoints = []

    def score_raw(raw):
        prediction = forecast(host, conditioner, gate, history, past,
                              torch.from_numpy(raw).to(device)[None], compact=True)
        terminal = decode(prediction[0, ..., -1, :]).cpu().numpy()
        endpoints.append(terminal)
        return score(terminal).T

    old_cost = score_raw(old)
    rng = np.random.default_rng(np.random.SeedSequence((scene.seed, root, 553)))
    proposals, costs = cem_population(
        scene, score.goal, decoded_current, HORIZON, rng,
        lambda plans: score_raw(np.repeat(plans[:, :, None], ACTION_BLOCK, axis=2)),
    )
    raw = np.concatenate((old, np.repeat(proposals[:, :, None], ACTION_BLOCK, axis=2)))
    return raw, np.concatenate(endpoints), np.concatenate((old_cost, costs), axis=1).T, decoded_current


def replay_bank(physical, start, plans):
    """Vectorized stock collisions, with canonical Env.step checked per root."""
    xy = np.broadcast_to(start, (2, len(plans), 2))
    trace = []
    for controls in plans.reshape(len(plans), -1, 2).transpose(1, 0, 2):
        xy = physical.advance(xy, controls)
        trace.append(xy)
    return np.stack(trace, axis=2).transpose(1, 0, 2, 3).reshape(
        len(plans), 2, HORIZON, ACTION_BLOCK, 2,
    )


def onset_raw(traces):
    steps = traces.reshape(*traces.shape[:-3], -1, 2)
    changed = (steps[..., 0, :, :] != steps[..., 1, :, :]).any(axis=-1)
    return np.where(changed.any(axis=-1), changed.argmax(axis=-1), -1).astype(np.int8)


def select_coverage(physical, start, goal, root, score, raw, actual, parent_xy, parent_cost):
    retained = (1, 3) if root < 2 else (0, 1)
    corner_xy = np.where(start < CENTER, physical.near - .5, physical.far + .5)
    terminal = actual[..., -1, -1, :]
    corner = (terminal == corner_xy).all(axis=(1, 2))
    separated = (actual[:, 0, :, -1] != actual[:, 1, :, -1]).any(axis=(1, 2))
    false_goal = (room(parent_xy) == room(goal)).any(axis=1)
    optimistic = parent_cost.min(axis=1)
    physical_best = score(terminal).min(axis=1)
    distinct = np.array([not any(np.array_equal(plan, raw[i]) for i in retained) for plan in raw])
    distinct[:4] = False
    if (corner & distinct).any():
        matching = np.flatnonzero(corner & distinct)
        first = matching[np.lexsort((matching, optimistic[matching], ~false_goal[matching]))[0]]
        first_role = 2
    else:
        first = np.flatnonzero(distinct)[optimistic[distinct].argmin()]
        first_role = 4
    distinct &= (raw != raw[first]).any(axis=(1, 2, 3))
    if (separated & distinct).any():
        matching = np.flatnonzero(separated & distinct)
        second = matching[physical_best[matching].argmin()]
        second_role = 3
    else:
        second = np.flatnonzero(distinct)[optimistic[distinct].argmin()]
        second_role = 5
    return (np.asarray((*retained, first, second)),
            np.asarray((1, 1, first_role, second_role), dtype=np.int8), corner, separated, false_goal)


def execute_selected(env, physical, start, goal, raw, full, chosen):
    """Verify every saved supervision trajectory and fixed sentinels against stock transitions."""
    sentinels = (4 + np.arange(ITERATIONS)[:, None] * SAMPLES + np.array((0, SAMPLES // 2))
                 if len(raw) > 4 else np.empty(0, dtype=int))
    verify = np.unique(np.concatenate((chosen, sentinels.ravel())))
    for mode in (0, 1):
        env.begin(start, goal, mode)
        for index in verify:
            # There is no hidden dynamical state besides position and lock. The
            # actual common root is restored, with fixed geometry and goal.
            env.agent_position = torch.from_numpy(start.copy())
            observed = np.stack([env.execute(command) for command in raw[index]])
            np.testing.assert_array_equal(observed, full[index, mode])

    local = np.full((len(chosen), 2, HORIZON, 2, ACTION_BLOCK, 2), np.nan, dtype=np.float32)
    valid = np.ones(local.shape[:3], dtype=bool)
    actual = full[chosen]
    for i, index in enumerate(chosen):
        for mode in (0, 1):
            for horizon, command in enumerate(raw[index]):
                source = actual[i, mode, horizon - 1, -1] if horizon else start
                x, y = source
                # Inside the expanded vertical slab, closing its occupied
                # aperture would overlap the disk with a wall. Boundary touch
                # is legal in the canonical transition's strict inequalities.
                alternate = 1 - mode
                if (physical.near < x < physical.far
                        and abs(y - physical.lock_centers[alternate, 0]) > physical.aperture):
                    valid[i, mode, horizon] = False
                    continue
                for counterfactual in (0, 1):
                    env.set_mode(counterfactual)
                    env.agent_position = torch.from_numpy(source.copy())
                    local[i, mode, horizon, counterfactual] = env.execute(command)
                np.testing.assert_array_equal(local[i, mode, horizon, mode], actual[i, mode, horizon])
    changed = (local[..., 0, :, :] != local[..., 1, :, :]).any(axis=-1)
    onset = np.where(valid & changed.any(axis=-1), changed.argmax(axis=-1), -1).astype(np.int8)
    np.testing.assert_array_equal(valid[:, 0, 0], valid[:, 1, 0])
    np.testing.assert_array_equal(local[:, 0, 0], local[:, 1, 0])
    return local, valid, onset, 2 * len(verify)


def encode_shortlist(env, host, common, selected, batch):
    future = selected["future_xy"][..., -1, :]
    local = selected["local_xy"][..., -1, :]
    valid = selected["local_valid"]
    (future_z, local_valid_z), frames = encode_groups(
        env, host, (future, local[valid]), batch, known=(common["root_xy"], common["root_z"]),
    )
    local_z = np.full((*local.shape[:-1], 192), np.nan, dtype=np.float32)
    local_z[valid] = local_valid_z
    for mode in (0, 1):
        available = valid[:, :, mode]
        np.testing.assert_array_equal(local_z[:, :, mode, :, mode][available],
                                      future_z[:, :, mode][available])
    return dict(common, **selected, future_z=future_z, local_z=local_z), frames


def collect_layout(seed, model, decode, batch, reference=None):
    scene = make_scene(seed)
    env = ReversibleFourRoom(scene)
    physical = PhysicalRollout(scene)
    host = model[0]
    try:
        root_xy, root_action = actual_common_windows(scene, env)
        env.begin(scene.northwest, scene.southeast, 0)
        (root_z,), root_frames = encode_groups(env, host, (root_xy,), batch)
        common = dict(seed=np.int64(seed), root_xy=root_xy, root_action=root_action, root_z=root_z)
        bank_roots, union_roots, selected, roles = [], [], [], []
        verified = 0
        for root, window in enumerate(root_xy):
            goal = scene.southeast if root in (0, 2) else scene.northwest
            start = window[-1]
            score = VisibleMapCost(scene, goal)
            old = original_h5(scene, physical, start, goal, root)
            raw, parent_xy, parent_cost, decoded_current = query_bank(
                scene, score, root, root_z[root], root_action[root], old, model, decode,
            )
            future = replay_bank(physical, start, raw)
            ids, role, corner, separated, false_goal = select_coverage(
                physical, start, goal, root, score, raw, future, parent_xy, parent_cost,
            )
            selected.append(ids)
            roles.append(role)
            union = np.unique(np.concatenate((np.arange(4), ids)))
            local, valid, local_onset, n_verified = execute_selected(env, physical, start, goal, raw, future, union)
            verified += n_verified
            bank = dict(action=raw, future_xy=future, parent_xy=parent_xy, parent_cost=parent_cost,
                        common_corner=corner, block_end_separated=separated, parent_goal_room=false_goal,
                        first_branch_divergence_raw=onset_raw(future))
            bank_roots.append(dict(bank, decoded_current=decoded_current))
            used = np.concatenate((np.arange(4), ids))
            local_index = np.searchsorted(union, used)
            local_arrays = dict(local_xy=local[local_index], local_valid=valid[local_index],
                                first_local_divergence_raw=local_onset[local_index])
            union_roots.append(dict({key: value[used] for key, value in bank.items()}, **local_arrays))

        bank = dict(common, **stack_rows(bank_roots), coverage_bank_index=np.stack(selected),
                    coverage_role=np.stack(roles))
        bank_index = np.concatenate((np.tile(np.arange(4), (len(root_xy), 1)), np.stack(selected)), axis=1)
        selection = np.concatenate((np.zeros((len(root_xy), 4), dtype=np.int8), np.stack(roles)), axis=1)
        # The shortlist contains exact indices into the immutable parent bank.
        shortlist = dict(**stack_rows(union_roots), bank_index=bank_index, selection_role=selection)
        env.begin(scene.northwest, scene.southeast, 0)
        combined, other_frames = encode_shortlist(env, host, common, shortlist, batch)
        if reference is not None:
            np.testing.assert_array_equal(root_xy, reference["root_xy"])
            np.testing.assert_array_equal(root_action, reference["root_action"])
            np.testing.assert_array_equal(combined["action"][:, :4, :4], reference["action"])
            np.testing.assert_array_equal(combined["future_xy"][:, :4, :, :4], reference["future_xy"])
        return combined, bank, verified, root_frames + other_frames
    finally:
        env.close()


def stack_rows(rows):
    return {key: np.stack([row[key] for row in rows]) for key in rows[0]}


def slice_candidates(data, candidate_slice):
    common = {"seed", "root_xy", "root_action", "root_z"}
    return {key: value if key in common else value[:, :, candidate_slice] for key, value in data.items()}


def save_npz(path, payload):
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def summarize(data, bank, verified, frames):
    starts = data["local_xy"]
    informative = (starts[..., 0, -1, :] != starts[..., 1, -1, :]).any(axis=-1) & data["local_valid"]
    count_by_root = lambda mask: mask.sum(axis=(0, 2)).tolist()
    return {
        "layout_seeds": data["seed"].tolist(),
        "root_names": ROOT_NAMES,
        "slot_roles_by_root": {role: count_by_root(data["selection_role"] == i) for i, role in enumerate(ROLES)},
        "shortlist_common_corner_by_root": count_by_root(data["common_corner"]),
        "shortlist_block_end_separated_by_root": count_by_root(data["block_end_separated"]),
        "shortlist_corner_parent_goal_room_by_root": count_by_root(data["common_corner"] & data["parent_goal_room"]),
        "cem_bank_common_corner_by_root": count_by_root(bank["common_corner"][..., 4:]),
        "cem_bank_block_end_separated_by_root": count_by_root(bank["block_end_separated"][..., 4:]),
        "cem_bank_corner_parent_goal_room_by_root": count_by_root(
            (bank["common_corner"] & bank["parent_goal_room"])[..., 4:]),
        "valid_local_pairs_by_horizon": data["local_valid"].sum(axis=(0, 1, 2, 3)).tolist(),
        "valid_informative_local_pairs_by_horizon": informative.sum(axis=(0, 1, 2, 3)).tolist(),
        "actual_canonical_env_trajectory_checks": verified,
        "unique_encoded_rgb_frames": frames,
        "shapes": {key: list(value.shape) for key, value in data.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    model = load_trained(RUN, "conditioned", args.device)
    decode = load_readout(args.device, RUN)
    with np.load(RUN / "data_train_physical.npz", allow_pickle=False) as old:
        np.testing.assert_array_equal(old["seed"][:len(TRAIN_SEEDS)], TRAIN_SEEDS)
        reference = {key: old[key][:len(TRAIN_SEEDS)] for key in ("root_xy", "root_action", "action", "future_xy")}
    summaries = {}
    for split, seeds in (("eval", EVAL_SEEDS), ("train", TRAIN_SEEDS)):
        shortlists, banks, checks, rgb = [], [], 0, 0
        for i, seed in enumerate(seeds):
            old_row = {key: value[i] for key, value in reference.items()} if split == "train" else None
            shortlist, bank, checked, frames = collect_layout(seed, model, decode, args.batch, old_row)
            shortlists.append(shortlist)
            banks.append(bank)
            checks += checked
            rgb += frames
            print(f"{split} {i + 1}/{len(seeds)} seed {seed}: {frames} RGB, {checked} Env.step tracks",
                  flush=True)
        data, bank = stack_rows(shortlists), stack_rows(banks)
        save_npz(args.output / f"data_{split}_bank.npz", bank)
        summaries[split] = summarize(data, bank, checks, rgb)
        if split == "eval":
            save_npz(args.output / "data_eval.npz", data)
        else:
            for arm, candidates_slice in (("original", slice(0, 4)), ("coverage", slice(4, 8))):
                arm_data = slice_candidates(data, candidates_slice)
                save_npz(args.output / f"data_train_{arm}.npz", arm_data)
                summaries[arm] = summarize(arm_data, bank, checks, rgb)
        (args.output / "data_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
