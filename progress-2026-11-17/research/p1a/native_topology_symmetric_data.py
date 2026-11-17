"""Select both physical gate directions and a static blocked/legal pair from the archived H5 banks."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .native_topology_condition import load_host
from .native_topology_paired_data import OUTPUT as PAIRED, clipping
from .native_topology_prefix_data import (
    OUTPUT as PREFIX, ROOT_NAMES, TRAIN_SEEDS, encode_shortlist, execute_selected, onset_raw, save_npz,
)
from .physical_topology_gate import ACTION_BLOCK, CENTER, PhysicalRollout, ReversibleFourRoom, VisibleMapCost, make_scene, room


OUTPUT = PAIRED.with_name("native_topology_symmetric_20261116")
ARCHIVES = (
    PAIRED / "data_train_paired.npz", PREFIX / "data_train_coverage.npz", PREFIX / "data_train_original.npz",
    PAIRED / "data_train_refresh_bank.npz", PREFIX / "data_train_bank.npz",
)
ROLES = ("upper_open_witness", "lower_open_witness", "common_blocked_corner", "common_legal_room_change")
ARCHIVED = ("action", "future_xy", "parent_xy", "parent_cost")


def read_archives():
    loaded = []
    for path in ARCHIVES:
        with np.load(path, allow_pickle=False) as f:
            np.testing.assert_array_equal(f["seed"], TRAIN_SEEDS)
            loaded.append(dict(f) if "local_z" in f else {key: f[key] for key in ARCHIVED})
    for data in loaded[1:3]:
        for key in ("root_xy", "root_z", "root_action"):
            np.testing.assert_array_equal(loaded[0][key], data[key])
    return loaded


def select(archive, i, root, physical, score):
    root_data = archive[0]
    start = root_data["root_xy"][i, root, -1]
    groups = [{k: data[k][i, root - 2 if j == 3 else root] for k in ARCHIVED}
              for j, data in enumerate(archive)]
    raw, xy, parent_xy, parent_cost = (np.concatenate([data[key] for data in groups]) for key in ARCHIVED)
    lengths = [len(data["action"]) for data in groups]
    source = np.repeat(np.arange(len(groups)), lengths)
    source_index = np.concatenate([np.arange(n) for n in lengths])
    actual = xy.reshape(len(raw), 2, -1, 2)
    terminal = actual[:, :, -1]
    cost = score(terminal)
    start_cost = score(start)
    clips = clipping(start, raw, xy)
    hit = clips.any(axis=-1)
    crossed = (actual[..., 0] >= CENTER) != (start[0] >= CENTER)
    advance = cost < start_cost
    changed = (actual[:, 0] != actual[:, 1]).any(axis=-1)
    first = changed.argmax(axis=-1)
    boundary = np.where(start < CENTER, physical.near - .5, physical.far + .5)
    first_clipped_x = actual[np.arange(len(raw)), :, first, 0] == boundary[0]
    side_route = (crossed[..., -1] & ~crossed[:, ::-1].any(axis=-1) & advance & (cost < cost[:, ::-1])
                  & first_clipped_x[:, ::-1] & ~first_clipped_x)
    false_goal = (room(parent_xy) == room(score.goal)) & (room(terminal) != room(score.goal))
    common = ~changed.any(axis=-1)
    common_corner = common & (terminal[:, 0] == boundary).all(axis=-1) & clips[:, 0, -ACTION_BLOCK:].any(axis=-1)
    common_legal = common & ~hit[:, 0] & (room(terminal[:, 0]) != room(start)) & advance[:, 0]

    def different_first(a, b):
        return ((raw[a, 0] != raw[b, 0]).any(axis=(-2, -1))
                & (xy[a, :, 0, -1] != xy[b, :, 0, -1]).any(axis=-1).all(axis=-1))

    a, b = [np.flatnonzero(side_route[:, m]) for m in (0, 1)]
    up, down = np.meshgrid(a, b, indexing="ij")
    upper, lower = up.ravel(), down.ravel()
    swap = (cost[upper, 0] < cost[lower, 0]) & (cost[lower, 1] < cost[upper, 1])
    clipped_routes = hit[upper, 0].astype(np.int8) + hit[lower, 1].astype(np.int8)
    ordering = np.lexsort((lower, upper, cost[upper, 0] + cost[lower, 1], clipped_routes, ~swap))
    # Alternating parity fixes the summary designation before seeing scores. Both true modes still train on all slots.
    designation = (i + root) % 2
    for rank in ordering:
        u, d = upper[rank], lower[rank]
        if not different_first(u, d):
            continue
        blocked = np.flatnonzero(common_corner & different_first(np.arange(len(raw)), u)
                                 & different_first(np.arange(len(raw)), d)
                                 & (cost[:, 0] > max(cost[u, 0], cost[d, 1])))
        if not len(blocked):
            continue
        priorities = np.lexsort((blocked, parent_cost[blocked, designation], ~false_goal[blocked, designation]))
        for stop in blocked[priorities]:
            legal = np.flatnonzero(common_legal & different_first(np.arange(len(raw)), stop)
                                   & (cost[:, 0] < cost[stop, 0]))
            if len(legal):
                go = legal[cost[legal, 0].argmin()]
                break
        else:
            continue
        break
    else:
        raise ValueError(f"Archived candidates lack a complete physical quartet: {TRAIN_SEEDS[i]} {ROOT_NAMES[root]}")

    chosen = np.asarray((u, d, stop, go))
    record = {
        "seed": TRAIN_SEEDS[i], "root": ROOT_NAMES[root], "failure_designation": designation,
        "archive_sources": [ARCHIVES[j].name for j in source[chosen]], "archive_indices": source_index[chosen].tolist(),
        "true_h1_xy": xy[chosen, :, 0, -1].tolist(), "true_h5_xy": terminal[chosen].tolist(),
        "true_h5_cost": cost[chosen].tolist(), "first_divergent_raw_step": onset_raw(xy[chosen]).tolist(),
        "route_open_unclipped": [bool(~hit[u, 0]), bool(~hit[d, 1])],
        "route_strict_order_reversal": bool(swap[rank]),
        "common_blocked_model_false_goal": false_goal[stop].tolist(),
    }
    values = dict(zip(ARCHIVED, (raw[chosen], xy[chosen], parent_xy[chosen], parent_cost[chosen]), strict=True))
    return values, source[chosen], source_index[chosen], record


def assemble(archive, host, output, batch):
    panel = {key: value.copy() for key, value in archive[0].items()}
    panel["archive_source"] = np.zeros(panel["bank_index"].shape, dtype=np.int8)
    panel["archive_index"] = np.broadcast_to(np.arange(4, dtype=np.int16), panel["bank_index"].shape).copy()
    panel["failure_designation"] = np.full(panel["bank_index"].shape[:2], -1, dtype=np.int8)
    records, checks, frames = [], 0, 0
    for i, seed in enumerate(TRAIN_SEEDS):
        scene = make_scene(seed)
        physical, env = PhysicalRollout(scene), ReversibleFourRoom(scene)
        try:
            for root, goal in ((2, scene.southeast), (3, scene.northwest)):
                score = VisibleMapCost(scene, goal)
                values, source, index, record = select(archive, i, root, physical, score)
                records.append(record)
                start = panel["root_xy"][i, root, -1]
                # Exact raw-bank sentinels and the retained selected trajectories are checked by the existing collector.
                refresh = archive[3]
                raw = np.concatenate((refresh["action"][i, root - 2], values["action"]))
                xy = np.concatenate((refresh["future_xy"][i, root - 2], values["future_xy"]))
                selected = np.arange(len(raw) - 4, len(raw))
                local, valid, onset, n_checks = execute_selected(env, physical, start, goal, raw, xy, selected)
                checks += n_checks
                values.update(local_xy=local, local_valid=valid, first_local_divergence_raw=onset,
                              first_branch_divergence_raw=onset_raw(values["future_xy"]))

                new = source >= 3
                for slot in np.flatnonzero(~new):
                    old = archive[source[slot]]
                    for key in ("local_xy", "local_valid", "first_local_divergence_raw"):
                        np.testing.assert_array_equal(values[key][slot], old[key][i, root, index[slot]])
                    for key in ("future_z", "local_z"):
                        panel[key][i, root, slot] = old[key][i, root, index[slot]]
                if new.any():
                    env.begin(scene.northwest, scene.southeast, 0)
                    common = {key: panel[key][i, root][None] for key in ("root_xy", "root_z", "root_action")}
                    encoded, n_frames = encode_shortlist(env, host, common, {key: value[new][None]
                                                          for key, value in values.items()}, batch)
                    for key in ("future_z", "local_z"):
                        panel[key][i, root, new] = encoded[key][0]
                    frames += n_frames

                for key, value in values.items():
                    panel[key][i, root] = value
                ends = values["future_xy"][:, :, -1, -1]
                panel["common_corner"][i, root] = (ends == np.where(start < CENTER, 103., 121.)).all(axis=(1, 2))
                panel["block_end_separated"][i, root] = (values["future_xy"][:, 0, :, -1]
                                                        != values["future_xy"][:, 1, :, -1]).any(axis=(1, 2))
                panel["parent_goal_room"][i, root] = (room(values["parent_xy"]) == room(goal)).any(axis=-1)
                panel["selection_role"][i, root] = np.arange(8, 12)
                panel["archive_source"][i, root], panel["archive_index"][i, root] = source, index
                panel["bank_index"][i, root] = -1  # archive_source/archive_index name the explicit source bank.
                panel["failure_designation"][i, root] = record["failure_designation"]
        finally:
            env.close()
        print(f"{i + 1}/32 seed {seed}, {frames} cumulative encoded RGB positions", flush=True)

    for key, baseline in archive[0].items():
        unchanged = key in ("seed", "root_xy", "root_action", "root_z")
        np.testing.assert_array_equal(panel[key] if unchanged else panel[key][:, :2],
                                      baseline if unchanged else baseline[:, :2])
    assert np.isfinite(panel["future_z"]).all()
    for mode in (0, 1):
        valid = panel["local_valid"][..., mode, :]
        np.testing.assert_array_equal(panel["local_z"][..., mode, :, mode, :][valid], panel["future_z"][..., mode, :, :][valid])
    np.testing.assert_array_equal(panel["future_z"][:, 2:, 2:, 0], panel["future_z"][:, 2:, 2:, 1])
    counts = np.bincount(panel["failure_designation"][:, 2:].ravel(), minlength=2)
    np.testing.assert_array_equal(counts, (32, 32))
    informative = ((panel["local_xy"][..., 0, -1, :] != panel["local_xy"][..., 1, -1, :]).any(axis=-1)
                   & panel["local_valid"])
    summary = {
        "layout_seeds": list(TRAIN_SEEDS), "root_names": ROOT_NAMES, "navigation_roots": len(records),
        "archives": [str(path) for path in ARCHIVES], "roles": dict(zip(range(8, 12), ROLES, strict=True)),
        "true_route_witnesses_by_open_mode": [len(records)] * 2,
        "open_route_unclipped_by_mode": np.sum([r["route_open_unclipped"] for r in records], axis=0).tolist(),
        "true_strict_order_reversals": sum(r["route_strict_order_reversal"] for r in records),
        "common_blocked_legal_pairs": len(records), "failure_designations": counts.tolist(),
        "archived_model_false_goal_by_mode": np.sum([r["common_blocked_model_false_goal"] for r in records], axis=0).tolist(),
        "designated_model_false_goal_count": sum(r["common_blocked_model_false_goal"][r["failure_designation"]] for r in records),
        "changed_navigation_slots_by_root": (panel["action"][:, 2:] != archive[0]["action"][:, 2:]).any(axis=(3, 4, 5)).sum(axis=(0, 2)).tolist(),
        "valid_local_pairs_h2_h5": panel["local_valid"][..., 1:].sum(axis=(0, 1, 2, 3)).tolist(),
        "informative_local_pairs_h2_h5": informative[..., 1:].sum(axis=(0, 1, 2, 3)).tolist(),
        "canonical_trajectory_checks": checks, "encoded_rgb_positions": frames,
        "shapes": {key: list(array.shape) for key, array in panel.items()}, "rows": records,
    }
    save_npz(output / "data_train_symmetric.npz", panel)
    (output / "data_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assemble(read_archives(), load_host(args.device), args.output, args.batch)


if __name__ == "__main__":
    main()
