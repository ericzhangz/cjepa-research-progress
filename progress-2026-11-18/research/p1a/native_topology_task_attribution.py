"""Attribute FourRoom H5 selection errors using only the recorded action bank."""

import argparse
import json
from pathlib import Path

import numpy as np

from .physical_topology_gate import (
    GOAL_RADIUS, LOCK, VisibleMapCost, door_xy, doors_between, make_scene, room,
)


RUN = Path(__file__).resolve().parents[2] / "runs" / "native_topology_prefix_20261115"


def open_route_cost(scene, goal, xy, mode):
    """Offline reference: omit paths through the simulator's hidden locked door."""
    source_rooms = room(xy)
    target = int(room(goal))
    cost = np.empty(xy.shape[:-1], dtype=np.float32)
    for source in range(4):
        mask = source_rooms == source
        distances = []
        for direction in (1, -1):
            doors = doors_between(source, target, direction)
            if LOCK[mode] in doors:
                continue
            points = [door_xy(scene, door) for door in doors] + [goal]
            remainder = sum(np.linalg.norm(b - a) for a, b in zip(points, points[1:]))
            distances.append(np.linalg.norm(xy[mask] - points[0], axis=-1) + remainder)
        cost[mask] = np.min(distances, axis=0)
    return cost


def selected(scores, truth, forecasts, target, raw_distance, visible_truth, controls, ids):
    index = int(np.argmin(scores))
    # Success is evaluated at every recorded original step, as in the closed-loop task.
    best_distance = float(raw_distance[index].min())
    return {
        "candidate": int(ids[index]),
        "first_command": controls[index, 0, 0].tolist(),
        "actual_hit_radius": bool(best_distance < GOAL_RADIUS),
        "actual_best_distance_px": best_distance,
        "actual_terminal_distance_px": float(np.linalg.norm(truth[index] - target)),
        "actual_visible_regret": float(visible_truth[index] - visible_truth.min()),
        "false_goal_room": bool(room(forecasts[index]) == room(target) and room(truth[index]) != room(target)),
    }


def summarize(rows):
    metrics = ("actual_hit_radius", "actual_best_distance_px", "actual_terminal_distance_px", "actual_visible_regret", "false_goal_room")
    values = {name: [row[name] for row in rows] for name in metrics}
    return {"roots": len(rows), **{
        name: int(sum(v)) if type(v[0]) is bool else float(np.mean(v)) for name, v in values.items()
    }}


def evaluate(bank, predictions):
    full_rows, local_rows = [], []
    for layout, seed in enumerate(bank["seed"]):
        scene = make_scene(int(seed))
        for root, target in ((2, scene.southeast), (3, scene.northwest)):
            visible = VisibleMapCost(scene, target)
            ids = np.arange(bank["action"].shape[2])
            compact_ids = predictions["bank_index"][layout, root, predictions["unique_slot"][layout, root]]
            for mode in (0, 1):
                raw = bank["future_xy"][layout, root, :, mode].reshape(len(ids), -1, 2)
                physical = raw[:, -1]
                free = predictions["F_bank"][layout, root, :, mode, -1]
                physical_cost = predictions["Y_bank_cost"][layout, root, :, mode, -1]
                free_cost = predictions["F_bank_cost"][layout, root, :, mode, -1]
                np.testing.assert_allclose(visible(physical), physical_cost, atol=1e-4, rtol=0)
                np.testing.assert_allclose(visible(free), free_cost, atol=1e-4, rtol=0)
                distance = np.linalg.norm(raw - target, axis=-1)
                controls = bank["action"][layout, root]
                key = {"layout": int(seed), "root": root, "mode": mode}
                full = {**key, "pool_has_actual_hit": bool((distance < GOAL_RADIUS).any())}
                for name, xy, cost in (
                    ("F_visible", free, free_cost), ("Y_visible", physical, physical_cost),
                    ("F_open_route", free, open_route_cost(scene, target, free, mode)),
                    ("Y_open_route", physical, open_route_cost(scene, target, physical, mode)),
                ):
                    full[name] = selected(cost, physical, xy, target, distance, physical_cost, controls, ids)
                full_rows.append(full)

                small = dict(key)
                slots = predictions["unique_slot"][layout, root]
                for name in ("Y", "C", "L", "F"):
                    xy = predictions[name][layout, root, slots, mode, -1]
                    small[name] = selected(visible(xy), physical[compact_ids], xy, target, distance[compact_ids],
                                           physical_cost[compact_ids], controls[compact_ids], compact_ids)
                local_rows.append(small)

    summary = {name: summarize([row[name] for row in full_rows])
               for name in ("F_visible", "Y_visible", "F_open_route", "Y_open_route")}
    for name in ("Y_visible", "F_open_route", "Y_open_route"):
        summary[name]["first_command_changed_vs_F_visible"] = sum(
            row[name]["first_command"] != row["F_visible"]["first_command"] for row in full_rows)
    return {
        "description": "Recorded H5 roots, true-mode scoring, no new simulation. Open-route cost uses the hidden lock offline.",
        "layouts": len(bank["seed"]), "plans_per_full_pool": bank["action"].shape[2],
        "root_mode_pools_with_actual_hit": sum(row["pool_has_actual_hit"] for row in full_rows),
        "full_pool": summary, "full_pool_per_root_mode": full_rows,
        "short_pool": {name: summarize([row[name] for row in local_rows]) for name in ("Y", "C", "L", "F")},
        "short_pool_per_root_mode": local_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with np.load(RUN / "data_eval_bank.npz", allow_pickle=False) as bank, np.load(
        RUN / "evaluation_prefix_arrays.npz", allow_pickle=False,
    ) as predictions:
        np.testing.assert_array_equal(bank["seed"], predictions["seed"])
        result = evaluate(bank, predictions)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
