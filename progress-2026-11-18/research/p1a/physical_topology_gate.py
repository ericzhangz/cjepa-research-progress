"""Gate 0: real FourRoom physics, reversible invisible locks, one shared CEM."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from stable_worldmodel.envs.four_room.env import DOOR_ORDER, FourRoomEnv


ACTION_BLOCK = 5
SPEED = 2.5
RADIUS = 3.5
HALF_WALL = 5
DOOR_SIZE = 16
CENTER = 112.0
DOOR_CLEARANCE = 1.75
GOAL_RADIUS = 16.0
DEADLINE = 12
HORIZONS = (4, 5, 6)
SAMPLES = 96
ITERATIONS = 4
ELITES = 12
SEEDS = tuple(range(2026111500, 2026111564))
ARMS = ("true", "uniform", "keep_old")
LOCK = ("bottom", "top")
# Clockwise room order: top-left, top-right, bottom-right, bottom-left.
RING_DOORS = ("top", "right", "bottom", "left")


@dataclass(frozen=True)
class Scene:
    seed: int
    door_positions: dict[str, int]
    northwest: np.ndarray
    southeast: np.ndarray


def make_scene(seed):
    rng = np.random.default_rng(seed)
    doors = {
        door: int(rng.integers(low, high + 1))
        for door, low, high in (
            ("top", 73, 82), ("bottom", 142, 151),
            ("left", 73, 82), ("right", 142, 151),
        )
    }
    return Scene(seed, doors, rng.integers(81, 88, 2).astype(np.float32),
                 rng.integers(137, 144, 2).astype(np.float32))


class ReversibleFourRoom(FourRoomEnv):
    """Only the stock collision call sees the lock. Rendering shows both openings."""

    def __init__(self, scene):
        super().__init__(
            agent_speed=SPEED, agent_radius=RADIUS, wall_thickness=2 * HALF_WALL,
            door_size=[DOOR_SIZE] * 4, door_positions=scene.door_positions,
            close_door_prob=0.0, terminate_on_goal=False,
        )
        self.scene = scene
        self.mode = 0

    def begin(self, start, goal, mode):
        self.set_mode(mode)
        self.reset(seed=self.scene.seed, options={"state": start, "target_state": goal})

    def set_mode(self, mode):
        if mode not in (0, 1):
            raise ValueError("Expected complementary lock mode 0 or 1")
        self.mode = mode

    def _apply_collisions(self, pos1, pos2):
        previous = self.closed_door
        self.closed_door = LOCK[self.mode]
        try:
            return super()._apply_collisions(pos1, pos2)
        finally:
            self.closed_door = previous

    @property
    def xy(self):
        return self.agent_position.numpy().copy()

    def execute(self, action_block):
        if action_block.shape != (ACTION_BLOCK, 2) or not np.all(np.abs(action_block) <= 1):
            raise ValueError("Expected one block of five legal native raw actions")
        points = []
        for action in action_block:
            self.step(action)
            points.append(self.xy)
        return np.stack(points)


def block(control):
    return np.repeat(np.asarray(control, dtype=np.float32)[None], ACTION_BLOCK, axis=0)


class PhysicalRollout:
    """Batched stock collision equations, checked against actual Env.step."""

    def __init__(self, scene):
        self.top, self.bottom, self.left, self.right = (
            np.float32(scene.door_positions[name]) for name in DOOR_ORDER
        )
        self.min_xy = float(FourRoomEnv.BORDER_SIZE) + RADIUS
        self.max_xy = float(FourRoomEnv.IMG_SIZE) - self.min_xy
        self.near = CENTER - HALF_WALL - RADIUS
        self.far = CENTER + HALF_WALL + RADIUS
        self.aperture = DOOR_SIZE + DOOR_CLEARANCE
        self.lock_centers = np.array([self.top, self.bottom], dtype=np.float32)[:, None]

    def advance(self, xy, action, mode_index=slice(None)):
        x, y = xy[..., 0], xy[..., 1]
        next_x, next_y = np.clip(
            xy + action * SPEED, self.min_xy, self.max_xy,
        ).transpose(2, 0, 1)
        vertical = np.abs(next_y - self.lock_centers[mode_index]) <= self.aperture
        next_x = np.where((x < CENTER) & (next_x > self.near) & ~vertical, self.near - .5, next_x)
        next_x = np.where((x >= CENTER) & (next_x < self.far) & ~vertical, self.far + .5, next_x)
        horizontal = ((np.abs(next_x - self.left) <= self.aperture) |
                      (np.abs(next_x - self.right) <= self.aperture))
        next_y = np.where((y < CENTER) & (next_y > self.near) & ~horizontal, self.near - .5, next_y)
        next_y = np.where((y >= CENTER) & (next_y < self.far) & ~horizontal, self.far + .5, next_y)
        return np.stack((next_x, next_y), axis=-1)

    def terminal(self, start, plans):
        xy = np.broadcast_to(start, (2, len(plans), 2))
        for controls in plans.transpose(1, 0, 2):
            for _ in range(ACTION_BLOCK):
                xy = self.advance(xy, controls)
        return xy


def room(xy):
    right = xy[..., 0] >= CENTER
    bottom = xy[..., 1] >= CENTER
    return np.where(bottom, np.where(right, 2, 3), right.astype(np.int64))


def door_xy(scene, name):
    position = scene.door_positions[name]
    return np.array((CENTER, position) if name in ("top", "bottom") else
                    (position, CENTER), dtype=np.float32)


def doors_between(start_room, target_room, direction):
    count = ((target_room - start_room) * direction) % 4
    return tuple(RING_DOORS[(start_room + direction * i - (direction == -1)) % 4]
                 for i in range(count))


class VisibleMapCost:
    """Stock door-centre path length, independent of the invisible lock mode."""

    def __init__(self, scene, goal):
        self.goal = goal
        target_room = int(room(goal))
        self.paths = {}
        for source_room in range(4):
            choices = []
            for direction in (1, -1):
                entries = doors_between(source_room, target_room, direction)
                points = [door_xy(scene, door) for door in entries] + [goal]
                residual = sum(np.linalg.norm(end - begin) for begin, end in zip(points, points[1:]))
                choices.append((points[0], residual))
            self.paths[source_room] = choices

    def __call__(self, xy):
        locations = room(xy)
        cost = np.empty(xy.shape[:-1], dtype=np.float32)
        for source_room, choices in self.paths.items():
            mask = locations == source_room
            points = xy[mask]
            cost[mask] = np.minimum(*(np.linalg.norm(points - first, axis=-1) + residual
                                      for first, residual in choices))
        return cost


def route_points(scene, start, goal, direction):
    source = int(room(start))
    doors = doors_between(source, int(room(goal)), direction)
    points = []
    for door in doors:
        center = door_xy(scene, door)
        axis = 0 if door in ("top", "bottom") else 1
        before = center.copy()
        before[axis] += (-10 if source in ((0, 3) if axis == 0 else (0, 1)) else 10)
        points.extend((before, 2 * center - before))
        source = (source + direction) % 4
    return points + [goal]


def route_seed(scene, start, goal, direction, horizon):
    """Visible-map centrelines give both CEM components equally legal warm starts."""
    position = start.copy()
    commands = []
    for target in route_points(scene, start, goal, direction):
        while np.max(np.abs(target - position)) > 1e-4 and len(commands) < horizon:
            command = np.clip((target - position) / (SPEED * ACTION_BLOCK), -1, 1)
            commands.append(command)
            position += command * SPEED * ACTION_BLOCK
    return np.asarray(commands + [np.zeros(2, dtype=np.float32)] * (horizon - len(commands)))


def cem_population(scene, goal, start, horizon, rng, candidate_cost):
    """Two equal proposal components, all candidates scored under both modes.

    One component optimizes each hypothesis. The proposal and raw sample draws are
    independent of which policy requests the action. All policies can select any
    candidate in the pooled bank using their specified beliefs.
    """
    means = np.stack([route_seed(scene, start, goal, direction, horizon)
                      for direction in (1, -1)])
    spreads = np.full_like(means, 0.65)
    plans_seen, costs_seen = [], []
    for _ in range(ITERATIONS):
        noise = rng.standard_normal((2, SAMPLES // 2, horizon, 2), dtype=np.float32)
        proposals = np.clip(means[:, None] + noise * spreads[:, None], -1, 1)
        proposals[:, 0] = means
        plans = proposals.reshape(SAMPLES, horizon, 2)
        costs = candidate_cost(plans)
        if costs.shape != (2, SAMPLES) or not np.isfinite(costs).all():
            raise ValueError("Each CEM candidate must have two finite hypothesis costs")
        plans_seen.append(plans)
        costs_seen.append(costs)
        # Both components may inherit elites from either part of the shared bank.
        elites = plans[np.argsort(costs, axis=1, kind="stable")[:, :ELITES]]
        means = elites.mean(axis=1)
        spreads = np.maximum(elites.std(axis=1), 0.10)
    return np.concatenate(plans_seen), np.concatenate(costs_seen, axis=1)


def shared_cem(scene, physical, score, start, horizon, rng):
    return cem_population(
        scene, score.goal, start, horizon, rng,
        lambda plans: score(physical.terminal(start, plans)),
    )


def select(costs, arm, current_mode, old_mode):
    scores = {"true": costs[current_mode], "uniform": costs.mean(axis=0),
              "keep_old": costs[old_mode]}[arm]
    return int(np.argmin(scores))


def record_execute(env, commands, event_log, tag):
    before = env.xy
    aftermath = env.execute(commands)
    event_log.append({"phase": tag, "start": before.tolist(), "raw_xy": aftermath.tolist(),
                      "raw_actions": commands.tolist()})
    return aftermath


def move_to(env, target, event_log, phase):
    """Script legal raw controls towards a visible waypoint."""
    target = np.asarray(target, dtype=np.float32)
    for _ in range(DEADLINE):
        if np.array_equal(env.xy, target):
            return
        position = env.xy
        commands = []
        for _ in range(ACTION_BLOCK):
            command = np.clip((target - position) / SPEED, -1, 1)
            commands.append(command)
            position += command * SPEED
        record_execute(env, np.stack(commands), event_log, phase)
    raise RuntimeError(f"Unreachable factual waypoint {target} from {env.xy}")


def probe_and_dormancy(env, anchor, event_log, phase):
    """Execute a physical doorway probe, return, and take unrelated movements."""
    from_left = anchor[0] < CENTER
    probe_y = env.scene.door_positions["top" if from_left else "bottom"]
    probe_start = (100 if from_left else 124, probe_y)
    move_to(env, probe_start, event_log, phase + "_approach")
    pre = env.render().copy()
    post_xy = record_execute(env, block((1 if from_left else -1, 0)), event_log, phase + "_probe")[-1]
    post = env.render().copy()
    crossed = (post_xy[0] >= CENTER) == from_left
    open_mode = 0 if from_left else 1
    if crossed != (env.mode == open_mode):
        raise AssertionError("Actual probe did not identify the lock")
    move_to(env, probe_start, event_log, phase + "_retreat")
    move_to(env, anchor, event_log, phase + "_return")
    dormant = len(event_log)
    away = -1 if from_left else 1
    move_to(env, anchor + np.array((15 * away, 0)), event_log, phase + "_dormant_out")
    move_to(env, anchor, event_log, phase + "_dormant_back")
    for _ in range(3):
        record_execute(env, block((0, 0)), event_log, phase + "_dormant_hold")
    return {"probe_crossed": bool(crossed), "probe_terminal": post_xy.tolist(),
            "dormant_raw_steps": (len(event_log) - dormant) * ACTION_BLOCK}, np.stack((pre, post, env.render()))


def plan_leg(env, physical, goal, arm, old_mode, horizon, rng_seed):
    """Replan after each legal five-step block, ending on actual radius crossing."""
    cost = VisibleMapCost(env.scene, goal)
    ticks = []
    for index in range(DEADLINE):
        before = env.xy
        rng = np.random.default_rng(np.random.SeedSequence([rng_seed, index]))
        candidates, scores = shared_cem(env.scene, physical, cost, before, horizon, rng)
        decisions = {name: select(scores, name, env.mode, old_mode) for name in ARMS}
        choice = decisions[arm]
        current_cost = scores[env.mode]
        vector_command = candidates[choice, 0]
        path = env.execute(block(vector_command))
        distance = np.linalg.norm(path - goal, axis=-1)
        ticks.append({
            "start": before.tolist(), "xy": path.tolist(), "control": vector_command.tolist(),
            "chosen": choice, "common_bank_selections": decisions,
            "within_bank_regret": {name: float(current_cost[i] - current_cost[decisions["true"]])
                                   for name, i in decisions.items()},
            "best_physical_distance": float(distance.min()),
        })
        if np.any(distance < GOAL_RADIUS):
            break
    best_distance = min(tick["best_physical_distance"] for tick in ticks)
    return {"success": best_distance < GOAL_RADIUS, "best_physical_distance": best_distance,
            "final_xy": env.xy.tolist(), "blocks": len(ticks), "ticks": ticks}


def switched_root(scene, initial_mode, physical, horizon, arm):
    """An independent same-origin assessment, with the old fact actually observed."""
    env = ReversibleFourRoom(scene)
    start, target = scene.northwest, scene.southeast
    env.begin(start, target, initial_mode)
    timeline = []
    old_evidence, _ = probe_and_dormancy(env, start, timeline, "old")
    switch_xy = env.xy.tolist()
    env.set_mode(1 - initial_mode)
    new_evidence, _ = probe_and_dormancy(env, start, timeline, "reversed")
    result = plan_leg(env, physical, target, arm, initial_mode, horizon, scene.seed)
    env.close()
    return {**result, "switch_xy": switch_xy, "old_evidence": old_evidence,
            "new_evidence": new_evidence, "prefix": timeline}


def macro(scene, initial_mode, physical, horizon, arm):
    """One continuous out-and-back episode with a real switch beyond the walls."""
    env = ReversibleFourRoom(scene)
    start, target = scene.northwest, scene.southeast
    env.begin(start, target, initial_mode)
    timeline = []
    evidence, _ = probe_and_dormancy(env, start, timeline, "initial")
    outbound = plan_leg(env, physical, target, arm, initial_mode, horizon, scene.seed)
    if not outbound["success"]:
        env.close()
        return {"success": False, "outbound": outbound, "return": None,
                "outbound_evidence": evidence, "prefix": timeline}
    returned_timeline = []
    move_to(env, target, returned_timeline, "reached_depot")
    env.set_mode(1 - initial_mode)
    env._set_goal_state(start)
    return_evidence, _ = probe_and_dormancy(env, target, returned_timeline, "reversed")
    response = plan_leg(env, physical, start, arm, initial_mode, horizon, scene.seed + 1)
    env.close()
    return {"success": response["success"], "outbound": outbound, "return": response,
            "outbound_evidence": evidence, "return_evidence": return_evidence,
            "prefix": timeline, "switch_and_return_prefix": returned_timeline}


def physical_contract(scene):
    """Check vector physics and executable probes before looking at planning results."""
    env = ReversibleFourRoom(scene)
    physical = PhysicalRollout(scene)
    rng = np.random.default_rng(96513)
    checked_positions = np.vstack([
        rng.uniform(15, 209, (256, 2)),
        [(x, y) for x in (103.0, 103.5, 111.9, 112.0, 120.5, 121.0)
         for center in (physical.top, physical.bottom)
         for y in (center, center - physical.aperture, center + physical.aperture)],
    ]).astype(np.float32)
    controls = rng.uniform(-1, 1, checked_positions.shape).astype(np.float32)
    predictions = physical.advance(np.broadcast_to(checked_positions, (2, *checked_positions.shape)), controls)
    evidence_frames, evidence_records = [], []
    for mode in (0, 1):
        env.begin(scene.northwest, scene.southeast, mode)
        for pos, control, expected in zip(checked_positions, controls, predictions[mode]):
            # Broad one-step source fixture, including wall-adjacent float cases.
            env.agent_position = torch.from_numpy(pos.copy())
            env.step(control)
            np.testing.assert_array_equal(env.xy, expected)
        env.begin(scene.northwest, scene.southeast, mode)
        log = []
        evidence, frames = probe_and_dormancy(env, scene.northwest, log, "contract")
        evidence_frames.append(frames)
        evidence_records.append(evidence)
    a, b = evidence_frames
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[-1], b[-1])
    if np.array_equal(a[1], b[1]):
        raise AssertionError("The actual probe must separate full-view RGB")
    env.close()
    return {"scalar_advance_comparisons": 2 * len(checked_positions), "rgb_gate_start_identical": True,
            "rgb_after_probe_distinct": True, "rgb_after_dormancy_identical": True,
            "probe_outcomes": evidence_records}, np.stack(evidence_frames)


def path_witness(scene, physical):
    """A centreline path is physically feasible under the shared action/deadline."""
    budgets = []
    for start, target in ((scene.northwest, scene.southeast), (scene.southeast, scene.northwest)):
        for mode in (0, 1):
            direction = next(d for d in (1, -1)
                             if LOCK[mode] not in doors_between(int(room(start)), int(room(target)), d))
            commands = route_seed(scene, start, target, direction, DEADLINE)
            position = np.broadcast_to(start, (2, 1, 2))
            hit = None
            for tick, command in enumerate(commands):
                for _ in range(ACTION_BLOCK):
                    position = physical.advance(position, command)
                    if np.linalg.norm(position[mode, 0] - target) < GOAL_RADIUS and hit is None:
                        hit = tick + 1
            if hit is None:
                raise AssertionError(f"Designed geometry is infeasible in scene {scene.seed}, mode {mode}")
            budgets.append(hit)
    return max(budgets)


def run_scene(seed):
    torch.set_num_threads(1)
    scene = make_scene(seed)
    physical = PhysicalRollout(scene)
    witness = path_witness(scene, physical)
    result = {"seed": seed, "doors": scene.door_positions, "northwest": scene.northwest.tolist(),
              "southeast": scene.southeast.tolist(), "witness_max_blocks": witness, "horizons": {}}
    for h in HORIZONS:
        result["horizons"][str(h)] = {
            str(mode): {arm: {"matched_switch": switched_root(scene, mode, physical, h, arm),
                             "macro": macro(scene, mode, physical, h, arm)}
                        for arm in ARMS}
            for mode in (0, 1)
        }
    return result


def exact_p(wins, losses):
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant
    return min(1.0, 2 * tail)


def summarize(rows):
    # Resample scenes as clusters; their two mode episodes share geometry.
    resampling = np.random.default_rng(11752026).integers(0, len(rows), (10000, len(rows)))
    output = {}
    for horizon in HORIZONS:
        data = [row["horizons"][str(horizon)] for row in rows]
        h_summary = {}
        for name in ("matched_switch", "outbound", "macro"):
            def leg(episode, arm):
                if name == "outbound":
                    return episode[arm]["macro"]["outbound"]
                return episode[arm][name]

            def gather(field, arm):
                return np.array([[leg(scene[str(mode)], arm)[field] for mode in (0, 1)]
                                 for scene in data])

            metrics = {
                "success": {},
                "paired_success": {},
                "episodes": 2 * len(rows),
            }
            true = gather("success", "true")
            for arm in ARMS:
                baseline = gather("success", arm)
                metrics["success"][arm] = {"count": int(baseline.sum()), "rate": float(baseline.mean())}
                if name != "macro":
                    metrics["success"][arm].update({
                        "mean_best_distance": float(gather("best_physical_distance", arm).mean()),
                        "mean_blocks": float(gather("blocks", arm).mean()),
                    })
                if arm == "true":
                    continue
                deltas = true.astype(float) - baseline
                bootstrap = deltas.mean(axis=1)[resampling].mean(axis=1)
                wins = int(np.sum(deltas == 1))
                losses = int(np.sum(deltas == -1))
                scene_delta = deltas.mean(axis=1)
                positive = int(np.sum(scene_delta > 0))
                negative = int(np.sum(scene_delta < 0))
                metrics["paired_success"][arm] = {
                    "gap": float(deltas.mean()), "cluster_bootstrap_95ci": np.quantile(bootstrap, [.025, .975]).tolist(),
                    "paired_episode_wins": wins, "paired_episode_losses": losses,
                    "scene_sign_test_p": exact_p(positive, negative),
                    "paired_scene_wins": positive, "paired_scene_losses": negative,
                }
                if name != "macro":
                    paired_rows = [leg(scene[str(mode)], "true")["ticks"][0]
                                   for scene in data for mode in (0, 1)]
                    metrics["paired_success"][arm]["first_common_root_regret"] = float(np.mean([
                        tick["within_bank_regret"][arm] for tick in paired_rows
                    ]))
            if name == "macro":
                metrics["returned_after_reaching_outbound"] = {
                    arm: sum(scene[str(mode)][arm]["macro"]["return"] is not None and
                             scene[str(mode)][arm]["macro"]["return"]["success"]
                             for scene in data for mode in (0, 1))
                    for arm in ARMS
                }
            h_summary[name] = metrics
        output[str(horizon)] = h_summary
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/physical_topology_gate_20261115"))
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    contract, frames = physical_contract(make_scene(2026111400))
    (args.output / "physical_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    np.savez_compressed(args.output / "contract_rgb.npz", frames=frames)
    if args.contract_only:
        print(json.dumps(contract, indent=2))
        return

    started = time.perf_counter()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            (args.output / "scenes.jsonl").open("w", encoding="utf-8") as file:
        for row in pool.map(run_scene, SEEDS, chunksize=1):
            rows.append(row)
            file.write(json.dumps(row, separators=(",", ":")) + "\n")
            file.flush()
            print(f"Processed {len(rows)}/{len(SEEDS)} scene seeds", flush=True)

    metrics = summarize(rows)
    primary = metrics["5"]
    switched = primary["matched_switch"]
    uniform = switched["paired_success"]["uniform"]
    keep_old = switched["paired_success"]["keep_old"]
    passed = (switched["success"]["true"]["rate"] >= .80 and uniform["gap"] >= .20 and
              uniform["cluster_bootstrap_95ci"][0] > 0 and keep_old["gap"] > 0)
    summary = {"settings": {"horizons": HORIZONS, "primary_horizon": 5, "seeds": SEEDS,
                            "deadline_blocks_per_leg": DEADLINE, "action_block": ACTION_BLOCK,
                            "samples": SAMPLES, "iterations": ITERATIONS, "elites_per_mode": ELITES,
                            "rollouts_per_candidate": 2, "gate_primary": "matched_switch"},
               "contract": contract, "max_witness_blocks": max(row["witness_max_blocks"] for row in rows),
               "gate_passed": passed, "metrics": metrics, "elapsed_seconds": time.perf_counter() - started}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
