"""Collect the fixed P1A action bank in the official FourRoom simulator."""

import argparse
import json
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from p1a_reference import build_action_bank
from scripts.generate_fourroom_eval_tasks import (
    NoRoomPairError,
    _room_crossing_count,
    _sample_crossing_task,
)
from stable_worldmodel.envs.four_room.env import ROOM_INDEX, FourRoomEnv


def env_config(path, protocol):
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    world = data["world"]
    env_name = world.pop("env_name")
    time_limit = world.pop("max_episode_steps")
    image_shape = tuple(world.pop("image_shape"))
    world.pop("num_envs")
    world.update(
        render_mode="rgb_array",
        min_distractors=protocol["active_distractors"],
        max_distractors=protocol["active_distractors"],
        terminate_on_goal=protocol["terminate_on_goal_collection_only"],
        retarget_on_goal=protocol["retarget_on_goal"],
    )
    if env_name != "swm/FourRoomDistractors-v0" or image_shape != (224, 224):
        raise ValueError("Expected the official FourRoom Distractors environment at 224×224.")
    return env_name, time_limit, image_shape, world, data.get("options") or {}


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic, torch.Tensor)):
        return value.tolist()
    return value


def rendering_state(env):
    return json_value({
        "variation_space": env.variation_space.value,
        "agent_xy": env.agent_position,
        "goal_xy": env.target_position,
        "closed_door": env.closed_door,
        "closed_door_idx": env._closed_door_idx(),
        "door_positions": env.door_positions,
        "door_sizes": env.door_sizes,
        "wall_thickness": env.wall_thickness,
        "render_target_flag": env.render_target_flag,
        "active_distractors": env._n_active,
        "distractor_colors": env.distractor_colors,
    })


def sample_root(sim, root_seed, conditions, base_options):
    rng = np.random.default_rng(root_seed)
    env = sim.unwrapped
    crossings = conditions["room_crossings"]
    for attempt in range(1, conditions["attempts_cap_per_root"] + 1):
        reset_seed = int(rng.integers(0, 1_000_000_000))
        sim.reset(seed=reset_seed, options=base_options)
        closed_door = env.closed_door
        try:
            start, goal, start_room, goal_room = _sample_crossing_task(env, rng, crossings)
        except NoRoomPairError:
            continue
        if np.linalg.norm(start - goal) < conditions["min_start_goal_euclidean"]:
            continue
        options = {**base_options, "state": start.tolist(), "target_state": goal.tolist()}
        _, info = sim.reset(seed=reset_seed, options=options)
        if not (
            env.closed_door == closed_door
            and np.array_equal(info["xy"], start)
            and np.array_equal(info["goal_state"], goal)
            and env._room_of(start) == start_room
            and env._room_of(goal) == goal_room
            and _room_crossing_count(env, start_room, goal_room) == crossings
            and info["distance_to_target"] >= conditions["min_start_goal_euclidean"]
        ):
            raise RuntimeError(f"Root {root_seed} changed the accepted geometry on reset.")
        _, doors = env.shortest_path(start, goal)
        return {
            "reset_seed": reset_seed,
            "reset_options": options,
            "attempt": attempt,
            "start_xy": start.tolist(),
            "goal_xy": goal.tolist(),
            "start_room": start_room,
            "goal_room": goal_room,
            "closed_door": closed_door,
            "closed_door_idx": info["closed_door_idx"],
            "start_goal_euclidean": info["distance_to_target"],
            "door_graph_crossings": crossings,
            "door_center_path": doors,
            "acceptance": "official geometry sampler, two room crossings and minimum start-goal distance",
            "rendering": rendering_state(env),
        }
    raise RuntimeError(f"Root {root_seed} exhausted {conditions['attempts_cap_per_root']} geometry attempts.")


def render_goals(sim, renderer, root, goal_seeds):
    env = sim.unwrapped
    view = renderer.unwrapped
    physical_before = json_value(env.np_random.bit_generator.state)
    renderer.reset(seed=root["reset_seed"], options=root["reset_options"])
    geometry = rendering_state(env)
    if rendering_state(view) != geometry or not torch.equal(view._target_img, env._target_img):
        raise RuntimeError("Independent render reset does not reproduce the standard goal image.")
    no_distractor = FourRoomEnv._render_frame(view, agent_pos=view.target_position)
    images = [view._target_img.numpy().transpose(1, 2, 0)]
    positions = [view.distractor_positions[0].tolist()]
    for seed in goal_seeds:
        view.np_random = np.random.default_rng(seed)
        position = view._sample_free_position()
        view._set_distractor_positions(position[None])
        if rendering_state(view) != geometry or not torch.equal(
            FourRoomEnv._render_frame(view, agent_pos=view.target_position), no_distractor
        ):
            raise RuntimeError("Goal variant changed non-distractor geometry or rendering.")
        images.append(view._render_frame(agent_pos=view.target_position).numpy().transpose(1, 2, 0))
        positions.append(position.tolist())
    if rendering_state(env) != geometry or json_value(env.np_random.bit_generator.state) != physical_before:
        raise RuntimeError("Independent goal rendering changed the live physical environment.")
    return np.stack(images), positions


def physical_row(env, info, collision):
    agent = info["xy"]
    goal = info["goal_state"]
    distractor = info["distractor0_xy"]
    dd = env.shortest_path(agent, goal)[0]
    minimum = env.shortest_path(agent, distractor)[0]
    if not np.isfinite(dd):
        raise RuntimeError(f"Official door-center distance is nonfinite for real XY {agent} → {goal}.")
    if not np.isfinite(minimum):
        # The official FourRoom distractor-distance backfill uses this rule.
        minimum = np.linalg.norm(agent - distractor)
    de = info["distance_to_target"]
    return {
        "agent_xy": agent,
        "distractor_xy": distractor,
        "closed_door_idx": info["closed_door_idx"],
        "room": ROOM_INDEX.get(env._room_of(agent), -1),
        "collision": collision,
        "hit": de < 16.0,
        "de": de,
        "dd": dd,
        "min_distractor_dist": minimum,
    }


def advance(sim, action, root):
    env = sim.unwrapped
    speed = float(env.variation_space["agent"]["speed"].value.item())
    desired = env.agent_position + torch.as_tensor(action).clamp(-1, 1) * speed
    _, reward, terminated, truncated, info = sim.step(action)
    if terminated or truncated or reward != 0.0:
        raise RuntimeError(f"Unexpected boundary or reward during root {root['root_id']}.")
    if info["closed_door_idx"] != root["closed_door_idx"] or not np.array_equal(
        info["goal_state"], root["goal_xy"]
    ):
        raise RuntimeError(f"Door or goal changed during root {root['root_id']}.")
    return physical_row(env, info, not torch.equal(desired, env.agent_position))


def collect_prefix(sim, actions, root):
    _, info = sim.reset(seed=root["reset_seed"], options=root["reset_options"])
    if rendering_state(sim.unwrapped) != root["rendering"]:
        raise RuntimeError(f"Reset geometry or colors changed for root {root['root_id']}.")
    rows = [physical_row(sim.unwrapped, info, False)]
    images = [sim.render()]
    for action in actions:
        rows.append(advance(sim, action, root))
        images.append(sim.render())
    return rows, np.stack(images)


def store_row(arrays, index, candidate, step, row):
    for key, value in row.items():
        arrays[key][index, candidate, step] = value


def candidate_coverage(arrays, index, horizons, prefix_steps, block_size):
    result = {}
    for horizon in horizons:
        end = prefix_steps + int(horizon) * block_size
        xy = arrays["agent_xy"][index, :, end]
        first = arrays["first_hit_step"][index]
        result[str(horizon)] = {
            "absolute_step": end,
            "xy_span": np.ptp(xy, axis=0).tolist(),
            "unique_endpoints": len(np.unique(xy, axis=0)),
            "candidate_set_has_hit": bool(np.any((first >= 0) & (first <= end))),
            "de_span": float(np.ptp(arrays["de"][index, :, end])),
            "dd_span": float(np.ptp(arrays["dd"][index, :, end])),
        }
    return result


def collect(args):
    torch.set_num_threads(1)
    protocol = json.loads(args.config.read_text(encoding="utf-8"))
    split = protocol["split"][args.split]
    roots = split["roots"]
    block_size = protocol["raw_steps_per_model_step"]
    prefix_steps = protocol["real_prefix_raw_steps"]
    max_h = max(protocol["horizons"])
    horizons = [protocol["smoke_horizon"], *protocol["horizons"]]
    candidates = protocol["candidates_per_root"]
    goal_variants = protocol["goal_variants"]
    end = prefix_steps + max_h * block_size
    prefix_times = np.arange(0, prefix_steps + 1, block_size)
    endpoint_times = [prefix_steps + h * block_size for h in horizons]
    true_times = sorted(set(endpoint_times + [t - block_size for t in endpoint_times[1:]]))
    true_slots = {time: i for i, time in enumerate(true_times)}
    env_name, limit, image_shape, world, base_options = env_config(args.world_config, protocol)
    rgb_shape = (*image_shape, 3)
    if limit <= end:
        raise ValueError("The official time limit must exceed the requested longest branch.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "physical.npz").exists() or (output / "roots.jsonl").exists():
        raise FileExistsError(f"Preserve the existing collection in {output} and choose an empty output.")
    sim = gym.make(env_name, max_episode_steps=limit, **world)
    renderer = gym.make(env_name, max_episode_steps=limit, **world)
    raw_shape = (roots, candidates, end + 1)
    arrays = {
        "split": np.array(args.split),
        "root_id": np.arange(roots, dtype=np.int64) + split["root_seed_base"],
        "candidate_id": np.arange(candidates),
        "H": np.array(horizons),
        "absolute_step": np.arange(end + 1),
        "goal_variant": np.arange(goal_variants),
        "prefix_absolute_step": prefix_times,
        "true_rgb_absolute_step": np.array(true_times),
        "prefix_rgb": np.empty((roots, len(prefix_times), *rgb_shape), np.uint8),
        "goal_rgb": np.empty((roots, goal_variants, *rgb_shape), np.uint8),
        "prefix_raw_actions": np.empty((roots, len(prefix_times) - 1, block_size, 2), np.float32),
        "raw_actions": np.empty((roots, candidates, max_h, block_size, 2), np.float32),
        "agent_xy": np.empty((*raw_shape, 2), np.float32),
        "distractor_xy": np.empty((*raw_shape, 2), np.float32),
        "goal_xy": np.empty((roots, 2), np.float32),
        "closed_door_idx": np.empty(raw_shape, np.int8),
        "room": np.empty(raw_shape, np.int8),
        "collision": np.empty(raw_shape, bool),
        "hit": np.empty(raw_shape, bool),
        "de": np.empty(raw_shape, np.float32),
        "dd": np.empty(raw_shape, np.float64),
        "min_distractor_dist": np.empty(raw_shape, np.float32),
        "first_hit_step": np.full((roots, candidates), -1, np.int16),
    }
    # Keep the larger assessment RGB buffer off the Python heap. Only selected
    # real frames are ever materialized and the retained artifact is compressed.
    with tempfile.NamedTemporaryFile(prefix="p1a-rgb-", suffix=".raw", dir=output, delete=False) as handle:
        rgb_path = Path(handle.name)
    true_rgb = np.memmap(rgb_path, dtype=np.uint8, mode="w+", shape=(roots, candidates, len(true_times), *rgb_shape))
    notes = []
    try:
        for i, root_seed in enumerate(arrays["root_id"]):
            root_seed = int(root_seed)
            root = sample_root(sim, root_seed, protocol["root_sampling"], base_options)
            root.update(root_id=root_seed, root_index=i, split=args.split, world_config=world, max_episode_steps=limit)
            arrays["goal_xy"][i] = root["goal_xy"]
            offset = split["seed_offset"]
            goal_seeds = [protocol["goal_variant_seed_base"] + offset + 100 * i + k for k in range(1, goal_variants)]
            goal_rgb, goal_positions = render_goals(sim, renderer, root, goal_seeds)
            arrays["goal_rgb"][i] = goal_rgb
            root.update(
                goal_variant_seeds=[None, *goal_seeds],
                goal_variant_distractor_xy=goal_positions,
                goal_render_checks={"g0_matches_official_reset": True, "same_non_distractor_render": True, "live_rng_unchanged": True},
            )
            theta = 2 * np.pi * (i % 8) / 8
            prefix = np.repeat(np.array([[np.cos(theta), np.sin(theta)]], np.float32), prefix_steps, axis=0)
            arrays["prefix_raw_actions"][i] = prefix.reshape(-1, block_size, 2)
            observed, observed_rgb = collect_prefix(sim, prefix, root)
            arrays["prefix_rgb"][i] = observed_rgb[prefix_times]
            # The bank only becomes a query after the shared real event.
            seed = protocol["candidate_seed_base"] + offset + i
            bank, names = build_action_bank(seed, max_h, block_size)
            arrays["raw_actions"][i] = bank
            arrays["candidate_name"] = np.array(names)
            root["candidate_seed"] = seed
            for candidate, actions in enumerate(bank):
                replay, replay_rgb = collect_prefix(sim, prefix, root)
                if not np.array_equal(replay_rgb, observed_rgb) or any(
                    not np.array_equal(actual[key], expected[key])
                    for actual, expected in zip(replay, observed, strict=True)
                    for key in expected
                ):
                    raise RuntimeError(f"Prefix replay differs for root {root_seed}, candidate {candidate}.")
                for step, row in enumerate(replay):
                    store_row(arrays, i, candidate, step, row)
                for step, action in enumerate(actions.reshape(-1, 2), start=prefix_steps + 1):
                    row = advance(sim, action, root)
                    store_row(arrays, i, candidate, step, row)
                    if row["hit"] and arrays["first_hit_step"][i, candidate] < 0:
                        arrays["first_hit_step"][i, candidate] = step
                    if step in true_slots:
                        true_rgb[i, candidate, true_slots[step]] = sim.render()
                if candidate and not np.array_equal(arrays["distractor_xy"][i, candidate], arrays["distractor_xy"][i, 0]):
                    raise RuntimeError(f"Distractor trajectory diverged for root {root_seed}, candidate {candidate}.")
            root["replay_checks"] = {
                "candidate_replays": candidates,
                "exact_prefix_rgb_frames_per_candidate": prefix_steps + 1,
                "exact_prefix_physical_frames_per_candidate": prefix_steps + 1,
                "exact_distractor_frames_per_candidate": end + 1,
            }
            root["coverage"] = candidate_coverage(arrays, i, horizons, prefix_steps, block_size)
            notes.append(root)
            print(f"{args.split}: {i + 1}/{roots} roots, seed {root_seed}, {candidates} branches.", flush=True)
        true_rgb.flush()
        with (output / "physical.npz").open("xb") as file:
            np.savez_compressed(file, true_rgb=true_rgb, **arrays)
        with (output / "roots.jsonl").open("x", encoding="utf-8") as file:
            for note in notes:
                file.write(json.dumps(note, ensure_ascii=False, allow_nan=False) + "\n")
    finally:
        sim.close()
        renderer.close()
        del true_rgb
        rgb_path.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("smoke", "development", "assessment"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("p1a_config.json"))
    parser.add_argument("--world-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    collect(parser.parse_args())
