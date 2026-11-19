"""Actual unmodified official TwoRoom rollouts for the fixed PRBD speed probe.

Only RGB and ordinary raw controls appear in learning NPZ files. Physical state,
speed, collision diagnostics, and released episode identities live separately.
No predictor or planner is imported or called.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/prbd_tworoom_speed_20261119"

def ordinary_commands(official_hdf5):
    """First 64 sufficient ordinary episodes in chronology, each used once."""
    import h5py

    # Read original HDF5 datasets directly instead of the execution machine's
    # acquired metadata prefix and extracted full action column.
    with h5py.File(official_hdf5, "r") as file:
        if file["action"].shape != (920809, 2):
            raise ValueError("Unexpected complete released action array")
        lengths, offsets = file["ep_len"][:1024], file["ep_offset"][:1024]
        full_actions = file["action"][:]
    np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum(lengths[:-1])])
    selected = []
    for episode, (length, offset) in enumerate(zip(lengths, offsets)):
        if length < 81:
            continue
        raw = np.array(full_actions[int(offset):int(offset)+80], dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError("Nonterminal published action contains NaN")
        selected.append((episode, int(offset), int(length), raw))
        if len(selected) == 64:
            break
    if len(selected) != 64:
        raise ValueError("Acquired metadata does not contain 64 sufficient episodes")
    return selected


def rollout(seed, speed, commands):
    """Execute every stored transition through actual env.step, stopping at done."""
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    blocks = len(commands)
    env = TwoRoomEnv(render_mode="rgb_array", render_target=False)
    obs, info = env.reset(seed=int(seed), options={"variation_values": {
        "agent.speed": np.array([speed], dtype=np.float32)}})
    if float(env.variation_space["agent"]["speed"].value.item()) != speed:
        raise ValueError("Native public speed variation was not applied")
    rgb = np.zeros((blocks+1, 224, 224, 3), dtype=np.uint8)
    rgb[0] = env.render()
    frame_valid = np.zeros(blocks+1, dtype=bool)
    frame_valid[0] = True
    block_valid = np.zeros(blocks, dtype=bool)
    primitive_valid = np.zeros((blocks, 5), dtype=bool)
    positions = np.full((blocks*5+1, 2), np.nan, dtype=np.float32)
    positions[0] = info["state"].copy()
    collision = np.zeros((blocks, 5), dtype=bool)
    terminated = np.zeros((blocks, 5), dtype=bool)
    truncated = np.zeros((blocks, 5), dtype=bool)
    goal = info["goal_state"].copy()
    count = 0
    done = False
    for block, actions in enumerate(commands):
        for primitive, action in enumerate(actions):
            before = positions[count].copy()
            obs, reward, term, trunc, info = env.step(action)
            count += 1
            actual = np.asarray(info["state"], dtype=np.float32).copy()
            positions[count] = actual
            expected = before + np.clip(action, -1, 1) * np.float32(speed)
            collision[block, primitive] = np.max(np.abs(actual - expected)) > 1e-5
            primitive_valid[block, primitive] = True
            terminated[block, primitive] = term
            truncated[block, primitive] = trunc
            if term or trunc:
                done = True
                break
        if primitive_valid[block].all():
            rgb[block+1] = env.render()
            frame_valid[block+1] = True
            block_valid[block] = True
        if done:
            break
    final_rgb = env.render()
    env.close()
    return {"rgb": rgb, "frame_valid": frame_valid, "block_valid": block_valid,
            "primitive_valid": primitive_valid, "actions": commands,
            "final_rgb": final_rgb}, {
        "positions": positions, "goal": goal, "collision": collision,
        "terminated": terminated, "truncated": truncated,
        "primitive_count": np.array(count, dtype=np.int64),
        "seed": np.array(seed, dtype=np.int64), "speed": np.array(speed, dtype=np.float32)}


def stack_records(records):
    return {name: np.stack([record[name] for record in records]) for name in records[0]}


def summarize(physical):
    return {"sequences": len(physical["speed"]),
            "executed_primitives": int(physical["primitive_count"].sum()),
            "primitive_lengths": physical["primitive_count"].tolist(),
            "collision_primitives": int(physical["collision"].sum()),
            "terminated_sequences": int(physical["terminated"].any(axis=(-1, -2)).sum()),
            "truncated_sequences": int(physical["truncated"].any(axis=(-1, -2)).sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--official-hdf5", type=Path, required=True,
                        help="Original tworoom.h5 extracted from the official tworoom.tar.zst")
    parser.add_argument("--reference-env-source", type=Path, required=True,
                        help="Unmodified inspected official stable_worldmodel/envs/two_room/env.py")
    args = parser.parse_args()
    import gymnasium
    from stable_worldmodel.envs.two_room import env as native_env

    started = time.perf_counter()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    # Verify cached source agrees with previously inspected official source by
    # direct text equality. No content hashes or provenance substitution.
    env_source = Path(native_env.__file__)
    if env_source.read_text(encoding="utf-8") != args.reference_env_source.read_text(encoding="utf-8"):
        raise ValueError("Cached native TwoRoom differs from the inspected official environment")
    episodes = ordinary_commands(args.official_hdf5)
    diagnostics = {}
    summary = {}
    fit_speeds = np.random.default_rng(2026111980).permutation(np.tile([2.5, 7.5], 16))
    for split, start, speeds in (("calibration", 0, np.full(32, 5.0)), ("fit", 32, fit_speeds)):
        learning, physical = [], []
        for index in range(32):
            source = episodes[start+index]
            seed = 2026111900 + start + index
            legal, truth = rollout(seed, float(speeds[index]), source[3].reshape(16, 5, 2))
            learning.append(legal)
            truth.update({"source_episode": np.array(source[0]), "source_offset": np.array(source[1]),
                          "source_episode_length": np.array(source[2])})
            physical.append(truth)
        legal, truth = stack_records(learning), stack_records(physical)
        legal["seed"] = truth["seed"].copy()
        np.savez_compressed(args.output / f"data_{split}.npz", **legal)
        diagnostics.update({f"{split}_{name}": value for name, value in truth.items()})
        summary[split] = summarize(truth)
        summary[split]["source_episodes"] = truth["source_episode"].tolist()
        summary[split]["raw_action_axis_min"] = legal["actions"].min(axis=(0, 1, 2)).tolist()
        summary[split]["raw_action_axis_max"] = legal["actions"].max(axis=(0, 1, 2)).tolist()
        print(json.dumps({"split": split, **summary[split]}), flush=True)

    cardinal = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
    validation_legal, validation_truth = [], []
    for index in range(16):
        seed = 2026111964 + index
        probe = cardinal[index % 4] * np.float32(.25)
        transverse = np.array([-probe[1], probe[0]], dtype=np.float32)
        for speed in (2.5, 7.5):
            for candidate in range(2):
                commands = np.zeros((10, 5, 2), dtype=np.float32)
                commands[2] = probe
                commands[3] = -probe
                if candidate == 0:
                    commands[6] = transverse
                legal, truth = rollout(seed, speed, commands)
                validation_legal.append(legal)
                truth.update({"probe_direction": np.array(index % 4), "candidate": np.array(candidate)})
                validation_truth.append(truth)
    legal, truth = stack_records(validation_legal), stack_records(validation_truth)
    legal = {name: value.reshape(16, 2, 2, *value.shape[1:]) for name, value in legal.items()}
    truth = {name: value.reshape(16, 2, 2, *value.shape[1:]) for name, value in truth.items()}
    # Root t6 has frames4,5,6. Source root t2 has frames0,1,2.
    # Flags describe all preregistered cases, including failures; none is replaced.
    complete_prefix = legal["frame_valid"][..., :7].all(axis=-1).all(axis=-1).all(axis=-1)
    rgb_prefix_replay = (legal["rgb"][:, :, 0, :7] == legal["rgb"][:, :, 1, :7]).all(axis=(1, 2, 3, 4, 5))
    rgb_root_speed = (legal["rgb"][:, 0, 0, 4:7] == legal["rgb"][:, 1, 0, 4:7]).all(axis=(1, 2, 3, 4))
    logical_root_speed = (truth["positions"][:, 0, 0, [20, 25, 30]] == truth["positions"][:, 1, 0, [20, 25, 30]]).all(axis=(1, 2))
    stationary_root = (truth["positions"][..., 20:31, :] == truth["positions"][..., 20:21, :]).all(axis=(1, 2, 3, 4))
    returned_start = (truth["positions"][..., 30, :] == truth["positions"][..., 0, :]).all(axis=(1, 2, 3))
    same_goals = (truth["goal"] == truth["goal"][:, :1, :1]).all(axis=(1, 2, 3))
    matched = complete_prefix & rgb_prefix_replay & rgb_root_speed & logical_root_speed & stationary_root & returned_start & same_goals
    truth.update({"complete_prefix_by_seed": complete_prefix,
                  "rgb_prefix_candidate_replay_equal_by_seed": rgb_prefix_replay,
                  "rgb_root_cross_speed_equal_by_seed": rgb_root_speed,
                  "logical_root_cross_speed_equal_by_seed": logical_root_speed,
                  "stationary_root_by_seed": stationary_root,
                  "returned_to_start_by_seed": returned_start,
                  "same_goals_by_seed": same_goals, "matched_windows_by_seed": matched})
    legal["seed"] = np.arange(2026111964, 2026111980, dtype=np.int64)
    legal["matched_rgb"] = complete_prefix & rgb_prefix_replay & rgb_root_speed
    np.savez_compressed(args.output / "data_validation.npz", **legal)
    diagnostics.update({f"validation_{name}": value for name, value in truth.items()})
    np.savez_compressed(args.output / "data_diagnostics.npz", **diagnostics)
    summary["validation"] = {
        "unique_native_reset_seeds": 16, "independent_actual_rollouts": 64,
        "executed_primitives": int(truth["primitive_count"].sum()),
        "primitive_lengths": truth["primitive_count"].tolist(),
        "collision_primitives": int(truth["collision"].sum()),
        "terminated_rollouts": int(truth["terminated"].any(axis=(-1, -2)).sum()),
        "matched_window_seeds": int(matched.sum()),
        "matched_windows_by_seed": matched.tolist(),
        "rgb_prefix_candidate_replay_equal_by_seed": rgb_prefix_replay.tolist(),
        "rgb_root_cross_speed_equal_by_seed": rgb_root_speed.tolist(),
        "logical_root_cross_speed_equal_by_seed": logical_root_speed.tolist(),
        "returned_to_start_by_seed": returned_start.tolist(),
        "stationary_root_by_seed": stationary_root.tolist()}
    manifest = {
        "scope": "Actual unmodified official swm/TwoRoom-v1 environment reset/step/render; no model, planner, tuning, or closed-loop evaluation",
        "runtime": {"python": sys.executable, "python_version": platform.python_version(),
                    "torch": torch.__version__, "gymnasium": gymnasium.__version__, "torch_threads": 1},
        "official_source": {"package": "stable-worldmodel", "version": "0.1.1", "native_source": str(env_source),
                            "public_repository": "https://github.com/galilai-group/stable-worldmodel",
                            "matches_saved_official_env_text": True, "source_not_modified": True},
        "speed_variation": {"interface": "reset options variation_values agent.speed", "official_default": 5.0,
                            "calibration": "all32 default5.0; nominal-only host residual covariance, fixed before model scores",
                            "fit": "balanced16 at2.5 and16 at7.5, permutation RNG2026111980 independent of reset seeds and command selection",
                            "validation": "each of16 reset seeds independently replayed at both2.5 and7.5"},
        "data_boundary": {"calibration_seeds": [2026111900, 2026111931], "fit_seeds": [2026111932, 2026111963],
                          "validation_seeds": [2026111964, 2026111979],
                          "learning_files": "Only RGB, raw actions, validity masks, and final actual RGB; no speed/state labels",
                          "physical_diagnostics": "data_diagnostics.npz only; forbidden model/fitter input"},
        "ordinary_commands": {"source_hdf5": str(args.official_hdf5),
                              "source_archive_url": "https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst",
                              "source_actions": "action dataset in original tworoom.h5", "rows": 920809,
                              "episode_metadata": "ep_len/ep_offset first1024 native HDF5 entries; direct complete-file access replaces local partial-file extraction only",
                              "selection": "First64 chronological released episodes having at least80 finite primitive actions; first80 controls each, first32 calibration,next32 fit; disjoint rows"},
        "schema": {"training": "rgb[N,17,224,224,3], actions[N,16,5,2], frame_valid[N,17], block_valid[N,16], primitive_valid[N,16,5], final_rgb[N,224,224,3]",
                   "validation": "Same keys prefixed shape[16seeds,2speeds,2candidates,...];11 frames and10 action blocks; speed axis2.5,7.5; candidate axis transverse,zero",
                   "frame_time": "RGB frame t is after5*t primitive steps; action block t drives frame t to t+1",
                   "termination": "Stop on first terminated/truncated; masks preserve only executed steps; incomplete last block retained by primitive_valid and final_rgb"},
        "fixed_validation": {"probe_directions": ["left", "right", "up", "down"],
                             "prefix": "blocks0,1 zero; block2 cardinal5x0.25; block3 inverse5x0.25;blocks4,5 zero; root t6 window[4,5,6]",
                             "suffix": "candidate0 block6 perpendicular5x0.25; candidate1 zero; both blocks7,8,9 zero; H1 frame7, H4 frame10",
                             "all_cases_retained": True, "selection_by_model_performance": False},
        "summary": summary, "wall_seconds": time.perf_counter()-started}
    (args.output / "data_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"validation": summary["validation"], "wall_seconds": manifest["wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
