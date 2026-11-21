"""Frozen Phase 2 data collection on the existing official TwoRoom path.

Learning archives contain RGB/controls and execution validity only. Speed,
change points, coordinates, and counterfactual outcomes stay in diagnostic files.
"""

from __future__ import annotations

import argparse
import json
from .prbd_paths import PHASE2 as OUT, output_path
import time

import numpy as np
import torch

from .prbd_consistency_data import program as static_program
from .prbd_tworoom_speed_data import rollout, stack_records, environment_class


SPLITS = {"train": (2026113000, 192), "static": (2026114000, 48),
          "switch": (2026114100, 48)}
SPEEDS = np.array([2.5, 7.5], dtype=np.float32)


def save_json(name, value):
    output_path(OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = {
        "scope": "Phase2 fixed data, ordinary actual single-trajectory training; no model-score selection",
        "native_path": "prbd_tworoom_speed_data.rollout -> unmodified official TwoRoomEnv reset/step/render",
        "splits": {key: {"first_seed": start, "roots": count} for key, (start, count) in SPLITS.items()},
        "train": {"blocks": 32, "starting_speeds": "balanced96/96, independent permutation RNG2026113101",
                  "switches": "RNG2026113100; cut1 integer8..13,cut2 integer20..25; speed changes before that action block",
                  "commands": "fixed seed2026113200+index, eight-block directional/return/zero/free motif; no state feedback",
                  "model_keys": ["z", "a", "lengths", "seed"]},
        "static": {"axes": ["root", "speed slow2.5/fast7.5", "candidate orthogonal/stop"],
                   "program": "existing consistency_data.program(dev,index,candidate)",
                   "source_root": 2, "matched_query_root": 6, "horizons": [1, 4]},
        "switch": {"axes": ["root", "initial-speed slow2.5/fast7.5"], "blocks": 32,
                   "switches": [10, 20], "commands": "same32-block program in both arms, fixed balanced motion/return/zero",
                   "post_switch_observation": "12 blocks after last change; all fixed cases retained; validity separately reported",
                   "counterfactual": "each actual block root independently re-rendered/replayed at slow and fast; diagnostic only"},
        "goal": "official reset publicly supplies goal_state; native reset-computed _target_img renders that public target in the same layout",
        "encoding": "CUDA float32, TF32 disabled, fixed image batch32 padded; canonical native CPU float32 action shape[1,1,10]",
        "termination": "stop at first native terminated/truncated, retain every fixed root, lengths counts complete five-primitive blocks",
        "diagnostic_boundary": "diag_*.npz and switch_counterfactual.pt forbidden training/method inputs",
        "no_hash_checks": True,
    }
    path = OUT / "DATA_PLAN.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Frozen data plan already exists with a different specification")
    save_json("DATA_PLAN.json", plan)


def train_program(index):
    rng = np.random.default_rng(2026113200 + index)
    cardinal = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
    along = cardinal[index % 4]
    across = np.array([-along[1], along[0]], dtype=np.float32)
    if index // 4 % 2:
        across = -across
    amplitude = [.125, .1875, .25, .3125][index // 8 % 4]
    waveform = np.asarray([.5, 1, 1.5, 1, .5], np.float32)[:, None]
    p, q = waveform * amplitude * along, waveform * amplitude * across
    zero = np.zeros((5, 2), dtype=np.float32)
    blocks = []
    for motif in range(4):
        # A free control and its exact reverse broaden controls without feedback.
        free = rng.uniform(-amplitude, amplitude, (5, 2)).astype(np.float32)
        first, second = (p, q) if motif % 2 == 0 else (q, -p)
        blocks.extend([zero, first, second, -second[::-1], -first[::-1], free, -free[::-1], zero])
    return np.stack(blocks)


def switch_program(index):
    # Small bounded excursions support repeated observations without requiring a
    # hidden-state controller. Both opposite-speed histories use these controls.
    cardinal = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
    along = cardinal[index % 4]
    across = np.array([-along[1], along[0]], dtype=np.float32)
    if index // 4 % 2:
        across = -across
    gain = [.125, .1875, .25][index // 8 % 3]
    wave = np.asarray([.5, 1, 1.5, 1, .5], np.float32)[:, None] * gain
    p, q, zero = wave * along, wave * across, np.zeros((5, 2), np.float32)
    pair = [p, -p[::-1], q, -q[::-1], zero, p, -p[::-1], zero,
            q, -q[::-1], p, -p[::-1], zero, q, -q[::-1], zero]
    return np.stack(pair + pair)


def public_goals(seeds):
    goals = []
    for seed in seeds:
        env = environment_class()(render_mode="rgb_array", render_target=False)
        _, info = env.reset(seed=int(seed))
        # This image is constructed by the official reset before any rollout.
        goal = env._target_img.cpu().numpy().transpose(1, 2, 0).copy()
        expected = env._render_frame(torch.as_tensor(info["goal_state"], dtype=torch.float32))
        np.testing.assert_array_equal(goal, expected.numpy().transpose(1, 2, 0))
        goals.append(goal)
        env.close()
    return np.stack(goals)


def counterfactual(seed, legal, truth):
    blocks = len(legal["actions"])
    rgb = np.zeros((blocks, 2, 224, 224, 3), dtype=np.uint8)
    complete = np.zeros((blocks, 2), dtype=bool)
    counts = np.zeros((blocks, 2), dtype=np.int64)
    positions = np.full((blocks, 2, 2), np.nan, dtype=np.float32)
    roots = np.zeros(blocks, dtype=bool)
    for block in range(blocks):
        if not legal["frame_valid"][block] or not legal["primitive_valid"][block].any():
            break
        roots[block] = True
        for speed_index, speed in enumerate(SPEEDS):
            env = environment_class()(render_mode="rgb_array", render_target=False)
            env.reset(seed=int(seed), options={"variation_values": {
                "agent.speed": np.array([speed], dtype=np.float32)}})
            env._set_state(truth["positions"][block * 5])
            np.testing.assert_array_equal(env.render(), legal["rgb"][block])
            for action in legal["actions"][block]:
                _, _, terminated, truncated, info = env.step(action)
                counts[block, speed_index] += 1
                if terminated or truncated:
                    break
            rgb[block, speed_index] = env.render()
            positions[block, speed_index] = info["state"].copy()
            complete[block, speed_index] = counts[block, speed_index] == 5
            actual_speed = float(truth["speed"][block])
            if float(speed) == actual_speed:
                actual_count = int(legal["primitive_valid"][block].sum())
                if int(counts[block, speed_index]) != actual_count:
                    raise ValueError("Actual-speed counterfactual primitive count differs")
                target = legal["rgb"][block + 1] if legal["block_valid"][block] else legal["final_rgb"]
                np.testing.assert_array_equal(rgb[block, speed_index], target)
            env.close()
    differing = np.any(rgb[:, 0] != rgb[:, 1], axis=(1, 2, 3)) & complete.all(axis=-1)
    return {"rgb": rgb, "root_valid": roots, "complete": complete,
            "primitive_count": counts, "positions": positions, "differing_rgb": differing}


def collect(split):
    freeze()
    torch.set_num_threads(1)
    start_time = time.perf_counter()
    start, count = SPLITS[split]
    learning, physics, cf = [], [], []
    cuts_rng = np.random.default_rng(2026113100)
    cuts = np.stack((cuts_rng.integers(8, 14, count), cuts_rng.integers(20, 26, count)), axis=-1)
    starts = np.random.default_rng(2026113101).permutation(np.tile(SPEEDS, count // 2))
    for index in range(count):
        arms = [0] if split == "train" else [0, 1]
        for arm in arms:
            candidates = [0, 1] if split == "static" else [0]
            for candidate in candidates:
                if split == "static":
                    commands = static_program("dev", index, candidate)
                    speed = float(SPEEDS[arm])
                    change = np.array([-1, -1], dtype=np.int64)
                else:
                    commands = train_program(index) if split == "train" else switch_program(index)
                    change = cuts[index] if split == "train" else np.array([10, 20])
                    initial = starts[index] if split == "train" else SPEEDS[arm]
                    speed = np.full(len(commands), initial, dtype=np.float32)
                    speed[change[0]:change[1]] = 10.0 - initial
                legal, truth = rollout(start + index, speed, commands)
                truth["switch_blocks"] = change
                learning.append(legal)
                physics.append(truth)
                if split == "switch":
                    cf.append(counterfactual(start + index, legal, truth))
        if (index + 1) % 16 == 0:
            print(f"collect {split} {index+1}/{count} {time.perf_counter()-start_time:.1f}s", flush=True)
    legal, truth = stack_records(learning), stack_records(physics)
    prefix_shape = (count,) if split == "train" else ((count, 2, 2) if split == "static" else (count, 2))
    if split != "train":
        legal = {key: value.reshape(*prefix_shape, *value.shape[1:]) for key, value in legal.items()}
        truth = {key: value.reshape(*prefix_shape, *value.shape[1:]) for key, value in truth.items()}
    legal["seed"] = np.arange(start, start + count, dtype=np.int64)
    if split == "static":
        legal["matched_rgb"] = np.array([
            legal["frame_valid"][i, ..., :7].all()
            and np.all(legal["rgb"][i, :, :, 4:7] == legal["rgb"][i, 0, 0, 4:7])
            and np.all(legal["actions"][i, :, :, 4:6] == legal["actions"][i, 0, 0, 4:6])
            for i in range(count)])
    np.savez_compressed(output_path(OUT / f"data_{split}.npz"), **legal)
    np.savez_compressed(output_path(OUT / f"diag_{split}.npz"), **truth)
    if split != "train":
        np.savez_compressed(output_path(OUT / f"goals_{split}.npz"), seed=legal["seed"], rgb=public_goals(legal["seed"]))
    if split == "switch":
        cf = {key: value.reshape(count, 2, *value.shape[1:]) for key, value in stack_records(cf).items()}
        np.savez_compressed(output_path(OUT / "data_switch_counterfactual.npz"), rgb=cf.pop("rgb"), root_valid=cf["root_valid"])
        np.savez_compressed(output_path(OUT / "diag_switch_counterfactual.npz"), **cf)
    lengths = legal["block_valid"].sum(axis=-1)
    meta = {"seed_range": [start, start + count - 1], "roots": count,
            "rollouts": len(learning), "lengths": lengths.tolist(),
            "complete_rollouts": int((lengths == legal["block_valid"].shape[-1]).sum()),
            "actual_primitives": int(truth["primitive_count"].sum()),
            "collisions": int(truth["collision"].sum()),
            "terminated": int(truth["terminated"].any(axis=(-1, -2)).sum()),
            "matched_rgb_roots": int(legal["matched_rgb"].sum()) if split == "static" else None,
            "wall_seconds": time.perf_counter() - start_time}
    if split == "switch":
        meta["cf_differing_rgb_counts"] = cf["differing_rgb"].sum(axis=-1).tolist()
        meta["cf_complete_slow_fast_counts"] = cf["complete"].all(axis=-1).sum(axis=-1).tolist()
        meta["after_last_switch_differing_counts"] = cf["differing_rgb"][..., 20:].sum(axis=-1).tolist()
        meta["at_least_eight_after_last_switch_differing"] = int((cf["differing_rgb"][..., 20:].sum(axis=-1) >= 8).sum())
    save_json(f"collection_{split}.json", meta)
    print(json.dumps({"split": split, **meta}), flush=True)


@torch.no_grad()
def encode_images(host, rgb, device):
    mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
    std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
    output = torch.zeros((len(rgb), 192), dtype=torch.float32)
    for start in range(0, len(rgb), 32):
        part = rgb[start:start+32]
        padded = np.concatenate((part, np.repeat(part[-1:], 32-len(part), axis=0)))
        images = torch.from_numpy(padded).to(device).permute(0, 3, 1, 2).float() / 255
        output[start:start+len(part)] = host.encode({"pixels": ((images-mean)/std)[:, None]})["emb"][:len(part), 0].cpu()
    return output


def encode(split):
    from .prbd_tworoom_speed_fit import load_host, encode_split, RELEASE
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen encoding protocol requires CUDA FP32")
    device = torch.device("cuda")
    host, action_host = load_host(device), load_host("cpu")
    stats = json.loads((RELEASE / "action_statistics.json").read_text(encoding="utf-8"))
    batch = encode_split(host, split, device, stats, directory=OUT, canonical_action_host=action_host)
    if split == "train" and set(batch) != {"z", "a", "lengths", "seed"}:
        raise ValueError("Training package contains a forbidden field")
    temporary = output_path(OUT / f"{split}.writing.pt")
    torch.save(batch, temporary)
    temporary.replace(output_path(OUT / f"{split}.pt"))
    checks = {"z_shape": list(batch["z"].shape), "a_shape": list(batch["a"].shape),
              "finite": bool(torch.isfinite(batch["z"]).all() and torch.isfinite(batch["a"]).all()),
              "keys": sorted(batch), "encoder_device": str(device), "dtype": str(batch["z"].dtype)}
    with np.load(OUT / f"data_{split}.npz") as source:
        # Direct equality of every actual repeated adjacent RGB is checked,
        # including the boundary between separately evaluated image batches.
        same = np.all(source["rgb"][..., 1:, :, :, :] == source["rgb"][..., :-1, :, :, :], axis=(-1,-2,-3))
        valid = source["block_valid"]
        selected = torch.from_numpy(same & valid)
        checks["equal_adjacent_rgb_pairs"] = int(selected.sum())
        checks["equal_adjacent_rgb_latents_bitwise"] = bool(torch.equal(batch["z"][..., 1:, :][selected], batch["z"][..., :-1, :][selected]))
        if not checks["equal_adjacent_rgb_latents_bitwise"]:
            raise ValueError("Identical RGB did not produce bitwise-identical CUDA FP32 latents")
    if split != "train":
        with np.load(OUT / f"goals_{split}.npz") as goals:
            goal_z = encode_images(host, goals["rgb"], device)
            torch.save({"seed": torch.from_numpy(goals["seed"]), "z_goal": goal_z}, output_path(OUT / f"goals_{split}.pt"))
    if split == "static":
        chosen = batch["matched_rgb"]
        checks["matched_root_latents_bitwise"] = bool(torch.equal(batch["z"][chosen, ..., 4:7, :],
            batch["z"][chosen, 0:1, 0:1, 4:7, :].expand(-1, 2, 2, -1, -1)))
        if not checks["matched_root_latents_bitwise"]:
            raise ValueError("Matched roots lost latent equality")
    if split == "switch":
        with np.load(OUT / "data_switch_counterfactual.npz") as cf:
            rgb, roots = cf["rgb"], cf["root_valid"]
            z_cf = torch.zeros((*roots.shape, 2, 192), dtype=torch.float32)
            z_cf[torch.from_numpy(roots)] = encode_images(host, rgb[roots].reshape(-1,224,224,3), device).reshape(-1,2,192)
        with np.load(OUT / "diag_switch.npz") as truth:
            actual_indices = torch.from_numpy((truth["speed"] == 7.5).astype(np.int64))
        actual_cf = torch.gather(z_cf, -2, actual_indices[...,None,None].expand(*actual_indices.shape,1,192)).squeeze(-2)
        mask = torch.arange(batch["a"].shape[-2])[None,None] < batch["lengths"][...,None]
        checks["actual_cf_latents_bitwise"] = bool(torch.equal(actual_cf[mask], batch["z"][...,1:,:][mask]))
        if not checks["actual_cf_latents_bitwise"]:
            raise ValueError("Actual-speed counterfactual latent differs from actual successor")
        torch.save({"seed": batch["seed"], "z_cf": z_cf, "root_valid": torch.from_numpy(roots)}, output_path(OUT / "switch_counterfactual.pt"))
    save_json(f"encoding_{split}.json", checks)
    print(json.dumps({"split":split, **checks}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "collect", "encode"])
    parser.add_argument("split", nargs="?", choices=SPLITS)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.split is None:
        parser.error("collect/encode require a split")
    else:
        (collect if args.stage == "collect" else encode)(args.split)
