"""Collect causal TwoRoom trajectories and encode them for the P3b protocol."""

from .prbd_paths import CONSISTENCY, output_path

import argparse
import json
from pathlib import Path

import numpy as np


OUT = CONSISTENCY
SPLITS = {"train": (2026112000, 64), "dev": (2026112100, 48),
          "confirm": (2026112300, 96)}


def program(split, index, candidate=0):
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.float32)
    along = directions[index % 4]
    across = np.array([-along[1], along[0]], dtype=np.float32)
    if ((index // 4) % 2 == 1) != (split == "confirm"):
        across = -across
    amplitude = [.1875, .25, .3125, .375][index // 8 % 4]
    waveform = [.5, 1, .5, 1.5, 1.5] if split == "confirm" else [.5, 1, 1.5, 1, .5]
    gains = np.asarray(waveform, dtype=np.float32)[:, None] * amplitude
    p, q = gains * along, gains * across
    zero = np.zeros_like(p)
    if split == "train":
        blocks = [zero, zero, p, q, zero, -q[::-1], -p[::-1], zero,
                  q, p, zero, -p[::-1], -q[::-1], zero, p, q]
    else:
        blocks = [zero, zero, p, -p[::-1], zero, zero,
                  q if candidate == 0 else zero, zero, zero, zero]
    return np.stack(blocks)


def collect(split):
    from .prbd_tworoom_speed_data import rollout, stack_records
    start, count = SPLITS[split]
    speeds = np.random.default_rng(20261120).permutation(np.tile([2.5, 7.5], count // 2))
    learning, physics = [], []
    for index in range(count):
        for speed in ([speeds[index]] if split == "train" else [2.5, 7.5]):
            for candidate in range(1 if split == "train" else 2):
                legal, truth = rollout(start + index, float(speed), program(split, index, candidate))
                learning.append(legal)
                physics.append(truth)
    legal, truth = stack_records(learning), stack_records(physics)
    if split != "train":
        def by_root(records):
            return {k: v.reshape(count, 2, 2, *v.shape[1:]) for k, v in records.items()}
        legal, truth = by_root(legal), by_root(truth)
        legal["matched_rgb"] = np.array([
            legal["frame_valid"][i, ..., :7].all()
            and np.all(legal["rgb"][i, :, :, 4:7] == legal["rgb"][i, 0, 0, 4:7])
            and np.all(legal["actions"][i, :, :, 4:6] == legal["actions"][i, 0, 0, 4:6])
            for i in range(count)])
    legal["seed"] = np.arange(start, start + count)
    np.savez_compressed(output_path(OUT / f"diag_{split}.npz"), **truth)
    np.savez_compressed(output_path(OUT / f"data_{split}.npz"), **legal)
    meta = {"reset_count": count, "seed_range": [start, start + count - 1],
            "actual_rollouts": len(physics), "actual_primitives": int(truth["primitive_count"].sum()),
            "complete_blocks": int(legal["block_valid"].sum()),
            "complete_zero_blocks": int(((legal["actions"] == 0).all(axis=(-1,-2)) & legal["block_valid"]).sum()),
            "matched_rgb_count": int(legal["matched_rgb"].sum()) if "matched_rgb" in legal else None}
    (output_path(OUT / f"collection_{split}.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(split, meta, flush=True)


def encode(split):
    import torch
    from .prbd_tworoom_speed_fit import load_host, RELEASE, encode_split
    from .prbd_native import encode_native_controls
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    host = load_host(device)
    action_host = load_host("cpu") if device.type != "cpu" else host
    stats = json.loads((RELEASE / "action_statistics.json").read_text(encoding="utf-8"))
    batch = encode_split(host, split, device, stats, directory=OUT, canonical_action_host=action_host)
    batch["null_control"] = encode_native_controls(action_host, np.zeros((5,2), np.float32), stats)
    temp = output_path(OUT / f"{split}.writing.pt")
    torch.save(batch, output_path(temp))
    temp.replace(output_path(OUT / f"{split}.pt"))
    print(split, tuple(batch["z"].shape), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["collect", "encode"])
    parser.add_argument("split", choices=SPLITS)
    args = parser.parse_args()
    (collect if args.stage == "collect" else encode)(args.split)
