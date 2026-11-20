"""Frozen released TwoRoom host, RGB/action encoding and factual episodes.

Set PRBD_LEWM_SOURCE / PRBD_MODEL_DIR for the unmodified author assets.
"""

import json
from pathlib import Path

import numpy as np
import torch

from .prbd_assets import load_model
from .prbd_paths import ROOT, SPEED_DATA as OUT, STATISTICS as RELEASE, LEWM_SOURCE, MODEL_DIR


def load_host(device, *, source=None, model_dir=None):
    return load_model(device, source=LEWM_SOURCE if source is None else source,
                      model_dir=MODEL_DIR if model_dir is None else model_dir).requires_grad_(False).eval()


@torch.no_grad()
def encode_split(host, split, device, stats, *, directory=OUT, canonical_action_host=None):
    with np.load(directory / f"data_{split}.npz", allow_pickle=False) as data:
        rgb, raw = data["rgb"], data["actions"]
        visible, valid = data["frame_valid"], data["block_valid"]
        if rgb.dtype != np.uint8 or rgb.shape != (*visible.shape, 224, 224, 3):
            raise ValueError("Invalid author RGB input")
        lengths = valid.sum(axis=-1)
        if (not np.array_equal(valid, np.arange(valid.shape[-1]) < lengths[..., None])
                or not np.array_equal(visible, np.arange(visible.shape[-1]) <= lengths[..., None])):
            raise ValueError("Incomplete physical blocks cannot enter a model window")
        if raw.shape != (*valid.shape, 5, 2) or not np.isfinite(raw[valid]).all():
            raise ValueError("Missing executed primitive control in a valid model block")
        z = torch.zeros((*visible.shape, 192), dtype=torch.float32)
        rgb_input = rgb[visible]
        observation_mean = torch.tensor([.485, .456, .406], device=device)[None, :, None, None]
        observation_std = torch.tensor([.229, .224, .225], device=device)[None, :, None, None]
        latents = []
        for start in range(0, len(rgb_input), 32):
            batch = rgb_input[start:start + 32]
            # Fixed batch shape for equal-frame comparisons, including the final batch.
            padded = np.concatenate((batch, np.repeat(batch[-1:], 32 - len(batch), axis=0)))
            images = torch.from_numpy(padded).to(device).permute(0, 3, 1, 2).float() / 255
            images = (images - observation_mean) / observation_std
            latents.append(host.encode({"pixels": images[:, None]})["emb"][:len(batch), 0].cpu())
        z[torch.from_numpy(visible)] = torch.cat(latents)
        mean = torch.tensor(stats["mean"], device=device)
        std = torch.tensor(stats["std_sample_correction_1"], device=device)
        a = torch.zeros((*valid.shape, 192), dtype=torch.float32)
        actual = torch.from_numpy(raw[valid]).to(device)
        # The author Embedder has a length-one temporal kernel. Independent
        # executed blocks can be embedded together without crossing an episode.
        if host.action_encoder.patch_embed.kernel_size != (1,):
            raise ValueError("Block-independent encoding requires the released pointwise action embedder")
        if canonical_action_host is None:
            embedded = host.action_encoder(((actual - mean) / std).reshape(-1, 1, 10))[:, 0].cpu()
        else:
            from .prbd_native import encode_native_controls
            embedded = encode_native_controls(canonical_action_host, raw[valid], stats).cpu()
        a[torch.from_numpy(valid)] = embedded
        batch = {"seed": torch.as_tensor(data["seed"]), "z": z, "a": a,
                 "lengths": torch.from_numpy(lengths)}
        if "matched_rgb" in data:
            matched = []
            for i, length in enumerate(lengths):
                same_images = np.all(rgb[i, :, :, 4:7] == rgb[i, 0, 0, 4:7])
                same_past = np.all(raw[i, :, :, 4:6] == raw[i, 0, 0, 4:6])
                matched.append(bool(np.all(length >= 6) and same_images and same_past))
            if not np.array_equal(matched, data["matched_rgb"]):
                raise ValueError("Reconstructed matched visual/command roots disagree with raw archive")
            batch["matched_rgb"] = torch.tensor(matched)
        return batch



def single_episodes(batch):
    return [(seed.item(), z[:n+1], a[:n], None)
            for seed, z, a, n in zip(batch["seed"], batch["z"], batch["a"], batch["lengths"], strict=True)
            if n >= 3]
