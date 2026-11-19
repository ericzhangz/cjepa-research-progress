"""Archived offline PRBD check on LeWM's published PushT frames and actions.

Uses the included captured panel; upstream code and checkpoint are external.
See runs/prbd_official_protocols_20261119/ASSETS.md. No environment is stepped.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "research_sources_2026-10-22/lewm_assets"
OUT = ROOT / "runs/prbd_official_protocols_20261119"
PANEL = OUT / "pusht_published_prefixes.npz"


def measure(args):
    import torch
    from .prbd import assimilate, issue, quadrature
    from .prbd_native import NativePrototypes, native_conditioned

    sys.path.insert(0, str(ASSETS))
    from prepare_assets import load_model

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    host = load_model("cpu", source=args.lewm_source, model_dir=args.model_dir).requires_grad_(False).eval()
    model = NativePrototypes(host).eval()
    with np.load(args.panel, allow_pickle=False) as data, torch.no_grad():
        # RGB in this HDF5 is already in the standard 224-square evaluation size.
        rgb = torch.from_numpy(data["rgb"])
        latents = []
        image_mean = torch.tensor([.485, .456, .406])[None, :, None, None]
        image_std = torch.tensor([.229, .224, .225])[None, :, None, None]
        for batch in rgb.flatten(0, 1).split(16):
            images = batch.permute(0, 3, 1, 2).float() / 255
            images = (images - image_mean) / image_std
            latents.append(host.encode({"pixels": images[:, None]})["emb"][:, 0])
        z = torch.cat(latents).reshape(*rgb.shape[:2], -1)
        raw = torch.from_numpy(data["actions"])
        commands = (raw - raw.new_tensor(data["mean"])) / raw.new_tensor(data["std"])
        actions = host.action_encoder(commands.reshape(len(raw), 8, 10))
        config = {"sample": "first 16 published training trajectories, frames 0..40 at stride 5",
                  "source_frames": [0, 5, 10], "real_write_frame": 15,
                  "held_actions_target_frames": [20, 25, 30, 35, 40],
                  "action_mean": data["mean"].tolist(), "action_std_ddof1": data["std"].tolist(),
                  "normalization_finite_rows": int(data["finite_action_rows"]),
                  "conditional_blocks": len(host.predictor.transformer.layers), "trainable_offsets": model.offsets.numel(),
                  "noise": "identity, uncalibrated host-interface check", "transport": "identity",
                  "weight_updates": 0, "source_prior": [0.25] * 4, "quadrature_nodes": 32, "nu": 3}
        responses = model(z[:, :3], actions[:, :3])
        neutral_window = z[:, 1:4]
        original_window = neutral_window
        future = model.rollout(neutral_window, actions[:, 1:]).double()
        max_anchor, raw_zero_error = [], []
        for h in range(5):
            control = actions[:, 1+h:4+h]
            original = host.predict(original_window, control)[:, -1]
            zero = native_conditioned(host, neutral_window, control, model.conditions[0])[:, -1]
            raw_zero_error.append(float((original-zero).abs().max()))
            max_anchor.append(float((original-future[:, 0, h]).abs().max()))
            torch.testing.assert_close(original, zero, atol=0, rtol=0)
            torch.testing.assert_close(original.double(), future[:, 0, h], atol=0, rtol=0)
            neutral_window = torch.cat((neutral_window[:, 1:], zero[:, None]), dim=1)
            original_window = torch.cat((original_window[:, 1:], original[:, None]), dim=1)
        transition = torch.eye(4, dtype=torch.float64)
        prior = transition.new_full((4,), .25)
        noise = torch.eye(z.size(-1), dtype=torch.float64)
        nodes, masses = quadrature()
        rows = []
        for i, response in enumerate(responses):
            signed = issue(response, prior, transition, noise)
            arrival = assimilate(signed, z[i, 3], nodes, masses)
            predicted = future[i]
            before = torch.einsum("k,khd->hd", signed.prior, predicted)
            after = torch.einsum("k,khd->hd", arrival.posterior, predicted)
            answer = z[i, 4:].double()
            if torch.equal(raw[i, 10:15], raw[i, 15:20]):
                raise ValueError("Expected the next executed action block to differ from the source block")
            rows.append({"episode": i, "posterior": arrival.posterior.tolist(),
                         "support_tv": float(.5 * (arrival.posterior - signed.prior).abs().sum()),
                         "issued_source_spread_l2": float(torch.pdist(response.double()).max()),
                         "prediction_shift_l2": (after-before).norm(dim=-1).tolist(),
                         "squared_error_gain_per_coordinate": ((before-answer).square()-(after-answer).square()).mean(dim=-1).tolist()})
    report = {"protocol": config, "nominal_max_abs_error_by_h": max_anchor,
              "raw_native_zero_max_abs_error_by_h": raw_zero_error, "records": rows}
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("measure",))
    parser.add_argument("--lewm-source", type=Path, required=True,
                        help="Official le-wm checkout containing jepa.py and module.py")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Official lewm-pusht directory containing config.json and weights.pt")
    parser.add_argument("--panel", type=Path, default=PANEL,
                        help="Captured published RGB/action panel; no extraction is needed")
    parser.add_argument("--output", type=Path, default=OUT / "pusht_native_probe.reproduced.json")
    measure(parser.parse_args())
