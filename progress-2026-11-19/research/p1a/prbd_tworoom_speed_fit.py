"""Fit and read PRBD on actual native TwoRoom speed variations, with disjoint reset seeds."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from .prbd import quadrature
from .prbd_native import NativePrototypes
from .prbd_public_fit import calibrate, fit, windows


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/prbd_tworoom_speed_20261119"
RELEASE = ROOT / "runs/prbd_official_protocols_20261119/tworoom"


def load_host(device, source, model_dir):
    sys.path.insert(0, str(ROOT / "research_sources_2026-10-22/lewm_assets"))
    from prepare_assets import load_model

    return load_model(device, source=source, model_dir=model_dir).requires_grad_(False).eval()


@torch.no_grad()
def encode_split(host, split, device, stats, output=OUT):
    with np.load(output / f"data_{split}.npz", allow_pickle=False) as data:
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
        a[torch.from_numpy(valid)] = host.action_encoder(((actual - mean) / std).reshape(-1, 1, 10))[:, 0].cpu()
        batch = {"seed": torch.as_tensor(data["seed"]), "z": z, "a": a,
                 "lengths": torch.from_numpy(lengths)}
        if split == "validation":
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["encode", "fit", "measure"])
    parser.add_argument("--source", type=Path, required=True,
                        help="Original LeWM directory containing jepa.py and module.py")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Official TwoRoom directory containing config.json and weights.pt")
    parser.add_argument("--output", type=Path, default=OUT,
                        help="Directory containing the raw/encoded splits and fitted.pt")
    parser.add_argument("--action-statistics", type=Path, default=RELEASE / "action_statistics.json")
    parser.add_argument("--report-dir", type=Path,
                        help="Separate measurement destination; defaults to --output")
    args = parser.parse_args()
    stage, output = args.stage, args.output
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20261119)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if stage == "encode":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        host = load_host(device, args.source, args.model_dir)
        stats = json.loads(args.action_statistics.read_text(encoding="utf-8"))
        for split in ("calibration", "fit", "validation"):
            encoded = encode_split(host, split, device, stats, output)
            torch.save(encoded, output / f"encoded_{split}.pt")
            print(split, encoded["z"].shape, flush=True)
        return
    model = NativePrototypes(load_host("cpu", args.source, args.model_dir)).eval()
    if stage == "fit":
        cal, training = [single_episodes(torch.load(output / f"encoded_{split}.pt", weights_only=True))
                         for split in ("calibration", "fit")]
        if any(row[0] == other[0] for row in cal for other in training):
            raise ValueError("Calibration and training must have different reset seeds")
        noise, noise_log = calibrate(model.host, cal)
        scales, masses = quadrature()
        observation_model = {"noise_cholesky": noise, "transition": torch.eye(4, dtype=torch.float64),
                             "scales": scales, "masses": masses}
        initial = model.offsets.detach().clone()
        start = perf_counter()
        logs = fit(model, training, noise, observation_model["transition"], scales, masses)
        elapsed = perf_counter() - start
        _, z, a, _ = training[0]
        with torch.no_grad():
            w, c = windows(z[:-1]), windows(a)
            torch.testing.assert_close(model(w, c)[:, 0], model.host.predict(w, c)[:, -1], atol=0, rtol=0)
        torch.save({**observation_model, "offsets": model.offsets.detach(), "initialized_offsets": initial},
                   output / "fitted.pt")
        report = {"calibration": noise_log, "calibration_seeds": [e[0] for e in cal],
                  "training": logs, "optimization_steps": len(logs), "actual_scored_arrivals": sum(r["transitions"] for r in logs),
                  "fit_seconds": elapsed, "nominal_anchor_exact_after_fit": True,
                  "offset_change_l2": (model.offsets.detach() - initial).flatten(1).norm(dim=1).tolist()}
        (output / "fit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{len(logs)} fitted episodes, {report['actual_scored_arrivals']} arrivals", flush=True)
        return
    from .prbd_tworoom_speed_measure import measure

    batch = torch.load(output / "encoded_validation.pt", weights_only=True)
    cases = [{key: batch[key][i] for key in ("z", "a", "lengths")} |
             {"seed": seed.item(), "matched_rgb": bool(batch["matched_rgb"][i])}
             for i, seed in enumerate(batch["seed"])]
    fitted = torch.load(output / "fitted.pt", weights_only=True)
    report_dir = args.report_dir or output
    report_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for field, name in [("initialized_offsets", "initialized"), ("offsets", "fitted")]:
            model.offsets.copy_(fitted[field])
            start = perf_counter()
            results = measure(model, fitted, cases)
            results["elapsed_seconds"] = perf_counter() - start
            (report_dir / f"{name}_measure.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            print(f"{name} held-out measurements complete", flush=True)


if __name__ == "__main__":
    main()
