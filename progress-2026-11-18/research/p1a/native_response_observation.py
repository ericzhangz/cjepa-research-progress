"""Calibrate public observation likelihoods for issued native full-view responses.

The native rollout predicts full-view embeddings for planning. The factual
likelihood scores the arrived encoder output in its public observation channel.
The frozen patch map connects those coordinates while preserving gradients to
the issued modes. Its only runtime inputs are the issued latent and public
camera center. The declared deterministic strip channel has one common mean.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn


FULL, STRIP, PATCH = range(3)
WIDTH = 192
SCALE_FLOOR = 1e-3
CANVAS_RADIUS = (96 - 1) / 2
LOG_NORMALIZER = 0.5 * math.log(2 * math.pi)
ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "JEPA_OIC_Codex_Handoff_2026-11-03/runs"


def _scales(values):
    return values.std(dim=0, unbiased=False).clamp_min(0.01)


def _residual_scale(residual):
    return residual.square().mean(dim=0).sqrt().clamp_min(SCALE_FLOOR)


def _normalize_center(center):
    return (center - CANVAS_RADIUS) / CANVAS_RADIUS


def _log_density(arrived, means, scales):
    normalized = (arrived[:, None] - means) / scales
    return -(0.5 * normalized.square().sum(dim=-1)
             + scales.log().sum() + arrived.shape[-1] * LOG_NORMALIZER)


def _bounded_scales(reference, coordinate):
    lower = (0.25 * reference).clamp_min(SCALE_FLOOR)
    upper = 4 * reference
    # Start at the offline residual scale even where the floor makes the bounds asymmetric.
    low, high = lower.log(), upper.log()
    reference_fraction = ((reference.log() - low) / (high - low)).clamp(1e-3, 1 - 1e-3)
    fraction = (coordinate + reference_fraction.logit()).sigmoid()
    return torch.lerp(low, high, fraction).exp()


class NativeResponseObservation(nn.Module):
    """One mode-blind map to the frozen public LeWM encoder's coordinates."""

    def __init__(self):
        super().__init__()
        self.patch_mean = nn.Sequential(nn.Linear(WIDTH + 2, 32), nn.SiLU(), nn.Linear(32, WIDTH))
        self.patch_precision_coordinate = nn.Parameter(torch.zeros(WIDTH))
        self.full_precision_coordinate = nn.Parameter(torch.zeros(WIDTH))
        self.register_buffer("input_offset", torch.zeros(WIDTH))
        self.register_buffer("input_scale", torch.ones(WIDTH))
        self.register_buffer("target_offset", torch.zeros(WIDTH))
        self.register_buffer("target_scale", torch.ones(WIDTH))
        self.register_buffer("patch_reference_sigma", torch.full((WIDTH,), float("nan")))
        self.register_buffer("full_reference_sigma", torch.full((WIDTH,), float("nan")))
        self.register_buffer("strip_mean", torch.zeros(WIDTH))

    def freeze_mean(self):
        self.patch_mean.requires_grad_(False)

    @torch.no_grad()
    def set_full_reference_sigma(self, sigma):
        sigma = torch.as_tensor(sigma).to(self.full_reference_sigma)
        if sigma.shape != (WIDTH,) or not torch.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError("Full-view calibration needs one finite nonnegative residual scale per latent coordinate")
        self.full_reference_sigma.copy_(sigma.clamp_min(SCALE_FLOOR))

    def _patch_means(self, means, center):
        features = (means - self.input_offset) / self.input_scale
        normalized_center = _normalize_center(center)[:, None].expand(*means.shape[:-1], 2)
        normalized = self.patch_mean(torch.cat((features, normalized_center), dim=-1))
        return self.target_offset + self.target_scale * normalized

    def _score_channel(self, means, arrived, channel, center):
        if channel == STRIP:
            fixed_sigma = self.strip_mean.new_full((WIDTH,), SCALE_FLOOR)
            return _log_density(arrived, self.strip_mean[None, None], fixed_sigma).expand_as(means[..., 0])
        if channel == PATCH:
            if center is None or center.shape != (len(means), 2):
                raise ValueError("The patch observation requires its public [batch, 2] pixel center")
            if not torch.isfinite(center).all() or ((center < 0) | (center > 2 * CANVAS_RADIUS)).any():
                raise ValueError("Patch centers must lie in the public image coordinates")
            means = self._patch_means(means, center.to(means))
            reference, coordinate = self.patch_reference_sigma, self.patch_precision_coordinate
        elif channel == FULL:
            reference, coordinate = self.full_reference_sigma, self.full_precision_coordinate
        else:
            raise ValueError("Unknown public observation channel")
        if not torch.isfinite(reference).all():
            raise ValueError("Calibrate the declared observation channel before scoring it")
        return _log_density(arrived, means, _bounded_scales(reference, coordinate))

    def forward(self, issued_means, arrived, channel, patch_center=None):
        """Return full Gaussian log densities, one per real arrival and issued mode."""
        if issued_means.ndim != 3 or issued_means.shape[-1] != WIDTH:
            raise ValueError("Expected issued means [batch, modes, 192]")
        if arrived.shape != (len(issued_means), WIDTH):
            raise ValueError("Expected a public arrival [batch, 192] aligned with the issued modes")
        if isinstance(channel, int):
            return self._score_channel(issued_means, arrived, channel, patch_center)
        if not torch.is_tensor(channel) or channel.dtype != torch.long or channel.shape != (len(issued_means),):
            raise ValueError("Expected one declared observation channel per arrival")
        if ((channel < FULL) | (channel > PATCH)).any():
            raise ValueError("Unknown public observation channel")
        result = issued_means.new_empty(issued_means.shape[:2])
        for kind in (FULL, STRIP, PATCH):
            rows = torch.where(channel == kind)[0]
            if rows.numel():
                center = None if patch_center is None else patch_center[rows]
                result[rows] = self._score_channel(issued_means[rows], arrived[rows], kind, center)
        return result

    @staticmethod
    def nll(log_weights, log_likelihood):
        return -(log_weights + log_likelihood).logsumexp(dim=-1) / WIDTH

    @classmethod
    def load(cls, path, device):
        module = cls().to(device)
        module.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        module.freeze_mean()
        return module


@dataclass
class PatchPanel:
    scene_ids: np.ndarray
    full: torch.Tensor
    public: torch.Tensor
    center: torch.Tensor
    strip: torch.Tensor
    matched: torch.Tensor


def _load_panel(name, count, device):
    latent_path = ARCHIVE / f"{name}_native{count}_latents.npz"
    with np.load(ARCHIVE / f"{name}.npz", allow_pickle=False) as raw, np.load(
        latent_path, allow_pickle=False,
    ) as cached:
        scene_ids = raw["scene_ids"]
        np.testing.assert_array_equal(scene_ids, cached["scene_ids"])
        np.testing.assert_array_equal(raw["observation_steps"], np.arange(0, 31, 5))
        if not raw["actual_step"].all():
            raise ValueError("Observation calibration requires actually executed prefix blocks")
        full, public = (cached[key] for key in ("full_latents", "obs_latents"))
        np.testing.assert_array_equal(full.shape, (count, 4, 7, WIDTH))
        np.testing.assert_array_equal(full.shape, public.shape)
        masks = raw["mask"]
        patch_columns = np.flatnonzero((masks.sum(axis=(-1, -2)) == 9).all(axis=0))
        if len(patch_columns) != 1:
            raise ValueError("Expected one calibrated public patch channel across scenes")
        index = patch_columns.item()
        _, y, x = np.nonzero(masks[:, index])
        center = np.stack((x, y), axis=-1).reshape(count, 9, 2).mean(axis=1)
        strip_columns = np.flatnonzero((masks == masks[0, -1]).all(axis=(0, 2, 3)))
        strip = public[:, :, strip_columns].reshape(-1, WIDTH)
        np.testing.assert_array_equal(strip, np.broadcast_to(strip[0], strip.shape))

        # Same camera within a scene, matching legal prefixes, and unlike public arrivals.
        first, second = np.triu_indices(public.shape[1], 1)
        prior = (public[:, first, :index] == public[:, second, :index]).all(axis=(-1, -2))
        image = raw["obs_rgb"][:, :, index]
        distinct = (image[:, first] != image[:, second]).any(axis=(-1, -2, -3))
        scene, pairs = np.nonzero(prior & distinct)
        matched = np.stack((scene, first[pairs], second[pairs]), axis=-1)

    return PatchPanel(scene_ids, torch.as_tensor(full[:, :, index].copy(), device=device),
                      torch.as_tensor(public[:, :, index].copy(), device=device),
                      torch.as_tensor(center, device=device, dtype=torch.float32),
                      torch.as_tensor(strip[0].copy(), device=device),
                      torch.as_tensor(matched, device=device))


def _centers(panel):
    return panel.center.repeat_interleave(panel.full.shape[1], dim=0)


@torch.no_grad()
def _evaluate(panel, obs, view_only, view_sigma, prior_sigma):
    full = panel.full.flatten(0, 1)
    public = panel.public.flatten(0, 1)
    center = _centers(panel)
    fitted = obs._patch_means(full[:, None], center)
    view = obs.target_offset + obs.target_scale * view_only(_normalize_center(center))
    prior = obs.target_offset.expand_as(view)

    def score(means, sigma):
        return (-_log_density(public, means, sigma).mean() / WIDTH).item()

    def rms(x):
        return x.square().mean().sqrt().item()

    scene, first, second = panel.matched.unbind(-1)
    predicted = fitted.reshape_as(panel.full)
    actual_delta = panel.public[scene, second] - panel.public[scene, first]
    predicted_delta = predicted[scene, second] - predicted[scene, first]
    alignment = torch.nn.functional.cosine_similarity(actual_delta, predicted_delta, dim=-1)
    log_odds_contrast = (actual_delta * predicted_delta / obs.patch_reference_sigma.square()).sum(-1)
    return {
        "scenes": len(panel.scene_ids),
        "public_latent_nll_per_dimension": {
            "full_to_public": score(fitted, obs.patch_reference_sigma),
            "prior_only": score(prior[:, None], prior_sigma),
            "view_only": score(view[:, None], view_sigma),
        },
        "public_latent_rmse": {
            "full_to_public": rms(fitted[:, 0] - public),
            "prior_only": rms(prior - public),
            "view_only": rms(view - public),
        },
        "same_camera_matched_prefix_pairs": {
            "count": len(scene),
            "scenes": scene.unique().numel(),
            "public_delta_rms": rms(actual_delta),
            "predicted_delta_rms": rms(predicted_delta),
            "mean_delta_cosine": alignment.mean().item(),
            "positive_log_odds_contrasts": (log_odds_contrast > 0).sum().item(),
            "median_log_odds_contrast": log_odds_contrast.median().item(),
        },
    }


def _fit(train, dev, device):
    torch.manual_seed(1611)
    torch.set_float32_matmul_precision("highest")
    obs = NativeResponseObservation().to(device)
    view_only = nn.Sequential(nn.Linear(2, 32), nn.SiLU(), nn.Linear(32, WIDTH)).to(device)
    full = train.full.flatten(0, 1)
    public = train.public.flatten(0, 1)
    center = _centers(train)
    normalized_center = _normalize_center(center)
    obs.input_offset.copy_(full.mean(0))
    obs.input_scale.copy_(_scales(full))
    obs.target_offset.copy_(public.mean(0))
    obs.target_scale.copy_(_scales(public))
    obs.strip_mean.copy_(train.strip)
    optimizer = torch.optim.Adam([*obs.patch_mean.parameters(), *view_only.parameters()], lr=1e-3)
    target = (public - obs.target_offset) / obs.target_scale
    features = torch.cat(((full - obs.input_offset) / obs.input_scale, normalized_center), dim=-1)
    for _ in range(512):
        optimizer.zero_grad(set_to_none=True)
        map_error = obs.patch_mean(features) - target
        view_error = view_only(normalized_center) - target
        (map_error.square().mean() + view_error.square().mean()).backward()
        optimizer.step()

    with torch.no_grad():
        fitted = obs._patch_means(full[:, None], center)[:, 0]
        view = obs.target_offset + obs.target_scale * view_only(normalized_center)
        obs.patch_reference_sigma.copy_(_residual_scale(public - fitted))
        view_sigma = _residual_scale(public - view)
        prior_sigma = _residual_scale(public - obs.target_offset)
    obs.freeze_mean()
    report = {
        "calibration": {"steps": 512, "lr": 1e-3, "seed": 1611, "full_batch": True},
        "fit": _evaluate(train, obs, view_only, view_sigma, prior_sigma),
        "historical_development": _evaluate(dev, obs, view_only, view_sigma, prior_sigma),
    }
    return obs, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    train = _load_panel("train_11033026", 96, args.device)
    dev = _load_panel("confirmation_11034026", 48, args.device)
    if np.intersect1d(train.scene_ids, dev.scene_ids).size:
        raise ValueError("Calibration and development scenes must be disjoint")
    torch.testing.assert_close(train.strip, dev.strip, rtol=0, atol=0)
    obs, report = _fit(train, dev, args.device)
    args.output.mkdir(parents=True)
    torch.save(obs.state_dict(), args.output / "observation.pt")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
