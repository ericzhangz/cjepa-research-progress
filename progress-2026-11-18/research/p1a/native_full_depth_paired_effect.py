"""Fit finite arrival evidence and multistep geometry through the same native explanations."""

import copy
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import native_full_depth_substep as substep
from .native_full_depth_first_effect import rollout
from .native_full_depth_reversible import action_embeddings, encode_rgb
from .native_full_depth_task_geometry import (
    OUTPUT as INITIAL, chart_at_latent, evaluate, json_array, pose_chart,
)
from .native_reversible_pusht import make_scene


PAIRS = INITIAL.parent / "native_paired_effect_pairs_20261115"
OUTPUT = INITIAL.parent / "native_full_depth_paired_effect_20261115"
UPDATES = 192
SEED = 3115


class ObservableChart(nn.Module):
    def __init__(self, initial):
        super().__init__()
        self.weight = nn.Parameter(initial["weight"].clone())
        self.bias = nn.Parameter(initial["bias"].clone())

    def forward(self, z):
        return chart_at_latent(z, self.weight, self.bias)


def finite_evidence(predicted, arrived, whitener, bias):
    """Center and half-contrast of the two real arrivals' conditional log odds."""
    phases, width = predicted.shape[-2:]

    def white(path):
        return (path.flatten(-2) @ whitener.T).unflatten(-1, (phases, width))

    mu, actual = white(predicted), white(arrived)
    d = mu[..., 1, :, :] - mu[..., 0, :, :]
    e = actual[..., 1, :, :] - actual[..., 0, :, :]
    residual = actual.mean(dim=-3) - mu.mean(dim=-3) - white(bias)
    center = (d * residual).sum(-1)
    contrast = 0.5 * (d * e).sum(-1)
    target = 0.5 * e.square().sum(-1)
    return center, contrast, target


@torch.no_grad()
def load_fit(model, device):
    base = substep.panel(model, "fit")
    with np.load(substep.PAIRS / "fit.npz", allow_pickle=False) as archive:
        initial_rgb = archive["rgb"][..., 0, :, :, :]
        state = archive["state"]
        np.testing.assert_array_equal(initial_rgb, np.broadcast_to(initial_rgb[:, :1, :1], initial_rgb.shape))
        np.testing.assert_array_equal(state[..., 0, :], np.broadcast_to(state[:, :1, :1, 0, :], state[..., 0, :].shape))
        base["source_q"] = pose_chart(torch.tensor(archive["path_state"][..., 2:5], device=device).float())
        base["coarse_q"] = pose_chart(torch.tensor(state[..., 2:5], device=device).float())
        base["goal_z"] = encode_rgb(model, archive["goal_rgb"])
        poses = np.stack([make_scene(int(seed)).goal for seed in base["scene"]])
        base["goal_q"] = pose_chart(torch.as_tensor(poses, device=device, dtype=torch.float32))
    base["embedded"] = action_embeddings(model, base["actions"])
    return base


def prefix_predictions(model, adapter, gate, prefix, history, past, commands):
    return torch.stack([substep.predict_step(model, adapter, gate, prefix, history, past, commands, phase)
                        for phase in range(1, substep.PHASES + 1)], dim=-2)


def free_batch(model, adapter, gate, hold, fit, rows):
    z = fit["z"][rows]
    initial = z[:, 0, :, 0].flatten(0, 1)[:, None].expand(-1, 3, -1)
    controls = fit["embedded"][rows].flatten(0, 1)[:, None]
    pred = rollout(model, adapter, gate, initial, hold.expand(len(initial), 2, -1), controls)[:, 0]
    z_target = z[:, :, :, 1:].transpose(1, 2).flatten(0, 1)
    q_target = fit["coarse_q"][rows, :, :, 1:].transpose(1, 2).flatten(0, 1)
    return pred, z_target, q_target


def train(model, adapter, gate, hold, prefix, chart, values, fit_drift, fit_cov, *, evidence):
    parameters = [*adapter.parameters(), *prefix.parameters(), *chart.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    q_scale = fit_cov.diagonal().mean().sqrt().float()
    z_scale = values["targets"].std().clamp_min(0.01)
    lower = torch.linalg.cholesky(fit_cov).float()
    whiten = torch.linalg.solve_triangular(lower, torch.eye(lower.size(0), device=hold.device), upper=False)
    common_bias = fit_drift.float().unflatten(-1, (substep.PHASES, -1))
    trace = []

    def mse(predicted, target, unit):
        return ((predicted - target) / unit).square().mean()

    for step in range(UPDATES):
        rows = rng.choice(len(values["scene"]), size=2, replace=False).tolist()
        tick = step % values["embedded"].size(2)
        history, past, commands = substep.root_controls(values, values["embedded"], rows, tick, hold)
        issued = prefix_predictions(model, adapter, gate, prefix, history, past, commands)
        source = values["targets"][rows, :, :, tick].flatten(0, 2)
        physical = values["source_q"][rows, :, :, tick].flatten(0, 2)
        source_q = chart(issued)
        seen_q = chart(source)
        source_fit = mse(source_q, physical, q_scale) + mse(issued, source, z_scale)
        forecast, future_z, future_q = free_batch(model, adapter, gate, hold, values, rows)
        future_fit = mse(chart(forecast), future_q, q_scale) + mse(forecast, future_z, z_scale)
        observation = 0.5 * (mse(seen_q, physical, q_scale) +
                             mse(chart(values["goal_z"][rows]), values["goal_q"][rows], q_scale))
        center, contrast, target = finite_evidence(source_q, seen_q.detach(), whiten, common_bias)
        transfer = ((center.square() + (contrast - target).square()) / (1 + target)).mean()
        loss = observation + 0.5 * (source_fit + future_fit)
        if evidence:
            loss = loss + transfer
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
        optimizer.step()
        if (step + 1) % 24 == 0:
            row = {"update": step + 1, "observation": observation.item(), "source": source_fit.item(),
                   "future": future_fit.item(), "evidence": transfer.item(), "gradient_norm": norm.item()}
            trace.append(row)
            print("finite" if evidence else "forecast_only", row, flush=True)
    return trace


@torch.no_grad()
def temperature_scale(model, adapter, gate, hold, prefix, chart, calibration, drift, cov):
    embedded = action_embeddings(model, calibration["actions"])
    signs = []
    for tick in range(embedded.size(2)):
        root, past, cmd = substep.root_controls(calibration, embedded, slice(None), tick, hold)
        issued = chart(prefix_predictions(model, adapter, gate, prefix, root, past, cmd))
        arrived = chart(calibration["targets"][:, :, :, tick].flatten(0, 2))
        factors = [substep.chronological_evidence(issued, arrived[:, mode], drift, cov)
                   .sum(-1).diff(dim=-1).flatten() for mode in range(2)]
        signs.extend((-factors[0], factors[1]))
    signed = torch.cat(signs).cpu()

    def derivative(scale):
        return -torch.mean(signed * (-signed * scale).sigmoid()).item()

    if derivative(0) >= 0:
        return 0.
    low, high = 0., 1.
    while derivative(high) < 0 and high < 1024:
        high *= 2
    for _ in range(48):
        mid = (low + high) / 2
        if derivative(mid) < 0:
            low = mid
        else:
            high = mid
    return (high + low) / 2


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, base, gate, hold, w, b, _ = substep.load_core(device)
    original_prefix = substep.NativeActionPrefixes(hold.numel()).to(device).eval()
    old = torch.load(substep.OUTPUT / "prefix.pt", map_location=device, weights_only=True)
    original_prefix.load_state_dict(old["prefix"])
    initial = torch.load(INITIAL / "coordinates.pt", map_location=device, weights_only=True)
    base_chart = ObservableChart(initial).eval()
    print("Encoding 20 fit / 4 calibration scenes", flush=True)
    fit = load_fit(model, device)
    calibration = substep.panel(model, "calibration")
    baseline = lambda observed, predicted: base_chart(observed) - base_chart(predicted)
    noise = substep.calibrate_residuals(model, base, gate, hold, original_prefix, fit, baseline)
    temperature = temperature_scale(model, base, gate, hold, original_prefix, base_chart,
                                    calibration, initial["drift"], initial["covariance"])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {"fit_scenes": fit["scene"], "calibration_scenes": calibration["scene"], "updates": UPDATES,
              "training_covariance_pairs": noise[2] // 2, "temperature_on_frozen_score": temperature, "arms": {}}
    for arm in ("forecast_only", "finite_effect"):
        adapter, prefix, chart = (copy.deepcopy(part) for part in (base, original_prefix, base_chart))
        trace = train(model, adapter, gate, hold, prefix, chart, fit, *noise[:2], evidence=(arm == "finite_effect"))
        drift, covariance, n = substep.calibrate_residuals(
            model, adapter, gate, hold, prefix, calibration,
            lambda arrived, predicted: chart(arrived) - chart(predicted),
        )
        with (OUTPUT / f"{arm}.pt").open("xb") as file:
            torch.save({"adapter": adapter.state_dict(), "prefix": prefix.state_dict(), "chart": chart.state_dict(),
                        "drift": drift, "covariance": covariance}, file)
        result["arms"][arm] = {"fit_trace": trace, "calibration_rows": n, **evaluate(
            model, adapter, gate, hold, prefix, chart, drift, covariance, w, b, old, PAIRS,
        )}
    result["arms"]["frozen"] = evaluate(
        model, base, gate, hold, original_prefix, base_chart, initial["drift"], initial["covariance"],
        w, b, old, PAIRS, path_scale=temperature,
    )
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, default=json_array, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
