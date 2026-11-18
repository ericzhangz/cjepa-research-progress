"""Read one sensor contrast per arrival from the two issued native explanations."""

import json

import numpy as np
import torch

from . import native_full_depth_substep as substep
from .native_full_depth_first_effect import rollout
from .native_full_depth_paired_effect import OUTPUT as FITTED, ObservableChart
from .native_full_depth_reversible import SWITCH, action_embeddings, block_windows, encode_rgb
from .native_full_depth_substep_candidates import candidate_metrics
from .native_full_depth_task_geometry import (
    HORIZONS, block_reversals, chart_pose, json_array, scene_support, shape_cost, source_predictions,
)
from .native_paired_effect_delivery import DELIVERY, physical_choice
from .native_pusht_visible_measurement import render_t_chroma, visible_t_chroma


PAIRS = FITTED.parent / "native_visible_source_pairs_20261115"
OUTPUT = FITTED.parent / "native_visible_source_20261115"
CHROMA_UNIT = 1.0 / 41


def contrast_samples(projected, rgb, common_mask=None):
    """One score/scale for each actual arrival, with a shared mask for both predictions."""
    seen, valid = visible_t_chroma(rgb)
    if common_mask is not None:
        valid = valid & common_mask
    mean = projected.mean(axis=-4)
    difference = (projected[..., 1, :, :, :] - projected[..., 0, :, :, :]) * valid
    energy = np.einsum("...ij,...ij->...", difference, difference, dtype=np.float64)
    strength = np.sqrt(energy)
    score = np.einsum("...ij,...ij->...", difference, np.where(valid, seen - mean, 0.), dtype=np.float64)
    return np.divide(score, strength, out=np.zeros_like(score), where=strength > 0), strength


def fit_noise(contrast, strength):
    """Preserve the observed phase variances and shrink covariance with co-visibility."""
    present = strength > 0
    counts = present.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("The calibration data must expose each camera phase")
    center = (contrast * present).sum(axis=0) / counts
    scaled = np.where(present, contrast - center, 0) / np.sqrt(counts / len(contrast))
    covariance = scaled.T @ scaled / len(scaled) + CHROMA_UNIT ** 2 * np.eye(scaled.shape[1])
    return center, covariance, counts


def arrival_factors(scores, magnitude, bias, covariance, *, endpoint=False):
    """Score both modes using each visible prefix's observed principal covariance."""
    factors = np.zeros((*scores.shape[:-1], 2, scores.shape[-1]))
    residual = scores[..., None, :] - np.array([-0.5, 0.5])[:, None] * magnitude[..., None, :] - bias
    if endpoint:
        factors[..., -1] = np.where((magnitude[..., -1] > 0)[..., None],
                                    -.5 * residual[..., -1] ** 2 / covariance[-1, -1], 0.)
        return factors
    size = scores.shape[-1]
    residual = residual.reshape(-1, 2, size)
    flat = factors.reshape(-1, 2, size)
    present = magnitude.reshape(-1, size) > 0
    for end in range(1, size + 1):
        patterns, labels = np.unique(present[:, :end], axis=0, return_inverse=True)
        for i, pattern in enumerate(patterns):
            if pattern[-1]:
                rows = labels == i
                index = np.flatnonzero(pattern)
                chol = np.linalg.cholesky(covariance[np.ix_(index, index)])
                # Need the last row of L^-1 to form the conditional residual.
                vector = np.linalg.solve(chol.T, np.eye(len(index))[:, -1])
                flat[rows, :, end - 1] = -0.5 * (residual[rows][:, :, index] @ vector) ** 2
    return factors


def support_for(factors, device):
    prior, posterior = scene_support(torch.as_tensor(factors, device=device))
    return {"prior": prior, "posterior": posterior}


def summarize_pair(left, right, factual, active, modes, ticks, device):
    left, right = tuple(torch.as_tensor(value, device=device) for value in (left, right))
    middle, directed, nll = substep.nll_pairs(left, right)
    result = support_for(factual, device)
    result.update(middle=middle, directed=directed, nll=nll, active_nll=nll[active].mean(),
                  neutral_abs=middle[~active].abs().mean(), signed_half_gap=directed[active].mean(),
                  **block_reversals(result, modes, active, ticks))
    return result


def rendered_arm(projected, rgb, modes, noise, active, ticks, device, *, endpoint=False):
    factual = np.stack([rgb[:, tick, mode] for tick, mode in enumerate(modes)], axis=1)

    def factors(images, mask=None):
        return arrival_factors(*contrast_samples(projected, images, mask), *noise, endpoint=endpoint)

    _, common = visible_t_chroma(rgb)
    shared = common[:, :, 0] & common[:, :, 1]
    left, right = [factors(rgb[:, :, k], shared) for k in (0, 1)]
    own = [factors(rgb[:, :, k]) for k in (0, 1)]
    arrived = factors(factual)
    np.testing.assert_array_equal(arrived, np.stack([own[m][:, t] for t, m in enumerate(modes)], axis=1))
    result = summarize_pair(left, right, arrived, active, modes, ticks, device)
    l, r = (torch.as_tensor(a, device=device) for a in own)
    _, _, own_nll = substep.nll_pairs(l, r)
    result.update(own_mask_active_nll=own_nll[active].mean(), common_mask_fraction=float(shared.mean()),
                  changed_pixel_masks=int(np.any(common[:, :, 0] != common[:, :, 1], axis=(-2, -1)).sum()),
                  actual_log_factors=arrived.tolist())
    return result


def pose_arm(projected, observed, modes, bundle, active, ticks, device):
    def factors(arrived):
        return substep.chronological_evidence(projected, arrived, bundle["drift"], bundle["covariance"])

    left, right = [factors(observed[:, :, k]) for k in (0, 1)]
    actual = torch.stack([observed[:, t, k] for t, k in enumerate(modes)], dim=1)
    return summarize_pair(left, right, factors(actual), active, modes, ticks, device)


def calibrate(model, adapter, gate, hold, prefix, chart):
    device = hold.device
    with np.load(substep.PAIRS / "calibration.npz", allow_pickle=False) as raw:
        values = {"z": encode_rgb(model, raw["rgb"]),
                  "actions": torch.as_tensor(raw["actions"], dtype=hold.dtype, device=device)}
        images = raw["path_rgb"]
        scenes = raw["scene_id"].tolist()
    commands = action_embeddings(model, values["actions"])
    residuals, magnitudes = [], []
    for tick in range(commands.size(2)):
        history, past, controls = substep.root_controls(values, commands, slice(None), tick, hold)
        outputs = torch.stack([substep.predict_step(model, adapter, gate, prefix, history, past, controls, phase)
                               for phase in range(1, substep.PHASES + 1)], dim=-2)
        projected = render_t_chroma(chart_pose(chart(outputs)).cpu().numpy())
        observed = images[:, :, :, tick].reshape(-1, *images.shape[-5:])
        for mode in (0, 1):
            score, magnitude = contrast_samples(projected, observed[:, mode])
            residuals.append(score - (mode - 0.5) * magnitude)
            magnitudes.append(magnitude)
    bias, covariance, counts = fit_noise(np.concatenate(residuals), np.concatenate(magnitudes))
    return (bias, covariance), {"scene_ids": scenes, "present_counts": counts.tolist(), "bias": bias.tolist(),
                                "covariance": covariance.tolist()}


def score_candidates(model, adapter, gate, hold, chart, coarse, embedded, scenes, phases, ticks, modes, source):
    device = hold.device
    with np.load(PAIRS / "post_switch_candidates.npz", allow_pickle=False) as archive:
        if (archive["scene_id"].tolist() != scenes or archive["phase"].tolist() != [phases[t] for t in ticks]
                or archive["mode"].tolist() != [modes[t] for t in ticks] or archive["future_mode"].tolist() != [0, 1]):
            raise ValueError("Candidate outcomes must use the same actual roots and mode order as source evidence")
        cmd = action_embeddings(model, torch.as_tensor(archive["actions"], dtype=hold.dtype, device=device))
        goal = chart(encode_rgb(model, archive["goal_rgb"]))
        true_cost = torch.as_tensor(archive["coverage"][..., HORIZONS], device=device)
        true_xy = torch.as_tensor(archive["state"][..., HORIZONS, 2:4], device=device)
    z = []
    for tick in ticks:
        root, past = block_windows(coarse, embedded, tick + 1, hold)
        z.append(rollout(model, adapter, gate, root, past, cmd[None].expand(len(root), -1, -1, -1),
                         compact=True)[..., HORIZONS, :])
    prediction = chart(torch.stack(z, dim=1))
    cost = shape_cost(prediction, goal[:, None, None, None, None])
    predicted_xy = chart_pose(prediction)[..., :2]
    coverage = torch.stack([true_cost[:, j, modes[t]] for j, t in enumerate(ticks)], dim=1)
    xy = torch.stack([true_xy[:, j, modes[t]] for j, t in enumerate(ticks)], dim=1)
    supports = {k: s["posterior"][:, ticks, -1] for k, s in source.items()}
    supports.update(uniform=torch.full_like(supports["pose"], 0.5),
                    skip_current=source["camera_path"]["prior"][:, ticks],
                    known_mode=supports["pose"].new_tensor([modes[t] for t in ticks]).expand_as(supports["pose"]))
    results = {}
    for key, p in supports.items():
        masses = torch.stack((1 - p, p), dim=-1)[:, :, None, :, None]
        results[key] = candidate_metrics((cost * masses).sum(dim=3), coverage,
                                         (predicted_xy * masses[..., None]).sum(dim=3), xy)
    return results


def evaluate_delivery(model, adapter, gate, hold, prefix, chart, bundle, noise):
    device = hold.device
    with np.load(DELIVERY, allow_pickle=False) as raw:
        ids, names = raw["scene_id"].tolist(), raw["candidate"].tolist()
        modes = torch.as_tensor(raw["actual_mode"][:, -1], device=device)
        coarse = encode_rgb(model, raw["actual_rgb"][:, ::5])
        images = raw["actual_rgb"][:, 1:].reshape(len(ids), -1, 5, 96, 96, 3)
        seen = chart(encode_rgb(model, images))
        commands = torch.tensor(raw["prefix_action"], dtype=hold.dtype, device=device)[None].expand(len(ids), -1, -1, -1)
        goal = chart(encode_rgb(model, raw["goal_rgb"]))
        action = torch.tensor(raw["candidate_action"], dtype=hold.dtype, device=device)
        plans = action_embeddings(model, action[:, None])[None].expand(len(ids), -1, -1, -1)
        physical = torch.as_tensor(raw["candidate_coverage"][..., -1], device=device)
        coverage = physical[torch.arange(len(ids), device=device), modes]
    issued, embedded = source_predictions(model, adapter, gate, hold, prefix, coarse, commands)
    issued = chart(issued)
    projected = render_t_chroma(chart_pose(issued).cpu().numpy())
    root, past = block_windows(coarse, embedded, commands.size(1), hold)
    candidate = chart(rollout(model, adapter, gate, root, past, plans, compact=True)[..., 0, :])
    predicted_cost = shape_cost(candidate, goal[:, None, None])
    factor_sets = {
        "pose": substep.chronological_evidence(issued, seen, bundle["drift"], bundle["covariance"]),
        "camera_path": arrival_factors(*contrast_samples(projected, images), *noise),
    }
    results = {}
    for arm, factors in factor_sets.items():
        history = support_for(factors, device)
        posterior = history["posterior"][:, -1, -1]
        base = history["prior"][:, -1]
        results[arm] = {**history, "brier_before": (base - modes).square().mean(),
                        "brier_after": (posterior - modes).square().mean(),
                        "decisions": physical_choice(predicted_cost, SWITCH + (1 - 2 * SWITCH) * posterior, coverage),
                        "skip": physical_choice(predicted_cost, SWITCH + (1 - 2 * SWITCH) * base, coverage)}
    results["uniform"] = physical_choice(predicted_cost, torch.full_like(base, 0.5), coverage)
    results["known_mode"] = physical_choice(predicted_cost, modes.double(), coverage)
    return {"scene_ids": ids, "candidate_names": names, "modes": modes, "results": results}


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, adapter, gate, hold, *_ = substep.load_core(device)
    prefix = substep.NativeActionPrefixes(hold.numel()).to(device).eval()
    bundle = torch.load(FITTED / "forecast_only.pt", map_location=device, weights_only=True)
    adapter.load_state_dict(bundle["adapter"])
    prefix.load_state_dict(bundle["prefix"])
    chart = ObservableChart(bundle["chart"]).eval()
    noise, calibration = calibrate(model, adapter, gate, hold, prefix, chart)
    print("Camera calibration", calibration["present_counts"], flush=True)
    with np.load(PAIRS / "fresh_switches.npz", allow_pickle=False) as raw:
        scenes, modes, phases = (raw[k].tolist() for k in ("scene_id", "mode", "phase"))
        coarse = encode_rgb(model, raw["rgb"])
        images = raw["path_rgb"]
        fine = chart(encode_rgb(model, images))
        commands = torch.as_tensor(raw["actions"], dtype=hold.dtype, device=device)
        actual = raw["path_state"][..., 2:5]
        active = torch.as_tensor(np.linalg.norm(actual[:, :, 1, :, :2] - actual[:, :, 0, :, :2], axis=-1).max(-1) >= 1,
                                 device=device)
    issued, embedded = source_predictions(model, adapter, gate, hold, prefix, coarse, commands)
    pose = chart(issued)
    projected = render_t_chroma(chart_pose(pose).cpu().numpy())
    projected_truth = render_t_chroma(actual)
    ticks = [phases.index(n) for n in ("reverse_open", "reverse_brake")]
    source = {"pose": pose_arm(pose, fine, modes, bundle, active, ticks, device),
              "camera_endpoint": rendered_arm(projected, images, modes, noise, active, ticks, device, endpoint=True),
              "camera_path": rendered_arm(projected, images, modes, noise, active, ticks, device),
              "camera_actual_pose_oracle": rendered_arm(projected_truth, images, modes, noise, active, ticks, device)}
    decisions = score_candidates(model, adapter, gate, hold, chart, coarse, embedded, scenes, phases, ticks, modes, source)
    result = {"scene_ids": scenes, "phases": phases, "modes": modes, "active": active, "calibration": calibration,
              "source": source, "fixed_candidates": decisions,
              "delivery_replay": evaluate_delivery(model, adapter, gate, hold, prefix, chart, bundle, noise)}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, default=json_array, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
