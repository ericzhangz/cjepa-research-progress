"""Fit native action-prefix measurements and consume only the arriving conditional innovations."""

import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import native_response_mixture_regulation as data
from .native_full_depth_conditioning import FirstEffectReadout, FullDepthConditioner, advance_log_support
from .native_full_depth_first_effect import OUTPUT as FIRST_EFFECT, rollout
from .native_full_depth_reversible import OUTPUT as COARSE, SWITCH, action_embeddings, block_windows, encode_rgb
from .native_full_depth_source_evidence import decompose_switches


PAIRS = data.RUNS / "native_full_depth_substep_pairs_20261115"
OUTPUT = data.RUNS / "native_full_depth_substep_20261115"
PHASES = 5
UPDATES = 320
SEED = 2915


class NativeActionPrefixes(nn.Module):
    """A single duration-conditioned affine change to the original action token."""

    def __init__(self, width):
        super().__init__()
        self.affine = nn.Parameter(torch.zeros(4, width))

    def forward(self, model, raw_action, step):
        if step == PHASES:
            return action_embeddings(model, raw_action)
        mean, scale = data._action_moments(raw_action)
        normalized = (raw_action - mean) / scale
        available = torch.arange(PHASES, device=normalized.device) < step
        masked = torch.where(available[None, :, None], normalized, 0).flatten(1)
        command = model.action_encoder(masked[:, None])[:, 0]
        elapsed = step / PHASES
        s0, s1, b0, b1 = self.affine
        return command + (1 - elapsed) * ((s0 + elapsed * s1) * command + b0 + elapsed * b1)


def load_core(device):
    model = data._native_model(device)
    old = torch.load(COARSE / "conditioner.pt", map_location=device, weights_only=True)
    trained = torch.load(FIRST_EFFECT / "conditioner.pt", map_location=device, weights_only=True)
    adapter = FullDepthConditioner(model, 2).to(device).eval()
    gate = FirstEffectReadout(model).to(device).eval()
    adapter.load_state_dict(trained["adapter"])
    gate.load_state_dict(trained["readout"])
    for part in (adapter, gate):
        part.requires_grad_(False)
    hold = action_embeddings(model, torch.zeros(1, PHASES, 2, device=device))[0]
    return model, adapter, gate, hold, old["w"], old["b"], old["scale"]


@torch.no_grad()
def panel(model, split):
    device = next(model.parameters()).device
    with np.load(PAIRS / f"{split}.npz", allow_pickle=False) as raw:
        coarse = encode_rgb(model, raw["rgb"])
        targets = encode_rgb(model, raw["path_rgb"])
        return {"z": coarse, "targets": targets, "position": torch.as_tensor(raw["path_state"][..., 2:4], device=device).float(),
                "actions": torch.as_tensor(raw["actions"], device=device).float(), "scene": raw["scene_id"].tolist()}


def root_controls(values, native_actions, rows, tick, hold):
    states, embedded = values["z"][rows], native_actions[rows]
    history, past = block_windows(states, embedded, tick, hold)
    raw = values["actions"][rows, :, tick]
    raw = raw[:, None].expand(-1, states.size(1), -1, -1, -1)
    past = past[:, None].expand(-1, states.size(1), -1, -1, -1)
    return [x.flatten(0, 2) for x in (history, past, raw)]


def predict_step(model, adapter, gate, prefix, history, past, raw_action, phase):
    command = prefix(model, raw_action, phase)
    return rollout(model, adapter, gate, history, past, command[:, None, None])[:, 0, :, 0]


def paired_objective(prediction, true_z, true_position, w, b, xy_scale, latent_scale):
    xy = prediction @ w + b
    shared = ((xy.mean(dim=1) - true_position.mean(dim=1)) / xy_scale).square().mean()
    difference = ((xy.diff(dim=1) - true_position.diff(dim=1)) / xy_scale).square().mean()
    latent = ((prediction - true_z) / latent_scale).square().mean()
    return 0.5 * shared + difference + 0.5 * latent


def fit(model, adapter, gate, hold, prefix, values, w, b):
    np_rng = np.random.default_rng(SEED)
    embedded = action_embeddings(model, values["actions"])
    latent_scale = values["targets"].std().clamp_min(0.01)
    xy_scale = values["position"][..., :4, :].diff(dim=-3).square().mean().sqrt().clamp_min(1)
    optimizer = torch.optim.AdamW(prefix.parameters(), lr=0.003, weight_decay=0.01)
    trace = []
    for step in range(UPDATES):
        rows = np_rng.choice(len(values["scene"]), size=2, replace=False).tolist()
        tick, phase = (step // 4) % 4, 1 + step % 4
        history, past, actions = root_controls(values, embedded, rows, tick, hold)
        predicted = predict_step(model, adapter, gate, prefix, history, past, actions, phase)
        features = values["targets"][rows, :, :, tick, :, phase - 1, :].flatten(0, 2)
        position = values["position"][rows, :, :, tick, :, phase - 1, :].flatten(0, 2)
        loss = paired_objective(predicted, features, position, w, b, xy_scale, latent_scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prefix.parameters(), 5.)
        optimizer.step()
        if (step + 1) % 40 == 0:
            trace.append([step + 1, loss.item()])
            print(trace[-1], flush=True)
    return trace


@torch.no_grad()
def calibrate(model, adapter, gate, hold, prefix, values, w, b):
    return calibrate_residuals(model, adapter, gate, hold, prefix, values, lambda observed, predicted: (observed - predicted) @ w)


@torch.no_grad()
def calibrate_residuals(model, adapter, gate, hold, prefix, values, residual):
    embedded = action_embeddings(model, values["actions"])
    residuals = []
    for tick in range(embedded.size(2)):
        history, past, commands = root_controls(values, embedded, slice(None), tick, hold)
        prediction = torch.stack([predict_step(model, adapter, gate, prefix, history, past, commands, j)
                                  for j in range(1, PHASES + 1)], dim=-2)
        observed = values["targets"][:, :, :, tick].flatten(0, 2)
        residuals.append(residual(observed, prediction).flatten(-2).flatten(0, 1).double())
    residuals = torch.cat(residuals)
    mean = residuals.mean(dim=0)
    centered = residuals - mean
    cov = centered.T @ centered / len(centered) + torch.eye(centered.size(-1), device=centered.device, dtype=centered.dtype)
    return mean, cov, len(residuals)


def chronological_evidence(predicted, arrived, drift, covariance):
    """Log likelihood factors for chronological phases using only present and earlier data."""
    residual = arrived.unsqueeze(-3).flatten(-2).double() - predicted.flatten(-2).double() - drift
    chol = torch.linalg.cholesky(covariance)
    factors = []
    width = predicted.size(-1)
    for end in range(width, residual.size(-1) + 1, width):
        innovations = torch.linalg.solve_triangular(chol[:end, :end], residual[..., :end, None], upper=False)
        factors.append(-0.5 * innovations[..., -width:, 0].square().sum(dim=-1))
    return torch.stack(factors, dim=-1)


def endpoint_evidence(predicted, arrived, drift, covariance):
    width = predicted.size(-1)
    residual = arrived[..., -1, :].unsqueeze(-2).double() - predicted[..., -1, :].double() - drift[-width:]
    precision = torch.linalg.inv(covariance[-width:, -width:])
    log_factor = -0.5 * ((residual @ precision) * residual).sum(dim=-1)
    return torch.cat((log_factor.new_zeros((*log_factor.shape, predicted.size(-2) - 1)), log_factor.unsqueeze(-1)), dim=-1)


def nll_pairs(left, right):
    left = left.sum(-1).diff(dim=-1).squeeze(-1)
    right = right.sum(-1).diff(dim=-1).squeeze(-1)
    center, gap = 0.5 * (left + right), 0.5 * (right - left)
    nll = 0.5 * (F.softplus(left) + F.softplus(-right))
    return center, gap, nll


@torch.no_grad()
def fresh(model, adapter, gate, hold, prefix, w, b, old_scale, drift, covariance):
    device = w.device
    with np.load(PAIRS / "fresh_switches.npz", allow_pickle=False) as archive:
        history = encode_rgb(model, archive["rgb"])
        fine = encode_rgb(model, archive["path_rgb"])
        commands = torch.as_tensor(archive["actions"], device=device).float()
        ids, modes, phases = [archive[k].tolist() for k in ("scene_id", "mode", "phase")]
        raw_gap = np.linalg.norm(archive["path_state"][:, :, 1, :, 2:4] - archive["path_state"][:, :, 0, :, 2:4], axis=-1)
    embedded = action_embeddings(model, commands)
    predictions = []
    for tick in range(len(phases)):
        past_z, past_a = block_windows(history, embedded, tick, hold)
        raw = commands[:, tick]
        out = [predict_step(model, adapter, gate, prefix, past_z, past_a, raw, j) for j in range(1, PHASES + 1)]
        native = rollout(model, adapter, gate, past_z, past_a, embedded[:, None, tick:tick + 1])[:, 0, :, 0]
        torch.testing.assert_close(out[-1], native, atol=1e-5, rtol=1e-5)
        for j in range(1, PHASES):
            changed = raw.clone()
            changed[:, j:] = -changed[:, j:] + 0.11
            torch.testing.assert_close(prefix(model, raw, j), prefix(model, changed, j), rtol=0, atol=0)
        predictions.append(torch.stack(out, dim=-2))
    predicted = torch.stack(predictions, dim=1) @ w + b
    measured = fine @ w + b
    # The paired archive records both arms. Recover the factual sensor stream
    # at fixture ingress before passing any measurements to the estimator.
    factual = torch.stack([measured[:, tick, mode] for tick, mode in enumerate(modes)], dim=1)
    active = torch.as_tensor(raw_gap.max(axis=-1) >= 1, device=device)
    active_end = torch.as_tensor(raw_gap[..., -1] >= 1, device=device)
    path_gap = (measured[:, :, 1] - measured[:, :, 0]).flatten(-2).double()
    all_d = ((path_gap @ torch.linalg.inv(covariance)) * path_gap).sum(-1)
    end_d = ((path_gap[..., -2:] @ torch.linalg.inv(covariance[-2:, -2:])) * path_gap[..., -2:]).sum(-1)
    torch.testing.assert_close((all_d - end_d).clamp_min(0), all_d - end_d)
    zero = torch.zeros_like(drift)
    old_cov = torch.block_diag(torch.eye(8, device=device).double(), old_scale.double().square().diag())
    arms = {
        "previous_endpoint": (endpoint_evidence, predicted, zero, old_cov),
        "matched_endpoint": (endpoint_evidence, predicted, drift, covariance),
        "native_path": (chronological_evidence, predicted, drift, covariance),
        "oracle_matched_endpoint": (endpoint_evidence, measured, zero, covariance),
        "oracle_path": (chronological_evidence, measured, zero, covariance),
    }
    result = {}
    for name, (fn, mu, offset, cov) in arms.items():
        left, right = (fn(mu, measured[:, :, k], offset, cov) for k in (0, 1))
        center, oriented, nll = nll_pairs(left, right)
        issued_factual = fn(mu, factual, offset, cov)
        torch.testing.assert_close(issued_factual, torch.stack(
            [(left if mode == 0 else right)[:, tick] for tick, mode in enumerate(modes)], dim=1,
        ))
        support = mu.new_full((len(ids), 2), -np.log(2)).double()
        phase_p, priors = [], []
        for tick in range(len(phases)):
            support = advance_log_support(support, SWITCH)
            priors.append(support.diff(dim=-1).flatten())
            current = []
            for phase in range(PHASES):
                support = (support + issued_factual[:, tick, :, phase]).log_softmax(-1)
                current.append(support.exp()[:, 1])
            phase_p.append(torch.stack(current, dim=-1))
        priors, phase_p = torch.stack(priors, dim=1), torch.stack(phase_p, dim=1)
        result[name] = {"active_nll": nll[active].mean().item(), "all_nll": nll.mean().item(),
                        "neutral_abs_evidence": center[~active].abs().mean().item(),
                        "mean_signed_strength_active": oriented[active].mean().item(),
                        "per_scene": {str(s): {"midpoint": center[i].tolist(), "oriented_half_gap": oriented[i].tolist(),
                                               "prior_log_odds": priors[i].tolist(),
                                               "posterior_braked": phase_p[i, :, -1].tolist(),
                                               "posterior_each_arrival": phase_p[i].tolist(), "paired_nll": nll[i].tolist()}
                                      for i, s in enumerate(ids)}}
    active_pair_gap = raw_gap.max(axis=-1).tolist()
    scope = {"scenes": ids, "phases": phases, "true_modes": modes,
             "physical_gap_by_scene": dict(zip(map(str, ids), active_pair_gap, strict=True)), "scores": result}
    oracle_overlap = [int((active & ~active_end).sum()), int(active.sum())]
    errors = ((predicted - measured).square().sum(dim=-1).mean(dim=(0, 1, 2))).sqrt()
    delta = measured[:, :, 1] - measured[:, :, 0]
    native_delta = predicted[:, :, 1] - predicted[:, :, 0]
    directed = (delta * native_delta).sum(dim=-1).mean(dim=(0, 1)) / delta.square().sum(dim=-1).mean(dim=(0, 1))
    return {"paired_sensor_rmse_by_raw_step": errors.tolist(), "signed_contrast_fraction_by_step": directed.tolist(),
            "physical_transient_only_and_total_active": oracle_overlap, "marginal_gap_mahalanobis_active": end_d[active].mean().item(),
            "conditional_early_addition_active": (all_d - end_d)[active].mean().item(), "scores": scope,
            "actual_reversal_attribution": decompose_switches(scope)}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model, adapter, gate, hold, w, b, old_scale = load_core(device)
    prefix = NativeActionPrefixes(hold.numel()).to(device)
    example = panel(model, "fit")
    print(f"Encoded raw control-step pairs on {len(example['scene'])} fitting scenes", flush=True)
    trace = fit(model, adapter, gate, hold, prefix, example, w, b)
    cov_data = panel(model, "calibration")
    drift, covariance, rows = calibrate(model, adapter, gate, hold, prefix, cov_data, w, b)
    evaluation = fresh(model, adapter, gate, hold, prefix, w, b, old_scale, drift, covariance)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump({"fit": example["scene"], "calibration": cov_data["scene"], "calibration_rows": rows,
                   "updates": UPDATES, "training_trace": trace, "drift": drift.tolist(), "covariance": covariance.tolist(),
                   **evaluation}, stream, indent=2, allow_nan=False)
    torch.save({"prefix": prefix.state_dict(), "drift": drift.cpu(), "covariance": covariance.cpu()}, OUTPUT / "prefix.pt")


if __name__ == "__main__":
    main()
