"""Measure the paired assignment tangent of two tied native response interpretations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import native_response_mixture_regulation as data
from .native_response_observation import _log_density
from .native_response_relations_intermediates import HOLD_CANDIDATE, QUERY_STEP, SOURCE_STEPS, _source


RUN = data.RUNS / "native_response_relations_20261114/action_effects_20261115"
TWIN_MODES = slice(1, 3)
TANGENT_STEP = 0.01
WITNESS_STEPS = (0.05, 0.025)


def _paired_inputs(model, panel):
    rows = slice(None)
    layout = panel.visible.shape[:3]
    visible, commands = data._factual_inputs(panel, rows)
    embedded = model.action_encoder(commands).double()
    sources = []
    for step in SOURCE_STEPS:
        end = step // 5
        window, controls, target = _source(visible, embedded, step)
        if not panel.valid[..., end - 1].all() or not torch.equal(target, panel.full[..., end, :].flatten(0, 2)):
            raise ValueError(f"The paired tape requires the executed public full-frame arrival at {step}")
        sources.append((window.double(), controls, target.double()))

    windows, past, candidates = data._query_inputs(panel, rows, (QUERY_STEP,))
    for value in (windows, past, candidates):
        negative, positive = value.unflatten(0, layout).unbind(2)
        if not torch.equal(negative, positive):
            raise ValueError("Each mirror pair must expose the same query inputs")
    mean, std = data._action_moments(commands)
    hold = ((commands.new_zeros(5, 2) - mean) / std).flatten()
    if not torch.equal(candidates[:, HOLD_CANDIDATE], hold.expand_as(candidates[:, HOLD_CANDIDATE])):
        raise ValueError("Expected the actual zero-command reference at candidate 3")
    if not torch.equal(candidates[:, :, 0], candidates[:, :, 1]):
        raise ValueError("Expected the recorded H2 to repeat the H1 command")
    query = (
        windows[::2].double(),
        model.action_encoder(past[::2]).double(),
        model.action_encoder(candidates[::2].flatten(1, 2)).unflatten(1, candidates.shape[1:3]).double(),
    )
    _, truth, valid = data._targets(panel, rows, (QUERY_STEP,))
    if not valid.all():
        raise ValueError("This panel requires every paired action and reference endpoint to be executed")
    truth = truth.squeeze(-4).double()
    truth_hold = truth[..., HOLD_CANDIDATE, :, :]
    truth_d = truth[..., :HOLD_CANDIDATE, :, :] - truth_hold.unsqueeze(-3)
    return sources, query, truth_hold.flatten(0, 1), truth_d.flatten(0, 1)


def _evaluate(model, inputs, sigma, coordinates, xy_variance, center, offset):
    sources, query, _, truth_d = inputs
    op = model.response_mixture
    op.relation_raw_strengths[:2].copy_(torch.stack((center + offset, center - offset)))
    batch = len(sources[0][0])
    state = op.prepare_state(op.initial_state(batch), batch)
    likelihoods = []
    for window, controls, target in sources:
        _, advanced, issued = model.predict_response(window, controls, state, factual=True)
        torch.testing.assert_close(advanced, state, atol=1e-14, rtol=0)
        likelihood = _log_density(target, issued.means, sigma)
        likelihoods.append(likelihood)
        _, state, _ = op.condition(issued, likelihood)
    log_likelihood = torch.stack(likelihoods, dim=1)[..., TWIN_MODES].unflatten(0, (-1, 2))
    log_p = state[:, TWIN_MODES].log_softmax(-1).unflatten(0, (-1, 2))
    expected = log_likelihood.sum(-2).log_softmax(-1)
    torch.testing.assert_close(log_p, expected, atol=1e-12, rtol=0)

    w, bias = coordinates
    xy = model.rollout_response_modes(*query)[..., TWIN_MODES, :] @ w + bias
    hold = xy[:, HOLD_CANDIDATE]
    effect = xy[:, :HOLD_CANDIDATE] - hold[:, None]
    error = effect[:, None] - truth_d[..., None, :]
    energy = (error.square() / xy_variance[:, None, :]).mean(dim=(2, 3, 5))
    return log_likelihood, log_p, hold, effect, energy


def _tangent(model, inputs, sigma, coordinates, xy_variance, center, step):
    source, response = [], []
    for offset in torch.eye(center.numel(), device=center.device, dtype=center.dtype) * step:
        log_l, _, _, _, energy = _evaluate(model, inputs, sigma, coordinates, xy_variance, center, offset)
        source.append((log_l[..., 0] - log_l[..., 1]) / (2 * step))
        response.append((energy[..., 1] - energy[..., 0]) / (2 * step))
    a_steps, b = (torch.stack(values, dim=-1).diff(dim=1).squeeze(1) for values in (source, response))
    a = a_steps.sum(1)
    cross = a.T @ b / len(a)
    return a, b, (cross + cross.T) / 2, a_steps


def _rms_xy(value):
    return value.square().sum(-1).mean(dim=tuple(range(value.ndim - 2))).sqrt().tolist()


def _witness(model, inputs, sigma, coordinates, xy_variance, center, direction, step, a, b):
    _, _, truth_hold, truth_d = inputs
    _, log_p, hold, effect, energy = _evaluate(model, inputs, sigma, coordinates, xy_variance, center, step * direction)
    # Exact 2x2 likelihood-ratio computation via the determinant, avoiding the
    # cancellation of four nearly equal log mixture scores at small offsets.
    u_negative, u_positive = (log_p[..., 0] - log_p[..., 1]).unbind(1)
    v_negative, v_positive = (energy[..., 1] - energy[..., 0]).unbind(1)
    product_scale = (u_negative + v_negative - F.softplus(u_positive + v_negative)
                     - F.softplus(u_negative + v_positive)).exp()
    ratio = torch.log1p(product_scale * (u_positive - u_negative).expm1() * (v_positive - v_negative).expm1())
    log_q = (log_p[:, :, None, :] - energy[:, None, :, :]).logsumexp(-1)
    direct_ratio = log_q[:, 1, 1] + log_q[:, 0, 0] - log_q[:, 1, 0] - log_q[:, 0, 1]
    torch.testing.assert_close(ratio, direct_ratio, atol=1e-12, rtol=0)
    coefficient = (a @ direction) * (b @ direction)

    p = log_p.exp()
    delta_p = p[:, 1] - p[:, 0]
    paired_d = torch.einsum("pm,pchmx->pchx", delta_p, effect)
    paired_hold = torch.einsum("pm,phmx->phx", delta_p, hold)
    true_d = truth_d[:, 1] - truth_d[:, 0]
    signed_dot = (paired_d * true_d).sum(-1).mean((0, 1))
    common_fit_difference = (2 * signed_dot - paired_d.square().sum(-1).mean((0, 1))) / 4
    return {
        "raw_strength_step": step,
        "mean_assignment_log_odds": ratio.mean().item(),
        "mean_assignment_log_odds_divided_by_step_squared": (ratio / step**2).mean().item(),
        "assignment_coefficient_rms_remainder": ((ratio / step**2 - coefficient).square().mean().sqrt().item()),
        "mean_twin_support_tv_across_histories": (delta_p.abs().sum(-1).mean() / 2).item(),
        "true_paired_action_effect_rms_h1_h2": _rms_xy(true_d),
        "predicted_paired_action_effect_rms_h1_h2": _rms_xy(paired_d),
        "paired_action_effect_mean_signed_dot_h1_h2": signed_dot.tolist(),
        "paired_action_mse_improvement_vs_mean_support_h1_h2": common_fit_difference.tolist(),
        "true_paired_hold_rms_h1_h2": _rms_xy(truth_hold[:, 1] - truth_hold[:, 0]),
        "predicted_paired_hold_rms_h1_h2": _rms_xy(paired_hold),
    }


@torch.no_grad()
def run(device):
    train = data.load_data("train", data.ARCHIVE_DIR, device)
    development = data.load_data("development", data.ARCHIVE_DIR, device)
    if np.intersect1d(train.scene_ids, development.scene_ids).size:
        raise ValueError("The fit and historical development scenes must be disjoint")
    _, xy_variance = data._reanchored_scales(train, (QUERY_STEP,))
    model = data._native_model(device)
    op = model.enable_response_mixture(parameterization="relational", num_modes=4)
    checkpoint = torch.load(RUN / "relational/model.pt", map_location=device, weights_only=True)
    model.predictor.transformer.layers[-1].load_state_dict(checkpoint["last_block"])
    model.pred_proj.load_state_dict(checkpoint["pred_proj"])
    op.load_state_dict(checkpoint["response_mixture"])
    model.double()
    # The frozen LeWM action encoder explicitly casts its inputs to float.
    model.action_encoder.float()
    sigma = checkpoint["full_sigma"].double()
    coordinates = tuple(value.double().to(device) for value in checkpoint["object_coordinates"])
    center = op.relation_raw_strengths[:2].mean(0)
    xy_variance = xy_variance.double()
    train_inputs = _paired_inputs(model, train)
    development_inputs = _paired_inputs(model, development)
    exact = _evaluate(model, train_inputs, sigma, coordinates, xy_variance, center, torch.zeros_like(center))
    torch.testing.assert_close(exact[0][..., 0], exact[0][..., 1], atol=1e-12, rtol=0)
    torch.testing.assert_close(exact[3][..., 0, :], exact[3][..., 1, :], atol=1e-12, rtol=0)

    fit_tangent = _tangent(model, train_inputs, sigma, coordinates, xy_variance, center, TANGENT_STEP)
    fit_c = fit_tangent[2]
    eigenvalues, eigenvectors = torch.linalg.eigh(fit_c)
    direction = eigenvectors[:, -1]
    direction = direction * direction[direction.abs().argmax()].sign()
    results = {
        "source_steps": list(SOURCE_STEPS),
        "query_step": QUERY_STEP,
        "hold_candidate": HOLD_CANDIDATE,
        "twin_mode_ids": [1, 2],
        "tied_raw_strengths": center.tolist(),
        "predictor_dtype": "float64; unchanged frozen action embeddings computed in float32",
        "tangent_central_difference_raw_step": TANGENT_STEP,
        "xy_variance_h1_h2": xy_variance.tolist(),
        "fit_eigenvalues": eigenvalues.tolist(),
        "fit_direction": direction.tolist(),
    }
    for label, inputs, (a, b, c, a_steps) in (
        ("train", train_inputs, fit_tangent),
        ("development", development_inputs,
         _tangent(model, development_inputs, sigma, coordinates, xy_variance, center, TANGENT_STEP)),
    ):
        results[label] = {
            "mirror_pairs": len(a),
            "mean_symmetrized_tangent_product": c.tolist(),
            "mean_assignment_tangent_quadratic_on_fit_direction": (direction @ c @ direction).item(),
            "rms_source_tangent_on_fit_direction": (a @ direction).square().mean().sqrt().item(),
            "rms_source_tangent_by_arrival_on_fit_direction": (a_steps @ direction).square().mean(0).sqrt().tolist(),
            "mean_assignment_coefficient_by_arrival_on_fit_direction": (
                (a_steps @ direction) * (b @ direction)[:, None]
            ).mean(0).tolist(),
            "rms_query_tangent_on_fit_direction": (b @ direction).square().mean().sqrt().item(),
            "finite_witness": [
                _witness(model, inputs, sigma, coordinates, xy_variance, center, direction, step, a, b)
                for step in WITNESS_STEPS
            ],
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda"))
    parser.add_argument("--output", type=Path, default=RUN / "pair_split/tangent.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = run(args.device)
    encoded = json.dumps(results, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
