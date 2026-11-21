"""Unlabelled sequential PRBD on the existing null-free native operator.

The fixed-k H1..4 suffix law approximates switches inside the prediction window
by a constant condition. Actual time always transports, then absorbs one actual
arrival. Uniform response geometry is independent of belief. No imagined
arrival changes public state. The native model itself is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .prbd import continuous_quadrature
from .prbd_native import NativePrototypes
from .prbd_paths import PHASE2 as OUT
from .prbd_tworoom_speed_fit import load_host


BUDGET = {"updates": 2048, "batch_episodes": 4, "learning_rate": .0003,
          "decay_updates": [1024, 1536], "seed": 20261121, "horizon": 4}
MIN_DEV_COEFFICIENT = .001


def fixed_transition(device="cpu"):
    """Design hazard 2 switches / 30 filterable steps; no cut-point access."""
    h = 2 / 30
    return (torch.eye(4, dtype=torch.float64, device=device) * (1 - h)
            + (1 - torch.eye(4, dtype=torch.float64, device=device)) * (h / 3))


class _BatchedBoundedDensity(torch.autograd.Function):
    """Batched G1 spectral values and analytic first derivatives.

    This is the same density/score as prbd._ContinuousBoundedDensity. Contracting
    over 185 quadrature nodes before forming d-by-k gradients avoids a large
    [batch,node,branch,d,k] tensor. No spectral derivative, rank cut or jitter.
    """

    @staticmethod
    def forward(ctx, basis, x, scales, masses):
        n, d, k = basis.shape
        # A tall 192x4 CUDA SVD dispatches one solver per item. QR followed by
        # the exact 4x4 SVD uses the batched small-matrix solver, with no rank
        # decision: B=Q R=(Q U_r) S V^T, including repeated and zero responses.
        orthonormal, triangular = torch.linalg.qr(basis, mode="reduced")
        small_vectors, singular, right = torch.linalg.svd(triangular, full_matrices=False)
        vectors = orthonormal @ small_vectors
        raw = singular.square()
        eigenvalues = raw / (1 + raw)
        variances = 1 + scales[None, :, None] * eigenvalues[:, None]
        projected_x = (vectors.transpose(1, 2) @ x[:, :, None])[:, :, 0]
        projected_basis = vectors.transpose(1, 2) @ basis
        coordinates = projected_x[:, :, None] - projected_basis
        orthogonal = x - (vectors @ projected_x[:, :, None])[:, :, 0]
        node_logs = -.5 * (variances.log().sum(-1)[:, :, None]
            + (coordinates[:, None].square() / variances[:, :, :, None]).sum(2))
        log_mass = masses.log()[None, :, None]
        log_joint = log_mass + node_logs
        reduced = torch.logsumexp(log_joint, dim=1)
        responsibility = (log_joint - reduced[:, None]).exp()
        absolute = reduced - .5 * (d * math.log(2 * math.pi)
                                  + orthogonal.square().sum(-1))[:, None]
        node_scores = -.5 * variances.log().sum(-1)[:, :, None] + .5 * (
            (scales[None, :, None] * eigenvalues[:, None] / variances)[:, :, :, None]
            * projected_x[:, None, :, None].square()
            + (2 * projected_x[:, :, None] * projected_basis
               - projected_basis.square())[:, None] / variances[:, :, :, None]).sum(2)
        scores = torch.logsumexp(log_mass + node_scores, dim=1)
        relative = scores - scores[:, :1]
        if any(ctx.needs_input_grad[:2]):
            projected = coordinates[:, None] / variances[:, :, :, None]
            # All quadrature-dependent contractions stay in the four-column
            # response coordinates. Ambient D only reappears after summing Q.
            # This is algebraically the same analytic Gaussian covariance score.
            u_projected = projected / (1 + raw)[:, None, :, None]
            m_coordinates = (singular / (1 + raw))[:, :, None] * right
            residual_m = projected.transpose(2, 3) @ m_coordinates[:, None]
            u_inverse_m_coordinates = ((singular / (1 + raw).square())[:, None, :, None]
                                      * right[:, None] / variances[:, :, :, None])
            weight = responsibility * scales[None, :, None]
            coordinate_score = torch.einsum("nqj,nqrj,nqjl->njrl", weight, u_projected, residual_m)
            coordinate_score -= torch.einsum("nqj,nqrl->njrl", weight, u_inverse_m_coordinates)
            score_basis = vectors[:, None] @ coordinate_score
            score_basis += orthogonal[:, None, :, None] * torch.einsum(
                "nqj,nqjl->njl", weight, residual_m)[:, :, None]
            mean_coordinates = (responsibility[:, :, None] * projected).sum(1)
            mean_score = (orthogonal[:, :, None] * responsibility.sum(1)[:, None]
                          + vectors @ mean_coordinates).transpose(1, 2)
            score_basis += mean_score[:, :, :, None] * torch.eye(
                k, dtype=basis.dtype, device=basis.device)[None, :, None]
            ctx.save_for_backward(score_basis, -mean_score)
        return absolute, relative

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, absolute_gradient, relative_gradient):
        score_basis, score_x = ctx.saved_tensors
        coefficients = absolute_gradient + relative_gradient
        coefficients = coefficients.clone()
        coefficients[:, 0] -= relative_gradient.sum(-1)
        return ((coefficients[:, :, None, None] * score_basis).sum(1),
                (coefficients[:, :, None] * score_x).sum(1), None, None)


def response_density(responses, observations, noise_cholesky, *, scales=None, masses=None,
                     chunk_size=64):
    """Return absolute/relative G1 logs and separation for [...,K,D] means.

    observations is [...,D], L is [D,D]. Everything probabilistic is FP64;
    gradients return to FP32 native predictions. Separation is max pairwise
    whitened Euclidean distance and never reads the observed target.
    """
    if responses.shape[:-2] != observations.shape[:-1] or responses.size(-1) != observations.size(-1):
        raise ValueError("Expected aligned responses [...,K,D] and observations [...,D]")
    shape, k, d = responses.shape[:-2], *responses.shape[-2:]
    r = responses.double().reshape(-1, k, d)
    y = observations.double().reshape(-1, d)
    lower = noise_cholesky.to(device=r.device, dtype=torch.float64)
    if scales is None:
        scales, masses = continuous_quadrature()
    scales, masses = scales.to(r.device), masses.to(r.device)
    difference = r - r[:, :1]
    center_difference = difference.mean(1)
    center = r[:, 0] + center_difference
    basis = torch.linalg.solve_triangular(lower, (difference - center_difference[:, None]).transpose(1, 2), upper=False)
    x = torch.linalg.solve_triangular(lower, (y - center)[:, :, None], upper=False)[:, :, 0]
    outputs = [_BatchedBoundedDensity.apply(basis[i:i + chunk_size], x[i:i + chunk_size], scales, masses)
               for i in range(0, len(r), chunk_size)]
    absolute, relative = (torch.cat([item[j] for item in outputs]) for j in (0, 1))
    absolute = absolute - lower.diagonal().log().sum()
    separation = (basis[:, :, :, None] - basis[:, :, None, :]).square().sum(1).amax((1, 2)).sqrt()
    return absolute.reshape(*shape, k), relative.reshape(*shape, k), separation.reshape(shape)


@dataclass
class StepResult:
    prior: torch.Tensor
    posterior: torch.Tensor
    log_likelihood: torch.Tensor
    relative_log_likelihood: torch.Tensor
    log_predictive: torch.Tensor
    separation: torch.Tensor
    silent: torch.Tensor


def assimilate_step(responses, weights, observation, noise_cholesky, *, transition=None,
                    deadzone=0., scales=None, masses=None):
    """Transport first, absorb actual arrival second; supports batched means.

    Deadzone skips only observation writing, returning the transported w^-.
    It does not alter response means, the density or predictive likelihood.
    Set deadzone=0 for the frozen-response diagnostic without the numeric rule.
    """
    if transition is None:
        transition = fixed_transition(responses.device)
    prior = weights.double() @ transition.to(responses.device)
    prior = prior / prior.sum(-1, keepdim=True)
    absolute, relative, separation = response_density(
        responses, observation, noise_cholesky, scales=scales, masses=masses)
    posterior = (prior.log() + relative).softmax(-1)
    silent = separation <= deadzone
    posterior = torch.where(silent[..., None], prior, posterior)
    return StepResult(prior, posterior, absolute, relative,
        torch.logsumexp(prior.log() + absolute, -1), separation, silent)


def load_episodes(path, device="cpu"):
    """Only four whitelisted actual-input tensors cross the training boundary."""
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if not {"z", "a", "seed", "lengths"} <= batch.keys():
        raise ValueError("Expected z,a,seed,lengths actual input tensors")
    z, a, lengths, seeds = (batch[key] for key in ("z", "a", "lengths", "seed"))
    if z.ndim != 3 or a.ndim != 3 or z.shape != (len(a), a.size(1) + 1, 192) or a.size(-1) != 192:
        raise ValueError("Expected z[N,T+1,192] and a[N,T,192]")
    if len(seeds) != len(z) or len(lengths) != len(z) or len(set(seeds.tolist())) != len(seeds):
        raise ValueError("Invalid episode IDs or lengths")
    episodes = []
    for seed, states, actions, length in zip(seeds, z, a, lengths, strict=True):
        size = int(length)
        if size < 3:
            continue
        states, actions = states[:size + 1].to(device).clone(), actions[:size].to(device).clone()
        if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
            raise ValueError("Nonfinite actual training input")
        episodes.append((int(seed), states, actions))
    return episodes


def episode_paths(model, episodes):
    """Batch every legal actual root, grouping only by remaining horizon."""
    slots, windows, controls, targets = [], {}, {}, {}
    for episode, (_, z, a) in enumerate(episodes):
        for root in range(2, len(a)):
            horizon = min(4, len(a) - root)
            slots.append((episode, root, horizon))
            windows.setdefault(horizon, []).append(z[root - 2:root + 1])
            controls.setdefault(horizon, []).append(a[root - 2:root + horizon])
            targets.setdefault(horizon, []).append(z[root + 1:root + horizon + 1])
    results, counters = {}, {h: 0 for h in windows}
    paths = {h: model.rollout(torch.stack(windows[h]), torch.stack(controls[h])) for h in windows}
    for e, root, h in slots:
        index = counters[h]
        results[e, root] = (paths[h][index], targets[h][index])
        counters[h] += 1
    return results


def batch_objective(model, episodes, noise, *, transition=None, deadzone=0., min_dev=MIN_DEV_COEFFICIENT,
                    return_trace=False):
    """Equal episodes; equal roots; equal legal H1..4 prefix scores / (H*D).

    Every root's single k owns its whole imagined suffix. The weights are
    transported for the first action and carry the full actual-prefix gradient.
    Future targets score an issued suffix but never update the public belief.
    Min-dev is 1e-3 mean_root sum_{k>0,d}(f_k-f0)^2 / D, raw latent units.
    """
    paths = episode_paths(model, episodes)
    if transition is None:
        transition = fixed_transition(noise.device)
    keys = list(paths)
    all_means = torch.cat([paths[key][0].transpose(0, 1) for key in keys])
    all_targets = torch.cat([paths[key][1] for key in keys])
    absolute, relative, separation = response_density(all_means, all_targets, noise)
    density, cursor = {}, 0
    for key in keys:
        h = paths[key][0].size(1)
        density[key] = (absolute[cursor:cursor + h], relative[cursor:cursor + h], separation[cursor:cursor + h])
        cursor += h
    nlls, penalties, traces = [], [], []
    for e, (_, z, a) in enumerate(episodes):
        belief = noise.new_full((4,), .25)
        root_nll, root_penalty, trace = [], [], []
        for root in range(2, len(a)):
            path, _ = paths[e, root]
            logs, rel, sep = density[e, root]
            prior = belief @ transition
            prior = prior / prior.sum()
            joint = logs.cumsum(0) + prior.log()[None]
            nll = -torch.logsumexp(joint, -1) / (torch.arange(1, len(logs) + 1, device=logs.device) * z.size(-1))
            root_nll.append(nll.mean())
            root_penalty.append((path[1:, 0] - path[:1, 0]).square().sum() / z.size(-1))
            arrival_relative = rel[0]
            updated = (prior.log() + arrival_relative).softmax(0)
            silent = sep[0] <= deadzone
            belief = torch.where(silent, prior, updated)
            if return_trace:
                trace.append({"root": root, "prior": prior, "posterior": belief,
                              "responses": path[:, 0], "separation": sep[0], "silent": silent,
                              "horizon_nll": nll, "relative_log_likelihood": arrival_relative,
                              "arrival_relative": arrival_relative})
        nlls.append(torch.stack(root_nll).mean())
        penalties.append(torch.stack(root_penalty).mean())
        traces.append(trace)
    nll, penalty = torch.stack(nlls).mean(), torch.stack(penalties).mean()
    result = {"nll_per_dimension": nll, "min_deviation_per_dimension": penalty,
              "weighted_min_deviation": min_dev * penalty,
              "read_roots": len(paths), "suffix_target_steps": len(all_targets)}
    if return_trace:
        result["trace"] = traces
    return nll + min_dev * penalty, result


def load_phase2_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = NativePrototypes(load_host(device), k=4, null_control=None).eval()
    with torch.no_grad():
        model.offsets.copy_(checkpoint["offsets"].to(model.offsets))
    if model.offsets.numel() != 3456 or model.null_control is not None:
        raise ValueError("Phase2 requires original null-free 3456-parameter operator")
    calibration = {"noise_cholesky": checkpoint["noise_cholesky"].to(device),
                   "transition": checkpoint["transition"].to(device),
                   "deadzone": float(checkpoint["deadzone"])}
    return model, calibration


def filter_episode(model, z, a, noise_cholesky, *, transition=None, deadzone=0.,
                   initial_belief=None):
    """Deployment replay returning pre-action priors and actual posteriors.

    Entry t has only z[:t+1],a[:t+1]. Caller reads its prior with a same-k
    rollout for a future candidate. No truth, speed, cut point or null input.
    """
    belief = noise_cholesky.new_full((4,), .25) if initial_belief is None else initial_belief
    responses = model(z[:-1].unfold(0, 3, 1).transpose(1, 2).contiguous(),
                      a.unfold(0, 3, 1).transpose(1, 2).contiguous())
    result = []
    for t, means in enumerate(responses, start=2):
        step = assimilate_step(means, belief, z[t + 1], noise_cholesky,
                               transition=transition, deadzone=deadzone)
        result.append({"root": t, "responses": means, "step": step})
        belief = step.posterior
    return result
