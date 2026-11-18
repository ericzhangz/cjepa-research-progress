"""Full-window latent forecasts and causal paired learning for §49.8."""

from __future__ import annotations

import torch

from .stage_c_native_check import native_conditioned


def two_offsets(common, antisymmetric):
    return torch.stack((common - antisymmetric, common + antisymmetric), dim=0)


def conditional_step(host, windows, actions, offsets):
    """The penultimate leading dimension of windows indexes the explanation."""
    count = windows.shape[:-3]
    modes, tokens, dimension = windows.shape[-3:]
    conditioned = offsets.expand(*count, *offsets.shape)
    conditioned = conditioned.reshape(-1, offsets.size(1), offsets.size(2)).transpose(0, 1)
    forecast = native_conditioned(
        host, windows.reshape(-1, tokens, dimension),
        actions.reshape(-1, tokens, actions.size(-1)), conditioned[:, :, None],
    )
    return forecast[:, -1].reshape(*count, modes, dimension)


def source_evidence(issued, arrival, scale_squared):
    minus, plus = issued.unbind(dim=-2)
    direction = plus - minus
    offset = arrival - (plus + minus) * .5
    return (direction * offset).sum(dim=-1) / scale_squared


def replace_support(support, evidence):
    mass = torch.stack(((-evidence).relu(), evidence.relu()), dim=-1)
    return (support + mass) / (1 + evidence.abs().unsqueeze(-1))


def factual_support(host, latents, actions, offsets, scale_squared):
    """Support at index t reads only the observations up to and including time t."""
    windows = latents[..., :-1, :].unfold(-2, 3, 1).transpose(-1, -2)
    controls = actions.unfold(-2, 3, 1).transpose(-1, -2)
    pending = conditional_step(host, windows.unsqueeze(-3).expand(*windows.shape[:-2], 2, *windows.shape[-2:]),
                               controls.unsqueeze(-3).expand(*controls.shape[:-2], 2, *controls.shape[-2:]), offsets)
    evidence = source_evidence(pending, latents[..., 3:, :], scale_squared)
    initial = evidence.new_full((*evidence.shape[:-1], 2), .5)
    support = initial
    timeline = [initial, initial, initial]
    for e in evidence.unbind(-1):
        support = replace_support(support, e)
        timeline.append(support)
    return torch.stack(timeline, dim=-2), evidence, pending


def menu_forecast(host, window, controls, offsets):
    """Run two whole native trajectories from a common actual root."""
    windows = window.unsqueeze(-3).unsqueeze(-3)
    windows = windows.expand(*controls.shape[:-2], 2, *window.shape[-2:])
    forecast = []
    for raw_window in controls.unfold(-2, 3, 1).transpose(-1, -2).unbind(-3):
        actions = raw_window.unsqueeze(-3).expand(*raw_window.shape[:-2], 2, *raw_window.shape[-2:])
        next_z = conditional_step(host, windows, actions, offsets)
        forecast.append(next_z)
        windows = torch.cat((windows[..., 1:, :], next_z.unsqueeze(-2)), dim=-2)
    return torch.stack(forecast, dim=-2)


def paired_energy(predicted, truth, support, directed, scale_squared, source_cost=None):
    """Select one assignment for all bridge roots and candidates in one group."""
    # predicted [group, bridge, candidate, explanation, horizon, latent].
    # truth [group, bridge, realized history, candidate, horizon, latent].
    predicted = predicted.transpose(-4, -3).unsqueeze(2)
    truth = truth.unsqueeze(3)
    error = (predicted - truth).square().sum(dim=-1).mean(dim=(-1, -2)) / (2 * scale_squared)
    penalties = -support.clamp_min(torch.finfo(support.dtype).tiny).log()
    costs = error + directed[:, None, None, None] * penalties
    if source_cost is not None:
        # One actual H1 source and one menu per endpoint, before the same min.
        if source_cost.shape != costs.shape:
            raise ValueError('H1 costs must pair each menu root, history, and explanation')
        costs = .5 * (costs + source_cost)
    costs = costs.mean(dim=1)
    unchanged = .5 * (costs[:, 0, 0] + costs[:, 1, 1])
    swapped = .5 * (costs[:, 0, 1] + costs[:, 1, 0])
    return torch.where(unchanged <= swapped, unchanged, swapped)
