"""G3 structural checks for fixed-reference, episode-consistent fitting."""

from .prbd_paths import output_path

import argparse
import json
from pathlib import Path

import torch

from .prbd import (_log_mass, assimilate_continuous, continuous_quadrature,
                   issue)
from .prbd_public_fit import (consistency_episode_loss, partition_paths,
                             partition_episode_log_predictive, windows)
from .prbd_trajectory_checks import ToyNative


def check():
    torch.set_num_threads(2)
    nodes, masses = continuous_quadrature()
    noise = torch.tensor([[.7]], dtype=torch.float64)
    transition = torch.eye(3, dtype=torch.float64)
    uniform = transition.new_full((3,), 1 / 3)
    z = torch.tensor([[.1], [.3], [-.2], [.6], [.15], [-.1], [.25],
                      [.4], [-.2], [.1]], dtype=torch.float64)
    a = torch.tensor([[.1], [.2], [.7], [-.4], [.5], [-.6], [.3],
                      [-.5], [.8]], dtype=torch.float64)
    model = ToyNative()

    posterior, predictive = uniform.clone(), []
    for response, target in zip(model(windows(z[:-1]), windows(a)), z[3:], strict=True):
        signed = issue(response, posterior, transition, noise, reference="uniform")
        arrival = assimilate_continuous(signed, target, nodes, masses)
        predictive.append(arrival.log_predictive)
        posterior = arrival.posterior
    sequential = -torch.stack(predictive).mean()
    sequential_grad, = torch.autograd.grad(sequential, model.offsets)
    episode, h1_counts = consistency_episode_loss(
        model, z, a, noise, transition, nodes, masses, dual=False)
    episode_grad, = torch.autograd.grad(episode, model.offsets)
    torch.testing.assert_close(sequential, episode, atol=2e-12, rtol=0)
    torch.testing.assert_close(sequential_grad, episode_grad, atol=2e-12, rtol=0)

    # Independent scalar-density expansion checks normalization and the one-k
    # placement. Every constituent Gaussian and every mixing mass is normalized.
    manual_scores, block_mixtures = [], []
    for root, path in partition_paths(model, z, a, 4):
        block_scores = []
        for step, response in enumerate(path.unbind(1), start=1):
            b = (response[:, 0] - response[:, 0].mean()) / noise[0, 0]
            energy = b.square().sum()
            variance = noise[0, 0].square() * (1 + nodes * energy / (1 + energy))
            residual = z[root + step, 0] - response[:, 0]
            normal = -.5 * (torch.log(2 * torch.pi * variance)[:, None]
                             + residual.square()[None] / variance[:, None])
            block_scores.append(torch.logsumexp(masses.log()[:, None] + normal, 0))
        total = torch.stack(block_scores).sum(0)
        manual_scores.extend(block_scores)
        block_mixtures.append(torch.logsumexp(_log_mass(uniform) + total, 0))
    oracle = torch.logsumexp(_log_mass(uniform) + torch.stack(manual_scores).sum(0), 0)
    h4_joint, h4_counts = partition_episode_log_predictive(
        model, z, a, noise, transition, nodes, masses, horizon=4)
    torch.testing.assert_close(h4_joint, oracle, atol=2e-12, rtol=0)
    remixed_blocks = torch.stack(block_mixtures).sum()
    assert abs(float(h4_joint - remixed_blocks)) > 1e-5
    torch.testing.assert_close(masses.sum(), masses.new_tensor(1.), atol=1e-15, rtol=0)

    dual, dual_counts = consistency_episode_loss(
        model, z, a, noise, transition, nodes, masses)
    torch.testing.assert_close(dual, (episode - h4_joint / (len(a) - 2)) / 2,
                               atol=2e-12, rtol=0)
    analytic, = torch.autograd.grad(dual, model.offsets)
    assert torch.isfinite(analytic).all() and (analytic != 0).all()
    finite_difference = torch.zeros_like(analytic)
    original = model.offsets.detach().clone()
    epsilon = 1e-5
    for index in range(original.numel()):
        values = []
        for sign in (-1, 1):
            with torch.no_grad():
                model.offsets.copy_(original)
                model.offsets.flatten()[index] += sign * epsilon
            score, _ = consistency_episode_loss(model, z, a, noise, transition, nodes, masses)
            values.append(float(score))
        finite_difference.flatten()[index] = (values[1] - values[0]) / (2 * epsilon)
    with torch.no_grad():
        model.offsets.copy_(original)
    torch.testing.assert_close(analytic, finite_difference, atol=2e-8, rtol=2e-6)

    # All terminal lengths, and both schemes, cover exactly the same targets.
    tail_counts = {}
    for length in range(3, 10):
        _, counts = consistency_episode_loss(
            model, z[:length + 1], a[:length], noise, transition, nodes, masses)
        expected = list(range(3, length + 1))
        for partition in counts["partitions"].values():
            assert partition["target_indices"] == expected
            assert len(set(partition["target_indices"])) == len(expected)
        assert counts["scored_target_steps"] == length - 2
        assert counts["density_factor_evaluations"] == 2 * (length - 2)
        assert counts["public_belief_writes_in_objective"] == 0
        tail_counts[str(length)] = counts["partitions"]["H4"]["block_lengths"]

    original_paths = partition_paths(model, z, a, 4)
    changed = z.clone()
    changed[3:] += 10
    changed_paths = partition_paths(model, changed, a, 4)
    assert torch.equal(original_paths[0][1], changed_paths[0][1])
    assert not torch.equal(original_paths[1][1], changed_paths[1][1])
    terminal = z.clone()
    terminal[-1] += 100
    assert all(torch.equal(left[1], right[1]) for left, right in zip(
        original_paths, partition_paths(model, terminal, a, 4), strict=True))
    paths = original_paths[0][1]
    teacher = model(z[None, 1:4], a[None, 1:4])[0]
    free_vs_teacher = float((paths[:, 1] - teacher).norm())
    assert free_vs_teacher > .01

    # Imagined or terminal targets never write the causal public first update.
    source = model(z[None, :3], a[None, :3])[0]
    signed = issue(source, uniform, transition, noise, reference="uniform")
    first = assimilate_continuous(signed, z[3], nodes, masses)
    consistency_episode_loss(model, terminal, a, noise, transition, nodes, masses)
    replay_first = assimilate_continuous(signed, z[3], nodes, masses)
    assert torch.equal(first.posterior, replay_first.posterior)

    # A later target changes the gradient to an earlier response through the
    # shared episode prototype; no posterior detach or independent remix occurs.
    response_history = torch.tensor([[[-.5], [.2], [.9]], [[-.6], [.1], [.8]]],
                                    dtype=torch.float64, requires_grad=True)
    gradients = []
    for later in (-.4, .7):
        factors = []
        for response, target in zip(response_history, (.65, later), strict=True):
            issued = issue(response, uniform, transition, noise, reference="uniform")
            factors.append(assimilate_continuous(
                issued, torch.tensor([target], dtype=torch.float64), nodes, masses).log_likelihood)
        score = -torch.logsumexp(_log_mass(uniform) + torch.stack(factors).sum(0), 0)
        gradient, = torch.autograd.grad(score, response_history)
        gradients.append(gradient[0])
    historical_gradient_change = float((gradients[0] - gradients[1]).norm())
    assert historical_gradient_change > 1e-4
    return {"passed": True, "continuous_integration_nodes": len(nodes),
            "h1_episode_vs_sequential_nll_abs_error": float(abs(sequential - episode)),
            "h1_episode_vs_sequential_gradient_max_error": float((sequential_grad - episode_grad).abs().max()),
            "h4_joint_vs_independent_scalar_density_abs_error": float(abs(h4_joint - oracle)),
            "h4_joint_minus_per_block_remix_log_density": float(h4_joint - remixed_blocks),
            "dual_gradient_max_finite_difference_error": float((analytic - finite_difference).abs().max()),
            "future_target_changes_historical_response_gradient_l2": historical_gradient_change,
            "free_vs_teacher_forced_path_l2": free_vs_teacher,
            "terminal_block_lengths_by_actual_T": tail_counts,
            "every_target_exactly_once_per_partition": True,
            "future_targets_leave_issued_block_paths_unchanged": True,
            "later_block_refresh_uses_actual_root": True,
            "future_targets_leave_public_first_write_unchanged": True,
            "h1_counts": h1_counts, "h4_counts": h4_counts,
            "dual_counts": dual_counts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] /
                        "runs/prbd_consistency_20261120/g3_checks.json")
    args = parser.parse_args()
    result = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
