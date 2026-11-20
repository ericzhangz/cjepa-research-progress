"""Deterministic structural checks for fixed-k causal-prefix trajectory learning."""

from .prbd_paths import output_path

import argparse
import itertools
import json
import math
from pathlib import Path

import torch
from torch import nn

from .prbd import assimilate, issue, quadrature
from .prbd_native import NativePrototypes
from .prbd_public_fit import (trajectory_episode_loss, trajectory_log_predictive,
                              trajectory_paths, windows)


class ToyNative(nn.Module):
    """Nonlinear window-sensitive fixture using the actual native rollout loop."""

    rollout = NativePrototypes.rollout
    forward = NativePrototypes.forward

    def __init__(self):
        super().__init__()
        self.offsets = nn.Parameter(torch.tensor([[[-.35]], [[.8]]], dtype=torch.float64))

    def _step(self, window, controls):
        conditions = torch.cat((torch.zeros_like(self.offsets[:1]), self.offsets))[:, 0]
        last = window[:, :, -1]
        return .8 * last + .15 * last.square() + controls[:, :, -1] * conditions


def check(geometries=("original", "bounded")):
    torch.set_num_threads(2)
    nodes, masses = quadrature(order=8)
    noise = torch.tensor([[.7]], dtype=torch.float64)
    transition = torch.eye(3, dtype=torch.float64)
    z = torch.tensor([[.1], [.3], [-.2], [.6], [.15], [-.1], [.25]], dtype=torch.float64)
    a = torch.tensor([[.1], [.2], [.7], [-.4], [.5], [-.6]], dtype=torch.float64)
    rows = {}
    for geometry in geometries:
        model = ToyNative()
        responses = model(windows(z[:-1]), windows(a))
        posterior = transition.new_full((3,), 1 / 3)
        losses = []
        for response, observation in zip(responses, z[3:], strict=True):
            arrival = assimilate(issue(response, posterior, transition, noise), observation,
                                  nodes, masses, geometry=geometry)
            losses.append(-arrival.log_predictive)
            posterior = arrival.posterior
        causal = torch.stack(losses).mean()
        causal_gradient, = torch.autograd.grad(causal, model.offsets)
        same, counts = trajectory_episode_loss(model, z, a, noise, transition, nodes, masses,
                                               horizon=1, geometry=geometry)
        trajectory_gradient, = torch.autograd.grad(same, model.offsets)
        torch.testing.assert_close(causal, same, atol=1e-12, rtol=0)
        torch.testing.assert_close(causal_gradient, trajectory_gradient, atol=1e-12, rtol=0)
        assert counts["transitions"] == counts["scored_target_steps"] == 4

        paths = trajectory_paths(model, z, a, 4)
        assert [path.size(1) for path in paths] == [4, 3, 2, 1]
        manual = z[None, :3, None].transpose(1, 2).expand(1, 3, 3, 1)
        manual_steps = []
        for step in range(4):
            controls = a[None, None, step:step + 3].expand(1, 3, 3, 1)
            predicted = model._step(manual, controls)
            manual_steps.append(predicted[0])
            manual = torch.cat((manual[:, :, 1:], predicted[:, :, None]), dim=2)
        torch.testing.assert_close(paths[0], torch.stack(manual_steps, dim=1), atol=0, rtol=0)
        torch.testing.assert_close(paths[1][:, 0], responses[1], atol=0, rtol=0)
        teacher = model(z[None, 1:4], a[None, 1:4])[0]
        own_teacher_difference = float((paths[0][:, 1] - teacher).norm())
        assert own_teacher_difference > .01
        d_loss, d_counts = trajectory_episode_loss(model, z, a, noise, transition, nodes, masses,
                                                   horizon=4, geometry=geometry)
        assert d_counts == {"transitions": 4, "scored_target_steps": 10,
                            "root_horizon_counts": {"1": 1, "2": 1, "3": 1, "4": 1}}
        d_gradient, = torch.autograd.grad(d_loss, model.offsets)
        assert torch.isfinite(d_gradient).all() and (d_gradient != 0).all()

        weights = torch.tensor([.2, .3, .5], dtype=torch.float64)
        path = torch.tensor([[[-.6], [.2]], [[.3], [-.5]], [[.9], [.7]]], dtype=torch.float64)
        observations = torch.tensor([[.4], [-.1]], dtype=torch.float64)
        joint, first = trajectory_log_predictive(path, weights, observations, noise,
                                                 nodes, masses, geometry=geometry)
        changed = observations.clone()
        changed[1] += 100
        _, changed_first = trajectory_log_predictive(path, weights, changed, noise,
                                                     nodes, masses, geometry=geometry)
        assert torch.equal(first.posterior, changed_first.posterior)
        # Independently expand all k and both quadrature-node combinations.
        # Each component is a normalized diagonal 2D Gaussian; total mass is 1.
        center = torch.einsum("k,khd->hd", weights, path)[:, 0]
        deviations = (path[:, :, 0] - center) / noise[0, 0]
        energy = deviations.square().sum(0)
        extra = energy if geometry == "original" else energy / (1 + energy)
        terms, component_mass = [], []
        for k, q1, q2 in itertools.product(range(3), range(len(nodes)), range(len(nodes))):
            variance = noise[0, 0].square() * (1 + nodes[[q1, q2]] * extra)
            residual = observations[:, 0] - path[k, :, 0]
            normal = -.5 * (2 * math.log(2 * math.pi) + variance.log().sum()
                             + (residual.square() / variance).sum())
            mass = weights[k] * masses[q1] * masses[q2]
            terms.append(mass.log() + normal)
            component_mass.append(mass)
        oracle = torch.logsumexp(torch.stack(terms), 0)
        torch.testing.assert_close(joint, oracle, atol=1e-12, rtol=0)
        torch.testing.assert_close(torch.stack(component_mass).sum(), joint.new_tensor(1),
                                   atol=1e-14, rtol=0)
        wrong = sum(assimilate(issue(path[:, h], weights, transition, noise), observations[h],
                               nodes, masses, geometry=geometry).log_predictive for h in range(2))
        assert abs(float(joint - wrong)) > 1e-3

        # A future target must train evidence-producing historical responses.
        source = torch.tensor([[-.5], [.2], [.9]], dtype=torch.float64, requires_grad=True)
        source_arrival = assimilate(issue(source, weights, transition, noise),
                                     torch.tensor([.65], dtype=torch.float64), nodes, masses,
                                     geometry=geometry)
        future, _ = trajectory_log_predictive(path, source_arrival.posterior, observations,
                                              noise, nodes, masses, geometry=geometry)
        historical_gradient, = torch.autograd.grad(-future, source)
        assert torch.isfinite(historical_gradient).all() and historical_gradient.norm() > 1e-4
        equal = torch.full((3, 2, 1), .25, dtype=torch.float64, requires_grad=True)
        equal_joint, equal_first = trajectory_log_predictive(
            equal, weights, observations, noise, nodes, masses, geometry=geometry)
        equal_gradient, = torch.autograd.grad(-equal_joint, equal)
        assert torch.equal(equal_first.posterior, weights)
        assert torch.isfinite(equal_gradient).all()
        rows[geometry] = {"h1_nll_absolute_difference": float(abs(causal - same)),
                          "h1_gradient_max_absolute_difference": float((causal_gradient - trajectory_gradient).abs().max()),
                          "own_window_vs_teacher_forced_l2": own_teacher_difference,
                          "joint_vs_expanded_density_absolute_difference": float(abs(joint - oracle)),
                          "expanded_gaussian_component_mass": float(torch.stack(component_mass).sum()),
                          "joint_vs_independently_remixed_log_density_difference": float(joint - wrong),
                          "future_target_to_historical_response_gradient_l2": float(historical_gradient.norm()),
                          "future_target_changes_public_first_write": False,
                          "zero_response_rank_preserves_prior_and_finite_gradients": True,
                          "tail_counts": d_counts, "passed": True}
    return {"quadrature_order_for_expanded_fixture": len(nodes),
            "training_quadrature_unchanged": "32 nodes from the archived checkpoint",
            "checks": rows, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", choices=("original", "bounded", "both"), default="both")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[2] /
                        "runs/prbd_evidence_trajectory_20261119/p3_checks.json")
    args = parser.parse_args()
    result = check((args.geometry,) if args.geometry != "both" else ("original", "bounded"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
