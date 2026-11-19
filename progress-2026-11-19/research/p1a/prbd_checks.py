"""Deterministic, independent numerical checks for the PRBD finite realization.

These are mathematical synthetic cases, not learned capability or task results.
The continuous-reference residuals and the quadrature order are fixed below;
their measured discrepancy is reported rather than used to select new nodes.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.integrate import quad, quad_vec
from scipy.special import gammaln, logsumexp

from research.p1a.prbd import assimilate, issue, quadrature


DTYPE = torch.float64


def _tensor(values, *, requires_grad=False):
    return torch.tensor(values, dtype=DTYPE, requires_grad=requires_grad)


def _values(value):
    return value.detach().cpu().tolist()


def _full_covariance_log_likelihood(responses, prior, noise, observation, scales, masses):
    """Direct ambient-space Gaussian densities, independent of the Gram formula."""
    means = responses.detach()
    mean = prior.detach() @ means
    deviations = means - mean
    noise = noise.detach()
    covariance = noise @ noise.T
    response_covariance = deviations.T @ deviations
    component_logs = []
    for scale in scales:
        cholesky = torch.linalg.cholesky(covariance + scale * response_covariance)
        residual = torch.linalg.solve_triangular(cholesky, (observation - means).T, upper=False)
        log_density = -0.5 * (
            means.shape[1] * math.log(2 * math.pi)
            + 2 * cholesky.diagonal().log().sum()
            + residual.square().sum(dim=0)
        )
        component_logs.append(log_density)
    return torch.logsumexp(torch.stack(component_logs) + masses.log()[:, None], dim=0).numpy()


def _continuous_1d(x, branch_mean, response_variance, nu):
    """Adaptive integral in log(xi), including the inverse-gamma Jacobian."""
    alpha = beta = nu / 2
    constant = alpha * math.log(beta) - gammaln(alpha)

    def integrand(log_scale):
        if log_scale < -700:
            return 0.0
        log_variance = np.logaddexp(0.0, log_scale + math.log(response_variance))
        normal = -0.5 * (
            math.log(2 * math.pi) + log_variance + (x - branch_mean) ** 2 * math.exp(-log_variance)
        )
        mixing = constant - alpha * log_scale - beta * math.exp(-log_scale)
        return math.exp(normal + mixing)

    return quad(integrand, -np.inf, np.inf, epsabs=1e-12, epsrel=1e-11, limit=300)


def _normalization_check(scales, masses):
    responses = _tensor([[-1.0], [0.4], [1.2]])
    weights = _tensor([0.2, 0.5, 0.3])
    issued = issue(responses, weights, torch.eye(3, dtype=DTYPE), _tensor([[1.3]]))

    def density(z):
        result = assimilate(issued, _tensor([z]), scales, masses)
        return np.r_[result.log_likelihood.detach().exp().numpy(), result.log_predictive.detach().exp().numpy()]

    integral, integration_error = quad_vec(density, -np.inf, np.inf, epsabs=1e-9, epsrel=1e-9)
    assert np.max(np.abs(integral - 1)) < 5e-8, (integral, integration_error)
    return {
        "branch_integrals": integral[:-1].tolist(),
        "predictive_integral": float(integral[-1]),
        "integration_error_estimate": float(integration_error),
        "absolute_error_max": float(np.max(np.abs(integral - 1))),
    }


def _full_covariance_check(scales, masses):
    responses = _tensor([[-0.9, 0.3, 0.2], [0.7, -0.2, 0.4], [1.3, 0.8, -0.5], [0.7, -0.2, 0.4]])
    weights = _tensor([0.13, 0.27, 0.41, 0.19])
    transition = _tensor([[0.7, 0.1, 0.1, 0.1], [0.05, 0.65, 0.2, 0.1], [0.1, 0.05, 0.75, 0.1], [0.2, 0.1, 0.1, 0.6]])
    noise = _tensor([[1.2, 0, 0], [0.3, 0.8, 0], [-0.2, 0.15, 1.1]])
    observation = _tensor([0.2, 0.45, -0.3])
    issued = issue(responses, weights, transition, noise)
    result = assimilate(issued, observation, scales, masses)
    prior = transition.T @ weights
    oracle = _full_covariance_log_likelihood(responses, prior, noise, observation, scales, masses)
    oracle_log_predictive = logsumexp(np.log(prior.numpy()) + oracle)
    oracle_posterior = np.exp(np.log(prior.numpy()) + oracle - oracle_log_predictive)
    likelihood_error = np.max(np.abs(result.log_likelihood.detach().numpy() - oracle))
    posterior_error = np.max(np.abs(result.posterior.detach().numpy() - oracle_posterior))
    assert torch.allclose(issued.prior, prior, atol=1e-14, rtol=0)
    assert likelihood_error < 1e-11, likelihood_error
    assert posterior_error < 1e-12, posterior_error
    assert abs(float(result.log_predictive) - oracle_log_predictive) < 1e-11
    prior_odds = float(prior[1] / prior[3])
    posterior_odds = float(result.posterior[1] / result.posterior[3])
    assert abs(prior_odds - posterior_odds) < 1e-12
    return {
        "noise_covariance": _values(noise @ noise.T),
        "prior": _values(prior),
        "log_likelihood_low_rank": _values(result.log_likelihood),
        "log_likelihood_full_covariance": oracle.tolist(),
        "log_likelihood_absolute_error_max": float(likelihood_error),
        "log_predictive": float(result.log_predictive),
        "posterior": _values(result.posterior),
        "posterior_absolute_error_max": float(posterior_error),
        "equal_response_prior_odds": prior_odds,
        "equal_response_posterior_odds": posterior_odds,
    }


def _continuous_reference_check(scales, masses):
    nu = 3.0
    responses = _tensor([[-1.0], [0.4], [1.2]])
    noise = _tensor([[1.3]])
    issued = issue(responses, _tensor([0.2, 0.5, 0.3]), torch.eye(3, dtype=DTYPE), noise)
    center = float(issued.prior @ responses[:, 0])
    branch_means = ((responses[:, 0] - center) / noise[0, 0]).numpy()
    response_variance = sum(float(value) ** 2 for value in branch_means)
    rows = []
    for x in (0.0, 1.0, 3.0, 8.0, 20.0):
        reference_pairs = [_continuous_1d(x, b, response_variance, nu) for b in branch_means]
        reference = np.asarray([pair[0] / float(noise[0, 0]) for pair in reference_pairs])
        reference_log = np.log(reference)
        result = assimilate(issued, _tensor([center + float(noise[0, 0]) * x]), scales, masses)
        finite_log = result.log_likelihood.detach().numpy()
        log_error = finite_log - reference_log
        pairwise_log_ratio_error = log_error[:, None] - log_error[None, :]
        assert np.all(reference > 0) and np.all(np.isfinite(finite_log))
        assert max(pair[1] for pair in reference_pairs) < 1e-10
        rows.append({
            "whitened_residual_from_center": x,
            "continuous_density_in_z": reference.tolist(),
            "finite_density_in_z": np.exp(finite_log).tolist(),
            "density_absolute_error_max": float(np.max(np.abs(np.exp(finite_log) - reference))),
            "density_relative_error_max": float(np.max(np.abs(np.expm1(log_error)))),
            "log_density_error_max": float(np.max(np.abs(log_error))),
            "pairwise_log_ratio_error_max": float(np.max(np.abs(pairwise_log_ratio_error))),
            "continuous_integral_error_estimate_max": max(pair[1] / float(noise[0, 0]) for pair in reference_pairs),
        })
    return {
        "nu": nu,
        "order": int(scales.numel()),
        "branch_means_whitened": branch_means.tolist(),
        "response_variance": response_variance,
        "rows": rows,
        "interpretation": "Measured approximation errors at fixed residuals; a finite Gaussian mixture has no continuous heavy-tail asymptotic guarantee.",
    }


def _zero_response_check(scales, masses):
    responses = _tensor([[0.4, -0.3]] * 3, requires_grad=True)
    weights = _tensor([0.6, 0.3, 0.1])
    transition = _tensor([[0.75, 0.2, 0.05], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]])
    noise = _tensor([[1.1, 0], [0.25, 0.8]])
    observation = _tensor([1.0, 0.2])
    issued = issue(responses, weights, transition, noise)
    result = assimilate(issued, observation, scales, masses)
    prior = transition.T @ weights
    assert torch.equal(result.posterior, issued.prior)
    assert torch.allclose(issued.prior, prior, atol=1e-14, rtol=0)
    assert not torch.equal(prior, weights)
    assert float((result.log_likelihood - result.log_likelihood[0]).abs().max()) < 1e-14
    posterior_gradient = torch.autograd.grad(result.posterior[0], responses)[0]
    assert float(posterior_gradient.abs().max()) > 1e-3, "Exact equal-likelihood output must preserve response derivatives."
    step = 1e-5
    plus, minus = responses.detach().clone(), responses.detach().clone()
    plus[0, 0] += step
    minus[0, 0] -= step
    plus_value = assimilate(issue(plus, weights, transition, noise), observation, scales, masses).posterior[0]
    minus_value = assimilate(issue(minus, weights, transition, noise), observation, scales, masses).posterior[0]
    finite_difference = float((plus_value - minus_value) / (2 * step))
    assert abs(float(posterior_gradient[0, 0]) - finite_difference) < 1e-7
    zero_support = _tensor([0.7, 0.3, 0.0])
    sparse = assimilate(issue(_tensor([[-1.0], [0.2], [1.1]]), zero_support, torch.eye(3, dtype=DTYPE), _tensor([[1.0]])), _tensor([0.5]), scales, masses)
    assert float(sparse.posterior[2]) == 0.0
    return {
        "old_weights": _values(weights),
        "transported_prior": _values(issued.prior),
        "posterior": _values(result.posterior),
        "transport_l1_increment": float((prior - weights).abs().sum()),
        "write_l1_increment": float((result.posterior - issued.prior).abs().sum()),
        "equal_response_posterior_gradient": _values(posterior_gradient),
        "gradient_00_finite_difference": finite_difference,
        "zero_prior_posterior": _values(sparse.posterior),
    }


def _orthogonal_check(scales, masses):
    white_responses = _tensor([[-1.0, 0.2, 0, 0, 0], [0.5, -0.6, 0, 0, 0], [1.3, 0.7, 0, 0, 0]])
    noise = _tensor([[1.2, 0, 0, 0, 0], [0.3, 0.8, 0, 0, 0], [0, 0, 1.1, 0, 0], [0, 0, 0.2, 0.9, 0], [0, 0, 0, 0.1, 1.3]])
    responses = white_responses @ noise.T
    issued = issue(responses, _tensor([0.2, 0.5, 0.3]), torch.eye(3, dtype=DTYPE), noise)
    center = issued.prior @ responses
    x = _tensor([0.25, -0.1, 0.1, -0.2, 0.3])
    orthogonal = _tensor([0, 0, 1e6, -2e6, 3e6])
    ordinary = assimilate(issued, center + noise @ x, scales, masses)
    perturbed = assimilate(issued, center + noise @ (x + orthogonal), scales, masses)
    log_ratios = ordinary.relative_log_likelihood - ordinary.relative_log_likelihood[0]
    perturbed_log_ratios = perturbed.relative_log_likelihood - perturbed.relative_log_likelihood[0]
    ratio_error = float((log_ratios - perturbed_log_ratios).abs().max())
    posterior_error = float((ordinary.posterior - perturbed.posterior).abs().max())
    expected_log_density_change = -0.5 * float(orthogonal @ orthogonal + 2 * x @ orthogonal)
    actual_log_density_change = float(perturbed.log_predictive - ordinary.log_predictive)
    assert ratio_error < 1e-10, ratio_error
    assert posterior_error < 1e-11, posterior_error
    assert abs(actual_log_density_change - expected_log_density_change) < 16 * np.finfo(float).eps * abs(expected_log_density_change)
    return {
        "orthogonal_whitened_norm": float(torch.linalg.vector_norm(orthogonal)),
        "log_ratios": _values(log_ratios),
        "perturbed_log_ratios": _values(perturbed_log_ratios),
        "log_ratio_absolute_error_max": ratio_error,
        "posterior_absolute_error_max": posterior_error,
        "expected_common_log_density_change": expected_log_density_change,
        "actual_log_predictive_change": actual_log_density_change,
    }


def _weak_response_check(scales, masses):
    weights = _tensor([0.2, 0.5, 0.3])
    pattern = _tensor([[-1.0, 0.2], [0.4, -0.6], [1.2, 0.7]])
    rows = []
    for amplitude in (1.0, 0.1, 0.01, 0.001, 0.0001, 0.000001, 0.0):
        issued = issue(amplitude * pattern, weights, torch.eye(3, dtype=DTYPE), torch.eye(2, dtype=DTYPE))
        result = assimilate(issued, _tensor([0.8, -0.25]), scales, masses)
        rows.append({
            "response_amplitude": amplitude,
            "log_likelihood_spread": float(result.relative_log_likelihood.max() - result.relative_log_likelihood.min()),
            "write_l1_increment": float((result.posterior - issued.prior).abs().sum()),
            "posterior": _values(result.posterior),
        })
    assert rows[-2]["log_likelihood_spread"] < 1e-5
    assert rows[-2]["write_l1_increment"] < 1e-5
    assert rows[-1]["write_l1_increment"] == 0.0
    assert all(rows[i + 1]["write_l1_increment"] < rows[i]["write_l1_increment"] for i in range(2, len(rows) - 1))
    return {"rows": rows}


def _gradient_check(scales, masses):
    prototypes = _tensor([-0.7, 0.35, 1.1], requires_grad=True)
    response_map = _tensor([[0.7, -0.1, 0.2], [-0.15, 0.55, 0.3]], requires_grad=True)
    transport_logits = _tensor([[1.5, 0.1, -0.2, -0.6], [-0.2, 1.1, 0.3, 0], [0.2, -0.3, 1.3, 0.1], [0, 0.2, -0.4, 1.4]], requires_grad=True)
    weights = _tensor([0.4, 0.2, 0.25, 0.15])
    noise = _tensor([[1.0, 0, 0], [0.2, 0.8, 0], [-0.1, 0.15, 1.1]])

    def objective(u, mapping, logits):
        coordinates = torch.cat((u.new_zeros(1), u))
        features = torch.stack((coordinates, coordinates.square()), dim=1)
        responses = _tensor([0.1, -0.2, 0.05]) + features @ mapping
        transition = logits.softmax(dim=1)
        first = assimilate(issue(responses, weights, transition, noise), _tensor([0.65, 0.1, -0.2]), scales, masses)
        later_responses = _tensor([-0.1, 0.05, 0.2]) + features @ (mapping * _tensor([[0.6, 1.2, -0.4], [1.1, -0.7, 0.8]]))
        second = assimilate(issue(later_responses, first.posterior, transition, noise), _tensor([-0.2, 0.5, 0.35]), scales, masses)
        return -first.log_predictive - second.log_predictive, first, second

    loss, first, second = objective(prototypes, response_map, transport_logits)
    gradients = torch.autograd.grad(loss, (prototypes, response_map, transport_logits))
    for gradient in gradients:
        assert torch.isfinite(gradient).all() and float(gradient.norm()) > 1e-6
    step = 1e-5
    checks = []
    for parameter_number, index in ((0, (1,)), (2, (0, 0))):
        parameters = [prototypes.detach(), response_map.detach(), transport_logits.detach()]
        plus = [parameter.clone() for parameter in parameters]
        minus = [parameter.clone() for parameter in parameters]
        plus[parameter_number][index] += step
        minus[parameter_number][index] -= step
        finite_difference = float((objective(*plus)[0] - objective(*minus)[0]) / (2 * step))
        automatic = float(gradients[parameter_number][index])
        assert abs(automatic - finite_difference) < 2e-7
        checks.append({"parameter": "prototype" if parameter_number == 0 else "transport_logit", "index": list(index), "autograd": automatic, "finite_difference": finite_difference})
    return {
        "causal_two_arrival_nll": float(loss),
        "first_posterior": _values(first.posterior),
        "second_posterior": _values(second.posterior),
        "prototype_gradients": _values(gradients[0]),
        "response_map_gradient_norm": float(gradients[1].norm()),
        "transport_gradient_norm": float(gradients[2].norm()),
        "finite_difference_checks": checks,
        "optimizer_steps": 0,
    }


def _cost_and_pruning_check():
    weights = _tensor([0.5, 0.5])
    trajectories = _tensor([[[-2.0, -1.0], [2.0, 1.0]], [[0.8, 0.8], [0.8, 0.8]]])
    full_cost = trajectories.square().sum(dim=-1) @ weights
    cost_of_mean = (trajectories * weights[None, :, None]).sum(dim=1).square().sum(dim=-1)
    assert int(full_cost.argmin()) == 1 and int(cost_of_mean.argmin()) == 0
    pruning_weights = _tensor([0.62, 0.25, 0.10, 0.03])
    branch_costs = _tensor([[0.1, 0.2, 0.8, 1.0], [0.7, 0.6, 0.2, 0.0], [0.8, 0.9, 0.1, 0.2]])
    discarded_mass = float(pruning_weights[-1])
    exact = branch_costs @ pruning_weights
    approximate = branch_costs[:, :-1] @ (pruning_weights[:-1] / pruning_weights[:-1].sum())
    bound = discarded_mass * 1.0  # Shared branch cost interval [0, 1].
    ordered = approximate.sort().values
    gap = float(ordered[1] - ordered[0])
    assert bool(((exact - approximate).abs() <= bound + 1e-14).all())
    assert gap > 2 * bound
    assert int(exact.argmin()) == int(approximate.argmin())
    return {
        "full_trajectories": _values(trajectories),
        "expected_full_trajectory_cost": _values(full_cost),
        "cost_of_mean_trajectory": _values(cost_of_mean),
        "full_expectation_choice": int(full_cost.argmin()),
        "mean_trajectory_choice": int(cost_of_mean.argmin()),
        "pruning": {
            "branch_cost_interval": [0.0, 1.0],
            "discarded_mass": discarded_mass,
            "per_candidate_error_bound": bound,
            "exact_costs": _values(exact),
            "pruned_costs": _values(approximate),
            "measured_absolute_errors": _values((exact - approximate).abs()),
            "approximate_best_runner_up_gap": gap,
            "required_gap_strictly_above": 2 * bound,
            "certified_choice": int(approximate.argmin()),
        },
    }


def _multimodal_check(scales, masses):
    coordinates = _tensor([0., -.8, .8])
    prior = _tensor([.5, .25, .25])
    issued = issue(coordinates.square()[:, None], prior, torch.eye(3, dtype=DTYPE), _tensor([[1.]]))
    result = assimilate(issued, _tensor([.64]), scales, masses)
    assert result.posterior[1] == result.posterior[2] and result.posterior[1] > prior[1]
    return {"coordinates": _values(coordinates), "responses": _values(coordinates.square()),
            "prior": _values(prior), "posterior": _values(result.posterior)}


def numerical_checks():
    """Run fixed synthetic checks; raise AssertionError on invariant failure."""
    scales, masses = quadrature(nu=3.0, order=32)
    assert scales.dtype == DTYPE and masses.dtype == DTYPE
    assert bool((scales > 0).all()) and bool((masses > 0).all())
    assert abs(float(masses.sum()) - 1.0) < 1e-14
    return {
        "quadrature": {"nu": 3.0, "order": 32, "scale_min": float(scales.min()), "scale_max": float(scales.max()), "mass_sum": float(masses.sum())},
        "probability_normalization": _normalization_check(scales, masses),
        "low_rank_full_covariance": _full_covariance_check(scales, masses),
        "continuous_inverse_gamma_reference": _continuous_reference_check(scales, masses),
        "zero_response_with_transport": _zero_response_check(scales, masses),
        "orthogonal_perturbation": _orthogonal_check(scales, masses),
        "weak_response_limit": _weak_response_check(scales, masses),
        "squared_response_preserves_two_explanations": _multimodal_check(scales, masses),
        "causal_gradients": _gradient_check(scales, masses),
        "full_cost_and_certified_pruning": _cost_and_pruning_check(),
    }
