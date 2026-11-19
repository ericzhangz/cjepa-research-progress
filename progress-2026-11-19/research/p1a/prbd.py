"""Finite-response PRBD issuance and Bayesian arrival update (proposal §3).

The online covariance calculation uses only a K-by-K Gram system.  Its shared
ambient Gaussian factor is kept outside the posterior calculation, so a large
response-orthogonal residual cannot erase the relative evidence numerically.
All online operations retain PyTorch gradients and run in float64.
"""

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class IssuedPRBD:
    """Geometry signed before the observation arrives; basis is B in [D, K]."""

    prior: torch.Tensor
    center: torch.Tensor
    basis: torch.Tensor
    gram: torch.Tensor
    noise_cholesky: torch.Tensor
    log_jacobian: torch.Tensor


@dataclass(frozen=True)
class PRBDResult:
    posterior: torch.Tensor
    log_likelihood: torch.Tensor
    log_predictive: torch.Tensor
    relative_log_likelihood: torch.Tensor
    common_log_density: torch.Tensor


def _tensor(value, device=None):
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device)
    if not torch.isfinite(tensor).all():
        raise ValueError("PRBD inputs must be finite")
    return tensor


def _probabilities(value, name, dim=-1):
    total = value.sum(dim=dim, keepdim=True)
    if (value < 0).any() or not torch.allclose(
        total, torch.ones_like(total), atol=1e-6, rtol=0
    ):
        raise ValueError(f"{name} must be nonnegative and sum to one")
    # Remove only floating roundoff in already normalized probability inputs.
    return value / total


def _log_mass(mass):
    # Avoid log(0) in autograd while retaining exact zero probability support.
    positive = mass > 0
    safe_mass = torch.where(positive, mass, torch.ones_like(mass))
    return safe_mass.log().masked_fill(~positive, -torch.inf)


def quadrature(nu=3.0, order=32):
    """Return positive scales and normalized masses as CPU float64 tensors.

    Gauss--Legendre integrates the uniform quantile variable on (0, 1), with
    scales given by the InvGamma(nu/2, nu/2) inverse CDF.  The fixed finite
    Gaussian mixture is normalized, but does not inherit the continuous
    mixture's infinite-tail limit.  The caller must assess its residual range.
    Nodes are fixed constants; nu and order are not learnable parameters.
    """
    from scipy.special import gammainccinv, roots_legendre

    if not math.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("order must be a positive integer")
    nodes, weights = roots_legendre(order)
    alpha = nu / 2
    scales = alpha / gammainccinv(alpha, (nodes + 1) / 2)
    scales = _tensor(scales)
    masses = _tensor(weights)
    if (scales <= 0).any() or (masses <= 0).any():
        raise ValueError("Quadrature produced nonpositive nodes or masses")
    return scales, masses / masses.sum()


def issue(responses, weights, transition, noise_cholesky):
    """Sign one window before arrival, with responses [K,D] and T [K,K].

    T is row-stochastic: the transported prior is T.T @ weights.  The shared
    fixed noise covariance is R = L @ L.T, where noise_cholesky is L [D,D].
    Inputs may have lower precision; conversion to float64 preserves gradients.
    Probability inputs are checked before correcting floating roundoff.
    """
    responses = _tensor(responses)
    if responses.ndim != 2 or min(responses.shape) < 1:
        raise ValueError("responses must have nonempty shape [K, D]")
    k, d = responses.shape
    weights = _tensor(weights, responses.device)
    transition = _tensor(transition, responses.device)
    noise_cholesky = _tensor(noise_cholesky, responses.device)
    if weights.shape != (k,) or transition.shape != (k, k):
        raise ValueError("weights and transition must have shapes [K] and [K,K]")
    if noise_cholesky.shape != (d, d):
        raise ValueError("noise_cholesky must have shape [D,D]")
    if not torch.equal(noise_cholesky, noise_cholesky.tril()) or (
        noise_cholesky.diagonal() <= 0
    ).any():
        raise ValueError("noise_cholesky must be lower triangular with positive diagonal")
    weights = _probabilities(weights, "weights")
    transition = _probabilities(transition, "transition rows")
    prior = transition.T @ weights
    prior = prior / prior.sum()
    # Center differences before adding the common response. Identical forecasts
    # then give a literal zero basis even with a large shared latent offset.
    differences = responses - responses[0]
    mean_difference = prior @ differences
    center = responses[0] + mean_difference
    basis = torch.linalg.solve_triangular(
        noise_cholesky, (differences - mean_difference).T, upper=False
    )
    return IssuedPRBD(
        prior=prior,
        center=center,
        basis=basis,
        gram=basis.T @ basis,
        noise_cholesky=noise_cholesky,
        log_jacobian=-noise_cholesky.diagonal().log().sum(),
    )


def assimilate(issued, observation, scales, masses):
    """Assimilate one actual observation [D] using fixed 1D quadrature.

    log_likelihood is the absolute density in original z coordinates, including
    |R|^(-1/2).  relative_log_likelihood omits only common_log_density; use it
    for ratios when an enormous shared residual exceeds absolute-log precision.
    No support is added to a zero prior, and no response rank threshold is used.
    """
    observation = _tensor(observation, issued.prior.device)
    scales = _tensor(scales, issued.prior.device)
    masses = _tensor(masses, issued.prior.device)
    d, k = issued.basis.shape
    if observation.shape != (d,):
        raise ValueError("observation must have shape [D]")
    if scales.ndim != 1 or scales.numel() < 1 or masses.shape != scales.shape:
        raise ValueError("scales and masses must have matching nonempty shape [Q]")
    if (scales <= 0).any():
        raise ValueError("quadrature scales must be positive")
    masses = _probabilities(masses, "quadrature masses")
    x = torch.linalg.solve_triangular(
        issued.noise_cholesky, (observation - issued.center)[:, None], upper=False
    )[:, 0]
    bx = issued.basis.T @ x
    residual_coordinates = bx[:, None] - issued.gram
    systems = torch.eye(k, dtype=x.dtype, device=x.device)[None] + (
        scales[:, None, None] * issued.gram[None]
    )
    cholesky = torch.linalg.cholesky(systems)
    solved = torch.cholesky_solve(
        residual_coordinates[None].expand(len(scales), -1, -1), cholesky
    )
    correction = scales[:, None] * (
        residual_coordinates[None] * solved
    ).sum(dim=1)
    logdet = 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    # ||x-b_k||² = ||x||² - 2 b_k.T x + ||b_k||².  Do not add
    # the common ||x||² before posterior normalization: it may be enormous.
    component_scores = (
        bx[None] - 0.5 * issued.gram.diagonal()[None]
        - 0.5 * logdet[:, None] + 0.5 * correction
    )
    relative = torch.logsumexp(_log_mass(masses)[:, None] + component_scores, dim=0)
    log_joint = _log_mass(issued.prior) + relative
    log_relative_predictive = torch.logsumexp(log_joint, dim=0)
    posterior = (log_joint - log_relative_predictive).exp()
    if torch.equal(relative, relative[0].expand_as(relative)):
        # Exact no-write value, with the generic Bayesian derivative intact.
        posterior = issued.prior.detach() + (posterior - posterior.detach())
    common = issued.log_jacobian - 0.5 * (
        d * math.log(2 * math.pi) + x.square().sum()
    )
    return PRBDResult(
        posterior=posterior,
        log_likelihood=common + relative,
        log_predictive=common + log_relative_predictive,
        relative_log_likelihood=relative,
        common_log_density=common,
    )
