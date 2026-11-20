"""Finite-response PRBD issuance and Bayesian arrival update (proposal §3).

The legacy covariance calculation uses only a K-by-K Gram system.  Its shared
ambient Gaussian factor is kept outside the posterior calculation, so a large
response-orthogonal residual cannot erase the relative evidence numerically.
The continuous bounded rule uses a thin spectral value calculation and an
analytic first-order backward, including rank-degenerate response families.
All observation calculations run in float64.
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
    reference: str = "prior"


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


def continuous_quadrature():
    """Fixed positive log-scale rule for the continuous InvGamma(3/2,3/2).

    The trapezoid spacing is 1/4 on log(xi) in [-6,40].  Before normalizing,
    the omitted inverse-Gamma mass is below 2e-26 (the lower tail is smaller
    than exp(-600)).  Normalizing makes the implemented Gaussian mixture a
    density for every response family.  This is an approximation to the
    continuous law, with likelihood-ratio and gradient checks in the G1 report;
    it is not a claim of uniform error for arbitrarily remote observations.
    """
    log_scales = torch.linspace(-6., 40., 185, dtype=torch.float64)
    alpha = 1.5
    log_masses = (alpha * math.log(alpha) - math.lgamma(alpha)
                  - alpha * log_scales - alpha * (-log_scales).exp())
    trapezoid = torch.ones_like(log_scales)
    trapezoid[[0, -1]] = .5
    masses = (log_masses + trapezoid.log()).softmax(0)
    return log_scales.exp(), masses


def issue(responses, weights, transition, noise_cholesky, *, reference="prior"):
    """Sign one window before arrival, with responses [K,D] and T [K,K].

    T is row-stochastic: the transported prior is T.T @ weights.  The shared
    fixed noise covariance is R = L @ L.T, where noise_cholesky is L [D,D].
    Inputs may have lower precision; conversion to float64 preserves gradients.
    Probability inputs are checked before correcting floating roundoff.
    reference='uniform' uses the fixed branch mean for the observation law,
    independently of transported belief. With identity transport this new path
    preserves the supplied valid public weights bitwise, including sum roundoff;
    its gradients retain the normalized probability calculation. The default
    preserves legacy issue.
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
    public_weights = weights
    weights = _probabilities(weights, "weights")
    transition = _probabilities(transition, "transition rows")
    prior = transition.T @ weights
    prior = prior / prior.sum()
    if reference == "uniform" and torch.equal(
        transition, torch.eye(k, dtype=transition.dtype, device=transition.device)
    ):
        # Identity transport has no numerical work to perform on public state.
        # Keep its literal value while retaining the ordinary normalized
        # transport derivative, including derivatives of transition parameters.
        prior = public_weights.detach() + (prior - prior.detach())
    # Center differences before adding the common response. Identical forecasts
    # then give a literal zero basis even with a large shared latent offset.
    differences = responses - responses[0]
    if reference not in ("prior", "uniform"):
        raise ValueError("reference must be 'prior' or 'uniform'")
    mean_difference = (prior @ differences if reference == "prior"
                       else differences.mean(0))
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
        reference=reference,
    )


class _ContinuousBoundedDensity(torch.autograd.Function):
    """Stable values and first derivatives, including repeated/zero responses.

    Thin SVD only evaluates the covariance function.  Backward uses its analytic
    Gaussian score, rather than differentiating singular vectors (undefined at
    repeated singular values).  No rank threshold or covariance jitter is used.
    """

    @staticmethod
    def forward(ctx, basis, x, scales, masses):
        d, k = basis.shape
        vectors, singular, right = torch.linalg.svd(basis, full_matrices=False)
        raw = singular.square()
        eigenvalues = raw / (1 + raw)
        variances = 1 + scales[:, None] * eigenvalues[None]
        coordinates = vectors.T @ (x[:, None] - basis)
        orthogonal = x - vectors @ (vectors.T @ x)
        node_logs = -.5 * (variances.log().sum(-1)[:, None]
                           + (coordinates[None].square() / variances[:, :, None]).sum(1))
        log_joint = _log_mass(masses)[:, None] + node_logs
        reduced = torch.logsumexp(log_joint, dim=0)
        responsibilities = (log_joint - reduced[None]).exp()
        absolute = reduced - .5 * (d * math.log(2 * math.pi) + orthogonal.square().sum())
        projected_x = vectors.T @ x
        projected_basis = vectors.T @ basis
        # Remove the ambient x² factor before summing. This expression uses
        # xi*lambda/(1+xi*lambda), not 1-1/(1+xi*lambda), so exact null and
        # arbitrarily weak directions do not pick up an orthogonal residual.
        node_scores = -.5 * variances.log().sum(-1)[:, None] + .5 * (
            (scales[:, None] * eigenvalues[None] / variances)[:, :, None]
            * projected_x[None, :, None].square()
            + (2 * projected_x[:, None] * projected_basis - projected_basis.square())[None]
            / variances[:, :, None]).sum(1)
        scores = torch.logsumexp(_log_mass(masses)[:, None] + node_scores, dim=0)
        ratios = scores - scores[0]

        if not any(ctx.needs_input_grad[:2]):
            return absolute, ratios

        # r=C^-1(x-b_j), M=(I+BB.T)^-1 B.  For C=I+xi*P,
        # d log N / dB = xi U(rr.T-C^-1) U B + r e_j.T.
        projected = coordinates[None] / variances[:, :, None]
        inverse_residual = orthogonal[None, :, None] + vectors[None] @ projected
        u_residual = orthogonal[None, :, None] + vectors[None] @ (
            projected / (1 + raw)[None, :, None])
        m_coordinates = (singular / (1 + raw))[:, None] * right
        residual_m = projected.transpose(1, 2) @ m_coordinates[None]
        u_inverse_m = vectors[None] @ (
            (singular / (1 + raw).square())[None, :, None]
            * right[None] / variances[:, :, None])
        covariance_score = scales[:, None, None, None] * (
            u_residual.permute(0, 2, 1)[:, :, :, None]
            * residual_m[:, :, None, :] - u_inverse_m[:, None])
        score_basis = (responsibilities.T[:, :, None, None]
                       * covariance_score.permute(1, 0, 2, 3)).sum(1)
        mean_score = (responsibilities[:, None] * inverse_residual).sum(0).T
        score_basis = score_basis + mean_score[:, :, None] * torch.eye(
            k, dtype=basis.dtype, device=basis.device)[:, None, :]
        score_x = -mean_score
        ctx.save_for_backward(score_basis, score_x)
        # Removing branch zero retains accurate ratios even when the common
        # orthogonal log density is too large for absolute-log subtraction.
        return absolute, ratios

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, absolute_gradient, relative_gradient):
        score_basis, score_x = ctx.saved_tensors
        coefficients = absolute_gradient + relative_gradient
        coefficients = coefficients.clone()
        coefficients[0] -= relative_gradient.sum()
        return ((coefficients[:, None, None] * score_basis).sum(0),
                (coefficients[:, None] * score_x).sum(0), None, None)


def assimilate_continuous(issued, observation, scales=None, masses=None):
    """Bounded continuous nu=3 observation law used for fit, write and readout.

    Supply continuous_quadrature() once to reuse constants across an episode.
    First derivatives propagate to native responses and transported belief;
    second derivatives through the spectral integration are not implemented.
    Observation geometry should be signed with issue(reference='uniform').
    """
    if issued.reference != "uniform":
        raise ValueError("Continuous PRBD requires issue(reference='uniform')")
    if (scales is None) != (masses is None):
        raise ValueError("Supply both continuous scales and masses, or neither")
    if scales is None:
        scales, masses = continuous_quadrature()
    observation = _tensor(observation, issued.prior.device)
    scales = _tensor(scales, issued.prior.device)
    masses = _tensor(masses, issued.prior.device)
    d, _ = issued.basis.shape
    if observation.shape != (d,):
        raise ValueError("observation must have shape [D]")
    if scales.ndim != 1 or masses.shape != scales.shape or scales.numel() < 1:
        raise ValueError("scales and masses must have matching nonempty shape [Q]")
    if (scales <= 0).any():
        raise ValueError("quadrature scales must be positive")
    masses = _probabilities(masses, "quadrature masses")
    x = torch.linalg.solve_triangular(
        issued.noise_cholesky, (observation - issued.center)[:, None], upper=False)[:, 0]
    absolute, relative = _ContinuousBoundedDensity.apply(issued.basis, x, scales, masses)
    log_likelihood = absolute + issued.log_jacobian
    log_joint = _log_mass(issued.prior) + relative
    posterior = log_joint.softmax(0)
    if torch.equal(relative, relative[0].expand_as(relative)):
        posterior = issued.prior.detach() + (posterior - posterior.detach())
    common = log_likelihood[0]
    return PRBDResult(
        posterior=posterior, log_likelihood=log_likelihood,
        log_predictive=torch.logsumexp(_log_mass(issued.prior) + log_likelihood, 0),
        relative_log_likelihood=relative, common_log_density=common)


def _bounded_components(issued, x, scales):
    """Use P=D(D.T D+I)^-1 D.T without changing the branch mean offsets.

    Both regularizers are fixed at one in the calibrated whitened coordinates.
    N=D chol(D.T D+I)^-T factors P; all factorizations remain K by K.
    """
    d, k = issued.basis.shape
    identity = torch.eye(k, dtype=x.dtype, device=x.device)
    regularized = torch.linalg.cholesky(identity + issued.gram)
    noise_basis = torch.linalg.solve_triangular(
        regularized, issued.basis.T, upper=False
    ).T
    noise_gram = noise_basis.T @ noise_basis
    systems = identity[None] + scales[:, None, None] * noise_gram[None]
    cholesky = torch.linalg.cholesky(systems)
    residual = x[:, None] - issued.basis
    coordinates = noise_basis.T @ residual
    solved = torch.cholesky_solve(
        coordinates[None].expand(len(scales), -1, -1), cholesky
    )
    correction = scales[:, None] * (coordinates[None] * solved).sum(dim=1)
    logdet = 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    relative = (
        (issued.basis.T @ x)[None] - 0.5 * issued.gram.diagonal()[None]
        - 0.5 * logdet[:, None] + 0.5 * correction
    )
    # The ridge minimum equals r.T (I+xi*P)^-1 r.  Its sum of squares
    # avoids subtracting two large quadratics at large mean separation.
    coefficients = scales[:, None, None] * solved
    remaining = residual[None] - noise_basis[None] @ coefficients
    quadratic = remaining.square().sum(dim=1) + (
        coefficients.square().sum(dim=1) / scales[:, None]
    )
    absolute = issued.log_jacobian - 0.5 * (
        d * math.log(2 * math.pi) + logdet[:, None] + quadratic
    )
    return relative, absolute


def assimilate(issued, observation, scales, masses, *, geometry="original"):
    """Assimilate one actual observation [D] using fixed 1D quadrature.

    log_likelihood is the absolute density in original z coordinates, including
    |R|^(-1/2).  relative_log_likelihood omits only common_log_density; use it
    for ratios when an enormous shared residual exceeds absolute-log precision.
    No support is added to a zero prior, and no response rank threshold is used.

    geometry='original' preserves the proposal's covariance I+xi*D D.T.
    geometry='bounded' uses I+xi*P, P=D(D.T D+I)^-1 D.T, with the raw
    D=issued.basis still defining the branch means.  Its fixed rho=lambda=1
    is in whitened units; no response-amplitude tuning is performed.
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
    if geometry not in ("original", "bounded"):
        raise ValueError("geometry must be 'original' or 'bounded'")
    masses = _probabilities(masses, "quadrature masses")
    x = torch.linalg.solve_triangular(
        issued.noise_cholesky, (observation - issued.center)[:, None], upper=False
    )[:, 0]
    if geometry == "bounded":
        component_scores, absolute_components = _bounded_components(issued, x, scales)
        relative = torch.logsumexp(
            _log_mass(masses)[:, None] + component_scores, dim=0
        )
        log_likelihood = torch.logsumexp(
            _log_mass(masses)[:, None] + absolute_components, dim=0
        )
        joint = _log_mass(issued.prior) + relative
        posterior = (joint - torch.logsumexp(joint, dim=0)).exp()
        if torch.equal(relative, relative[0].expand_as(relative)):
            posterior = issued.prior.detach() + (posterior - posterior.detach())
        common = issued.log_jacobian - 0.5 * (
            d * math.log(2 * math.pi) + x.square().sum()
        )
        return PRBDResult(
            posterior=posterior,
            log_likelihood=log_likelihood,
            log_predictive=torch.logsumexp(_log_mass(issued.prior) + log_likelihood, dim=0),
            relative_log_likelihood=relative,
            common_log_density=common,
        )
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
