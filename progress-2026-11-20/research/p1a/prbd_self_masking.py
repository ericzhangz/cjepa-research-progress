"""Fixed P0 evidence checks, with no fitting or response-scale selection.

The prescribed scalar example is f_-= -v, f_+=v, R=1, prior=(1/2,1/2)
and z=+-v.  The grid, nu=3, Q=32 and bounded rho=lambda=1 are fixed below.
Continuous and scalar finite-mixture references evaluate scalar variances
directly, avoiding the cancellation of the original low-rank formula.
"""

from __future__ import annotations

from .prbd_paths import EVIDENCE, output_path

import argparse
import csv
import json
import math
import platform
from pathlib import Path

import numpy as np
import scipy
import torch
from scipy.integrate import quad, quad_vec
from scipy.special import expit, gammaln, logsumexp

from .prbd import assimilate, issue, quadrature
from .prbd_checks import numerical_checks


DTYPE = torch.float64
NU = 3.0
ORDER = 32
SCALAR_GRID = (0.0, 1e-12, 1e-8, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1e4)
OUT = EVIDENCE


def tensor(value, *, requires_grad=False):
    return torch.tensor(value, dtype=DTYPE, requires_grad=requires_grad)


def values(value):
    return value.detach().cpu().tolist()


def continuous_log_density(residual, response_variance):
    """Independent adaptive integral in log(xi), rescaled before integration.

    Rescaling only improves integration precision at very small densities.
    It does not change the noise distribution or choose quadrature nodes.
    """
    if response_variance == 0:
        return -0.5 * (math.log(2 * math.pi) + residual**2), 0.0
    alpha = beta = NU / 2
    constant = alpha * math.log(beta) - gammaln(alpha)
    log_response_variance = math.log(response_variance)

    def log_integrand(log_scale):
        if log_scale < -700:
            return -math.inf
        log_variance = np.logaddexp(0.0, log_scale + log_response_variance)
        normal = -0.5 * (
            math.log(2 * math.pi) + log_variance
            + residual**2 * math.exp(-log_variance)
        )
        return normal + constant - alpha * log_scale - beta * math.exp(-log_scale)

    approximate_peak = math.log((NU + residual**2 / response_variance) / (NU + 1))
    pivot = max(log_integrand(0.0), log_integrand(approximate_peak))
    integral, error = quad(
        lambda t: math.exp(log_integrand(t) - pivot),
        -np.inf, np.inf, epsabs=1e-11, epsrel=2e-11, limit=400,
    )
    if integral <= 0 or not math.isfinite(integral):
        raise RuntimeError("Continuous reference integration failed")
    return math.log(integral) + pivot, error / integral


def scalar_reference(v, sign, geometry, scales, masses):
    raw_variance = 2 * v * v
    noise_variance = raw_variance if geometry == "original" else raw_variance / (1 + raw_variance)
    means = np.asarray([-v, v])
    residuals = sign * v - means
    variances = 1 + scales[:, None] * noise_variance
    logs = logsumexp(
        np.log(masses)[:, None] - 0.5 * (
            np.log(2 * math.pi * variances) + residuals[None]**2 / variances
        ), axis=0,
    )
    continuous_pairs = [continuous_log_density(float(r), noise_variance) for r in residuals]
    continuous = np.asarray([pair[0] for pair in continuous_pairs])
    correct = 1 if sign == 1 else 0
    other = 1 - correct
    return {
        "direct32_log_likelihood": logs.tolist(),
        "direct32_log_predictive": float(logsumexp(logs) - math.log(2)),
        "direct32_correct_posterior": float(expit(logs[correct] - logs[other])),
        "direct32_log_bayes_factor": float(logs[correct] - logs[other]),
        "continuous_log_likelihood": continuous.tolist(),
        "continuous_log_predictive": float(logsumexp(continuous) - math.log(2)),
        "continuous_correct_posterior": float(expit(continuous[correct] - continuous[other])),
        "continuous_log_bayes_factor": float(continuous[correct] - continuous[other]),
        "continuous_relative_integration_error_max": max(pair[1] for pair in continuous_pairs),
        "noise_response_variance": noise_variance,
    }


def scalar_curve(scales, masses):
    rows = []
    scale_array, mass_array = scales.numpy(), masses.numpy()
    for v in SCALAR_GRID:
        for sign in (-1, 1):
            signed = issue(tensor([[-v], [v]]), tensor([0.5, 0.5]), torch.eye(2, dtype=DTYPE), tensor([[1.0]]))
            for geometry in ("original", "bounded"):
                result = assimilate(signed, tensor([sign * v]), scales, masses, geometry=geometry)
                reference = scalar_reference(v, sign, geometry, scale_array, mass_array)
                correct = 1 if sign == 1 else 0
                other = 1 - correct
                api_log_bf = float(result.relative_log_likelihood[correct] - result.relative_log_likelihood[other])
                row = {
                    "v": v, "observation_sign": sign, "geometry": geometry,
                    "api_correct_posterior": float(result.posterior[correct]),
                    "api_log_bayes_factor": api_log_bf,
                    "api_log_predictive": float(result.log_predictive),
                    **reference,
                }
                row["api_direct32_posterior_absolute_error"] = abs(row["api_correct_posterior"] - row["direct32_correct_posterior"])
                row["api_direct32_log_bayes_factor_absolute_error"] = abs(api_log_bf - row["direct32_log_bayes_factor"])
                row["api_direct32_log_predictive_absolute_error"] = abs(row["api_log_predictive"] - row["direct32_log_predictive"])
                row["finite32_continuous_posterior_absolute_error"] = abs(row["direct32_correct_posterior"] - row["continuous_correct_posterior"])
                row["finite32_continuous_log_predictive_error"] = row["direct32_log_predictive"] - row["continuous_log_predictive"]
                row["finite32_continuous_log_bayes_factor_error"] = row["direct32_log_bayes_factor"] - row["continuous_log_bayes_factor"]
                row["finite32_continuous_branch_log_density_error_max"] = float(np.max(np.abs(
                    np.asarray(row["direct32_log_likelihood"]) - np.asarray(row["continuous_log_likelihood"])
                )))
                row["gaussian_correct_posterior"] = float(expit(2 * v * v))
                row["gaussian_log_bayes_factor"] = 2 * v * v
                rows.append(row)
    # The reference is direct scalar arithmetic even where the archived
    # original Woodbury subtraction loses precision; report that loss intact.
    assert all(math.isfinite(row["api_log_predictive"]) for row in rows)
    assert max(row["api_direct32_log_predictive_absolute_error"] for row in rows if row["geometry"] == "bounded") < 1e-7
    assert max(row["continuous_relative_integration_error_max"] for row in rows) < 1e-8
    assert all(rows[i]["api_correct_posterior"] == rows[i + 2]["api_correct_posterior"] for i in range(0, len(rows), 4))
    assert all(rows[i + 1]["api_correct_posterior"] == rows[i + 3]["api_correct_posterior"] for i in range(0, len(rows), 4))
    scaled_logs = np.log(mass_array)[:, None] - 0.5 * (
        np.log(4 * math.pi * scale_array[:, None])
        + np.asarray([4.0, 0.0])[None] / (2 * scale_array[:, None])
    )
    asymptotic_logs = logsumexp(scaled_logs, axis=0)
    return {
        "grid": list(SCALAR_GRID), "means": "[-v,+v]", "observations": "-v and +v",
        "prior": [0.5, 0.5], "nu": NU, "quadrature_order": ORDER,
        "bounded_rho": 1.0, "bounded_lambda": 1.0,
        "original_continuous_correct_posterior_limit": 25 / 34,
        "original_finite32_correct_posterior_limit": float(expit(asymptotic_logs[1] - asymptotic_logs[0])),
        "bounded_continuous_correct_posterior_limit": 1.0,
        "rows": rows,
    }


def fixed_scale_check():
    rows = []
    xi = 1.0
    for v in SCALAR_GRID:
        for sign in (-1, 1):
            signed = issue(tensor([[-v], [v]]), tensor([0.5, 0.5]), torch.eye(2, dtype=DTYPE), tensor([[1.0]]))
            for geometry in ("original", "bounded"):
                result = assimilate(signed, tensor([sign * v]), tensor([xi]), tensor([1.0]), geometry=geometry)
                variance = 2 * v * v
                noise_variance = variance if geometry == "original" else variance / (1 + variance)
                analytic = 2 * v * v / (1 + xi * noise_variance)
                correct = 1 if sign == 1 else 0
                measured = float(result.relative_log_likelihood[correct] - result.relative_log_likelihood[1 - correct])
                rows.append({"v": v, "observation_sign": sign, "geometry": geometry,
                             "xi": xi, "analytic_log_bayes_factor": analytic,
                             "api_log_bayes_factor": measured, "absolute_error": abs(measured - analytic)})
    assert max(row["absolute_error"] for row in rows if row["v"] <= 100) < 1e-7
    return {"original_formula": "2v^2/(1+2xi*v^2)",
            "bounded_formula": "2v^2/(1+xi*2v^2/(1+2v^2))", "rows": rows}


def invariant_checks(scales, masses):
    responses = tensor([[-1.0, 0.2, 0, 0, 0], [-1.0, 0.2, 0, 0, 0], [0.5, -0.6, 0, 0, 0], [1.3, 0.7, 0, 0, 0]])
    weights = tensor([0.1, 0.2, 0.3, 0.4])
    transition = torch.eye(4, dtype=DTYPE)
    noise = torch.diag(tensor([1.2, 0.8, 1.1, 0.9, 1.3]))
    signed = issue(responses @ noise.T, weights, transition, noise)
    x = tensor([0.25, -0.1, 0.1, -0.2, 0.3])
    orthogonal = tensor([0, 0, 1e6, -2e6, 3e6])
    result = {}
    for geometry in ("original", "bounded"):
        ordinary = assimilate(signed, signed.center + noise @ x, scales, masses, geometry=geometry)
        if geometry == "original":
            default = assimilate(signed, signed.center + noise @ x, scales, masses)
            assert all(torch.equal(getattr(default, name), getattr(ordinary, name)) for name in default.__dataclass_fields__)
        perturbed = assimilate(signed, signed.center + noise @ (x + orthogonal), scales, masses, geometry=geometry)
        odds_error = abs(float(ordinary.posterior[0] / ordinary.posterior[1] - weights[0] / weights[1]))
        log_ratios = ordinary.relative_log_likelihood - ordinary.relative_log_likelihood[0]
        perturbed_ratios = perturbed.relative_log_likelihood - perturbed.relative_log_likelihood[0]
        ratio_error = float((log_ratios - perturbed_ratios).abs().max())
        posterior_error = float((ordinary.posterior - perturbed.posterior).abs().max())
        expected_common_change = -0.5 * float(orthogonal @ orthogonal + 2 * x @ orthogonal)
        actual_change = float(perturbed.log_predictive - ordinary.log_predictive)
        zero_responses = tensor([[0.4, -0.3]] * 3, requires_grad=True)
        zero = issue(zero_responses, tensor([0.6, 0.3, 0.1]), torch.eye(3, dtype=DTYPE), torch.eye(2, dtype=DTYPE))
        zero_result = assimilate(zero, tensor([1.0, 0.2]), scales, masses, geometry=geometry)
        zero_gradient = torch.autograd.grad(zero_result.posterior[0], zero_responses)[0]
        sparse = issue(tensor([[-1.0], [0.2], [1.1]]), tensor([0.7, 0.3, 0.0]), torch.eye(3, dtype=DTYPE), tensor([[1.0]]))
        sparse_result = assimilate(sparse, tensor([0.5]), scales, masses, geometry=geometry)
        assert torch.equal(zero_result.posterior, zero.prior)
        assert float(sparse_result.posterior[2]) == 0.0
        assert float(zero_gradient.norm()) > 0
        assert odds_error < 1e-12 and ratio_error < 1e-10 and posterior_error < 1e-11
        assert abs(actual_change - expected_common_change) < 0.1
        result[geometry] = {"equal_response_odds_error": odds_error,
                            "orthogonal_log_bayes_factor_error_max": ratio_error,
                            "orthogonal_posterior_error_max": posterior_error,
                            "orthogonal_expected_log_predictive_change": expected_common_change,
                            "orthogonal_measured_log_predictive_change": actual_change,
                            "zero_response_exact_prior": True,
                            "zero_prior_support_preserved": True,
                            "zero_response_posterior_gradient_norm": float(zero_gradient.norm())}
    return result


def direct_density(responses, weight_logits, noise, observation, scales, masses, geometry):
    """Ambient covariance reference with independent full-D factorization."""
    prior = weight_logits.softmax(dim=0)
    center = prior @ responses
    deviations = torch.linalg.solve_triangular(noise, (responses - center).T, upper=False)
    if geometry == "original":
        response_covariance = deviations @ deviations.T
    else:
        regularized = torch.eye(len(prior), dtype=DTYPE) + deviations.T @ deviations
        response_covariance = deviations @ torch.linalg.solve(regularized, deviations.T)
    d = responses.shape[1]
    white_covariance = torch.eye(d, dtype=DTYPE)[None] + scales[:, None, None] * response_covariance[None]
    covariance = noise[None] @ white_covariance @ noise.T[None]
    cholesky = torch.linalg.cholesky(covariance)
    residual = (observation - responses).T[None].expand(len(scales), -1, -1)
    solved = torch.linalg.solve_triangular(cholesky, residual, upper=False)
    components = -0.5 * (
        d * math.log(2 * math.pi)
        + 2 * cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)[:, None]
        + solved.square().sum(dim=1)
    )
    logs = torch.logsumexp(masses.log()[:, None] + components, dim=0)
    joint = prior.log() + logs
    return logs, torch.logsumexp(joint, dim=0), joint.softmax(dim=0), torch.linalg.cond(white_covariance).max()


def ambient_and_gradient_checks(scales, masses):
    generator = torch.Generator().manual_seed(20261119)
    rows = []
    for d in (3, 7, 192):
        responses = (0.7 * torch.randn(4, d, generator=generator, dtype=DTYPE) / math.sqrt(d)).requires_grad_()
        logits = tensor([-0.3, 0.2, 0.1, -0.1], requires_grad=True)
        observation = (0.5 * torch.randn(d, generator=generator, dtype=DTYPE)).requires_grad_()
        noise = torch.diag(torch.linspace(0.8, 1.2, d, dtype=DTYPE))
        noise += 0.03 * torch.randn(d, d, generator=generator, dtype=DTYPE).tril(-1) / math.sqrt(d)
        parameters = (responses, logits, observation)
        for geometry in ("original", "bounded"):
            signed = issue(responses, logits.softmax(0), torch.eye(4, dtype=DTYPE), noise)
            actual = assimilate(signed, observation, scales, masses, geometry=geometry)
            logs, predictive, posterior, condition = direct_density(responses, logits, noise, observation, scales, masses, geometry)
            probe = tensor([0.2, -0.1, 0.4, -0.3])
            objective = -actual.log_predictive + 0.17 * (actual.posterior * probe).sum() + 0.03 * actual.log_likelihood.sum()
            reference_objective = -predictive + 0.17 * (posterior * probe).sum() + 0.03 * logs.sum()
            gradients = torch.autograd.grad(objective, parameters)
            references = torch.autograd.grad(reference_objective, parameters)
            errors = [float((a - b).abs().max()) for a, b in zip(gradients, references)]
            density_error = float((actual.log_likelihood - logs).detach().abs().max())
            predictive_error = abs(float((actual.log_predictive - predictive).detach()))
            posterior_error = float((actual.posterior - posterior).detach().abs().max())
            assert float(condition.detach()) < 1000
            assert density_error < 1e-10 and predictive_error < 1e-10 and posterior_error < 1e-11
            assert max(errors) < 1e-9
            rows.append({"ambient_dimension": d, "prototypes": 4, "geometry": geometry,
                         "direct_covariance_condition_max": float(condition.detach()),
                         "log_likelihood_absolute_error_max": density_error,
                         "H1_absolute_log_predictive": float(actual.log_predictive.detach()),
                         "H1_absolute_log_predictive_error": predictive_error,
                         "posterior_absolute_error_max": posterior_error,
                         "response_gradient_absolute_error_max": errors[0],
                         "prior_logit_gradient_absolute_error_max": errors[1],
                         "observation_gradient_absolute_error_max": errors[2]})
    return rows


def normalization_check(scales, masses):
    signed = issue(tensor([[-1.0], [0.4], [1.2]]), tensor([0.2, 0.5, 0.3]), torch.eye(3, dtype=DTYPE), tensor([[1.3]]))

    def densities(z):
        result = assimilate(signed, tensor([z]), scales, masses, geometry="bounded")
        return np.r_[result.log_likelihood.detach().exp().numpy(), float(result.log_predictive.detach().exp())]

    integral, error = quad_vec(densities, -np.inf, np.inf, epsabs=1e-9, epsrel=1e-9)
    assert float(np.max(np.abs(integral - 1))) < 5e-8
    return {"branch_density_integrals": integral[:-1].tolist(),
            "predictive_density_integral": float(integral[-1]),
            "integration_error_estimate": float(error),
            "absolute_integral_error_max": float(np.max(np.abs(integral - 1)))}


def stability_gradients(scales, masses):
    rows = []
    for v in (0.0, 1e-12, 1e-8, 1.0, 100.0, 1e4):
        for sign in (-1, 1):
            responses = tensor([[-v], [v]], requires_grad=True)
            observation = tensor([sign * v], requires_grad=True)
            signed = issue(responses, tensor([0.5, 0.5]), torch.eye(2, dtype=DTYPE), tensor([[1.0]]))
            result = assimilate(signed, observation, scales, masses, geometry="bounded")
            gradients = torch.autograd.grad(-result.log_predictive, (responses, observation))
            assert all(torch.isfinite(g).all() for g in gradients)
            rows.append({"v": v, "observation_sign": sign,
                         "absolute_log_predictive": float(result.log_predictive.detach()),
                         "response_nll_gradient": values(gradients[0]),
                         "observation_nll_gradient": values(gradients[1])})
    return rows


def write_outputs(report, out):
    out = output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    (output_path(out / "p0.json")).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    flat_rows = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in report["scalar_curve"]["rows"]]
    with (out / "p0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    for geometry, color in (("original", "#0072B2"), ("bounded", "#D55E00")):
        rows = [r for r in flat_rows if r["geometry"] == geometry and r["observation_sign"] == 1 and r["v"] >= 0.1]
        x = [r["v"] for r in rows]
        axes[0].plot(x, [r["continuous_correct_posterior"] for r in rows], "-o", color=color, label=f"{geometry}, continuous")
        axes[0].plot(x, [r["direct32_correct_posterior"] for r in rows], "--x", color=color, label=f"{geometry}, Q32")
        axes[1].plot(x, [r["finite32_continuous_branch_log_density_error_max"] for r in rows], "-o", color=color, label=geometry)
        axes[2].plot(x, [r["finite32_continuous_log_predictive_error"] for r in rows], "-o", color=color, label=geometry)
    gaussian_rows = [r for r in flat_rows if r["geometry"] == "original" and r["observation_sign"] == 1 and r["v"] >= 0.1]
    axes[0].plot([r["v"] for r in gaussian_rows], [r["gaussian_correct_posterior"] for r in gaussian_rows], ":", color="#009E73", label="Gaussian")
    axes[0].axhline(25 / 34, color="#555555", linewidth=0.8, label="original limit 25/34")
    axes[0].set_ylim(0.48, 1.02)
    axes[0].set_ylabel("Posterior of observed branch")
    axes[1].set_ylabel("Max branch log-density error: Q32 vs continuous")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("Q32 minus continuous log predictive density")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Half separation v (whitened units)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.savefig(out / "p0.png", dpi=180)
    plt.close(fig)
    curve = report["scalar_curve"]
    largest = [r for r in curve["rows"] if r["v"] == 1e4 and r["observation_sign"] == 1]
    lines = [
        "# P0: fixed scalar evidence and bounded-geometry verification", "",
        "No fitting, node selection, hyperparameter tuning, simulator rollout or FourRoom execution.", "",
        "The branch means remain [-v,+v]. The original covariance is I+xi*D D^T. "
        "The bounded covariance is I+xi*P with P=D(G+I)^-1 D^T, rho=lambda=1 in fixed whitened units. "
        "Bounding changes the noise geometry, not the branch means.", "",
        "The prescribed fixed-xi log Bayes factor is 2v^2/(1+2xi*v^2) for the original geometry. "
        "It saturates at 1/xi. For the bounded geometry it is 2v^2/[1+xi*2v^2/(1+2v^2)].", "",
        f"For nu=3 the original continuous posterior tends to 25/34 = {25/34:.12f}; "
        f"the fixed Q32 limit is {curve['original_finite32_correct_posterior_limit']:.12f}. "
        "The bounded continuous posterior tends to 1. Both observation signs are recorded.", "",
    ]
    for row in largest:
        lines.append(f"At v=10000, {row['geometry']}: continuous posterior {row['continuous_correct_posterior']:.12g}, "
                     f"Q32 posterior {row['direct32_correct_posterior']:.12g}; "
                     f"archived-formula vs direct Q32 log-Bayes-factor error {row['api_direct32_log_bayes_factor_absolute_error']:.6g}; "
                     f"Q32 minus continuous absolute log predictive density {row['finite32_continuous_log_predictive_error']:.6g} nat.")
        lines.append(f"Its Q32 log Bayes factor is {row['direct32_log_bayes_factor']:.12g}, "
                     f"versus {row['continuous_log_bayes_factor']:.12g} continuously; "
                     f"the maximum branch log-density error is {row['finite32_continuous_branch_log_density_error_max']:.6g} nat. "
                     "A saturated posterior can conceal the wrong-branch tail-density error.")
    max_density = max(r["log_likelihood_absolute_error_max"] for r in report["ambient_and_gradients"])
    max_gradient = max(max(r[key] for key in ("response_gradient_absolute_error_max", "prior_logit_gradient_absolute_error_max", "observation_gradient_absolute_error_max")) for r in report["ambient_and_gradients"])
    lines += ["", f"Direct well-conditioned ambient dimensions 3, 7 and 192 agree in absolute likelihood to {max_density:.3g} "
              f"and gradients to {max_gradient:.3g}. Equal-response odds, exact zero-response no-write, "
              "orthogonal evidence invariance, normalized absolute densities and bounded small/large-v gradients pass. "
              "The original default and its previous numerical checks are retained.", "",
              "Finite Q32 is a normalized finite Gaussian mixture; it does not inherit the continuous infinite-tail law. "
              "At large separation the original low-rank subtraction also loses numerical precision. "
              "The direct scalar references and approximation errors are retained, without selecting a new rule.", "",
              "Reproduce: `python -m research.p1a.prbd_self_masking`. "
              "This run used `an isolated Python environment`; "
              "the base Conda interpreter encountered conflicting OpenMP runtimes, so no duplicate-runtime override was used.", ""]
    (output_path(out / "p0.md")).write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    scales, masses = quadrature(nu=NU, order=ORDER)
    report = {
        "scope": "P0 fixed mathematical checks only; application goal remains paused",
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "scalar_curve": scalar_curve(scales, masses),
        "fixed_xi": fixed_scale_check(),
        "invariants": invariant_checks(scales, masses),
        "ambient_and_gradients": ambient_and_gradient_checks(scales, masses),
        "bounded_absolute_density_normalization": normalization_check(scales, masses),
        "bounded_numerical_stability": stability_gradients(scales, masses),
        "original_regression": numerical_checks(),
    }
    write_outputs(report, args.out)
    print(json.dumps({"output": str(args.out), "scalar_rows": len(report["scalar_curve"]["rows"]), "checks": "passed"}))


if __name__ == "__main__":
    main()
