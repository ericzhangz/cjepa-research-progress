"""Localized Q32/continuous check at archived initial factual sources only.

Both checkpoints use the same calibrated R, uniform prior, nu=3 and fixed
rho=lambda=1. Program duplicates are checked exactly before deduplication.
No subsequent write, H4 prediction, fitting, collection or P4 is performed.
"""

from __future__ import annotations

from .prbd_paths import EVIDENCE, SPEED_DATA, output_path

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import quad
from scipy.special import gammaln, logsumexp

from .prbd import assimilate, issue, quadrature


OUT = EVIDENCE
FIT = SPEED_DATA / "fitted.pt"
NU = 3.0


def reduced_geometry(signed, observation, geometry):
    """Independent thin-SVD diagonalization; retain every singular direction.

    SVD is used in this offline numerical reference only. Keeping all K thin
    directions avoids a rank threshold, including the centered null direction.
    """
    basis = signed.basis.numpy()
    x = np.linalg.solve(signed.noise_cholesky.numpy(), observation.numpy() - signed.center.numpy())
    vectors, singular, _ = np.linalg.svd(basis, full_matrices=False)
    eigenvalues = singular**2
    if geometry == "bounded":
        eigenvalues = eigenvalues / (1 + eigenvalues)
    coordinates = vectors.T @ (x[:, None] - basis)
    orthogonal = x - vectors @ (vectors.T @ x)
    common = float(signed.log_jacobian) - 0.5 * (
        (len(x) - len(singular)) * math.log(2 * math.pi) + orthogonal @ orthogonal
    )
    return coordinates, eigenvalues, common


def reduced_node_logs(coordinates, eigenvalues, scales):
    variances = 1 + scales[:, None] * eigenvalues[None]
    return -0.5 * (
        coordinates.shape[0] * math.log(2 * math.pi)
        + np.log(variances).sum(axis=1)[:, None]
        + (coordinates[None]**2 / variances[:, :, None]).sum(axis=1)
    )


def continuous_reduced_logs(coordinates, eigenvalues):
    alpha = beta = NU / 2
    constant = alpha * math.log(beta) - gammaln(alpha)
    log_eigenvalues = np.full_like(eigenvalues, -np.inf)
    np.log(eigenvalues, out=log_eigenvalues, where=eigenvalues > 0)
    logs, relative_errors, split_errors = [], [], []
    for residual in coordinates.T:
        def log_integrand(t):
            if t < -700:
                return -math.inf
            log_variances = np.logaddexp(0.0, t + log_eigenvalues)
            normal = -0.5 * (
                len(residual) * math.log(2 * math.pi)
                + log_variances.sum() + (residual**2 * np.exp(-log_variances)).sum()
            )
            return normal + constant - alpha * t - beta * math.exp(-t)

        # Fixed reference evaluations only rescale a tiny integral; they do
        # not modify nu, Q32, covariance geometry or any inference output.
        pivot = max(log_integrand(t) for t in (-5.0, 0.0, 3.0, 5.0, 10.0, 20.0, 40.0))
        def integrand(t):
            return math.exp(log_integrand(t) - pivot)

        integral, error = quad(integrand, -np.inf, np.inf, epsabs=1e-11, epsrel=2e-11, limit=400)
        boundaries = (-np.inf, -10.0, 0.0, 10.0, 20.0, np.inf)
        split = [quad(integrand, left, right, epsabs=1e-12, epsrel=2e-11, limit=400)
                 for left, right in zip(boundaries[:-1], boundaries[1:])]
        split_integral = sum(pair[0] for pair in split)
        if integral <= 0 or split_integral <= 0:
            raise RuntimeError("Continuous reduced-density integration failed")
        split_error = abs(math.log(integral) - math.log(split_integral))
        relative_error = max(error / integral, sum(pair[1] for pair in split) / split_integral)
        if split_error > 1e-8 or relative_error > 1e-8:
            raise RuntimeError(f"Continuous quadrature failed its error check: {split_error}, {relative_error}")
        logs.append(math.log(split_integral) + pivot)
        relative_errors.append(relative_error)
        split_errors.append(split_error)
    return np.asarray(logs), max(relative_errors), max(split_errors)


def winning_indices(posterior):
    return np.flatnonzero(posterior == posterior.max()).tolist()


def evaluate(path, fitted):
    archived = torch.load(path, weights_only=True, map_location="cpu")
    noise = fitted["noise_cholesky"].double()
    scales, masses = fitted["scales"].double(), fitted["masses"].double()
    matched = [case for case in archived["cases"] if case["matched"]]
    if len(matched) != 11:
        raise RuntimeError(f"Expected 11 matched archived roots, got {len(matched)}")
    rows = []
    for case in matched:
        responses, arrivals = case["source_responses"], case["source_actual_arrivals"]
        if not torch.equal(responses[:, 0], responses[:, 1]) or not torch.equal(arrivals[:, 0], arrivals[:, 1]):
            raise RuntimeError("Programs differ at the source; cannot deduplicate")
        for speed_index, speed in enumerate((2.5, 7.5)):
            response, observation = responses[speed_index, 0], arrivals[speed_index, 0]
            k = len(response)
            signed = issue(response, torch.full((k,), 1 / k, dtype=torch.float64), torch.eye(k, dtype=torch.float64), noise)
            for geometry in ("bounded", "original"):
                finite = assimilate(signed, observation, scales, masses, geometry=geometry)
                coordinates, eigenvalues, common = reduced_geometry(signed, observation, geometry)
                finite_logs = logsumexp(np.log(masses.numpy())[:, None] + reduced_node_logs(coordinates, eigenvalues, scales.numpy()), axis=0)
                continuous_logs, relative_error, split_error = continuous_reduced_logs(coordinates, eigenvalues)
                finite_posterior = np.exp(finite_logs - logsumexp(finite_logs))
                continuous_posterior = np.exp(continuous_logs - logsumexp(continuous_logs))
                error = finite_logs - continuous_logs
                api_log_ratios = finite.relative_log_likelihood.numpy() - float(finite.relative_log_likelihood[0])
                reference_log_ratios = finite_logs - finite_logs[0]
                api_error = float(np.max(np.abs(api_log_ratios - reference_log_ratios)))
                archived_index = archived["methods"].index(f"{geometry}_source_only")
                archived_posterior = case["supports"][archived_index, speed_index, 0].numpy()
                archive_error = float(np.max(np.abs(finite_posterior - archived_posterior)))
                absolute_error = float(np.max(np.abs(common + finite_logs - finite.log_likelihood.numpy())))
                if max(api_error, archive_error, absolute_error) > 1e-8:
                    raise RuntimeError("Reduced reference or archived source no longer reproduces Q32")
                finite_winners = winning_indices(finite_posterior)
                continuous_winners = winning_indices(continuous_posterior)
                rows.append({
                    "seed": case["seed"], "speed": speed, "geometry": geometry,
                    "finite32_posterior": finite_posterior.tolist(),
                    "continuous_posterior": continuous_posterior.tolist(),
                    "posterior_total_variation": float(0.5 * np.abs(finite_posterior - continuous_posterior).sum()),
                    "max_pair_log_bayes_factor_discrepancy": float(error.max() - error.min()),
                    "max_branch_log_density_discrepancy": float(np.abs(error).max()),
                    "log_predictive_finite32_minus_continuous": float(logsumexp(finite_logs) - logsumexp(continuous_logs)),
                    "finite32_winning_prototypes": finite_winners,
                    "continuous_winning_prototypes": continuous_winners,
                    "winning_prototype_changed": finite_winners != continuous_winners,
                    "continuous_relative_integral_error_bound_max": relative_error,
                    "continuous_split_log_integral_discrepancy_max": split_error,
                    "finite32_reduced_api_log_bayes_factor_error_max": api_error,
                    "finite32_reduced_api_absolute_log_density_error_max": absolute_error,
                    "finite32_reduced_archived_posterior_error_max": archive_error,
                    "response_noise_eigenvalue_max": float(eigenvalues.max()),
                })
    summaries = {}
    for geometry in ("bounded", "original"):
        group = [r for r in rows if r["geometry"] == geometry]
        summaries[geometry] = {
            "unique_initial_sources": len(group),
            "posterior_total_variation_median": float(np.median([r["posterior_total_variation"] for r in group])),
            "posterior_total_variation_max": max(r["posterior_total_variation"] for r in group),
            "max_pair_log_bayes_factor_discrepancy": max(r["max_pair_log_bayes_factor_discrepancy"] for r in group),
            "max_branch_log_density_discrepancy": max(r["max_branch_log_density_discrepancy"] for r in group),
            "winning_prototype_changes": sum(r["winning_prototype_changed"] for r in group),
            "continuous_relative_integral_error_bound_max": max(r["continuous_relative_integral_error_bound_max"] for r in group),
            "continuous_split_log_integral_discrepancy_max": max(r["continuous_split_log_integral_discrepancy_max"] for r in group),
            "finite32_reference_api_or_archive_error_max": max(max(r[key] for key in (
                "finite32_reduced_api_log_bayes_factor_error_max", "finite32_reduced_api_absolute_log_density_error_max",
                "finite32_reduced_archived_posterior_error_max")) for r in group),
        }
    return {"archive": str(path), "matched_roots": len(matched), "summary": summaries, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT / "p0_observed_sources.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    fitted = torch.load(FIT, weights_only=True, map_location="cpu")
    if fitted["scales"].numel() != 32:
        raise RuntimeError("Expected archived Q32")
    reference_scales, reference_masses = quadrature(nu=NU, order=32)
    torch.testing.assert_close(fitted["scales"].double(), reference_scales, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(fitted["masses"].double(), reference_masses, atol=1e-12, rtol=1e-12)
    result = {
        "scope": "Initial factual source only, 11 matched roots x 2 speeds, program duplicates removed; no subsequent write/H4/tail/learning/P4 test",
        "protocol": {"nu": NU, "finite_order": 32, "prior": "uniform over four prototypes",
                     "bounded_rho": 1.0, "bounded_lambda": 1.0, "noise": str(FIT),
                     "reference": "all K thin-SVD directions; direct reduced Gaussian variance; adaptive log-xi integral with independent fixed split check",
                     "rank_threshold": None, "fitting_or_node_selection": False},
        "checkpoints": {
            "p12_frozen": evaluate(OUT / "p12_tensors.pt", fitted),
            "p3_bounded_d": evaluate(OUT / "p3_bounded_d_evaluation_tensors.pt", fitted),
        },
    }
    output_path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: value["summary"] for name, value in result["checkpoints"].items()}, indent=2))


if __name__ == "__main__":
    main()
