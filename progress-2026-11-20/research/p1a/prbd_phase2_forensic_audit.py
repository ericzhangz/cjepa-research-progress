"""Fixed-checkpoint audit of the archived Phase 2 response covariance.

This is a descriptive re-evaluation and local derivative decomposition.  It
does not fit a model or identify the causal contribution of a training path.
"""

import json
import math
from time import perf_counter

import torch
import numpy as np

from .prbd import continuous_quadrature
from .prbd_paths import REPAIR
from .prbd_phase2 import (OUT, episode_paths, fixed_transition,
                         load_episodes, load_phase2_checkpoint, response_density)


AUDIT = REPAIR / "audit"


class SplitDensity(torch.autograd.Function):
    """The archived normalized G1 density with separate covariance/residual inputs.

    d log N / d B = xi A^-1 (s s^T - C^-1) A^-1 B,
    A=I+BB^T, C=I+xi(I-A^-1), s=C^-1 r.  Quadrature
    responsibilities average the component scores.  No spectral derivatives.
    """

    @staticmethod
    def forward(ctx, basis, residual, scales, masses):
        n, d, k = basis.shape
        q, triangular = torch.linalg.qr(basis, mode="reduced")
        u_small, singular, right = torch.linalg.svd(triangular, full_matrices=False)
        vectors = q @ u_small
        raw = singular.square()
        variances = 1 + scales[None, :, None] * (raw / (1 + raw))[:, None]
        coordinates = vectors.transpose(1, 2) @ residual.transpose(1, 2)
        orthogonal = residual.transpose(1, 2) - vectors @ coordinates
        node_logs = -.5 * (variances.log().sum(-1)[:, :, None]
            + (coordinates[:, None].square() / variances[:, :, :, None]).sum(2))
        joint = masses.log()[None, :, None] + node_logs
        reduced = torch.logsumexp(joint, dim=1)
        responsibility = (joint - reduced[:, None]).exp()
        absolute = reduced - .5 * (d * math.log(2 * math.pi)
                                  + orthogonal.square().sum(1))
        if any(ctx.needs_input_grad[:2]):
            projected = coordinates[:, None] / variances[:, :, :, None]
            u_projected = projected / (1 + raw)[:, None, :, None]
            m_coordinates = (singular / (1 + raw))[:, :, None] * right
            residual_m = projected.transpose(2, 3) @ m_coordinates[:, None]
            u_inverse_m = ((singular / (1 + raw).square())[:, None, :, None]
                           * right[:, None] / variances[:, :, :, None])
            weight = responsibility * scales[None, :, None]
            coordinate_score = torch.einsum("nqj,nqrj,nqjl->njrl", weight, u_projected, residual_m)
            coordinate_score -= torch.einsum("nqj,nqrl->njrl", weight, u_inverse_m)
            score_basis = vectors[:, None] @ coordinate_score
            score_basis += orthogonal.transpose(1, 2)[:, :, :, None] * torch.einsum(
                "nqj,nqjl->njl", weight, residual_m)[:, :, None]
            mean_score = vectors @ (responsibility[:, :, None] * projected).sum(1)
            mean_score += orthogonal * responsibility.sum(1)[:, None]
            ctx.save_for_backward(score_basis, -mean_score.transpose(1, 2))
        return absolute

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, derivative):
        score_basis, score_residual = ctx.saved_tensors
        return ((derivative[:, :, None, None] * score_basis).sum(1),
                derivative[:, :, None] * score_residual, None, None)


def white_inputs(means, target, noise):
    means = means.double()
    # The uniform center is the archived G1 center, independent of belief.
    centered = means - means.mean(1, keepdim=True)
    basis = torch.linalg.solve_triangular(noise, centered.transpose(1, 2), upper=False)
    residual = torch.linalg.solve_triangular(
        noise, (target.double()[:, None] - means).transpose(1, 2), upper=False).transpose(1, 2)
    return basis, residual


def density(basis, residual, noise):
    scales, masses = continuous_quadrature()
    return SplitDensity.apply(basis, residual, scales.to(basis), masses.to(basis)) - noise.diagonal().log().sum()


def gaussian_logs(residual, noise):
    return -.5 * (residual.size(-1) * math.log(2 * math.pi)
                  + residual.square().sum(-1)) - noise.diagonal().log().sum()


def independent_dense_check():
    """Independent full-dimensional solve, including rank-degenerate B."""
    torch.manual_seed(20261121)
    errors, derivative_errors = [], []
    for kind in ("random", "repeated", "zero"):
        basis = torch.randn(2, 7, 4, dtype=torch.float64)
        if kind == "repeated":
            basis[:, :, 1:] = basis[:, :, :1]
        elif kind == "zero":
            basis.zero_()
        basis.requires_grad_()
        residual = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
        scales, masses = continuous_quadrature()
        split = SplitDensity.apply(basis, residual, scales, masses)
        p = basis @ torch.linalg.solve(torch.eye(4, dtype=basis.dtype) + basis.transpose(1, 2) @ basis,
                                      basis.transpose(1, 2))
        covariance = torch.eye(7, dtype=basis.dtype)[None, None] + scales[None, :, None, None] * p[:, None]
        solved = torch.linalg.solve(covariance, residual.transpose(1, 2)[:, None])
        quadratic = (residual.transpose(1, 2)[:, None] * solved).sum(2)
        log = -.5 * (7 * math.log(2 * math.pi) + torch.linalg.slogdet(covariance)[1][:, :, None] + quadratic)
        dense = torch.logsumexp(masses.log()[None, :, None] + log, 1)
        probe = torch.arange(1, 9, dtype=basis.dtype).reshape(2, 4) / 8
        grad_a = torch.autograd.grad((split * probe).sum(), (basis, residual), retain_graph=True)
        grad_b = torch.autograd.grad((dense * probe).sum(), (basis, residual))
        errors.append(float((split - dense).abs().max().detach()))
        for a, b in zip(grad_a, grad_b, strict=True):
            derivative_errors.append(float((a-b).abs().max()))
            torch.testing.assert_close(a, b, atol=2e-8, rtol=2e-7)
        torch.testing.assert_close(split, dense, atol=2e-8, rtol=2e-9)
    return {"reference": "Dense D-dimensional covariance solve and slogdet, ordinary autograd",
            "cases": ["random", "repeated", "zero"], "max_log_density_error": max(errors),
            "max_first_derivative_error": max(derivative_errors)}


def objective_from_logs(absolute, separation, keys, paths, episodes, cal, *, detach_prefix=False):
    """Literal archived equal-episode/root/horizon mixture and actual-prefix law."""
    densities, cursor = {}, 0
    for key in keys:
        h = paths[key][0].size(1)
        densities[key] = (absolute[cursor:cursor+h], separation[cursor:cursor+h])
        cursor += h
    per_episode = []
    for e, (_, z, a) in enumerate(episodes):
        belief = absolute.new_full((4,), .25)
        terms = []
        for root in range(2, len(a)):
            logs, sep = densities[e, root]
            prior = belief @ cal["transition"]
            prior = prior / prior.sum()
            joint = logs.cumsum(0) + prior.log()[None]
            divisor = torch.arange(1, len(logs) + 1, dtype=logs.dtype, device=logs.device) * z.size(-1)
            terms.append((-torch.logsumexp(joint, -1) / divisor).mean())
            updated = (prior.log() + logs[0] - logs[0, 0]).softmax(0)
            belief = torch.where(sep[0] <= cal["deadzone"], prior, updated)
            if detach_prefix:
                belief = belief.detach()
        per_episode.append(torch.stack(terms).mean())
    return torch.stack(per_episode).mean()


def describe(x):
    x = x.detach().double().flatten()
    return {"mean": float(x.mean()), "minimum": float(x.min()),
            "median": float(x.median()), "p95": float(torch.quantile(x, .95)),
            "maximum": float(x.max()), "positive": int((x > 0).sum()), "count": len(x)}


def vector_report(gradients):
    total = gradients["total"]
    out = {}
    for name, g in gradients.items():
        dot = float((g.double() * total.double()).sum())
        out[name] = {"L2": float(g.norm()), "per_prototype_L2": g.flatten(1).norm(dim=1).tolist(),
                     "dot_total": dot, "signed_fraction_total_squared_norm": dot / float(total.double().square().sum())}
    return out


def native_audit(model, cal, episodes):
    paths = episode_paths(model, episodes)
    keys = list(paths)
    means = torch.cat([paths[key][0].transpose(0, 1) for key in keys])
    targets = torch.cat([paths[key][1] for key in keys])
    archived, _, separation = response_density(means, targets, cal["noise_cholesky"])
    basis, residual = white_inputs(means, targets, cal["noise_cholesky"])
    b_proxy = basis.detach().requires_grad_()
    r_proxy = residual.detach().requires_grad_()
    split = density(b_proxy, r_proxy, cal["noise_cholesky"])
    value_error = float((split - archived).abs().max().detach())
    torch.testing.assert_close(split, archived, atol=2e-9, rtol=2e-10)
    original = objective_from_logs(archived, separation, keys, paths, episodes, cal)
    split_loss = objective_from_logs(split, separation.detach(), keys, paths, episodes, cal)
    g_b, g_r = torch.autograd.grad(split_loss, (b_proxy, r_proxy), retain_graph=True)
    local_loss = objective_from_logs(split, separation.detach(), keys, paths, episodes, cal, detach_prefix=True)
    g_b_local, g_r_local = torch.autograd.grad(local_loss, (b_proxy, r_proxy))
    # Common-scale radial decomposition is unique even at repeated singular values.
    coefficient = (g_b * basis.detach()).sum((1, 2), keepdim=True) / basis.detach().square().sum((1, 2), keepdim=True)
    radial = coefficient * basis.detach()
    # Singular-amplitude versus orientation split; the three nonzero singular
    # values are reported so any near-degeneracy remains visible.
    u, s, vh = torch.linalg.svd(basis.detach(), full_matrices=False)
    local = u.transpose(1, 2) @ g_b @ vh.transpose(1, 2)
    amplitude = u @ torch.diag_embed(local.diagonal(dim1=-2, dim2=-1)) @ vh
    grads = {}
    def native_grad(value, score):
        return torch.autograd.grad(value, model.offsets, score, retain_graph=True)[0]
    grads["mean"] = native_grad(residual, g_r)
    grads["covariance"] = native_grad(basis, g_b)
    grads["covariance_common_scale"] = native_grad(basis, radial)
    grads["covariance_shape_orientation_after_common_scale"] = native_grad(basis, g_b - radial)
    grads["covariance_singular_amplitude"] = native_grad(basis, amplitude)
    grads["covariance_orientation"] = native_grad(basis, g_b - amplitude)
    grads["direct_suffix_mean"] = native_grad(residual, g_r_local)
    grads["direct_suffix_covariance"] = native_grad(basis, g_b_local)
    grads["historical_prefix_mean"] = grads["mean"] - grads["direct_suffix_mean"]
    grads["historical_prefix_covariance"] = grads["covariance"] - grads["direct_suffix_covariance"]
    repair_logs, _, _ = response_density(means, targets, cal["noise_cholesky"],
                                        freeze_covariance_eigenvalues=True)
    repair_loss = objective_from_logs(repair_logs, separation, keys, paths, episodes, cal)
    repair_gradient, = torch.autograd.grad(repair_loss, model.offsets, retain_graph=True)
    repair_expected = grads["mean"] + grads["covariance_orientation"]
    torch.testing.assert_close(repair_gradient, repair_expected, atol=3e-6, rtol=3e-4)
    grads["total"], = torch.autograd.grad(original, model.offsets)
    recomposition_error = float((grads["mean"] + grads["covariance"] - grads["total"]).abs().max())
    torch.testing.assert_close(grads["mean"] + grads["covariance"], grads["total"], atol=3e-6, rtol=3e-4)
    with torch.no_grad():
        zero = gaussian_logs(residual, cal["noise_cholesky"])
        zero_objective = objective_from_logs(zero, separation, keys, paths, episodes, cal)
    first_indices, cursor = [], 0
    for key in keys:
        first_indices.append(cursor)
        cursor += paths[key][0].size(1)
    result = {"nll_per_dimension": float(original.detach()),
              "same_means_frozen_P_zero_nll_per_dimension": float(zero_objective),
              "density_value_max_abs_error_vs_archived": value_error,
              "native_gradient_recomposition_max_abs_error": recomposition_error,
              "repair_endpoint_gradient_check": {
                  "freeze_eigenvalue_gradient_L2": float(repair_gradient.norm()),
                  "mean_plus_covariance_orientation_L2": float(repair_expected.norm()),
                  "max_absolute_error": float((repair_gradient-repair_expected).abs().max())},
              "native_gradients": vector_report(grads),
              "singular_values": [describe(s[:, j]) for j in range(s.size(1))],
              "nominal_log_density_G1_minus_P_zero": describe(archived[:, 0] - zero[:, 0]),
              "actual_H1_nominal_log_density_G1_minus_P_zero": describe(archived[first_indices, 0] - zero[first_indices, 0]),
              "all_branch_log_density_G1_minus_P_zero": describe(archived - zero),
              "read_roots": len(keys), "suffix_targets": len(means)}
    frozen = {"means": means.detach().cpu(), "targets": targets.detach().cpu(),
              "basis": basis.detach().cpu(), "residual": residual.detach().cpu(),
              "separation": separation.detach().cpu(), "gradient": {k: v.detach().cpu() for k, v in grads.items()}}
    return result, frozen


@torch.no_grad()
def static_roots(model, cal, batch):
    z, a = batch["z"], batch["a"]
    with np.load(OUT / "diag_static.npz") as data:
        positions = torch.from_numpy(data["positions"].copy())
    matched_seeds = json.loads((OUT / "static_matched_roots.json").read_text(encoding="utf-8"))["1"]["seed"]
    matched = torch.tensor([int(seed) in matched_seeds for seed in batch["seed"]])
    records, tensors = [], []
    for t in range(2, 6):
        means = model(z[..., t-2:t+1, :].reshape(-1, 3, 192), a[..., t-2:t+1, :].reshape(-1, 3, 192))
        target = z[..., t+1, :].reshape(-1, 192)
        basis, residual = white_inputs(means, target, cal["noise_cholesky"])
        g1, _, sep = response_density(means, target, cal["noise_cholesky"])
        zero = gaussian_logs(residual, cal["noise_cholesky"])
        event_type = "source_action" if t < 4 else "stationary_arrival"
        g1 = g1.reshape(*z.shape[:3], 4)[:, :, 0]
        zero = zero.reshape(*z.shape[:3], 4)[:, :, 0]
        sep = sep.reshape(*z.shape[:3])[:, :, 0]
        valid = batch["lengths"][:, :, 0] > t
        if t >= 4:
            valid &= (positions[:, :, 0, 5*t:5*(t+1)+1]
                      == positions[:, :, 0, 5*t:5*t+1]).flatten(2).all(-1)
        scoped = {}
        for scope, mask in (("all_valid", valid), ("strict_matched", valid & matched[:, None])):
            logs, fixed = g1[mask], zero[mask]
            scoped[scope] = {
                "nominal_log_G1_minus_P_zero": describe(logs[:, 0] - fixed[:, 0]),
                "response_separation": describe(sep[mask]),
                "log_likelihood_spread": describe(logs.max(-1).values - logs.min(-1).values),
                "G1_nominal_log_density": describe(logs[:, 0]),
                "P_zero_nominal_log_density": describe(fixed[:, 0])}
        records.append({"root": t, "kind": event_type, "scopes": scoped})
        tensors.append({"root": t, "seed": batch["seed"], "G1_logs": g1.cpu(),
                        "P_zero_logs": zero.cpu(), "separation": sep.cpu(),
                        "all_valid": valid, "strict_matched": valid & matched[:, None]})
    return records, tensors


def main():
    AUDIT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = perf_counter()
    episodes = load_episodes(OUT / "train.pt")[:4]
    batch = torch.load(OUT / "static.pt", map_location="cpu", weights_only=True)
    report = {"scope": "Fixed archived original checkpoints, CPU, no optimizer updates or parameter search",
              "seeds": [r[0] for r in episodes], "checkpoints": {},
              "independent_dense_validation": independent_dense_check()}
    frozen, cals = {}, {}
    for name in ("initial", "final"):
        print(f"Auditing {name}", flush=True)
        model, cal = load_phase2_checkpoint(OUT / "A" / f"{name}.pt", "cpu")
        result, native = native_audit(model, cal, episodes)
        result["static_roots"], static_tensors = static_roots(model, cal, batch)
        report["checkpoints"][name] = result
        frozen[name], cals[name] = native, cal
        torch.save(native, AUDIT / f"original_{name}_fixed_tensors.pt")
        torch.save(static_tensors, AUDIT / f"original_{name}_static_density.pt")
        (AUDIT / "original_checkpoint_audit.partial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        del model
    assert torch.equal(cals["initial"]["noise_cholesky"], cals["final"]["noise_cholesky"])
    cal = cals["initial"]
    paths = {(e, root): (torch.empty(4, min(4, len(a)-root), 192), None)
             for e, (_, _, a) in enumerate(episodes) for root in range(2, len(a))}
    keys = list(paths)
    report["nominal_initial_final_mean_max_abs_difference"] = float(
        (frozen["initial"]["means"][:, 0] - frozen["final"]["means"][:, 0]).abs().max())
    report["same_frozen_means_covariance_swap"] = {}
    for mean_name in ("initial", "final"):
        record = frozen[mean_name]
        swaps = {}
        for covariance_name in ("initial", "final"):
            log = density(frozen[covariance_name]["basis"], record["residual"], cal["noise_cholesky"])
            score = objective_from_logs(log, record["separation"], keys, paths, episodes, cal)
            swaps[covariance_name] = {"nominal_log_density": describe(log[:, 0]),
                "all_branch_log_density": describe(log), "mixture_prefix_nll_per_dimension": float(score)}
        zero = gaussian_logs(record["residual"], cal["noise_cholesky"])
        score = objective_from_logs(zero, record["separation"], keys, paths, episodes, cal)
        swaps["P_zero"] = {"nominal_log_density": describe(zero[:, 0]),
            "all_branch_log_density": describe(zero), "mixture_prefix_nll_per_dimension": float(score)}
        report["same_frozen_means_covariance_swap"][mean_name] = swaps
    history = json.loads((OUT / "A/objective_history.json").read_text(encoding="utf-8"))
    report["archived_full_training_composite_nll"] = {"initial": history[0]["nll_per_dimension"],
        "final": history[-1]["nll_per_dimension"], "change": history[-1]["nll_per_dimension"] - history[0]["nll_per_dimension"],
        "meaning": "Equal episode/root/H normalized fixed-k suffix mixture, recursively fitted actual-prefix belief; covariance, branch means and prefix all contribute"}
    old_static = json.loads((OUT / "static_results.json").read_text(encoding="utf-8"))
    report["archived_static_strongest_write_scope"] = {
        "strongest_TV_event": old_static["probes"]["trained"]["matched"]["strongest_write"],
        "matched_stationary_response_separation": old_static["probes"]["trained"]["matched"]["statistics"]["separation"],
        "interpretation": "5.21924 is response separation at the largest-TV event; the response-separation maximum is 6.69630"}
    report["seconds"] = perf_counter() - started
    report["limitations"] = ["Local derivatives and fixed-checkpoint swaps do not identify training causes.",
        "Mean/covariance parameter gradients share the same offsets and can cancel; L2 norms are not additive contributions.",
        "Singular-amplitude/orientation split uses the fixed local SVD; a repeated nonzero singular value would make this split nonunique.",
        "Common-scale radial/rest split is well defined without a spectral gap."]
    (AUDIT / "original_checkpoint_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (AUDIT / "original_checkpoint_audit.partial.json").unlink()
    print(json.dumps({"seconds": report["seconds"], "report": str(AUDIT / "original_checkpoint_audit.json")}), flush=True)


if __name__ == "__main__":
    main()
