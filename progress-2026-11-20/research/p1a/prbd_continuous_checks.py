"""G1 frozen continuous-law validation; no fitting, collection or node search.

Run with the loco interpreter: python -m research.p1a.prbd_continuous_checks.
The fixed contract was written before the first measurements in g1_lock.json.
"""

from .prbd_paths import ROOT as RELEASE_ROOT, EVIDENCE, CONSISTENCY, SPEED_DATA, output_path

from itertools import combinations
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import gammaln, logsumexp
import torch

from .prbd import assimilate, assimilate_continuous, continuous_quadrature, issue


ROOT = RELEASE_ROOT
OUT = CONSISTENCY
OLD = EVIDENCE
DATA = SPEED_DATA
LOCK = {
    "nu": 3., "geometry": "bounded rho=lambda=1", "reference": "fixed uniform branch mean",
    "log_xi_range": [-6., 40.], "log_xi_step": .25, "nodes": 185,
    "max_pair_log_bayes_factor_error_nat": 1e-4,
    "posterior_total_variation": 1e-5,
    "gradient_l2_error": "1e-4*(1+reference_gradient_l2), every tested pair and absolute branch",
    "max_absolute_branch_log_density_error_nat": 1e-4,
    "signed_h1_h4": "same signs as independent continuous reference, aggregated and per seed",
    "native_gradient_scope": "all six logBF pairs and all four absolute branches, full learnable offset gradients, 22 initial sources per archived checkpoint",
    "native_directional_controls": "three deterministic full-parameter directions for all pairs and absolute branches; reference gradient directional contraction",
    "selection_or_tuning": False,
}


def continuous_reference(responses, observation, noise):
    """Adaptive continuous logs and complete response derivatives.

    Independent NumPy thin SVD retains every direction. SciPy adaptively
    integrates Gaussian covariance scores as well as density. No fixed PyTorch
    nodes or PyTorch backwards are used. Missing log-xi tails outside [-12,100]
    have prior mass below 1e-64; actual observation errors are checked directly.
    """
    responses, observation, noise = map(np.asarray, (responses, observation, noise))
    k, d = responses.shape
    center = responses.mean(0)
    basis = np.linalg.solve(noise, (responses-center).T)
    x = np.linalg.solve(noise, observation-center)
    vectors, singular, right = np.linalg.svd(basis, full_matrices=False)
    raw = singular**2
    eigen = raw/(1+raw)
    coordinates = vectors.T @ (x[:, None]-basis)
    orthogonal = x-vectors@(vectors.T@x)
    common = -np.log(np.diag(noise)).sum()-.5*(d*math.log(2*math.pi)+orthogonal@orthogonal)
    constant = 1.5*math.log(1.5)-gammaln(1.5)
    m_coordinates = (singular/(1+raw))[:, None]*right

    def terms(t):
        scale = math.exp(t)
        variance = 1+scale*eigen
        projected = coordinates/variance[:, None]
        log_normal = -.5*(np.log(variance).sum()+(coordinates**2/variance[:, None]).sum(0))
        log_prior = constant-1.5*t-1.5*math.exp(-t)
        inverse_residual = orthogonal[:, None]+vectors@projected
        u_residual = orthogonal[:, None]+vectors@(projected/(1+raw)[:, None])
        residual_m = projected.T@m_coordinates
        u_inverse_m = vectors@((singular/(1+raw)**2)[:, None]*right/variance[:, None])
        covariance = scale*(u_residual.T[:, :, None]*residual_m[:, None, :]-u_inverse_m[None])
        covariance -= covariance.mean(-1, keepdims=True)
        derivative = covariance+inverse_residual.T[:, :, None]*np.eye(k)[:, None, :]
        return log_normal+log_prior, derivative

    pivot = np.max(np.stack([terms(t)[0] for t in np.arange(-6., 41., 1.)]), 0)

    def integrand(t):
        logs, derivative = terms(t)
        density = np.exp(logs-pivot)
        return np.concatenate((density[:, None],
                               (density[:, None, None]*derivative).reshape(k, -1)), 1)

    bounds = (-12., 0., 10., 20., 40., 100.)
    parts = [quad_vec(integrand, left, right, epsabs=2e-11, epsrel=2e-11,
                      norm="max", limit=400, quadrature="gk21")
             for left, right in zip(bounds[:-1], bounds[1:])]
    integrated = sum(part[0] for part in parts)
    density = integrated[:, 0]
    if np.any(density <= 0) or not np.isfinite(integrated).all():
        raise ArithmeticError("Continuous adaptive integration failed")
    reduced = pivot+np.log(density)
    grad_white = (integrated[:, 1:]/density[:, None]).reshape(k, d, k)
    gradients = np.stack([np.linalg.solve(noise.T, gradient).T for gradient in grad_white])
    return {"logs": common+reduced, "ratios": reduced-reduced[0],
            "response_gradients": gradients,
            "integration_error_bound": sum(part[1] for part in parts)/float(density.min())}


def objectives(result):
    return ([("absolute", [k], result.log_likelihood[k]) for k in range(len(result.posterior))]
            + [("logBF", [i, j], result.relative_log_likelihood[i]-result.relative_log_likelihood[j])
               for i, j in combinations(range(len(result.posterior)), 2)])


def check_arrays(responses, observation, noise, weights=None):
    responses = responses.detach().double().requires_grad_(True)
    observation, noise = observation.double(), noise.double()
    k = len(responses)
    if weights is None:
        weights = responses.new_full((k,), 1/k)
    signed = issue(responses, weights, torch.eye(k), noise, reference="uniform")
    result = assimilate_continuous(signed, observation)
    ref = continuous_reference(responses.detach().numpy(), observation.numpy(), noise.numpy())
    log_error = result.log_likelihood.detach().numpy()-ref["logs"]
    ratio_error = result.relative_log_likelihood.detach().numpy()-ref["ratios"]
    reference_posterior = torch.from_numpy(ref["ratios"]).add(weights.log()).softmax(0)
    gradient_rows = []
    for kind, indices, objective in objectives(result):
        actual = torch.autograd.grad(objective, responses, retain_graph=True)[0]
        expected = ref["response_gradients"][indices[0]]
        if kind == "logBF":
            expected = expected-ref["response_gradients"][indices[1]]
        expected = torch.from_numpy(expected)
        gradient_rows.append({"type": kind, "indices": indices,
                              "error_l2": float((actual-expected).norm()),
                              "reference_l2": float(expected.norm())})
    row = {"max_absolute_log_density_error": float(np.abs(log_error).max()),
           "max_pair_logBF_error": float(ratio_error.max()-ratio_error.min()),
           "posterior_TV": float(.5*(result.posterior-reference_posterior).abs().sum()),
           "integration_error_bound": float(ref["integration_error_bound"]), "gradients": gradient_rows}
    assert row["max_absolute_log_density_error"] <= 1e-4, row
    assert row["max_pair_logBF_error"] <= 1e-4, row
    assert row["posterior_TV"] <= 1e-5, row
    assert all(g["error_l2"] <= 1e-4*(1+g["reference_l2"]) for g in gradient_rows), row
    return row, result, ref


def synthetic_checks():
    generator = torch.Generator().manual_seed(20261120)
    rows = []
    for d in (1, 3, 7, 192):
        for amplitude in (0., 1e-8, .1, 1., 100., 1e4):
            b = torch.randn(4, d, generator=generator, dtype=torch.float64)*amplitude
            z = torch.randn(d, generator=generator, dtype=torch.float64)
            if amplitude >= 100:
                z = b[-1].clone()
            row, *_ = check_arrays(b, z, torch.eye(d, dtype=torch.float64))
            rows.append({"dimension": d, "amplitude": amplitude, **row})
    b = torch.tensor([[1., .3], [1., .3], [-.2, .1], [1.8, -.4]], dtype=torch.float64)
    w = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
    row, result, _ = check_arrays(b, torch.tensor([.6, .2]), torch.eye(2), w)
    assert abs(float(result.posterior[0]/result.posterior[1]-w[0]/w[1])) < 1e-12
    rows.append({"case": "repeated responses", **row})
    equal = b[:1].expand_as(b).clone()
    row, result, _ = check_arrays(equal, torch.tensor([.6, .2]), torch.eye(2), w)
    assert torch.equal(result.posterior, w)
    rows.append({"case": "all responses physically equal", **row})
    b = b.requires_grad_(True)
    def density(values):
        return assimilate_continuous(issue(values, w, torch.eye(4), torch.eye(2),
                                           reference="uniform"), torch.tensor([.6, .2])).log_likelihood
    assert torch.autograd.gradcheck(density, (b,), eps=1e-5, atol=1e-6, rtol=1e-5)
    assert torch.autograd.gradcheck(density, (equal.requires_grad_(True),), eps=1e-5, atol=1e-6, rtol=1e-5)
    laws, priors = [], []
    for prior in (w, torch.tensor([.4, .3, .2, .1], dtype=torch.float64)):
        for transition in (torch.eye(4), torch.full((4, 4), .25)):
            issued = issue(b, prior, transition, torch.eye(2), reference="uniform")
            laws.append(assimilate_continuous(issued, torch.tensor([.6, .2])).log_likelihood)
            priors.append(issued.prior)
    assert all(torch.equal(laws[0], law) for law in laws)
    assert not torch.equal(priors[0], priors[1])
    b3 = torch.cat((b.detach(), torch.zeros(4, 1)), dim=1)
    a = issue(b3, torch.tensor([.7, .3, 0., 0.]), torch.eye(4), torch.eye(3), reference="uniform")
    ordinary = assimilate_continuous(a, torch.tensor([.6, .2, 0.]))
    shifted = assimilate_continuous(a, torch.tensor([.6, .2, 1e6]))
    torch.testing.assert_close(ordinary.relative_log_likelihood, shifted.relative_log_likelihood, atol=1e-10, rtol=0)
    assert torch.equal(ordinary.posterior[2:], torch.zeros(2, dtype=torch.float64))
    return {"rows": rows, "finite_difference_gradcheck": True,
            "prior_independent_likelihood_bitwise": True, "transport_preserved_separately": True,
            "sparse_support_preserved": True,
            "orthogonal_logBF_error": float((ordinary.relative_log_likelihood-shifted.relative_log_likelihood).abs().max())}


def identity_prior_roundoff_checks():
    """Exact public-value identity with normalized probability derivatives."""
    from dataclasses import replace
    base = torch.tensor([.7, .2, .07, .03], dtype=torch.float64)
    priors = [base, base*torch.nextafter(base.new_tensor(1.), base.new_tensor(0.)),
              base*torch.nextafter(base.new_tensor(1.), base.new_tensor(2.))]
    rows = []
    for original in priors:
        public = original.clone().requires_grad_(True)
        transition = torch.eye(4, dtype=torch.float64, requires_grad=True)
        means = torch.tensor([[.3, -.2]]*4, dtype=torch.float64, requires_grad=True)
        signed = issue(means, public, transition, torch.eye(2), reference="uniform")
        arrival = assimilate_continuous(signed, torch.tensor([.6, .4]))
        assert torch.equal(signed.prior, public)
        assert torch.equal(arrival.posterior, public)
        normalized = transition/transition.sum(1, keepdim=True)
        baseline = normalized.T@(public/public.sum())
        baseline = baseline/baseline.sum()
        reference = assimilate_continuous(replace(signed, prior=baseline), torch.tensor([.6, .4]))
        actual_gradients = torch.autograd.grad(arrival.posterior[0], (public, transition, means), retain_graph=True)
        reference_gradients = torch.autograd.grad(reference.posterior[0], (public, transition, means), retain_graph=True)
        gradient_error = max(float((actual-expected).abs().max())
                             for actual, expected in zip(actual_gradients, reference_gradients, strict=True))
        assert gradient_error < 1e-14
        assert actual_gradients[2].norm() > 0
        legacy = issue(means, public, transition, torch.eye(2))
        assert torch.equal(legacy.prior, baseline)
        changed_transition = torch.full((4, 4), .25, dtype=torch.float64)
        transported = issue(means, public, changed_transition, torch.eye(2), reference="uniform")
        torch.testing.assert_close(transported.prior, torch.full((4,), .25, dtype=torch.float64), atol=0, rtol=0)
        rows.append({"input_sum": float(public.sum()), "supplied_prior_preserved_bitwise": True,
                     "posterior_preserved_bitwise": True, "normalized_derivative_error_max": gradient_error,
                     "ordinary_response_derivative_norm": float(actual_gradients[2].norm()),
                     "legacy_normalization_unchanged": True, "nonidentity_transport_preserved": True})
    return {"rows": rows, "passed": True}


def native_gradients(model, response, target, noise):
    reference = continuous_reference(response.detach().double().numpy(), target.numpy(), noise.numpy())
    result = assimilate_continuous(issue(response, noise.new_full((4,), .25),
                                         torch.eye(4), noise, reference="uniform"), target)
    generator = torch.Generator().manual_seed(20261120)
    directions = torch.randn(3, *model.offsets.shape, generator=generator)
    directions /= directions.flatten(1).norm(dim=1)[:, None, None, None]
    rows = []
    for kind, indices, objective in objectives(result):
        ref_gradient = reference["response_gradients"][indices[0]]
        if kind == "logBF":
            ref_gradient = ref_gradient-reference["response_gradients"][indices[1]]
        actual = torch.autograd.grad(objective, model.offsets, retain_graph=True)[0]
        expected = torch.autograd.grad(response, model.offsets,
                                        torch.from_numpy(ref_gradient).to(response), retain_graph=True)[0]
        error, norm = float((actual-expected).norm()), float(expected.norm())
        assert error <= 1e-4*(1+norm), (kind, indices, error, norm)
        rows.append({"type": kind, "indices": indices, "error_l2": error, "reference_l2": norm,
                     "directional_error_max": float(((directions*(actual-expected)).flatten(1).sum(1)).abs().max())})
    return rows


def archive_checks(name, tensors, fitted):
    from .prbd_native import NativePrototypes
    from .prbd_tworoom_speed_fit import load_host
    saved = torch.load(tensors, weights_only=True, map_location="cpu")
    noise = fitted["noise_cholesky"].double()
    model = NativePrototypes(load_host("cpu")).eval()
    model.offsets.data.copy_(fitted["offsets"])
    validation = torch.load(DATA/"encoded_validation.pt", weights_only=True, map_location="cpu")
    lookup = {int(seed): idx for idx, seed in enumerate(validation["seed"])}
    rows, outputs = [], {}
    for case in saved["cases"]:
        if not case["matched"]:
            continue
        assert torch.equal(case["source_responses"][:, 0], case["source_responses"][:, 1])
        for speed_index, speed in enumerate((2.5, 7.5)):
            response = case["source_responses"][speed_index, 0].double()
            target = case["source_actual_arrivals"][speed_index, 0].double()
            row, result, ref = check_arrays(response, target, noise)
            row.update(seed=case["seed"], speed=speed)
            outputs[case["seed"], speed] = (result.posterior.detach().numpy(),
                                            np.exp(ref["ratios"]-logsumexp(ref["ratios"])))
            legacy = assimilate(issue(response, response.new_full((4,), .25), torch.eye(4), noise),
                                target, fitted["scales"], fitted["masses"], geometry="bounded")
            index = saved["methods"].index("bounded_source_only")
            error = float((legacy.posterior-case["supports"][index, speed_index, 0]).abs().max())
            assert error < 1e-8
            row["legacy_Q32_archived_posterior_error_max"] = error
            idx = lookup[case["seed"]]
            z, a = validation["z"][idx, speed_index, 0], validation["a"][idx, speed_index, 0]
            actual = model(z[None, :3], a[None, :3])[0]
            torch.testing.assert_close(actual.double(), response, atol=2e-5, rtol=0)
            row["native_offset_gradients"] = native_gradients(model, actual, target, noise)
            rows.append(row)
            print(name, len(rows), case["seed"], speed, "passed", flush=True)
    signed = {"per_seed": []}
    numerators, reference_numerators, denominators = [], [], []
    for case in saved["cases"]:
        if not case["matched"]:
            continue
        slow, slow_ref = outputs[case["seed"], 2.5]
        fast, fast_ref = outputs[case["seed"], 7.5]
        family, truth = case["move_minus_stop"].double().numpy(), case["true_interaction"].double().numpy()
        prediction = slow@family[0].reshape(4, -1)-fast@family[1].reshape(4, -1)
        ref_prediction = slow_ref@family[0].reshape(4, -1)-fast_ref@family[1].reshape(4, -1)
        dot = (prediction.reshape(truth.shape)*truth).sum(1)
        ref_dot = (ref_prediction.reshape(truth.shape)*truth).sum(1)
        denom = (truth**2).sum(1)
        assert np.array_equal(np.sign(dot[[0, 3]]), np.sign(ref_dot[[0, 3]]))
        signed["per_seed"].append({"seed": case["seed"], "fixed_numerator_h1_h4": dot[[0, 3]].tolist(),
                                   "reference_numerator_h1_h4": ref_dot[[0, 3]].tolist()})
        numerators.append(dot)
        reference_numerators.append(ref_dot)
        denominators.append(denom)
    signed["fixed_signed_recovery_percent"] = (100*np.sum(numerators, 0)/np.sum(denominators, 0)).tolist()
    signed["reference_signed_recovery_percent"] = (100*np.sum(reference_numerators, 0)/np.sum(denominators, 0)).tolist()
    assert np.array_equal(np.sign(signed["fixed_signed_recovery_percent"]), np.sign(signed["reference_signed_recovery_percent"]))
    return {"rows": rows, "signed_recovery": signed}


def native_finite_differences(fitted):
    """Direct adaptive-reference differences on two real float64 sources.

    Three full parameter directions supplement all-source float32 full VJPs.
    Float64 avoids measuring float32 quantization instead of a derivative.
    """
    from .prbd_native import NativePrototypes
    from .prbd_tworoom_speed_fit import load_host
    validation = torch.load(DATA/"encoded_validation.pt", weights_only=True, map_location="cpu")
    idx = int(torch.where(validation["matched_rgb"])[0][0])
    model = NativePrototypes(load_host("cpu")).eval().double()
    model.offsets.data.copy_(fitted["offsets"].double())
    initial = model.offsets.detach().clone()
    generator = torch.Generator().manual_seed(20261120)
    directions = torch.randn(3, *initial.shape, generator=generator, dtype=torch.float64)
    directions /= directions.flatten(1).norm(dim=1)[:, None, None, None]
    rows = []
    for speed_index, speed in enumerate((2.5, 7.5)):
        z, a = (validation[key][idx, speed_index, 0].double() for key in ("z", "a"))
        response = model(z[None, :3], a[None, :3])[0]
        target, noise = z[3], fitted["noise_cholesky"].double()
        result = assimilate_continuous(issue(response, response.new_full((4,), .25),
                                             torch.eye(4), noise, reference="uniform"), target)
        actual_derivatives = []
        requested = objectives(result)
        for _, _, objective in requested:
            gradient = torch.autograd.grad(objective, model.offsets, retain_graph=True)[0]
            actual_derivatives.append((directions*gradient).flatten(1).sum(1).numpy())
        references = []
        epsilon = 1e-4
        for direction in directions:
            logs = []
            for sign in (-1, 1):
                with torch.no_grad():
                    model.offsets.copy_(initial+sign*epsilon*direction)
                    value = model(z[None, :3], a[None, :3])[0]
                logs.append(continuous_reference(value.numpy(), target.numpy(), noise.numpy())["logs"])
            references.append((logs[1]-logs[0])/(2*epsilon))
        model.offsets.data.copy_(initial)
        references = np.stack(references)
        for actual, (kind, indices, _) in zip(actual_derivatives, requested, strict=True):
            ref = references[:, indices[0]]
            if kind == "logBF":
                ref = ref-references[:, indices[1]]
            errors = np.abs(actual-ref)
            assert np.all(errors <= 1e-4*(1+np.abs(ref)))
            rows.append({"seed": int(validation["seed"][idx]), "speed": speed, "type": kind,
                         "indices": indices, "actual_derivatives": actual.tolist(),
                         "continuous_finite_difference": ref.tolist(), "absolute_errors": errors.tolist()})
    return {"epsilon": 1e-4, "dtype": "float64 native parameter and host calculations",
            "three_full_parameter_directions": True, "rows": rows}


def benchmark():
    fitted = torch.load(DATA/"fitted.pt", weights_only=True, map_location="cpu")
    saved = torch.load(OLD/"p12_tensors.pt", weights_only=True, map_location="cpu")
    case = next(case for case in saved["cases"] if case["matched"])
    responses = case["source_responses"][0, 0].double()
    target = case["source_actual_arrivals"][0, 0].double()
    nodes, masses = continuous_quadrature()
    loops = 100
    start = perf_counter()
    for _ in range(loops):
        with torch.no_grad():
            signed = issue(responses, responses.new_full((4,), .25), torch.eye(4),
                           fitted["noise_cholesky"], reference="uniform")
            assimilate_continuous(signed, target, nodes, masses)
    inference = (perf_counter()-start)/loops
    start = perf_counter()
    for _ in range(loops):
        value = responses.detach().requires_grad_(True)
        signed = issue(value, value.new_full((4,), .25), torch.eye(4),
                       fitted["noise_cholesky"], reference="uniform")
        result = assimilate_continuous(signed, target, nodes, masses)
        torch.autograd.grad(result.log_predictive, value)
    return {"D": 192, "K": 4, "Q": len(nodes), "loops": loops,
            "inference_seconds_per_issue_and_arrival": inference,
            "forward_backward_seconds_per_issue_and_arrival": (perf_counter()-start)/loops,
            "scope": "observation kernel only, no native forward, CPU one thread"}


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    lock_path = OUT/"g1_lock.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == LOCK, "G1 contract is locked"
    else:
        output_path(lock_path).write_text(json.dumps(LOCK, indent=2)+"\n", encoding="utf-8")
    started = perf_counter()
    report = {"lock": LOCK, "synthetic": synthetic_checks(),
              "identity_prior_roundoff": identity_prior_roundoff_checks(), "checkpoints": {}}
    for name, archive, checkpoint in (("p12_frozen", "p12_tensors.pt", DATA/"fitted.pt"),
                                       ("p3_bounded_d", "p3_bounded_d_evaluation_tensors.pt", OLD/"p3_bounded_d.pt")):
        fitted = torch.load(checkpoint, weights_only=True, map_location="cpu")
        report["checkpoints"][name] = archive_checks(name, OLD/archive, fitted)
        report["checkpoints"][name]["native_direct_finite_differences"] = native_finite_differences(fitted)
        (output_path(OUT/"g1_partial.json")).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    from .prbd_checks import numerical_checks
    report["legacy_synthetic"] = numerical_checks()
    report["benchmark"] = benchmark()
    report["seconds"], report["passed"] = perf_counter()-started, True
    (output_path(OUT/"g1.json")).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(OUT/"g1.json", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
