"""Phase 1 fixed-response comparison: shared full-latent t3 versus G3 law.

No fitting, new environment histories, independent confirmation, or planning.
"""

import json
import math
from time import perf_counter

import numpy as np
import torch
from scipy.stats import multivariate_t

from .prbd import (assimilate_continuous, assimilate_shared_student_t_nu3,
                   continuous_quadrature, issue)
from .prbd_native import NativePrototypes
from .prbd_tworoom_speed_fit import load_host
from .prbd_paths import ROOT, output_path


HISTORY = ROOT / "runs/prbd_consistency_20261120"
OUT = ROOT / "runs/prbd_phase1_20261121"
LAWS = {"historical_bounded_185": assimilate_continuous,
        "candidate_shared_student_t_nu3": assimilate_shared_student_t_nu3}
COMPARISONS = {"single_vs_uniform": ("single_source", "uniform"),
               "all_vs_uniform": ("all_arrivals", "uniform"),
               "all_vs_omit": ("all_arrivals", "omit_source")}


def evaluate(law, means, observation, prior, noise):
    signed = issue(means, prior, torch.eye(len(prior), dtype=torch.float64),
                   noise, reference="uniform")
    return LAWS[law](signed, observation)


def structural_checks():
    torch.manual_seed(20261121)
    errors = []
    for d in (1, 2, 7, 192):
        means = torch.randn(4, d, dtype=torch.float64)
        lower = torch.eye(d, dtype=torch.float64)*1.3
        if d > 1:
            lower[1, 0] = .2
        observation = torch.randn(d, dtype=torch.float64)
        prior = torch.tensor([.1, .2, .3, .4], dtype=torch.float64)
        out = evaluate("candidate_shared_student_t_nu3", means, observation, prior, lower)
        reference = np.array([multivariate_t.logpdf(observation.numpy(),
            loc=mean.numpy(), shape=(lower@lower.T).numpy(), df=3) for mean in means])
        errors.append(float(np.max(np.abs(reference-out.log_likelihood.numpy()))))
        np.testing.assert_allclose(out.log_likelihood.numpy(), reference, atol=1e-10, rtol=0)
        np.testing.assert_allclose(out.relative_log_likelihood.numpy(), reference-reference[0], atol=1e-10, rtol=0)
    means = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    observation = torch.randn(5, dtype=torch.float64, requires_grad=True)
    prior = torch.tensor([.0, .2, .3, .5], dtype=torch.float64)
    noise = torch.eye(5, dtype=torch.float64)
    def objective(m, y):
        result = evaluate("candidate_shared_student_t_nu3", m, y, prior, noise)
        return torch.cat((result.log_likelihood, result.relative_log_likelihood,
                          result.posterior, result.log_predictive[None]))
    assert torch.autograd.gradcheck(objective, (means, observation), atol=1e-5, rtol=1e-4)
    result = evaluate("candidate_shared_student_t_nu3", means, observation, prior, noise)
    assert result.posterior[0] == 0
    identical = means.detach()[:1].expand_as(means).clone().requires_grad_()
    same = evaluate("candidate_shared_student_t_nu3", identical, observation, prior, noise)
    assert torch.equal(same.posterior, prior)
    assert torch.isfinite(torch.autograd.grad(same.log_predictive, identical)[0]).all()
    transition = .7*torch.eye(4, dtype=torch.float64)+.3*torch.eye(4, dtype=torch.float64).roll(1, 1)
    transported = issue(identical, prior, transition, noise, reference="uniform")
    no_write = assimilate_shared_student_t_nu3(transported, observation)
    assert torch.equal(no_write.posterior, transported.prior)
    paired = means.detach().clone()
    paired[2] = paired[1]
    same_pair = evaluate("candidate_shared_student_t_nu3", paired, observation, prior, noise)
    assert abs(float(same_pair.posterior[1]/same_pair.posterior[2])-2/3) < 1e-14
    permutation = torch.tensor([2, 0, 3, 1])
    permuted = evaluate("candidate_shared_student_t_nu3", means[permutation], observation,
                        prior[permutation], noise)
    torch.testing.assert_close(permuted.posterior, result.posterior[permutation], atol=1e-14, rtol=0)
    changed_prior = evaluate("candidate_shared_student_t_nu3", means, observation,
                            prior.flip(0), noise)
    torch.testing.assert_close(changed_prior.log_likelihood, result.log_likelihood, atol=0, rtol=0)
    return {"scipy_absolute_logpdf_max_error": max(errors),
            "value_and_first_gradient_check": True, "equal_response_exact_no_write": True,
            "equal_pair_preserves_odds": True, "zero_prior_support_preserved": True,
            "prototype_permutation_equivariance": True,
            "density_independent_of_public_prior": True,
            "zero_response_preserves_transported_prior_exactly": True,
            "degenerate_response_gradients_finite": True}


def controlled_checks():
    d = 192
    means = torch.zeros(2, d, dtype=torch.float64)
    means[:, 0] = torch.tensor([-1., 1.])
    prior, noise = torch.full((2,), .5, dtype=torch.float64), torch.eye(d, dtype=torch.float64)
    orthogonal = []
    for size in (0., 3., 30., 300., 3000.):
        observation = torch.zeros(d, dtype=torch.float64)
        observation[0], observation[1] = 2., size
        outputs = {name: evaluate(name, means, observation, prior, noise) for name in LAWS}
        orthogonal.append({"orthogonal_residual": size, **{name: {
            "logBF_1_vs_0": float(value.relative_log_likelihood[1]),
            "support_1": float(value.posterior[1])} for name, value in outputs.items()}})
    base = orthogonal[0]["historical_bounded_185"]["logBF_1_vs_0"]
    assert max(abs(row["historical_bounded_185"]["logBF_1_vs_0"]-base) for row in orthogonal) < 1e-10
    assert orthogonal[-1]["candidate_shared_student_t_nu3"]["support_1"] < .501
    tail = []
    for size in (2., 10., 100., 1e3, 1e4, 1e6, 1e8, 1e12):
        observation = torch.zeros(d, dtype=torch.float64)
        observation[0] = size
        outputs = {name: evaluate(name, means, observation, prior, noise) for name in LAWS}
        tail.append({"response_direction_residual": size, **{name: float(
            value.relative_log_likelihood[1]) for name, value in outputs.items()},
            "historical_185_direct_absolute_log_difference": float(
                outputs["historical_bounded_185"].log_likelihood[1]
                -outputs["historical_bounded_185"].log_likelihood[0])})
    initial = torch.tensor([.1, .9], dtype=torch.float64)
    supports = {law: initial.clone() for law in LAWS}
    reversal, first_flip = [], {law: None for law in LAWS}
    for index, location in enumerate((1., -1., -1., 1.), start=1):
        observation = torch.zeros(d, dtype=torch.float64)
        observation[0] = location
        event = {"arrival": index, "observation_first_coordinate": location, "laws": {}}
        for law in LAWS:
            before = supports[law]
            result = evaluate(law, means, observation, before, noise)
            supports[law] = result.posterior
            flipped = bool(result.posterior[0] > .5)
            if flipped and first_flip[law] is None:
                first_flip[law] = index
            event["laws"][law] = {"before": before.tolist(), "after": result.posterior.tolist(),
                "logBF_0_vs_1": -float(result.relative_log_likelihood[1]),
                "supports_branch_0_above_half": flipped}
        reversal.append(event)
    observation = torch.zeros(d, dtype=torch.float64)
    observation[0] = -1.
    single = {}
    for law in LAWS:
        result = evaluate(law, means, observation, initial, noise)
        single[law] = {"same_initial_support": initial.tolist(), "posterior": result.posterior.tolist(),
            "logBF_0_vs_1": -float(result.relative_log_likelihood[1]),
            "required_logBF_to_cross_half": math.log(9),
            "crosses_half": bool(result.posterior[0] > .5)}
    assert single["candidate_shared_student_t_nu3"]["crosses_half"]
    assert not single["historical_bounded_185"]["crosses_half"]
    # This is an analytic witness, not an empirical parameter sweep.
    sigma, _ = continuous_quadrature()
    largest_variance = float(1+sigma[-1]*(2/3))
    gradient_bound = (d+3)/(2*math.sqrt(3))
    score_checks = []
    for radius in (0., math.sqrt(3), 10., 1e8):
        observation = torch.zeros(d, dtype=torch.float64)
        observation[0] = radius
        observation.requires_grad_()
        result = evaluate("candidate_shared_student_t_nu3", means[:1]*0, observation, prior[:1]*2, noise)
        gradient = torch.autograd.grad(result.log_likelihood[0], observation)[0]
        score_checks.append({"radius": radius, "score_l2": float(gradient.norm())})
        assert float(gradient.norm()) <= gradient_bound+1e-12
    return {"response_orthogonal_counterexample": orthogonal,
            "fixed_response_tail_range": tail,
            "controlled_counterevidence_sequence": reversal,
            "controlled_first_branch_0_majority_arrival": first_flip,
            "single_counterevidence_from_fixed_0_9": single,
            "controlled_sequence_is_not_a_real_environment_reversal": True,
            "t3_whitened_log_density_gradient_bound": gradient_bound,
            "measured_whitened_score_bound_checks": score_checks,
            "fixed_185_far_tail_dominant_node_logBF_slope": 2/largest_variance,
            "fixed_185_tail_limit": "For means -1,+1, logBF ~ 2*x/(1+xi_max*2/3), not zero.",
            "fixed_185_tail_precision": "At extreme synthetic residuals the historical relative-score subtraction can round small logBF to zero; direct absolute-log differences are retained. These residuals exceed the historical audited maximum 147.4014. No historical kernel change is made.",
            "candidate_tail_limit": "For every fixed finite pair of means, logBF tends to zero in every direction.",
            "boundedness": "Shared t3 log density has bounded whitened residual score; posterior remains in the simplex."}


def contrast(path, horizon):
    return path[0, 0, horizon]-path[0, 1, horizon]-path[1, 0, horizon]+path[1, 1, horizon]


@torch.no_grad()
def historical_replay(label, batch):
    fitted = torch.load(HISTORY/f"g3_{label}.pt", weights_only=True, map_location="cpu")
    saved = torch.load(HISTORY/f"g3_{label}_measure_tensors.pt", weights_only=True, map_location="cpu")
    archived = json.loads((HISTORY/f"g3_{label}_measure.json").read_text())
    metadata = {row["seed"]: row for row in archived["seeds"]}
    lookup = {int(seed): i for i, seed in enumerate(batch["seed"])}
    model = NativePrototypes(load_host("cpu"), null_control=batch["null_control"]).eval().requires_grad_(False)
    model.offsets.copy_(fitted["offsets"])
    noise, uniform = fitted["noise_cholesky"].double(), torch.full((4,), .25, dtype=torch.float64)
    rows, max_error, null_count = [], 0., 0
    for case in saved["cases"]:
        seed, supports, events = case["seed"], {law: {} for law in LAWS}, []
        for law in LAWS:
            supports[law] = {name: torch.empty(2, 4, dtype=torch.float64)
                             for name in ("single_source", "all_arrivals", "omit_source")}
            supports[law]["uniform"] = uniform.expand(2, -1)
        for speed in (0, 1):
            z, actions = batch["z"][lookup[seed], speed, 0], batch["a"][lookup[seed], speed, 0]
            states = {law: {"all_arrivals": uniform.clone(), "omit_source": uniform.clone()} for law in LAWS}
            for step in range(2, 6):
                means = model(z[None, step-2:step+1], actions[None, step-2:step+1])[0]
                old = metadata[seed]["actual_write_traces"][speed][step-2]["weights"]["all_arrivals"]
                archived_prior = torch.tensor(old["before"], dtype=torch.float64)
                fixed = {law: evaluate(law, means, z[step+1], archived_prior, noise) for law in LAWS}
                discrepancy = fixed["historical_bounded_185"].relative_log_likelihood-torch.tensor(
                    old["relative_log_likelihood"], dtype=torch.float64)
                max_error = max(max_error, float(discrepancy.abs().max()))
                assert float(discrepancy.abs().max()) < 1e-8
                event = {"speed_index": speed, "action_t": step, "arrival_index": step+1,
                         "event": {2: "source_action", 3: "return_action_same_physical_speed",
                                   4: "native_null", 5: "native_null"}[step],
                         "same_archived_prior": archived_prior.tolist(),
                         "fixed_prior_density_comparison": {law: {
                             "relative_log_likelihood": value.relative_log_likelihood.tolist(),
                             "posterior": value.posterior.tolist(),
                             "log_predictive": float(value.log_predictive)} for law, value in fixed.items()},
                         "sequential": {}}
                for law in LAWS:
                    event["sequential"][law] = {}
                    for chronology in ("all_arrivals", "omit_source"):
                        before = states[law][chronology]
                        value = evaluate(law, means, z[step+1], before, noise)
                        after = before if chronology == "omit_source" and step == 2 else value.posterior
                        if step in (4, 5):
                            assert torch.equal(after, before)
                            null_count += 1
                        states[law][chronology] = after
                        event["sequential"][law][chronology] = {"before": before.tolist(), "after": after.tolist(),
                            "update_TV": float(.5*(after-before).abs().sum())}
                    if step == 2:
                        supports[law]["single_source"][speed] = states[law]["all_arrivals"]
                events.append(event)
            for law in LAWS:
                for chronology in ("all_arrivals", "omit_source"):
                    supports[law][chronology][speed] = states[law][chronology]
        path = case["future_responses"].double()
        common = path[:, :, 0]
        truth = case["true_future"].double()
        predictions = {law: {name: common+torch.einsum("sk,spkhd->sphd", weights, path-common[:, :, None])
                            for name, weights in methods.items()} for law, methods in supports.items()}
        for name in supports["historical_bounded_185"]:
            torch.testing.assert_close(supports["historical_bounded_185"][name], case["supports"][name], atol=1e-12, rtol=0)
        horizons = {}
        for horizon in (0, 3):
            actual = contrast(truth, horizon)
            laws = {}
            for law, methods in predictions.items():
                comparisons = {}
                for name, (after, before) in COMPARISONS.items():
                    a, b = contrast(methods[after], horizon), contrast(methods[before], horizon)
                    comparisons[name] = {"directed_numerator": float((a-b)@actual),
                        "interaction_error_gain": float((b-actual).square().sum()-(a-actual).square().sum()),
                        "endpoint_mse_gain": float(((methods[before][:, :, horizon]-truth[:, :, horizon]).square()
                            -(methods[after][:, :, horizon]-truth[:, :, horizon]).square()).mean())}
                laws[law] = comparisons
            horizons[str(horizon+1)] = {"truth_squared_norm": float(actual.square().sum()), "laws": laws}
        rows.append({"seed": seed, "physical_nonzero": metadata[seed]["horizons"]["1"]["physical_nonzero"],
            "events": events, "supports": {law: {name: weight.tolist() for name, weight in methods.items()}
                for law, methods in supports.items()}, "horizons": horizons})
    panels = {}
    for panel, selected in (("all_matched", rows), ("physical_nonzero", [r for r in rows if r["physical_nonzero"]])):
        aggregate = {}
        for h in ("1", "4"):
            denominator = sum(row["horizons"][h]["truth_squared_norm"] for row in selected)
            aggregate[h] = {law: {name: {
                "directed_recovery": sum(row["horizons"][h]["laws"][law][name]["directed_numerator"] for row in selected)/denominator,
                "interaction_error_gain_sum": sum(row["horizons"][h]["laws"][law][name]["interaction_error_gain"] for row in selected),
                "interaction_error_improved_seeds": sum(row["horizons"][h]["laws"][law][name]["interaction_error_gain"] > 0 for row in selected),
                "endpoint_mse_gain_mean": sum(row["horizons"][h]["laws"][law][name]["endpoint_mse_gain"] for row in selected)/len(selected)}
                for name in COMPARISONS} for law in LAWS}
        panels[panel] = {"reset_count": len(selected), "horizons": aggregate}
    return {"rows": rows, "panels": panels, "archived_logBF_reproduction_max_error": max_error,
            "exact_null_checks": null_count, "actual_null_arrivals": len(rows)*4,
            "real_environment_hidden_condition_reversals": 0,
            "interpretation": "Frozen-weight observation-law substitution; weights were trained with historical law. Not a retrained candidate or independent capability test."}


def render(report):
    controlled = report["controlled_checks"]
    old = controlled["single_counterevidence_from_fixed_0_9"]["historical_bounded_185"]
    candidate = controlled["single_counterevidence_from_fixed_0_9"]["candidate_shared_student_t_nu3"]
    lines = ["# Phase 1: shared full-latent Student-t observation candidate", "",
        "The candidate uses exactly the historical native means, fixed whitening L, and initial public support. Its scale is R=LL^T; nu=3 gives covariance 3R. There is no temperature or covariance matching. Historical training and probability defaults are unchanged.", "",
        "For r_k=L^-1(z-f_k), q_k=||r_k||^2 and D=192, log l_k=lgamma((D+3)/2)-lgamma(3/2)-(D/2)log(3*pi)-log|L|-((D+3)/2)log(1+q_k/3). This is the normalized integral of N(r_k;0,xi I), xi~InvGamma(3/2,3/2). No SVD or nodes are needed; gradients use ordinary differentiable arithmetic.", "",
        "This is a different probability law. Historical conditional covariance is I+xi P(B), with independent orthogonal Gaussian noise; the continuous law's marginal whitened covariance is I+3P(B). The candidate's whitened covariance is 3I. The candidate shares one radial scale across all latent coordinates. Thus an orthogonal residual enters the denominator of every Bayes factor and generally attenuates evidence. Equal means still give equal likelihoods and preserve their odds; zero response gives an exact no-write.", "",
        "For fixed means, all candidate pair logBFs tend to zero in every distant direction. Its whitened residual score has norm <=(D+3)/(2sqrt(3)); pair logBF magnitude is therefore bounded by this constant times whitened mean separation. The 185-node law is normalized but has a finite Gaussian-mixture far tail, whose pair logBF need not tend to zero. The controlled file contains measured tail values and the analytic largest-node asymptote, without claiming the finite rule equals the continuous law at infinity.", "",
        "Historical G3 contains a source action, an opposite return action, and two null arrivals. Physical speed remains static within each episode. It contains no real hidden-condition switch. Controlled contradictory observations below test mathematical updating only; they do not establish real-environment reversal capability.", "",
        ("The controlled two-branch witness fixes means -e1,+e1 in D=192 and prior (0.1,0.9). "
         f"One arrival at -e1 gives branch-zero support {old['posterior'][0]:.6f} under the historical law "
         f"(logBF {old['logBF_0_vs_1']:.6f} < log 9), and 1-{candidate['posterior'][1]:.3g} under shared t3 "
         f"(logBF {candidate['logBF_0_vs_1']:.6f} > log 9). Thus only t3 crosses one half on this single counterevidence. "
         "On the identical predeclared four-arrival sequence (+e1,-e1,-e1,+e1), t3 first flips to branch zero at "
         f"arrival {controlled['controlled_first_branch_0_majority_arrival']['candidate_shared_student_t_nu3']}; "
         "the old law never crosses one half in those four arrivals. Neither result is a claim of universal reversal speed."), "",
        "| Weights | Law | H | Source vs uniform | All vs uniform | All vs omit |", "|---|---|---:|---:|---:|---:|"]
    for label, data in report["historical_replay"].items():
        for h, laws in data["panels"]["physical_nonzero"]["horizons"].items():
            for law, values in laws.items():
                result = [100*values[name]["directed_recovery"] for name in COMPARISONS]
                lines.append(f"| {label} | {law} | {h} | {result[0]:+.7f}% | {result[1]:+.7f}% | {result[2]:+.7f}% |")
    lines.extend(["", "These are fixed-weight replay diagnostics, not candidate training results or independent confirmation. The JSON retains every support timeline, identical-prior likelihood comparison, H1/H4 readout, and exact null check. The accompanying controlled counterexample shows that the candidate sacrifices historical response-orthogonal evidence invariance.", "",
        "The previous G1 audit's response_deviation_norm_max is a maximum B-column norm, not the Frobenius or spectral norm. Learned dual train H4 reaches 12.754951 and dual development reaches 11.944585. They are not uniformly near zero. Exact audit ranges are copied to the JSON with their source and definition.", "",
        f"Reproduce: python -m research.p1a.prbd_shared_student_t_phase1. Runtime {report['seconds']:.3f} s. No training, new data collection, G4 or P4.", ""])
    return "\n".join(lines)


def main():
    torch.set_num_threads(2)
    started = perf_counter()
    report = {"scope": "Phase 1 only; fixed-response law comparison; no default replacement",
              "candidate": "shared full-dimensional Student-t nu=3; fixed scale R, covariance 3R",
              "structural_checks": structural_checks(), "controlled_checks": controlled_checks(),
              "historical_replay": {}}
    audit = json.loads((HISTORY/"g1_g3_audit.json").read_text())
    report["historical_B_column_norm_audit"] = {"source": str(HISTORY/"g1_g3_audit.json"),
        "definition": "max_k ||L^-1(f_k-uniform_mean(f))||_2; not Frobenius or spectral norm",
        "ranges": {label: {split: {family: values["ranges"] for family, values in groups.items()}
                            for split, groups in checkpoint["splits"].items()}
                   for label, checkpoint in audit["checkpoints"].items()}}
    batch = torch.load(HISTORY/"dev.pt", weights_only=True, map_location="cpu")
    for label in ("episode_h1", "episode_dual"):
        print(label, "fixed-response density substitution", flush=True)
        report["historical_replay"][label] = historical_replay(label, batch)
    report["seconds"] = perf_counter()-started
    output_path(OUT/"shared_student_t_phase1.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    output_path(OUT/"SHARED_STUDENT_T_PHASE1.md").write_text(render(report), encoding="utf-8")
    print(output_path(OUT/"shared_student_t_phase1.json"), report["seconds"], flush=True)


if __name__ == "__main__":
    main()
