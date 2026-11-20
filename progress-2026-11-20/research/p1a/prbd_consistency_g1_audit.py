"""G1 numerical coverage on actual G3 train/development and final offsets.

Selection uses numerical range endpoints and chronology, never task gains.
No optimization, collection, G4, or P4 is performed.
"""

from .prbd_paths import ROOT as RELEASE_ROOT, CONSISTENCY, output_path

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .prbd import assimilate_continuous, continuous_quadrature, issue
from .prbd_consistency_fit import load_episodes
from .prbd_continuous_checks import check_arrays, native_gradients
from .prbd_native import NativePrototypes
from .prbd_public_fit import partition_paths, windows
from .prbd_tworoom_speed_fit import load_host


ROOT = RELEASE_ROOT
OUT = CONSISTENCY
AUDIT_LOCK = {
    "probability_law": "bounded continuous nu3, 185 fixed log-xi nodes, uniform response reference",
    "tolerances": {"max_pair_logBF_error": 1e-4, "posterior_TV": 1e-5,
                   "absolute_log_density_error": 1e-4,
                   "gradient_error": "1e-4*(1+reference norm)"},
    "range_coverage": "all factual H1 and all disjoint free H<=4 target factors in each supplied split",
    "selection_per_split_and_factor_family": ["first", "middle", "last", "largest whitened residual",
        "smallest response deviation", "largest response deviation", "smallest prior entropy",
        "largest logBF span", "largest posterior TV update", "first native-null action",
        "first full H4 endpoint"],
    "gradient_coverage": "all pairs and absolute branches in response space for every selected factor",
    "native_gradient_coverage": "all pairs and absolute branches in full native offsets for selections caused by largest residual, largest response, smallest entropy, native-null action, or full H4 endpoint",
    "posterior_priors": "actual causal nonuniform prefixes, uniform initialization; H4 factors inherit only factual prefix available at their root",
    "reference": "independent adaptive SciPy continuous density and Gaussian score integral",
    "task_score_or_checkpoint_selection": False,
}


def load_numerical_split(path):
    """Ordinary train episodes or paired actual dev replays, without fitting."""
    batch = torch.load(path, weights_only=True, map_location="cpu")
    if batch["z"].ndim == 3:
        return load_episodes(path)
    z, a, lengths, seeds = (batch[name] for name in ("z", "a", "lengths", "seed"))
    if (z.ndim != 5 or a.shape[:-2] != z.shape[:-2] or z.shape[1:3] != (2, 2)
            or z.shape[-1] != 192 or a.shape[-1] != 192 or z.size(-2) != a.size(-2)+1
            or lengths.shape != z.shape[:3] or seeds.shape != z.shape[:1]):
        raise ValueError("Paired dev requires actual z[S,2,2,T+1,192], aligned a and lengths")
    episodes, skipped = [], []
    for row, seed in enumerate(seeds):
        for speed in (0, 1):
            for program in (0, 1):
                identity = (int(seed), speed, program)
                length = int(lengths[row, speed, program])
                if length < 3:
                    skipped.append({"identity": identity, "length": length})
                    continue
                states = z[row, speed, program, :length+1].clone()
                actions = a[row, speed, program, :length].clone()
                if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
                    raise ValueError("Nonfinite actual paired dev replay")
                episodes.append((identity, states, actions, None))
    return episodes, batch["null_control"].clone(), skipped


def metrics(response, target, prior, result, noise):
    white = torch.linalg.solve_triangular(noise, response.double().T, upper=False).T
    residuals = torch.linalg.solve_triangular(noise, (target.double()-response.double()).T,
                                              upper=False).T
    positive = prior > 0
    return {"residual_norm_max": float(residuals.norm(dim=1).max()),
            "response_deviation_norm_max": float((white-white.mean(0)).norm(dim=1).max()),
            "prior_entropy": float(-(prior[positive]*prior[positive].log()).sum()),
            "prior_min": float(prior.min()), "prior_max": float(prior.max()),
            "prior_sum_roundoff": float(prior.sum()-1),
            "logBF_span": float(result.relative_log_likelihood.max()-result.relative_log_likelihood.min()),
            "posterior_update_TV": float(.5*(result.posterior-prior).abs().sum()),
            "absolute_branch_log_density_min": float(result.log_likelihood.min()),
            "absolute_branch_log_density_max": float(result.log_likelihood.max())}


@torch.no_grad()
def pool_factors(model, episodes, noise, transition, nodes, masses):
    pools = {"factual_H1": [], "free_H4_partition": []}
    uniform = noise.new_full((len(transition),), 1/len(transition))
    for seed, z, a, _ in episodes:
        responses = model(windows(z[:-1]), windows(a))
        belief, priors = uniform.clone(), {}
        for root, response in enumerate(responses, start=2):
            prior = belief.clone()
            priors[root] = prior
            signed = issue(response, prior, transition, noise, reference="uniform")
            arrival = assimilate_continuous(signed, z[root+1], nodes, masses)
            belief = arrival.posterior
            native_null = bool(torch.equal(a[root], model.null_control))
            if native_null:
                assert torch.equal(response, response[:1].expand_as(response))
                assert torch.equal(arrival.posterior, prior)
            pools["factual_H1"].append({"seed": seed, "root": root, "step": 1, "horizon": 1,
                "response": response.detach().clone(), "prior": prior,
                "target": z[root+1], "native_null": native_null,
                "metrics": metrics(response, z[root+1], prior, arrival, noise)})
        for root, path in partition_paths(model, z, a, 4):
            for step, response in enumerate(path.unbind(1), start=1):
                prior = priors[root]
                signed = issue(response, prior, transition, noise, reference="uniform")
                arrival = assimilate_continuous(signed, z[root+step], nodes, masses)
                pools["free_H4_partition"].append({"seed": seed, "root": root, "step": step,
                    "horizon": path.size(1), "response": response.detach().clone(), "prior": prior,
                    "target": z[root+step], "native_null": bool(torch.equal(a[root+step-1], model.null_control)),
                    "metrics": metrics(response, z[root+step], prior, arrival, noise)})
    return pools


def choose(records):
    selections = [("first", 0), ("middle", len(records)//2), ("last", len(records)-1)]
    for criterion, metric, largest in (("largest_residual", "residual_norm_max", True),
                                     ("smallest_response", "response_deviation_norm_max", False),
                                     ("largest_response", "response_deviation_norm_max", True),
                                     ("smallest_entropy", "prior_entropy", False),
                                     ("largest_logBF", "logBF_span", True),
                                     ("largest_update", "posterior_update_TV", True)):
        selection = max if largest else min
        index = selection(range(len(records)), key=lambda i: records[i]["metrics"][metric])
        selections.append((criterion, index))
    null = next((i for i, row in enumerate(records) if row["native_null"]), None)
    if null is not None:
        selections.append(("native_null", null))
    endpoint = next((i for i, row in enumerate(records) if row["step"] == 4), None)
    if endpoint is not None:
        selections.append(("full_H4_endpoint", endpoint))
    selected = {}
    for criterion, index in selections:
        selected.setdefault(index, []).append(criterion)
    return selected


def summary(rows):
    response_gradients = [gradient for row in rows for gradient in row["gradients"]]
    native = [gradient for row in rows for gradient in row.get("native_offset_gradients", [])]
    return {"selected_factors": len(rows),
            "max_pair_logBF_error": max(row["max_pair_logBF_error"] for row in rows),
            "max_posterior_TV": max(row["posterior_TV"] for row in rows),
            "max_absolute_log_density_error": max(row["max_absolute_log_density_error"] for row in rows),
            "max_response_gradient_error_over_1_plus_norm": max(
                gradient["error_l2"]/(1+gradient["reference_l2"]) for gradient in response_gradients),
            "native_gradient_factors": sum("native_offset_gradients" in row for row in rows),
            "max_native_gradient_error_over_1_plus_norm": max(
                (gradient["error_l2"]/(1+gradient["reference_l2"]) for gradient in native), default=0.),
            "max_reference_integral_error_bound": max(row["integration_error_bound"] for row in rows),
            "nonuniform_prefix_factors": sum(abs(row["metrics"]["prior_max"]-.25) > 1e-12 for row in rows),
            "native_null_factors": sum(row["native_null"] for row in rows)}


def audit_split(model, episodes, noise, transition, nodes, masses):
    pools = pool_factors(model, episodes, noise, transition, nodes, masses)
    lookup = {seed: (z, a) for seed, z, a, _ in episodes}
    output = {}
    for family, records in pools.items():
        ranges = {metric: {"min": min(row["metrics"][metric] for row in records),
                           "max": max(row["metrics"][metric] for row in records)}
                  for metric in records[0]["metrics"]}
        rows = []
        for index, criteria in choose(records).items():
            factor = records[index]
            row, _, _ = check_arrays(factor["response"], factor["target"], noise, factor["prior"])
            row.update({key: factor[key] for key in ("seed", "root", "step", "horizon", "native_null", "metrics")})
            row["selection_criteria"] = criteria
            row["prior"] = factor["prior"].tolist()
            if set(criteria) & {"largest_residual", "largest_response", "smallest_entropy", "native_null", "full_H4_endpoint"}:
                z, a = lookup[factor["seed"]]
                root = factor["root"]
                if family == "factual_H1":
                    response = model(z[None, root-2:root+1], a[None, root-2:root+1])[0]
                else:
                    response = model.rollout(z[None, root-2:root+1],
                                             a[None, root-2:root+factor["horizon"]])[0, :, factor["step"]-1]
                row["native_offset_gradients"] = native_gradients(model, response, factor["target"].double(), noise)
            rows.append(row)
        output[family] = {"actual_factors_scanned": len(records), "actual_episodes": len(episodes),
                          "native_null_factors_scanned": sum(row["native_null"] for row in records),
                          "factual_native_null_preserves_public_bitwise": True if family == "factual_H1" else None,
                          "ranges": ranges, "rows": rows, "summary": summary(rows)}
    return output


def render_report(report):
    lines = ["# G1 numerical coverage on new G3 data and weights", "",
        "The fixed 185-node continuous-law approximation is checked against adaptive SciPy integration on the actual new training and development archives, at the shared initialization and both final G3 checkpoints. Selection uses chronology and numerical range endpoints. No training, task-score selection, G4, or P4 occurs in this verifier.", "",
        "All factual H1 factors and all factors in each disjoint free H<=4 partition are scanned. Each selected observation uses its actual causal prefix belief; the geometry remains independent of that belief. Paired development histories are used only as actual replay inputs for numerical validation. They do not enter fitting.", "",
        "The existing limits remain 1e-4 nat for every pair logBF and absolute branch log density, 1e-5 posterior TV, and gradient L2 error <=1e-4*(1+reference norm). Every selected case checks all four absolute branches and all six pairs in response space. Range-stress cases and a full H4 endpoint also check all native-offset coordinates.", "",
        "| Weights | Split | Factors | Scanned | Selected | Max logBF error | Max TV | Max absolute-log error | Max response gradient error/(1+norm) | Max native gradient error/(1+norm) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    groups = []
    for label, checkpoint in report["checkpoints"].items():
        for split, families in checkpoint["splits"].items():
            for family, details in families.items():
                s = details["summary"]
                lines.append(f"| {label} | {split} | {family} | {details['actual_factors_scanned']} | {s['selected_factors']} | {s['max_pair_logBF_error']:.3g} | {s['max_posterior_TV']:.3g} | {s['max_absolute_log_density_error']:.3g} | {s['max_response_gradient_error_over_1_plus_norm']:.3g} | {s['max_native_gradient_error_over_1_plus_norm']:.3g} |")
                groups.append((label, split, family, details))
    lines.extend(["", "| Weights | Split | Factors | Whitened residual max | Branch deviation norm range | Prefix entropy range | Native-null factors scanned |",
                  "|---|---|---|---:|---:|---:|---:|"])
    for label, split, family, details in groups:
        r = details["ranges"]
        spread, entropy = r["response_deviation_norm_max"], r["prior_entropy"]
        lines.append(f"| {label} | {split} | {family} | {r['residual_norm_max']['max']:.6g} | [{spread['min']:.6g}, {spread['max']:.6g}] | [{entropy['min']:.6g}, {entropy['max']:.6g}] | {details['native_null_factors_scanned']} |")
    lines.extend(["", "Factual native-null arrivals are checked across the complete scanned cohorts: all prototype means are exactly equal and the returned posterior equals the supplied public weights bitwise. The identity-transport value correction retains normalized weight and transition derivatives; its independent regression is in `g1_identity_prior_roundoff.json`. Legacy paths retain their previous normalization.", "",
        "The JSON retains every selected identity, actual prefix weights, selection criterion, response/range metrics, integration error bound, all per-objective gradient errors, and full numerical ranges. Numerical agreement does not establish predictive usefulness; that requires the separate G3 development results.", "",
        "The independent replay of final G3 directed H1/H4 conclusions through adaptive continuous likelihoods is reported separately in `G1_G3_SIGNED_REFERENCE.md` and `g1_g3_signed_reference.json`.", "",
        f"Runtime: {report['seconds']:.3f} s, CPU, one PyTorch thread.", "",
        "Reproduce: `python -m research.p1a.prbd_consistency_g1_audit`. The selection/tolerance contract is `g1_g3_audit_lock.json`; full output is `g1_g3_audit.json`.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=OUT/"train.pt")
    parser.add_argument("--development", type=Path, default=OUT/"dev.pt")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock_path = OUT/"g1_g3_audit_lock.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == AUDIT_LOCK
    else:
        output_path(lock_path).write_text(json.dumps(AUDIT_LOCK, indent=2)+"\n", encoding="utf-8")
    splits = {}
    for name, path in (("train", args.train), ("development", args.development)):
        episodes, null, skipped = load_numerical_split(path)
        splits[name] = (episodes, null, skipped)
    physical_seed = lambda identity: identity[0] if isinstance(identity, tuple) else identity
    assert not {physical_seed(row[0]) for row in splits["train"][0]} & {
        physical_seed(row[0]) for row in splits["development"][0]}
    assert torch.equal(splits["train"][1], splits["development"][1])
    model = NativePrototypes(load_host("cpu"), null_control=splits["train"][1]).eval()
    nodes, masses = continuous_quadrature()
    report = {"lock": AUDIT_LOCK, "sources": {"train": str(args.train.resolve()),
                                               "development": str(args.development.resolve())},
              "paired_dev_replays_used_only_for_numerical_validation": True,
              "checkpoints": {}}
    started = perf_counter()
    for label, path in (("initialized", OUT/"g3_episode_h1.pt"),
                        ("episode_h1", OUT/"g3_episode_h1.pt"),
                        ("episode_dual", OUT/"g3_episode_dual.pt")):
        checkpoint = torch.load(path, weights_only=True, map_location="cpu")
        assert checkpoint["observation_law"] == "continuous_bounded_uniform_nu3"
        torch.testing.assert_close(checkpoint["scales"], nodes, atol=0, rtol=0)
        torch.testing.assert_close(checkpoint["masses"], masses, atol=0, rtol=0)
        noise, transition = checkpoint["noise_cholesky"].double(), checkpoint["transition"].double()
        model.offsets.data.copy_(checkpoint["initialized_offsets"] if label == "initialized" else checkpoint["offsets"])
        output = {"checkpoint": str(path.resolve()), "uses_initialized_offsets": label == "initialized", "splits": {}}
        for split, (episodes, _, _) in splits.items():
            print(label, split, "scanning actual H1 and free H4 factors", flush=True)
            output["splits"][split] = audit_split(model, episodes, noise, transition, nodes, masses)
            print(label, split, {key: value["summary"] for key, value in output["splits"][split].items()}, flush=True)
        report["checkpoints"][label] = output
        (output_path(OUT/"g1_g3_audit_partial.json")).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    report["passed"], report["seconds"] = True, perf_counter()-started
    (output_path(OUT/"g1_g3_audit.json")).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output_path(OUT/"G1_G3_AUDIT.md")).write_text(render_report(report), encoding="utf-8")
    print(OUT/"g1_g3_audit.json", report["seconds"], flush=True)


if __name__ == "__main__":
    main()
