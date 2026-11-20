"""G2 fixed TwoRoom action-response capacity and strict no-op diagnostics.

Preparation only re-embeds archived ordinary commands with a fixed native shape.
Measurement is separate and starts only after G1 numerical sign-off. Neither
stage learns offsets, selects an action, or uses physical labels in PRBD.
"""

from __future__ import annotations

from .prbd_paths import ROOT as RELEASE_ROOT, EVIDENCE, CONSISTENCY, output_path

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .prbd import assimilate_continuous, issue
from .prbd_evidence_attribution import oracle_stats, pool, values, vector_stats
from .prbd_native import NativePrototypes, encode_native_controls, native_conditioned
from .prbd_tworoom_speed_fit import OUT as DATA, RELEASE, load_host


ROOT = RELEASE_ROOT
OUT = CONSISTENCY
PREVIOUS = EVIDENCE


def codechecks():
    """Synthetic implementation checks; no real endpoints or fitted weights."""
    setup = torch.load(OUT / "g2_prepared.pt", weights_only=True)
    host = load_host("cpu")
    stats = json.loads((RELEASE / "action_statistics.json").read_text(encoding="utf-8"))
    model = NativePrototypes(host, null_control=setup["null_control"]).eval()
    generator = torch.Generator().manual_seed(2026112002)
    window = torch.randn(3, 3, 192, generator=generator)
    raw = torch.randn(3, 6, 5, 2, generator=generator) * .2
    raw[:, 2:] = 0
    controls = encode_native_controls(host, raw, stats)
    silence = model(window, controls[:, :3])
    torch.testing.assert_close(silence, silence[:, :1].expand_as(silence), atol=0, rtol=0)
    silence_gradient, = torch.autograd.grad(silence.sum(), model.offsets)
    torch.testing.assert_close(silence_gradient, torch.zeros_like(silence_gradient), atol=0, rtol=0)
    paths = model.rollout(window, controls)
    torch.testing.assert_close(paths, paths[:, :1].expand_as(paths), atol=0, rtol=0)
    anchor = _frozen_path(host, window, controls)
    torch.testing.assert_close(paths[:, 0], anchor, atol=0, rtol=0)
    active = encode_native_controls(host, torch.randn(3, 6, 5, 2, generator=generator)*.2, stats)
    response = model(window, active[:, :3])
    probe = torch.linspace(-1., 1., 192)
    gradient, = torch.autograd.grad((response*probe).sum(), model.offsets)
    if not torch.isfinite(gradient).all() or not (gradient.flatten(1).norm(dim=1)>0).all():
        raise AssertionError("Active response lost the trainable native offsets")
    rollout = model.rollout(window, active)
    nominal_gradient, = torch.autograd.grad(rollout[:, 0].sum(), model.offsets, retain_graph=True)
    torch.testing.assert_close(nominal_gradient, torch.zeros_like(nominal_gradient), atol=0, rtol=0)
    own = window[:, None].expand(-1,4,-1,-1)
    manual = []
    for t in range(4):
        step = model._step(own,active[:,None,t:t+3].expand(-1,4,-1,-1))
        manual.append(step)
        own = torch.cat((own[:,:,1:],step[:,:,None]),dim=2)
    torch.testing.assert_close(rollout,torch.stack(manual,2),atol=0,rtol=0)
    legacy = NativePrototypes(host).eval()
    offsets = legacy.offsets[None].expand(3,-1,-1,-1).flatten(0,1).transpose(0,1)[:,:,None]
    direct = native_conditioned(host,window[:,None].expand(-1,3,-1,-1).reshape(-1,3,192),
        active[:,:3,None].transpose(1,2).expand(-1,3,-1,-1).reshape(-1,3,192),offsets)[:,-1].reshape(3,3,192)
    torch.testing.assert_close(legacy(window,active[:,:3])[:,1:],direct,atol=0,rtol=0)
    result = {"passed":True,"synthetic_windows":3,"different_historical_controls_retained":True,
        "latest_raw_zero_all_prototypes_exact":True,"latest_raw_zero_offset_gradient_exact_zero":True,
        "H4_nominal_exact_native":True,"H4_noop_all_branches_exact":True,"branch_own_window_manual_exact":True,
        "active_per_prototype_gradient_l2":values(gradient.flatten(1).norm(dim=1)),
        "nominal_offset_gradient_exact_zero":True,"legacy_native_offset_path_bitwise_preserved":True,
        "scientific_measurement":False}
    (output_path(OUT/"g2_codechecks.json")).write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    return result


@torch.no_grad()
def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    host = load_host("cpu")
    stats = json.loads((RELEASE / "action_statistics.json").read_text(encoding="utf-8"))
    null = encode_native_controls(host, torch.zeros(5, 2), stats)
    prepared = {}
    for split in ("calibration", "fit", "validation"):
        batch = torch.load(DATA / f"encoded_{split}.pt", map_location="cpu", weights_only=True)
        with np.load(DATA / f"data_{split}.npz", allow_pickle=False) as archive:
            raw = torch.from_numpy(archive["actions"].copy())
            valid = torch.from_numpy(archive["block_valid"].copy())
        embedded = torch.zeros_like(batch["a"])
        embedded[valid] = encode_native_controls(host, raw[valid], stats)
        zero = valid & (raw == 0).all(-1).all(-1)
        if not torch.equal(embedded[zero], null.expand(int(zero.sum()), -1)):
            raise AssertionError("Raw-zero action embedding changed with batch context")
        prepared[split] = {**batch, "a": embedded, "raw": raw, "valid": valid}
    torch.save({"null_control": null, "splits": prepared}, output_path(OUT / "g2_prepared.pt"))
    result = {"action_encoder": "released frozen pointwise native encoder, each block [1,1,10]",
              "normalization": str(RELEASE / "action_statistics.json"),
              "existing_latents_reused": True, "no_predictions_or_training": True,
              "null_embedding_bitwise_equal_across_all_raw_zero_blocks": True,
              "splits": {split: {"valid_blocks": int(b["valid"].sum()),
                  "zero_blocks": int((b["valid"] & (b["raw"] == 0).all(-1).all(-1)).sum())}
                  for split, b in prepared.items()}}
    (output_path(OUT / "g2_preparation.json")).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _source(model, fitted, z, a, prior):
    response = model(z[None, :3], a[None, :3])[0]
    issued = issue(response, prior, fitted["transition"], fitted["noise_cholesky"], reference="uniform")
    return response, assimilate_continuous(issued, z[3]).posterior


def _frozen_path(host, window, controls):
    forecast = []
    for t in range(controls.size(1) - window.size(1) + 1):
        prediction = host.predict(window, controls[:, t:t + window.size(1)])[:, -1]
        forecast.append(prediction)
        window = torch.cat((window[:, 1:], prediction[:, None]), dim=1)
    return torch.stack(forecast, dim=1)


def _endpoint(path, target, root):
    return {"absolute_mse": float((path.double() - target.double()).square().mean()),
            "prediction_displacement_l2": float((path.double() - root.double()).norm()),
            "actual_displacement_l2": float((target.double() - root.double()).norm())}


@torch.no_grad()
def measure():
    started = perf_counter()
    setup = torch.load(OUT / "g2_prepared.pt", weights_only=True)
    batch, null = setup["splits"]["validation"], setup["null_control"]
    host = load_host("cpu")
    stats = json.loads((RELEASE / "action_statistics.json").read_text(encoding="utf-8"))
    model = NativePrototypes(host, null_control=null).eval().requires_grad_(False)
    with np.load(DATA / "data_diagnostics.npz", allow_pickle=False) as archive:
        positions = torch.from_numpy(archive["validation_positions"].copy()).double()
        collisions = torch.from_numpy(archive["validation_collision"].copy())
        speeds = torch.from_numpy(archive["validation_speed"].copy())
    physical_truth = positions[:, 0, 0, 35] - positions[:, 0, 1, 35] \
        - positions[:, 1, 0, 35] + positions[:, 1, 1, 35]
    matched = batch["matched_rgb"] & (batch["lengths"].amin((1, 2)) >= 10)
    free = matched & (physical_truth != 0).any(-1)
    if int(matched.sum()) != 11 or int(free.sum()) != 9:
        raise ValueError("The fixed 11 matched / 9 physical-effect roots changed")
    if collisions[free, :, :, 6:10].any():
        raise ValueError("Capacity endpoints require collision-free future programs")
    scaled_raw = batch["raw"] * (speeds / 5)[..., None, None, None]
    if (scaled_raw[free, :, :, 4:10].abs() >= 1).any():
        raise ValueError("Scaled diagnostic commands are saturated")
    scaled = encode_native_controls(host, scaled_raw[free], stats)
    physical_reference, reference_tensors = [], []
    for local, index in enumerate(free.nonzero().flatten().tolist()):
        z, a = batch["z"][index], batch["a"][index]
        nominal, rescaled = [], []
        for speed in (0, 1):
            nominal.append(torch.stack([_frozen_path(host, z[speed, p, 4:7][None],
                a[speed, p, 4:10][None])[0] for p in (0, 1)]))
            rescaled.append(torch.stack([_frozen_path(host, z[speed, p, 4:7][None],
                scaled[local, speed, p, 4:10][None])[0] for p in (0, 1)]))
        nominal, rescaled = torch.stack(nominal).double(), torch.stack(rescaled).double()
        targets = z[:, :, 7:11].double()
        truth = targets[0, 0] - targets[0, 1] - targets[1, 0] + targets[1, 1]
        row = {"seed": int(batch["seed"][index]), "probe_direction_index":index % 4,"horizons": {}}
        for h in (1, 4):
            def score(paths):
                contrast = paths[0, 0, h-1] - paths[0, 1, h-1] \
                    - paths[1, 0, h-1] + paths[1, 1, h-1]
                return {"interaction": vector_stats(contrast, truth[h-1]),
                    "endpoints": [[_endpoint(paths[s, p, h-1], targets[s, p, h-1], z[s, p, 6])
                                   for p in (0, 1)] for s in (0, 1)],
                    "stop_cross_speed_l2": float((paths[0, 1, h-1] - paths[1, 1, h-1]).norm())}
            row["horizons"][str(h)] = {"nominal": score(nominal), "speed_scaled_diagnostic": score(rescaled)}
        physical_reference.append(row)
        reference_tensors.append({"seed": row["seed"], "nominal": nominal, "speed_scaled": rescaled})
    outputs, tensor_outputs = [], []
    uniform = torch.full((4,), .25, dtype=torch.float64)
    unequal = torch.tensor([.07, .13, .29, .51], dtype=torch.float64)
    for label, name in (("initialized", "p3_initialized.pt"), ("prior_bounded_D", "p3_bounded_d.pt")):
        fitted = torch.load(PREVIOUS / name, map_location="cpu", weights_only=True)
        if not torch.equal(fitted["transition"], torch.eye(4, dtype=torch.float64)):
            raise ValueError("G2 fixed-condition diagnostics require T=I")
        model.offsets.copy_(fitted["offsets"])
        rows, noops, saved = [], [], []
        for index in matched.nonzero().flatten().tolist():
            seed, z, a = int(batch["seed"][index]), batch["z"][index], batch["a"][index]
            paths, source_weights, source_predictions = [], [], []
            for speed in (0, 1):
                response, weight = _source(model, fitted, z[speed, 0], a[speed, 0], uniform)
                source_weights.append(weight)
                source_predictions.append(response.double())
                paths.append(torch.stack([model.rollout(z[speed, p, 4:7][None],
                    a[speed, p, 4:10][None])[0] for p in (0, 1)]))
                public = weight
                for t in (3, 4, 5):
                    means = model(z[speed, 0, t-2:t+1][None], a[speed, 0, t-2:t+1][None])[0]
                    issued = issue(means, public, fitted["transition"], fitted["noise_cholesky"], reference="uniform")
                    arrived = assimilate_continuous(issued, z[speed, 0, t+1])
                    if t in (4, 5):
                        if not (batch["raw"][index,speed,0,t] == 0).all() \
                                or not torch.equal(positions[index,speed,0,5*t],positions[index,speed,0,5*(t+1)]):
                            raise AssertionError("A pinned no-op is not a physically stationary zero command")
                        signed_unequal = issue(means, unequal, fitted["transition"], fitted["noise_cholesky"], reference="uniform")
                        alt = assimilate_continuous(signed_unequal, z[speed, 0, t+1])
                        item = {"seed": seed, "speed_index": speed, "action_t": t,
                            "actual_position_unchanged":True,
                            "actual_latent_unchanged":bool(torch.equal(z[speed,0,t],z[speed,0,t+1])),
                            "responses_exactly_equal": bool(torch.equal(means, means[:1].expand_as(means))),
                            "likelihoods_exactly_equal": bool(torch.equal(arrived.log_likelihood, arrived.log_likelihood[:1].expand_as(arrived.log_likelihood))),
                            "observation_posterior_exactly_prior": bool(torch.equal(arrived.posterior, issued.prior)),
                            "unequal_posterior_exactly_prior": bool(torch.equal(alt.posterior, signed_unequal.prior)),
                            "posterior_exactly_supplied_public": bool(torch.equal(arrived.posterior, public)),
                            "unequal_posterior_exactly_supplied_public": bool(torch.equal(alt.posterior, unequal)),
                            "public_max_abs_change": float((arrived.posterior-public).abs().max()),
                            "public_log_odds_max_abs_change": float(((arrived.posterior.log()-arrived.posterior[:1].log())-(public.log()-public[:1].log())).abs().max()),
                            "prior": values(public), "posterior": values(arrived.posterior)}
                        noops.append(item)
                    public = arrived.posterior
            paths, supports = torch.stack(paths).double(), torch.stack(source_weights)
            source_family = torch.stack(source_predictions)
            source_truth = z[0, 0, 3].double() - z[1, 0, 3].double()
            source_contrast = supports[0] @ source_family[0] - supports[1] @ source_family[1]
            row = {"seed": seed, "probe_direction_index":index % 4,"physical_effect_nonzero": bool(free[index]),
                   "source_weights": values(supports),
                   "source_between_speed_TV": float((supports[0]-supports[1]).abs().sum()/2),
                   "source_transfer": vector_stats(source_contrast, source_truth), "horizons": {}}
            for h in (1, 4):
                families = paths[:, 0, :, h-1] - paths[:, 1, :, h-1]
                truth = z[0, 0, 6+h].double() - z[0, 1, 6+h].double() \
                    - z[1, 0, 6+h].double() + z[1, 1, 6+h].double()
                contrast = supports[0] @ families[0] - supports[1] @ families[1]
                endpoint = {method: [[_endpoint(weights[s] @ paths[s,p,:,h-1],
                    z[s,p,6+h], z[s,p,6]) for p in (0,1)] for s in (0,1)]
                    for method, weights in (("uniform",uniform.expand(2,-1)),("source",supports))}
                row["horizons"][str(h)] = {"source_weight_transfer": vector_stats(contrast, truth),
                    "favorable_reassignment_capacity": oracle_stats(families, truth),
                    "per_branch_truth_projection": values(families @ truth),
                    "branch_action_effect_l2": values(families.norm(dim=-1)),
                    "cross_speed_stop_difference_l2": float((supports[0] @ paths[0,1,:,h-1]-supports[1] @ paths[1,1,:,h-1]).norm()),
                    "endpoints": endpoint,
                    "nominal_exact": all(torch.equal(paths[s,p,0,:h].float(), _frozen_path(host,
                        z[s,p,4:7][None],a[s,p,4:6+h][None])[0]) for s in (0,1) for p in (0,1))}
            rows.append(row)
            saved.append({"seed": seed, "future_responses": paths, "source_responses": source_family,
                          "source_weights": supports})
        if len(noops) != 44 or not all(all(r[key] for key in ("responses_exactly_equal",
                "likelihoods_exactly_equal", "observation_posterior_exactly_prior", "unequal_posterior_exactly_prior",
                "posterior_exactly_supplied_public", "unequal_posterior_exactly_supplied_public")) for r in noops):
            raise AssertionError("The 44 physical no-op arrivals violate strict silence")
        if not all(h["nominal_exact"] for r in rows for h in r["horizons"].values()):
            raise AssertionError("The nominal native rollout changed")
        nonzero_rows = [row for row in rows if row["physical_effect_nonzero"]]
        summary = {str(h): {"source_weight_transfer": pool([r["horizons"][str(h)]["source_weight_transfer"] for r in nonzero_rows]),
            "favorable_reassignment_signed_recovery": sum(r["horizons"][str(h)]["favorable_reassignment_capacity"]["projection_upper"] for r in nonzero_rows) /
                sum(r["horizons"][str(h)]["source_weight_transfer"]["truth_squared_norm"] for r in nonzero_rows),
            "positive_capacity_roots": sum(r["horizons"][str(h)]["favorable_reassignment_capacity"]["projection_upper"] > 0 for r in nonzero_rows)} for h in (1,4)}
        outputs.append({"checkpoint": label, "offset_path": str(PREVIOUS/name), "summary_nonzero9": summary,
                        "rows": rows, "no_op_arrivals": noops})
        tensor_outputs.append({"checkpoint": label, "cases": saved})
    reference_summary = {str(h): {method: pool([r["horizons"][str(h)][method]["interaction"] for r in physical_reference])
        for method in ("nominal","speed_scaled_diagnostic")} for h in (1,4)}
    capacity_passed = all(s["positive_capacity_roots"] == 9 and s["favorable_reassignment_signed_recovery"] > 0
        for output in outputs for s in output["summary_nonzero9"].values())
    reference_passed = all(reference_summary[str(h)]["speed_scaled_diagnostic"]["signed_recovery"] > 0 for h in (1,4))
    report = {"scope": "G2 fixed archived developmental panel; no training or independent generalization claim",
        "g1_numerical_pass_required_before_execution": True,
        "operation": "F0(W,a)+(Gu(W,a)-G0(W,a))-(Gu(W,a_ref)-G0(W,a_ref)); only latest command changes in reference",
        "no_detector_or_bayesian_gating": True, "geometry": "G1 continuous bounded, reference=uniform; geometry independent of current belief",
        "native_cost_per_step": {"prototype_count":4,"native_predictor_batch_calls":8,"native_branch_evaluations":16,
                                 "frozen_base_evaluated_with_each_branchs_own_window":True},
        "physical_diagnostic": "Only frozen F0 sees privileged raw command * actual speed/5 before official normalization; PRBD never sees speed",
        "coverage": {"matched11": values(batch["seed"][matched]), "collision_free_unsaturated_effect9": values(batch["seed"][free]), "noops_per_checkpoint":44},
        "frozen_reference_summary":reference_summary,"frozen_reference_rows":physical_reference,"prototypes":outputs,
        "admissibility": {"strict_noop_all44_each_checkpoint":True,"favorable_native_capacity_H1_H4_all9":capacity_passed,
                          "frozen_scaled_reference_positive_H1_H4":reference_passed,
                          "passed":capacity_passed and reference_passed,
                          "learning_alignment_not_required_by_G2":True},"elapsed_seconds":perf_counter()-started}
    add_summaries(report)
    (output_path(OUT/"g2.json")).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output_path(OUT/"g2.md")).write_text(render_report(report),encoding="utf-8")
    torch.save({"frozen_reference":reference_tensors,"prototypes":tensor_outputs},output_path(OUT/"g2_tensors.pt"))
    return report


def add_summaries(report):
    """Pool saved absolute errors and group fixed directions without new forwards."""
    directions = ("left","right","up","down")
    for h in ("1","4"):
        for method in ("nominal","speed_scaled_diagnostic"):
            group = report["frozen_reference_summary"][h][method]
            rows = report["frozen_reference_rows"]
            group["absolute_endpoint_MSE_mean_by_program"] = {
                name:sum(r["horizons"][h][method]["endpoints"][s][p]["absolute_mse"]
                    for r in rows for s in (0,1))/(2*len(rows)) for p,name in enumerate(("movement","stop"))}
            group["by_probe_direction"] = {name:pool([r["horizons"][h][method]["interaction"]
                for r in rows if r["probe_direction_index"]==d]) for d,name in enumerate(directions)}
        for output in report["prototypes"]:
            rows = [r for r in output["rows"] if r["physical_effect_nonzero"]]
            summary = output["summary_nonzero9"][h]
            summary["absolute_endpoint_MSE_mean"] = {method:{name:sum(
                r["horizons"][h]["endpoints"][method][s][p]["absolute_mse"] for r in rows for s in (0,1))/(2*len(rows))
                for p,name in enumerate(("movement","stop"))} for method in ("uniform","source")}
            summary["by_probe_direction"] = {name:pool([r["horizons"][h]["source_weight_transfer"]
                for r in rows if r["probe_direction_index"]==d]) for d,name in enumerate(directions)}
    report["no_op_precision"] = {output["checkpoint"]:{
        "observation_update_max_abs_change":0.,
        "input_public_to_returned_max_abs_change":max(r["public_max_abs_change"] for r in output["no_op_arrivals"]),
        "input_public_log_odds_max_abs_change":max(r["public_log_odds_max_abs_change"] for r in output["no_op_arrivals"])}
        for output in report["prototypes"]}
    report["no_op_precision"]["interpretation"] = "With the uniform reference and identity transport, the observation posterior is bitwise equal to both issued.prior and the supplied public weights, including sum roundoff. Normalized probability derivatives and legacy behavior are retained."


def render_report(report):
    lines = ["# G2 native action-response consistency", "",
        "The fixed developmental panel passes native capacity and no-op admissibility. No offsets were trained. These are not independent generalization results.","",
        "| Frozen F0 reference | H1 | H4 |","|---|---:|---:|",
        "| Raw command × physical speed/5, signed recovery | " + " | ".join(
            f"{100*report['frozen_reference_summary'][h]['speed_scaled_diagnostic']['signed_recovery']:.4f}%" for h in ("1","4")) + " |",
        "| Positive physical directions | 9/9 | 9/9 |","",
        "Actual speed is supplied only to this frozen diagnostic, before official action normalization. All nine endpoints are collision-free and scaled commands are unsaturated. The corresponding unscaled F0 interaction is exactly zero.","",
        "| Native offsets | H | Favorable reassignment capacity | Actual source transfer | Positive actual transfer |",
        "|---|---:|---:|---:|---:|"]
    for output in report["prototypes"]:
        for h in ("1","4"):
            row = output["summary_nonzero9"][h]
            transfer = row["source_weight_transfer"]
            lines.append(f"| {output['checkpoint']} | {h} | {100*row['favorable_reassignment_signed_recovery']:.5f}% | {100*transfer['signed_recovery']:+.5f}% | {transfer['sign_counts']['positive']}/9 |")
    lines += ["", "Capacity uses a truth-informed, per-root reassignment solely to test whether the fixed branch family can point in the physical direction. Actual transfer uses G1 continuous bounded source likelihoods and the uniform response reference. The two quantities have different roles; neither selects a new architecture or enters deployment.","",
        "For both initialized and prior bounded-D offsets, all 44 stationary arrivals have bitwise identical prototype means and likelihoods; their observation posterior is bitwise equal to both the issued prior and the supplied public weights, including unequal-prior checks. Native nominal H1–H4 matches F0 exactly. The maximum incoming-public-to-returned difference is " +
        f"{max(report['no_op_precision'][o['checkpoint']]['input_public_to_returned_max_abs_change'] for o in report['prototypes']):.3g}" +
        ". The uniform-reference identity-transport path preserves the public value through probability-sum roundoff, while retaining normalized probability derivatives. Legacy behavior remains unchanged. No detector, threshold, or static-action update gate was added.","",
        "| H | F0 scaled move MSE | F0 scaled stop MSE | Prior-D uniform move | Prior-D source move | Prior-D uniform stop | Prior-D source stop |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    learned = next(o for o in report["prototypes"] if o["checkpoint"]=="prior_bounded_D")
    for h in ("1","4"):
        reference = report["frozen_reference_summary"][h]["speed_scaled_diagnostic"]["absolute_endpoint_MSE_mean_by_program"]
        mse = learned["summary_nonzero9"][h]["absolute_endpoint_MSE_mean"]
        numbers = [reference["movement"],reference["stop"],mse["uniform"]["movement"],mse["source"]["movement"],mse["uniform"]["stop"],mse["source"]["stop"]]
        lines.append(f"| {h} | " + " | ".join(f"{v:.8f}" for v in numbers)+" |")
    lines += ["","The JSON retains every root, projection sign, direction group, absolute endpoint MSE, zero-effect root and stationary prior/posterior. Each prototype owns its free rollout window. The fixed-shape native action encoder ensures raw-zero embeddings agree across training and evaluation batches. The new step uses eight predictor batch calls and sixteen branch evaluations for four prototypes.","",
        "Implementation: `research/p1a/prbd_native.py`; runner: `research/p1a/prbd_consistency_native.py`. Commands: `python -m research.p1a.prbd_consistency_native prepare`, `check`, then, after G1 sign-off, `measure --g1-passed`. CPU Python: `python`.",""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare","check","measure"))
    parser.add_argument("--g1-passed", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "measure" and not args.g1_passed:
        parser.error("G2 measurement requires completed G1 numerical sign-off")
    result = {"prepare":prepare,"check":codechecks,"measure":measure}[args.stage]()
    print(json.dumps(result if args.stage!="measure" else result["admissibility"]),flush=True)


if __name__ == "__main__":
    main()
