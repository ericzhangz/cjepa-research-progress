"""Frozen TwoRoom evidence attribution, with future truth used only for scoring.

The three source likelihoods share the same issued means and calibrated R.
Native FP32 forwards preserve the archived batching; all evidence, differences,
oracle bounds, and pooled statistics use FP64. No data or parameters are fit.
"""

from __future__ import annotations

from .prbd_paths import ROOT as RELEASE_ROOT, output_path

import argparse
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import torch

from .prbd import assimilate, issue
from .prbd_native import NativePrototypes
from .prbd_tworoom_speed_fit import OUT as DATA, load_host


ROOT = RELEASE_ROOT
OUTPUT = ROOT / "runs/prbd_evidence_trajectory_20261119/p12.json"
HORIZONS = (1, 2, 3, 4)
SOURCE_METHODS = ("original", "bounded", "gaussian")
METHODS = ("uniform", "nominal", "original_source_only", "bounded_source_only",
           "gaussian_source_only", "original_all_arrivals", "original_skip_source",
           "bounded_all_arrivals", "bounded_skip_source")


def ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def values(tensor):
    return tensor.detach().cpu().tolist()


def vector_stats(prediction, truth):
    prediction, truth = prediction.double(), truth.double()
    squared, dot = float(truth @ truth), float(prediction @ truth)
    norm_squared = float(prediction @ prediction)
    return {"truth_squared_norm": squared, "prediction_truth_dot": dot,
            "prediction_squared_norm": norm_squared,
            "error_squared_norm": float((prediction - truth).square().sum()),
            "signed_recovery": ratio(dot, squared),
            "relative_squared_error_gain_vs_zero": ratio(2 * dot - norm_squared, squared),
            "sign": "positive" if dot > 0 else "negative" if dot < 0 else "zero"}


def pool(rows):
    keys = ("truth_squared_norm", "prediction_truth_dot", "prediction_squared_norm",
            "error_squared_norm")
    result = {"count": len(rows), "zero_truth_count": sum(row["truth_squared_norm"] == 0 for row in rows)}
    result.update({key + "_sum": math.fsum(row[key] for row in rows) for key in keys})
    denominator = result["truth_squared_norm_sum"]
    result["signed_recovery"] = ratio(result["prediction_truth_dot_sum"], denominator)
    result["relative_squared_error_gain_vs_zero"] = ratio(
        2 * result["prediction_truth_dot_sum"] - result["prediction_squared_norm_sum"], denominator)
    result["sign_counts"] = {sign: sum(row["sign"] == sign for row in rows)
                             for sign in ("positive", "negative", "zero")}
    return result


def posterior(responses, prior, actual, fitted, geometry):
    issued = issue(responses, prior, fitted["transition"], fitted["noise_cholesky"])
    if geometry == "gaussian":
        x = torch.linalg.solve_triangular(issued.noise_cholesky,
            (actual.double() - issued.center)[:, None], upper=False)[:, 0]
        relative = issued.basis.T @ x - .5 * issued.gram.diagonal()
        return torch.softmax(prior.log() + relative, dim=0), relative
    result = assimilate(issued, actual, fitted["scales"], fitted["masses"], geometry=geometry)
    return result.posterior, result.relative_log_likelihood


@torch.no_grad()
def predict_prefix(model, fitted, z_prefix, actions):
    """Forecast boundary: accepts exactly actual z0..z6; no future target argument.

    This is also the evaluator boundary for a separately trained checkpoint.
    The two programs are independently replayed, preserving archive batch sizes.
    """
    if z_prefix.shape != (2, 2, 7, 192) or actions.shape != (2, 2, 10, 192):
        raise ValueError("Forecasts require exactly z0..z6 and ten executed/planned controls")
    count = len(model.conditions)
    uniform = fitted["transition"].new_full((count,), 1 / count)
    nominal = torch.zeros_like(uniform)
    nominal[0] = 1
    supports = torch.empty((len(METHODS), 2, 2, count), dtype=torch.float64)
    source_responses = torch.empty((2, 2, count, 192), dtype=torch.float64)
    source_relative = torch.empty((3, 2, 2, count), dtype=torch.float64)
    paths = torch.empty((2, 2, count, 4, 192), dtype=torch.float64)
    timeline = torch.empty((2, 2, 2, 2, 4, count), dtype=torch.float64)
    # timeline axes: geometry, speed, program, all/skip, arrival z3..z6, K.
    for speed in (0, 1):
        for program in (0, 1):
            z, a = z_prefix[speed, program], actions[speed, program]
            response = model(z[None, :3], a[None, :3])[0].double()
            source_responses[speed, program] = response
            state = {"uniform": uniform, "nominal": nominal}
            for mi, name in enumerate(SOURCE_METHODS):
                weight, relative = posterior(response, uniform, z[3], fitted, name)
                state[name + "_source_only"] = weight
                source_relative[mi, speed, program] = relative
                if name != "gaussian":
                    state[name + "_all_arrivals"] = weight
                    state[name + "_skip_source"] = uniform
            for gi, geometry in enumerate(("original", "bounded")):
                for ci, chronology in enumerate(("all_arrivals", "skip_source")):
                    timeline[gi, speed, program, ci, 0] = state[geometry + "_" + chronology]
            for t in range(3, 6):
                response = model(z[None, t-2:t+1], a[None, t-2:t+1])[0].double()
                for gi, geometry in enumerate(("original", "bounded")):
                    for ci, chronology in enumerate(("all_arrivals", "skip_source")):
                        key = geometry + "_" + chronology
                        state[key] = posterior(response, state[key], z[t+1], fitted, geometry)[0]
                        timeline[gi, speed, program, ci, t-2] = state[key]
            for mi, method in enumerate(METHODS):
                supports[mi, speed, program] = state[method]
            paths[speed, program] = model.rollout(z[None, 4:7], a[None, 4:10])[0].double()
    # Independent programs have the same actual history through their fork.
    torch.testing.assert_close(source_responses[:, 0], source_responses[:, 1], atol=0, rtol=0)
    torch.testing.assert_close(supports[:, :, 0], supports[:, :, 1], atol=0, rtol=0)
    torch.testing.assert_close(timeline[:, :, 0], timeline[:, :, 1], atol=0, rtol=0)
    return {"supports": supports, "source_responses": source_responses,
            "source_relative_log_likelihood": source_relative, "future_responses": paths,
            "chronological_supports": timeline}


def oracle_stats(families, truth):
    """Exact directional simplex cap; each speed uses one support for both actions."""
    scores = families @ truth
    slow, fast = int(scores[0].argmax()), int(scores[1].argmin())
    upper = float(scores[0, slow] - scores[1, fast])
    lower = float(scores[0].min() - scores[1].max())
    squared = float(truth @ truth)
    return {"native_move_minus_stop_truth_dots": values(scores),
            "slow_max_prototype": slow, "fast_min_prototype": fast,
            "slow_max_exact_ties": values((scores[0] == scores[0].max()).nonzero().flatten()),
            "fast_min_exact_ties": values((scores[1] == scores[1].min()).nonzero().flatten()),
            "oracle_slow_support": values(torch.eye(len(scores[0]), dtype=torch.float64)[slow]),
            "oracle_fast_support": values(torch.eye(len(scores[0]), dtype=torch.float64)[fast]),
            "projection_lower": lower, "projection_upper": upper,
            "signed_recovery_upper": ratio(upper, squared),
            "zero_truth": squared == 0}


def score_case(case, physical, predicted):
    z = case["z"].double()
    supports, paths = predicted["supports"], predicted["future_responses"]
    source = predicted["source_responses"]
    # State-by-program truth is slow movement - slow stop - fast movement + fast stop.
    truth = z[0, 0, 7:11] - z[0, 1, 7:11] - z[1, 0, 7:11] + z[1, 1, 7:11]
    xy = torch.from_numpy(physical[:, :, 35:51:5].copy()).double()
    physical_truth = xy[0, 0] - xy[0, 1] - xy[1, 0] + xy[1, 1]
    source_truth = z[0, 0, 3] - z[1, 0, 3]
    source_families = source[:, 0]
    row = {"seed": case["seed"], "matched_rgb": case["matched_rgb"],
           "matched_encoded_root_exact": bool(torch.equal(z[:, :, 4:7], z[0, 0, 4:7].expand(2, 2, -1, -1))),
           "matched_encoded_root_allclose": bool(torch.allclose(z[:, :, 4:7], z[0, 0, 4:7].expand(2, 2, -1, -1), atol=1e-6, rtol=1e-6)),
           "source_truth_squared_norm": float(source_truth @ source_truth), "source": {},
           "horizons": {}, "endpoints": {}}
    same_controls = bool(torch.equal(case["a"][0], case["a"][1]))
    shared_past = bool(torch.equal(case["a"][:, :, 4:6], case["a"][0, 0, 4:6].expand(2, 2, -1, -1)))
    row["matched"] = case["matched_rgb"] and row["matched_encoded_root_allclose"] and same_controls and shared_past
    for method in ("uniform", "original_source_only", "bounded_source_only", "gaussian_source_only"):
        weight = supports[METHODS.index(method), :, 0]
        contrast = weight[0] @ source_families[0] - weight[1] @ source_families[1]
        row["source"][method] = {**vector_stats(contrast, source_truth),
            "supports_slow_fast": values(weight), "between_speed_support_TV": float((weight[0] - weight[1]).abs().sum() / 2),
            "from_uniform_TV_by_speed": values((weight - .25).abs().sum(-1) / 2)}
        if method != "uniform":
            geometry_index = SOURCE_METHODS.index(method.removesuffix("_source_only"))
            row["source"][method]["relative_log_likelihood_slow_fast"] = values(
                predicted["source_relative_log_likelihood"][geometry_index, :, 0])
    for hi, horizon in enumerate(HORIZONS):
        branches, actual = paths[:, :, :, hi], z[:, :, 6+horizon]
        families = branches[:, 0] - branches[:, 1]
        oracle = oracle_stats(families, truth[hi])
        hrow = {"truth_squared_norm": float(truth[hi] @ truth[hi]),
                "physical_truth_xy": values(physical_truth[hi]),
                "physical_truth_zero": bool((physical_truth[hi] == 0).all()),
                "oracle": oracle, "methods": {}, "retention": {}}
        if hrow["physical_truth_zero"] and hrow["truth_squared_norm"] != 0:
            raise ValueError("Physical-zero matched contrast unexpectedly has nonzero encoded truth")
        midpoint = (families[0] + families[1]) / 2
        mid_scores = midpoint @ truth[hi]
        range_dot = float(mid_scores.max() - mid_scores.min())
        for mi, method in enumerate(METHODS):
            weight = supports[mi]
            # Both independently executed programs must use the same retained support.
            prediction = (weight[0, 0] @ branches[0, 0] - weight[0, 1] @ branches[0, 1]
                          - weight[1, 0] @ branches[1, 0] + weight[1, 1] @ branches[1, 1])
            delta = weight[0, 0] - weight[1, 0]
            support_part = delta @ midpoint
            window_part = ((weight[0, 0] + weight[1, 0]) / 2) @ (families[0] - families[1])
            error = float((prediction - support_part - window_part).abs().max())
            if error > 1e-11:
                raise ValueError("Support/input midpoint identity failed")
            tv = float(delta.abs().sum() / 2)
            signed_contributions = torch.stack((weight[0, 0] * (families[0] @ truth[hi]),
                                               -weight[1, 0] * (families[1] @ truth[hi])))
            hrow["methods"][method] = {**vector_stats(prediction, truth[hi]),
                "supports_slow_fast": values(weight[:, 0]), "between_speed_support_TV": tv,
                "signed_per_prototype_projection_contributions_slow_fast": values(signed_contributions),
                "support_delta_midpoint_projection_contributions": values(delta * mid_scores),
                "contribution_sum_error": abs(float(signed_contributions.sum()) - float(prediction @ truth[hi])),
                "support_only_midpoint": vector_stats(support_part, truth[hi]),
                "actual_window_midpoint": vector_stats(window_part, truth[hi]),
                "fixed_TV_directional_cap": tv * range_dot,
                "orientation_efficiency": ratio(float(support_part @ truth[hi]), tv * range_dot),
                "slow_mass_on_oracle_max": float(weight[0, 0, oracle["slow_max_prototype"]]),
                "fast_mass_on_oracle_min": float(weight[1, 0, oracle["fast_min_prototype"]]),
                "midpoint_identity_linf_error": error}
        for geometry in ("original", "bounded"):
            after = supports[METHODS.index(geometry + "_all_arrivals")]
            before = supports[METHODS.index(geometry + "_skip_source")]
            revision_by_speed = torch.stack([(after[s, 0] - before[s, 0]) @ families[s] for s in (0, 1)])
            revision = revision_by_speed[0] - revision_by_speed[1]
            hrow["retention"][geometry] = {**vector_stats(revision, truth[hi]),
                "interaction_squared_error_gain":
                    hrow["methods"][geometry + "_skip_source"]["error_squared_norm"]
                    - hrow["methods"][geometry + "_all_arrivals"]["error_squared_norm"],
                "retained_increment_TV_by_speed": values((after[:, 0] - before[:, 0]).abs().sum(-1) / 2),
                "source_increment_remaining_TV_ratio_by_speed": [ratio(float((after[s, 0] - before[s, 0]).abs().sum()),
                    float((supports[METHODS.index(geometry + "_source_only"), s, 0] - .25).abs().sum())) for s in (0, 1)]}
        row["horizons"][str(horizon)] = hrow
        endpoints = {}
        for program, name in enumerate(("movement", "stop")):
            endpoint = {"physical_actual_displacement_squared_by_speed": values(torch.from_numpy(
                physical[:, program, (6+horizon)*5] - physical[:, program, 30]).double().square().sum(-1)),
                "methods": {}}
            for mi, method in enumerate(METHODS):
                forecast = torch.stack([supports[mi, speed, program] @ branches[speed, program] for speed in (0, 1)])
                actual_error = (forecast - actual[:, program]).square().mean(-1)
                displacement = (forecast - z[:, program, 6]).square().sum(-1)
                endpoint["methods"][method] = {"mse_by_speed": values(actual_error),
                    "prediction_displacement_squared_by_speed": values(displacement),
                    "speed_contrast": vector_stats(forecast[0] - forecast[1], actual[0, program] - actual[1, program])}
            endpoints[name] = endpoint
        row["endpoints"][str(horizon)] = endpoints
    return row, {"seed": case["seed"], "matched": row["matched"], **predicted,
                 "move_minus_stop": paths[:, 0] - paths[:, 1],
                 "true_interaction": truth, "physical_interaction": physical_truth,
                 "actual_targets": z[:, :, 7:11], "actual_root": z[:, :, 4:7],
                 "source_actual_arrivals": z[:, :, 3]}


def summarize(rows):
    summary = {"seed_count": len(rows), "seeds": [row["seed"] for row in rows], "horizons": {}, "source": {}}
    for method in ("uniform", "original_source_only", "bounded_source_only", "gaussian_source_only"):
        summary["source"][method] = {**pool([row["source"][method] for row in rows]),
            "between_speed_support_TV_mean": float(np.mean([row["source"][method]["between_speed_support_TV"] for row in rows])) if rows else None,
            "from_uniform_TV_mean": float(np.mean([row["source"][method]["from_uniform_TV_by_speed"] for row in rows])) if rows else None}
    for horizon in HORIZONS:
        key = str(horizon)
        hrows = [row["horizons"][key] for row in rows]
        denominator = math.fsum(row["truth_squared_norm"] for row in hrows)
        summary["horizons"][key] = {
            "physical_zero_seeds": [row["seed"] for row in rows if row["horizons"][key]["physical_truth_zero"]],
            "truth_squared_norm_sum": denominator,
            "oracle_projection_upper_sum": math.fsum(row["oracle"]["projection_upper"] for row in hrows),
            "oracle_signed_recovery_upper": ratio(math.fsum(row["oracle"]["projection_upper"] for row in hrows), denominator),
            "methods": {method: pool([row["methods"][method] for row in hrows]) for method in METHODS},
            "retention": {geometry: pool([row["retention"][geometry] for row in hrows]) for geometry in ("original", "bounded")},
            "endpoints": {}}
        for method in METHODS:
            measurements = [row["methods"][method] for row in hrows]
            cap = math.fsum(row["fixed_TV_directional_cap"] for row in measurements)
            summary["horizons"][key]["methods"][method].update({
                "fixed_TV_directional_cap_sum": cap,
                "orientation_efficiency": ratio(math.fsum(row["support_only_midpoint"]["prediction_truth_dot"] for row in measurements), cap),
                "between_speed_support_TV_mean": float(np.mean([row["between_speed_support_TV"] for row in measurements])) if measurements else None})
        for geometry in ("original", "bounded"):
            gains = [row["retention"][geometry]["interaction_squared_error_gain"] for row in hrows]
            summary["horizons"][key]["retention"][geometry].update({
                "interaction_squared_error_gain_sum": math.fsum(gains),
                "improved_interaction_count": sum(gain > 0 for gain in gains)})
        for program in ("movement", "stop"):
            endpoints = [row["endpoints"][key][program] for row in rows]
            summary["horizons"][key]["endpoints"][program] = {method: {
                "absolute_mse_mean": float(np.mean([e["methods"][method]["mse_by_speed"] for e in endpoints])) if rows else None,
                "predicted_displacement_l2_rms": math.sqrt(float(np.mean([e["methods"][method]["prediction_displacement_squared_by_speed"] for e in endpoints]))) if rows else None,
                "speed_contrast": pool([e["methods"][method]["speed_contrast"] for e in endpoints])}
                for method in METHODS}
    return summary


def archive_regression(rows, checkpoint):
    if checkpoint.resolve() != (DATA / "fitted.pt").resolve():
        return {"applicable": False, "reason": "Checkpoint differs from archived original fit"}
    archived = json.loads((DATA / "fitted_measure.json").read_text(encoding="utf-8"))
    posterior_errors, recovery_errors, mse_errors = [], [], []
    for row in rows:
        old = next(item for item in archived["seeds"] if item["seed"] == row["seed"])
        if row["matched"] != old["root_checks"]["matched_subset_eligible"]:
            raise ValueError("Matched set changed from frozen archive")
        for name, legacy in (("original_source_only", "prbd_source_only"), ("gaussian_source_only", "gaussian_source_only")):
            for speed in (0, 1):
                prior = next(s for s in old["sources"] if s["speed_index"] == speed and s["program"] == "movement")
                legacy_weights = prior["posterior" if legacy == "prbd_source_only" else "gaussian_posterior"]
                posterior_errors.append(float(np.max(np.abs(np.array(row["source"][name]["supports_slow_fast"][speed]) - legacy_weights))))
            for key in ("1", "2", "3", "4"):
                recovery_errors.append(abs(row["horizons"][key]["methods"][name]["prediction_truth_dot"] - old["interactions"][key]["full_prediction"][legacy]["prediction_truth_dot"]))
                for program in ("movement", "stop"):
                    for speed in (0, 1):
                        previous = next(e for e in old["endpoints"] if e["horizon"] == int(key) and e["program"] == program and e["speed_index"] == speed)
                        mse_errors.append(abs(row["endpoints"][key][program]["methods"][name]["mse_by_speed"][speed] - previous["strategies"][legacy]["mse"]))
    maxima = {"posterior_max_abs_error": max(posterior_errors), "interaction_dot_max_abs_error": max(recovery_errors),
              "endpoint_mse_max_abs_error": max(mse_errors)}
    if maxima["posterior_max_abs_error"] > 1e-10 or maxima["interaction_dot_max_abs_error"] > 1e-9 or maxima["endpoint_mse_max_abs_error"] > 1e-10:
        raise ValueError(f"Archive regression failed: {maxima}")
    return {"applicable": True, "passed": True, **maxima}


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path=DATA / "fitted.pt"):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = perf_counter()
    checkpoint = Path(checkpoint_path)
    fitted = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not torch.equal(fitted["transition"], torch.eye(4, dtype=torch.float64)):
        raise ValueError("Frozen fixed-speed attribution requires four prototypes and identity transport")
    model = NativePrototypes(load_host("cpu")).eval().requires_grad_(False)
    model.offsets.copy_(fitted["offsets"])
    batch = torch.load(DATA / "encoded_validation.pt", map_location="cpu", weights_only=True)
    with np.load(DATA / "data_diagnostics.npz", allow_pickle=False) as archive:
        positions = archive["validation_positions"].copy()
        speeds = archive["validation_speed"].copy()
    if not np.all(speeds[:, 0] == 2.5) or not np.all(speeds[:, 1] == 7.5):
        raise ValueError("Archived slow/fast physical order changed")
    rows, tensors, truncated = [], [], []
    for index, seed in enumerate(batch["seed"]):
        case = {key: batch[key][index] for key in ("z", "a", "lengths")}
        case.update(seed=int(seed), matched_rgb=bool(batch["matched_rgb"][index]))
        if int(case["lengths"].min()) < 10:
            if int(case["lengths"].max()) >= 3:
                raise ValueError("Frozen panel has unexpected partial endpoint coverage")
            truncated.append({"seed": int(seed), "lengths": values(case["lengths"]), "status": "truncated_before_source"})
            continue
        predicted = predict_prefix(model, fitted, case["z"][:, :, :7], case["a"])
        row, tensor = score_case(case, positions[index], predicted)
        rows.append(row)
        tensors.append(tensor)
    matched = [row for row in rows if row["matched"]]
    nonzero = [row for row in matched if not row["horizons"]["1"]["physical_truth_zero"]]
    if len(rows) != 15 or len(matched) != 11 or len(nonzero) != 9:
        raise ValueError("Frozen panel coverage changed")
    for row in matched:
        if any(row["horizons"][str(h)]["physical_truth_zero"] != row["horizons"]["1"]["physical_truth_zero"] for h in HORIZONS):
            raise ValueError("Physical-zero mask changes across the frozen quiet suffix")
    validation = archive_regression(rows, checkpoint)
    validation.update({"nominal_anchor_exact_by_construction": True,
        "independent_program_source_responses_exact": True, "independent_program_supports_exact": True,
        "forecast_input_boundary": "predict_prefix accepts exactly seven actual states z0..z6; no target argument",
        "future_states_used_for_scoring_only": True,
        "max_midpoint_identity_linf_error": max(h["methods"][m]["midpoint_identity_linf_error"] for r in rows for h in r["horizons"].values() for m in METHODS),
        "max_projection_contribution_sum_error": max(h["methods"][m]["contribution_sum_error"] for r in rows for h in r["horizons"].values() for m in METHODS)})
    report = {"checkpoint": str(checkpoint.resolve()), "scope": "P1/P2 fixed-checkpoint attribution; no collection, training, tuning, or policy execution",
        "protocol": {"source": "issued t2 from z0..z2; only actual z3 writes source-only support",
            "root": "actual z4,z5,z6, same past controls; independent movement/stop execution",
            "future": "each native prototype freely predicts its own z7..z10; future truth enters scoring only",
            "orientation": "slow(2.5) movement-minus-stop minus fast(7.5) movement-minus-stop",
            "A": "m dot d = sum_k p_s[k](V_s[k] dot d) - p_f[k](V_f[k] dot d)",
            "oracle": "per root and horizon, slow argmax V_s dot d; fast argmin V_f dot d; shared support across movement and stop",
            "oracle_selection": "offline diagnostic only, never used in likelihoods or deployment",
            "ties": "exact equality, all tied indices retained; deterministic first index for one-hot display",
            "aggregation": "sum numerator / sum truth squared norm, retaining every sign and zero case",
            "precision": "native archived float32 with TF32 off; all evidence and scoring float64",
            "bounded_noise": "rho=lambda=1 fixed; means/R/quadrature unchanged",
            "global_permutation": "not computed; per-root signed contributions and cap distinguish orientation without relabeling learned prototypes"},
        "runtime": {"python": sys.executable, "python_version": platform.python_version(),
                    "torch": torch.__version__, "torch_threads": torch.get_num_threads()},
        "coverage": {"all_frozen_seeds": values(batch["seed"]), "complete_seeds": len(rows),
            "matched_raw_roots": len(matched), "matched_nonzero_roots": len(nonzero),
            "matched_physical_zero_seeds": [r["seed"] for r in matched if r not in nonzero], "truncated": truncated},
        "all_complete": summarize(rows), "matched_raw": summarize(matched), "matched_nonzero": summarize(nonzero),
        "seeds": rows, "validation": validation, "elapsed_seconds": perf_counter() - started}
    return report, {"methods": list(METHODS), "source_likelihood_methods": list(SOURCE_METHODS),
                    "axes": {"supports": "method,speed,program,K", "future_responses": "speed,program,K,H,D",
                             "move_minus_stop": "speed,K,H,D", "true_interaction": "H,D",
                             "chronological_supports": "geometry,speed,program,all_or_skip,arrival_z3_to_z6,K"},
                    "cases": tensors}


def render_report(report):
    lines = ["# Frozen TwoRoom P1/P2 evidence attribution", "",
        f"Checkpoint: `{report['checkpoint']}`. Fixed 16 seeds; 15 complete, 11 raw matched, 9 with nonzero physical interaction.", "",
        "All percentages are signed projection recovery; endpoints are raw 192-dimensional latent MSE. The physical-zero seeds remain in the raw panel.", "",
        "| Panel | H | Original source | Bounded source | Gaussian source | Directional oracle cap |", "|---|---:|---:|---:|---:|---:|"]
    for panel in ("matched_raw", "matched_nonzero"):
        for key, h in report[panel]["horizons"].items():
            recovery = [h["methods"][m + "_source_only"]["signed_recovery"] for m in SOURCE_METHODS]
            lines.append(f"| {panel} | {key} | " + " | ".join(f"{100*r:+.5f}%" for r in recovery) + f" | {100*h['oracle_signed_recovery_upper']:.4f}% |")
    lines += ["", "| Source (11 roots) | Signed recovery | Slow/fast support TV | TV from uniform |", "|---|---:|---:|---:|"]
    for name in SOURCE_METHODS:
        row = report["matched_raw"]["source"][name + "_source_only"]
        lines.append(f"| {name} | {100*row['signed_recovery']:+.4f}% | {row['between_speed_support_TV_mean']:.6f} | {row['from_uniform_TV_mean']:.6f} |")
    lines += ["", "| H | Method | Movement endpoint MSE | Stop endpoint MSE | Stop predicted displacement RMS |", "|---:|---|---:|---:|---:|"]
    for key, h in report["matched_raw"]["horizons"].items():
        for method in ("uniform", "original_source_only", "bounded_source_only", "gaussian_source_only"):
            move, stop = (h["endpoints"][p][method] for p in ("movement", "stop"))
            lines.append(f"| {key} | {method} | {move['absolute_mse_mean']:.8f} | {stop['absolute_mse_mean']:.8f} | {stop['predicted_displacement_l2_rms']:.6f} |")
    lines += ["", "| H | Original retained source increment | Bounded retained source increment |", "|---:|---:|---:|"]
    for key, h in report["matched_raw"]["horizons"].items():
        lines.append(f"| {key} | {100*h['retention']['original']['signed_recovery']:+.5f}% | {100*h['retention']['bounded']['signed_recovery']:+.5f}% |")
    lines += ["", "The JSON retains every root's source likelihood, support, prototype-wise signed contribution, exact oracle support pair and ties, midpoint orientation efficiency, absolute endpoint errors, stop effects, and all-writes-minus-skip-source increment. The tensor artifact contains all native source and future responses and chronological supports.", "",
        "Forecast boundary: exactly actual z0..z6 enter `predict_prefix`; z7..z10 enter only `score_case`. Independent movement/stop source responses and supports are checked exactly. Original/Gaussian source, endpoint, and interaction numbers are checked against the archived fit when evaluating that checkpoint.", "",
        "Seed 2026111974 remains included; see its per-root signed values and oracle pairs below.", "",
        "| H | Original recovery | Bounded recovery | Gaussian recovery | Oracle slow/fast |", "|---:|---:|---:|---:|---|"]
    seed = next(row for row in report["seeds"] if row["seed"] == 2026111974)
    for key, h in seed["horizons"].items():
        rs = [h["methods"][m + "_source_only"]["signed_recovery"] for m in SOURCE_METHODS]
        lines.append(f"| {key} | " + " | ".join(f"{100*r:+.5f}%" for r in rs) + f" | {h['oracle']['slow_max_prototype']} / {h['oracle']['fast_min_prototype']} |")
    lines += ["", "Validation: `" + json.dumps(report["validation"], ensure_ascii=False) + "`", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DATA / "fitted.pt")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output = output_path(args.output)
    report, tensors = evaluate_checkpoint(args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    torch.save(tensors, output_path(args.output.with_name(args.output.stem + "_tensors.pt")))
    output_path(args.output.with_suffix(".md")).write_text(render_report(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "seconds": report["elapsed_seconds"],
                      "validation": report["validation"]}), flush=True)


if __name__ == "__main__":
    main()
