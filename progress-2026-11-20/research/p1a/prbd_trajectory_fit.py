"""Fixed TwoRoom 2x2 fit: observation geometry x H1/free-trajectory objective.

Every control starts at the archived initial offsets and uses the same 30
eligible factual episodes, frozen noise, seed, Adam(0.001), and 30 updates.
Equation (D) uses every H1 root with H_t=min(4, remaining actual blocks).
Overlapping suffixes form a conditional composite criterion; the episode's
summed joint NLL is divided by its total scored target steps.
"""

from .prbd_paths import EVIDENCE, output_path

import argparse
import json
import math
import platform
from pathlib import Path
from time import perf_counter

import torch

from .prbd_native import NativePrototypes
from .prbd_public_fit import fit, windows
from .prbd_tworoom_speed_fit import OUT as DATA, load_host, single_episodes


OUT = EVIDENCE
CONTROLS = {"original_h1": ("original", "causal", 1),
            "bounded_h1": ("bounded", "causal", 1),
            "original_d": ("original", "trajectory", 4),
            "bounded_d": ("bounded", "trajectory", 4)}


def train_control(name):
    geometry, objective, horizon = CONTROLS[name]
    torch.manual_seed(20261119)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(DATA / "fitted.pt", weights_only=True, map_location="cpu")
    episodes = single_episodes(torch.load(DATA / "encoded_fit.pt", weights_only=True,
                                         map_location="cpu"))
    original_report = json.loads((DATA / "fit.json").read_text(encoding="utf-8"))
    ids = [eid for eid, *_ in episodes]
    if ids != [row["episode"] for row in original_report["training"]] or len(ids) != 30:
        raise ValueError("The frozen fit cohort must match all original 30 eligible episodes")
    model = NativePrototypes(load_host("cpu")).train()
    initial = source["initialized_offsets"]
    with torch.no_grad():
        model.offsets.copy_(initial)
    fixed = {key: source[key] for key in ("noise_cholesky", "transition", "scales", "masses")}
    OUT.mkdir(parents=True, exist_ok=True)
    initial_path = output_path(OUT / "p3_initialized.pt")
    if not initial_path.exists():
        torch.save({**fixed, "offsets": initial, "initialized_offsets": initial}, output_path(initial_path))
    started = perf_counter()
    logs = fit(model, episodes, fixed["noise_cholesky"], fixed["transition"],
               fixed["scales"], fixed["masses"], objective=objective,
               horizon=horizon, geometry=geometry)
    elapsed = perf_counter() - started
    model.eval()
    with torch.no_grad():
        _, z, a, _ = episodes[0]
        window, controls = windows(z[:-1]), windows(a)
        torch.testing.assert_close(model(window, controls)[:, 0],
                                   model.host.predict(window, controls)[:, -1], atol=0, rtol=0)
    checkpoint = {**fixed, "offsets": model.offsets.detach(), "initialized_offsets": initial,
                  "training_geometry": geometry, "training_objective": objective,
                  "training_horizon": horizon}
    checkpoint_path = output_path(OUT / f"p3_{name}.pt")
    torch.save(checkpoint, output_path(checkpoint_path))
    roots = sum(row["transitions"] for row in logs)
    scored = sum(row.get("scored_target_steps", row["transitions"]) for row in logs)
    report = {"control": name, "geometry": geometry, "objective": objective,
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                          "device": "cpu", "torch_threads": torch.get_num_threads()},
              "horizon": horizon, "seed": 20261119, "learning_rate": .001,
              "optimizer": "Adam; only native prototype offsets; one update per episode",
              "horizon_policy": "All original t>=2 roots; min(H, remaining actual blocks)",
              "criterion": "Summed overlapping conditional suffix joint NLL / total target steps"
              if objective == "trajectory" else "Mean causal one-step predictive NLL",
              "joint_mixture": "One fixed-k mixture over the full suffix; all geometries issued ex ante"
              if objective == "trajectory" else None,
              "training_episode_ids": ids, "updates": len(logs), "factual_roots": roots,
              "scored_target_steps": scored, "fit_seconds": elapsed,
              "fixed_noise_source": str((DATA / "fitted.pt").resolve()),
              "nominal_anchor_exact_after_fit": True,
              "offset_change_l2": (model.offsets.detach() - initial).flatten(1).norm(dim=1).tolist(),
              "training": logs, "checkpoint": str(checkpoint_path.resolve())}
    if name == "original_h1":
        report["legacy_regression"] = {
            "offsets_bitwise_equal": torch.equal(model.offsets.detach(), source["offsets"]),
            "nll_max_abs_difference": max(abs(left["nll"] - right["nll"])
                                           for left, right in zip(logs, original_report["training"], strict=True))}
        if not report["legacy_regression"]["offsets_bitwise_equal"]:
            raise RuntimeError("Original H1 no longer reproduces the archived fitted offsets")
    (output_path(OUT / f"p3_{name}_fit.json")).write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{name}: {len(logs)} updates, {roots} factual roots, {scored} scored targets, {elapsed:.3f}s",
          flush=True)
    return report


def summarize_controls():
    """Read the four completed fixed runs; no fitting or selection occurs."""
    rows = []
    for name, (geometry, _, _) in CONTROLS.items():
        fit_log = json.loads((OUT / f"p3_{name}_fit.json").read_text(encoding="utf-8"))
        evaluation = json.loads((OUT / f"p3_{name}_evaluation.json").read_text(encoding="utf-8"))
        panel = evaluation["matched_raw"]
        source = panel["source"][geometry + "_source_only"]
        row = {"control": name, "evaluation_geometry": geometry,
               "updates": fit_log["updates"], "factual_roots": fit_log["factual_roots"],
               "scored_target_steps": fit_log["scored_target_steps"],
               "source_signed_recovery": source["signed_recovery"],
               "source_between_speed_support_TV_mean": source["between_speed_support_TV_mean"],
               "horizons": {}}
        for key, horizon in panel["horizons"].items():
            isolated = horizon["methods"][geometry + "_source_only"]
            retention = horizon["retention"][geometry]
            endpoints = horizon["endpoints"]
            row["horizons"][key] = {
                "source_only_signed_recovery": isolated["signed_recovery"],
                "source_only_relative_squared_error_gain_vs_zero": isolated["relative_squared_error_gain_vs_zero"],
                "source_only_sign_counts": isolated["sign_counts"],
                "directional_oracle_signed_recovery_cap": horizon["oracle_signed_recovery_upper"],
                "all_arrivals_signed_recovery": horizon["methods"][geometry + "_all_arrivals"]["signed_recovery"],
                "skip_source_signed_recovery": horizon["methods"][geometry + "_skip_source"]["signed_recovery"],
                "actual_source_increment_signed_recovery": retention["signed_recovery"],
                "actual_source_increment_interaction_squared_error_gain_sum": retention["interaction_squared_error_gain_sum"],
                "actual_source_increment_sign_counts": retention["sign_counts"],
                "endpoints": {program: {
                    "uniform_mse": endpoints[program]["uniform"]["absolute_mse_mean"],
                    "source_only_mse": endpoints[program][geometry + "_source_only"]["absolute_mse_mean"],
                    "all_arrivals_mse": endpoints[program][geometry + "_all_arrivals"]["absolute_mse_mean"],
                    "skip_source_mse": endpoints[program][geometry + "_skip_source"]["absolute_mse_mean"],
                    "source_only_predicted_displacement_l2_rms": endpoints[program][geometry + "_source_only"]["predicted_displacement_l2_rms"]}
                    for program in ("movement", "stop")}}
        rows.append(row)
    cohort = [json.loads((OUT / f"p3_{name}_fit.json").read_text(encoding="utf-8"))["training_episode_ids"]
              for name in CONTROLS]
    if not all(ids == cohort[0] for ids in cohort):
        raise ValueError("The 2x2 controls must share the same factual episodes")
    summary = {"training_episode_ids": cohort[0], "matched_raw_roots": 11,
               "matched_physical_nonzero_roots": 9, "physical_zero_roots_included": 2,
               "all_complete_held_roots": 15, "total_held_seeds": 16,
               "horizon_policy": "All 418 original roots; H_t=min(4, remaining), 1492 overlapping target steps for D",
               "normalization": "Per episode: sum conditional suffix joint NLL / total scored target steps",
               "controls": rows}
    bounded = json.loads((OUT / "p3_bounded_d_evaluation.json").read_text(encoding="utf-8"))
    matched = [row for row in bounded["seeds"] if row["matched"]]
    influence = []
    for row in matched:
        h4 = row["horizons"]["4"]
        influence.append({"seed": row["seed"], "truth_squared_norm": h4["truth_squared_norm"],
                          "physical_truth_zero": h4["physical_truth_zero"],
                          "latent_truth_zero": h4["truth_squared_norm"] == 0,
                          "source_only_numerator": h4["methods"]["bounded_source_only"]["prediction_truth_dot"],
                          "actual_source_increment_numerator": h4["retention"]["bounded"]["prediction_truth_dot"]})
    denominator = math.fsum(row["truth_squared_norm"] for row in influence)
    sums = {name: math.fsum(row[name] for row in influence)
            for name in ("source_only_numerator", "actual_source_increment_numerator")}
    leave_one_out = [{"omitted_seed": row["seed"], "omitted_physical_truth_zero": row["physical_truth_zero"],
                      **{name.removesuffix("_numerator") + "_signed_recovery":
                         (total - row[name]) / (denominator - row["truth_squared_norm"])
                         for name, total in sums.items()}} for row in influence]
    influence_summary = {"per_seed": influence, "leave_one_seed_out": leave_one_out,
                         "ranges": {name: {"minimum": min(row[name] for row in leave_one_out),
                                           "maximum": max(row[name] for row in leave_one_out)}
                                    for name in ("source_only_signed_recovery", "actual_source_increment_signed_recovery")},
                         "strongest_numerators": {name: {
                             "positive": sorted((row for row in influence if row[name] > 0), key=lambda row: -row[name])[:3],
                             "negative": sorted((row for row in influence if row[name] < 0), key=lambda row: row[name])[:3]}
                             for name in sums}}
    tensors = torch.load(OUT / "p3_bounded_d_evaluation_tensors.pt", weights_only=True, map_location="cpu")
    import numpy as np

    with np.load(DATA / "data_diagnostics.npz", allow_pickle=False) as archive:
        physical_positions = archive["validation_positions"].copy()
    with np.load(DATA / "data_validation.npz", allow_pickle=False) as archive:
        raw_actions, held_seeds = archive["actions"].copy(), archive["seed"].copy()
    held_lookup = {int(seed): index for index, seed in enumerate(held_seeds)}
    timelines = []
    for case in tensors["cases"]:
        if not case["matched"]:
            continue
        idx = held_lookup[case["seed"]]
        latent = torch.cat((case["source_actual_arrivals"][:, 0, None],
                            case["actual_root"][:, 0]), dim=1)
        # Independent program histories were checked exactly by the evaluator;
        # retain one copy per actual slow/fast history to avoid duplicate rows.
        chronological = case["chronological_supports"][1, :, 0]
        histories = []
        for speed in (0, 1):
            all_writes, skip_source = chronological[speed]
            arrivals = []
            for j, label in enumerate(("retract_z4", "zero_z5", "zero_z6"), start=1):
                block = j + 2
                physical_delta = physical_positions[idx, speed, 0, 5 * (block + 1)] - physical_positions[idx, speed, 0, 5 * block]
                latent_delta = latent[speed, j] - latent[speed, j-1]
                arrivals.append({"arrival": label, "raw_action_all_zero": bool((raw_actions[idx, speed, 0, block] == 0).all()),
                                 "physical_displacement_xy": physical_delta.tolist(),
                                 "physical_displacement_zero": bool((physical_delta == 0).all()),
                                 "latent_change_l2": float(latent_delta.norm()),
                                 "latent_change_exact_zero": bool((latent_delta == 0).all()),
                                 "all_write_support": all_writes[j].tolist(),
                                 "skip_source_support": skip_source[j].tolist(),
                                 "all_write_update_TV": float((all_writes[j] - all_writes[j-1]).abs().sum() / 2),
                                 "retained_source_increment_TV": float((all_writes[j] - skip_source[j]).abs().sum() / 2)})
            histories.append({"speed_index_for_scoring_only": speed,
                              "source_z3_support": all_writes[0].tolist(), "arrivals": arrivals})
        timelines.append({"seed": case["seed"], "histories": histories})
    chronology_values = torch.cat([case["chronological_supports"].flatten() for case in tensors["cases"]])
    summary["bounded_d_influence"] = influence_summary
    summary["bounded_d_posterior_timelines"] = timelines
    summary["bounded_d_held_filter_numerics"] = {
        "minimum_posterior_mass": float(chronology_values.min()),
        "zero_mass_count": int((chronology_values == 0).sum()),
        "all_finite": bool(torch.isfinite(chronology_values).all()),
        "scope": "All 15 complete held prefixes, original and bounded observation geometries; no nominal one-hot control"}
    (output_path(OUT / "p3_comparison.json")).write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# P3: fixed single-track trajectory learning", "",
        "All four controls use the same archived initial offsets, frozen official three-frame TwoRoom LeWM, calibrated R, 30 eligible actual fit episodes, seed 20261119, Adam at 0.001 and 30 updates. No tuning or further collection was performed.", "",
        "Every original H1 root remains enrolled. At each root t, H_t=min(4, remaining actual blocks); D scores 1492 overlapping target steps from the same 418 roots. Each episode's joint NLL sum is divided by its scored target steps. These overlapping suffixes define a conditional composite criterion rather than independent observations.", "",
        "For D, each fixed prototype freely rolls its own native window. Before scoring future truth, every horizon's geometry is signed from the full K-path family and the same causal root support. The product of normalized per-horizon observation densities is mixed over k once. Only the first actual arrival advances the public factual filter; its gradient history remains connected. Identity transport is required for this fixed-condition suffix model.", "",
        "The unchanged held panel has 16 seeds, 15 complete roots, 11 raw matched roots and 9 physical-nonzero roots. Both physical-zero roots remain included. Percentages below are pooled signed projection recovery. Each row is evaluated using its training observation geometry; each separate evaluation JSON also retains all three frozen source-likelihood alternatives.", "",
        "| Training | Source recovery | Source slow/fast TV | H1 source-only | H4 source-only | H4 directional cap | H4 actual source increment |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        h1, h4 = (row["horizons"][str(h)] for h in (1, 4))
        lines.append(f"| {row['control']} | {100*row['source_signed_recovery']:+.4f}% | {row['source_between_speed_support_TV_mean']:.6f} | {100*h1['source_only_signed_recovery']:+.4f}% | {100*h4['source_only_signed_recovery']:+.4f}% | {100*h4['directional_oracle_signed_recovery_cap']:.4f}% | {100*h4['actual_source_increment_signed_recovery']:+.4f}% |")
    lines += ["", "Actual source increment compares all actual z3..z6 writes against replaying the same factual sequence while omitting only source z3. It is distinct from freezing source-only support until the query.", "",
        "Bounded D gives a positive pooled isolated-source effect, but only 3 of the 9 physical-nonzero roots have the correct H4 sign. Its all-arrivals H4 recovery is -5.3210%, and the actual source increment is -1.5839%. At H4, movement/stop source-only endpoint MSE is 0.24547/0.10510 versus uniform 0.23604/0.09471. Thus the fixed run establishes an implemented training path and a partial isolated effect; sustained useful prediction is not established.", "",
        "Validation: original H1 reproduces the archived fitted offsets bitwise and every recorded NLL exactly. Synthetic original/bounded checks verify H1 loss/gradient identity, a normalized expanded Gaussian-mixture joint, one fixed-k mixture, own windows, terminal indexing, historical-evidence gradients, no future public write, and finite gradients at zero response rank. See p3_checks.json. Every fitted control retains an exact nominal F0 anchor. The 2x2 budget matches updates rather than forward work: D fits took about 12–13 seconds versus about 2–3 seconds for H1 on CPU.", "",
        "Artifacts: p3_<control>.pt; p3_<control>_fit.json; p3_<control>_evaluation.json/.md/_tensors.pt; p3_comparison.json. The evaluation artifacts retain absolute endpoints, stop responses, every root's prototype supports and likelihoods, directional caps, and actual source increments. No hidden-condition reversal or planning/P4 run was performed.", ""]
    ranges = influence_summary["ranges"]
    lines += ["## Bounded-D root influence and actual arrivals", "",
        f"Leaving out one of the 11 matched reset seeds gives H4 source-only recovery from {100*ranges['source_only_signed_recovery']['minimum']:+.4f}% to {100*ranges['source_only_signed_recovery']['maximum']:+.4f}%; the actual source increment ranges from {100*ranges['actual_source_increment_signed_recovery']['minimum']:+.4f}% to {100*ranges['actual_source_increment_signed_recovery']['maximum']:+.4f}%. Physical-zero and latent-zero future interactions are recorded separately in JSON; both frozen zero cases are retained.", "",
        "| Seed | H4 source-only signed numerator | H4 actual source increment numerator | Physical/latent zero truth |",
        "|---|---:|---:|---|"]
    for row in sorted(influence, key=lambda row: -row["source_only_numerator"]):
        lines.append(f"| {row['seed']} | {row['source_only_numerator']:+.6f} | {row['actual_source_increment_numerator']:+.6f} | {row['physical_truth_zero']}/{row['latent_truth_zero']} |")
    lines += ["", "The JSON also retains each slow/fast history's posterior after source z3, retract z4, and the zero-command arrivals z5/z6. It records each arrival's raw-zero action flag, actual physical displacement, exact latent-change flag, update TV, and retained all-minus-skip-source TV. Physical stationarity and latent stationarity are measured independently.", "",
        f"Held filter numerical audit: minimum posterior mass {float(chronology_values.min()):.6g}, zero masses {int((chronology_values == 0).sum())}; all values finite. Likelihood products and their joint mixture are computed in log space. This fixed positive-node Gaussian mixture is normalized, but does not inherit the continuous inverse-gamma mixture's infinite-tail likelihood-ratio guarantee. No clipping, temperature, or mass floor was introduced.", ""]
    (output_path(OUT / "p3.md")).write_text("\n".join(lines), encoding="utf-8")
    print(OUT / "p3_comparison.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", nargs="*", metavar="CONTROL")
    parser.add_argument("--summarize", action="store_true",
                        help="Summarize all four completed fit and frozen-evaluation artifacts")
    args = parser.parse_args()
    if not args.controls and not args.summarize:
        parser.error("Select at least one control or --summarize")
    if any(name not in CONTROLS for name in args.controls):
        parser.error("Controls must be one of: " + ", ".join(CONTROLS))
    for name in args.controls:
        train_control(name)
    if args.summarize:
        summarize_controls()


if __name__ == "__main__":
    main()
