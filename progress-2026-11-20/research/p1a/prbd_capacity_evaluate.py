"""Frozen physical readout, fresh records, and held-out oracle capacity scoring.

The physical state is diagnostic privilege: it fits a training-only linear
readout and assigns known-speed oracle branches, never a deployed estimator.
No future target is accepted by the forecast functions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .prbd_native import encode_native_controls
from .prbd_tworoom_speed_fit import load_host, RELEASE
from .prbd_paths import ROOT, output_path


OUT = ROOT / "runs/prbd_capacity_phase1_20261121"
OLD = ROOT / "runs/prbd_consistency_20261120"
HORIZONS = (1, 4)
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 4000, 20261121


def write_json(path, value):
    output_path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def decode(z, coef):
    return z.double() @ coef[:-1] + coef[-1]


def readout_metadata(reader, path):
    namespace = reader.get("record_namespace", "simulator_reset_seeds" if "training_seeds" in reader else None)
    if namespace not in ("simulator_reset_seeds", "author_tworoom_training_hdf5"):
        raise ValueError("Physical readout requires an explicit, supported record namespace")
    identity_key = "training_episode_ids" if namespace == "author_tworoom_training_hdf5" else "training_seeds"
    if identity_key not in reader:
        raise ValueError(f"Physical readout is missing {identity_key}")
    if namespace == "author_tworoom_training_hdf5" and "training_seeds" in reader:
        raise ValueError("Author HDF5 episode rows must not be represented as simulator reset seeds")
    return {"record_namespace": namespace, "training_frames": int(reader["training_frames"]),
            identity_key: torch.as_tensor(reader[identity_key]).tolist(),
            "construction": reader.get("construction", "FP64 ordinary linear least squares with intercept"),
            "path": str(Path(path).resolve())}


def eligibility(batch, positions, raw):
    included, excluded = [], []
    for index, seed in enumerate(batch["seed"].tolist()):
        states, controls, p = batch["z"][index], batch["a"][index], positions[index]
        root, past = states[:, :, 4:7], controls[:, :, 4:6]
        checks = {"all_endpoints_complete": bool((batch["lengths"][index] >= 10).all()),
                  "actual_rgb_matched": bool(batch["matched_rgb"][index]),
                  "native_root_exactly_matched": torch.equal(root, root[:1, :1].expand_as(root)),
                  "recent_controls_exactly_matched": torch.equal(past, past[:1, :1].expand_as(past)),
                  "same_program_controls_equal_across_speeds": torch.equal(controls[0], controls[1]),
                  "physical_root_exactly_matched": torch.equal(p[:, :, [20, 25, 30]], p[:1, :1, [20, 25, 30]].expand(2, 2, -1, -1)),
                  "program_actual_prefix_exactly_replayed": torch.equal(states[:, 0, :7], states[:, 1, :7]) and torch.equal(controls[:, 0, :6], controls[:, 1, :6]),
                  "raw_zero_references_exact": torch.equal(controls[:, :, 4:6], batch["null_control"].expand(2, 2, 2, -1)),
                  "same_raw_future_across_speeds": bool(np.array_equal(raw[index, 0], raw[index, 1]))}
        if all(checks.values()):
            included.append(index)
        else:
            excluded.append({"seed": seed, "checks": checks, "reasons": [k for k, v in checks.items() if not v]})
    return included, excluded


@torch.no_grad()
def forecast(model, states_prefix, controls):
    """Only the last three actual states; all four future tokens are free."""
    if states_prefix.shape != (2, 2, 7, 192):
        raise ValueError("Forecast boundary must contain actual z0..z6 only")
    result = []
    for speed in range(2):
        result.append(torch.stack([model.rollout(states_prefix[speed, p, 4:7][None],
                                               controls[speed, p, 4:10][None])[0]
                                   for p in range(2)]))
    result = torch.stack(result).double()
    if not torch.equal(result[0], result[1]):
        raise AssertionError("Matched roots must emit the same branch family across speeds")
    return result


@torch.no_grad()
def frozen_forecast(host, states_prefix, embedded):
    paths = []
    for speed in range(2):
        programs = []
        for program in range(2):
            window = states_prefix[speed, program, 4:7][None]
            controls = embedded[speed, program, 4:10]
            values = []
            for h in range(4):
                value = host.predict(window, controls[None, h:h+3])[:, -1]
                values.append(value[0])
                window = torch.cat((window[:, 1:], value[:, None]), 1)
            programs.append(torch.stack(values))
        paths.append(torch.stack(programs))
    return torch.stack(paths).double()


def contrast(value):
    return value[0, 0] - value[0, 1] - value[1, 0] + value[1, 1]


def vector_score(prediction, truth):
    return {"prediction": prediction.tolist(), "truth": truth.tolist(),
            "dot": float(prediction @ truth), "truth_squared_norm": float(truth.square().sum()),
            "error_squared_norm": float((prediction-truth).square().sum()),
            "prediction_squared_norm": float(prediction.square().sum())}


def score(seed, states, positions, methods, coef, direction, free_reference):
    row = {"seed": seed, "probe_direction": direction, "free_unsaturated_reference": free_reference, "horizons": {}}
    root_z, root_xy = states[:, :, 6].double(), positions[:, :, 30].double()
    root_decoded = decode(root_z, coef)
    for h in HORIZONS:
        target_z, target_xy = states[:, :, 6+h].double(), positions[:, :, 30+5*h].double()
        truth_displacement = target_xy-root_xy
        target_decoded = decode(target_z, coef)
        true_interaction_z, true_interaction_xy = contrast(target_z), contrast(truth_displacement)
        item = {"physical_nonzero": bool((true_interaction_xy != 0).any()),
                "true_displacement_xy": truth_displacement.tolist(),
                "real_encoding_readout": {
                    "absolute_position_error_xy": (target_decoded-target_xy).tolist(),
                    "root_position_error_xy": (root_decoded-root_xy).tolist(),
                    "displacement_error_xy": (target_decoded-root_decoded-truth_displacement).tolist(),
                    "physical_interaction": vector_score(contrast(target_decoded-root_decoded), true_interaction_xy)},
                "methods": {}}
        for name, path in methods.items():
            pred_z = path[:, :, h-1]
            pred_displacement = decode(pred_z, coef)-root_decoded
            error_xy = pred_displacement-truth_displacement
            item["methods"][name] = {"latent_interaction": vector_score(contrast(pred_z), true_interaction_z),
                "physical_interaction": vector_score(contrast(pred_displacement), true_interaction_xy),
                "latent_endpoint_mse_by_speed_program": (pred_z-target_z).square().mean(-1).tolist(),
                "physical_displacement_error_xy": error_xy.tolist(),
                "physical_displacement_L2_by_speed_program": error_xy.norm(dim=-1).tolist(),
                "physical_displacement_prediction_xy": pred_displacement.tolist()}
        row["horizons"][str(h)] = item
    return row


def ratio(a, b):
    return float(a/b) if b > 0 else None


def ci(value):
    return np.quantile(value, [.025, .975], axis=0).tolist()


def summarize(rows):
    result = {"reset_count": len(rows), "seeds": [r["seed"] for r in rows], "horizons": {}}
    if not rows:
        return result
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(rows), (BOOTSTRAP_DRAWS, len(rows)))
    for h in map(str, HORIZONS):
        scored = [r["horizons"][h] for r in rows]
        methods = {}
        for name in scored[0]["methods"]:
            stats = {}
            for space in ("latent", "physical"):
                values = [r["methods"][name][space+"_interaction"] for r in scored]
                dot = np.array([v["dot"] for v in values]); norm = np.array([v["truth_squared_norm"] for v in values])
                error = np.array([v["error_squared_norm"] for v in values])
                denom = norm[draws].sum(-1); valid = denom > 0
                stats[space+"_interaction"] = {"signed_recovery": ratio(dot.sum(), norm.sum()),
                    "signed_recovery_ci95": ci(dot[draws].sum(-1)[valid]/denom[valid]) if valid.any() else None,
                    "truth_squared_norm_sum": float(norm.sum()), "error_squared_norm_sum": float(error.sum()),
                    "positive_direction_roots": int((dot > 0).sum()), "negative_direction_roots": int((dot < 0).sum()),
                    "zero_direction_roots": int((dot == 0).sum()),
                    "leave_one_out_signed_recovery": [ratio(dot.sum()-d, norm.sum()-n) for d, n in zip(dot, norm)]}
            lat = np.array([r["methods"][name]["latent_endpoint_mse_by_speed_program"] for r in scored])
            phy = np.array([r["methods"][name]["physical_displacement_error_xy"] for r in scored])
            root_lat = lat.mean(1)
            stats["latent_endpoint_MSE_by_program"] = root_lat.mean(0).tolist()
            stats["latent_endpoint_MSE_by_program_ci95"] = ci(root_lat[draws].mean(1))
            stats["physical_displacement_RMSE_by_program_map_units"] = np.sqrt((phy**2).mean((0, 1, 3))).tolist()
            stats["physical_displacement_mean_L2_by_program_map_units"] = np.linalg.norm(phy, axis=-1).mean((0, 1)).tolist()
            stats["physical_displacement_RMSE_by_program_ci95"] = ci(np.sqrt((phy[draws]**2).mean((1, 2, 4))))
            methods[name] = stats
        real = [r["real_encoding_readout"] for r in scored]
        absolute = np.array([r["absolute_position_error_xy"] for r in real])
        displacement = np.array([r["displacement_error_xy"] for r in real])
        comparisons = {}
        for label, baseline in (("oracle_vs_frozen", "nominal_frozen"), ("oracle_vs_pooled", "pooled_single_branch")):
            if "oracle_speed_branch" not in methods:
                continue
            comparison = {}
            for space in ("latent", "physical"):
                before = np.array([r["methods"][baseline][space+"_interaction"]["error_squared_norm"] for r in scored])
                after = np.array([r["methods"]["oracle_speed_branch"][space+"_interaction"]["error_squared_norm"] for r in scored])
                gain = before-after
                denominator = before[draws].sum(-1)
                valid = denominator > 0
                comparison[space+"_interaction"] = {"squared_error_gain_sum": float(gain.sum()),
                    "relative_squared_error_gain": ratio(gain.sum(), before.sum()),
                    "relative_squared_error_gain_ci95": ci(gain[draws].sum(-1)[valid]/denominator[valid]) if valid.any() else None,
                    "improved_roots": int((gain > 0).sum())}
            latent_gain = np.array([np.asarray(r["methods"][baseline]["latent_endpoint_mse_by_speed_program"])
                -np.asarray(r["methods"]["oracle_speed_branch"]["latent_endpoint_mse_by_speed_program"]) for r in scored]).mean(1)
            physical_gain = np.array([(np.asarray(r["methods"][baseline]["physical_displacement_error_xy"])**2
                -np.asarray(r["methods"]["oracle_speed_branch"]["physical_displacement_error_xy"])**2).mean((0, 2)) for r in scored])
            comparison.update({"latent_endpoint_MSE_gain_by_program": latent_gain.mean(0).tolist(),
                "latent_endpoint_MSE_gain_by_program_ci95": ci(latent_gain[draws].mean(1)),
                "physical_displacement_MSE_gain_by_program_map_units_squared": physical_gain.mean(0).tolist(),
                "physical_displacement_MSE_gain_by_program_ci95": ci(physical_gain[draws].mean(1)),
                "latent_endpoint_improved_roots_by_program": (latent_gain > 0).sum(0).tolist(),
                "physical_displacement_improved_roots_by_program": (physical_gain > 0).sum(0).tolist()})
            comparisons[label] = comparison
        result["horizons"][h] = {"methods": methods, "comparisons": comparisons, "real_encoding_readout": {
            "absolute_position_RMSE_map_units": float(np.sqrt((absolute**2).mean())),
            "absolute_position_RMSE_by_program_map_units": np.sqrt((absolute**2).mean((0, 1, 3))).tolist(),
            "displacement_RMSE_by_program_map_units": np.sqrt((displacement**2).mean((0, 1, 3))).tolist()}}
    return result


def checks():
    coef = torch.zeros(193, 2, dtype=torch.float64)
    coef[:2] = torch.eye(2)
    z = torch.zeros(2, 2, 11, 192, dtype=torch.float64)
    xy = torch.zeros(2, 2, 51, 2, dtype=torch.float64)
    for speed, delta in ((0, 1.), (1, 3.)):
        z[speed, 0, 7:, 0] = delta
        xy[speed, 0, 35:, 0] = delta
    methods = {"perfect": z[:, :, 7:].clone(), "frozen": z[:, :, 7:]*0}
    row = score(1, z, xy, methods, coef, "up", True)
    result = summarize([row])
    for h in ("1", "4"):
        methods = result["horizons"][h]["methods"]
        assert methods["perfect"]["physical_interaction"]["signed_recovery"] == 1.
        assert methods["perfect"]["latent_interaction"]["signed_recovery"] == 1.
        assert methods["frozen"]["physical_interaction"]["signed_recovery"] == 0.
        assert methods["perfect"]["physical_displacement_RMSE_by_program_map_units"] == [0., 0.]
        assert result["horizons"][h]["real_encoding_readout"]["absolute_position_RMSE_map_units"] == 0.
    report = {"perfect_latent_and_physical_recovery": True, "zero_frozen_recovery": True,
              "exact_physical_displacement_fixture": True, "forecast_boundary_accepts_no_future": True}
    write_json(OUT / "evaluation_checks.json", report)
    print(json.dumps(report), flush=True)


def render_report(report):
    coverage = report["coverage"]
    meta = report.get("physical_readout_metadata", {"record_namespace": "simulator_reset_seeds", "training_frames": 1056})
    corpus = "the author's released TwoRoom training HDF5" if meta["record_namespace"] == "author_tworoom_training_hdf5" else "the old simulator-reset G3 training split"
    description = f"{corpus} ({meta['training_frames']} actual encoded frames)"
    lines = [f"# Phase 1 {report['split']} oracle capacity", "",
        f"Offered {coverage['offered']}; matched {coverage['matched']}; physical nonzero {coverage['physical_nonzero']}. All exclusions and per-root scores remain in JSON.", "",
        f"Oracle labels select the preassigned slow/fast branches. This diagnoses supervised native capacity, without evaluating a deployed belief estimator. Physical decoding is one OLS readout fitted only to {description} and frozen before this held-out scoring. Its source record namespace is {meta['record_namespace']}.", "",
        "| H | Method | Latent recovery | Physical recovery | Move latent MSE | Stop latent MSE | Move displacement RMSE | Stop displacement RMSE |",
        "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for h, group in report["panels"]["physical_nonzero"]["horizons"].items():
        for name, m in group["methods"].items():
            values = [100*m[s+"_interaction"]["signed_recovery"] for s in ("latent", "physical")]
            values += m["latent_endpoint_MSE_by_program"] + m["physical_displacement_RMSE_by_program_map_units"]
            lines.append(f"| {h} | {name} | " + " | ".join(f"{v:.7f}" for v in values) + " |")
        real = group["real_encoding_readout"]
        lines += ["", f"H{h} actual encoded endpoint absolute XY RMSE: {real['absolute_position_RMSE_map_units']:.6f} map units; actual encoded displacement RMSE (move, stop): {real['displacement_RMSE_by_program_map_units']}.", ""]
    lines += ["Signed recoveries are percentages. RMSE uses the mean over both XY coordinates, speeds and roots, separately for movement/stop. Physical displacement is D(predicted latent) minus D(actual root latent), compared with actual endpoint-minus-root position. Physical recovery projects the predicted speed-by-program interaction onto the corresponding real interaction. The raw-velocity frozen control uses true speed only for action scaling before official normalization; its collision-free, unsaturated subset is reported separately.", "",
        "The JSON also retains 4,000 paired-reset bootstrap intervals, direction strata, leave-one-root-out recovery, physical-zero/all-matched panels, endpoint vectors, actual-encoding position/displacement errors, and all free latent trajectories in the accompanying PT file.", ""]
    return "\n".join(lines)


def rescore_readout(source, reader_path, destination):
    """Score an independently frozen linear readout on saved predictions only."""
    source, reader_path, destination = map(Path, (source, reader_path, destination))
    destination = output_path(destination)
    previous = json.loads(source.read_text(encoding="utf-8"))
    split = previous["split"]
    directory = OLD if split == "dev" else OUT
    reader = torch.load(reader_path, map_location="cpu", weights_only=True)
    cases = torch.load(source.with_name(source.stem+"_tensors.pt"), map_location="cpu", weights_only=True)["cases"]
    batch = torch.load(directory / f"{split}.pt", map_location="cpu", weights_only=True)
    lookup = {int(seed): index for index, seed in enumerate(batch["seed"])}
    previous_rows = {row["seed"]: row for row in previous["rows"]}
    with np.load(directory / f"diag_{split}.npz", allow_pickle=False) as data:
        positions = torch.from_numpy(data["positions"].copy()).double()
    metadata = readout_metadata(reader, reader_path)
    if metadata["record_namespace"] == "simulator_reset_seeds" and set(lookup) & set(metadata["training_seeds"]):
        raise ValueError("Physical readout training includes held-out simulator reset seeds")
    rows = []
    for case in cases:
        seed = case["seed"]
        original = previous_rows[seed]
        states = torch.cat((case["actual_prefix_z"], case["true_future"]), dim=2)
        row = score(seed, states, positions[lookup[seed]], case["predictions"], reader["coef"],
                    original["probe_direction"], original["free_unsaturated_reference"])
        for h in map(str, HORIZONS):
            for name in row["horizons"][h]["methods"]:
                if row["horizons"][h]["methods"][name]["latent_interaction"] != original["horizons"][h]["methods"][name]["latent_interaction"]:
                    raise AssertionError("Readout rescoring changed a latent prediction score")
        rows.append(row)
    nonzero = [r for r in rows if r["horizons"]["1"]["physical_nonzero"]]
    previous["panels"] = {"all_matched": summarize(rows), "physical_nonzero": summarize(nonzero),
        "physical_zero": summarize([r for r in rows if not r["horizons"]["1"]["physical_nonzero"]]),
        "free_unsaturated_physical_nonzero": summarize([r for r in nonzero if r["free_unsaturated_reference"]])}
    previous["by_probe_direction"] = {d: summarize([r for r in nonzero if r["probe_direction"] == d]) for d in ("left", "right", "up", "down")}
    previous["rows"] = rows
    previous["physical_readout"] = str(reader_path.resolve())
    previous["physical_readout_metadata"] = metadata
    previous["readout_rescored_from_fixed_predictions"] = str(source.resolve())
    write_json(destination, previous)
    destination.with_suffix(".md").write_text(render_report(previous), encoding="utf-8")
    return previous


@torch.no_grad()
def evaluate(checkpoint, split):
    from .prbd_capacity_oracle import load_oracle_checkpoint
    started = perf_counter()
    directory = OLD if split == "dev" else OUT
    batch = torch.load(directory / f"{split}.pt", map_location="cpu", weights_only=True)
    reader = torch.load(OUT / "physical_readout.pt", map_location="cpu", weights_only=True)
    fitted = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = readout_metadata(reader, OUT / "physical_readout.pt")
    readout_seeds = set(metadata.get("training_seeds", []))
    if set(batch["seed"].tolist()) & (set(fitted["training_seeds"]) | readout_seeds):
        raise ValueError("Held-out evaluation overlaps a training seed")
    model = load_oracle_checkpoint(checkpoint, device="cpu").eval().requires_grad_(False)
    host = model.host
    with np.load(directory / f"diag_{split}.npz", allow_pickle=False) as data:
        positions = torch.from_numpy(data["positions"].copy()).double()
        collision = data["collision"].copy()
    with np.load(directory / f"data_{split}.npz", allow_pickle=False) as data:
        raw = data["actions"].copy()
    eligible, excluded = eligibility(batch, positions, raw)
    stats = json.loads((RELEASE / "action_statistics.json").read_text(encoding="utf-8"))
    rows, saved = [], []
    directions = ("left", "right", "up", "down")
    for index in eligible:
        seed = int(batch["seed"][index]); z = batch["z"][index]; a = batch["a"][index]
        family = forecast(model, z[:, :, :7], a)
        methods = {"oracle_speed_branch": torch.stack((family[0, :, 1], family[1, :, 2])),
                   "nominal_frozen": family[:, :, 0], "pooled_single_branch": family[:, :, 3]}
        scales = np.array([.5, 1.5], dtype=np.float32)[:, None, None, None, None]
        scaled_raw = raw[index] * scales
        scaled_controls = encode_native_controls(host, scaled_raw, stats)
        methods["oracle_velocity_frozen"] = frozen_forecast(host, z[:, :, :7], scaled_controls)
        nominal = frozen_forecast(host, z[:, :, :7], a)
        if not torch.equal(nominal, methods["nominal_frozen"]):
            raise AssertionError("Oracle native branch0 differs from frozen host")
        free = not bool(collision[index, :, :, 6:10].any()) and bool((np.abs(scaled_raw[:, :, 4:10]) < 1).all())
        row = score(seed, z, positions[index], methods, reader["coef"], directions[index % 4], free)
        rows.append(row)
        saved.append({"seed": seed, "future_family": family, "predictions": methods,
                      "true_future": z[:, :, 7:11], "physical_future": positions[index, :, :, 35:51:5],
                      "actual_root_window": z[:, :, 4:7], "actual_prefix_z": z[:, :, :7],
                      "future_controls_with_history": a[:, :, 4:10],
                      "future_raw_controls_with_history": torch.from_numpy(raw[index, :, :, 4:10].copy()),
                      "physical_root": positions[index, :, :, 30],
                      "future_branch_action_effect": family[:, 0]-family[:, 1]})
    nonzero = [r for r in rows if r["horizons"]["1"]["physical_nonzero"]]
    panels = {"all_matched": summarize(rows), "physical_nonzero": summarize(nonzero),
              "physical_zero": summarize([r for r in rows if not r["horizons"]["1"]["physical_nonzero"]]),
              "free_unsaturated_physical_nonzero": summarize([r for r in nonzero if r["free_unsaturated_reference"]])}
    report = {"checkpoint": str(checkpoint.resolve()), "split": split, "oracle_privilege": "Known true speed selects branch1/2; not a deployed belief result",
        "physical_units": "Official TwoRoom map coordinates; 224x224 render, five primitives per model block",
        "physical_readout": str(OUT / "physical_readout.pt"), "test_fitting": False,
        "physical_readout_metadata": metadata,
        "forecast_boundary": "Only actual z0..z6 accepted; z7..z10 and physics consumed after all forecasts",
        "coverage": {"offered": len(batch["seed"]), "matched": len(rows), "physical_nonzero": len(nonzero), "excluded": excluded},
        "panels": panels, "by_probe_direction": {d: summarize([r for r in nonzero if r["probe_direction"] == d]) for d in directions},
        "rows": rows, "elapsed_seconds": perf_counter()-started, "phase": 1,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "paired reset"}}
    output = output_path(OUT / f"{checkpoint.parent.name}_{checkpoint.stem}_{split}_evaluation.json")
    write_json(output, report)
    output.with_suffix(".md").write_text(render_report(report), encoding="utf-8")
    torch.save({"cases": saved}, output.with_name(output.stem+"_tensors.pt"))
    brief = {h: {name: {space+"_recovery": m[space+"_interaction"]["signed_recovery"] for space in ("latent", "physical")}
                 for name, m in group["methods"].items()} for h, group in panels["physical_nonzero"]["horizons"].items()}
    print(json.dumps({"output": str(output), "coverage": report["coverage"], "summary": brief, "seconds": report["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("evaluate", "check"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--split", choices=("dev", "fresh"), default="dev")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.stage == "check":
        checks()
    elif args.checkpoint is None:
        parser.error("--checkpoint required")
    else:
        evaluate(args.checkpoint, args.split)


if __name__ == "__main__":
    main()
