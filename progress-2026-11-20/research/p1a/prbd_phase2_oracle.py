"""One prespecified labelled capacity diagnostic for the Phase2 response repair.

Only training diagnostics supply labels during fitting. Independent static and
switch labels select a branch after the final checkpoint is frozen; they do not
modify weights. This is a capacity witness, not a deployable inference method.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import torch

from .prbd_paths import REPAIR
from .prbd_phase2 import BUDGET, OUT as DATA, load_episodes
from .prbd_phase2_response import load_response_state, make_response_model, response_state


OUT = REPAIR / "oracle"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(name):
    return torch.load(DATA / name, map_location="cpu", weights_only=True)


def protocol():
    return {
        "question": "Can the native response family, with author-comparable offline capacity, express the actual speed-dependent futures?",
        "budget": dict(BUDGET),
        "parameters": {"shared_final_block": 1798400, "shared_final_norm": 384,
                       "shared_pred_proj": 792768, "prototype_offsets": 3456,
                       "total": 2595008, "author_AdaJEPA": 2591552},
        "architecture": "existing NativePrototypes; literal F0 nominal; shared independently copied native final block/norm/pred_proj for k>0",
        "initialization": "released host suffix copied exactly; original deterministic seed20261119 std0.01 offsets",
        "module_semantics": "eval dropout and frozen BatchNorm running moments; offline gradient updates only",
        "train_data": str(DATA / "train.pt"),
        "train_labels": str(DATA / "diag_train.npz"),
        "train_label_use": "actual speed of each executed action selects slow branch1 or fast branch2 during free H1..4 rollout; no coordinates or counterfactuals",
        "branch3": "retained native fourth prototype; no labelled assignment; offsets remain at initialization, shared suffix is trained",
        "objective": "equal episodes; equal all legal roots t>=2; equal legal H1..4; each H averages raw unwhitened squared latent error over its full prefix and 192 dimensions",
        "fixed_scale": 1.0,
        "objective_difference_from_method": "supervised raw latent squared error; no G1 likelihood, belief, whitening or min-deviation penalty",
        "selection": "single predeclared parameterization; final update2048 only; no test-dependent tuning or early stopping",
        "independent_test_access": "only after final checkpoint; exact original48 static and48 paired switch roots, all valid transitions retained",
        "test_oracle": "true speed at the first issued action selects one branch fixed over the whole candidate; no future speed or cut point selects later rollout branches",
        "test_inputs": "same public actual three-frame window and same embedded controls as every Phase2 arm",
        "switch_metrics": "original cut10/20 windows and both directions; paired bootstrap samples original48 reset IDs, keeping arms/cuts together",
        "uncertainty": "4000 paired root bootstrap draws, seed20261121;95% percentile intervals",
        "units": "exact raw latent endpoint MSE mean192, and unscaled latent interaction SSE sum192",
        "deployment_updates": 0,
        "test_labels_are_privileged_diagnostic_only": True,
        "no_null_input": True,
        "no_parameter_sweep": True,
    }


def training_paths(model, episodes, speeds):
    """One selected native path; train labels may change per executed action."""
    losses = []
    for seed, z, a in episodes:
        roots = torch.arange(2, len(a), device=z.device)
        windows = z[:-1].unfold(0, 3, 1).transpose(1, 2).contiguous()
        errors = []
        speed = speeds[seed].to(z.device)
        for step in range(4):
            source = (roots + step).clamp(max=len(a) - 1)
            commands = torch.stack([a[(roots + step - 2 + j).clamp(max=len(a) - 1)] for j in range(3)], 1)
            branch = torch.where(speed[source] == 2.5, 1, 2).long()
            predicted = model.selected_step(windows, commands, branch)
            target = z[(source + 1).clamp(max=len(z) - 1)]
            errors.append((predicted - target).square().mean(-1))
            windows = torch.cat((windows[:, 1:], predicted[:, None]), 1)
        errors = torch.stack(errors, -1)
        horizon = torch.arange(1, 5, device=z.device)
        legal = roots[:, None] + horizon <= len(a)
        prefix = errors.cumsum(-1) / horizon
        losses.append(((prefix * legal).sum(-1) / legal.sum(-1)).mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def measure_training_checkpoints():
    """Descriptive initial/final fit, without independent evaluation or updates."""
    episodes = load_episodes(DATA / "train.pt", "cuda")
    with np.load(DATA / "diag_train.npz") as diagnostic:
        speeds = {int(seed): torch.from_numpy(row.copy()).cuda()
                  for seed, row in zip(diagnostic["seed"], diagnostic["speed"], strict=True)}
    model = make_response_model("cuda")
    result = {}
    for name in ("initial", "final"):
        load_response_state(model, torch.load(OUT / f"{name}.pt", weights_only=True))
        values = [float(training_paths(model, [episode], speeds)) for episode in episodes]
        result[name] = {"raw_predictive_prefix_MSE": float(np.mean(values)),
                        "episode_MSE": values, "seed": [row[0] for row in episodes]}
    write("training_full_measure.json", result)
    return result


@torch.no_grad()
def anchors(model, episode):
    _, z, a = episode
    path = model.rollout(z[None, :3], a[None, :6])
    state, native = z[None, :3], []
    for t in range(4):
        value = model.host.predict(state, a[None, t:t + 3])[:, -1]
        native.append(value)
        state = torch.cat((state[:, 1:], value[:, None]), 1)
    error = (path[:, 0] - torch.stack(native, 1)).abs().amax((0, 2))
    if not bool((error == 0).all()):
        raise AssertionError("Literal nominal H1..4 differs from independently evaluated F0")
    return {"nominal_H1_H4_bitwise_F0": True, "max_abs_by_H": error.tolist(),
            "trainable_parameters": sum(p.numel() for p in model.response_parameters()),
            "frozen_host": not any(p.requires_grad for p in model.host.parameters()),
            "independent_suffix_storage": all(p.data_ptr() != q.data_ptr() for p, q in zip(
                model.final_block.parameters(), model.host.predictor.transformer.layers[-1].parameters(), strict=True)),
            "null_control": model.null_control}


def bootstrap(x):
    values = np.asarray(x, np.float64)
    if not len(values):
        return None
    draw = np.random.default_rng(20261121).integers(len(values), size=(4000, len(values)))
    return {"mean": float(values.mean()), "ci95": np.percentile(values[draw].mean(1), [2.5, 97.5]).tolist(),
            "positive_roots": int((values > 0).sum()), "roots": len(values)}


def contrast(x):
    return (x[:, 0, 0] - x[:, 0, 1]) - (x[:, 1, 0] - x[:, 1, 1])


@torch.no_grad()
def forecast_static(model):
    batch = load("static.pt")
    device = model.offsets.device
    # Inputs stop at z6; future actual latents are read only by scoring below.
    window = batch["z"][..., 4:7, :].flatten(0, 2).to(device)
    commands = batch["a"][..., 4:10, :].flatten(0, 2).to(device)
    family = model.rollout(window, commands).double().cpu().reshape(48, 2, 2, 4, 4, 192)
    # Static axis0 is slow and axis1 fast by the frozen data plan.
    pred = torch.stack((family[:, 0, :, 1], family[:, 1, :, 2]), 1)
    models = {"oracle": pred, "uniform": family.mean(3), "nominal": family[:, :, :, 0]}
    for name in ("frozen", "gru", "adajepa", "optta"):
        package = load(f"baselines/static_{name}.pt")
        if not torch.equal(package["seed"], batch["seed"]):
            raise ValueError("Static baseline root order differs")
        models[name] = package["predictions"].double()
    torch.save({"seed": batch["seed"], "predictions": models, "family": family}, OUT / "static_predictions.pt")
    return batch, models, family


def score_static(batch, models, family):
    from .prbd_phase2_score import static_masks

    with np.load(DATA / "diag_static.npz") as diagnostic:
        pos = torch.from_numpy(diagnostic["positions"].copy()).double()
    with np.load(DATA / "data_static.npz") as legal:
        matched, nonzero, exclusions = static_masks(batch, pos, legal["actions"])
    results, per_root = {}, {}
    complete = torch.nonzero((batch["lengths"] >= 10).flatten(1).all(-1)).flatten().tolist()
    for panel, rows in (("complete", complete), ("matched", matched), ("physical_nonzero", nonzero)):
        truth = batch["z"][rows, :, :, 7:11].double()
        delta = contrast(truth)
        curves = {key: contrast(value[rows]) for key, value in models.items()}
        error = {key: (value - delta).square().sum(-1) for key, value in curves.items()}
        results[panel] = {"n": len(rows), "horizon": {}}
        per_root[panel] = {"seed": batch["seed"][rows].tolist()}
        for h in range(4):
            estimates = {}
            for key, value in models.items():
                oracle_gain = error[key][:, h] - error["oracle"][:, h]
                estimates[key] = {
                    "endpoint_raw_latent_mse": float((value[rows, :, :, h] - truth[:, :, :, h]).square().mean()),
                    "endpoint_mse_move_stop": (value[rows, :, :, h] - truth[:, :, :, h]).square().mean((0, 1, 3)).tolist(),
                    "interaction_SSE": float(error[key][:, h].mean()),
                    "oracle_SSE_gain_over_arm": bootstrap(oracle_gain.tolist()),
                }
            denominator = delta[:, h].square().sum()
            uniform_sse = error["uniform"][:, h].sum()
            estimates["oracle"]["directed_recovery_percent_from_uniform"] = float(
                100 * ((curves["oracle"][:, h] - curves["uniform"][:, h]) * delta[:, h]).sum() / denominator) if denominator else None
            estimates["oracle"]["interaction_SSE_gain_percent_from_uniform"] = float(
                100 * (uniform_sse - error["oracle"][:, h].sum()) / uniform_sse) if uniform_sse else None
            results[panel]["horizon"][str(h + 1)] = estimates
            per_root[panel][str(h + 1)] = {"truth_energy": delta[:, h].square().sum(-1).tolist(),
                "interaction_SSE": {key: err[:, h].tolist() for key, err in error.items()}}
    separation = family[matched, :, :, 1] - family[matched, :, :, 2]
    results["heads"] = {"matched_roots": len(matched),
        "slow_fast_output_rms_by_H": separation.square().mean((0, 1, 2, 4)).sqrt().tolist()}
    # Output-space differences establish separation independently of parameter differences.
    results["offered"] = 48
    results["exclusions"] = exclusions
    write("static_results.json", results)
    write("static_per_root.json", per_root)
    return results


@torch.no_grad()
def forecast_switch(model):
    batch = load("switch.pt")
    device = model.offsets.device
    z, a = (batch[key].flatten(0, 1).to(device) for key in ("z", "a"))
    lengths = batch["lengths"].flatten().to(device)
    # Issue all branch families without diagnostic labels or cut points.
    family = torch.full((len(z), 32, 4, 4, 192), float("nan"), dtype=torch.float64)
    for root in range(2, 32):
        horizon = min(4, 32 - root)
        valid = lengths >= root + horizon
        if valid.any():
            family[valid.cpu(), root, :, :horizon] = model.rollout(
                z[valid, root - 2:root + 1], a[valid, root - 2:root + horizon]).double().cpu()
    family = family.reshape(48, 2, 32, 4, 4, 192)
    with np.load(DATA / "diag_switch.npz") as diagnostic:
        speeds = torch.from_numpy(diagnostic["speed"].copy())
    # The first-action speed selects the SAME k for every H; no future labels.
    branch = torch.where(speeds == 2.5, 1, 2)
    index = branch[..., None, None, None].expand(48, 2, 32, 1, 4, 192)
    oracle = family.gather(3, index).squeeze(3)
    models = {"oracle": oracle[..., 0, :], "uniform": family[..., 0, :].mean(3),
              "nominal": family[:, :, :, 0, 0]}
    for name in ("frozen", "gru", "adajepa", "optta"):
        package = load(f"baselines/switch_{name}.pt")
        if not torch.equal(package["seed"], batch["seed"]):
            raise ValueError("Switch baseline root order differs")
        models[name] = package["predictions"].double()
    torch.save({"seed": batch["seed"], "predictions": models, "family": family,
                "oracle_constant_current_speed_H1_H4": oracle}, OUT / "switch_predictions.pt")
    return batch, models


def score_switch(batch, models):
    from .prbd_phase2_score import recovered

    with np.load(DATA / "diag_switch.npz") as diagnostic:
        speed = torch.from_numpy(diagnostic["speed"].copy())
        cuts = diagnostic["switch_blocks"].copy()
    truth = batch["z"][..., 1:, :].double()
    clock = torch.arange(32)
    valid = (clock >= 2) & (clock < batch["lengths"][..., None])
    post = valid & (clock >= torch.from_numpy(cuts)[..., :1])
    errors = {name: (pred - truth).square().mean(-1) for name, pred in models.items()}
    counterfactual = load("switch_counterfactual.pt")
    if not torch.equal(counterfactual["seed"], batch["seed"]):
        raise ValueError("Switch diagnostic root IDs differ")
    pairs = counterfactual["z_cf"].double()
    with np.load(DATA / "diag_switch_counterfactual.npz") as diagnostic:
        informative = torch.from_numpy(diagnostic["differing_rgb"].copy())
    informative &= valid & (pairs[:, :, :, 0] != pairs[:, :, :, 1]).any(-1)
    results, roots = {}, {}
    for direction, target_speed in (("to_slow", 2.5), ("to_fast", 7.5)):
        mask = post & (speed == target_speed)
        totals = mask.sum((1, 2))
        if not bool((totals > 0).all()):
            raise ValueError("Paired switch root missing post-cut transitions")
        values = {key: torch.where(mask, err, 0.).sum((1, 2)) / totals for key, err in errors.items()}
        results[direction] = {key: {"raw_endpoint_mse": float(torch.where(mask, err, 0.).sum() / mask.sum()),
            "mean48_root_MSE": bootstrap(values[key].tolist()),
            "oracle_gain_over_arm": bootstrap((values[key] - values["oracle"]).tolist())}
            for key, err in errors.items()}
        roots[direction] = {"seed": batch["seed"].tolist(), "valid_blocks": totals.tolist(),
                            "raw_endpoint_mse": {key: value.tolist() for key, value in values.items()}}
    for name, pred in models.items():
        distance = (pred[..., None, :] - pairs).square().mean(-1)
        correct = torch.where(speed == 2.5, distance[..., 0] < distance[..., 1],
                              distance[..., 1] < distance[..., 0])
        events = []
        for i, seed in enumerate(batch["seed"].tolist()):
            for arm in range(2):
                c1, c2 = map(int, cuts[i, arm])
                for left, cut, end in ((2, c1, c2), (c1, c2, 32)):
                    direction = "to_fast" if ((arm == 0) == (cut == c1)) else "to_slow"
                    times = clock[informative[i, arm] & (clock >= cut) & (clock < end)]
                    past = clock[informative[i, arm] & (clock >= left) & (clock < cut)]
                    lag, confirmed = recovered(times.tolist(), correct[i, arm, times].tolist(), cut)
                    events.append({"seed": seed, "arm": arm, "direction": direction,
                        "cut": cut, "informative": len(times),
                        "pre_correct": bool(len(past) >= 2 and correct[i, arm, past[-2:]].all()),
                        "lag": lag, "confirmed_lag": confirmed,
                        "correct": int(correct[i, arm, times].sum())})
        for direction in ("to_slow", "to_fast"):
            offered = [event for event in events if event["direction"] == direction]
            eligible = [event for event in offered if event["informative"] >= 2]
            success = [event for event in eligible if event["lag"] is not None]
            results[direction][name]["original_two_consecutive_judge"] = {
                "offered": len(offered), "opportunities": len(eligible),
                "recovered": len(success), "right_censored": len(eligible) - len(success),
                "median_lag": float(np.median([event["lag"] for event in success])) if success else None,
                "pre_correct_opportunities": sum(event["pre_correct"] for event in eligible),
                "pre_correct_recovered": sum(event["pre_correct"] for event in success)}
        write(f"switch_{name}_events.json", events)
    results["overall"] = {key: float(err[valid].mean()) for key, err in errors.items()}
    write("switch_results.json", results)
    write("switch_per_root.json", roots)
    return results


def score_saved_predictions():
    """Rescore issued forecasts on CPU, including the original matched controls."""
    def saved(split):
        batch = load(f"{split}.pt")
        package = torch.load(OUT / f"{split}_predictions.pt", map_location="cpu", weights_only=True)
        if not torch.equal(package["seed"], batch["seed"]):
            raise ValueError(f"Oracle {split} evaluation root order differs")
        return batch, package

    batch, package = saved("static")
    static = score_static(batch, package["predictions"], package["family"])
    batch, package = saved("switch")
    switch = score_switch(batch, package["predictions"])
    fit = json.loads((OUT / "training_full_measure.json").read_text(encoding="utf-8"))
    write("result.json", {"static": static, "switch": switch,
        "full_training_fit": {key: row["raw_predictive_prefix_MSE"] for key, row in fit.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("score", "train"), help="rescore issued forecasts on CPU or run the fixed-budget capacity fit")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.stage == "score":
        score_saved_predictions()
        return
    torch.manual_seed(BUDGET["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "final.pt").exists():
        raise FileExistsError("Final capacity diagnostic already exists; no silent rerun")
    write("protocol.json", protocol())
    episodes = load_episodes(DATA / "train.pt", "cuda")
    with np.load(DATA / "diag_train.npz") as diagnostic:
        seeds = diagnostic["seed"].copy()
        speed = diagnostic["speed"].copy()
    if not np.array_equal(seeds, load("train.pt")["seed"].numpy()) or not {row[0] for row in episodes} <= set(seeds.tolist()):
        raise ValueError("Training diagnostic labels do not match actual training episodes")
    speeds = {int(seed): torch.from_numpy(row).cuda() for seed, row in zip(seeds, speed, strict=True)}
    train_ids = {row[0] for row in episodes}
    for split in ("static", "switch"):
        if train_ids.intersection(load(f"{split}.pt")["seed"].tolist()):
            raise ValueError("Training and independent reset IDs overlap")
    model = make_response_model("cuda")
    write("initial_checks.json", anchors(model, episodes[0]))
    torch.save(response_state(model), OUT / "initial.pt")
    optimizer = torch.optim.Adam(model.response_parameters(), lr=BUDGET["learning_rate"])
    generator = torch.Generator().manual_seed(BUDGET["seed"])
    started = perf_counter()
    with (OUT / "updates.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, BUDGET["updates"] + 1):
            if update in BUDGET["decay_updates"]:
                for group in optimizer.param_groups:
                    group["lr"] *= .1
            indices = torch.randperm(len(episodes), generator=generator)[:BUDGET["batch_episodes"]].tolist()
            selected = [episodes[i] for i in indices]
            optimizer.zero_grad(set_to_none=True)
            loss = training_paths(model, selected, speeds)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite oracle capacity loss at update{update}")
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.response_parameters()):
                raise FloatingPointError(f"Nonfinite capacity gradient at update{update}")
            optimizer.step()
            row = {"update": update, "raw_predictive_prefix_MSE": float(loss.detach()),
                   "seconds": perf_counter() - started, "episode_seeds": [x[0] for x in selected]}
            stream.write(json.dumps(row) + "\n")
            if update % 8 == 0:
                stream.flush()
            if update % 128 == 0:
                torch.save({**response_state(model), "update": update, "budget": BUDGET}, OUT / "latest.pt")
                print(json.dumps(row), flush=True)
    torch.save({**response_state(model), "update": update, "budget": BUDGET,
                "training_seeds": sorted(train_ids)}, OUT / "final.pt")
    write("final_checks.json", {**anchors(model, episodes[0]),
        "slow_fast_offset_L2": float((model.offsets[0] - model.offsets[1]).norm()),
        "unused_branch3_offset_matches_initial": bool(torch.equal(model.offsets[2].cpu(),
            torch.load(OUT / "initial.pt", weights_only=True)["learned_state"]["offsets"][2]))})
    write("training_result.json", {"updates": update, "training_seconds": perf_counter() - started,
        "final_batch_raw_predictive_prefix_MSE": float(loss.detach()), "selection": "final2048 only"})
    model.requires_grad_(False)
    fit = measure_training_checkpoints()
    static_batch, static_models, static_family = forecast_static(model)
    static_result = score_static(static_batch, static_models, static_family)
    switch_batch, switch_models = forecast_switch(model)
    switch_result = score_switch(switch_batch, switch_models)
    write("result.json", {"elapsed_seconds_including_evaluation": perf_counter() - started,
        "static": static_result, "switch": switch_result,
        "full_training_fit": {key: value["raw_predictive_prefix_MSE"] for key, value in fit.items()}})
    print(json.dumps({"done": True, "seconds": perf_counter() - started,
        "to_slow_MSE": switch_result["to_slow"]["oracle"]["raw_endpoint_mse"],
        "to_fast_MSE": switch_result["to_fast"]["oracle"]["raw_endpoint_mse"]}), flush=True)


if __name__ == "__main__":
    main()
