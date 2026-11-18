"""Compare fixed latent and decoded-XY writes on the same FourRoom forecasts."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .native_full_depth_conditioning import advance_log_support, write_observation
from .native_topology_condition import (
    CALIBRATE, HOLDOUT, ROOT, RUN as REFERENCE, SWITCH, forecast, load_readout, load_trained,
    predict_sources, read_panel,
)
from .native_topology_prefix_eval import RUN as PREFIX, factual_masks, predict_local
from .native_topology_writein_attribution import first_block_groups, regret_parts
from .native_topology_writein_replay import POLICIES, RUN as PAIRED, STAGES, decisions, score_banks
from .physical_topology_gate import ReversibleFourRoom, VisibleMapCost, make_scene


def read_npz(path):
    with np.load(path, allow_pickle=False) as stream:
        return dict(stream)


def calibrate(model, decode, panel):
    """Gaussian covariances about issued means, fitted from factual residuals only."""
    issued, _, index = predict_sources(*model, panel)
    rows, initial, time = index
    arrived = panel["life_z"][rows, initial, time + 1]
    mode = panel["life_mode"][index].long()
    matched = issued[torch.arange(len(issued), device=issued.device), mode]
    residual = decode(arrived).double() - decode(matched).double()
    covariance = residual.T @ residual / len(residual)
    factor = torch.linalg.cholesky(covariance)
    return factor, {"layouts": panel["seed"].tolist(), "arrivals": len(residual),
                    "covariance_px_squared": covariance.tolist(),
                    "mean_residual_px": residual.mean(0).tolist()}


def signed_paired_gaps(issued, arrivals, scale, decode=None, transform=None):
    """One issued pair scored against each legally arrived endpoint separately."""
    prior = scale.new_zeros((*arrivals.shape[:-1], 2))
    _, ll = write_observation(prior, issued[:, None], arrivals, scale, decode=decode)
    return paired_directions(ll if transform is None else transform(ll))


def paired_directions(ll):
    return torch.stack((ll[..., 0, 0] - ll[..., 0, 1], ll[..., 1, 1] - ll[..., 1, 0]), dim=-1)


def student_t(gaussian):
    """Use the same 2D scatter, with 3 degrees of freedom fixed before evaluation."""
    return -2.5 * torch.log1p((-2.0 / 3.0) * gaussian)


def replay_scores(archive, index, scores):
    """Replay identical factual arrivals using fixed per-mode likelihood scores."""
    rows, initial, time = index
    shape = archive["life_valid"].shape
    actual = scores.new_full((*shape[:2], 2), -np.log(2))
    omitted = actual.clone()
    tracks = {key: actual.new_full((*shape, 2), torch.nan)
              for key in ("prior", "actual", "omit_reversed", "log_likelihood")}
    for t in range(2, shape[-1]):
        select = time == t
        a, b = rows[select], initial[select]
        prior = advance_log_support(actual[a, b], SWITCH)
        without = advance_log_support(omitted[a, b], SWITCH)
        ll = scores[select]
        actual[a, b] = (prior + ll).log_softmax(-1)
        include = archive["life_event"][a, b, t] != 3
        omitted[a, b] = (without + ll * include[:, None]).log_softmax(-1)
        for key, value in zip(tracks, (prior, actual[a, b], omitted[a, b], ll), strict=True):
            tracks[key][a, b, t] = value
    return {k: v.cpu().numpy() for k, v in tracks.items()}


def replay(archive, index, issued, arrived, scale, decode=None, transform=None):
    _, scores = write_observation(scale.new_zeros((len(issued), 2)), issued, arrived, scale, decode=decode)
    return replay_scores(archive, index, scores if transform is None else transform(scores))


def sizes(values):
    return ({"mean_abs": float(np.abs(values).mean()),
             "quartiles_abs": np.quantile(np.abs(values), [0.25, 0.5, 0.75]).tolist(),
             "p99_abs": float(np.quantile(np.abs(values), .99)), "max_abs": float(np.abs(values).max())}
            if len(values) else None)


def summarize_pairs(gaps, raw_effect, end_effect):
    result = {}
    for name, mask in (("raw_neutral", ~raw_effect), ("converged_at_endpoint", raw_effect & ~end_effect),
                       ("endpoint_informative", end_effect)):
        values = gaps[mask]
        result[name] = {"sources": int(mask.sum()), "magnitude_nats": sizes(values.ravel()),
                        "directions_correct_mode_0_1": (values > 0).sum(0).tolist(),
                        "directions_wrong_mode_0_1": (values < 0).sum(0).tolist(),
                        "ties_both": int((values == 0).all(-1).sum()),
                        "correct_both": int((values > 0).all(-1).sum()),
                        "wrong_both": int((values < 0).all(-1).sum())}
    return result


def lifecycle_report(tracks, index, events, modes, raw_neutral):
    sign = 1 - 2 * modes
    prior, post, omitted, ll = (tracks[key][index] for key in ("prior", "actual", "omit_reversed", "log_likelihood"))
    before = (prior[:, 0] - prior[:, 1]) * sign
    after = (post[:, 0] - post[:, 1]) * sign
    gap = (ll[:, 0] - ll[:, 1]) * sign
    result = {}
    for name, take in ([(st, events == i + 1) for i, st in enumerate(STAGES[:3])]
                       + [("raw_neutral", raw_neutral), ("all", np.ones_like(raw_neutral, dtype=bool))]):
        p = np.exp(post[take, modes[take]])
        prev = np.exp(prior[take, modes[take]])
        result[name] = {
            "arrivals": len(p), "pre_true_majority": int((before[take] > 0).sum()),
            "post_true_majority": int((after[take] > 0).sum()), "post_wrong_majority": int((after[take] < 0).sum()),
            "gain_true_majority": int(((before[take] <= 0) & (after[take] > 0)).sum()),
            "lose_true_majority": int(((before[take] > 0) & (after[take] < 0)).sum()),
            "mean_true_probability_pre": float(prev.mean()), "mean_true_probability_post": float(p.mean()),
            "mean_true_probability_omit_reversal": float(np.exp(omitted[take, modes[take]]).mean()),
            "mean_true_nll": float(-post[take, modes[take]].mean()),
            "actual_correct_direction": int((gap[take] > 0).sum()),
            "actual_wrong_direction": int((gap[take] < 0).sum()),
            "magnitude_nats": sizes(gap[take]),
        }
    return result


def downstream(archive, banks, tracks, learned):
    selected = decisions(archive, banks, tracks, learned)
    pool = banks["choices"][:, -1]
    physical = banks["physical_cost"][pool, selected["mode"]].astype(np.float64)
    group, group_min = (np.stack(values) for values in zip(*(
        first_block_groups(action, truth) for action, truth in zip(banks["action"][pool], physical, strict=True)
    )))
    regret = regret_parts(physical, group_min, selected["best"])
    np.testing.assert_allclose(regret[..., 2], selected["regret"], atol=1e-5, rtol=0)
    first = group[np.arange(len(pool))[:, None], selected["best"]]
    by_stage = {}
    for i, name in enumerate(STAGES):
        take = selected["stage"] == i
        stage = {"queries": int(take.sum()), "physical_decision_space": int(selected["physical_decision_space"][take].sum()),
                 "mean_regret_first_suffix_plan": dict(zip(POLICIES, regret[take].mean(0).tolist(), strict=True))}
        for j, policy in enumerate(POLICIES[1:], start=1):
            delta = regret[take, 0] - regret[take, j]
            stage[f"actual_minus_{policy}"] = {"mean_regret_first_suffix_plan": delta.mean(0).tolist(),
                                               "first_improved": int((delta[:, 0] < 0).sum()),
                                               "first_worsened": int((delta[:, 0] > 0).sum()),
                                               "first_changed": int((first[take, 0] != first[take, j]).sum())}
        by_stage[name] = stage
    return by_stage, dict(selected, first_group=first, regret_parts=regret)


def cost_from_saved_forecasts(path, banks):
    with np.load(path, allow_pickle=False) as stream:
        xy = stream["life_xy"][:, :, :, -1].transpose(0, 2, 1, 3)
    if xy.shape[:-1] != banks["physical_cost"].shape:
        raise ValueError("Archived lifecycle forecasts do not match the fixed H5 bank dimensions")
    return np.stack([VisibleMapCost(scene, scene.southeast)(forecast_xy)
                     for scene, forecast_xy in zip(map(make_scene, map(int, banks["seed"])), xy, strict=True)])


def archived_spike(model, decode, latent_scale, xy_factor, device):
    logged = json.loads((REFERENCE / "cem_archive_attribution.json").read_text(encoding="utf-8"))
    case = min(logged["evidence_cases"], key=lambda row: row["signed_log_likelihood_gap"])
    assert case["physically_neutral"] and case["actual_raw_xy"] == case["other_raw_xy"]
    with (REFERENCE / "mpc_remaining.jsonl").open(encoding="utf-8") as stream:
        episode = next(row for row in map(json.loads, stream) if row.get("seed") == case["seed"]
                       and row.get("arm") == "persistent" and row["initial_mode"] == 1 - case["mode"])
    # Archived block numbering is one-based. Recover the corresponding exact command by contents.
    chosen, = [i for i, tick in enumerate(episode["ticks"]) if tick["raw_action"] == case["raw_action"]
               and tick["actual_raw_xy"] == case["actual_raw_xy"]]
    assert chosen + 1 == case["block"]
    scene = make_scene(case["seed"])
    position = [scene.northwest] + [step["raw_xy"][-1] for step in episode["scripted_prefix"]]
    controls = [step["raw_actions"] for step in episode["scripted_prefix"]]
    position += [tick["actual_raw_xy"][-1] for tick in episode["ticks"][:chosen + 1]]
    controls += [tick["raw_action"] for tick in episode["ticks"][:chosen + 1]]
    np.testing.assert_array_equal(position[-2], case["actual_start"])
    assert len(position) == len(controls) + 1
    env = ReversibleFourRoom(scene)
    env.begin(scene.northwest, scene.southeast, 0)
    try:
        frames = [env._render_frame(agent_pos=torch.as_tensor(xy, dtype=torch.float32)) for xy in position[-4:]]
    finally:
        env.close()
    frames = torch.stack(frames).to(device=device, dtype=torch.float32) / 255
    mean = frames.new_tensor((.485, .456, .406))[:, None, None]
    std = frames.new_tensor((.229, .224, .225))[:, None, None]
    # The archived deployed writer encoded each arrival as a single frame.
    z = torch.stack([model[0].encode({"pixels": ((frame - mean) / std)[None, None]})["emb"][0, 0]
                     for frame in frames])
    u = torch.as_tensor(controls[-3:], device=device, dtype=z.dtype)
    issued = forecast(*model, z[None, :3], u[None, :2], u[None, None, 2:], compact=True)[:, 0, :, 0]
    p = case["pre_block_p_true"]
    prior = z.new_tensor([[p, 1 - p]]).log()
    result = {"archive": case, "forecast_xy_mode_0_1": decode(issued)[0].tolist(), "arrived_xy": decode(z[-1]).tolist()}
    for name, scale, read in (("latent", latent_scale, None), ("xy", xy_factor, decode)):
        post, ll = write_observation(prior, issued, z[None, -1], scale, decode=read)
        result[name] = {"log_likelihood": ll[0].tolist(), "signed_gap": float(ll[0, 0] - ll[0, 1]),
                        "p_true_post": float(post[0, 0].exp())}
    result["latent_gap_minus_archive"] = result["latent"]["signed_gap"] - case["signed_log_likelihood_gap"]
    return result


def student_spike(case):
    p = case["archive"]["pre_block_p_true"]
    g = torch.tensor(case["xy"]["log_likelihood"], dtype=torch.float64)
    rare = student_t(g)
    post = (torch.tensor([p, 1-p], dtype=torch.float64).log() + rare).softmax(-1)
    return {"log_likelihood": rare.tolist(), "signed_gap": float(rare[0] - rare[1]), "p_true_post": float(post[0])}


def frozen_student_comparison(args, device):
    old = args.frozen_xy
    info = json.loads((old / "summary.json").read_text(encoding="utf-8"))
    if Path(info["checkpoint"]).resolve() != args.checkpoint.resolve():
        raise ValueError("Frozen scorer artifacts and forecast checkpoint differ")
    saved = read_npz(old / "xy.npz")
    lifecycle = read_npz(REFERENCE / "data_holdout.npz")
    indices = tuple(saved["source_index"])
    neutral = ~saved["raw_effect"]
    modes = saved["mode"]
    log_xy = torch.from_numpy(saved["log_likelihood"][indices])
    scores = student_t(log_xy)
    tracks = replay_scores({"life_valid": torch.from_numpy(lifecycle["life_valid"]),
                            "life_event": torch.from_numpy(lifecycle["life_event"])},
                           tuple(torch.from_numpy(v) for v in indices), scores)
    banks = read_npz(PAIRED / "writein_fixed_h5_banks.npz")
    by_stage, selected = downstream(lifecycle, banks, tracks, np.load(old / "fixed_learned_cost.npy"))

    # The old short cache saved likelihood differences only. Student-t requires both
    # radii, so issue exactly those same frozen short-window forecasts once.
    factor = torch.load(old / "xy_cholesky.pt", map_location=device, weights_only=True)
    latent_scale = torch.load(REFERENCE / "likelihood_scale.pt", map_location=device, weights_only=True)
    model = load_trained(args.checkpoint, "conditioned", device)
    decode = load_readout(device, REFERENCE)
    small = read_npz(PREFIX / "data_eval.npz")
    terms = {}
    predict_local(*model, decode, latent_scale, small, evidence_terms=terms, xy_cholesky=factor)
    local = factual_masks(small)
    mask = local["eligible"]
    original = read_npz(old / "local_sources.npz")
    np.testing.assert_array_equal(mask, original["eligible"])
    ll = torch.from_numpy(terms["xy_log_likelihood"][mask])
    g = paired_directions(ll).numpy()
    np.testing.assert_allclose(g, original["gap_xy"][mask], atol=1e-5, rtol=2e-5)
    t = paired_directions(student_t(ll)).numpy()
    np.testing.assert_array_equal(np.sign(g), np.sign(t))
    spike = info["archived_neutral_spike"]
    result = {"scope": "Zero-update, identical forecasts and Cholesky factor; Student-t df=3",
              "frozen_xy_archive": str(old.resolve()), "calibration": info["calibration"],
              "archived_neutral_spike": {"gaussian": spike["xy"],
                                         "student_t": student_spike(spike)},
              "factual_lifecycle": {"gaussian": info["scores"]["xy"]["factual_lifecycle"],
                                     "student_t": lifecycle_report(tracks, indices, lifecycle["life_event"][indices], modes, neutral)},
              "fixed_short_bank_sources": {k: summarize_pairs(x, local["raw_change"][mask], local["end_change"][mask])
                                           for k, x in (("gaussian", g), ("student_t", t))},
              "fixed_h5_decisions": {"gaussian": info["scores"]["xy"]["fixed_h5_decisions"], "student_t": by_stage}}
    np.savez_compressed(args.output / "student_t.npz", source_index=saved["source_index"],
                        gaussian_log_likelihood=log_xy.numpy(), mode=modes, **tracks,
                        **{f"decision_{k}": v for k, v in selected.items()})
    np.savez_compressed(args.output / "local_xy_radii.npz", eligible=mask, gaussian_log_likelihood=ll.numpy(),
                        gaussian_pair_gaps=g, student_t_pair_gaps=t)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output / "summary.json", flush=True)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Directory of conditioned.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    bank = parser.add_mutually_exclusive_group()
    bank.add_argument("--cached-cost", type=Path, help="Matching checkpoint's archived write-in learned_cost")
    bank.add_argument("--cached-forecast", type=Path, help="Matching checkpoint's life_xy forecasts on the fixed H5 bank")
    bank.add_argument("--skip-bank", action="store_true")
    parser.add_argument("--skip-local", action="store_true", help="Skip the additional fixed short bank evidence sources")
    calibration_input = parser.add_mutually_exclusive_group()
    calibration_input.add_argument("--frozen-xy", type=Path, help="Score saved XY predictions under fixed Student-t df=3")
    calibration_input.add_argument("--calibration-only", action="store_true",
                                   help="Save the checkpoint's calibration factor without reading evaluation data")
    calibration_input.add_argument("--xy-cholesky", type=Path,
                                   help="Score new forecasts with this existing factor without recalibrating")
    parser.add_argument("--with-student-t", action="store_true", help="Also score the same new forecasts under Student-t df=3")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.frozen_xy:
        frozen_student_comparison(args, device)
        return
    decode = load_readout(device, REFERENCE)
    model = load_trained(args.checkpoint, "conditioned", device)
    if args.xy_cholesky is None:
        train = read_panel(REFERENCE / "data_train.npz", CALIBRATE, device)
        factor, calibration = calibrate(model, decode, train)
        print(f"Calibrated {calibration['arrivals']} train-only factual XY residuals", flush=True)
    else:
        factor = torch.load(args.xy_cholesky, map_location=device, weights_only=True)
        if (not isinstance(factor, torch.Tensor) or factor.shape != (2, 2)
                or not factor.is_floating_point() or not factor.isfinite().all()
                or not torch.equal(factor, factor.tril()) or not (factor.diagonal() > 0).all()):
            raise ValueError("Expected a finite 2x2 lower Cholesky factor with a positive diagonal")
        calibration = {"fixed_xy_cholesky": str(args.xy_cholesky.resolve()),
                       "covariance_px_squared": (factor @ factor.T).tolist()}
        print(f"Loaded fixed XY calibration factor from {args.xy_cholesky}", flush=True)
    factor_path = args.output / "xy_cholesky.pt"
    if args.xy_cholesky is None or args.xy_cholesky.resolve() != factor_path.resolve():
        torch.save(factor.cpu(), factor_path)
    if args.calibration_only:
        report = {"scope": "Calibration facts only", "checkpoint": str(args.checkpoint.resolve()),
                  "device": str(device), "calibration": calibration}
        path = args.output / "summary.json"
        path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(path, flush=True)
        return

    latent_scale = torch.load(REFERENCE / "likelihood_scale.pt", map_location=device, weights_only=True)
    archive = read_npz(REFERENCE / "data_holdout.npz")
    np.testing.assert_array_equal(archive["seed"], list(HOLDOUT))
    heldout = {k: torch.as_tensor(v, device=device) for k, v in archive.items() if k != "future_z"}
    issued, exposure, index = predict_sources(*model, heldout)
    rows, initial, time = index
    arrived = heldout["life_z"][rows, initial, time + 1]
    arrivals = heldout["life_cf_z"][index]
    modes = heldout["life_mode"][index].long()
    torch.testing.assert_close(arrived, arrivals[torch.arange(len(modes), device=device), modes], atol=0, rtol=0)
    idx = tuple(k.cpu().numpy() for k in index)
    raw = archive["life_cf_xy"][idx]
    raw_effect = (raw[:, 0] != raw[:, 1]).any(axis=(-2, -1))
    end_effect = (raw[:, 0, -1] != raw[:, 1, -1]).any(axis=-1)
    np.testing.assert_array_equal(raw_effect, archive["life_informative"][idx])
    modes = modes.cpu().numpy()
    report = {"scope": "Fixed old data and actions; comparing writes within each frozen checkpoint, no closed loop.",
              "checkpoint": str(args.checkpoint.resolve()), "calibration": calibration, "scores": {}}
    report["archived_neutral_spike"] = archived_spike(model, decode, latent_scale, factor, device)
    old_case = report["archived_neutral_spike"]
    if args.with_student_t:
        old_case["student_t"] = student_spike(old_case)
    print(f"Archived neutral case: {old_case['latent']['signed_gap']:+g} latent, "
          f"{old_case['xy']['signed_gap']:+g} XY nat", flush=True)
    if args.checkpoint.resolve() == REFERENCE.resolve():
        # The archived RGB encoder and predictor ran on CUDA; CPU kernels differ slightly.
        np.testing.assert_allclose(report["archived_neutral_spike"]["latent_gap_minus_archive"], 0, atol=1.)
    banks = None if args.skip_bank else read_npz(PAIRED / "writein_fixed_h5_banks.npz")
    if banks is not None:
        if args.cached_cost:
            with np.load(args.cached_cost, allow_pickle=False) as stream:
                learned = stream["learned_cost"]
        elif args.cached_forecast:
            learned = cost_from_saved_forecasts(args.cached_forecast, banks)
        else:
            learned = score_banks(model, banks, decode, device)
        np.save(args.output / "fixed_learned_cost.npy", learned)
        print(f"Fixed H5 bank: {len(banks['choices'])} queries", flush=True)
    short_gaps = {}
    if not args.skip_local:
        small = read_npz(PREFIX / "data_eval.npz")
        terms = {}
        _, _, local_exposure, gap_latent = predict_local(*model, decode, latent_scale, small,
                                                        evidence_terms=terms, xy_cholesky=factor)
        gap_xy = terms.pop("xy_signed_gaps")
        local = factual_masks(small)
        pick = local["eligible"]
        short_gaps = {"latent": gap_latent[pick], "xy": gap_xy[pick]}
        if args.with_student_t:
            short_gaps["student_t"] = paired_directions(student_t(torch.from_numpy(terms["xy_log_likelihood"][pick]))).numpy()
            np.testing.assert_array_equal(np.sign(short_gaps["student_t"]), np.sign(short_gaps["xy"]))
        print(f"Fixed short bank: {int(pick.sum())} legal paired sources", flush=True)
        np.savez_compressed(args.output / "local_sources.npz", seed=small["seed"], bank_index=small["bank_index"],
                            eligible=pick, raw_effect=local["raw_change"], endpoint_effect=local["end_change"],
                            gap_latent=gap_latent, gap_xy=gap_xy, exposure=local_exposure, **terms)
    scorers = [("latent", latent_scale, None, None), ("xy", factor, decode, None)]
    if args.with_student_t:
        scorers.append(("student_t", factor, decode, student_t))
    for name, scale, read, transform in scorers:
        gaps = signed_paired_gaps(issued, arrivals, scale, read, transform).cpu().numpy()
        tracks = replay(heldout, index, issued, arrived, scale, read, transform)
        ll = tracks["log_likelihood"][idx]
        actual_signed = (ll[:, 0] - ll[:, 1]) * (1 - 2 * modes)
        np.testing.assert_allclose(gaps[np.arange(len(modes)), modes], actual_signed, atol=1e-2, rtol=5e-5)
        if (~end_effect).any():
            np.testing.assert_allclose(gaps[~end_effect].sum(-1), 0, atol=1e-2, rtol=0)
        report["scores"][name] = {"paired_sources": summarize_pairs(gaps, raw_effect, end_effect),
                                  "factual_lifecycle": lifecycle_report(tracks, idx, archive["life_event"][idx], modes, ~raw_effect)}
        if not args.skip_local:
            source = short_gaps[name]
            report["scores"][name]["fixed_short_bank_sources"] = summarize_pairs(
                source, local["raw_change"][pick], local["end_change"][pick])
            # Neutral short sources do not come with a support history. Count
            # ability to overturn a common .9 prior without counting it as a replayed flip.
            short_neutral = source[~local["raw_change"][pick]]
            report["scores"][name]["fixed_short_bank_sources"]["raw_neutral"]["could_flip_p09"] = int(
                (short_neutral < -np.log(9)).any(axis=-1).sum())
        arrays = dict(source_index=np.stack(idx), exposure=exposure.cpu().numpy(), gap_pair=gaps,
                      raw_effect=raw_effect, endpoint_effect=end_effect, mode=modes, **tracks)
        if banks is not None:
            summary, selected = downstream(archive, banks, tracks, learned)
            report["scores"][name]["fixed_h5_decisions"] = summary
            arrays |= {f"decision_{k}": v for k, v in selected.items()}
        np.savez_compressed(args.output / f"{name}.npz", **arrays)
    if args.with_student_t and not args.skip_local:
        measured = report["scores"]["student_t"]
        life = measured["factual_lifecycle"]
        short = measured["fixed_short_bank_sources"]
        if life["reversed_probe"]["arrivals"] != 64 or short["endpoint_informative"]["sources"] != 640:
            raise ValueError("The fixed evidence gate requires all 64 reversals and 640 informative sources")
        report["fixed_evidence_gate"] = {
            "abs_archived_lr_lt_0p5": abs(old_case["student_t"]["signed_gap"]) < 0.5,
            "all_64_reversals_true_majority": life["reversed_probe"]["post_true_majority"] == 64,
            "no_actual_neutral_flip": life["raw_neutral"]["lose_true_majority"] == 0,
            "short_both_correct_ge_487_of_640": short["endpoint_informative"]["correct_both"] >= 487,
            "short_neutral_could_flip_at_p09": short["raw_neutral"]["could_flip_p09"],
        }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output / "summary.json", flush=True)


if __name__ == "__main__":
    main()
