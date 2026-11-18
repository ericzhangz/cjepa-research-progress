"""Score matched H5 training arms offline on one frozen FourRoom candidate bank."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .native_full_depth_conditioning import write_observation
from .native_topology_condition import ROOT, RUN as REFERENCE, forecast, gate_counts, load_readout, load_trained
from .physical_topology_gate import SPEED, PhysicalRollout, VisibleMapCost, make_scene, room


RUN = ROOT / "runs/native_topology_prefix_20261115"
SEEDS = np.arange(2026112000, 2026112016)
ARMS = ("reference", "old", "query", "prefix")
HORIZON = 5
ROOT_NAMES = ("top_contact", "bottom_contact", "northwest_navigation", "southeast_navigation")
ROLES = {0: "original", 1: "retained_original", 2: "common_corner", 3: "block_separated",
         4: "common_corner_fallback", 5: "block_separated_fallback"}


def load_panels(run):
    panels = []
    for name in ("data_eval.npz", "data_eval_bank.npz"):
        with np.load(run / name, allow_pickle=False) as source:
            panels.append(dict(source))
    small, bank = panels
    layouts, roots, candidates = len(SEEDS), len(ROOT_NAMES), 388
    for key in ("seed", "root_xy", "root_z", "root_action"):
        np.testing.assert_array_equal(small[key], bank[key])
    np.testing.assert_array_equal(bank["seed"], SEEDS)
    if bank["action"].shape != (layouts, roots, candidates, HORIZON, 5, 2):
        raise ValueError("Expected four original and 384 round-major frozen CEM plans per root")
    index = small["bank_index"]
    if index.shape != (layouts, roots, 8) or not ((0 <= index) & (index < candidates)).all():
        raise ValueError("Expected eight indexed C/L/F diagnostic plans per root")
    np.testing.assert_array_equal(index[..., :4], np.broadcast_to(np.arange(4), index[..., :4].shape))
    indices = (np.arange(layouts)[:, None, None], np.arange(roots)[None, :, None], index)
    for key in ("action", "future_xy"):
        np.testing.assert_array_equal(small[key], bank[key][indices])
    if not np.isin(small["selection_role"], tuple(ROLES)).all():
        raise ValueError("Unexpected diagnostic candidate selection role")
    return small, bank


def execute_block(physical, before, commands):
    """Open-loop replay from per-mode starts. Both paths execute the same input."""
    dims = before.shape
    state = before.reshape(-1, 2, 2).transpose(1, 0, 2)
    raw = []
    for command in commands.reshape(-1, 5, 2).transpose(1, 0, 2):
        state = physical.advance(state, command)
        raw.append(state.transpose(1, 0, 2).reshape(dims))
    return np.stack(raw, axis=-2)


def wall_depth(physical, xy, true_mode):
    """Use the collision-centre strips, with apertures open or locked as specified."""
    x, y = np.moveaxis(xy, -1, 0)
    if true_mode:
        centers = np.array([physical.top, physical.bottom], dtype=np.float32)[:, None]
        vertical = np.abs(y - centers)
    else:
        vertical = np.minimum(np.abs(y - physical.top), np.abs(y - physical.bottom))
    horizontal = np.minimum(np.abs(x - physical.left), np.abs(x - physical.right))
    col = np.minimum(np.minimum(x - physical.near, physical.far - x), vertical - physical.aperture)
    row = np.minimum(np.minimum(y - physical.near, physical.far - y), horizontal - physical.aperture)
    return np.maximum(0, np.maximum(col, row))


def scores(panel, xy):
    cost = np.empty(xy.shape[:-1], dtype=np.float32)
    for i, seed in enumerate(panel["seed"]):
        scene = make_scene(int(seed))
        for root in range(len(ROOT_NAMES)):
            goal = scene.southeast if root in (0, 2) else scene.northwest
            cost[i, root] = VisibleMapCost(scene, goal)(xy[i, root])
    return cost


def geometry(panel, xy, root_c, *, prior_xy=None, verify=False):
    """Keep occupancy, wrong room and replay from the prior decoded state distinct."""
    truth = panel["future_xy"][..., -1, :]
    depth = {"visible": np.empty(xy.shape[:-1], dtype=np.float32),
             "true_mode": np.empty(xy.shape[:-1], dtype=np.float32)}
    replay = np.empty_like(xy)
    prior_xy = xy if prior_xy is None else prior_xy
    for i, seed in enumerate(panel["seed"]):
        physical = PhysicalRollout(make_scene(int(seed)))
        for name in depth:
            depth[name][i] = wall_depth(physical, xy[i], name == "true_mode")
        for h in range(HORIZON):
            root_y = np.broadcast_to(panel["root_xy"][i, :, None, None, -1], truth[i, ..., h, :].shape)
            root_decoded = np.broadcast_to(root_c[i, :, None, None], truth[i, ..., h, :].shape)
            commands = panel["action"][i, ..., h, :, :]
            before = root_decoded if h == 0 else prior_xy[i, ..., h - 1, :]
            replay[i, ..., h, :] = execute_block(physical, before, commands)[..., -1, :]
            if verify:
                factual = root_y if h == 0 else truth[i, ..., h - 1, :]
                np.testing.assert_array_equal(execute_block(physical, factual, commands),
                                              panel["future_xy"][i, ..., h, :, :])
    endpoint_room = room(xy)
    return {"visible_wall_depth": depth["visible"], "true_mode_wall_depth": depth["true_mode"],
            "wrong_room": endpoint_room != room(truth), "prior_decoded_replay_xy": replay,
            "wrong_room_prior_decoded_replay": endpoint_room != room(replay)}


def factual_masks(panel):
    """Mark common inner-wall blocks and physically traversed navigation routes."""
    raw = panel["future_xy"]
    shape = (*raw.shape[:3], 2, 1, 2)
    starts = np.broadcast_to(panel["root_xy"][:, :, None, None, -1:, :], shape)
    before = np.concatenate((starts, raw[..., :-1, -1, :]), axis=-2)
    raw_before = np.concatenate((before[..., None, :], raw[..., :-1, :]), axis=-2)
    raw_change = (panel["local_xy"][..., 0, :, :] != panel["local_xy"][..., 1, :, :]).any(axis=(-2, -1))
    end_change = (panel["local_xy"][..., 0, -1, :] != panel["local_xy"][..., 1, -1, :]).any(axis=-1)
    valid = panel["local_valid"].copy()
    indices = panel["bank_index"]
    unique = np.ones(indices.shape, dtype=bool)
    for slot in range(1, indices.shape[-1]):
        unique[..., slot] = ~(indices[..., slot, None] == indices[..., :slot]).any(axis=-1)
    eligible = valid & unique[..., None, None]
    eligible[..., 1, 0] = False  # The shared H1 source is one genuine pair.
    np.testing.assert_array_equal(valid[..., 0, 0], valid[..., 1, 0])
    np.testing.assert_array_equal(panel["local_xy"][..., 0, 0, :, :, :], panel["local_xy"][..., 1, 0, :, :, :])
    if "first_local_divergence_raw" in panel:
        np.testing.assert_array_equal(raw_change[valid], (panel["first_local_divergence_raw"] >= 0)[valid])
    if (end_change[valid] & ~raw_change[valid]).any():
        raise AssertionError("A block-end mode effect requires a raw-step effect")
    command = panel["action"][:, :, :, None]
    layout = PhysicalRollout(make_scene(int(panel["seed"][0])))
    unrestricted = np.clip(raw_before + SPEED * command, layout.min_xy, layout.max_xy)
    wall_clips = (np.abs(raw - unrestricted) > 1e-5).any(axis=-1).any(axis=-1)
    common = (before[..., 0, :, :] == before[..., 1, :, :]).all(axis=-1)
    common_blocked = (common & valid.all(axis=-2) & ~raw_change.any(axis=-2) & wall_clips.all(axis=-2))
    transition = (room(raw_before) != room(raw)).any(axis=(-2, -1))
    root_cost = scores(panel, np.broadcast_to(starts, before.shape))
    truth_cost = scores(panel, raw[..., -1, :])
    legal_route = transition & (truth_cost[..., -1] < root_cost[..., -1])
    legal_route[:, :2] = False
    intent = panel["action"].sum(axis=-2)
    norm = np.linalg.norm(intent, axis=-1, keepdims=True)
    direction = np.divide(intent, norm, out=np.zeros_like(intent), where=norm > 0)
    common_blocked &= (norm[..., 0] > 0) & unique[..., None]
    legal_route &= unique[..., None]
    return {"before": before, "raw_change": raw_change, "end_change": end_change, "eligible": eligible, "valid": valid,
            "common_blocked": common_blocked, "legal_route": legal_route, "direction": direction, "unique_slot": unique,
            "root_cost": root_cost}


@torch.inference_mode()
def predict_bank(host, conditioner, gate, decode, panel, *, force=None):
    shape = panel["future_xy"].shape[:-2]
    output = np.empty((*shape, 2), dtype=np.float32)
    exposure = np.empty(shape, dtype=np.float32)
    device = host.action_mean.device
    as_tensor = lambda x: torch.as_tensor(x, device=device)
    root_c = decode(as_tensor(panel["root_z"][..., -1, :])).cpu().numpy()
    for i in range(len(panel["seed"])):
        history = as_tensor(panel["root_z"][i])
        past = as_tensor(panel["root_action"][i])
        plans = as_tensor(panel["action"][i])
        for root in range(len(ROOT_NAMES)):
            for first, last in ((0, 4), *((k, k + 96) for k in range(4, plans.size(1), 96))):
                end = slice(first, last)
                trace = {}
                issued = forecast(host, conditioner, gate, history[root:root + 1], past[root:root + 1],
                                  plans[root:root + 1, end], compact=True, trace=trace, force=force)
                decoded = decode(issued[0])
                decoded[..., -1, :] = decode(issued[0, ..., -1, :])
                output[i, root, end] = decoded.cpu().numpy()
                exposure[i, root, end] = trace["exposure"][0].cpu().numpy()
        if (i + 1) % 4 == 0:
            print(f"  Frozen bank {i + 1}/{len(panel['seed'])} layouts", flush=True)
    return output, exposure, root_c


@torch.inference_mode()
def predict_local(host, conditioner, gate, decode, scale, panel, *, evidence_terms=None, xy_cholesky=None):
    device = host.action_mean.device
    as_tensor = lambda x: torch.as_tensor(x, device=device)
    target = as_tensor(panel["future_z"])
    root = as_tensor(panel["root_z"])[:, :, None, None].expand(*target.shape[:4], 3, -1)
    past = as_tensor(panel["root_action"])[:, :, None, None].expand(*target.shape[:4], 2, 5, 2)
    action = as_tensor(panel["action"])[:, :, :, None].expand(*target.shape[:4], HORIZON, 5, 2)
    actual_history = torch.cat((root, target), dim=-2).flatten(0, 3)
    controls = torch.cat((past, action), dim=-3).flatten(0, 3)
    origin = torch.arange(2, device=device).expand(target.shape[:4]).flatten()
    counterfactual = as_tensor(panel["local_z"]).flatten(0, 3)
    valid = as_tensor(panel["local_valid"]).flatten(0, 3)
    c = decode(target).cpu().numpy()
    local = np.empty((len(origin), HORIZON, 2), dtype=np.float32)
    exposure = np.empty((len(origin), HORIZON), dtype=np.float32)
    gaps = np.full((len(origin), HORIZON, 2), np.nan, dtype=np.float32)
    if xy_cholesky is not None:
        if evidence_terms is None:
            raise ValueError("Supply evidence_terms to receive decoded XY gaps")
        gaps_xy = np.full(gaps.shape, np.nan, dtype=np.float64)
        xy_log_likelihood = np.full((*gaps.shape, 2), np.nan, dtype=np.float64)
    if evidence_terms is not None:
        terms = {name: np.full(exposure.shape, np.nan) for name in (
            "prediction_gap_std", "prediction_gap_px", "midpoint_residual_std",
            "midpoint_residual_px", "midpoint_projection_nat", "true_gap_projection_nat",
        )}
    for first in range(0, len(origin), 256):
        end = slice(first, first + 256)
        for h in range(HORIZON):
            trace = {}
            issued = forecast(host, conditioner, gate, actual_history[end, h:h + 3], controls[end, h:h + 2],
                              controls[end, None, h + 2:h + 3], compact=True, trace=trace)[:, 0, :, 0]
            chosen = origin[end, None, None].expand(-1, 1, issued.size(-1))
            local[end, h] = decode(issued.gather(1, chosen).squeeze(1)).cpu().numpy()
            exposure[end, h] = trace["exposure"][:, 0, 0, 0].cpu().numpy()
            eligible = valid[end, h]
            observed = counterfactual[end, h][eligible]
            original_arrival = observed.gather(1, chosen[eligible]).squeeze(1)
            torch.testing.assert_close(original_arrival, actual_history[end, h + 3][eligible], rtol=0, atol=0)
            pair = issued[eligible]
            prior = pair.new_full((len(pair), 2, 2), -np.log(2))
            _, likelihood = write_observation(prior, pair[:, None], observed, scale)
            signed = torch.stack((likelihood[:, 0, 0] - likelihood[:, 0, 1],
                                  likelihood[:, 1, 1] - likelihood[:, 1, 0]), dim=-1)
            gaps[end, h][eligible.cpu().numpy()] = signed.cpu().numpy()
            if xy_cholesky is not None:
                _, xy_ll = write_observation(prior, pair[:, None], observed, xy_cholesky, decode=decode)
                xy_signed = torch.stack((xy_ll[:, 0, 0] - xy_ll[:, 0, 1],
                                         xy_ll[:, 1, 1] - xy_ll[:, 1, 0]), dim=-1)
                gaps_xy[end, h][eligible.cpu().numpy()] = xy_signed.cpu().numpy()
                xy_log_likelihood[end, h][eligible.cpu().numpy()] = xy_ll.cpu().numpy()
            if evidence_terms is not None:
                pred, actual = pair.double(), observed.double()
                difference = (pred[:, 1] - pred[:, 0]) / scale
                midpoint = (actual.mean(dim=1) - pred.mean(dim=1)) / scale
                actual_difference = (actual[:, 1] - actual[:, 0]) / scale
                pred_xy, actual_xy = decode(pair), decode(observed)
                values = (
                    difference.norm(dim=-1), (pred_xy[:, 1] - pred_xy[:, 0]).norm(dim=-1),
                    midpoint.norm(dim=-1), (actual_xy.mean(dim=1) - pred_xy.mean(dim=1)).norm(dim=-1),
                    (midpoint * difference).sum(dim=-1), .5 * (actual_difference * difference).sum(dim=-1),
                )
                for term, value in zip(terms.values(), values, strict=True):
                    term[end, h][eligible.cpu().numpy()] = value.cpu().numpy()
    shape = target.shape[:-1]
    if evidence_terms is not None:
        evidence_terms.update({name: value.reshape(shape) for name, value in terms.items()})
        if xy_cholesky is not None:
            evidence_terms["xy_signed_gaps"] = gaps_xy.reshape(*shape, 2)
            evidence_terms["xy_log_likelihood"] = xy_log_likelihood.reshape(*shape, 2, 2)
    return c, local.reshape(*shape, 2), exposure.reshape(shape), gaps.reshape(*shape, 2)


def horizon_mean(x, mask):
    mask = np.broadcast_to(mask, x.shape)
    return [float(x[..., h][mask[..., h]].mean()) if mask[..., h].any() else None
            for h in range(HORIZON)]


def endpoint_report(c, local, free, truth, mask=True):
    truth, c, local, free = (a.astype(np.float64) for a in (truth, c, local, free))
    parts = {"C-Y": c - truth, "L-C": local - c, "F-L": free - local,
             "L-Y": local - truth, "F-Y": free - truth}
    np.testing.assert_allclose(parts["F-Y"], parts["C-Y"] + parts["L-C"] + parts["F-L"], rtol=0, atol=1e-10)
    mask = np.broadcast_to(mask, truth.shape[:-1])
    squared_error = {name: np.square(delta).sum(axis=-1) for name, delta in parts.items()}
    mode_truth = truth[..., 1, :, :] - truth[..., 0, :, :]
    mode_error = {name: (a[..., 1, :, :] - a[..., 0, :, :]) - mode_truth
                  for name, a in (("C", c), ("L", local), ("F", free))}
    mode_gain = {}
    for name, error in mode_error.items():
        numerator = horizon_mean(((error + mode_truth) * mode_truth).sum(-1), mask.all(axis=-2))
        denominator = horizon_mean(np.square(mode_truth).sum(-1), mask.all(axis=-2))
        mode_gain[name] = [n / d if d else None for n, d in zip(numerator, denominator, strict=True)]
    return {"branches_by_h": mask.reshape(-1, HORIZON).sum(axis=0).tolist(),
            "rmse_px": {name: [float(np.sqrt(mse)) if mse is not None else None
                              for mse in horizon_mean(values, mask)] for name, values in squared_error.items()},
            "common_component_rmse_px": {name: [float(np.sqrt(mse)) if mse is not None else None for mse in
                horizon_mean(np.square(delta.mean(axis=-3)).sum(axis=-1), mask.all(axis=-2))]
                for name, delta in parts.items()},
            "mode_gap_gain": mode_gain}


def first_event(events):
    first = np.where(events.any(axis=-1), events.argmax(axis=-1) + 1, 0)
    n = int(first.size)
    return {"branches": n, "blocks": int(events.size), "failed_branches": int((first > 0).sum()),
            "failed_blocks": int(events.sum()), "failed_branch_rate": float((first > 0).mean()) if n else None,
            "events_by_h": events.reshape(-1, HORIZON).sum(axis=0).tolist(),
            "first_event_0_none_1_to_5": np.bincount(first.ravel(), minlength=HORIZON + 1).tolist()}


def error_events(geom, reference_c=None):
    events = {"visible_wall": geom["visible_wall_depth"] > 0,
              "true_mode_wall": geom["true_mode_wall_depth"] > 0,
              "room_vs_factual_endpoint": geom["wrong_room"],
              "room_vs_prior_decoded_replay": geom["wrong_room_prior_decoded_replay"]}
    if reference_c is not None:
        events.update(new_visible_wall=events["visible_wall"] & ~(reference_c["visible_wall_depth"] > 0),
                      wrong_room_when_actual_encoding_correct=events["room_vs_factual_endpoint"] & ~reference_c["wrong_room"])
    return {name: first_event(value) for name, value in events.items()}


def progress_report(c, local, free, truth, mask, estimates):
    common = mask["common_blocked"]
    legal = mask["legal_route"]
    direction = mask["direction"][:, :, :, None]
    before = mask["before"]
    projection = {name: ((a - before) * direction).sum(axis=-1)
                  for name, a in (("Y", truth), ("C", c), ("L", local), ("F", free))}
    projection.update({"L-C": projection["L"] - projection["C"],
                       "F-L": projection["F"] - projection["L"], "F-Y": projection["F"] - projection["Y"]})
    shared = {name: values.mean(axis=-2) for name, values in projection.items()}
    progress = {name: mask["root_cost"] - value for name, value in estimates.items()}
    progress["F-Y"] = progress["F"] - progress["Y"]
    root = before[..., :1, :]
    displacement = truth - root
    length = np.linalg.norm(displacement, axis=-1, keepdims=True)
    travel = np.divide(displacement, length, out=np.zeros_like(displacement), where=length > 0)
    moved = legal[..., None] & (length[..., 0] > 0)
    legal_projection = {name: horizon_mean(((a - root) * travel).sum(axis=-1), moved)
                        for name, a in (("Y", truth), ("C", c), ("L", local), ("F", free))}
    return {"common_blocked": {"paired_blocks_by_h": common.reshape(-1, HORIZON).sum(0).tolist(),
                               "mean_forward_px": {name: horizon_mean(values, common) for name, values in shared.items()}},
            "legal_route": {"branches": int(legal.sum()), "moved_by_h": moved.reshape(-1, HORIZON).sum(0).tolist(),
                            "mean_physical_displacement_projection_px": legal_projection, "mean_goal_progress_px": {
                name: horizon_mean(values, legal[..., None]) for name, values in progress.items()}}}


def gap_stats(values):
    n = len(values)
    return {"n": n, "positive": int((values > 0).sum()), "zero": int((values == 0).sum()),
            "negative": int((values < 0).sum()), "mean_nat": float(values.mean(dtype=np.float64)) if n else None,
            "median_nat": float(np.median(values)) if n else None,
            "p95_nat": float(np.quantile(values, .95)) if n else None,
            "p99_nat": float(np.quantile(values, .99)) if n else None, "max_nat": float(values.max()) if n else None}


def evidence_report(gaps, exposure, masks):
    eligible, end, raw = (masks[name] for name in ("eligible", "end_change", "raw_change"))
    if not np.isfinite(gaps[eligible]).all():
        raise FloatingPointError("Nonfinite likelihood on a valid paired source")
    relevant = gaps[eligible & end]
    min_gap = relevant.min(axis=-1)
    jointly = gap_stats(min_gap)
    jointly["positive_rate"] = jointly["positive"] / jointly["n"] if jointly["n"] else None
    negative = (relevant < 0).any(axis=-1)
    jointly["wrong_on_either"] = int(negative.sum())
    jointly["both_wrong"] = int((relevant < 0).all(axis=-1).sum())
    jointly["both_tied"] = int((relevant == 0).all(axis=-1).sum())
    neutral = {}
    for name, pick in (("all_block_equal", eligible & ~end), ("all_raw_equal", eligible & ~raw),
                       ("temporarily_distinct_then_equal", eligible & ~end & raw),
                       ("common_wall_blocked", eligible & masks["common_blocked"][..., None, :])):
        false = gaps[pick]
        np.testing.assert_array_equal(false[:, 0], -false[:, 1])
        neutral[name] = {"absolute": gap_stats(np.abs(false[:, 1])), "signed_mode1": gap_stats(false[:, 1])}
    by_h = [gate_counts(end[..., h][eligible[..., h]], exposure[..., h][eligible[..., h]] > 0)
            for h in range(HORIZON)]
    return {"paired_sources": int(eligible.sum()),
            "invalid_origin_block_rows": int((~masks["valid"] & masks["unique_slot"][..., None, None]).sum()),
            "physically_distinct_by_raw_block": {"raw": int((raw & eligible).sum()),
                                                "block_end": int((end & eligible).sum())},
            "informative_signed": {"true_mode0": gap_stats(relevant[:, 0]), "true_mode1": gap_stats(relevant[:, 1]),
                                   "min_both": jointly}, "neutral_false_evidence": neutral,
            "gate_by_h": by_h, "gate_total": gate_counts(end[eligible], exposure[eligible] > 0)}


def ranking_report(forecast_cost, true_cost, forecast_xy, true_xy, goal_room):
    """Use the deployed stable argmin and score only the matching true-mode branch."""
    n_plans = forecast_cost.shape[-2]
    shape = forecast_cost.shape[:-2] + (2,)
    model = forecast_cost.swapaxes(-2, -1).reshape(-1, n_plans)
    actual = true_cost.swapaxes(-2, -1).reshape(-1, n_plans)
    model_rooms = room(forecast_xy).swapaxes(-2, -1).reshape(-1, n_plans)
    actual_rooms = room(true_xy).swapaxes(-2, -1).reshape(-1, n_plans)
    goal_room = np.broadcast_to(goal_room[..., None], shape).reshape(-1)
    pick = model.argmin(axis=-1)
    best = actual.argmin(axis=-1)
    rows = np.arange(len(pick))
    chosen = actual[rows, pick]
    optimal = actual[rows, best]
    p, q = np.triu_indices(n_plans, 1)
    rank = np.empty((len(pick), 3), dtype=np.int64)
    for i, (forecast, truth) in enumerate(zip(model, actual, strict=True)):
        d_true, d_forecast = truth[p] - truth[q], forecast[p] - forecast[q]
        counts = np.sign(d_true) * np.sign(d_forecast)
        rank[i] = ((d_true != 0).sum(), (counts < 0).sum(), ((d_true != 0) & (d_forecast == 0)).sum())
    false_goal = (model_rooms == goal_room[:, None]) & (actual_rooms != goal_room[:, None])
    elites = np.argsort(model, axis=-1, kind="stable")[:, :12]
    result = {"selected_index": pick.reshape(shape), "true_best_index": best.reshape(shape),
              "selected_true_cost": chosen.reshape(shape), "selected_forecast_cost": model[rows, pick].reshape(shape),
              "regret": (chosen - optimal).reshape(shape), "true_best_cost": optimal.reshape(shape),
              "selected_true_rank": (1 + (actual < chosen[:, None]).sum(axis=-1)).reshape(shape),
              "forecast_rank_of_physical_best": (1 + (model < model[rows, best, None]).sum(axis=-1)).reshape(shape),
              "rank_pairs_truth_non_tied_inverted_forecast_tied": rank.reshape(*shape, 3),
              "selected_false_goal_room": false_goal[rows, pick].reshape(shape),
              "elites_false_goal_count": np.take_along_axis(false_goal, elites, axis=-1).sum(axis=-1).reshape(shape)}
    return result


def reduce_ranks(ranks):
    comparable, inversions, ties = ranks["rank_pairs_truth_non_tied_inverted_forecast_tied"].reshape(-1, 3).sum(axis=0)
    return {"root_mode_pairs": int(ranks["regret"].size), "mean_regret_px": float(ranks["regret"].mean()),
            "zero_regret": int((ranks["regret"] == 0).sum()), "mean_selected_true_cost_px": float(ranks["selected_true_cost"].mean()),
            "mean_selected_forecast_cost_px": float(ranks["selected_forecast_cost"].mean()),
            "mean_true_best_cost_px": float(ranks["true_best_cost"].mean()),
            "mean_selected_true_rank": float(ranks["selected_true_rank"].mean()),
            "mean_forecast_rank_of_physical_best": float(ranks["forecast_rank_of_physical_best"].mean()),
            "true_cost_non_tied_pairs": int(comparable), "inverted_pairs": int(inversions), "forecast_tied_pairs": int(ties),
            "inverted_fraction": float(inversions / comparable) if comparable else None,
            "selected_false_goal_room": int(ranks["selected_false_goal_room"].sum()),
            "top12_false_goal_count": int(ranks["elites_false_goal_count"].sum())}


def select_layout(data, row):
    return {key: value[row] for key, value in data.items()}


def summarize(row, small, c, local, free, truth, geom, bank_geometry, masks, costs, gaps, exposure, ranks):
    c, local, free, truth = (a[row] for a in (c, local, free, truth))
    roles = small["selection_role"][row]
    local_masks = select_layout(masks, row)
    unique = local_masks["unique_slot"]
    geom = {key: select_layout(select_layout(value, row), unique) for key, value in geom.items()}
    evidence = evidence_report(gaps[row], exposure[row], local_masks)
    corners = roles == 2
    return {
        "diagnostic_slots": int(roles.size),
        "bank_indices_distinct_within_root": int(unique.sum()),
        "selection_role_counts": {name: int((roles == key).sum()) for key, name in ROLES.items()},
        "endpoint": endpoint_report(c, local, free, truth, unique[..., None, None]),
        "endpoint_by_root": {name: endpoint_report(c[:, i], local[:, i], free[:, i], truth[:, i], unique[:, i, :, None, None])
                             for i, name in enumerate(ROOT_NAMES)},
        "endpoint_common_corner_selected": endpoint_report(c, local, free, truth, corners[..., None, None]),
        "geometry_diagnostic": {name: error_events(values, None if name == "C" else geom["C"]) for name, values in geom.items()},
        "geometry_full_bank": error_events(select_layout(bank_geometry, row)),
        "geometry_bank_common_corner": error_events(select_layout(select_layout(bank_geometry, row), masks["bank_common_corner"][row])),
        "progress": progress_report(c, local, free, truth, local_masks, select_layout(costs, row)),
        "evidence": evidence,
        "ranking_h5": {name: {"all_roots": reduce_ranks(select_layout(pool, row)),
                            "navigation_roots": reduce_ranks({key: value[row, 2:] for key, value in pool.items()})}
                       for name, pool in ranks.items()},
    }


def selected_measures(report):
    prog = report["progress"]
    e = report["evidence"]
    gate = e["gate_total"]
    rank = report["ranking_h5"]["full388"]["navigation_roots"]
    scalars = {
        "bank_free_any_visible_wall_rate": report["geometry_full_bank"]["visible_wall"]["failed_branch_rate"],
        "bank_free_any_wrong_room_rate": report["geometry_full_bank"]["room_vs_factual_endpoint"]["failed_branch_rate"],
        "bank_free_any_wrong_room_from_prior_decoded_rate": report["geometry_full_bank"]["room_vs_prior_decoded_replay"]["failed_branch_rate"],
        "navigation_h5_mean_regret_px": rank["mean_regret_px"],
        "navigation_h5_inverted_rank_fraction": rank["inverted_fraction"],
        "local_gate_fp_rate": gate["false_positive"] / gate["inactive"] if gate["inactive"] else None,
        "local_gate_fn_rate": gate["false_negative"] / gate["active"] if gate["active"] else None,
        "local_both_likelihood_signs_correct_rate": e["informative_signed"]["min_both"]["positive_rate"],
        "local_neutral_mean_abs_evidence_nat": e["neutral_false_evidence"]["all_block_equal"]["absolute"]["mean_nat"],
        "local_neutral_p95_abs_evidence_nat": e["neutral_false_evidence"]["all_block_equal"]["absolute"]["p95_nat"],
    }
    for h in range(HORIZON):
        for component in ("C-Y", "L-C", "F-L", "F-Y"):
            scalars[f"h{h + 1}_rmse_{component}_px"] = report["endpoint"]["rmse_px"][component][h]
            if component != "C-Y":
                scalars[f"h{h + 1}_common_blocked_{component}_px"] = prog["common_blocked"]["mean_forward_px"][component][h]
        scalars[f"h{h + 1}_legal_route_F-Y_px"] = prog["legal_route"]["mean_goal_progress_px"]["F-Y"][h]
        travel = prog["legal_route"]["mean_physical_displacement_projection_px"]
        scalars[f"h{h + 1}_legal_route_projection_F-Y_px"] = (
            travel["F"][h] - travel["Y"][h] if travel["Y"][h] is not None else None
        )
    return scalars


def paired_contrasts(layouts_by_arm):
    rng = np.random.default_rng(20261115)
    pairs = (("query", "old"), ("prefix", "query"), ("prefix", "old"), ("old", "reference"))
    results = {}
    for newer, earlier in pairs:
        if newer not in layouts_by_arm or earlier not in layouts_by_arm:
            continue
        current = [selected_measures(layout) for layout in layouts_by_arm[newer]]
        baseline = [selected_measures(layout) for layout in layouts_by_arm[earlier]]
        group = {}
        for key in current[0]:
            complete = np.array([a[key] is not None and b[key] is not None for a, b in zip(current, baseline, strict=True)])
            delta = np.array([a[key] - b[key] for a, b, good in zip(current, baseline, complete, strict=True) if good])
            if len(delta):
                resampled = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=-1)
                interval = np.quantile(resampled, [.025, .975]).tolist()
            else:
                interval = None
            values = iter(delta.tolist())
            group[key] = {"layouts": len(delta), "mean_delta": float(delta.mean()) if len(delta) else None,
                          "percentile_bootstrap_95": interval, "positive": int((delta > 0).sum()),
                          "negative": int((delta < 0).sum()), "zero": int((delta == 0).sum()),
                          "delta_by_seed": {str(seed): next(values) if good else None for seed, good in zip(SEEDS, complete, strict=True)}}
        results[f"{newer}_minus_{earlier}"] = group
    return results


def run_arm(run, name, device, decode, scale, small, bank, masks, truth_cost, goal_room):
    host, conditioner, gate = load_trained(REFERENCE if name == "reference" else run / name, "conditioned", device)
    bank_free, bank_exposure, root_c = predict_bank(host, conditioner, gate, decode, bank)
    c, local, local_gate, gaps = predict_local(host, conditioner, gate, decode, scale, small)
    if not all(np.isfinite(a).all() for a in (bank_free, c, local, local_gate)):
        raise FloatingPointError(f"Nonfinite prediction in {name}")
    indices = (np.arange(len(SEEDS))[:, None, None], np.arange(len(ROOT_NAMES))[None, :, None], small["bank_index"])
    free = bank_free[indices]
    np.testing.assert_allclose(local[..., 0, :], free[..., 0, :], rtol=2e-4, atol=2e-4)
    bank_geom = geometry(bank, bank_free, root_c, verify=name == "reference")
    geom = {"C": geometry(small, c, root_c), "L": geometry(small, local, root_c, prior_xy=c),
            "F": select_layout(bank_geom, indices)}
    truth = small["future_xy"][..., -1, :]
    bank_truth = bank["future_xy"][..., -1, :]
    bank_cost = scores(bank, bank_free)
    costs = {"Y": truth_cost[indices], "C": scores(small, c), "L": scores(small, local), "F": bank_cost[indices]}
    if name == "reference":
        np.testing.assert_allclose(bank_free[..., -1, :], bank["parent_xy"], rtol=2e-4, atol=2e-4)
        np.testing.assert_allclose(bank_cost[..., -1], bank["parent_cost"], rtol=2e-4, atol=2e-4)
    ranks = {}
    for key, start in (("full388", 0), ("cem384", 4)):
        pred = bank_cost[:, :, start:, :, -1]
        actual = truth_cost[:, :, start:, :, -1]
        ranks[key] = ranking_report(pred, actual, bank_free[:, :, start:, :, -1], bank_truth[:, :, start:, :, -1], goal_room)
        for index in ("selected_index", "true_best_index"):
            ranks[key][index] += start
    raw = {"seed": small["seed"], "bank_index": small["bank_index"], "selection_role": small["selection_role"],
           "Y": truth, "C": c, "L": local, "F": free, "F_bank": bank_free, "Y_bank": bank_truth,
           "F_bank_exposure": bank_exposure, "L_exposure": local_gate, "signed_likelihood_gaps": gaps,
           "Y_bank_cost": truth_cost, "F_bank_cost": bank_cost, "root_C": root_c}
    raw.update(masks)
    for key, values in geom.items():
        raw.update({f"{key}_{name}": value for name, value in values.items()})
    raw.update({f"bank_F_{key}": value for key, value in bank_geom.items()})
    for key, values in ranks.items():
        raw.update({f"{key}_{name}": value for name, value in values.items()})
    np.savez_compressed(run / f"evaluation_{name}_arrays.npz", **raw)
    inputs = small, c, local, free, truth, geom, bank_geom, masks, costs, gaps, local_gate, ranks
    full = summarize(slice(None), *inputs)
    per_layout = [summarize(slice(i, i + 1), *inputs) for i in range(len(SEEDS))]
    return full, per_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    options = parser.parse_args()
    run, device = options.run, options.device
    torch.set_num_threads(1)
    small, bank = load_panels(run)
    masks = factual_masks(small)
    masks["bank_common_corner"] = bank["common_corner"]
    decode = load_readout(device, REFERENCE)
    scale = torch.load(REFERENCE / "likelihood_scale.pt", map_location=device, weights_only=True)
    if not (torch.isfinite(scale) & (scale > 0)).all():
        raise ValueError("Expected archived finite positive likelihood scales")
    truth_cost = scores(bank, bank["future_xy"][..., -1, :])
    goal_room = np.broadcast_to(np.array([2, 0, 2, 0])[None], bank["root_z"].shape[:2])
    summary, layouts = {}, {}
    for arm in options.arms:
        print(f"Scoring {arm}", flush=True)
        summary[arm], layouts[arm] = run_arm(run, arm, device, decode, scale, small, bank, masks, truth_cost, goal_room)
    protocol = {
        "seed": SEEDS.tolist(), "roots": ROOT_NAMES,
        "populations": {"full388": "four original H5 plans then 384 round-major frozen parent CEM queries",
                        "diagnostic": "eight archived slots per root, with six distinct bank indices used once in metrics"},
        "C/L/F": "decoded actual arrival / one-step true mode at prior actual three-frame window / matching free-rollout mode",
        "errors": "Euclidean endpoint RMSE and the first positive collision-strip depth or wrong endpoint room",
        "prior_decoded_replay": "exact same mode and five controls from preceding F for F or preceding C for C/L (frozen root readout at H1)",
        "progress": "common-block projection from factual start onto net command; legal-route goal progress and signed projection along actual root displacement",
        "legal_route": "navigation branch with an actual raw crossing and lower physical H5 goal cost than root",
        "evidence": "frozen archived scales, paired arrivals from each valid actual same-prefix source, shared H1 counted once per bank index",
        "paired": "layout metric deltas, complete layouts only, 10000 resamples of whole paired layouts",
        "selection": "selected frozen CEM tail is conditioned on the original scorer; bank ranking and room flags also use all proposals",
    }
    report = {"protocol": protocol, "absolute": summary, "per_layout": {
                    str(seed): {arm: rows[i] for arm, rows in layouts.items()} for i, seed in enumerate(SEEDS)},
              "paired_contrasts": paired_contrasts(layouts)}
    (run / "evaluation_summary.json").write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n",
                                                 encoding="utf-8")


if __name__ == "__main__":
    main()
