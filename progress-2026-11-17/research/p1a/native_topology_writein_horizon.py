"""Replay each frozen FourRoom plan and read the same learned/physical prefixes."""

import argparse
import csv
import json
from itertools import combinations

import numpy as np
import torch
from scipy.stats import rankdata

from .native_topology_prefix_data import replay_bank
from .native_topology_writein_attribution import supports
from .native_topology_writein_replay import (
    BATCH, CHECKPOINTS, HORIZON, REFERENCE, RUN, STAGES, forecast, load_readout, load_trained,
)
from .physical_topology_gate import PhysicalRollout, VisibleMapCost, make_scene, room


OUTPUT = RUN / "writein_attribution_20261116"
SUPPORTS = ("pre", "post", "omit", "uniform", "oracle")


def read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return dict(source)


@torch.no_grad()
def replay_prefixes(banks, device, *, force=None, collect_gate=False, replay_physical=True):
    torch.backends.cuda.matmul.allow_tf32 = False
    modules = load_trained(CHECKPOINTS["paired"], "conditioned", device)
    decode = load_readout(device, REFERENCE)
    shape = (*banks["physical_cost"].shape, HORIZON)
    arms = ("physical", "learned") if replay_physical else ("learned",)
    traces = {f"{arm}_xy": np.empty((*shape, 2), dtype=np.float32) for arm in arms}
    traces.update({f"{arm}_cost": np.empty(shape, dtype=np.float32) for arm in arms})
    if collect_gate:
        traces.update(exposure=np.empty(shape, dtype=np.float32), gate_logits=np.empty(shape, dtype=np.float32),
                      decoded_root_xy=np.empty((shape[0], 2), dtype=np.float32))
    plans = banks["action"]
    for pool, seed in enumerate(banks["seed"]):
        scene = make_scene(int(seed))
        geometry = VisibleMapCost(scene, scene.southeast)
        if replay_physical:
            physical = PhysicalRollout(scene)
            xy = replay_bank(physical, banks["root_xy"][pool], plans[pool])[:, :, :, -1].transpose(1, 0, 2, 3)
            traces["physical_xy"][pool] = xy
            traces["physical_cost"][pool] = geometry(xy)

        h, past, controls = (torch.as_tensor(banks[k][pool:pool + 1], device=device)
                             for k in ("root_z", "root_action", "action"))
        if collect_gate:
            traces["decoded_root_xy"][pool] = decode(h[:, -1]).cpu().numpy()[0]
        for start in range(0, controls.size(1), BATCH):
            sl = slice(start, start + BATCH)
            trace = {} if collect_gate else None
            forecast_z = forecast(*modules, h, past, controls[:, sl], compact=True, trace=trace, force=force)[0]
            if collect_gate:
                for key, trace_key in (("exposure", "exposure"), ("gate_logits", "logits")):
                    traces[key][pool, :, sl] = trace[trace_key][0].cpu().numpy().transpose(1, 0, 2)
            for horizon in range(HORIZON):
                xy = decode(forecast_z[..., horizon, :]).cpu().numpy().transpose(1, 0, 2)
                traces["learned_xy"][pool, :, sl, horizon] = xy
                traces["learned_cost"][pool, :, sl, horizon] = geometry(xy)
        if (pool + 1) % 12 == 0:
            print(f"Replayed {pool + 1}/{len(banks['seed'])} fixed plan pools", flush=True)
    return traces


def check_terminals(traces, banks, replay):
    errors = {}
    for arm, reference in (("physical", banks["physical_cost"]), ("learned", replay["learned_cost"])):
        actual = traces[f"{arm}_cost"][..., -1]
        np.testing.assert_array_equal(actual, reference)
        errors[f"{arm}_h5_max_abs_cost_error"] = float(np.max(np.abs(actual - reference)))
    for values in traces.values():
        assert np.isfinite(values).all()
    return errors


def rank_correlation(a, b):
    a, b = (rankdata(cost, axis=2) for cost in (a, b))
    a, b = (ranks - ranks.mean(axis=2, keepdims=True) for ranks in (a, b))
    covariance = (a * b).sum(axis=2)
    norm = np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)
    return np.divide(covariance, norm, out=np.full_like(covariance, np.nan), where=norm > 0)


def score_queries(traces, banks, replay):
    rows, initial, time, stage, pool = banks["choices"].T
    query_index = np.arange(len(rows))
    mode = replay["choice_mode"]
    np.testing.assert_array_equal(stage, replay["choice_stage"])
    np.testing.assert_array_equal(banks["seed"][pool], np.unique(banks["seed"])[rows])
    weights, _, _ = supports(replay, banks["choices"], mode)

    expected = {}
    for arm in ("learned", "physical"):
        cost = traces[f"{arm}_cost"][pool].astype(np.float64)
        b = cost.mean(axis=1)
        d = cost[:, 1] - cost[:, 0]
        expected[arm] = b[:, None] + (weights[..., 1, None, None] - 0.5) * d[:, None]
    best = expected["learned"].argmin(axis=2)
    physical_best = expected["physical"].argmin(axis=2)
    np.testing.assert_array_equal(physical_best[:, -1, -1], replay["choice_physical_best"])
    np.testing.assert_array_equal(physical_best[:, 3, -1], replay["choice_physical_uniform"])

    truth = traces["physical_cost"][pool, mode]
    q, h = query_index[:, None, None], np.arange(HORIZON)[None, None, :]
    best_cost = truth[q, best, h]
    min_cost = truth.min(axis=1)[:, None]
    # The old H5 replay used a per-probe omission at all three probe stages.
    legacy_best = (replay["choice_weights"] @ traces["learned_cost"][pool, ..., -1]).argmin(axis=2)
    legacy_cost = truth[query_index[:, None], legacy_best, -1]
    np.testing.assert_array_equal(legacy_best, replay["choice_best"])
    np.testing.assert_array_equal(legacy_cost, replay["choice_true_cost"])
    np.testing.assert_array_equal(legacy_cost - min_cost[..., -1], replay["choice_regret"])
    first = banks["action"][pool, :, 0]
    chosen_first = first[q, best]
    same_first = (first[:, None, :, None] == chosen_first[:, :, None]).all(axis=(-1, -2))
    first_cost = np.where(same_first, truth[:, None], np.inf).min(axis=2)
    target_room = 2
    actual_room = room(traces["physical_xy"][pool, mode])[q, best, h]
    learned_room = room(traces["learned_xy"][pool, mode])[q, best, h]

    # Retain the H5-selected pair across all prefixes, in the order learned uniform, true physical.
    pair = np.stack((best[:, 3, -1], physical_best[:, -1, -1]), axis=1)
    pair_first = first[query_index[:, None], pair]
    pair_same_first = (pair_first[:, 0] == pair_first[:, 1]).all(axis=(-1, -2))
    pair_truth_h5_margin = truth[query_index, pair[:, 0], -1] - truth[query_index, pair[:, 1], -1]
    pair_stats = {}
    for arm in ("learned", "physical"):
        for field in ("xy", "cost"):
            values = traces[f"{arm}_{field}"].astype(np.float64)
            pair_values = values[pool[:, None], :, pair]
            pair_stats[f"pair_{arm}_{field}"] = pair_values
    return dict(
        rows=rows, initial=initial, time=time, stage=stage, pool=pool, mode=mode, weights=weights,
        learned_best=best, physical_best=physical_best,
        learned_best_true_cost=best_cost, physical_true_optimal_cost=min_cost[:, 0],
        plan_regret=best_cost - min_cost, first_regret=first_cost - min_cost,
        suffix_regret=best_cost - first_cost, true_cost_rank=1 + (truth[:, None] < best_cost[:, :, None]).sum(axis=2),
        same_support_spearman=rank_correlation(expected["learned"], expected["physical"]),
        true_mode_spearman=rank_correlation(expected["learned"], truth[:, None]),
        chosen_actual_room=actual_room, chosen_learned_room=learned_room,
        chosen_false_target=(learned_room == target_room) & (actual_room != target_room),
        pair=pair, pair_same_first=pair_same_first, pair_truth_h5_margin=pair_truth_h5_margin,
        **pair_stats,
    )


def effect_metrics(learned, physical):
    """Measure signed agreement when a physical effect exists, and false effects on zero rows."""
    predicted_norm = np.linalg.norm(learned, axis=-1)
    physical_norm = np.linalg.norm(physical, axis=-1)
    dot = (learned * physical).sum(axis=-1)
    active = physical_norm > 0
    predicted = predicted_norm > 0
    both = active & predicted
    aligned = both & (dot > 0)
    reversed_ = both & (dot < 0)

    def median(values):
        return float(np.median(values)) if values.size else None

    return dict(
        count=int(active.size), true_nonzero=int(active.sum()), true_zero=int((~active).sum()),
        aligned=int(aligned.sum()), reversed=int(reversed_.sum()), orthogonal=int((both & (dot == 0)).sum()),
        missed=int((active & ~predicted).sum()), predicted_nonzero_on_true_zero=int((~active & predicted).sum()),
        median_physical_magnitude=median(physical_norm[active]),
        median_learned_magnitude_on_true_nonzero=median(predicted_norm[active]),
        median_learned_magnitude_on_true_zero=median(predicted_norm[~active]),
        median_magnitude_ratio=median(predicted_norm[active] / physical_norm[active]),
        median_signed_gain=median(dot[active] / physical_norm[active] ** 2),
        median_cosine_when_both_nonzero=median(dot[both] / (physical_norm[both] * predicted_norm[both])),
    )


def prefix_metrics(traces, queries, pools, query_ids, horizon):
    learned_xy, physical_xy = (traces[f"{arm}_xy"][pools, ..., horizon, :].astype(np.float64)
                               for arm in ("learned", "physical"))
    learned_cost, physical_cost = (traces[f"{arm}_cost"][pools, ..., horizon, None].astype(np.float64)
                                   for arm in ("learned", "physical"))
    xy_error = learned_xy.mean(axis=1) - physical_xy.mean(axis=1)
    cost_error = learned_cost.mean(axis=1) - physical_cost.mean(axis=1)
    errors = {}
    for name, error in (
        ("xy_midpoint", xy_error), ("mean_cost", cost_error),
        ("xy_mode_effect", (learned_xy - physical_xy)[:, 1] - (learned_xy - physical_xy)[:, 0]),
        ("cost_mode_effect", (learned_cost - physical_cost)[:, 1] - (learned_cost - physical_cost)[:, 0]),
    ):
        magnitude = np.linalg.norm(error, axis=-1)
        errors[name] = dict(mean_magnitude=float(magnitude.mean()), median_magnitude=float(np.median(magnitude)),
                            rmse=float(np.sqrt(np.square(magnitude).mean())))
    same_plan = {
        "xy_mode_effect": effect_metrics(learned_xy[:, 1] - learned_xy[:, 0], physical_xy[:, 1] - physical_xy[:, 0]),
        "cost_mode_effect": effect_metrics(learned_cost[:, 1] - learned_cost[:, 0], physical_cost[:, 1] - physical_cost[:, 0]),
        "errors": errors,
        "learned_room_error": int((room(learned_xy) != room(physical_xy)).sum()),
        "both_modes_false_target_room": int(((room(learned_xy) == 2) & (room(physical_xy) != 2)).all(axis=1).sum()),
    }
    pairs = {}
    for name, take in (
        ("all", query_ids),
        ("strict_h5_first_command_miss", query_ids & ~queries["pair_same_first"] & (queries["pair_truth_h5_margin"] > 0)),
    ):
        metrics, alignment = {}, {}
        for field in ("xy", "cost"):
            pair = (queries[f"pair_{arm}_{field}"][take] for arm in ("learned", "physical"))
            learned, physical = (values[..., horizon, :] if field == "xy" else values[..., horizon, None] for values in pair)
            learned, physical = (values[:, 0] - values[:, 1] for values in (learned, physical))
            learned_d, physical_d = (values[:, 1] - values[:, 0] for values in (learned, physical))
            metrics[f"{field}_action_mode_interaction"] = effect_metrics(learned_d, physical_d)
            alignment[field] = (learned_d * physical_d).sum(axis=-1)
            metrics[f"{field}_common_action_difference"] = effect_metrics(learned.mean(axis=1), physical.mean(axis=1))
            common_error = np.linalg.norm((learned - physical).mean(axis=1), axis=-1)
            metrics[f"median_{field}_common_action_error"] = float(np.median(common_error)) if common_error.size else None
        metrics["xy_aligned_cost_reversed"] = int(((alignment["xy"] > 0) & (alignment["cost"] < 0)).sum())
        pairs[name] = metrics

    ranking = {}
    for j, name in enumerate(SUPPORTS):
        def values(key):
            return queries[key][query_ids, j, horizon]
        ranking[name] = {
            "count": int(query_ids.sum()),
            "true_optimal_first": int((values("first_regret") == 0).sum()),
            "true_optimal_plan": int((values("plan_regret") == 0).sum()),
            "same_plan_as_physical_with_same_support": int((values("learned_best") == values("physical_best")).sum()),
            "mean_first_regret": float(values("first_regret").mean()),
            "mean_suffix_regret": float(values("suffix_regret").mean()),
            "mean_plan_regret": float(values("plan_regret").mean()),
            "median_true_cost_rank": float(np.median(values("true_cost_rank"))),
            "median_same_support_spearman": finite_median(values("same_support_spearman")),
            "median_true_mode_spearman": finite_median(values("true_mode_spearman")),
            "chosen_false_target": int(values("chosen_false_target").sum()),
        }
    return dict(same_plan=same_plan, fixed_pair=pairs, ranking=ranking)


def finite_median(values):
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def training_pair_coverage():
    """Read the physical cost contrasts already present in the four-candidate fit archive."""
    metadata = json.loads((RUN / "data_summary.json").read_text(encoding="utf-8"))
    with np.load(RUN / "data_train_paired.npz", allow_pickle=False) as data:
        seeds, roles, endpoint = (data[key] for key in ("seed", "selection_role", "future_xy"))
    with np.load(RUN / "data_train_refresh_bank.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(seeds, data["seed"])
        original_xy = data["future_xy"][:, :, :4, :, -1, -1]
        original_cost = data["actual_cost"][:, :, :4]
    control_path = CHECKPOINTS["start"].parent / "data_train_coverage.npz"
    with np.load(control_path, allow_pickle=False) as data:
        np.testing.assert_array_equal(seeds, data["seed"])
        control_xy = data["future_xy"][:, 2:, :, :, -1, -1]
    np.testing.assert_array_equal(seeds, metadata["layout_seeds"])
    rows = []
    for row in metadata["rows"]:
        i = np.searchsorted(seeds, row["seed"])
        np.testing.assert_equal(seeds[i], row["seed"])
        root = metadata["root_names"].index(row["root"])
        bad, good, mode = row["bad_slot"], row["legal_slot"], row["mode"]
        np.testing.assert_array_equal(roles[i, root, [bad, good]], [6, 7])
        scene = make_scene(row["seed"])
        target = (scene.southeast, scene.northwest)[root - 2]
        geometry = VisibleMapCost(scene, target)
        paired_cost = geometry(endpoint[i, root, :, :, -1, -1])
        old_cost = geometry(original_xy[i, root - 2])
        np.testing.assert_array_equal(paired_cost[[bad, good], mode], row["actual_h5_costs"])
        np.testing.assert_array_equal(old_cost, original_cost[i, root - 2])
        control_cost = geometry(control_xy[i, root - 2])
        for library, cost in (("paired_fit", paired_cost), ("coverage_fit", control_cost), ("original_menu", old_cost)):
            for a, b in combinations(range(len(cost)), 2):
                designated = library == "paired_fit" and {a, b} == {bad, good}
                if designated:
                    a, b = bad, good
                d0, d1 = cost[a].astype(np.float64) - cost[b]
                rows.append(dict(seed=row["seed"], root=row["root"], library=library, source_true_mode=mode,
                                 slot_a=a, slot_b=b, designated_pair=designated,
                                 slot_a_mode0_cost=float(cost[a, 0]), slot_a_mode1_cost=float(cost[a, 1]),
                                 slot_b_mode0_cost=float(cost[b, 0]), slot_b_mode1_cost=float(cost[b, 1]),
                                 delta_J0=float(d0), delta_J1=float(d1), delta_B=float((d0 + d1) / 2),
                                 delta_D=float(d1 - d0), ranking_reverses=bool(d0 * d1 < 0)))

    def summary(records):
        d0, d1, b, d = (np.array([row[k] for row in records]) for k in ("delta_J0", "delta_J1", "delta_B", "delta_D"))
        return dict(pairs=len(records), ranking_reverses=int((d0 * d1 < 0).sum()),
                    roots_with_ranking_reversal=len({(row["seed"], row["root"]) for row in records if row["ranking_reverses"]}),
                    slot_b_better_both=int(((d0 > 0) & (d1 > 0)).sum()),
                    slot_a_better_both=int(((d0 < 0) & (d1 < 0)).sum()),
                    cost_tie_either_mode=int(((d0 == 0) | (d1 == 0)).sum()),
                    delta_D_nonzero=int((d != 0).sum()),
                    median_delta_J0=float(np.median(d0)), median_delta_J1=float(np.median(d1)),
                    median_abs_delta_D=float(np.median(np.abs(d))), median_abs_delta_B=float(np.median(np.abs(b))))

    designated = [row for row in rows if row["designated_pair"]]
    original = [row for row in rows if row["library"] == "original_menu"]
    return rows, {
        "designated": summary(designated),
        "designated_by_root": {root: summary([row for row in designated if row["root"] == root])
                               for root in metadata["root_names"][2:]},
        "all_slot_pairs": summary([row for row in rows if row["library"] == "paired_fit"]),
        "coverage_fit_slot_pairs": summary([row for row in rows if row["library"] == "coverage_fit"]),
        "original_menu_slot_pairs": summary(original),
        "original_visible_route_pairs": summary([row for row in original if (row["slot_a"], row["slot_b"]) == (0, 1)]),
        "original_mode_cem_pairs": summary([row for row in original if (row["slot_a"], row["slot_b"]) == (2, 3)]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summarize", action="store_true", help="Read the completed fixed-prefix replay")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    banks = read_npz(RUN / "writein_fixed_h5_banks.npz")
    replay = read_npz(RUN / "writein_paired.npz")
    path = OUTPUT / "horizon_traces.npz"
    if args.summarize:
        traces = read_npz(path)
    else:
        traces = replay_prefixes(banks, torch.device(args.device))
    errors = check_terminals(traces, banks, replay)
    if not args.summarize:
        np.savez_compressed(path, **traces)
    print(json.dumps(errors), flush=True)
    queries = score_queries(traces, banks, replay)
    np.testing.assert_allclose(queries["first_regret"] + queries["suffix_regret"], queries["plan_regret"], atol=1e-5)
    np.savez_compressed(OUTPUT / "horizon_queries.npz", **queries)
    quiet = queries["stage"] == len(STAGES) - 1
    northwest = np.unique(queries["pool"][quiet])
    np.testing.assert_array_equal(northwest, np.arange(2, len(banks["seed"]), 3))
    summary = {"h5_checks": errors, "supports": SUPPORTS, "scopes": {}}
    summary["quiet_pre_post"] = {
        "max_abs_support_difference": float(np.max(np.abs(queries["weights"][quiet, 0] - queries["weights"][quiet, 1]))),
        "changed_learned_plan_by_prefix": (queries["learned_best"][quiet, 0] != queries["learned_best"][quiet, 1]).sum(0).tolist(),
        "changed_physical_plan_by_prefix": (queries["physical_best"][quiet, 0] != queries["physical_best"][quiet, 1]).sum(0).tolist(),
        "equal_log_likelihoods": int(quiet.sum()),
    }
    ranking_rows = []
    for name, pools, query_ids in (
        ("all", np.arange(len(banks["seed"])), np.ones_like(quiet)), ("northwest_quiet", northwest, quiet),
    ):
        horizons = {}
        for h in range(HORIZON):
            prefix = prefix_metrics(traces, queries, pools, query_ids, h)
            horizon = h + 1
            horizons[f"H{horizon}"] = prefix
            for support, metrics in prefix["ranking"].items():
                ranking_rows.append(dict(scope=name, horizon=horizon, support=support, **metrics))
        summary["scopes"][name] = {"pools": len(pools), "queries": int(query_ids.sum()), "horizons": horizons}
    training_rows, summary["train_pair_h5_coverage"] = training_pair_coverage()
    (OUTPUT / "horizon_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    for name, rows in (("horizon_rankings", ranking_rows), ("horizon_training_pair_coverage", training_rows)):
        with (OUTPUT / f"{name}.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
