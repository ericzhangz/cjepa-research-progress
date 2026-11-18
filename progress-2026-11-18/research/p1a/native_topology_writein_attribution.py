"""Decompose support and fixed-bank H5 costs without running the model or planner."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/native_topology_paired_20261115"
OUT = RUN / "writein_attribution_20261116"
ARCHIVE = ROOT / "runs/native_topology_condition_20261115/data_holdout.npz"
STAGES = ("first_probe", "reentry_probe", "reversed_probe", "after_reversal_quiet")
POLICIES = ("pre", "post", "omit", "uniform", "true_mode")
MIXES = ("pred_B_pred_D", "true_B_pred_D", "pred_B_true_D", "true_B_true_D")
REGRETS = ("first_block_regret", "suffix_regret", "plan_regret")
PAIR_TYPES = ("uniform_vs_physical", "pre_vs_post")
EDITS = ("pred", "true_D_sign", "true_D_magnitude", "true_D", "true_B", "true_BD")
ATOL = 1e-9


def midpoint_gap(cost):
    cost = cost.astype(np.float64)
    return cost.mean(axis=-2), cost[..., 1, :] - cost[..., 0, :]


def first_block_groups(actions, true_cost):
    _, group = np.unique(actions[:, 0].reshape(len(actions), -1), axis=0, return_inverse=True)
    minima = np.full(group.max() + 1, np.inf)
    np.minimum.at(minima, group, true_cost)
    return group, minima[group]


def regret_parts(physical, group_minima, chosen):
    floor = physical.min(axis=-1, keepdims=True)
    true = np.take_along_axis(physical, chosen, axis=-1)
    first = np.take_along_axis(group_minima, chosen, axis=-1)
    return np.stack((first - floor, true - first, true - floor), axis=-1)


def sign(values):
    return np.sign(np.where(np.abs(values) <= ATOL, 0.0, values)).astype(int)


def pair_components(b_pred, d_pred, b_true, d_true):
    b = np.stack((b_pred, b_pred, b_pred, b_pred, b_true, b_true), axis=-1)
    d = np.stack((d_pred, sign(d_true) * np.abs(d_pred), sign(d_pred) * np.abs(d_true),
                  d_true, d_pred, d_true), axis=-1)
    return b, d


def pair_threshold(b, d):
    """Return p at tie and whether B can be strictly preferred at some p in [0,1]."""
    threshold = np.full_like(b, np.nan)
    nonzero = sign(d) != 0
    threshold[nonzero] = 0.5 - b[nonzero] / d[nonzero]
    reachable = sign(b + 0.5 * np.abs(d)) > 0
    feasible = nonzero & (threshold >= 0) & (threshold <= 1)
    return threshold, feasible, reachable


def verify_examples():
    actions = np.zeros((3, 2, 5, 2))
    actions[1, 0, -1] = 1
    groups, group_min = first_block_groups(actions, np.array([10., 5., 7.]))
    np.testing.assert_array_equal(groups, [0, 1, 0])
    regret = regret_parts(np.array([10., 5., 7.]), group_min, np.array([0, 1, 2]))
    np.testing.assert_array_equal(regret, [[2, 3, 5], [0, 0, 0], [2, 0, 2]])

    b, d = pair_components(np.array([-1., -3.]), np.array([-4., 2.]),
                           np.array([0., -3.]), np.array([6., 8.]))
    np.testing.assert_array_equal(b + 0.5 * d, [[-3, 1, -4, 2, -2, 3],
                                                 [-2, -2, 1, 1, -2, 1]])
    threshold, feasible, reachable = pair_threshold(b, d)
    assert threshold[1, 2] == 0.875 and feasible[1, 2] and reachable[1, 2]
    assert not feasible[1, 0] and not reachable[1, 0]
    threshold, feasible, reachable = pair_threshold(np.array([0., 1.]), np.zeros(2))
    assert np.isnan(threshold).all() and not feasible.any()
    np.testing.assert_array_equal(reachable, [False, True])


def check_roots(banks, archive):
    row, initial, time, stage, pool = banks["choices"].T
    index = row, initial, time
    np.testing.assert_array_equal(np.unique(np.stack((row, initial, stage), axis=1), axis=0).shape,
                                  (256, 3))
    np.testing.assert_array_equal(np.bincount(row), np.full(32, 8))
    np.testing.assert_array_equal(np.bincount(stage), np.full(4, 64))
    np.testing.assert_array_equal(np.unique(pool), np.arange(96))
    np.testing.assert_array_equal(archive["seed"][row], banks["seed"][pool])
    np.testing.assert_array_equal(archive["life_valid"][index], np.ones(256, dtype=bool))
    np.testing.assert_array_equal(archive["life_event"][index], (stage + 1) % 4)
    np.testing.assert_array_equal(archive["life_valid"][row[stage == 3], initial[stage == 3]].sum(-1) - 1,
                                  time[stage == 3])
    np.testing.assert_array_equal(archive["life_xy"][row, initial, time + 1], banks["root_xy"][pool])
    np.testing.assert_array_equal(
        archive["life_z"][row[:, None], initial[:, None], time[:, None] + [-1, 0, 1]],
        banks["root_z"][pool],
    )
    np.testing.assert_array_equal(
        archive["life_action"][row[:, None], initial[:, None], time[:, None] + [-1, 0]],
        banks["root_action"][pool],
    )
    return archive["life_mode"][index]


def supports(saved, choices, mode):
    row, initial, time, stage, _ = choices.T
    idx = row, initial, time
    pre, post, omit = (saved[key][idx].astype(np.float64) for key in ("prior", "actual", "omit_reversed"))
    logs = np.stack((pre, post, omit), axis=1)
    weights = np.exp(logs - logs.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    weights = np.concatenate((weights, np.full((len(mode), 1, 2), 0.5), np.eye(2)[mode, None]), axis=1)
    p = weights[..., 1]
    quiet = stage == 3
    np.testing.assert_array_equal(saved["log_likelihood"][idx][quiet, 0],
                                  saved["log_likelihood"][idx][quiet, 1])
    np.testing.assert_array_equal(p[stage < 2, 1], p[stage < 2, 2])
    omit_recenter_error = float(np.max(np.abs(p[stage == 2, 0] - p[stage == 2, 2])))
    assert omit_recenter_error < 1e-7
    # Equal likelihoods mathematically leave p alone. The archive used float32
    # addition of the likelihood to both entries before recentering.
    quiet_error = float(np.max(np.abs(p[quiet, 0] - p[quiet, 1])))
    assert quiet_error < 2e-6
    return weights, quiet_error, omit_recenter_error


def evaluate(banks, mode, saved):
    choices = banks["choices"]
    _, _, _, stage, pool = choices.T
    queries = np.arange(len(pool))
    weights, quiet_error, omit_error = supports(saved, choices, mode)
    learned = saved["learned_cost"].astype(np.float64)
    physical = banks["physical_cost"].astype(np.float64)
    assert np.isfinite(learned).all() and np.isfinite(physical).all()
    np.testing.assert_array_equal(mode, saved["choice_mode"])
    np.testing.assert_array_equal(stage, saved["choice_stage"])
    b_pred, d_pred = midpoint_gap(learned)
    b_true, d_true = midpoint_gap(physical)
    b = np.stack((b_pred, b_true, b_pred, b_true), axis=1)[pool]
    d = np.stack((d_pred, d_pred, d_true, d_true), axis=1)[pool]
    p = weights[..., 1]
    score = b[:, None] + (p[:, :, None, None] - 0.5) * d[:, None]
    np.testing.assert_allclose(score[:, :, 0], weights @ learned[pool], rtol=0, atol=ATOL)
    np.testing.assert_allclose(score[:, :, 3], weights @ physical[pool], rtol=0, atol=ATOL)
    np.testing.assert_array_equal(score[:, 3, [0, 2]], np.broadcast_to(b_pred[pool, None], score[:, 3, [0, 2]].shape))
    np.testing.assert_array_equal(score[:, 3, [1, 3]], np.broadcast_to(b_true[pool, None], score[:, 3, [1, 3]].shape))
    best = score.argmin(axis=-1)
    np.testing.assert_array_equal(best[:, [1, 3, 4], 0], saved["choice_best"][:, [0, 2, 3]])
    old_omit = np.where(stage == 3, best[:, 2, 0], best[:, 0, 0])
    np.testing.assert_array_equal(old_omit, saved["choice_best"][:, 1])
    selection_ties = ((score - score.min(axis=-1, keepdims=True)) <= ATOL).sum(axis=-1)
    physical_true = physical[pool, mode]
    group, group_min = (np.stack(items) for items in zip(
        *(first_block_groups(banks["action"][i], cost) for i, cost in zip(pool, physical_true))
    ))
    regret = regret_parts(physical_true, group_min, best.reshape(len(pool), -1)).reshape(*best.shape, 3)
    np.testing.assert_allclose(regret[..., 0] + regret[..., 1], regret[..., 2], rtol=0, atol=ATOL)
    np.testing.assert_array_equal(best[:, 4, 3], saved["choice_physical_best"])
    np.testing.assert_array_equal(best[:, 3, 3], saved["choice_physical_uniform"])
    np.testing.assert_array_equal(regret[:, 4, 3], 0)
    np.testing.assert_array_equal(best[stage == 3, 0], best[stage == 3, 1])
    np.testing.assert_array_equal(best[stage == 2, 0], best[stage == 2, 2])
    best_group = group[queries[:, None, None], best]
    np.testing.assert_array_equal(best[:, 3, 0], best[:, 3, 2])
    np.testing.assert_array_equal(best[:, 3, 1], best[:, 3, 3])
    np.testing.assert_allclose(regret[:, [1, 3, 4], 0, 2], saved["choice_regret"][:, [0, 2, 3]],
                               atol=1e-5, rtol=0)
    np.testing.assert_allclose(regret[:, 3, 3, 2], saved["choice_physical_uniform_regret"], atol=1e-5, rtol=0)
    assert np.min(regret) >= 0
    physical_space = regret[:, 3, 3, 0] > ATOL
    np.testing.assert_array_equal(physical_space, saved["choice_physical_decision_space"])
    return dict(weights=weights, best=best, best_group=best_group, selection_ties=selection_ties,
                regret=regret, physical_space=physical_space, group=group, group_min=group_min,
                physical_true=physical_true, quiet_error=quiet_error, omit_error=omit_error, b_pred=b_pred, d_pred=d_pred,
                b_true=b_true, d_true=d_true)


def evaluate_pairs(result, choices, mode):
    best = result["best"]
    pool = choices[:, -1]
    rows = np.arange(len(pool))[:, None]
    a = best[:, [3, 0], 0]
    b = np.stack((best[:, 4, 3], best[:, 1, 0]), axis=-1)
    b_pred, d_pred, b_true, d_true = [
        result[key][pool[:, None], a] - result[key][pool[:, None], b]
        for key in ("b_pred", "d_pred", "b_true", "d_true")
    ]
    edited_b, edited_d = pair_components(b_pred, d_pred, b_true, d_true)
    threshold, feasible, reachable = pair_threshold(edited_b, edited_d)
    p = result["weights"][..., 1]
    margins = edited_b[:, :, None] + (p[:, None, :, None] - 0.5) * edited_d[:, :, None]
    np.testing.assert_allclose(margins[:, :, 1, 0] - margins[:, :, 0, 0],
                               (p[:, 1] - p[:, 0])[:, None] * d_pred, rtol=0, atol=ATOL)
    true_plan = result["physical_true"][rows, a] - result["physical_true"][rows, b]
    true_first = result["group_min"][rows, a] - result["group_min"][rows, b]
    same_first = result["group"][rows, a] == result["group"][rows, b]
    np.testing.assert_allclose(margins[:, :, 4, -1], true_plan, rtol=0, atol=ATOL)
    error_b = b_pred - b_true
    error_d = (p[:, None, :] - 0.5) * (d_pred - d_true)[:, :, None]
    error_support = (p[:, None, :] - mode[:, None, None]) * d_true[:, :, None]
    np.testing.assert_allclose(margins[..., 0] - true_plan[:, :, None],
                               error_b[:, :, None] + error_d + error_support, rtol=0, atol=ATOL)
    assert np.all(sign(margins[:, 0, 3, 0]) <= 0)
    assert np.all(sign(margins[:, 1, 0, 0]) <= 0)
    assert np.all(sign(margins[:, 1, 1, 0]) >= 0)
    assert np.all(true_first[same_first] == 0)
    assert np.all(true_plan[:, 0] >= 0)
    return dict(a=a, b=b, b_pred=b_pred, d_pred=d_pred, b_true=b_true, d_true=d_true, true_plan=true_plan,
                true_first=true_first, same_first=same_first, threshold=threshold, feasible=feasible,
                reachable=reachable, margins=margins, edited_b=edited_b, edited_d=edited_d,
                error_b=error_b, error_d=error_d, error_support=error_support)


def layout_stats(delta, layout, seeds):
    cells = np.asarray([delta[layout == i].mean(axis=0) for i in np.unique(layout)])
    means = cells.mean(axis=0)
    radius = t.ppf(0.975, len(cells) - 1) * cells.std(axis=0, ddof=1) / np.sqrt(len(cells))
    largest = np.abs(cells).argmax(axis=0)
    omitted = (cells.sum(axis=0) - cells[largest, np.arange(cells.shape[1])]) / (len(cells) - 1)
    return {name: dict(mean=float(means[i]), ci95=[float(means[i] - radius[i]), float(means[i] + radius[i])],
                       largest_abs_layout=int(seeds[largest[i]]), mean_without_largest=float(omitted[i]))
            for i, name in enumerate(REGRETS)}


def contrast(result, take, layout, seeds, left, right):
    """Positive differences mean the first listed policy or mix improved physical outcomes."""
    regret, best, group = (result[key][take] for key in ("regret", "best", "best_group"))
    a, b = left, right
    delta = regret[:, b[0], b[1]] - regret[:, a[0], a[1]]
    return dict(plan_changes=int(np.count_nonzero(best[:, a[0], a[1]] != best[:, b[0], b[1]])),
                first_changes=int(np.count_nonzero(group[:, a[0], a[1]] != group[:, b[0], b[1]])),
                improved=dict(zip(REGRETS, np.count_nonzero(sign(delta) > 0, axis=0).tolist())),
                worsened=dict(zip(REGRETS, np.count_nonzero(sign(delta) < 0, axis=0).tolist())),
                stats=layout_stats(delta, layout[take], seeds))


def summarize_pairs(pairs, take):
    summary = {}
    for pair, name in enumerate(PAIR_TYPES):
        subset = take & ((pairs["a"][:, pair] != pairs["b"][:, pair]) if pair else True)
        first = sign(pairs["true_first"][subset, pair])
        plan = sign(pairs["true_plan"][subset, pair])
        pd, td = sign(pairs["d_pred"][subset, pair]), sign(pairs["d_true"][subset, pair])
        margins = sign(pairs["margins"][subset, pair])
        needs_b = first > 0
        edit_summary = {}
        for ei, edit in enumerate(EDITS):
            post_b = margins[:, 1, ei] > 0
            pred_post_b = margins[:, 1, 0] > 0
            edit_summary[edit] = dict(
                post_prefers_B=int((needs_b & post_b).sum()),
                newly_prefers_B=int((needs_b & post_b & ~pred_post_b).sum()),
                lost_B=int((needs_b & ~post_b & pred_post_b).sum()),
                B_reachable=int(pairs["reachable"][subset, pair, ei][needs_b].sum()),
                endpoint_thresholds=int(pairs["feasible"][subset, pair, ei][needs_b].sum()),
            )
        errors = {}
        for pi in (1, 4):
            values = (pairs["error_b"][subset, pair], pairs["error_d"][subset, pair, pi],
                      pairs["error_support"][subset, pair, pi])
            errors[POLICIES[pi]] = {
                name: dict(mean=float(value[needs_b].mean()), rms=float(np.sqrt(np.mean(value[needs_b] ** 2))))
                for name, value in zip(("B", "D", "support"), values) if needs_b.any()
            }
        summary[name] = dict(
            counted_pairs=int(subset.sum()),
            same_candidate=int((take & (pairs["a"][:, pair] == pairs["b"][:, pair])).sum()),
            same_first=int(pairs["same_first"][subset, pair].sum()),
            distinct_first_true_tie=int((~pairs["same_first"][subset, pair] & (first == 0)).sum()),
            plan_true_tie=int((plan == 0).sum()),
            first_B_better=int((first > 0).sum()), first_A_better=int((first < 0).sum()),
            plan_B_better=int((plan > 0).sum()), plan_A_better=int((plan < 0).sum()),
            pred_D_zero=int((pd == 0).sum()), true_D_zero=int((td == 0).sum()),
            D_opposite_where_B_first_better=int((needs_b & (pd * td < 0)).sum()),
            D_same_nonzero_where_B_first_better=int((needs_b & (pd * td > 0)).sum()),
            D_pred_zero_where_B_first_better=int((needs_b & (pd == 0)).sum()),
            D_true_zero_where_B_first_better=int((needs_b & (td == 0)).sum()),
            pre_strict_A_to_post_strict_B=int(((margins[:, 0, 0] < 0) & (margins[:, 1, 0] > 0)).sum()),
            strict_B_at_true_mode_where_B_first_better=int((needs_b & (margins[:, 4, 0] > 0)).sum()),
            edits_where_B_first_better=edit_summary,
            margin_error_where_B_first_better=errors,
        )
    return summary


def summarize(result, pairs, choices, mode, seeds):
    row, _, _, stage, _ = choices.T
    probability = result["weights"][np.arange(len(mode))[:, None], np.arange(len(POLICIES)), mode[:, None]]
    summary = {}
    for name, take in [("all", np.ones(len(mode), dtype=bool)), *[(name, stage == i) for i, name in enumerate(STAGES)]]:
        by_mix = {}
        for mi, mix in enumerate(MIXES):
            performance = {
                policy: dict(zip(REGRETS, result["regret"][take, pi, mi].mean(axis=0).tolist()))
                for pi, policy in enumerate(POLICIES)
            }
            comparisons = {
                f"{POLICIES[pi]}_vs_{POLICIES[other]}": contrast(
                    result, take, row, seeds, (pi, mi), (other, mi)
                )
                for pi, other in ((1, 0), (1, 2), (1, 3), (4, 3), (4, 1))
            }
            improvement = result["regret"][take, 3, mi] - result["regret"][take, 1, mi]
            learned = result["regret"][take, 3, 0] - result["regret"][take, 1, 0]
            by_mix[mix] = dict(performance=performance, comparisons=comparisons,
                              post_uniform_increment_vs_pred=layout_stats(improvement - learned, row[take], seeds),
                              post_absolute_gain_vs_pred=contrast(result, take, row, seeds, (1, mi), (1, 0)))
        summary[name] = dict(
            queries=int(take.sum()), layouts=int(len(np.unique(row[take]))),
            physical_decision_space_queries=int(result["physical_space"][take].sum()),
            physical_decision_space_post_vs_uniform_first={
                mix: float((result["regret"][take & result["physical_space"], 3, mi, 0] -
                            result["regret"][take & result["physical_space"], 1, mi, 0]).mean())
                for mi, mix in enumerate(MIXES)
            },
            support={
                policy: dict(true_mean=float(probability[take, pi].mean()),
                             true_majority=int((probability[take, pi] > 0.5).sum()),
                             ties=int((probability[take, pi] == 0.5).sum()))
                for pi, policy in enumerate(POLICIES)
            },
            mixes=by_mix, pairs=summarize_pairs(pairs, take),
        )
    return summary


def csv_output(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, first.keys())
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def query_keys(arm, banks, mode, result, q):
    row, initial, time, stage, pool = banks["choices"][q]
    return dict(arm=arm, query=q, layout=row, seed=banks["seed"][pool], initial_mode=initial,
                time=time, stage=STAGES[stage], pool=pool, true_mode=mode[q],
                physical_decision_space=result["physical_space"][q])


def choice_rows(results, banks, mode):
    for arm, result in results.items():
        for q in range(len(mode)):
            keys = query_keys(arm, banks, mode, result, q)
            for pi, policy in enumerate(POLICIES):
                for mi, mix in enumerate(MIXES):
                    candidate = result["best"][q, pi, mi]
                    first = banks["action"][keys["pool"], candidate, 0].tolist()
                    yield keys | dict(policy=policy, p1=result["weights"][q, pi, 1], mix=mix,
                                      candidate=candidate, first_block=first,
                                      first_group=result["best_group"][q, pi, mi],
                                      predicted_best_ties=result["selection_ties"][q, pi, mi],
                                      true_h5_selected=result["physical_true"][q, candidate],
                                      true_h5_best_suffix=result["group_min"][q, candidate]) | dict(
                                          zip(REGRETS, result["regret"][q, pi, mi]))


def pair_rows(results, pair_results, banks, mode):
    for arm, pairs in pair_results.items():
        for q in range(len(mode)):
            keys = query_keys(arm, banks, mode, results[arm], q)
            for i, name in enumerate(PAIR_TYPES):
                a, b = pairs["a"][q, i], pairs["b"][q, i]
                yield keys | dict(
                    pair=name, candidate_A=a, candidate_B=b, same_candidate=(a == b),
                    same_first=pairs["same_first"][q, i],
                    pred_delta_B=pairs["b_pred"][q, i], pred_delta_D=pairs["d_pred"][q, i],
                    true_delta_B=pairs["b_true"][q, i], true_delta_D=pairs["d_true"][q, i],
                    true_plan_A_minus_B=pairs["true_plan"][q, i],
                    true_first_A_minus_B=pairs["true_first"][q, i],
                    true_plan_preference=sign(pairs["true_plan"][q, i]).item(),
                    true_first_preference=sign(pairs["true_first"][q, i]).item(),
                    margin_error_B=pairs["error_b"][q, i],
                ) | {
                    f"p1_{policy}": results[arm]["weights"][q, pi, 1] for pi, policy in enumerate(POLICIES)
                } | {
                    f"margin_error_{component}_{POLICIES[pi]}": pairs[f"error_{component.lower()}"][q, i, pi]
                    for component in ("D", "support")
                    for pi in (1, 4)
                } | {
                    f"{edit}_{field}": pairs[key][q, i, ei]
                    for ei, edit in enumerate(EDITS)
                    for field, key in (("delta_B", "edited_b"), ("delta_D", "edited_d"),
                                       ("p1_threshold", "threshold"), ("threshold_feasible", "feasible"),
                                       ("B_strict_reachable", "reachable"))
                } | {
                    f"{edit}_delta_score_{policy}": pairs["margins"][q, i, pi, ei]
                    for ei, edit in enumerate(EDITS)
                    for pi, policy in enumerate(POLICIES)
                }


def layout_rows(results, choices, seeds):
    row, _, _, stage, _ = choices.T
    for arm, result in results.items():
        for i, seed in enumerate(seeds):
            for name, section in [("all", row == i), *[(name, (row == i) & (stage == j)) for j, name in enumerate(STAGES)]]:
                for mi, mix in enumerate(MIXES):
                    for pi, policy in enumerate(POLICIES):
                        yield dict(arm=arm, layout=i, seed=seed, stage=name, queries=int(section.sum()),
                                   mix=mix, policy=policy) | dict(
                                       zip(REGRETS, result["regret"][section, pi, mi].mean(axis=0)))


def load_npz(path):
    with np.load(path, allow_pickle=False) as contents:
        return dict(contents)


def main():
    verify_examples()
    banks = load_npz(RUN / "writein_fixed_h5_banks.npz")
    archive = load_npz(ARCHIVE)
    mode = check_roots(banks, archive)
    seeds = archive["seed"]
    results, pairs, summary = {}, {}, {}
    for arm in ("paired", "control"):
        saved = load_npz(RUN / f"writein_{arm}.npz")
        results[arm] = evaluate(banks, mode, saved)
        pairs[arm] = evaluate_pairs(results[arm], banks["choices"], mode)
        summary[arm] = summarize(results[arm], pairs[arm], banks["choices"], mode, seeds)
        summary[arm]["probability_roundoff_max"] = {
            "quiet_pre_vs_post": results[arm]["quiet_error"],
            "reversed_probe_pre_vs_omit": results[arm]["omit_error"],
        }
    metadata = dict(
        policies=POLICIES, mixes=MIXES, regrets=REGRETS, stages=STAGES, pair_types=PAIR_TYPES,
        pair_edits=EDITS, atol=ATOL,
        omit="Omit the reversed probe likelihood only, keeping all other factual writes.",
        selection="Canonical lowest stored candidate index at a minimum over the full fixed bank.",
        difference="Contrast = right regret minus left regret. Pair margin = plan A minus plan B.",
        ci="Two-sided paired Student t interval over 32 layout means, 31 degrees of freedom.",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attributed_summary.json").write_text(
        json.dumps(dict(metadata=metadata, results=summary), indent=2, allow_nan=False), encoding="utf-8")
    csv_output(OUT / "attributed_choices.csv", choice_rows(results, banks, mode))
    csv_output(OUT / "attributed_pairs.csv", pair_rows(results, pairs, banks, mode))
    csv_output(OUT / "attributed_layouts.csv", layout_rows(results, banks["choices"], seeds))
    np.savez_compressed(OUT / "attributed_arrays.npz", choices=banks["choices"], mode=mode, seeds=seeds,
                        **{f"{arm}_{key}": vals for arm, result in results.items() for key, vals in result.items()},
                        **{f"{arm}_pair_{key}": vals for arm, result in pairs.items() for key, vals in result.items()})
    print(f"Verified and saved {len(mode)} queries per checkpoint over {len(seeds)} layouts")


if __name__ == "__main__":
    main()
