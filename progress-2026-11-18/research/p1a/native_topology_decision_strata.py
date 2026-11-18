"""Re-score archived lifecycle decisions within fixed H5 false-target strata."""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t


RUN = Path(__file__).resolve().parents[2] / "runs/native_topology_paired_20261115/writein_attribution_20261116"
OUT = RUN / "decision_root_followup"
EPS = 1e-9


def interval(delta, layout):
    means = np.array([delta[layout == key].mean() for key in np.unique(layout)])
    center = means.mean()
    radius = t.ppf(.975, len(means) - 1) * means.std(ddof=1) / np.sqrt(len(means)) if len(means) > 1 else None
    return dict(query_mean=float(delta.mean()), layout_mean=float(center),
                ci95=None if radius is None else [float(center - radius), float(center + radius)],
                higher=int((delta > EPS).sum()), lower=int((delta < -EPS).sum()))


def target_room(points):
    return (points >= 112).all(axis=-1)


def retention_check(q, nw, post, uniform, oracle):
    """Disable switch prior only on the archived post-probe leg, with every actual arrival unchanged."""
    archive_path = RUN.parent / "writein_paired.npz"
    with np.load(archive_path) as f:
        ll, actual, cost = [f[key].astype(np.float64) for key in ("log_likelihood", "actual", "learned_cost")]
    row, initial, time, stage, pool = q["choices"].T
    take = np.flatnonzero(nw)
    prev = np.array([np.flatnonzero((row == row[n]) & (initial == initial[n]) & (stage == 2)).item() for n in take])
    assert np.all(time[take] - time[prev] == 10)
    m = q["mode"][take]
    np.testing.assert_array_equal(m, q["mode"][prev])
    trajectory = ll[row[take, None], initial[take, None], time[prev, None] + np.arange(1, 11)]
    obs_delta = trajectory[..., 1] - trajectory[..., 0]
    belief = actual[row[prev], initial[prev], time[prev]].copy()
    zero_switch = belief + (trajectory - trajectory[..., :1]).sum(axis=1)
    p = np.exp(zero_switch[:, 1] - np.logaddexp(zero_switch[:, 0], zero_switch[:, 1]))
    for arrived in trajectory.transpose(1, 0, 2):
        belief = np.logaddexp(belief + np.log(.99), belief[:, ::-1] + np.log(.01))
        belief = belief + (arrived - arrived[:, :1])
        belief -= np.logaddexp(belief[:, :1], belief[:, 1:])
    p_native = np.exp(belief[:, 1])
    np.testing.assert_allclose(p_native, q["weights"][take, post, 1], atol=5e-6, rtol=0)
    candidate = cost[pool[take]]
    top = q["best"][0, take, post, 0, -1]
    np.testing.assert_array_equal(
        ((1 - p_native[:, None]) * candidate[:, 0] + p_native[:, None] * candidate[:, 1]).argmin(axis=1), top
    )
    scores = (1 - p[:, None]) * candidate[:, 0] + p[:, None] * candidate[:, 1]
    pick = scores.argmin(axis=1)
    first_regret = q["group_min"][take, pick, -1] - q["group_min"][take, :, -1].min(axis=1)
    args = np.arange(len(take))
    factual_p = q["weights"][take, post, m]
    no_switch_p = np.where(m == 1, p, 1 - p)
    return dict(queries=len(take), after_reversal_blocks=10, baseline_probability_max_abs_error=float(
                    np.max(np.abs(p_native - q["weights"][take, post, 1]))),
                exact_equal_ll_blocks=int((obs_delta == 0).sum()),
                all_equal_ll_histories=int((obs_delta == 0).all(axis=1).sum()),
                ll_favors_wrong=int(((2 * m[:, None] - 1) * obs_delta < 0).sum()),
                factual_true_p_range=[float(factual_p.min()), float(factual_p.max())],
                no_switch_true_p_range=[float(no_switch_p.min()), float(no_switch_p.max())],
                pick_changes=int((pick != top).sum()),
                same_oracle_pick=int((pick == q["best"][0, take, oracle, 0, -1]).sum()),
                mean_first_regret=float(first_regret.mean()),
                relative_to_uniform=interval(q["regret"][0, take, uniform, 0, -1, 0] - first_regret, row[take]),
                relative_to_actual=interval(q["regret"][0, take, post, 0, -1, 0] - first_regret, row[take]),
                per_query=[dict(query=int(take[i]), actual_p_true=float(factual_p[i]), no_switch_p_true=float(no_switch_p[i]),
                                ll_gap_mode1=obs_delta[i].tolist(), no_switch_pick=int(pick[i]),
                                no_switch_first_regret=float(first_regret[i])) for i in args])


def main():
    src = RUN / "gate_followup"
    meta = json.loads((src / "gate_summary.json").read_text(encoding="utf-8"))["metadata"]
    with np.load(src / "gate_queries.npz") as f:
        q = dict(f)
    with np.load(RUN / "horizon_traces.npz") as f:
        model, truth = f["learned_xy"], f["physical_xy"]
    layout, _, _, stage, pool = q["choices"].T
    mode = q["mode"]
    idx = np.arange(len(mode))
    nw = q["position"] == meta["positions"].index("northwest")
    assert nw.sum() == 64 and len(np.unique(layout[nw])) == 32
    pre, post, omit, uniform, oracle = [meta["policies"].index(key) for key in ("pre", "post", "omit", "uniform", "true_mode")]
    native_top = q["best"][0, :, oracle, 0, -1]
    model_target = target_room(model[pool, :, native_top, -1])
    truth_target = target_room(truth[pool, :, native_top, -1])
    false_by_mode = model_target & ~truth_target
    b = false_by_mode[idx, mode]
    np.testing.assert_array_equal(b, q["label_native_false_target_h5"])
    static = q["label_static_occupancy"] | q["label_static_skip"]
    common_false = false_by_mode.all(axis=1)
    mode_value = q["physical_mode_value"][:, -1]
    structured = {
        "northwest/all": nw, "northwest/A_no_false_target": nw & ~b,
        "northwest/B_false_target": nw & b,
        "northwest/B_both_modes_false_same_plan": nw & common_false,
        "northwest/B_only_actual_mode_false": nw & b & ~common_false,
        "northwest/static_wall_label": nw & static, "northwest/no_static_wall_label": nw & ~static,
        "northwest/A_no_false_target_and_no_static_wall_label": nw & ~b & ~static,
        "northwest/B_false_target_and_static_wall_label": nw & b & static,
        "northwest/physics_mode_value": nw & mode_value,
        "northwest/A_and_physics_mode_value": nw & ~b & mode_value,
        "northwest/B_and_physics_mode_value": nw & b & mode_value,
    }
    for position, label in enumerate(meta["positions"]):
        if label == "northwest":
            continue
        at = q["position"] == position
        for name, left in (("all", at), ("reversal", at & (stage == meta["stages"].index("reversed_probe")))):
            structured.update({f"{label}/{name}": left, f"{label}/{name}/A_no_false_target": left & ~b,
                               f"{label}/{name}/B_false_target": left & b,
                               f"{label}/{name}/A_no_false_target_and_no_static_wall_label": left & ~b & ~static})

    regret = q["regret"][..., -1, :]  # gate, query, support, B/D table, regret components
    pair_d = q["pair_delta_d"][:, :, -1]
    pair_dx = q["pair_delta_d_xy"][:, :, -1]
    pair_b = q["pair_delta_b"][:, :, -1]
    strict = ~q["pair_same_first"] & (q["pair_physical_first_gap"][:, -1] > EPS)
    cost_effect = np.abs(pair_d[2]) > EPS
    xy_effect = np.linalg.norm(pair_dx[2], axis=-1) > EPS
    summaries = {}
    for label, take in structured.items():
        if not take.any():
            summaries[label] = dict(queries=0)
            continue
        here = dict(queries=int(take.sum()), layouts=int(np.unique(layout[take]).size),
                    static=int(static[take].sum()), both_mode_false=int(common_false[take].sum()),
                    physics_mode_value=int(mode_value[take].sum()), arms={})
        for arm, gate in enumerate(meta["arms"]):
            x = regret[arm, take]
            use = take & strict & cost_effect
            use_xy = take & strict & xy_effect
            here["arms"][gate] = dict(
                regret=x.mean(axis=0).tolist(),
                first_gain={key: interval(x[:, uniform, 0, 0] - x[:, policy, 0, 0], layout[take])
                            for key, policy in (("post", post), ("true_mode", oracle))},
                plan_gain={key: interval(x[:, uniform, 0, 2] - x[:, policy, 0, 2], layout[take])
                           for key, policy in (("post", post), ("true_mode", oracle))},
                physics_first_gain=float(x[:, uniform, 3, 0].mean()),
                fixed_pair_count=int((take & strict).sum()),
                fixed_pair_nonzero_d=int(use.sum()),
                fixed_pair_d_same_sign=int((pair_d[arm, use] * pair_d[2, use] > EPS).sum()),
                fixed_pair_d_opposite=int((pair_d[arm, use] * pair_d[2, use] < -EPS).sum()),
                fixed_pair_d_amp_median=None if not use.any() else float(np.median(np.abs(pair_d[arm, use] / pair_d[2, use]))),
                fixed_pair_xy_nonzero=int(use_xy.sum()),
                fixed_pair_xy_aligned=int((np.sum(pair_dx[arm, use_xy] * pair_dx[2, use_xy], axis=-1) > EPS).sum()),
                fixed_pair_common_gap_median_error=None if not (take & strict).any() else float(
                    np.median(np.abs(pair_b[arm, take & strict] - pair_b[2, take & strict]))),
            )
        summaries[label] = here

    rows = []
    for n in idx:
        row = dict(query=int(n), layout=int(layout[n]), pool=int(pool[n]), mode=int(mode[n]),
                   position=meta["positions"][q["position"][n]], stage=meta["stages"][stage[n]],
                   native_true_pick=int(native_top[n]), A_or_B="B" if b[n] else "A", both_mode_false=bool(common_false[n]),
                   static_occupancy=bool(q["label_static_occupancy"][n]), static_skip=bool(q["label_static_skip"][n]),
                   physical_mode_value=bool(mode_value[n]), p1_post=float(q["weights"][n, post, 1]),
                   pair_a=int(q["pair_candidates"][n, 0]), pair_b=int(q["pair_candidates"][n, 1]),
                   pair_strict_first_block=bool(strict[n]))
        for arm, gate in enumerate(meta["arms"]):
            for policy, name in enumerate(meta["policies"]):
                row[f"{gate}_{name}_pick"] = int(q["best"][arm, n, policy, 0, -1])
                row[f"{gate}_{name}_first_regret"] = float(regret[arm, n, policy, 0, 0])
                row[f"{gate}_{name}_plan_regret"] = float(regret[arm, n, policy, 0, 2])
        for source, key in enumerate(meta["pair_arms"]):
            row[f"{key}_pair_delta_B"] = float(pair_b[source, n])
            row[f"{key}_pair_delta_D"] = float(pair_d[source, n])
        rows.append(row)
    OUT.mkdir(exist_ok=True)
    with (OUT / "lifecycle_strata.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "lifecycle_strata.json").write_text(json.dumps({
        "definition": "A/B fixed by original gate, actual-mode H5 Top-1; same plan tested in both modes as secondary label",
        "gain": "uniform regret minus post or true-mode regret; positive is better",
        "arrays": {key: meta[key] for key in ("arms", "regrets", "policies", "mixes")},
        "strata": summaries,
        "archived_retention_check": retention_check(q, nw, post, uniform, oracle),
    }, indent=2), encoding="utf-8")
    print({key: val["queries"] for key, val in summaries.items()})


if __name__ == "__main__":
    main()
