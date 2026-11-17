"""Compare trained checkpoints on the unchanged archived FourRoom candidate banks."""

import argparse
import csv
import json

import numpy as np
import torch

from .native_topology_condition import forecast, load_readout, load_trained
from .native_topology_paired_data import clipping
from .native_topology_prefix_eval import wall_depth
from .native_topology_writein_attribution import (
    ARCHIVE, OUT, POLICIES, REGRETS, RUN, check_roots, csv_output,
    first_block_groups, load_npz, regret_parts, sign, supports,
)
from .native_topology_writein_gate import prefix_decisions
from .native_topology_writein_replay import REFERENCE, replay_evidence
from .physical_topology_gate import CENTER, SPEED, PhysicalRollout, VisibleMapCost, make_scene, room

OUTPUT = RUN.with_name("native_topology_symmetric_20261116")
OLD = OUT / "decision_root_followup"
DOORS = RUN.with_name("native_topology_prefix_20261115") / "data_eval_bank.npz"
ARMS = ("baseline", "arm1", "arm2")
EXAMPLES = (42, 26, 11)
INFERENCE_CONTROL = "arm1_inference_r1"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def count(mask):
    return int(np.count_nonzero(mask))


def signs(pred, truth, take):
    pred, truth = pred[take], truth[take]
    p, t = sign(pred), sign(truth)
    nz = t != 0
    return dict(n=len(pred), true_nonzero=count(nz), aligned=count(p * t > 0), opposite=count(p * t < 0),
                zero_prediction=count(nz & (p == 0)),
                median_amplitude_ratio=float(np.median(abs(pred[nz] / truth[nz]))) if nz.any() else None)


def diff_stats(value):
    s = sign(value)
    return dict(mean=float(value.mean()), harmed=count(s > 0), helped=count(s < 0), tied=count(s == 0))


@torch.inference_mode()
def predict(modules, decode, device, bank, *, lifecycle=False, pairs=None):
    """Replay former batch splits and decoding; never propose or execute a new plan."""
    if lifecycle:
        roots = list(range(len(bank["seed"])))
        intervals = [(i, i + 128) for i in range(0, 384, 128)]
        leading = (len(roots), 384)
    else:
        roots = [(i, j) for i in range(len(bank["seed"])) for j in ((0, 1) if pairs is not None else (2, 3))]
        intervals = [(i, i + 96) for i in range(4, 388, 96)]
        leading = (len(roots), 2 if pairs is not None else 384)
    xy = np.empty((*leading, 2, 5, 2), np.float32)
    exposure = np.empty(xy.shape[:-1], np.float32)
    root_decoded = np.empty((len(roots), 2), np.float32)
    for k, key in enumerate(roots):
        if pairs is not None and pairs[k] is None:
            xy[k], exposure[k], root_decoded[k] = 0, 0, 0
            continue
        h, past, plans = (torch.as_tensor(bank[n][key], device=device)[None] for n in ("root_z", "root_action", "action"))
        root_decoded[k] = decode(h[:, -1]).cpu().numpy()[0]
        chunks = [(0, 2)] if pairs is not None else intervals
        chosen = plans[:, pairs[k]] if pairs is not None else plans
        for lo, hi in chunks:
            trace = {}
            z = forecast(*modules, h, past, chosen[:, lo:hi], compact=True, trace=trace)[0]
            if lifecycle:
                decoded = torch.stack([decode(z[..., j, :]) for j in range(5)], dim=-2)
            else:
                decoded = decode(z)
                decoded[..., -1, :] = decode(z[..., -1, :])
            dst = slice(lo - 4, hi - 4) if not lifecycle and pairs is None else slice(lo, hi)
            xy[k, dst] = decoded.cpu().numpy()
            exposure[k, dst] = trace["exposure"][0].cpu().numpy()
        if (k + 1) % (12 if lifecycle else 8) == 0:
            print(f"  {'lifecycle' if lifecycle else 'doors' if pairs is not None else 'CEM'} {k + 1}/{len(roots)} roots", flush=True)
    for array in (xy, exposure, root_decoded):
        assert np.isfinite(array).all()
    return xy, exposure, root_decoded


def nav_data(bank):
    truth = bank["future_xy"][:, 2:, 4:, :, :, -1, :].reshape(-1, 384, 2, 5, 2)
    action = bank["action"][:, 2:, 4:].reshape(-1, 384, 5, 5, 2)
    scenes = [make_scene(int(seed)) for seed in bank["seed"]]
    specs = [(scene, target) for scene in scenes for target in (scene.southeast, scene.northwest)]
    files = load_npz(RUN / "pred_paired_scored_paired.npz")
    prior = files["xy"][:, :, 4:].reshape(truth.shape)
    with (OLD / "newcem_rows.csv").open(encoding="utf-8", newline="") as handle:
        frozen = list(csv.DictReader(handle))
    return truth, action, specs, prior, frozen, files["cost"][:, :, 4:, :, -1].reshape(-1, 384, 2)


def nav_geometry(bank, nav, arrays):
    seeds, start = bank["seed"], bank["root_xy"][:, 2:, -1]
    actions, raw = bank["action"][:, 2:, 4:], bank["future_xy"][:, 2:, 4:]
    y = raw[..., -1, :]
    hits = np.array([[clipping(start[i, j], actions[i, j], raw[i, j]) for j in (0, 1)] for i in range(len(seeds))])
    identical = (raw[..., 0, :, :, :] == raw[..., 1, :, :, :]).all(axis=(-1, -2, -3))
    blocked = identical & hits[..., 0, -actions.shape[-2]:].any(-1)
    root = start[:, :, None, None, None, :]
    travel = y - root
    length = np.linalg.norm(travel, axis=-1)
    legal = (room(y[..., -1, :]) != room(root[..., 0, :])) & ~hits.any(-1)
    direction = np.divide(travel, length[..., None], out=np.zeros_like(travel), where=length[..., None] > 0)
    intent = actions.sum(-2)
    intent /= np.maximum(np.linalg.norm(intent, axis=-1, keepdims=True), 1e-12)
    before = np.concatenate((np.broadcast_to(root, y[..., :1, :].shape), y[..., :-1, :]), axis=-2)

    def forward(x):
        return ((x - before) * intent[:, :, :, None]).sum(-1)[..., -1][blocked]

    arms = {"baseline": nav[3].reshape(y.shape), **{k: v["nav_xy"].reshape(y.shape) for k, v in arrays.items()}}
    geometry = {}
    for name, x in arms.items():
        depth = np.array([wall_depth(PhysicalRollout(make_scene(int(s))), x[i], False) for i, s in enumerate(seeds)])
        error_sq = np.square(x - y).sum(-1)
        projection = ((x - root) * direction).sum(-1)
        geometry[name] = {
            "fixed_bank_H1_H5_position_RMSE_px": np.sqrt(error_sq.mean(axis=(0, 1, 2, 3))).tolist(),
            "fixed_bank_H1_H5_room_error": (room(x) != room(y)).mean(axis=(0, 1, 2, 3)).tolist(),
            "visible_wall_endpoint_any_block_rate": float((depth > 0).any(-1).mean()),
            "both_modes_identical_final_blocked_xy_offset_RMSE_px": float(np.sqrt(error_sq[..., -1][blocked].mean())),
            "both_modes_identical_final_blocked_forward_projection_px": float(forward(x).mean()),
            "clean_room_change_H5_legal_projected_displacement_px": float(projection[..., -1][legal].mean()),
        }
    return {"physical_clean_room_change_branches": count(legal), "both_modes_identical_final_blocked_plans": count(blocked),
            "physical_H5_legal_projected_displacement_px": float(length[..., -1][legal].mean()),
            "physical_common_final_blocked_forward_projection_px": float(forward(y).mean()), "arms": geometry}


def nav_report(data, arrays, out):
    truth, action, specs, prior, frozen, prior_cost = data
    observed = {"baseline": prior, **{k: v["nav_xy"] for k, v in arrays.items()}}
    all_rows = {}
    for name, xy in observed.items():
        rows = []
        for i, (scene, goal) in enumerate(specs):
            score = VisibleMapCost(scene, goal)
            p, t = (score(values[i, ..., -1, :]).astype(np.float64) for values in (xy, truth))
            if name == "baseline":
                np.testing.assert_array_equal(p, prior_cost[i])
            uniform = int(p.mean(axis=1).argmin())
            for mode in (0, 1):
                fs = frozen[2 * i + mode]
                assert (int(fs["seed"]), int(fs["root_index"]), int(fs["true_mode"])) == (scene.seed, i % 2 + 2, mode)
                gt = t[:, mode]
                group, group_min = first_block_groups(action[i], gt)
                native, best = int(p[:, mode].argmin()), int(gt.argmin())
                baseline_u = int(fs["learned_uniform_slot"]) - 4
                if name == "baseline":
                    assert uniform == baseline_u and native + 4 == int(fs["learned_true_slot"])
                assert best + 4 == int(fs["physical_true_slot"])
                own_false = bool((room(xy[i, native, mode, -1]) == room(goal)) & (room(truth[i, native, mode, -1]) != room(goal)))
                if name == "baseline":
                    assert own_false == (fs["stratum"] == "B")
                row = dict(arm=name, seed=scene.seed, root=fs["root"], mode=mode, baseline_stratum=fs["stratum"],
                           own_false_target=own_false, predicted_goal=bool(room(xy[i, native, mode, -1]) == room(goal)),
                           actual_goal=bool(room(truth[i, native, mode, -1]) == room(goal)),
                           fixed_uniform_candidate=baseline_u + 4, physical_best_candidate=best + 4,
                           uniform_candidate=uniform + 4, true_candidate=native + 4,
                           selected_H5_midpoint_error=float(np.linalg.norm(xy[i, native, :, -1].mean(axis=0) - truth[i, native, :, -1].mean(axis=0))),
                           **{f"{pol}_{label}": float(v) for pol, ix in (("uniform", uniform), ("true", native))
                              for label, v in zip(REGRETS, regret_parts(gt, group_min, np.array([ix]))[0])})
                for label, a in (("fixed", baseline_u), ("reselected", uniform), ("true_selected", native)):
                    d = p[a] - p[best]
                    dt = t[a] - t[best]
                    row.update({f"{label}_first_gap": float(group_min[a] - gt[best]),
                                f"{label}_pred_delta_B": float(d.mean()), f"{label}_phys_delta_B": float(dt.mean()),
                                f"{label}_pred_delta_D": float(d[1] - d[0]), f"{label}_phys_delta_D": float(dt[1] - dt[0])})
                    if label == "fixed" and name == "baseline":
                        for ours, archived in ((group_min[a] - gt[best], "physical_first_gap"), (d[1] - d[0], "delta_D_learned"),
                                               (dt[1] - dt[0], "delta_D_actual")):
                            assert ours == float(fs["uniform_vs_physical_true_" + archived])
                rows.append(row)
        assert len(rows) == 64
        all_rows[name] = rows
    result = {}
    for name, rows in all_rows.items():
        result[name] = {}
        a = lambda key: np.array([row[key] for row in rows])
        for scope, take in (("all", np.ones(len(rows), bool)), ("A", a("baseline_stratum") == "A"), ("B", a("baseline_stratum") == "B")):
            cell = dict(n=count(take), own_false_target=count(take & a("own_false_target")),
                        actual_goal=count(take & a("actual_goal")), predicted_goal=count(take & a("predicted_goal")),
                        mean_selected_H5_midpoint_error=float(a("selected_H5_midpoint_error")[take].mean()),
                        regrets={p: {r: float(a(p + "_" + r)[take].mean()) for r in REGRETS} for p in ("uniform", "true")},
                        true_minus_uniform={r: diff_stats((a("true_" + r) - a("uniform_" + r))[take]) for r in REGRETS})
            for label in ("fixed", "reselected", "true_selected"):
                eligible = take & (a(label + "_first_gap") > 0)
                cell[label] = {"delta_" + k: signs(a(label + "_pred_delta_" + k), a(label + "_phys_delta_" + k), eligible)
                               for k in ("B", "D")}
            result[name][scope] = cell
    assert result["baseline"]["all"]["own_false_target"] == 32
    assert result["baseline"]["A"]["fixed"]["delta_D"]["opposite"] == 14
    assert result["baseline"]["A"]["fixed"]["delta_D"]["true_nonzero"] == 31
    csv_output(out / "newcem_queries.csv", [row for rows in all_rows.values() for row in rows])
    return result


def new_supports(evidence, choices, mode):
    row, initial, time, _, _ = choices.T
    logs = np.stack([evidence[k][row, initial, time].astype(np.float64) for k in ("prior", "actual", "omit_reversed")], axis=1)
    weights = np.exp(logs - logs.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return np.concatenate((weights, np.full((len(mode), 1, 2), .5), np.eye(2)[mode, None]), axis=1)


def lifecycle_data(bank):
    archive = load_npz(ARCHIVE)
    saved = load_npz(RUN / "writein_paired.npz")
    reference = load_npz(OUT / "gate_followup" / "gate_queries.npz")
    trace = load_npz(OUT / "horizon_traces.npz")
    gate = load_npz(OUT / "gate_followup" / "gate_forecasts.npz")
    mode = check_roots(bank, archive)
    weights = supports(saved, bank["choices"], mode)[0]
    np.testing.assert_array_equal(bank["choices"], reference["choices"])
    np.testing.assert_array_equal(weights, reference["weights"])
    np.testing.assert_array_equal(weights, new_supports(saved, bank["choices"], mode))
    np.testing.assert_array_equal(trace["physical_cost"][..., -1], bank["physical_cost"])
    np.testing.assert_array_equal(trace["learned_cost"][..., -1], saved["learned_cost"])
    return archive, saved, reference, trace, gate, mode, weights


def lifecycle_report(bank, data, arrays, out):
    _, _, ref, trace, gate, mode, weights = data
    physical_xy, physical_cost = trace["physical_xy"], trace["physical_cost"]
    row, initial, time, stage, pool = bank["choices"].T
    n = np.arange(len(pool))
    post, uniform, true = [POLICIES.index(k) for k in ("post", "uniform", "true_mode")]
    rev = stage == 2
    old_picks = ref["best"][0, :, [post, uniform], 0, -1].T
    assert old_picks.shape == (len(n), 2)
    truth = physical_cost[pool, mode]
    qp = pool[:, None]
    physical_pair = physical_cost[qp, :, old_picks].astype(np.float64)
    physical_diff = physical_pair[:, 0] - physical_pair[:, 1]
    delta_D_truth = physical_diff[:, 1] - physical_diff[:, 0]
    harmed = rev & (sign(physical_diff[n, mode, -1]) > 0)
    assert rev.sum() == 64 and harmed.sum() == 13
    baseline_weights = dict(baseline=weights, **{name: a["weights"] for name, a in arrays.items()})
    baseline_gate = gate["native_exposure"]
    models = {"baseline": (trace["learned_xy"], trace["learned_cost"], baseline_gate)}
    for name, a in arrays.items():
        pred_xy = a["life_xy"].transpose(0, 2, 1, 3, 4)
        pred_cost = np.stack([VisibleMapCost(s, s.southeast)(x) for x, s in zip(pred_xy, map(make_scene, map(int, bank["seed"])))])
        models[name] = (pred_xy, pred_cost, a["life_exposure"].transpose(0, 2, 1, 3))
    rows, pairs, result, examples = [], [], {}, {}
    for name, (xy, cost, exposure) in models.items():
        decisions = {label: prefix_decisions(bank, mode, w, (xy, xy, physical_xy), (cost, cost), physical_cost)
                     for label, w in (("fixed_support", weights), ("own_support", baseline_weights[name]))}
        if name == "baseline":
            np.testing.assert_array_equal(decisions["fixed_support"]["best"][0], ref["best"][0])
            np.testing.assert_array_equal(decisions["fixed_support"]["regret"][0], ref["regret"][0])
        own_pair = cost[qp, :, old_picks].astype(np.float64)
        diff = own_pair[:, 0] - own_pair[:, 1]
        db, dd = diff.mean(axis=1), diff[:, 1] - diff[:, 0]
        rho = weights[:, post, 1] - .5
        margin = db + rho[:, None] * dd
        result[name] = dict(fixed_reversal_pairs_before_reselection=dict(
            all=signs(dd[:, -1], delta_D_truth[:, -1], rev), harmed_baseline=signs(dd[:, -1], delta_D_truth[:, -1], harmed),
            predicts_original_uniform_better=count(harmed & (sign(margin[:, -1]) > 0)),
            mean_abs_delta_B_error_on_original_harms=float(np.abs(db[harmed, -1] - physical_diff[harmed, :, -1].mean(axis=1)).mean())))
        if name == "baseline":
            assert result[name]["fixed_reversal_pairs_before_reselection"]["harmed_baseline"]["opposite"] == 8
        for label, d in decisions.items():
            b = d["best"][0, :, :, 0, -1]
            rg = d["regret"][0, :, :, 0, -1]
            false = (room(xy[pool, mode, b[:, true], -1]) == 2) & (room(physical_xy[pool, mode, b[:, true], -1]) != 2)
            summary = {}
            for scope, take in (("all", np.ones(len(n), bool)), ("reversed", rev),
                                ("reversed_top_crossed", rev & (mode == 0)), ("reversed_top_blocked", rev & (mode == 1)),
                                ("northwest", stage == 3), ("northwest_A", (stage == 3) & ~ref["label_native_false_target_h5"]),
                                ("northwest_B", (stage == 3) & ref["label_native_false_target_h5"])):
                w = weights if label == "fixed_support" else baseline_weights[name]
                factual_p = w[n, post, mode]
                summary[scope] = dict(n=count(take), own_false_target=count(take & false),
                                      p_true_mean=float(factual_p[take].mean()), p_true_majority=count(take & (factual_p > .5)),
                                      mean_regret={p: dict(zip(REGRETS, rg[take, j].mean(axis=0).tolist())) for j, p in enumerate(POLICIES)},
                                      post_minus_uniform={r: diff_stats((rg[:, post, j] - rg[:, uniform, j])[take]) for j, r in enumerate(REGRETS)},
                                      post_minus_pre={r: diff_stats((rg[:, post, j] - rg[:, 0, j])[take]) for j, r in enumerate(REGRETS)},
                                      changed_plan=count(take & (b[:, post] != b[:, uniform])))
            result[name][label] = summary
            for q in n:
                rows.append(dict(arm=name, support=label, query=q, seed=bank["seed"][pool[q]], stage=stage[q],
                                 mode=mode[q], original_false_target=int(ref["label_native_false_target_h5"][q]),
                                 own_false_target=int(false[q]), original_pair_harm=int(harmed[q]),
                                 p1_pre=w[q, 0, 1], p1_post=w[q, post, 1],
                                 **{f"{p}_candidate": b[q, j] for j, p in enumerate(POLICIES)},
                                 **{f"{p}_{r}": rg[q, j, k] for j, p in enumerate(POLICIES) for k, r in enumerate(REGRETS)}))
            for q in EXAMPLES:
                examples.setdefault(str(q), {}).setdefault(name, {})[label] = {
                    "p1_post": float(w[q, post, 1]), "candidate": dict(zip(POLICIES, b[q].tolist())),
                    "regret": dict(zip(POLICIES, [dict(zip(REGRETS, vs)) for vs in rg[q].tolist()]))}
        pair_xy = xy[qp, :, old_picks]
        actual_pair_xy = physical_xy[qp, :, old_picks]
        pair_exposure = exposure[qp, :, old_picks]
        for q in n[rev]:
            p, u = old_picks[q]
            for h in range(5):
                r = dict(arm=name, query=q, seed=bank["seed"][pool[q]], mode=mode[q], pool=pool[q], horizon=h + 1,
                         baseline_harm=int(harmed[q]), post_candidate=p, uniform_candidate=u, p1_post=weights[q, post, 1],
                         predicted_delta_B=db[q, h], predicted_delta_D=dd[q, h], rho_delta_D=rho[q] * dd[q, h], post_margin=margin[q, h],
                         physical_delta_B=physical_diff[q, :, h].mean(), physical_delta_D=delta_D_truth[q, h],
                         physical_first_gap=ref["group_min"][q, p, h] - ref["group_min"][q, u, h],
                         physical_plan_gap=physical_diff[q, mode[q], h])
                for j, picked in enumerate(("post", "uniform")):
                    for m in (0, 1):
                        x, y = pair_xy[q, j, m, h]
                        ax, ay = actual_pair_xy[q, j, m, h]
                        r.update({f"{picked}_mode{m}_{key}": value for key, value in dict(
                            cost=own_pair[q, j, m, h], actual_cost=physical_pair[q, j, m, h], x=x, y=y,
                            actual_x=ax, actual_y=ay, exposure=pair_exposure[q, j, m, h]).items()})
                pairs.append(r)
            if q in EXAMPLES:
                examples[str(q)][name]["fixed_pair_H1_H5"] = [
                    {k: float(row[k]) for k in ("horizon", "predicted_delta_B", "predicted_delta_D", "rho_delta_D", "post_margin",
                                                 "physical_delta_D", "physical_first_gap", "physical_plan_gap")}
                    for row in pairs[-5:]]
                examples[str(q)][name]["original_candidates"] = [int(p), int(u)]
    assert result["baseline"]["fixed_support"]["reversed"]["post_minus_uniform"]["plan_regret"]["harmed"] == 13
    csv_output(out / "lifecycle_queries.csv", rows)
    csv_output(out / "reversal_fixed_pair_traces.csv", pairs)
    write_json(out / "reversal_examples.json", examples)
    return result


def quiet_source_report(bank, life, evidence, arm_weights):
    archive, _, _, _, _, mode, _ = life
    choice_row, initial, time, stage, _ = bank["choices"].T
    reverse = stage == 2
    rows, arms, steps = choice_row[reverse], initial[reverse], time[reverse]
    idx = rows[:, None], arms[:, None], steps[:, None] + np.arange(1, 11)
    true_mode = mode[reverse]
    same_arrival = (archive["life_cf_z"][idx][..., 0, :] == archive["life_cf_z"][idx][..., 1, :]).all(axis=-1)
    np.testing.assert_array_equal(same_arrival, ~archive["life_informative"][idx])
    northwest = stage == 3
    assert count(same_arrival) == 608 and count(~same_arrival) == 32
    result = {}
    for name, traces in evidence.items():
        gap = np.diff(traces["log_likelihood"][idx], axis=-1)[..., 0] * (2 * true_mode[:, None] - 1)
        values = arm_weights[name][np.arange(len(mode)), POLICIES.index("post"), mode]
        result[name] = {
            "arrivals": {label: dict(n=count(take), favors_true=count(take & (gap > 0)),
                                     favors_wrong=count(take & (gap < 0)), equal=count(take & (gap == 0)))
                         for label, take in (("all", np.ones_like(same_arrival)), ("same_actual_arrival", same_arrival),
                                             ("different_actual_arrival", ~same_arrival))},
            "northwest_posterior": {label: dict(n=count(take), correctly_prefers_true=count(take & (values > .5)),
                                                incorrectly_prefers_wrong=count(take & (values < .5)), equal_support=count(take & (values == .5)),
                                                p_true_range=[float(values[take].min()), float(values[take].max())])
                                    for label, take in (("all", northwest), ("mode0", northwest & (mode == 0)),
                                                        ("mode1", northwest & (mode == 1)))}}
    return result


def strict_door_pairs(bank):
    """Choose only stock-replayed legal held-out witnesses, before scoring any model."""
    pairs, meta = [], []
    for i, seed in enumerate(bank["seed"]):
        scene = make_scene(int(seed))
        physical = PhysicalRollout(scene)
        for j, (door, goal) in enumerate((("top", scene.southeast), ("bottom", scene.northwest))):
            start = bank["root_xy"][i, j, -1]
            assert start[0] <= physical.near or start[0] >= physical.far
            acts = bank["action"][i, j].reshape(-1, 25, 2)
            xy = bank["future_xy"][i, j].reshape(-1, 2, 25, 2)
            score = VisibleMapCost(scene, goal)
            costs = score(xy[..., -1, :])
            before = np.concatenate((np.broadcast_to(start, (*xy.shape[:2], 1, 2)), xy[..., :-1, :]), axis=-2)
            attempted = before + SPEED * acts[:, None]
            clipped = (attempted != xy).any(axis=-1)
            x_side = xy[..., 0] >= CENTER
            source_right = start[0] >= CENTER
            traverses = x_side != source_right
            has_split = (xy[:, 0] != xy[:, 1]).any(axis=-1)
            first = has_split.argmax(axis=1)
            k = np.arange(len(xy))
            side_stop = physical.far + .5 if source_right else physical.near - .5
            causal = (has_split.any(axis=1) & (xy[k, 1-j, first, 0] == side_stop)
                      & (xy[k, 1-j, first, 1] == xy[k, j, first, 1]) & ~clipped[k, j, first]
                      & (abs(xy[k, j, first, 1] - scene.door_positions[door]) <= physical.aperture))
            single = causal & traverses[:, j, -1] & ~traverses[:, 1-j].any(axis=1)
            improves = (costs[:, j] < costs[:, 1-j]) & (costs[:, j] < score(start))
            legal = single & improves & ~clipped[:, j].any(axis=1)
            common = ~has_split.any(axis=1)
            blocked = common & (room(xy[:, 0, -1]) == room(start)) & clipped[:, 0, -5:].any(axis=1)
            legal_common = common & ~clipped[:, 0].any(axis=1) & (room(xy[:, 0, -1]) != room(start)) & (costs[:, 0] < score(start))
            same_first = (acts[:, None, :5] == acts[None, :, :5]).all(axis=(2, 3))
            same_end = (xy[:, j, None, 4] == xy[None, :, j, 4]).all(axis=-1)
            pair_switch = ((costs[:, None, j] < costs[None, :, j]) & (costs[:, None, 1-j] > costs[None, :, 1-j]))
            eligible = legal[:, None] & (blocked | legal_common)[None, :] & ~same_first & ~same_end & pair_switch
            if eligible.any():
                a, b = np.where(eligible)
                order = np.lexsort((b, a, -np.minimum(costs[b, j] - costs[a, j], costs[a, 1-j] - costs[b, 1-j])))
                chosen = [int(a[order[0]]), int(b[order[0]])]
            else:
                chosen = None
            pairs.append(chosen)
            meta.append(dict(seed=int(seed), root=door, open_mode=j, candidate_pair=chosen, root_xy=start.tolist(),
                             strict_open_candidates=count(legal), otherwise_valid_clipped=count(single & improves & ~legal),
                             common_blocked_candidates=count(blocked), common_legal_candidates=count(legal_common),
                             strict_switch_pairs=count(eligible),
                             common_branch_type=("common_blocked" if blocked[chosen[1]] else "common_legal") if chosen else None))
    return pairs, meta


def doors_report(bank, pairs, meta, arrays, out):
    base = load_npz(RUN / "evaluation_paired_arrays.npz")
    np.testing.assert_array_equal(base["seed"], bank["seed"])
    actual, baseline, base_exposure = (base[key] for key in ("Y_bank", "F_bank", "F_bank_exposure"))
    np.testing.assert_array_equal(actual, bank["future_xy"][..., -1, :])
    rows = []
    for k, (indices, m) in enumerate(zip(pairs, meta)):
        if indices is None:
            continue
        i, j = divmod(k, 2)
        scene = make_scene(m["seed"])
        score = VisibleMapCost(scene, scene.southeast if j == 0 else scene.northwest)
        truth = score(actual[i, j, indices]).astype(np.float64)
        delta_true = truth[0] - truth[1]
        assert delta_true[j, -1] < 0 < delta_true[1-j, -1]
        forecasts = {"baseline": (baseline[i, j, indices], base_exposure[i, j, indices]),
                     **{a: (v["door_xy"][k], v["door_exposure"][k]) for a, v in arrays.items()}}
        for arm, (xy, exp) in forecasts.items():
            cost = score(xy).astype(np.float64)
            delta = cost[0] - cost[1]
            for h in range(5):
                rows.append(dict(arm=arm, **m, horizon=h+1, predicted_delta_D=delta[1, h] - delta[0, h],
                                 physical_delta_D=delta_true[1, h] - delta_true[0, h],
                                 predicted_delta_B=delta[:, h].mean(), physical_delta_B=delta_true[:, h].mean(),
                                 predicted_open_witness_cheaper=bool(delta[j, h] < 0), predicted_closed_common_cheaper=bool(delta[1-j, h] > 0),
                                 predicted_witness_crosses_in_open=bool((xy[0, j, h, 0] >= CENTER) != (m["root_xy"][0] >= CENTER)),
                                 predicted_witness_crosses_in_closed=bool((xy[0, 1-j, h, 0] >= CENTER) != (m["root_xy"][0] >= CENTER)),
                                 witness_open_exposure=exp[0, j, h], witness_closed_exposure=exp[0, 1-j, h],
                                 common_open_exposure=exp[1, j, h], common_closed_exposure=exp[1, 1-j, h]))
    result = {}
    for door in ("top", "bottom"):
        these = [m for m in meta if m["root"] == door]
        result[door] = dict(layouts=len(these), strict_witness_coverage=sum(m["strict_open_candidates"] > 0 for m in these),
                            strict_switch_pair_coverage=sum(m["candidate_pair"] is not None for m in these))
        for arm in ("baseline", *arrays):
            part = [r for r in rows if r["root"] == door and r["arm"] == arm and r["horizon"] == 5]
            p = np.array([r["predicted_delta_D"] for r in part])
            t = np.array([r["physical_delta_D"] for r in part])
            result[door][arm] = dict(delta_D=signs(p, t, np.ones(len(part), dtype=bool)),
                                     predicts_switch=sum(r["predicted_open_witness_cheaper"] and r["predicted_closed_common_cheaper"] for r in part),
                                     open_plan_predicted_crossing=sum(r["predicted_witness_crosses_in_open"] for r in part),
                                     closed_plan_false_crossing=sum(r["predicted_witness_crosses_in_closed"] for r in part))
    write_json(out / "strict_door_selection.json", meta)
    if rows:
        csv_output(out / "strict_door_witnesses.csv", rows)
    return result


@torch.inference_mode()
def forecast_arm(name, device, nav, bank, life, doors, pairs):
    print(f"Scoring {name} on existing banks", flush=True)
    arm_dir = OUTPUT / ("arm1" if name == INFERENCE_CONTROL else name)
    suffix = "_inference_r1" if name == INFERENCE_CONTROL else ""
    mods = load_trained(arm_dir, "conditioned", device)
    if name == INFERENCE_CONTROL:
        mods[2].forced_exposure = 1.0
    decode = load_readout(device, REFERENCE)
    nav_values = predict(mods, decode, device, nav)
    door_values = predict(mods, decode, device, doors, pairs=pairs)
    life_values = predict(mods, decode, device, bank, lifecycle=True)
    scale = torch.load(REFERENCE / "likelihood_scale.pt", map_location=device, weights_only=True)
    evidence = replay_evidence(mods, life[0], scale, device)
    weights = new_supports(evidence, bank["choices"], life[-2])
    output = {f"{panel}_{key}": values[i] for panel, values in (("nav", nav_values), ("life", life_values), ("door", door_values))
              for i, key in enumerate(("xy", "exposure", "root_decoded"))}
    if name in ("arm2", INFERENCE_CONTROL):
        np.testing.assert_array_equal(output["nav_exposure"], 1.)
        np.testing.assert_array_equal(output["life_exposure"], 1.)
    np.savez_compressed(arm_dir / f"fixed_bank_forecasts{suffix}.npz", **output, weights=weights)
    np.savez_compressed(arm_dir / f"fixed_lifecycle_evidence{suffix}.npz", **evidence)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forecast", choices=(*ARMS[1:], INFERENCE_CONTROL))
    p.add_argument("--device", default="cuda")
    p.add_argument("--summary", action="store_true", help="Summarize both trained checkpoints from forecast caches")
    p.add_argument("--include-inference-control", action="store_true", help="Also read Arm1 with its exposure forced at inference")
    p.add_argument("--baseline-only", action="store_true", help="Check archived counts on CPU before trained forecasts")
    a = p.parse_args()
    if not (a.forecast or a.summary or a.baseline_only):
        p.error("provide --forecast, --summary or --baseline-only")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    nav = load_npz(RUN / "bank_paired.npz")
    bank = load_npz(RUN / "writein_fixed_h5_banks.npz")
    doors = load_npz(DOORS)
    life = lifecycle_data(bank)
    pairs, meta = strict_door_pairs(doors)
    if a.forecast:
        forecast_arm(a.forecast, torch.device(a.device), nav, bank, life, doors, pairs)
    if a.summary or a.baseline_only:
        arrays = {name: load_npz(OUTPUT / name / "fixed_bank_forecasts.npz") for name in ARMS[1:]} if a.summary else {}
        if a.include_inference_control:
            arrays[INFERENCE_CONTROL] = load_npz(OUTPUT / "arm1/fixed_bank_forecasts_inference_r1.npz")
        evidence = {"baseline": life[1], **{name: load_npz(OUTPUT / name / "fixed_lifecycle_evidence.npz")
                                           for name in ARMS[1:] if name in arrays}}
        if INFERENCE_CONTROL in arrays:
            evidence[INFERENCE_CONTROL] = load_npz(OUTPUT / "arm1/fixed_lifecycle_evidence_inference_r1.npz")
        out = OUTPUT / ("fixed_bank_eval" if arrays else "archive_eval_check")
        out.mkdir(exist_ok=True)
        nav_inputs = nav_data(nav)
        combined = dict(
            metadata=dict(arm_order=("baseline", *arrays), original_candidate_pair_orientation="first minus second", cost_gap="mode1 minus mode0",
                          lifecycle_fixed_support="archived actual factual support, held fixed across checkpoints",
                          lifecycle_own_support="replay each checkpoint on identical actual arrivals with fixed old latent likelihood scale"),
            newcem=nav_report(nav_inputs, arrays, out), lifecycle=lifecycle_report(bank, life, arrays, out),
            post_reversal_arrivals=quiet_source_report(bank, life, evidence, {"baseline": life[-1], **{k: v["weights"] for k, v in arrays.items()}}),
            separate_heldout_doors=doors_report(doors, pairs, meta, arrays, out))
        write_json(out / "summary.json", combined)
        write_json(out / "geometry.json", nav_geometry(nav, nav_inputs, arrays))
        print(out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
