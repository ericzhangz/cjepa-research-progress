"""Measure the gate alone on the frozen lifecycle plans, keeping factual support fixed."""

import argparse
import json

import numpy as np
import torch
from scipy.stats import t
from stable_worldmodel.envs.four_room.env import FourRoomEnv

from .native_topology_writein_attribution import (
    ARCHIVE, ATOL, MIXES, POLICIES, REGRETS, check_roots, csv_output, evaluate,
    first_block_groups, regret_parts, sign, supports,
)
from .native_topology_writein_horizon import OUTPUT, check_terminals, read_npz, replay_prefixes
from .native_topology_writein_replay import HORIZON, RUN, STAGES
from .physical_topology_gate import (
    CENTER, DOOR_SIZE, HALF_WALL, LOCK, RADIUS, SPEED, PhysicalRollout, VisibleMapCost, make_scene, room,
)


OUT = OUTPUT / "gate_followup"
ARMS = ("native", "forced")
POSITIONS = ("top_door_crossed", "top_door_blocked", "northwest")
COMPARISONS = ((1, 3), (1, 2), (1, 0), (4, 3))
FAILURES = ("static_occupancy", "static_skip", "lock_only", "invalid_static_prior", "border_occupancy",
            "no_geometry_mode_value", "other")


def forecast_cache(banks, saved, reference, device):
    native = replay_prefixes(banks, device, collect_gate=True)
    for k in reference:
        np.testing.assert_array_equal(native[k], reference[k])
    print(check_terminals(native, banks, saved), flush=True)
    forced = replay_prefixes(banks, device, force=1.0, collect_gate=True, replay_physical=False)
    np.testing.assert_array_equal(forced["decoded_root_xy"], native["decoded_root_xy"])
    np.testing.assert_array_equal(forced["exposure"], 1.0)
    result = {f"native_{k}": native[k] for k in ("exposure", "gate_logits")}
    result.update({f"forced_{k}": v for k, v in forced.items()})
    np.savez_compressed(OUT / "gate_forecasts.npz", **result)
    return result


def prefix_decisions(banks, mode, weights, xy, forecast_cost, physical_cost):
    pool = banks["choices"][:, -1]
    n = len(pool)
    truth = physical_cost[pool, mode].astype(np.float64)
    # A group compares the full five primitive commands of the first block.
    groups = np.stack([first_block_groups(banks["action"][i], c[:, 0])[0] for i, c in zip(pool, truth)])
    group_min = np.empty_like(truth)
    for j, (group, cost) in enumerate(zip(groups, truth)):
        minima = np.full((group.max() + 1, HORIZON), np.inf)
        np.minimum.at(minima, group, cost)
        group_min[j] = minima[group]

    best = np.empty((2, n, len(POLICIES), len(MIXES), HORIZON), np.int64)
    regret = np.empty((*best.shape, len(REGRETS)))
    b_true = physical_cost.astype(np.float64).mean(axis=1)
    d_true = physical_cost[:, 1].astype(np.float64) - physical_cost[:, 0]
    common_error = np.empty((2, len(physical_cost), HORIZON, 2))
    for arm, cost in enumerate(forecast_cost):
        b = cost.astype(np.float64).mean(axis=1)
        d = cost[:, 1].astype(np.float64) - cost[:, 0]
        common_error[arm, ..., 0] = np.square(xy[arm].astype(np.float64).mean(axis=1) - xy[2].mean(axis=1, dtype=np.float64)).sum(axis=-1).mean(axis=1)
        common_error[arm, ..., 1] = np.square(b - b_true).mean(axis=1)
        for h in range(HORIZON):
            center = np.stack((b[..., h], b_true[..., h], b[..., h], b_true[..., h]), axis=1)[pool]
            gap = np.stack((d[..., h], d[..., h], d_true[..., h], d_true[..., h]), axis=1)[pool]
            score = center[:, None] + (weights[:, :, 1, None, None] - 0.5) * gap[:, None]
            chosen = score.argmin(axis=-1)
            best[arm, ..., h] = chosen
            regret[arm, ..., h, :] = regret_parts(truth[..., h], group_min[..., h], chosen.reshape(n, -1)).reshape(n, len(POLICIES), len(MIXES), len(REGRETS))
    np.testing.assert_array_equal(best[0, :, :, 3], best[1, :, :, 3])
    np.testing.assert_array_equal(regret[:, :, 4, 3], 0.)
    np.testing.assert_allclose(regret[..., 0] + regret[..., 1], regret[..., 2], rtol=0, atol=ATOL)
    marker = regret[0, :, 3, 3, :, 0] > ATOL
    return dict(best=best, regret=regret, groups=groups, group_min=group_min, common_error=common_error,
                physical_mode_value=marker, chosen_group=groups[np.arange(n)[None, :, None, None, None], best])


def wall_occupancy(physical, xy, closed=None):
    """Strict interior of stock collision-centre strips where the aperture is absent."""
    x, y = xy
    fits = lambda value, center: center - physical.aperture <= value <= center + physical.aperture
    vert_open = ((closed != "top" and fits(y, physical.top)) or (closed != "bottom" and fits(y, physical.bottom)))
    horiz_open = fits(x, physical.left) or fits(x, physical.right)
    return (physical.near < x < physical.far and not vert_open) or (physical.near < y < physical.far and not horiz_open)


def raw_stock(env, physical, before, commands, closed):
    """Apply stock primitive collision rules, returning axis-specific wall responses."""
    env.closed_door = closed
    pos = torch.as_tensor(before, dtype=torch.float32)
    path, contact = [], []
    for cmd in torch.as_tensor(commands):
        attempted = pos + cmd * SPEED
        pos = env._apply_collisions(pos, attempted)
        path.append(pos.numpy())
        contact.append((pos != attempted.clamp(physical.min_xy, physical.max_xy)).numpy())
    return np.stack(path), np.stack(contact)


def geometry_flags(env, physical, before, end, commands, mode):
    closed = LOCK[mode]
    border = lambda pos: bool(((pos < physical.min_xy) | (pos > physical.max_xy)).any())
    invalid = wall_occupancy(physical, before) or border(before)
    invalid_locked = wall_occupancy(physical, before, closed) or border(before)
    static_occupancy = wall_occupancy(physical, end)
    lock_occ = wall_occupancy(physical, end, closed) and not static_occupancy
    open_path, open_contact = raw_stock(env, physical, before, commands, None)
    lock_path, lock_contact = raw_stock(env, physical, before, commands, closed)
    side, predicted, opened, locked = (p >= CENTER for p in (before, end, open_path[-1], lock_path[-1]))
    flipped = side != predicted
    static_skip = not invalid and bool((flipped & (predicted != opened) & open_contact.any(axis=0)).any())
    lock_skip = not invalid_locked and bool((flipped & (predicted == opened) & (predicted != locked) & lock_contact.any(axis=0)).any())
    return dict(static_occupancy=static_occupancy, static_skip=static_skip, lock_occupancy=lock_occ,
                lock_skip=lock_skip, lock_only=(lock_occ or lock_skip) and not (static_occupancy or static_skip),
                invalid_static_prior=invalid, invalid_locked_prior=invalid_locked, border_occupancy=border(end),
                open_room_mismatch=bool(np.any(predicted != opened)),
                locked_room_mismatch=bool(np.any(predicted != locked))), open_path, lock_path, open_contact, lock_contact


def geometry(banks, mode, decoded_root, xy, decision):
    row, _, _, _, pool = banks["choices"].T
    selected = decision["best"][:, :, 4, 0, -1]
    # The third row uses the forced forecast of the native-selected plan, before reselection.
    best = np.concatenate((selected, selected[:1]), axis=0)
    n = len(mode)
    flags = {}
    axy = np.empty((len(best), n, HORIZON, 2), np.float32)
    raw_paths = np.empty((len(best), n, HORIZON, 2, 5, 2), np.float32)
    raw_contacts = np.empty_like(raw_paths, dtype=bool)
    for layout in np.unique(row):
        sel = np.where(row == layout)[0]
        scene = make_scene(int(banks["seed"][pool[sel[0]]]))
        physical = PhysicalRollout(scene)
        env = FourRoomEnv(agent_speed=SPEED, agent_radius=RADIUS, wall_thickness=2 * HALF_WALL,
                          door_size=[DOOR_SIZE] * 4, door_positions=scene.door_positions,
                          close_door_prob=0.0, terminate_on_goal=False)
        env.reset(seed=scene.seed, options={"state": scene.northwest, "target_state": scene.southeast})
        assert env.closed_door is None and env.door_positions == scene.door_positions
        if layout == 0:
            inward = np.tile(np.float32([1, 0]), (5, 1))
            before = np.float32([100, 103])
            example, open_path, _, _, _ = geometry_flags(env, physical, before, np.float32([121, 103]), inward, 1)
            assert example["static_skip"] and not example["lock_only"] and not example["static_occupancy"]
            np.testing.assert_array_equal(open_path[-1], [103, 103])
            before = np.float32([100, physical.top])
            example, open_path, _, _, _ = geometry_flags(env, physical, before, np.float32([112.5, physical.top]), inward, 1)
            assert example["lock_only"] and example["lock_skip"] and not example["static_skip"]
            env.closed_door = None
            env.agent_position = torch.as_tensor(before)
            for cmd, point in zip(inward, open_path):
                env.step(cmd)
                np.testing.assert_array_equal(env.agent_position.numpy(), point)
        for q in sel:
            p, m = pool[q], mode[q]
            for arm, xy_arm in enumerate((0, 1, 1)):
                c = best[arm, q]
                axy[arm, q] = xy[xy_arm][p, m, c]
                before = decoded_root[p]
                factual = banks["root_xy"][p]
                for h, end in enumerate(axy[arm, q]):
                    cmds = banks["action"][p, c, h]
                    checks, *raw = geometry_flags(env, physical, before, end, cmds, m)
                    for name, value in checks.items():
                        if name not in flags:
                            flags[name] = np.zeros(axy.shape[:-1], dtype=bool)
                        flags[name][arm, q, h] = value
                    raw_paths[arm, q, h] = raw[:2]
                    raw_contacts[arm, q, h] = raw[2:]
                    before = end
                    path, _ = raw_stock(env, physical, factual, cmds, LOCK[m])
                    factual = xy[2][p, m, c, h]
                    np.testing.assert_array_equal(path[-1], factual)
    actual = xy[2][pool[None], mode[None], best]
    return dict(flags=flags, chosen_xy=axy, same_plan_actual_xy=actual, raw_paths=raw_paths, raw_contacts=raw_contacts,
                chosen_false_target=(room(axy) == 2) & (room(actual) != 2))


def fixed_pairs(banks, mode, xy, costs, decision):
    q = np.arange(len(mode))
    p = banks["choices"][:, -1]
    best = decision["best"]
    a, b = best[0, :, 3, 0, -1], best[0, :, 4, 3, -1]
    candidates = np.stack((a, b), axis=1)
    pair_xy = np.stack([values[p[:, None], :, candidates] for values in xy]).astype(np.float64)
    pair_cost = np.stack([values[p[:, None], :, candidates] for values in costs]).astype(np.float64)
    d_cost = pair_cost[..., 1, :] - pair_cost[..., 0, :]
    d_xy = pair_xy[..., 1, :, :] - pair_xy[..., 0, :, :]
    by_candidate = pair_cost.mean(axis=3)
    db = by_candidate[:, :, 0] - by_candidate[:, :, 1]
    dd = d_cost[:, :, 0] - d_cost[:, :, 1]
    dd_xy = d_xy[:, :, 0] - d_xy[:, :, 1]
    same_first = decision["groups"][q, a] == decision["groups"][q, b]
    first_gap = decision["group_min"][q, a] - decision["group_min"][q, b]
    plan_gap = costs[2][p, mode, a].astype(np.float64) - costs[2][p, mode, b]
    np.testing.assert_allclose(db[2] + (mode[:, None] - 0.5) * dd[2], plan_gap, atol=ATOL, rtol=0)
    return dict(pair_candidates=candidates, pair_xy=pair_xy, pair_cost=pair_cost, pair_d_cost=d_cost, pair_d_xy=d_xy,
                pair_delta_b=db, pair_delta_d=dd, pair_delta_d_xy=dd_xy, pair_same_first=same_first,
                pair_physical_first_gap=first_gap, pair_physical_plan_gap=plan_gap)


def exposure_summary(values, physical_effect):
    return dict(count=int(values.size), exactly_zero=int((values == 0).sum()),
                strictly_between=int(((values > 0) & (values < 1)).sum()), exactly_one=int((values == 1).sum()),
                zero_on_physical_effect=int(((values == 0) & physical_effect).sum()),
                between_on_physical_effect=int((((values > 0) & (values < 1)) & physical_effect).sum()),
                physical_effect_rows=int(physical_effect.sum()), min=float(values.min()), max=float(values.max()),
                mean=float(values.mean()))


def data_scopes(banks, mode, flags, false_target, decision):
    row, _, _, stage, pool = banks["choices"].T
    position = pool % len(POSITIONS)
    np.testing.assert_array_equal(row, pool // len(POSITIONS))
    np.testing.assert_array_equal(mode[position != 2], position[position != 2])
    np.testing.assert_array_equal(position == 2, stage == len(STAGES) - 1)
    scenes = [make_scene(int(seed)) for seed in banks["seed"][::len(POSITIONS)]]
    root = banks["root_xy"].reshape(-1, len(POSITIONS), 2)
    np.testing.assert_array_equal(root[:, :2, 1], np.repeat([[scene.door_positions["top"]] for scene in scenes], 2, axis=1))
    np.testing.assert_array_equal(root[:, 0, 0], CENTER + .5)
    np.testing.assert_array_equal(root[:, 1, 0], CENTER - HALF_WALL - RADIUS - .5)
    np.testing.assert_array_equal(root[:, 2], np.stack([scene.northwest for scene in scenes]))
    labels = {key: value[0].any(axis=-1) for key, value in flags.items()}
    excluded = np.logical_or.reduce([labels[k] for k in FAILURES[:5]])
    labels["physical_mode_value"] = decision["physical_mode_value"][:, -1]
    labels["no_geometry_mode_value"] = ~excluded & labels["physical_mode_value"]
    labels["native_false_target_h5"] = false_target[0, :, -1]
    labels["other"] = ~excluded & ~labels["no_geometry_mode_value"]
    primary = np.select([labels[k] for k in FAILURES], FAILURES, default="unclassified")
    assert np.all(primary != "unclassified")

    def grouped(in_scope):
        for m in np.unique(mode[in_scope]):
            this_mode = in_scope & (mode == m)
            yield f"mode_{m}", this_mode
            for i, s in enumerate(STAGES):
                yield f"mode_{m}/{s}", this_mode & (stage == i)

    scopes = {"all": np.ones(len(mode), dtype=bool)}
    scopes.update((f"all/{k}", v) for k, v in grouped(scopes["all"]))
    for p, name in enumerate(POSITIONS):
        in_scope = position == p
        scopes[name] = in_scope
        scopes.update((f"{name}/{k}", v) for k, v in grouped(in_scope))
        for f in FAILURES:
            scopes[f"{name}/exclusive/{f}"] = in_scope & (primary == f)
    scopes.update((f"flag/{k}", v) for k, v in labels.items())
    scopes.update((f"exclusive/{k}", primary == k) for k in FAILURES)
    return position, labels, primary, {k: v for k, v in scopes.items() if v.any()}


def comparison(values, layout):
    means = np.stack([values[layout == i].mean(axis=0) for i in np.unique(layout)])
    delta = values.mean(axis=0)
    layout_delta = means.mean(axis=0)
    if len(means) > 1:
        radius = t.ppf(.975, len(means) - 1) * means.std(axis=0, ddof=1) / np.sqrt(len(means))
        ci = np.stack((layout_delta - radius, layout_delta + radius), axis=-1).tolist()
    else:
        ci = None
    return dict(query_mean=delta.tolist(), layout_mean=layout_delta.tolist(), paired_layout_ci95=ci,
                positive_queries=(values > ATOL).sum(axis=0).tolist(), negative_queries=(values < -ATOL).sum(axis=0).tolist())


def pair_summary(pairs, take, h):
    result = {}
    if not take.any():
        return result
    cost, location, b = (pairs[k][:, take, h] for k in ("pair_delta_d", "pair_delta_d_xy", "pair_delta_b"))
    for name, values in (("cost", cost[..., None]), ("xy", location)):
        norms = np.linalg.norm(values, axis=-1)
        active = norms[2] > 0
        with_true = (values[:2] * values[2]).sum(axis=-1)
        result[name] = dict(physical_nonzero=int(active.sum()),
                            aligned=(with_true > 0).sum(axis=1).tolist(), reversed=(with_true < 0).sum(axis=1).tolist(),
                            missed=((norms[:2] == 0) & active).sum(axis=1).tolist(),
                            spurious_on_true_zero=((norms[:2] > 0) & ~active).sum(axis=1).tolist(),
                            recovered_alignment=int(((with_true[0] <= 0) & (with_true[1] > 0)).sum()),
                            lost_alignment=int(((with_true[0] > 0) & (with_true[1] <= 0)).sum()),
                            median_ratio=np.median(norms[:2, active] / norms[2, active], axis=1).tolist() if active.any() else [None, None],
                            median_amp_change=float(np.median(norms[1, active] - norms[0, active])) if active.any() else None)
    error_b = b[:2] - b[2]
    result["common_cost_gap"] = dict(median=np.median(b, axis=1).tolist(),
                                     alignment=(sign(b[:2]) * sign(b[2]) > 0).sum(axis=1).tolist(),
                                     median_abs_error=np.median(np.abs(error_b), axis=1).tolist(),
                                     median_signed_change=float(np.median(b[1] - b[0])),
                                     median_abs_error_change=float(np.median(np.abs(error_b[1]) - np.abs(error_b[0]))))
    return result


def summarize(banks, scopes, decision, pairs, geo, chosen_exposure, chosen_effect):
    row, _, _, _, pool = banks["choices"].T
    result = {}
    for key, take in scopes.items():
        ids = np.unique(pool[take])
        r = decision["regret"][:, take]
        choices = decision["best"][:, take]
        group = decision["chosen_group"][:, take]
        cf = {}
        for (a, b) in COMPARISONS:
            gain = r[:, :, b] - r[:, :, a]
            cf[f"{POLICIES[a]}_vs_{POLICIES[b]}"] = dict(mean_gain=gain.mean(axis=1).tolist(),
                                                         force_gain_change=comparison(gain[1] - gain[0], row[take]))
        result[key] = dict(
            queries=int(take.sum()), layouts=int(len(np.unique(row[take]))), pools=int(len(ids)),
            physical_mode_value=decision["physical_mode_value"][take].sum(axis=0).tolist(),
            mean_regret=r.mean(axis=1).tolist(), first_optimal_count=(r[..., 0] <= ATOL).sum(axis=1).tolist(),
            first_group_changed=(group[0] != group[1]).sum(axis=0).tolist(),
            plan_changed=(choices[0] != choices[1]).sum(axis=0).tolist(),
            native_minus_forced_regret=comparison(r[0, :, :, 0] - r[1, :, :, 0], row[take]),
            support_gain=cf, common_xy_cost_rmse=np.sqrt(decision["common_error"][:, ids].mean(axis=1)).tolist(),
            chosen_prefix_flags={k: v[:, take].sum(axis=1).tolist() for k, v in geo["flags"].items()},
            chosen_any_prefix_flags={k: v[:, take].any(axis=-1).sum(axis=1).tolist() for k, v in geo["flags"].items()},
            chosen_false_target=geo["chosen_false_target"][:, take].sum(axis=1).tolist(),
            fixed_pair_all=[pair_summary(pairs, take, h) for h in range(HORIZON)],
            fixed_pair_strict_h5_first_miss=[pair_summary(pairs, take & ~pairs["pair_same_first"] &
                                                         (pairs["pair_physical_first_gap"][:, -1] > ATOL), h) for h in range(HORIZON)],
            native_selected_exposure=[exposure_summary(chosen_exposure[take, :, h], chosen_effect[take, :, h])
                                      for h in range(HORIZON)],
        )
    return result


def query_rows(banks, mode, labels, primary, position, weights, decisions, geo, pairs,
               chosen_exposure, chosen_effect, learned_first_split, physical_first_split):
    for q, (row, initial, time, stage, pool) in enumerate(banks["choices"]):
        key = dict(query=q, layout=row, seed=banks["seed"][pool], pool=pool, position=POSITIONS[position[q]],
                   actual_root_x=banks["root_xy"][pool, 0], actual_root_y=banks["root_xy"][pool, 1],
                   initial_mode=initial, true_mode=mode[q], time=time, stage=STAGES[stage], failure=primary[q],
                   pair_A=pairs["pair_candidates"][q, 0], pair_B=pairs["pair_candidates"][q, 1],
                   native_selected_first_learned_divergence_block=learned_first_split[q],
                   native_selected_first_physical_divergence_block=physical_first_split[q])
        key.update({f"native_any_{name}": value[q] for name, value in labels.items()})
        for i, arm in enumerate(ARMS):
            for h in range(HORIZON):
                geometry_columns = {name: flag[i, q, h] for name, flag in geo["flags"].items()}
                fixed_flags = {f"forced_on_native_plan_{name}": flag[2, q, h] for name, flag in geo["flags"].items()}
                opt = geo["chosen_xy"][i, q, h]
                actual = geo["same_plan_actual_xy"][i, q, h]
                wide = {}
                for p, policy in enumerate(POLICIES):
                    wide[f"p1_{policy}"] = weights[q, p, 1]
                    for m, mix in enumerate(MIXES):
                        c = decisions["best"][i, q, p, m, h]
                        terms = decisions["regret"][i, q, p, m, h]
                        prefix = f"{policy}_{mix}"
                        wide.update({f"{prefix}_{field}": value for field, value in
                                     zip(("candidate", "first_group", *REGRETS),
                                         (c, decisions["chosen_group"][i, q, p, m, h], *terms))})
                yield key | dict(arm=arm, horizon=h+1, physical_mode_value=decisions["physical_mode_value"][q, h],
                                 fixed_h5_true_mode_candidate=decisions["best"][i, q, 4, 0, -1],
                                 fixed_h5_plan_predicted_x=opt[0], fixed_h5_plan_predicted_y=opt[1],
                                 fixed_h5_plan_physical_x=actual[0], fixed_h5_plan_physical_y=actual[1],
                                 fixed_h5_plan_false_target=geo["chosen_false_target"][i, q, h],
                                 forced_on_native_plan_false_target=geo["chosen_false_target"][2, q, h],
                                 native_fixed_h5_plan_exposure_mode0=chosen_exposure[q, 0, h],
                                 native_fixed_h5_plan_exposure_mode1=chosen_exposure[q, 1, h],
                                 native_fixed_h5_plan_physical_mode_effect=chosen_effect[q, 0, h],
                                 pair_first_physical_gap=pairs["pair_physical_first_gap"][q, h],
                                 pair_plan_physical_gap=pairs["pair_physical_plan_gap"][q, h],
                                 pair_delta_D=pairs["pair_delta_d"][i, q, h],
                                 pair_delta_B=pairs["pair_delta_b"][i, q, h],
                                 pair_physical_delta_D=pairs["pair_delta_d"][2, q, h],
                                 pair_physical_delta_B=pairs["pair_delta_b"][2, q, h]) | geometry_columns | fixed_flags | wide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summarize", action="store_true", help="Reuse completed native and forced gate traces")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    banks = read_npz(RUN / "writein_fixed_h5_banks.npz")
    paired = read_npz(RUN / "writein_paired.npz")
    reference = read_npz(OUTPUT / "horizon_traces.npz")
    weight_reference = read_npz(OUTPUT / "horizon_queries.npz")
    mode = check_roots(banks, read_npz(ARCHIVE))
    weights, quiet_error, omit_error = supports(paired, banks["choices"], mode)
    np.testing.assert_array_equal(weights, weight_reference["weights"])
    np.testing.assert_array_equal(mode, weight_reference["mode"])
    np.testing.assert_equal(banks["action"].shape, (96, 384, HORIZON, 5, 2))
    np.testing.assert_equal(banks["choices"].shape, (256, 5))
    terminal_checks = check_terminals(reference, banks, paired)
    cache = read_npz(OUT / "gate_forecasts.npz") if args.summarize else forecast_cache(banks, paired, reference, torch.device(args.device))
    assert all(np.isfinite(values).all() for values in cache.values())
    np.testing.assert_array_equal(cache["forced_exposure"], 1.0)
    assert np.all((cache["native_exposure"] >= 0) & (cache["native_exposure"] <= 1))
    xy, costs = ([reference[f"learned_{k}"], cache[f"forced_learned_{k}"], reference[f"physical_{k}"]]
                 for k in ("xy", "cost"))
    terminal_checks["forced_h5_max_abs_cost_error_vs_rescored_xy"] = 0.
    for p, seed in enumerate(banks["seed"]):
        scene = make_scene(int(seed))
        scored = VisibleMapCost(scene, scene.southeast)(xy[1][p, ..., -1, :])
        np.testing.assert_array_equal(scored, costs[1][p, ..., -1])
    print("Verified archive roots, measured gate arrays, frozen support and all endpoint costs", flush=True)
    decision = prefix_decisions(banks, mode, weights, xy, costs[:2], costs[2])
    np.testing.assert_array_equal(decision["best"][0, :, :, 0], weight_reference["learned_best"])
    np.testing.assert_array_equal(decision["best"][0, :, :, 3], weight_reference["physical_best"])
    attr = evaluate(banks, mode, paired)
    np.testing.assert_array_equal(decision["best"][0, ..., -1], attr["best"])
    np.testing.assert_array_equal(decision["regret"][0, ..., -1, :], attr["regret"])
    np.testing.assert_array_equal(decision["physical_mode_value"][:, -1], attr["physical_space"])
    print("Verified native H1–H5 selection, H5 four cells and full first-block regret", flush=True)
    geo = geometry(banks, mode, cache["forced_decoded_root_xy"], xy, decision)
    position, labels, primary, scopes = data_scopes(banks, mode, geo["flags"], geo["chosen_false_target"], decision)
    pairs = fixed_pairs(banks, mode, xy, costs, decision)
    np.testing.assert_array_equal(pairs["pair_candidates"], weight_reference["pair"])
    p = banks["choices"][:, -1]
    c = decision["best"][0, :, 4, 0, -1]
    native_exposure = cache["native_exposure"][p, :, c]
    native_diff, true_diff = [(values[p, 0, c] != values[p, 1, c]).any(axis=-1) for values in (xy[0], xy[2])]
    first_split = lambda diff: np.where(diff.any(axis=-1), diff.argmax(axis=-1) + 1, 0)
    learned_first_split, physical_first_split = first_split(native_diff), first_split(true_diff)
    chosen_effect = np.broadcast_to(true_diff[:, None], native_exposure.shape)
    summaries = summarize(banks, scopes, decision, pairs, geo, native_exposure, chosen_effect)
    fullbank_effect = (xy[2][:, 0] != xy[2][:, 1]).any(axis=-1)
    exp = cache["native_exposure"]
    fullbank_gate = [exposure_summary(exp[..., h], np.broadcast_to(fullbank_effect[:, None, :, h], exp[..., h].shape))
                     for h in range(HORIZON)]
    overlap = {a: {b: int((mask & other).sum()) for b, other in labels.items()} for a, mask in labels.items()}
    metadata = dict(arms=ARMS, pair_arms=(*ARMS, "physical"), policies=POLICIES, mixes=MIXES, regrets=REGRETS, stages=STAGES,
                    positions=POSITIONS, exclusive_priority=FAILURES, strictly_positive_atol=ATOL,
                    position_coordinates="top-door open and blocked arrivals share y=top doorway; no lower-door query",
                    geometry_arms=(*ARMS, "forced_on_native_plan"),
                    checkpoint="paired", factual_support="read from existing files; same in each arm", quiet_error=quiet_error,
                    omit_error=omit_error, paired_interval="95% paired t interval over unique layout means",
                    geometry="native H5 true-mode best fixes all labels; inspect all its H1–H5 endpoints and predicted-start transitions",
                    fixed_pair="A = original learned uniform H5 best; B = original physical true-mode H5 best; same pair all horizons/arms")
    (OUT / "gate_summary.json").write_text(json.dumps(dict(metadata=metadata, terminal_checks=terminal_checks, flag_overlap=overlap,
                                                           fullbank_native_gate=fullbank_gate, scopes=summaries),
                                                     separators=(",", ":"), allow_nan=False), encoding="utf-8")
    csv_output(OUT / "gate_queries.csv", query_rows(banks, mode, labels, primary, position, weights, decision, geo, pairs,
                                                    native_exposure, chosen_effect, learned_first_split, physical_first_split))
    flags = {f"geo_{k}": v for k, v in geo["flags"].items()}
    flags.update({f"label_{k}": v for k, v in labels.items()})
    np.savez_compressed(OUT / "gate_queries.npz", choices=banks["choices"], mode=mode, weights=weights, position=position,
                        factual_root_xy=banks["root_xy"][p],
                        primary_label=primary, learned_first_split=learned_first_split, physical_first_split=physical_first_split,
                        native_selected_exposure=native_exposure, native_selected_physical_effect=true_diff,
                        **decision, **pairs, **flags, **{k: v for k, v in geo.items() if k != "flags"})
    print(f"Saved {len(mode)} queries with {len(scopes)} nonempty frozen strata", flush=True)


if __name__ == "__main__":
    main()
