"""Score archived common roots without running a policy, encoder, or predictor."""

import csv
import json
from argparse import ArgumentParser
from collections import deque
from pathlib import Path

import numpy as np

from research.p1a.physical_topology_gate import (
    ACTION_BLOCK, ARMS, CENTER, DEADLINE, GOAL_RADIUS, LOCK, SAMPLES, SEEDS, SPEED,
    PhysicalRollout, ReversibleFourRoom, VisibleMapCost, doors_between, make_scene,
    probe_and_dormancy, room, route_seed, select, shared_cem,
)


OUT = Path(__file__).resolve().parent
RUNS = OUT.parents[2]
ATOL = 1e-5  # Position/scoring roundoff only, in native map units.


def raw_trajectories(physics, start, actions):
    xy = np.broadcast_to(start, (2, len(actions), 2))
    sequence = []
    for command in actions.reshape(len(actions), -1, 2).transpose(1, 0, 2):
        xy = physics.advance(xy, command)
        sequence.append(xy)
    return np.stack(sequence, axis=2).reshape(2, len(actions), actions.shape[1], ACTION_BLOCK, 2)


def legal_both(physics, xy):
    x, y = xy
    in_x_wall = physics.near < x < physics.far
    in_y_wall = physics.near < y < physics.far
    open_horizontal = min(abs(x - physics.left), abs(x - physics.right)) <= physics.aperture
    return bool(
        np.all((xy >= physics.min_xy) & (xy <= physics.max_xy))
        and (not in_y_wall or open_horizontal)
        and (not in_x_wall or np.all(np.abs(y - physics.lock_centers) <= physics.aperture))
    )


def evaluate(actions, raw, goal, visible, h, *, verify_gate_cost=None):
    # First-block groups include all five commands, irrespective of proposal.
    _, inverse = np.unique(actions[:, 0].reshape(len(actions), -1), axis=0, return_inverse=True)
    end = raw[:, :, h - 1, -1]
    immediate = raw[:, :, 0, -1]
    cost = visible(end).astype(np.float64)
    immediate_cost = visible(immediate).astype(np.float64)
    hit = np.any(np.linalg.norm(raw[:, :, :h] - goal, axis=-1) < GOAL_RADIUS, axis=(-2, -1))

    q = np.full((2, int(inverse.max() + 1)), np.inf)
    group_hit = np.zeros(q.shape, dtype=bool)
    for mode in (0, 1):
        np.minimum.at(q[mode], inverse, cost[mode])
        np.logical_or.at(group_hit[mode], inverse, hit[mode])
    floor = cost.min(axis=1)
    np.testing.assert_array_equal(floor, q.min(axis=1))
    regret = q - floor[:, None]
    disjoint = np.all(np.max(regret, axis=0) > ATOL)
    ambiguity = float(np.min(np.max(regret, axis=0)))
    modes = cost.argmin(axis=1)
    uniform = int(cost.mean(axis=0).argmin())
    if verify_gate_cost is not None:
        np.testing.assert_array_equal(cost, verify_gate_cost)
        uniform = select(verify_gate_cost, "uniform", 0, 0)
    horizon = []
    for mode in (0, 1):
        oracle = int(modes[mode])
        stale = int(modes[1 - mode])
        actual_group = inverse[uniform]
        same_first = actual_group == inverse[oracle]
        r_first = regret[mode, actual_group]
        r_total = cost[mode, uniform] - floor[mode]
        suffix = cost[mode, uniform] - q[mode, actual_group]
        np.testing.assert_allclose(r_first + suffix, r_total, atol=ATOL, rtol=0)
        if same_first:
            assert r_first == 0
        lo = float(immediate_cost[mode, uniform] - immediate_cost[mode, oracle])
        # Immediate executed outcomes need not have the same sign as optimal H suffixes.
        horizon.append(dict(mode=mode, horizon=h, candidates=len(actions), first_groups=q.shape[1],
                            oracle=oracle, uniform=uniform, stale=stale,
                            mode_optimal_sets_disjoint=bool(disjoint),
                            minimax_blind_first_regret=ambiguity,
                            uniform_first_changed=bool(not same_first),
                            uniform_first_regret=float(r_first), uniform_plan_regret=float(r_total),
                            uniform_suffix_regret=float(suffix),
                            stale_first_regret=float(regret[mode, inverse[stale]]),
                            immediate_uniform_minus_true_cost=lo,
                            immediate_uniform_gt_true=bool(lo > ATOL),
                            any_candidate_hit=bool(hit[mode].any()), oracle_plan_hit=bool(hit[mode, oracle]),
                            uniform_plan_hit=bool(hit[mode, uniform]),
                            uniform_first_can_hit=bool(group_hit[mode, actual_group]),
                            oracle_first_can_hit=bool(group_hit[mode, inverse[oracle]]),
                            any_first_raw_mode_effect=bool(np.any(raw[0, :, 0] != raw[1, :, 0])),
                            any_horizon_raw_mode_effect=bool(np.any(raw[0, :, :h] != raw[1, :, :h])),
                            oracle_first=actions[oracle, 0].tolist(), uniform_first=actions[uniform, 0].tolist(),
                            oracle_h1_xy=immediate[mode, oracle].tolist(),
                            uniform_h1_xy=immediate[mode, uniform].tolist()))
    return cost, horizon


def gate_rows():
    # The large JSONL has terminal scores for chosen candidates only. Reconstruct
    # precisely the original first pools, verifying against all saved first choices.
    result, completion, checks = [], [], 0
    with (RUNS / "physical_topology_gate_20261115/scenes.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            seed = source["seed"]
            scene = make_scene(seed)
            physics = PhysicalRollout(scene)
            goal = scene.southeast
            visible = VisibleMapCost(scene, goal)
            for h in (4, 5, 6):
                archive = source["horizons"][str(h)]
                start = np.asarray(archive["0"]["true"]["matched_switch"]["ticks"][0]["start"], dtype=np.float32)
                np.testing.assert_array_equal(start, scene.northwest)
                plans, saved_cost = shared_cem(
                    scene, physics, visible, start, h,
                    np.random.default_rng(np.random.SeedSequence([seed, 0])),
                )
                actions = np.repeat(plans[:, :, None], ACTION_BLOCK, axis=2)
                raw = raw_trajectories(physics, start, actions)
                _, scores = evaluate(actions, raw, goal, visible, h, verify_gate_cost=saved_cost)
                for initial_string, arms in archive.items():
                    initial = int(initial_string)
                    mode = 1 - initial
                    score = scores[mode]
                    for arm in ARMS:
                        # The matched switch and the outbound are opposite-mode
                        # evaluations of the same start, goal, and proposal seed.
                        for leg, truth in ((arms[arm]["matched_switch"], mode),
                                           (arms[arm]["macro"]["outbound"], initial)):
                            tick = leg["ticks"][0]
                            expected = {name: select(saved_cost, name, truth, initial) for name in ARMS}
                            assert tick["common_bank_selections"] == expected
                            candidate = expected[arm]
                            assert tick["chosen"] == candidate
                            np.testing.assert_array_equal(tick["start"], start)
                            np.testing.assert_array_equal(tick["control"], plans[candidate, 0])
                            np.testing.assert_array_equal(tick["xy"], raw[truth, candidate, 0])
                            for name, index in expected.items():
                                assert tick["within_bank_regret"][name] == float(
                                    saved_cost[truth, index] - saved_cost[truth, expected["true"]])
                            checks += 1
                        macro = arms[arm]["macro"]
                        completion.append(dict(seed=seed, horizon=h, mode=mode, arm=arm,
                                               matched=arms[arm]["matched_switch"]["success"],
                                               outbound=macro["outbound"]["success"], macro=macro["success"]))
                    true_leg = arms["true"]["matched_switch"]
                    uniform_leg = arms["uniform"]["matched_switch"]
                    result.append(dict(source="gate0_exact_first_pool", seed=seed, root="northwest_matched_switch",
                                       x=float(start[0]), y=float(start[1]), legal_both=legal_both(physics, start),
                                       matched_true_leg=true_leg["success"], matched_uniform_leg=uniform_leg["success"],
                                       **score))
    assert len(result) == 384 and checks == 2304
    return result, completion, checks


def reachability_rows():
    result = []
    with np.load(RUNS / "native_topology_prefix_20261115/data_eval_bank.npz", allow_pickle=False) as f:
        seed, roots, actions, raw = (f[k] for k in ("seed", "root_xy", "action", "future_xy"))
    for i, s in enumerate(seed):
        scene = make_scene(int(s))
        physics = PhysicalRollout(scene)
        for root_index, name, goal in ((2, "northwest_navigation", scene.southeast),
                                       (3, "southeast_navigation", scene.northwest)):
            xy = roots[i, root_index, -1]
            visible = VisibleMapCost(scene, goal)
            for h in (4, 5):
                _, scores = evaluate(actions[i, root_index], raw[i, root_index].swapaxes(0, 1), goal, visible, h)
                for score in scores:
                    result.append(dict(source="archived_prefix_model_bank", seed=int(s), root=name,
                                       x=float(xy[0]), y=float(xy[1]), legal_both=legal_both(physics, xy), **score))
    assert len(result) == 128
    assert sum(r["any_candidate_hit"] for r in result if r["horizon"] == 5) == 8
    assert sum(r["oracle_plan_hit"] for r in result if r["horizon"] == 5) == 8
    return result


def lifecycle_rows():
    base = OUT.parents[1]
    with np.load(base / "writein_fixed_h5_banks.npz", allow_pickle=False) as f:
        seed, xy, actions, choice, reference = (f[k] for k in ("seed", "root_xy", "action", "choices", "physical_cost"))
    with np.load(OUT.parent / "horizon_traces.npz", allow_pickle=False) as f:
        reference_trace = f["physical_xy"]
    with np.load(RUNS / "native_topology_condition_20261115/data_holdout.npz", allow_pickle=False) as f:
        factual_mode, factual_x, cf_xy = (f[k] for k in ("life_mode", "life_xy", "life_cf_xy"))
    result = []
    table = {}
    for i, s in enumerate(seed):
        scene = make_scene(int(s))
        physics = PhysicalRollout(scene)
        visible = VisibleMapCost(scene, scene.southeast)
        raw = raw_trajectories(physics, xy[i], actions[i])
        np.testing.assert_array_equal(raw[:, :, :, -1], reference_trace[i])
        cost, scores = evaluate(actions[i], raw, scene.southeast, visible, 5)
        np.testing.assert_array_equal(cost, reference[i])
        table[i] = (legal_both(physics, xy[i]), scores)
    for i, initial, time, stage, pool in choice:
        mode = factual_mode[i, initial, time]
        np.testing.assert_array_equal(factual_x[i, initial, time + 1], xy[pool])
        common_source, scores = table[pool]
        pos = ("northwest" if stage == 3 else
               "top_probe_open_inside_wall_band" if not common_source else "top_probe_blocked_near_side")
        fact = cf_xy[i, initial, time]
        result.append(dict(source="archived_lifecycle_bank", seed=int(seed[pool]), root=pos,
                           x=float(xy[pool, 0]), y=float(xy[pool, 1]), legal_both=common_source, pool=int(pool),
                           stage=("first_probe", "reentry_probe", "reversed_probe", "after_reversal_quiet")[stage],
                           factual_block_has_raw_mode_effect=bool(np.any(fact[0] != fact[1])), **scores[mode]))
    assert len(result) == 256 and sum(r["legal_both"] for r in result) == 160
    return result


def mean_ci(rows, key):
    by_seed = np.array([np.mean([r[key] for r in rows if r["seed"] == s])
                        for s in sorted({r["seed"] for r in rows})])
    samples = np.random.default_rng(11752026).integers(0, len(by_seed), (10000, len(by_seed)))
    return [float(by_seed.mean()), *np.quantile(by_seed[samples].mean(axis=1), [.025, .975]).tolist()]


def summary(rows):
    metrics = {
        "queries": len(rows),
        "unique_physical_roots": len({(r["seed"], r["x"], r["y"]) for r in rows}),
        "legal_both": sum(r["legal_both"] for r in rows),
        "uniform_strict_first_regret": sum(r["uniform_first_regret"] > ATOL for r in rows),
        "uniform_regret_at_least_one_raw_step": sum(r["uniform_first_regret"] >= SPEED for r in rows),
        "minimax_blind_margin_at_least_one_raw_step": sum(r["minimax_blind_first_regret"] >= SPEED for r in rows),
    }
    for key in ("mode_optimal_sets_disjoint", "uniform_first_changed", "immediate_uniform_gt_true",
                "any_candidate_hit", "oracle_plan_hit", "uniform_plan_hit", "uniform_first_can_hit",
                "oracle_first_can_hit", "any_first_raw_mode_effect", "any_horizon_raw_mode_effect"):
        metrics[key] = sum(r[key] for r in rows)
    for key in ("uniform_first_regret", "uniform_plan_regret", "stale_first_regret",
                "immediate_uniform_minus_true_cost", "minimax_blind_first_regret"):
        metrics[key + "_mean_and_layout_bootstrap_95ci"] = mean_ci(rows, key)
    return metrics


def serialize(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class FactualWindow(ReversibleFourRoom):
    """Record RGB observations after actual zero blocks; leave stock physics intact."""

    def __init__(self, scene):
        super().__init__(scene)
        self.frames = deque(maxlen=3)

    def execute(self, action_block):
        path = super().execute(action_block)
        if not np.any(action_block):
            self.frames.append(self.render().copy())
        return path


def h6_facts(env, start, goal, physical):
    reference, factual = None, []
    for initial in (0, 1):
        env.begin(start, goal, initial)
        trace, decision_indices = [], []
        for stage, phase in enumerate(("first", "reversed")):
            env.frames.clear()
            probe_and_dormancy(env, start, trace, phase)
            current = initial ^ stage
            quiet = [tick for tick in trace if tick["phase"].startswith(phase + "_dormant_")]
            assert len(quiet) >= 5 and len(env.frames) == 3
            for tick in quiet:
                actions = np.asarray(tick["raw_actions"], dtype=np.float32)[None, None]
                raw = raw_trajectories(physical, np.asarray(tick["start"], dtype=np.float32), actions)
                np.testing.assert_array_equal(raw[0, 0, 0], raw[1, 0, 0])
                np.testing.assert_array_equal(raw[current, 0, 0], tick["raw_xy"])
            assert all(tick["phase"] == phase + "_dormant_hold" for tick in trace[-3:])
            actual = dict(root_xy=np.asarray([tick["raw_xy"][-1] for tick in trace[-3:]], dtype=np.float32),
                          root_action=np.asarray([tick["raw_actions"] for tick in trace[-2:]], dtype=np.float32),
                          root_rgb=np.stack(env.frames))
            np.testing.assert_array_equal(actual["root_xy"], np.broadcast_to(start, (3, 2)))
            if reference is None:
                reference = actual
            for key in reference:
                np.testing.assert_array_equal(actual[key], reference[key])
            decision_indices.append(len(trace))
            if stage == 0:
                env.set_mode(1 - initial)
        np.testing.assert_array_equal([tick["start"] for tick in trace[1:]],
                                      [tick["raw_xy"][-1] for tick in trace[:-1]])
        factual.append(dict(initial_mode=initial, switch_block=decision_indices[0],
                            decision_arrivals=decision_indices, blocks=trace))
    return reference, factual


def h6_witnesses(env, scene, start, goal, physical, plans):
    actions, paths, checks = [], [], []
    for mode in (0, 1):
        option, direction = next((i, d) for i, d in enumerate((1, -1))
                                 if LOCK[mode] not in doors_between(int(room(start)), int(room(goal)), d))
        witness_slot = option * (SAMPLES // 2)
        route = route_seed(scene, start, goal, direction, DEADLINE)
        control = np.repeat(route[:, None], ACTION_BLOCK, axis=1)
        np.testing.assert_array_equal(control[:6], plans[witness_slot])
        env.begin(start, goal, mode)
        actual = np.stack([env.execute(command) for command in control])
        np.testing.assert_array_equal(actual, raw_trajectories(physical, start, control[None])[mode, 0])
        raw = actual.reshape(-1, 2)
        prev = np.vstack((start, raw[:-1]))
        crosses = (prev[:, 0] >= CENTER) != (raw[:, 0] >= CENTER)
        open_door = LOCK[1 - mode]
        legal_crossing = crosses.any() and (np.abs(raw[crosses, 1] - scene.door_positions[open_door]) <= physical.aperture).all()
        successful_steps = np.flatnonzero(np.linalg.norm(raw - goal, axis=-1) < GOAL_RADIUS)
        checks.append(dict(mode=mode, slot=witness_slot, open_door=open_door,
                           crosses_open_door=bool(legal_crossing),
                           first_hit_block=int(successful_steps[0] // ACTION_BLOCK + 1) if len(successful_steps) else None))
        actions.append(control)
        paths.append(actual)
    return np.stack(actions), np.stack(paths), checks


def h6_scene(seed):
    scene = make_scene(seed)
    physical = PhysicalRollout(scene)
    env = FactualWindow(scene)
    roots, rows, factual = [], [], []
    try:
        for root, start, goal in (("northwest", scene.northwest, scene.southeast),
                                  ("southeast_mirror", scene.southeast, scene.northwest)):
            score = VisibleMapCost(scene, goal)
            controls, archived_style_cost = shared_cem(scene, physical, score, start, 6,
                                                        np.random.default_rng(np.random.SeedSequence([seed, 0])))
            plans = np.repeat(controls[:, :, None], ACTION_BLOCK, axis=2)
            physical_xy = raw_trajectories(physical, start, plans)
            window, episodes = h6_facts(env, start, goal, physical)
            witness_actions, witness_xy, witness = h6_witnesses(env, scene, start, goal, physical, plans)
            witnesses_pass = all(x["crosses_open_door"] and x["first_hit_block"] is not None for x in witness)
            joint = legal_both(physical, start) and witnesses_pass
            for h in (4, 5, 6):
                physical_cost, decisions = evaluate(plans, physical_xy, goal, score, h,
                                                    verify_gate_cost=archived_style_cost if h == 6 else None)
                qualifies = joint and decisions[0]["mode_optimal_sets_disjoint"] and (
                    decisions[0]["minimax_blind_first_regret"] >= SPEED)
                rows.extend(dict(seed=seed, root=root, x=float(start[0]), y=float(start[1]),
                                 legal_both=legal_both(physical, start), matched_window=True,
                                 both_route_witnesses=witnesses_pass, discriminating=qualifies, **d) for d in decisions)
            roots.append(dict(**window, action=plans, physical_xy=physical_xy, physical_cost=physical_cost,
                              target=goal, witness_actions=witness_actions, witness_xy=witness_xy,
                              discriminating=np.asarray(qualifies)))
            factual.append(dict(seed=seed, root=root, witnesses=witness, episodes=episodes))
    finally:
        env.close()
    return roots, rows, factual


def freeze_h6(path):
    path.mkdir(parents=True, exist_ok=False)
    roots, rows, facts = zip(*map(h6_scene, SEEDS), strict=True)
    rows = [row for batch in rows for row in batch]
    arrays = {key: np.stack([[root[key] for root in pair] for pair in roots]) for key in roots[0][0]}
    np.savez_compressed(path / "fixed_h6_bank.npz", seed=np.asarray(SEEDS), **arrays)
    serialize(path / "first_actions.csv", rows)
    with (path / "factual_paths.jsonl").open("x", encoding="utf-8") as f:
        for batch in facts:
            for report in batch:
                f.write(json.dumps(report, allow_nan=False) + "\n")
    h6 = [r for r in rows if r["horizon"] == 6]
    assert len(h6) == len(SEEDS) * 4
    output = dict(scope="Scripted physical probes and one fixed candidate bank per root; no learned predictions or closed loop",
                  candidates_per_root=int(arrays["action"].shape[2]), minimum_blind_margin_map_units=SPEED,
                  directions={}, both_directions_qualifying_layouts=int(arrays["discriminating"].all(axis=1).sum()),
                  all_roots_legal_matched_and_two_witnesses=all(r["legal_both"] and r["matched_window"] and
                                                               r["both_route_witnesses"] for r in h6))
    for root in ("northwest", "southeast_mirror"):
        part = [r for r in h6 if r["root"] == root]
        output["directions"][root] = dict(qualifying_layouts=sum(r["discriminating"] and r["mode"] == 0 for r in part),
                                          modes={m: summary([r for r in part if r["mode"] == m]) for m in (0, 1)})
    (path / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(path / "summary.json", flush=True)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-h6", type=Path, help="Freeze prospective common-root H6 panel in a new directory")
    args = parser.parse_args()
    if args.freeze_h6 is not None:
        freeze_h6(args.freeze_h6)
        return
    gate, completion, checks = gate_rows()
    prefix = reachability_rows()
    lifecycle = lifecycle_rows()
    cross_horizon = {}
    for arm in ARMS:
        cross_horizon[arm] = {}
        for key in ("matched", "outbound", "macro"):
            s = {h: {(r["seed"], r["mode"]) for r in completion
                     if r["arm"] == arm and r["horizon"] == h and r[key]} for h in (5, 6)}
            cross_horizon[arm][key] = dict(h5=len(s[5]), h6=len(s[6]), both=len(s[5] & s[6]),
                                           h5_only=len(s[5] - s[6]), h6_only=len(s[6] - s[5]))
    out = dict(original_first_decisions_verified=checks, numeric_tolerance_map_units=ATOL,
               gate0={h: summary([r for r in gate if r["horizon"] == h]) for h in (4, 5, 6)},
               gate0_mode={f"H{h}_mode{m}": summary([r for r in gate if r["horizon"] == h and r["mode"] == m])
                           for h in (4, 5, 6) for m in (0, 1)},
               gate0_h5_h6_completion_overlap=cross_horizon,
               prefix={h: summary([r for r in prefix if r["horizon"] == h]) for h in (4, 5)},
               prefix_root={f"H{h}_{p}": summary([r for r in prefix if r["horizon"] == h and r["root"] == p])
                            for h in (4, 5) for p in sorted({r["root"] for r in prefix})},
               lifecycle={p: summary([r for r in lifecycle if r["root"] == p])
                          for p in sorted({r["root"] for r in lifecycle})})
    with (OUT / "task_physics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write("\n")
    serialize(OUT / "task_physics_roots.csv", gate + prefix + lifecycle)
    print(f"Verified {checks} archived first choices and scored {len(gate) + len(prefix) + len(lifecycle)} rows.")


if __name__ == "__main__":
    main()
