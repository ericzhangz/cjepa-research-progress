"""Collect same-root H5 physical counterparts to failures of a current predictor."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .native_topology_condition import CALIBRATE, HOLDOUT, RUN as READOUT_RUN, forecast, load_readout, load_trained
from .native_topology_prefix_data import (
    EVAL_SEEDS, HORIZON, OUTPUT as PREFIX_RUN, ROOT_NAMES, TRAIN_SEEDS, actual_common_windows,
    encode_groups, encode_shortlist, execute_selected, onset_raw, original_h5, query_bank,
    replay_bank, save_npz, stack_rows,
)
from .physical_topology_gate import (
    ACTION_BLOCK, CENTER, ITERATIONS, SAMPLES, SPEED, PhysicalRollout, ReversibleFourRoom,
    VisibleMapCost, make_scene, room, shared_cem,
)


OUTPUT = PREFIX_RUN.with_name("native_topology_paired_20261115")
BAD_ROLE, LEGAL_ROLE = 6, 7
LIVE_OUTPUT = PREFIX_RUN.with_name("native_topology_live_20261116")
LIVE_SEEDS = tuple(range(2026112100, 2026112228))
LIVE_PANEL = "data_train_live.npz"
LIVE_BAD_ROLE = 10
LIVE_ROLES = {0: "contact", 8: "upper_open", 9: "lower_open", LIVE_BAD_ROLE: "live_bad", 11: "common_legal"}
CUMULATIVE_KINDS = {-1: "unused", 0: "fixed", 1: "false_goal", 2: "interior_blocked", 3: "archived"}
CUMULATIVE_MINING_RULE = "false_goal_or_interior_blocked_exact_control_dedup"
CUMULATIVE_RANKING_RULE = "mined_distinct_first_command_and_endpoint_same_mode_gap_strictly_above_5"
LIVE_COMMON_KEYS = ("seed", "root_xy", "root_action", "root_z")
LIVE_TRAJECTORY_KEYS = ("action", "future_xy", "local_xy", "local_valid", "first_local_divergence_raw",
                        "first_branch_divergence_raw")
LIVE_BANK = "data_train_live_bank.npz"
LIVE_SUMMARY = "data_train_live_summary.json"


def clipping(start, actions, xy):
    """Use the stock position update before clipping, at each actual position."""
    trace = xy.reshape(len(actions), 2, -1, 2)
    source = np.concatenate((np.broadcast_to(start, (*trace.shape[:2], 1, 2)), trace[:, :, :-1]), axis=2)
    expected = source + SPEED * actions.reshape(len(actions), 1, -1, 2)
    return (expected != trace).any(axis=-1)


def choose_pair(scene, physical, score, start, root, raw, xy, parent_cost, false_goal, blocked):
    """Rank bad live queries and find a same-mode legal witness independently of their model rank."""
    candidates, modes = np.nonzero((false_goal | blocked)[4:])
    candidates += 4
    order = np.lexsort((modes, candidates, parent_cost[candidates, modes], ~false_goal[candidates, modes]))
    if not len(order):
        return None, "no_bad_live_query", False

    def legal_pool(actions, traces):
        terminal = traces[:, :, -1, -1]
        cost = score(terminal)
        legal = (~clipping(start, actions, traces).any(axis=-1)
                 & (room(terminal) != room(start)) & (cost < score(start)))
        return actions, traces, cost, legal

    menu = legal_pool(raw[:4], xy[:4])
    physical_pool = None
    for i in order:
        bad, mode = int(candidates[i]), int(modes[i])
        bad_cost = float(score(xy[bad, mode, -1, -1]))

        def matches(pool):
            actions, traces, cost, legal = pool
            different_command = (actions[:, 0] != raw[bad, 0]).any(axis=(1, 2))
            different_first_end = (traces[:, mode, 0, -1] != xy[bad, mode, 0, -1]).any(axis=-1)
            return np.flatnonzero(legal[:, mode] & (cost[:, mode] < bad_cost)
                                  & different_command & different_first_end)

        witnesses = matches(menu)
        if len(witnesses):
            routes = witnesses[witnesses < 2]
            preferred = routes if len(routes) else witnesses
            legal_index = int(preferred[menu[2][preferred, mode].argmin()])
            source = "visible_route" if legal_index < 2 else "archived_physical_cem"
            return (bad, mode, source, legal_index, raw[legal_index], xy[legal_index]), None, physical_pool is not None

        if physical_pool is None:
            rng = np.random.default_rng(np.random.SeedSequence((scene.seed, root, 554)))
            controls, _ = shared_cem(scene, physical, score, start, raw.shape[1], rng)
            actions = np.repeat(controls[:, :, None], ACTION_BLOCK, axis=2)
            physical_pool = legal_pool(actions, replay_bank(physical, start, actions))
        witnesses = matches(physical_pool)
        if len(witnesses):
            actions, traces, cost, _ = physical_pool
            legal_index = int(witnesses[cost[witnesses, mode].argmin()])
            return (bad, mode, "physical_cem", legal_index, actions[legal_index], traces[legal_index]), None, True
    return None, "no_clean_lower_cost_first_command", True


def install_pair(coverage, root, pair, raw, xy, model, decode, scene, env, physical, score, batch):
    bad, mode, source, legal_index, legal_action, legal_xy = pair
    start = coverage["root_xy"][root, -1]
    history = coverage["root_z"][root]
    past = coverage["root_action"][root]
    old_actions = coverage["action"][root]
    existing_legal = np.flatnonzero((old_actions == legal_action).all(axis=(1, 2, 3)))
    legal_slot = int(existing_legal[0]) if len(existing_legal) else 3
    bad_slot = 2 if legal_slot != 2 else 3
    duplicate_bad = np.flatnonzero((old_actions == raw[bad]).all(axis=(1, 2, 3)))
    if len(duplicate_bad):
        bad_slot = int(duplicate_bad[0])
        if bad_slot == legal_slot:
            legal_slot = 3 if bad_slot != 3 else 2

    witness = legal_index
    if source == "physical_cem":
        witness = len(raw)
        raw = np.concatenate((raw, legal_action[None]))
        xy = np.concatenate((xy, legal_xy[None]))
    chosen = np.array((bad, witness))
    local, valid, local_onset, checks = execute_selected(env, physical, start, score.goal, raw, xy, chosen)
    actions, actual = raw[chosen], xy[chosen]
    selected = {
        "action": actions[None], "future_xy": actual[None], "local_xy": local[None],
        "local_valid": valid[None], "first_local_divergence_raw": local_onset[None],
        "first_branch_divergence_raw": onset_raw(actual)[None],
    }
    env.begin(scene.northwest, scene.southeast, 0)
    common = {key: coverage[key][root:root + 1] for key in ("root_xy", "root_z", "root_action")}
    encoded, frames = encode_shortlist(env, model[0], common, selected, batch)
    device = model[0].action_mean.device
    prediction = forecast(*model, torch.as_tensor(history, device=device)[None],
                          torch.as_tensor(past, device=device)[None],
                          torch.as_tensor(actions, device=device)[None], compact=True)
    predicted = decode(prediction[0, :, :, -1]).cpu().numpy()
    selected_cost = score(predicted)
    corner_xy = np.where(start < CENTER, physical.near - .5, physical.far + .5)
    actual_terminal = actual[:, :, -1, -1]
    coverage["selection_role"][root, [bad_slot, legal_slot]] = BAD_ROLE, LEGAL_ROLE
    for index, slot in enumerate((bad_slot, legal_slot)):
        # Keep an existing witness's actual latent and history byte-for-byte.
        if not np.array_equal(old_actions[slot], actions[index]):
            for key in (*selected, "future_z", "local_z"):
                coverage[key][root, slot] = encoded[key][0, index]
        else:
            for key in selected:
                np.testing.assert_array_equal(coverage[key][root, slot], encoded[key][0, index])
        coverage["parent_xy"][root, slot] = predicted[index]
        coverage["parent_cost"][root, slot] = selected_cost[index]
        coverage["parent_goal_room"][root, slot] = (room(predicted[index]) == room(score.goal)).any()
        coverage["common_corner"][root, slot] = (actual_terminal[index] == corner_xy).all()
        coverage["block_end_separated"][root, slot] = (actual[index, 0, :, -1] != actual[index, 1, :, -1]).any()
        coverage["bank_index"][root, slot] = bad if index == 0 else (legal_index if witness == legal_index else -1)
    if len(np.unique(coverage["action"][root].reshape(4, -1), axis=0)) != 4:
        raise AssertionError("A paired root must contain four distinct plans")
    return {
        "mode": mode, "bad_live_index": bad, "legal_source": source, "legal_index": legal_index,
        "bad_slot": bad_slot, "legal_slot": legal_slot, "first_commands": actions[:, 0, 0].tolist(),
        "actual_h1_endpoints": actual[:, mode, 0, -1].tolist(),
        "actual_h5_endpoints": actual_terminal[:, mode].tolist(),
        "actual_h5_costs": score(actual_terminal[:, mode]).tolist(),
        "prefix_h5_costs": selected_cost[:, mode].tolist(),
    }, checks, frames


@torch.inference_mode()
def collect_layout(seed, coverage, original, model, decode, batch):
    scene = make_scene(seed)
    physical = PhysicalRollout(scene)
    env = ReversibleFourRoom(scene)
    banks, records = [], []
    try:
        for root, goal in ((2, scene.southeast), (3, scene.northwest)):
            start = coverage["root_xy"][root, -1]
            score = VisibleMapCost(scene, goal)
            raw, parent_xy, parent_cost, decoded = query_bank(
                scene, score, root, coverage["root_z"][root], coverage["root_action"][root],
                original["action"][root, :4], model, decode,
            )
            xy = replay_bank(physical, start, raw)
            np.testing.assert_array_equal(xy[:4], original["future_xy"][root, :4])
            terminal = xy[:, :, -1, -1]
            false_goal = (room(parent_xy) == room(goal)) & (room(terminal) != room(goal))
            blocked = (room(terminal) == room(start)) & clipping(start, raw, xy)[..., -ACTION_BLOCK:].any(axis=-1)
            pair, missing, searched = choose_pair(scene, physical, score, start, root, raw, xy,
                                                  parent_cost, false_goal, blocked)
            record = dict(seed=seed, root=ROOT_NAMES[root], fallback=missing, physical_cem_searched=searched,
                          negative_counts_by_mode=(false_goal | blocked)[4:].sum(axis=0).tolist(),
                          false_goal_counts_by_mode=false_goal[4:].sum(axis=0).tolist(),
                          blocked_counts_by_mode=blocked[4:].sum(axis=0).tolist(), checks=0, rgb_frames=0)
            if pair is not None:
                detail, record["checks"], record["rgb_frames"] = install_pair(
                    coverage, root, pair, raw, xy, model, decode, scene, env, physical, score, batch,
                )
                detail["false_goal_room"] = bool(false_goal[pair[0], pair[1]])
                detail["blocked"] = bool(blocked[pair[0], pair[1]])
                record.update(detail)
            banks.append(dict(action=raw, future_xy=xy, parent_xy=parent_xy, parent_cost=parent_cost,
                              decoded_current=decoded, actual_cost=score(terminal), false_goal_room=false_goal,
                              blocked=blocked, pair_mode=np.int8(pair[1] if pair else -1)))
            records.append(record)
    finally:
        env.close()
    return coverage, stack_rows(banks), records


def far_door_route(physical, start):
    """A fixed path through the near horizontal opening into the far vertical opening."""
    sign = 1 if start[0] < CENTER else -1
    canonical = lambda value: CENTER + sign * (value - CENTER)
    horizontal = canonical(physical.left if sign == 1 else physical.right)
    vertical = canonical(physical.bottom if sign == 1 else physical.top)
    x, y = canonical(start)
    moves = []
    for _ in range(HORIZON * ACTION_BLOCK):
        next_y = min(y + SPEED, vertical)
        if next_y < physical.far + .5:
            limit = horizontal + physical.aperture - .5
        elif next_y < vertical - physical.aperture + .25:
            limit = physical.near - .5
        else:
            limit = physical.max_xy
        next_x = min(x + SPEED, limit)
        moves.append((sign * (next_x - x) / SPEED, sign * (next_y - y) / SPEED))
        x, y = next_x, next_y
    return np.asarray(moves, dtype=np.float32).reshape(HORIZON, ACTION_BLOCK, 2)


def live_geometry(physical, score, start, raw, xy):
    trace = xy.reshape(len(raw), 2, -1, 2)
    terminal = trace[:, :, -1]
    cost = score(terminal)
    hits = clipping(start, raw, xy)
    common = (trace[:, 0] == trace[:, 1]).all(axis=(-2, -1))
    same_room = room(terminal[:, 0]) == room(start)
    blocked = common & same_room & hits[:, 0].any(axis=-1)
    legal = common & ~same_room & ~hits[:, 0].any(axis=-1) & (cost[:, 0] < score(start))
    crossed = (trace[..., 0] >= CENTER) != (start[0] >= CENTER)
    first = np.maximum(0, onset_raw(xy))
    boundary = physical.near - .5 if start[0] < CENTER else physical.far + .5
    clipped_x = trace[np.arange(len(raw)), :, first, 0] == boundary
    witness = (crossed[..., -1] & ~crossed[:, ::-1].any(axis=-1) & (cost < score(start))
               & (cost < cost[:, ::-1]) & clipped_x[:, ::-1] & ~clipped_x)
    return cost, hits, blocked, legal, witness


def different_first(raw, xy, a, b):
    return ((raw[a, 0] != raw[b, 0]).any(axis=(-2, -1))
            & (xy[a, :, 0, -1] != xy[b, :, 0, -1]).any(axis=-1).all(axis=-1))


def fixed_live_anchors(scene, physical, score, start, root, old):
    """Physical-only symmetric witnesses, independent of both compared arms."""
    rng = np.random.default_rng(np.random.SeedSequence((scene.seed, root, 554)))
    plans, _ = shared_cem(scene, physical, score, start, HORIZON, rng)
    raw = np.concatenate((old, np.repeat(plans[:, :, None], ACTION_BLOCK, axis=2),
                          far_door_route(physical, start)[None]))
    xy = replay_bank(physical, start, raw)
    cost, hits, _, legal_mask, witness = live_geometry(physical, score, start, raw, xy)
    u, d = (np.flatnonzero(witness[:, mode]) for mode in (0, 1))
    u, d = (a.ravel() for a in np.meshgrid(u, d, indexing="ij"))
    distinct = different_first(raw, xy, u, d)
    u, d = u[distinct], d[distinct]
    swap = (cost[u, 0] < cost[d, 0]) & (cost[d, 1] < cost[u, 1])
    clips = hits[u, 0].any(axis=-1).astype(np.int8) + hits[d, 1].any(axis=-1)
    legal = np.flatnonzero(legal_mask)
    if not (len(u) and len(legal)):
        raise ValueError(f"Missing fixed physical gate or legal witness: {scene.seed} {ROOT_NAMES[root]}")
    top = np.lexsort((d, u, cost[u, 0] + cost[d, 1], clips, ~swap))[0]
    return raw[[u[top], d[top], legal[cost[legal, 0].argmin()]]]


@torch.inference_mode()
def collect_live_layout(seed, i, model, decode, batch, reference=None):
    scene = make_scene(seed)
    physical, env = PhysicalRollout(scene), ReversibleFourRoom(scene)
    rows, banks, records = [], [], []
    checks = frames = 0
    try:
        root_xy, root_action = actual_common_windows(scene, env)
        if reference is None:
            env.begin(scene.northwest, scene.southeast, 0)
            (root_z,), frames = encode_groups(env, model[0], (root_xy,), batch)
        else:
            for key, observed in (("root_xy", root_xy), ("root_action", root_action)):
                np.testing.assert_array_equal(reference[key], observed)
            root_z = reference["root_z"]
        common = dict(seed=np.int64(seed), root_xy=root_xy, root_action=root_action, root_z=root_z)
        for root, window in enumerate(root_xy):
            start = window[-1]
            goal = scene.southeast if root in (0, 2) else scene.northwest
            score = VisibleMapCost(scene, goal)
            old = original_h5(scene, physical, start, goal, root)
            if root < 2:
                raw, chosen = old, np.arange(4)
                xy = replay_bank(physical, start, raw)
                roles = np.zeros(4, dtype=np.int8)
                bank_indices = chosen
            else:
                anchor_slots = [0, 1, 3]
                anchors = (fixed_live_anchors(scene, physical, score, start, root, old) if reference is None
                           else reference["action"][root, anchor_slots])
                raw, parent_xy, parent_cost, decoded = query_bank(
                    scene, score, root, root_z[root], root_action[root], old, model, decode,
                )
                n = 4 + ITERATIONS * SAMPLES
                if len(raw) != n:
                    raise ValueError("Live model CEM returned the wrong candidate budget")
                raw = np.concatenate((raw, anchors))
                xy = replay_bank(physical, start, raw)
                actual_cost, hits, blocked, legal, witness = live_geometry(physical, score, start, raw, xy)
                if not (witness[n, 0] and witness[n + 1, 1] and legal[n + 2]):
                    raise ValueError(f"Changed physical witnesses: {seed} {ROOT_NAMES[root]}")
                terminal = xy[:n, :, -1, -1]
                false_goal = (room(parent_xy) == room(goal)) & (room(terminal) != room(goal))
                designation = (i + root) % 2
                live = np.arange(4, n)
                eligible = blocked[live] & (actual_cost[live, 0] > actual_cost[n + 2, 0])
                for anchor in range(n, n + 3):
                    eligible &= different_first(raw, xy, live, anchor)
                negatives = live[eligible]
                if not len(negatives):
                    raise ValueError(f"No same-root common blocked/live and legal pair: {seed} {ROOT_NAMES[root]}")
                best = np.lexsort((negatives, ~hits[negatives, 0, -ACTION_BLOCK:].any(axis=-1),
                                   parent_cost[negatives, designation], ~false_goal[negatives, designation]))[0]
                bad = negatives[best]
                chosen = np.asarray((n, n + 1, bad, n + 2))
                roles = np.arange(8, 12, dtype=np.int8)
                bank_indices = np.asarray((-1, -1, bad, -1))
                banks.append(dict(action=raw[:n], future_xy=xy[:n], parent_xy=parent_xy,
                                  parent_cost=parent_cost, decoded_current=decoded,
                                  false_goal=false_goal, common_blocked=blocked[:n]))
                records.append(dict(
                    seed=seed, root=ROOT_NAMES[root], designation=designation, live_bad_index=int(bad),
                    kind="false_goal" if false_goal[bad, designation] else "common_blocked_hard_negative",
                    selected_false_goal_by_mode=false_goal[bad].tolist(),
                    live_false_goal_by_mode=false_goal[4:].sum(axis=0).tolist(),
                    eligible_false_goal_by_mode=false_goal[negatives].sum(axis=0).tolist(),
                    eligible_common_blocked=len(negatives),
                    true_cost=actual_cost[chosen].tolist(),
                    strict_gate_swap=bool(actual_cost[n, 0] < actual_cost[n + 1, 0]
                                          and actual_cost[n + 1, 1] < actual_cost[n, 1]),
                ))
            local, valid, local_onset, n_checks = execute_selected(env, physical, start, goal, raw, xy, chosen)
            checks += n_checks
            rows.append(dict(action=raw[chosen], future_xy=xy[chosen], local_xy=local, local_valid=valid,
                             first_local_divergence_raw=local_onset, first_branch_divergence_raw=onset_raw(xy[chosen]),
                             selection_role=roles, bank_index=bank_indices))
        selected = stack_rows(rows)
        env.begin(scene.northwest, scene.southeast, 0)
        if reference is None:
            panel, n_frames = encode_shortlist(env, model[0], common, selected, batch)
        else:
            changed = selected["selection_role"] == LIVE_BAD_ROLE
            for key in ("action", "future_xy", "local_xy", "local_valid", "first_local_divergence_raw",
                        "first_branch_divergence_raw", "selection_role", "bank_index"):
                np.testing.assert_array_equal(selected[key][~changed], reference[key][~changed])
            dynamic = {key: value[changed, None] for key, value in selected.items()}
            encoded, n_frames = encode_shortlist(env, model[0], common, dynamic, batch)
            panel = dict(common, **selected)
            for key in ("future_z", "local_z"):
                panel[key] = reference[key].copy()
                panel[key][changed] = encoded[key][:, 0]
        if not (np.isfinite(panel["future_z"]).all()
                and np.isfinite(panel["local_z"][panel["local_valid"]]).all()):
            raise FloatingPointError(f"Nonfinite RGB targets in train scene {seed}")
        return panel, stack_rows(banks), records, checks, frames + n_frames
    finally:
        env.close()


def collect_live_training(output, checkpoint, *, device="cuda", batch=32, reference=None):
    reserved = set((*TRAIN_SEEDS, *EVAL_SEEDS, *CALIBRATE, *HOLDOUT, *range(2026111900, 2026111932)))
    if len(LIVE_SEEDS) != 128 or len(set(LIVE_SEEDS)) != 128 or reserved.intersection(LIVE_SEEDS):
        raise ValueError("Live training must use 128 distinct layouts outside the original, calibration and evaluation splits")
    output.mkdir(parents=True, exist_ok=True)
    if (output / LIVE_PANEL).exists():
        raise FileExistsError(output / LIVE_PANEL)
    if reference is not None:
        with np.load(reference, allow_pickle=False) as f:
            control = dict(f)
        np.testing.assert_array_equal(control["seed"], LIVE_SEEDS)
    model = load_trained(checkpoint, "conditioned", device)
    decode = load_readout(device, READOUT_RUN)
    panels, banks, records = [], [], []
    checks = frames = 0
    for i, seed in enumerate(LIVE_SEEDS):
        row = {key: value[i] for key, value in control.items()} if reference is not None else None
        panel, bank, detail, n_checks, n_frames = collect_live_layout(seed, i, model, decode, batch, row)
        panels.append(panel)
        banks.append(bank)
        records.extend(detail)
        checks += n_checks
        frames += n_frames
        if i == 0 or (i + 1) % 8 == 0:
            print(f"live data {i + 1}/{len(LIVE_SEEDS)}: {frames} RGB, "
                  f"{sum(r['kind'] == 'false_goal' for r in records)}/{len(records)} designated false goals", flush=True)
    panel, bank = stack_rows(panels), stack_rows(banks)
    summary = dict(
        layout_seeds=list(LIVE_SEEDS), root_names=ROOT_NAMES, slot_roles=LIVE_ROLES,
        checkpoint=str(checkpoint / "conditioned.pt"), fixed_anchor_reference=str(reference) if reference else None,
        navigation_roots=len(records), model_cem_budget_per_navigation_root=[ITERATIONS, SAMPLES],
        upper_open_witnesses=len(records), lower_open_witnesses=len(records),
        strict_gate_swaps=sum(r["strict_gate_swap"] for r in records), common_blocked_legal_pairs=len(records),
        designations=np.bincount([r["designation"] for r in records], minlength=2).tolist(),
        selected_false_goals_by_designation=[sum(r["kind"] == "false_goal" and r["designation"] == m for r in records)
                                             for m in (0, 1)],
        selected_hard_negatives_by_designation=[sum(r["kind"] != "false_goal" and r["designation"] == m for r in records)
                                                for m in (0, 1)],
        canonical_trajectory_checks=checks, encoded_rgb_positions=frames,
        changed_live_plans=(int((panel["action"] != control["action"]).any(axis=(-3, -2, -1)).sum())
                            if reference is not None else None),
        shapes={key: list(value.shape) for key, value in panel.items()}, rows=records,
    )
    save_npz(output / "data_train_live_bank.npz", dict(seed=panel["seed"], **bank))
    (output / "data_train_live_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_npz(output / LIVE_PANEL, panel)


def cumulative_geometry(physical, score, start, action, xy, legal_action, legal_xy, false_goal):
    """Mine physical prediction errors independently of their eligibility for a rank pair."""
    trace = xy.reshape(len(action), 2, -1, 2)
    before = np.concatenate((np.broadcast_to(start, trace[:, :, :1].shape), trace[:, :, :-1]), axis=2)
    expected = np.clip(before + SPEED * action.reshape(len(action), 1, -1, 2), physical.min_xy, physical.max_xy)
    interior = (expected != trace).any(axis=(-2, -1))
    terminal = trace[:, :, -1]
    actual_cost = score(terminal)
    gap = actual_cost - score(legal_xy[:, -1, -1])
    distinct_command = (action[:, 0] != legal_action[0]).any(axis=(-2, -1))
    distinct_endpoint = (xy[:, :, 0, -1] != legal_xy[:, 0, -1]).any(axis=-1)
    blocked = interior & (room(terminal) != room(score.goal))
    mining_eligible = false_goal | blocked
    rank_eligible = mining_eligible & (gap > 5.) & distinct_command[:, None] & distinct_endpoint
    return actual_cost, gap, interior, mining_eligible, rank_eligible


def choose_cumulative_negatives(action, predicted_cost, false_goal, eligible, retained):
    """Prefer false goals, then low predicted costs, with two distinct controls per root."""
    indices, modes = np.nonzero(eligible[4:])
    indices += 4
    order = np.lexsort((modes, indices, predicted_cost[indices, modes], ~false_goal[indices, modes]))
    chosen = []
    for index, mode in zip(indices[order], modes[order]):
        if any(np.array_equal(action[index], action[prior]) for prior, _ in chosen):
            continue
        if (retained == action[index]).all(axis=(1, 2, 3)).any():
            continue
        chosen.append((index, mode))
        if len(chosen) == 2:
            break
    return chosen


def extend_live_candidates(panel, count):
    out = {key: value if key in LIVE_COMMON_KEYS else np.concatenate(
        (value, np.repeat(value[:, :1], count, axis=1)), axis=1,
    ) for key, value in panel.items()}
    for key in ("candidate_valid", "local_valid", "mining_eligible", "rank_eligible", "false_goal", "interior_collision",
                "mining_round", "actual_cost", "actual_cost_gap"):
        out[key][:, -count:] = 0
    for key in ("selection_role", "bank_index", "candidate_kind", "candidate_mode",
                "first_local_divergence_raw", "first_branch_divergence_raw"):
        out[key][:, -count:] = -1
    for key in ("local_z", "local_xy"):
        out[key][:, -count:] = np.nan
    return out


def read_live_archive(path):
    with np.load(path, allow_pickle=False) as saved:
        panel = dict(saved)
    np.testing.assert_array_equal(panel["seed"], LIVE_SEEDS)
    np.testing.assert_array_equal(panel["selection_role"][:, 2:, 2], LIVE_BAD_ROLE)
    indices = panel["bank_index"][:, 2:, 2]
    if not ((4 <= indices) & (indices < 4 + ITERATIONS * SAMPLES)).all():
        raise ValueError(f"Archived negatives must index model CEM candidates in {path}")
    rows = np.arange(len(LIVE_SEEDS))[:, None], np.arange(2)[None], indices
    with np.load(path.with_name(LIVE_BANK), allow_pickle=False) as bank:
        np.testing.assert_array_equal(bank["seed"], LIVE_SEEDS)
        for key in ("action", "future_xy"):
            np.testing.assert_array_equal(panel[key][:, 2:, 2], bank[key][rows])
        false_goal = bank["false_goal"][rows]
    return panel, false_goal


def cumulative_seed_layout(i, reference, archive, reference_false, archive_false):
    """Keep the two archived negatives and exactly one copy of each fixed witness."""
    panel = {key: value[i].copy() for key, value in reference.items()}
    shape = panel["selection_role"].shape
    panel.update(candidate_valid=np.ones(shape, dtype=bool), candidate_kind=np.zeros(shape, dtype=np.int8),
                 candidate_mode=np.full(shape, -1, dtype=np.int8), mining_round=np.zeros(shape, dtype=np.int8))
    for key in ("false_goal", "interior_collision", "mining_eligible", "rank_eligible"):
        panel[key] = np.zeros((*shape, 2), dtype=bool)
    for key in ("actual_cost", "actual_cost_gap"):
        panel[key] = np.zeros((*shape, 2), dtype=np.float32)
    panel["bank_index"][panel["selection_role"] != LIVE_BAD_ROLE] = -1
    panel = extend_live_candidates(panel, 1)
    for key, value in archive.items():
        if key not in LIVE_COMMON_KEYS:
            panel[key][2:, 4] = value[i, 2:, 2]
    panel["candidate_valid"][2:, 4] = True
    panel["candidate_kind"][2:, [2, 4]] = 3
    panel["candidate_mode"][2:, [2, 4]] = ((i + np.arange(2, 4)) % 2)[:, None]
    panel["mining_round"][2:, [2, 4]] = (-1, -2)
    panel["false_goal"][2:, 2] = reference_false[i]
    panel["false_goal"][2:, 4] = archive_false[i]
    return panel


@torch.inference_mode()
def collect_cumulative_layout(seed, previous, model, decode, batch, mining_round):
    scene = make_scene(seed)
    physical, env = PhysicalRollout(scene), ReversibleFourRoom(scene)
    panel = extend_live_candidates(previous, 2)
    first_slot = panel["action"].shape[1] - 2
    banks, records = [], []
    checks = frames = 0
    try:
        root_xy, past = actual_common_windows(scene, env)
        np.testing.assert_array_equal(root_xy, panel["root_xy"])
        np.testing.assert_array_equal(past, panel["root_action"])
        for root, window in enumerate(root_xy):
            start = window[-1]
            goal = scene.southeast if root in (0, 2) else scene.northwest
            score = VisibleMapCost(scene, goal)
            valid = previous["candidate_valid"][root]
            retained = previous["action"][root, valid]
            if root < 2:
                panel["actual_cost"][root, np.flatnonzero(valid)] = score(previous["future_xy"][root, valid, :, -1, -1])
                continue
            legal, legal_xy = (panel[key][root, 3] for key in ("action", "future_xy"))
            if mining_round == 1:
                slots = np.flatnonzero(valid)
                costs = cumulative_geometry(physical, score, start, retained, previous["future_xy"][root, valid],
                                            legal, legal_xy, previous["false_goal"][root, valid])
                for key, value in zip(("actual_cost", "actual_cost_gap", "interior_collision", "mining_eligible",
                                       "rank_eligible"), costs):
                    panel[key][root, slots] = value
                panel["mining_eligible"][root, [0, 1, 3]] = False
                panel["rank_eligible"][root, [0, 1, 3]] = False
            old = original_h5(scene, physical, start, goal, root)
            raw, parent_xy, parent_cost, decoded = query_bank(
                scene, score, root, panel["root_z"][root], past[root], old, model, decode,
            )
            if len(raw) != 4 + ITERATIONS * SAMPLES:
                raise ValueError("Cumulative CEM returned the wrong candidate budget")
            xy = replay_bank(physical, start, raw)
            false = (room(parent_xy) == room(goal)) & (room(xy[:, :, -1, -1]) != room(goal))
            actual, gap, interior, mining_eligible, rank_eligible = cumulative_geometry(
                physical, score, start, raw, xy, legal, legal_xy, false,
            )
            mining_eligible[:4] = False
            rank_eligible[:4] = False
            selected = choose_cumulative_negatives(raw, parent_cost, false, mining_eligible, retained)
            indices = np.asarray([index for index, _ in selected], dtype=np.intp)
            local, valid, onset, n_checks = execute_selected(env, physical, start, goal, raw, xy, indices)
            checks += n_checks
            for j, (index, mode) in enumerate(selected):
                slot = first_slot + j
                kind = 1 if false[index, mode] else 2
                values = dict(action=raw[index], future_xy=xy[index], local_xy=local[j], local_valid=valid[j],
                              first_local_divergence_raw=onset[j], first_branch_divergence_raw=onset_raw(xy[index:index + 1])[0],
                              selection_role=LIVE_BAD_ROLE, bank_index=index, candidate_valid=True,
                              candidate_mode=mode, candidate_kind=kind, mining_round=mining_round,
                              false_goal=false[index], interior_collision=interior[index],
                              mining_eligible=mining_eligible[index], rank_eligible=rank_eligible[index],
                              actual_cost=actual[index], actual_cost_gap=gap[index])
                for key, value in values.items():
                    panel[key][root, slot] = value
            banks.append(dict(action=raw, future_xy=xy, parent_xy=parent_xy, parent_cost=parent_cost,
                              decoded_current=decoded, false_goal=false, interior_collision=interior,
                              mining_eligible=mining_eligible, rank_eligible=rank_eligible, actual_cost=actual, actual_cost_gap=gap))
            records.append(dict(seed=seed, root=ROOT_NAMES[root],
                                mining_eligible_false_goals=(mining_eligible & false).sum(axis=0).tolist(),
                                mining_eligible_blocked=(mining_eligible & ~false).sum(axis=0).tolist(),
                                rank_eligible_false_goals=(rank_eligible & false).sum(axis=0).tolist(),
                                candidates=[dict(bank_index=int(index), mode=int(mode),
                                                 kind=CUMULATIVE_KINDS[1 if false[index, mode] else 2],
                                                 actual_cost=actual[index].tolist(), actual_cost_gap=gap[index].tolist(),
                                                 mining_eligible=mining_eligible[index].tolist(),
                                                 rank_eligible=rank_eligible[index].tolist(),
                                                 repeated_control=bool((retained == raw[index]).all(axis=(1, 2, 3)).any()))
                                            for index, mode in selected]))
        fresh = panel["mining_round"] == mining_round
        if fresh.any():
            env.begin(scene.northwest, scene.southeast, 0)
            shortlist = {key: panel[key][fresh, None] for key in LIVE_TRAJECTORY_KEYS}
            common = {key: panel[key] for key in LIVE_COMMON_KEYS}
            encoded, frames = encode_shortlist(env, model[0], common, shortlist, batch)
            for key in ("future_z", "local_z"):
                panel[key][fresh] = encoded[key][:, 0]
        return panel, stack_rows(banks), records, checks, frames
    finally:
        env.close()


def collect_cumulative_training(output, checkpoint, mining_round, *, device, batch, reference, archive, replay):
    reserved = set((*TRAIN_SEEDS, *EVAL_SEEDS, *CALIBRATE, *HOLDOUT, *range(2026111900, 2026111932)))
    if len(LIVE_SEEDS) != 128 or len(set(LIVE_SEEDS)) != 128 or reserved.intersection(LIVE_SEEDS):
        raise ValueError("Cumulative training requires the same 128 distinct training layouts")
    output.mkdir(parents=True, exist_ok=True)
    if (output / LIVE_PANEL).exists():
        raise FileExistsError(output / LIVE_PANEL)
    if mining_round == 1:
        control, control_false = read_live_archive(reference)
        shared, shared_false = read_live_archive(archive)
        fixed = control["selection_role"] != LIVE_BAD_ROLE
        for key, value in control.items():
            np.testing.assert_array_equal(value if key in LIVE_COMMON_KEYS else value[fixed],
                                          shared[key] if key in LIVE_COMMON_KEYS else shared[key][fixed])
        if reference.resolve() == archive.resolve():
            raise ValueError("The shared and prior Arm 2 archives must be separate panels")
        banks = {"-2": str(archive.with_name(LIVE_BANK).resolve()), "-1": str(reference.with_name(LIVE_BANK).resolve())}
    else:
        previous = json.loads(replay.with_name(LIVE_SUMMARY).read_text(encoding="utf-8"))
        if previous["mining_round"] != mining_round - 1:
            raise ValueError("Cumulative replay must come from the immediately preceding mining round")
        if (previous.get("mining_rule") != CUMULATIVE_MINING_RULE
                or previous.get("ranking_rule") != CUMULATIVE_RANKING_RULE):
            raise ValueError("Cumulative replay uses a different mining or ranking rule")
        banks = previous["banks_by_mining_round"]
        with np.load(replay, allow_pickle=False) as saved:
            control = dict(saved)
        if control.pop("cumulative_round").item() != mining_round - 1:
            raise ValueError("Cumulative replay panel and mining summary disagree on its round")
        np.testing.assert_array_equal(control["seed"], LIVE_SEEDS)
        if control["action"].shape[:3] != (len(LIVE_SEEDS), 4, 5 + 2 * (mining_round - 1)):
            raise ValueError("Cumulative replay has the wrong number of candidate slots")
    banks[str(mining_round)] = str((output / LIVE_BANK).resolve())
    model = load_trained(checkpoint, "conditioned", device)
    if model[2].forced_exposure != 1.:
        raise ValueError("Cumulative mining requires a fixed-r=1 Arm 2 checkpoint")
    decode = load_readout(device, READOUT_RUN)
    panels, rounds, records = [], [], []
    checks = frames = 0
    for i, seed in enumerate(LIVE_SEEDS):
        row = (cumulative_seed_layout(i, control, shared, control_false, shared_false) if mining_round == 1
               else {key: value[i] for key, value in control.items()})
        panel, bank, detail, n_checks, n_frames = collect_cumulative_layout(seed, row, model, decode, batch, mining_round)
        panels.append(panel)
        rounds.append(bank)
        records.extend(detail)
        checks += n_checks
        frames += n_frames
        if i == 0 or (i + 1) % 8 == 0:
            print(f"cumulative round {mining_round}: {i + 1}/{len(LIVE_SEEDS)} layouts, {frames} RGB", flush=True)
    panel, bank = stack_rows(panels), stack_rows(rounds)
    valid = panel["candidate_valid"]
    if not (np.isfinite(panel["future_z"][valid]).all() and np.isfinite(panel["local_z"][panel["local_valid"]]).all()):
        raise FloatingPointError("Nonfinite cumulative supervision in a valid candidate")
    if (panel["local_valid"][~valid].any() or panel["mining_eligible"][~valid].any()
            or panel["rank_eligible"][~valid].any()):
        raise AssertionError("A padded candidate has a valid supervision source")
    current = panel["mining_round"] == mining_round
    layout, root, slot = np.nonzero(current)
    index = panel["bank_index"][layout, root, slot]
    for key in ("action", "future_xy", "false_goal", "mining_eligible", "rank_eligible", "actual_cost", "actual_cost_gap"):
        np.testing.assert_array_equal(panel[key][current], bank[key][layout, root - 2, index])
    gaps = panel["actual_cost_gap"][current]
    ranked = panel["rank_eligible"][current]
    mineable = panel["mining_eligible"][current]
    selected = np.arange(len(index)), panel["candidate_mode"][current]
    if not mineable[selected].all():
        raise AssertionError("A current negative was mined in an ineligible mode")
    distinct_command = (panel["action"][current][:, 0] != panel["action"][layout, root, 3, 0]).any(axis=(-2, -1))
    distinct_endpoint = (panel["future_xy"][current][:, :, 0, -1]
                         != panel["future_xy"][layout, root, 3, :, 0, -1]).any(axis=-1)
    np.testing.assert_array_equal(ranked, mineable & (gaps > 5.) & distinct_command[:, None] & distinct_endpoint)
    summary = dict(
        mining_round=mining_round, checkpoint=str((checkpoint / "conditioned.pt").resolve()), forced_exposure=1.,
        layout_seeds=list(LIVE_SEEDS), root_names=ROOT_NAMES, model_cem_budget_per_navigation_root=[ITERATIONS, SAMPLES],
        mining_rule=CUMULATIVE_MINING_RULE, ranking_rule=CUMULATIVE_RANKING_RULE,
        banks_by_mining_round=banks, candidate_kinds=CUMULATIVE_KINDS, navigation_roots=len(records),
        candidates_by_mining_round={source: int((panel["mining_round"][valid] == int(source)).sum()) for source in ("0", *banks)},
        selected_false_goals=int((panel["candidate_kind"][current] == 1).sum()),
        selected_interior_blocked=int((panel["candidate_kind"][current] == 2).sum()),
        selected_per_root=np.bincount(current[:, 2:].sum(axis=-1).ravel(), minlength=3).tolist(),
        repeated_controls=sum(candidate["repeated_control"] for row in records for candidate in row["candidates"]),
        minimum_selected_actual_cost_gap=float(gaps[selected].min()) if len(index) else None,
        rank_eligible_selected_branches=ranked.sum(axis=0).tolist(),
        minimum_selected_rank_eligible_actual_cost_gap=float(gaps[ranked].min()) if ranked.any() else None,
        canonical_trajectory_checks=checks, encoded_rgb_positions=frames,
        shapes={key: list(value.shape) for key, value in panel.items()}, rows=records,
    )
    save_npz(output / LIVE_BANK, dict(seed=panel["seed"], cumulative_round=np.int8(mining_round), **bank))
    (output / LIVE_SUMMARY).write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_npz(output / LIVE_PANEL, dict(cumulative_round=np.int8(mining_round), **panel))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--live-checkpoint", type=Path, help="Mine the 128-layout live panel using this checkpoint directory")
    parser.add_argument("--reference", type=Path, help="Round 1 live panel; freeze all anchors and refresh only the live bad slots")
    parser.add_argument("--cumulative-round", type=int, choices=(1, 2, 3), help="Append negatives to fixed-r=1 Arm 2 training replay")
    parser.add_argument("--archive", type=Path, help="Shared original live panel to retain in cumulative round 1")
    parser.add_argument("--replay", type=Path, help="Immediately preceding cumulative live panel")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.cumulative_round is not None:
        if args.live_checkpoint is None or args.output is None:
            parser.error("Cumulative mining requires --live-checkpoint and --output")
        if args.cumulative_round == 1:
            if args.reference is None or args.archive is None or args.replay is not None:
                parser.error("Cumulative round 1 requires --reference and --archive")
        elif args.replay is None or args.reference is not None or args.archive is not None:
            parser.error("Subsequent cumulative rounds require --replay only")
        collect_cumulative_training(args.output, args.live_checkpoint, args.cumulative_round, device=args.device,
                                    batch=args.batch, reference=args.reference, archive=args.archive, replay=args.replay)
        return
    if args.archive is not None or args.replay is not None:
        parser.error("--archive and --replay require --cumulative-round")
    if args.live_checkpoint is not None:
        collect_live_training(args.output or LIVE_OUTPUT, args.live_checkpoint, device=args.device,
                              batch=args.batch, reference=args.reference)
        return
    if args.reference is not None:
        parser.error("--reference requires --live-checkpoint")
    args.output = args.output or OUTPUT
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = PREFIX_RUN / "prefix"
    model = load_trained(checkpoint, "conditioned", args.device)
    decode = load_readout(args.device, READOUT_RUN)
    with np.load(PREFIX_RUN / "data_train_coverage.npz", allow_pickle=False) as f:
        control = dict(f)
    with np.load(PREFIX_RUN / "data_train_bank.npz", allow_pickle=False) as f:
        original = {key: f[key] for key in ("seed", "action", "future_xy")}
    np.testing.assert_array_equal(control["seed"], TRAIN_SEEDS)
    np.testing.assert_array_equal(original["seed"], TRAIN_SEEDS)
    panels, banks, records = [], [], []
    for i, seed in enumerate(TRAIN_SEEDS):
        panel, bank, rows = collect_layout(seed, {key: value[i].copy() for key, value in control.items()},
                                           {key: value[i] for key, value in original.items()}, model, decode, args.batch)
        panels.append(panel)
        banks.append(bank)
        records.extend(rows)
        print(f"{i + 1}/{len(TRAIN_SEEDS)} seed {seed}: {sum(row['fallback'] is None for row in rows)}/2 pairs", flush=True)
    paired = stack_rows(panels)
    for key, value in control.items():
        np.testing.assert_array_equal(paired[key] if key == "seed" else paired[key][:, :2],
                                      value if key == "seed" else value[:, :2])
    for key in ("root_xy", "root_z", "root_action"):
        np.testing.assert_array_equal(paired[key], control[key])
    changed = (paired["action"] != control["action"]).any(axis=(3, 4, 5))
    matched = [row for row in records if row["fallback"] is None]
    summary = {
        "checkpoint": str(checkpoint / "conditioned.pt"), "control": str(PREFIX_RUN / "data_train_coverage.npz"),
        "layout_seeds": list(TRAIN_SEEDS), "root_names": ROOT_NAMES, "navigation_roots": len(records),
        "live_budget_per_navigation_root": [ITERATIONS, SAMPLES], "candidate_pairs": len(matched),
        "false_goal_pairs": sum(row["false_goal_room"] for row in matched),
        "blocked_pairs": sum(row["blocked"] for row in matched),
        "physical_cem_searches": sum(row["physical_cem_searched"] for row in records),
        "legal_witness_sources": {source: sum(row["legal_source"] == source for row in matched)
                                  for source in ("visible_route", "archived_physical_cem", "physical_cem")},
        "fallbacks": {cause: sum(row["fallback"] == cause for row in records)
                      for cause in ("no_bad_live_query", "no_clean_lower_cost_first_command")},
        "changed_slots_by_root": changed.sum(axis=(0, 2)).tolist(),
        "actual_canonical_env_trajectory_checks": sum(row["checks"] for row in records),
        "unique_encoded_rgb_frames": sum(row["rgb_frames"] for row in records),
        "selection_role": {"6": "prefix_model_bad_query", "7": "same_mode_physical_counterpart"},
        "rows": records,
    }
    save_npz(args.output / "data_train_paired.npz", paired)
    save_npz(args.output / "data_train_refresh_bank.npz", dict(seed=paired["seed"], **stack_rows(banks)))
    (args.output / "data_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
