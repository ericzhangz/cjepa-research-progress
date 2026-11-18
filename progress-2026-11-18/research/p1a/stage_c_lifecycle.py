"""Record clean paired histories and executed H12 menus for Stage C §49.4."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from .native_topology_condition import load_host
from .stage_a_fourroom import (
    ACTION_BLOCK, CENTER, DOORS, GOAL_RADIUS, HORIZON, LOCKS, SPEED, OfficialFourRoom, OpenPhysics,
    Task, actual_crossings, candidate_bank, fixed_tasks, lifecycle_episode, route_controls,
    validate_lifecycle_pair, write_json,
)
from .stage_a_fit_gru import encode_frames
from .stage_c_data import ROOT, command


OUT = ROOT / 'runs/stage_c_clean_20261117'
FIT_SEED = 2026113400
FIT_CASES = 8


def encode(encoder, rgb):
    # The released GPU encoder can round differently in a short final batch.
    # Keep identical RGB inputs bitwise matched by using the same batch shape.
    flat = rgb.reshape(-1, *rgb.shape[-3:])
    padded = np.pad(flat, ((0, -len(flat) % 32), (0, 0), (0, 0), (0, 0)), mode='edge')
    latent = encode_frames(encoder, padded).numpy()[:len(flat)]
    return latent.reshape(*rgb.shape[:-3], latent.shape[-1])


def restore_root(env, seed, goal, lock, xy, rgb):
    # In the clean simulator the distractor never moves, and the author collision
    # update is Markov in ego position and the active door constraint.
    env.ego_lock = lock
    env.reset(seed=seed, options={'state': xy, 'target_state': goal})
    np.testing.assert_array_equal(env.xy, xy)
    np.testing.assert_array_equal(env.render(), rgb)


def execute(env, physics, controls):
    expected = physics.rollout(env.xy, controls[None], env.ego_lock)[0]
    paths, frames = [], []
    for raw in controls:
        paths.append(env.execute(raw))
        frames.append(env.render())
    actual = np.concatenate(paths)
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=0)
    return np.stack(frames), actual[ACTION_BLOCK - 1::ACTION_BLOCK]


def approach(start, mouth):
    xy = start.copy()
    controls = []
    for _ in range(2 * ACTION_BLOCK):
        raw = np.clip((mouth - xy) / SPEED, -1, 1)
        xy += raw * SPEED
        controls.append(raw)
    np.testing.assert_array_equal(xy, mouth)
    return np.stack(controls).reshape(2, ACTION_BLOCK, 2)


def extend_bridge(env, physics, task, goal, lock, fork_rgb, start, mouth, controls):
    restore_root(env, task.seed, goal, lock, start, fork_rgb)
    frames, xy = execute(env, physics, controls)
    np.testing.assert_array_equal(xy[-1], mouth)
    return frames


def quiet_menu(first, second):
    programs = [np.tile(np.stack((block, -block)), (HORIZON // 2, 1, 1)) for block in (first, second)]
    return np.broadcast_to(np.stack(programs), (2, 2, HORIZON, ACTION_BLOCK, 2)).copy()


def record_group(env, physics, encoder, task, goal, start, mouth, locks, prefixes, read_t, menu_a,
                 *, directed, common):
    a, b = prefixes
    for t in read_t:
        np.testing.assert_array_equal(a[0][t - 2:t + 1], b[0][t - 2:t + 1])
        np.testing.assert_array_equal(a[1][t - 2:t], b[1][t - 2:t])

    choices = []
    for t, xy, programs in zip(read_t, (start, mouth), menu_a, strict=True):
        targets = []
        for lock, (rgb, _) in zip(locks, prefixes, strict=True):
            futures = []
            for program in programs:
                restore_root(env, task.seed, goal, lock, xy, rgb[t])
                future, _ = execute(env, physics, program)
                futures.append(future)
            targets.append(np.stack(futures))
        left, right = targets
        if common:
            np.testing.assert_array_equal(left, right)
        elif not np.any(left != right):
            raise AssertionError('The real H12 menu must contain a visible condition effect')
        choices.append(np.stack(targets))

    prefix_a = np.stack((a[1], b[1]))
    prefix_z = encode(encoder, np.stack((a[0], b[0])))
    for t in read_t:
        np.testing.assert_array_equal(prefix_z[0, t - 2:t + 1], prefix_z[1, t - 2:t + 1])
    target_z = encode(encoder, np.stack(choices))
    if common:
        np.testing.assert_array_equal(target_z[:, 0], target_z[:, 1])
    elif not np.all(np.linalg.norm((target_z[:, 0] - target_z[:, 1]).reshape(2, -1), axis=-1) > 0):
        raise AssertionError('Each menu source pair must separate in the clean latent representation')
    return {'prefix_z': prefix_z, 'prefix_a': prefix_a, 'read_t': np.asarray(read_t),
            'menu_a': menu_a, 'target_z': target_z, 'directed': np.bool_(directed)}


def collect_case(env, physics, encoder, i, seed):
    # Cover all four approach directions with the same AAB and ABB constructions.
    # Mild changes to the fork and goal keep the independent Stage A query roots out of fitting.
    template = [task for task in fixed_tasks() if task.family == 'two_doors'][2 * (i % 4)]
    start = np.array(template.start, dtype=np.float32)
    goal = np.array(template.goal, dtype=np.float32)
    start[0] += (-2.5, 2.5)[(i // 4) % 2]
    goal[1] += (2.5, -2.5)[(i // 4) % 2]
    task = Task(template.name, template.family, tuple(start), tuple(goal), seed, ())
    door = 'top' if start[1] < CENTER else 'bottom'
    sign = 1 if start[0] < CENTER else -1
    mouth = np.array((CENTER - sign * 11.5, DOORS[door]), dtype=np.float32)
    schedule = (0, 0, 1) if (i // 4) % 2 == 0 else (0, 1, 1)
    initial = (i // 2) % 2
    ordered_initial = (initial, 1 - initial)
    pairs = [lifecycle_episode(env, physics, door, 0, sign, mouth, goal, seed, mode,
                               mode_schedule=schedule, anchor=start) for mode in ordered_initial]
    timeline, trace = validate_lifecycle_pair(pairs)
    away = command(0, -sign)
    tangent = command(1, 1 if start[1] < CENTER else -1)
    bridge = approach(start, mouth)
    routes = np.stack([np.stack([route_controls(task, xy, direction, HORIZON) for direction in (1, -1)])
                       for xy in (start, mouth)])
    probes = timeline['episode_a']['probes']
    source_targets = []
    rows = []

    # Initial cover sees both physical outcomes while the factual window has no evidence.
    initial_fork = timeline['episode_a']['initial_fork_block']
    first_probe = probes[0]['pre_block']
    initial_read = (initial_fork, first_probe)
    initial_locks = tuple(LOCKS[mode] for mode in ordered_initial)
    prefix = tuple((rgb[:first_probe + 1], a[:first_probe])
                   for rgb, a in zip(trace['rgb'], trace['raw_actions'], strict=True))
    rows.append(record_group(env, physics, encoder, task, goal, start, mouth, initial_locks,
                             prefix, initial_read, routes, directed=False, common=False))
    # Both true conditions supply the same arrivals on each of these neutral programs.
    rows.append(record_group(env, physics, encoder, task, goal, start, mouth, initial_locks,
                             prefix, initial_read, quiet_menu(np.zeros_like(away), away),
                             directed=False, common=True))
    rows.append(record_group(env, physics, encoder, task, goal, start, mouth, initial_locks,
                             prefix, initial_read, quiet_menu(tangent, .5 * (away + tangent)),
                             directed=False, common=True))

    for probe, relative in zip(probes, schedule, strict=True):
        pre, fork = probe['pre_block'], probe['fork_block']
        actual = trace['rgb'][:, pre + 1]
        if np.array_equal(actual[0], actual[1]):
            raise AssertionError('The two actual first probe arrivals must visibly separate')
        source_targets.append(encode(encoder, actual))
        locks = tuple(LOCKS[(mode + relative) % 2] for mode in ordered_initial)
        prefix = []
        for lock, rgb, controls in zip(locks, trace['rgb'], trace['raw_actions'], strict=True):
            continuation = extend_bridge(env, physics, task, goal, lock, rgb[fork], start, mouth, bridge)
            prefix.append((np.concatenate((rgb[:fork + 1], continuation)),
                           np.concatenate((controls[:fork], bridge))))
        rows.append(record_group(env, physics, encoder, task, goal, start, mouth, locks, prefix,
                                 (fork, fork + len(bridge)), routes, directed=True, common=False))

    return rows, source_targets, {'scenario': i, 'seed': seed, 'template': template.name,
                                  'relative_order': list(schedule), 'fixed_read_roots': [start.tolist(), mouth.tolist()]}


def collate(rows, sources):
    frames = max(row['prefix_z'].shape[1] for row in rows)
    for row in rows:
        n = row['prefix_z'].shape[1]
        row['prefix_z'] = np.pad(row['prefix_z'], ((0, 0), (0, frames - n), (0, 0)))
        row['prefix_a'] = np.pad(row['prefix_a'], ((0, 0), (0, frames - n), (0, 0), (0, 0)))
    data = {f'fit_{key}': np.stack([row[key] for row in rows]) for key in rows[0]}
    data['source_target_z'] = np.stack(sources)
    data['source_delta'] = data['source_target_z'][:, 1] - data['source_target_z'][:, 0]
    if not np.all(np.linalg.norm(data['source_delta'], axis=-1) > 0):
        raise AssertionError('The H1 scale requires strictly separated actual probe targets')
    if np.count_nonzero(data['fit_directed']) * 2 != len(rows):
        raise AssertionError('Fitting must receive equal directed and cover groups')
    return data


def candidate_geometry(physics, xy, programs, goal):
    paths = np.stack([physics.rollout(xy, programs, lock) for lock in LOCKS])
    previous = np.concatenate((np.broadcast_to(xy, (*paths.shape[:2], 1, 2)), paths[..., :-1, :]), axis=-2)
    free = np.clip(previous + SPEED * programs.reshape(len(programs), -1, 2), *physics.bounds)
    collision = np.any(np.abs(paths - free) > 1e-5, axis=(-1, -2))
    common = np.all(paths[0] == paths[1], axis=(-1, -2))
    success = np.linalg.norm(paths[..., -1, :] - goal, axis=-1) <= GOAL_RADIUS
    return {'common': common, 'common_blocked': common & collision.any(axis=0) & ~success.any(axis=0),
            'success': success}


def additional_candidates(task, xy, neutral, physics):
    if neutral:
        away = command(0, -1 if xy[0] < CENTER else 1)
        tangent = command(1, 1 if xy[1] < CENTER else -1)
        blocks = (.25 * away, .75 * away, .25 * tangent, .75 * tangent,
                  .25 * (away + tangent), .75 * (away + tangent))
        programs = np.stack([np.tile(np.stack((block, -block)), (HORIZON // 2, 1, 1))
                             for block in blocks])
        return programs, None
    bank = candidate_bank(task, xy, 0)
    random = np.random.default_rng(task.seed)
    half = len(bank) // 2
    geometry = candidate_geometry(physics, xy, bank, np.asarray(task.goal))
    priority = random.permutation(np.r_[1:half, half + 1:len(bank)])
    chosen = []
    for eligible in (geometry['common_blocked'], *geometry['success']):
        candidates = priority[eligible[priority] & ~np.isin(priority, chosen)]
        if not len(candidates):
            raise ValueError('The fitting bank cannot meet the frozen common-blocked / two-success coverage quota')
        chosen.append(int(candidates[0]))
    for base in (0, half):
        count = np.count_nonzero((np.asarray(chosen) >= base) & (np.asarray(chosen) < base + half))
        candidates = priority[(priority >= base) & (priority < base + half) & ~np.isin(priority, chosen)]
        chosen.extend(candidates[:3 - count].tolist())
    indices = np.asarray(chosen)
    if len(indices) != 6:
        raise AssertionError('Coverage selection must produce exactly six new candidates')
    return bank[indices], indices.tolist()


def augment_coverage(env, physics, encoder, source, output):
    started = time.perf_counter()
    with np.load(source / 'lifecycle.npz', allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads((source / 'lifecycle_cases.json').read_text(encoding='utf-8'))
    expected_order = ['initial_cover', 'quiet_outward_cover', 'quiet_lateral_cover',
                      'first_probe', 'second_probe', 'third_probe']
    if metadata['group_order_per_case'] != expected_order:
        raise ValueError('The coverage extension requires the original six-group clean lifecycle')
    old_menu, old_target = original['fit_menu_a'], original['fit_target_z']
    cases = metadata['fit_cases']
    if old_menu.shape[:3] != (6 * len(cases), 2, 2) or old_target.shape[3] != 2:
        raise ValueError('The source archive must contain exactly two candidates at each original fit root')
    if source.resolve() == output.resolve():
        raise ValueError('Coverage data must be written to a separate directory')
    output.mkdir(parents=True, exist_ok=True)
    selection_contract = ('At each varied fitting root select one identical blocked trajectory, one successful '
                          'non-nominal program per physical condition, then seeded fill to three candidates '
                          'from each bank family. Use only exact fitting physics and the original goal radius.')
    write_json(output / 'coverage_protocol.json', {
        'source_archive': str((source / 'lifecycle.npz').resolve()),
        'preserved_candidates': 2, 'appended_candidates': 6,
        'selection_contract': selection_contract, 'goal_radius': GOAL_RADIUS,
        'shared_blocked_definition': 'Identical full physical paths in both conditions, at least one wall collision, and neither endpoint within the goal radius.',
        'neutral_definition': 'Six conservative out-and-back programs with all actual RGB arrivals equal in both conditions.',
        'candidate_source': 'Full original candidate_bank at each fitting root, excluding the two retained nominal indices.',
    })
    templates = {task.name: task for task in fixed_tasks()}
    additions, observations, audit_rows = [], [], []
    root_error = 0.
    for case in cases:
        i, seed = case['scenario'], case['seed']
        template = templates[case['template']]
        roots = np.asarray(case['fixed_read_roots'], dtype=np.float32)
        goal = np.asarray(template.goal, dtype=np.float32)
        goal[1] += (2.5, -2.5)[(i // 4) % 2]
        task = Task(template.name, template.family, tuple(roots[0]), tuple(goal), seed, ())
        initial = (i // 2) % 2
        for local in range(6):
            group = 6 * i + local
            relative = 0 if local < 3 else case['relative_order'][local - 3]
            locks = tuple(LOCKS[(mode + relative) % 2] for mode in (initial, 1 - initial))
            neutral = local in (1, 2)
            programs, targets = [], []
            for bridge, xy in enumerate(roots):
                menu, indices = additional_candidates(task, xy, neutral, physics)
                combined = np.concatenate((old_menu[group, bridge], menu))
                if len(np.unique(combined.reshape(8, -1), axis=0)) != 8:
                    raise AssertionError('The augmented menu must contain eight distinct programs')
                actual = []
                for history, lock in enumerate(locks):
                    env.ego_lock = lock
                    env.reset(seed=seed, options={'state': xy, 'target_state': goal})
                    root_rgb = env.render()
                    root_z = encode(encoder, root_rgb[None])[0]
                    read = original['fit_read_t'][group, bridge]
                    expected_z = original['fit_prefix_z'][group, history, read]
                    root_error = max(root_error, float(np.abs(root_z - expected_z).max()))
                    np.testing.assert_array_equal(root_z, expected_z)
                    futures = []
                    for program in menu:
                        restore_root(env, seed, goal, lock, xy, root_rgb)
                        future, _ = execute(env, physics, program)
                        futures.append(future)
                    actual.append(np.stack(futures))
                if neutral:
                    np.testing.assert_array_equal(actual[0], actual[1])
                z = encode(encoder, np.stack(actual))
                common = np.all(z[0] == z[1], axis=(-1, -2))
                if neutral and not common.all():
                    raise AssertionError('Every appended neutral candidate must have the same actual targets')
                programs.append(menu)
                targets.append(z)
                geometry = candidate_geometry(physics, xy, menu, goal)
                audit_rows.append({'group': group, 'bridge': bridge, 'neutral': neutral,
                                   'candidate_bank_indices': indices,
                                   'new_common_blocked_slots': geometry['common_blocked'].tolist(),
                                   'new_goal_success_slots_by_physical_condition': geometry['success'].tolist(),
                                   'new_common_target_slots': common.tolist(),
                                   'new_target_difference_rms': np.sqrt(np.mean((z[0] - z[1])**2,
                                                                             axis=(-1, -2))).tolist()})
            additions.append(np.stack(programs))
            observations.append(np.stack(targets))
        print(f"Augmented case {i + 1}/{len(cases)} with six additional programs at every fit root", flush=True)

    arrays = original.copy()
    arrays['fit_menu_a'] = np.concatenate((old_menu, np.stack(additions)), axis=2)
    arrays['fit_target_z'] = np.concatenate((old_target, np.stack(observations)), axis=3)
    np.testing.assert_array_equal(arrays['fit_menu_a'][:, :, :2], old_menu)
    np.testing.assert_array_equal(arrays['fit_target_z'][:, :, :, :2], old_target)
    preserved = [key for key in original if key not in ('fit_menu_a', 'fit_target_z')]
    for key in preserved:
        np.testing.assert_array_equal(arrays[key], original[key])
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise AssertionError('All augmented archive arrays must be finite')
    source_strength = np.square(original['source_delta']).sum(-1) * .5
    directed = original['fit_directed']
    old_difference = old_target[directed, :, 1] - old_target[directed, :, 0]
    output.mkdir(parents=True, exist_ok=True)
    temp = output / 'lifecycle.tmp.npz'
    np.savez_compressed(temp, **arrays)
    os.replace(temp, output / 'lifecycle.npz')
    metadata['shapes'] = {key: list(value.shape) for key, value in arrays.items()}
    metadata['coverage_source'] = str(source.resolve())
    metadata['coverage_extension'] = 'Original slots 0 and 1 retained. Slots 2 through 7 are executed fit-only targets.'
    write_json(output / 'lifecycle_cases.json', metadata)
    write_json(output / 'coverage_data_audit.json', {
        'source_archive': str((source / 'lifecycle.npz').resolve()),
        'fitting_groups': len(directed), 'directed_groups': int(directed.sum()),
        'fit_read_roots': int(original['fit_read_t'].size), 'candidate_count': 8,
        'preserved_arrays': preserved, 'original_candidate_slots_exact': True,
        'reconstructed_root_latent_max_abs_error': root_error,
        'source_s2_obs': float(np.quantile(source_strength, .10) / 3),
        'source_s2_bundle': float(np.square(old_difference).sum(-1).mean()),
        'selection_contract': selection_contract,
        'actual_added_H12_programs': len(directed) * 2 * 2 * 6,
        'actual_added_raw_steps': len(directed) * 2 * 2 * 6 * HORIZON * ACTION_BLOCK,
        'wall_seconds': time.perf_counter() - started, 'roots': audit_rows,
    })
    print(f"Wrote {output / 'lifecycle.npz'}", flush=True)


def displacement_controls(delta):
    remaining = np.asarray(delta, dtype=np.float32).copy()
    controls = []
    for _ in range(int(np.ceil(np.abs(remaining).max() / SPEED))):
        action = np.clip(remaining / SPEED, -1., 1.)
        controls.append(action)
        remaining -= SPEED * action
    np.testing.assert_allclose(remaining, 0., atol=1e-5, rtol=0)
    return np.asarray(controls, dtype=np.float32).reshape(-1, 2)


def blocked_controls(controls):
    raw = np.concatenate(controls)
    raw = np.pad(raw, ((0, -len(raw) % ACTION_BLOCK), (0, 0)))
    return raw.reshape(-1, ACTION_BLOCK, 2)


def bridge_probe(anchor, physics):
    northwest = bool(np.all(anchor < CENTER))
    southeast = bool(np.all(anchor > CENTER))
    if not (northwest or southeast):
        raise ValueError('Bridge probes require a northwest or southeast anchor')
    direction = 1. if northwest else -1.
    mouth = np.array((CENTER - direction * 11.5, 56. if northwest else 168.), dtype=np.float32)
    approach = blocked_controls((displacement_controls((mouth[0] - anchor[0], 0.)),
                                 displacement_controls((0., mouth[1] - anchor[1]))))
    probe = np.stack((command(0, direction), command(0, direction)))
    boundary = physics.bounds[0 if northwest else 1]
    recovery = blocked_controls((np.tile((-direction, 0.), (44, 1)),
                                  displacement_controls((anchor[0] - boundary, 0.)),
                                  displacement_controls((0., anchor[1] - mouth[1]))))
    neutral = np.stack((command(0, -direction), command(0, direction)))
    return np.concatenate((approach, probe, recovery, neutral)), len(approach)


def connector_controls(source, destination, physics):
    if np.all(source > CENTER) and np.all(destination < CENTER):
        return -connector_controls(2 * CENTER - source, 2 * CENTER - destination, physics)
    if not (np.all(source < CENTER) and np.all(destination > CENTER)):
        raise ValueError('The tested connector joins northwest and southeast anchors')
    upper = physics.bounds[1]
    return blocked_controls((
        displacement_controls((0., 56. - source[1])),
        np.tile((1., 0.), (int(np.ceil((upper - source[0]) / SPEED)), 1)),
        displacement_controls((-38.5, 0.)),
        np.tile((0., 1.), (int(np.ceil((upper - 56.) / SPEED)), 1)),
        displacement_controls((0., -38.5)),
        np.tile((1., 0.), (int(np.ceil((upper - 64.5) / SPEED)), 1)),
        displacement_controls((destination[0] - upper, 0.)),
        displacement_controls((0., destination[1] - 168.)),
    ))


def collect_bridge(env, physics, encoder, i):
    northwest = np.array(((77.5, 80.), (82.5, 80.), (80., 77.5), (80., 82.5))[i % 4], dtype=np.float32)
    southeast = np.array(((144., 146.5), (144., 141.5), (146.5, 144.), (141.5, 144.))[i % 4], dtype=np.float32)
    roots = (northwest, southeast) if i < 4 else (southeast, northwest)
    seed = 2026114000 + i
    near, near_probe = bridge_probe(roots[0], physics)
    transfer = connector_controls(*roots, physics)
    far, far_probe = bridge_probe(roots[1], physics)
    controls = np.concatenate((near, transfer, far))
    reads = (len(near), len(controls))
    probe_roots = (near_probe, len(near) + len(transfer) + far_probe)
    locks = LOCKS if i % 2 == 0 else LOCKS[::-1]
    prefixes, paths = [], []
    for lock in locks:
        env.ego_lock = lock
        env.reset(seed=seed, options={'state': roots[0], 'target_state': roots[1]})
        first = env.render()
        arrived, xy = execute(env, physics, controls)
        for t, expected in zip(reads, roots, strict=True):
            np.testing.assert_array_equal(xy[t - 1], expected)
        prefixes.append((np.concatenate((first[None], arrived)), controls.copy()))
        raw_path = physics.rollout(roots[0], controls[None], lock)[0]
        if physics.wall_penetration(raw_path).any():
            raise AssertionError('A factual connector entered a solid wall')
        paths.append(raw_path)
    for pre in probe_roots:
        np.testing.assert_array_equal(prefixes[0][0][pre - 2:pre + 1], prefixes[1][0][pre - 2:pre + 1])
        if np.array_equal(prefixes[0][0][pre + 1], prefixes[1][0][pre + 1]):
            raise AssertionError('Each bridge probe must produce a visible actual condition effect')
    tasks = [Task(f'bridge_{i}_{j}', 'two_doors', tuple(root), tuple(roots[1 - j]), seed, ())
             for j, root in enumerate(roots)]
    menus, menu_audits = [], []
    for task, xy in zip(tasks, roots, strict=True):
        nominal = np.stack([route_controls(task, xy, direction, HORIZON) for direction in (1, -1)])
        extra, indices = additional_candidates(task, xy, False, physics)
        menu = np.concatenate((nominal, extra))
        if len(np.unique(menu.reshape(8, -1), axis=0)) != 8:
            raise AssertionError('Each bridge root requires eight distinct controls')
        geometry = candidate_geometry(physics, xy, menu, np.asarray(task.goal))
        menus.append(menu)
        menu_audits.append({'goal': np.asarray(task.goal).tolist(), 'extra_candidate_indices': indices,
                            'common_blocked_slots': geometry['common_blocked'].tolist(),
                            'success_slots_by_physical_condition': geometry['success'].tolist()})
    row = record_group(env, physics, encoder, tasks[0], roots[1], *roots, locks, prefixes, reads,
                       np.stack(menus), directed=True, common=False)
    return row, {
        'connector': i, 'group': 48 + i, 'seed': seed, 'direction': 'NW_to_SE' if i < 4 else 'SE_to_NW',
        'read_roots': [root.tolist() for root in roots], 'read_blocks': list(reads),
        'probe_pre_blocks': list(probe_roots), 'factual_blocks': len(controls),
        'first_probe_arrival': probe_roots[0] + 1, 'second_probe_arrival': probe_roots[1] + 1,
        'transfer_blocks': len(transfer), 'factual_resets_per_history': 1,
        'physical_row_order': list(locks), 'identical_controls_both_histories': True,
        'matched_read_rgb_and_raw_controls': True, 'solid_wall_endpoints': 0,
        'actual_crossings_by_history': [actual_crossings(np.vstack((roots[0], path)), physics) for path in paths],
        'menus': menu_audits,
    }


def augment_bridges(env, physics, encoder, source, output):
    started = time.perf_counter()
    with np.load(source / 'lifecycle.npz', allow_pickle=False) as archive:
        original = {key: archive[key].copy() for key in archive.files}
    if original['fit_menu_a'].shape != (48, 2, 8, HORIZON, ACTION_BLOCK, 2):
        raise ValueError('The bridge extension requires the frozen 48-group eight-candidate coverage archive')
    if source.resolve() == output.resolve():
        raise ValueError('Bridge data must be written to a separate directory')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'bridge_protocol.json', {
        'source_archive': str((source / 'lifecycle.npz').resolve()),
        'added_directed_groups': 8, 'read_roots_per_group': 2, 'candidates_per_root': 8,
        'factual_sequence': 'Near probe, blind recovery, neutral round trip, continuous connector, far probe, blind recovery, neutral round trip.',
        'factual_lock_contract': 'Both histories use identical controls and one constant physical condition throughout. Reset only once at each history start.',
        'candidate_contract': 'Reuse the coverage menu construction independently at each fitting root. Menu consequences restore their actual source positions.',
        'fit_seed_start': 2026114000,
    })
    rows, audits = [], []
    for i in range(8):
        row, audit = collect_bridge(env, physics, encoder, i)
        rows.append(row)
        audits.append(audit)
        print(f'Collected connector {i + 1}/8 with two paired actual read roots', flush=True)
    arrays = original.copy()
    frames = max(original['fit_prefix_z'].shape[2], *(row['prefix_z'].shape[1] for row in rows))
    for key in ('prefix_z', 'prefix_a'):
        name = f'fit_{key}'
        padding = frames - int(key == 'prefix_a') - original[name].shape[2]
        old_pad = [(0, 0)] * original[name].ndim
        old_pad[2] = (0, padding)
        base = np.pad(original[name], old_pad)
        added = []
        for row in rows:
            row_pad = [(0, 0)] * row[key].ndim
            row_pad[1] = (0, frames - int(key == 'prefix_a') - row[key].shape[1])
            added.append(np.pad(row[key], row_pad))
        arrays[name] = np.concatenate((base, np.stack(added)))
        np.testing.assert_array_equal(arrays[name][:48, :, :original[name].shape[2]], original[name])
    for key in ('read_t', 'menu_a', 'target_z', 'directed'):
        name = f'fit_{key}'
        arrays[name] = np.concatenate((original[name], np.stack([row[key] for row in rows])))
        np.testing.assert_array_equal(arrays[name][:48], original[name])
    for key in ('source_target_z', 'source_delta'):
        np.testing.assert_array_equal(arrays[key], original[key])
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise AssertionError('The bridge archive must be finite')
    temp = output / 'lifecycle.tmp.npz'
    np.savez_compressed(temp, **arrays)
    os.replace(temp, output / 'lifecycle.npz')
    metadata = json.loads((source / 'lifecycle_cases.json').read_text(encoding='utf-8'))
    metadata['shapes'] = {key: list(value.shape) for key, value in arrays.items()}
    metadata['bridge_source'] = str(source.resolve())
    metadata['bridge_cases'] = audits
    write_json(output / 'lifecycle_cases.json', metadata)
    write_json(output / 'bridge_data_audit.json', {
        'source_archive': str((source / 'lifecycle.npz').resolve()), 'original_groups': 48,
        'bridge_groups': 8, 'fitting_groups': 56, 'directed_groups': int(arrays['fit_directed'].sum()),
        'original_prefix_values_exact': True, 'original_group_arrays_exact': True,
        'source_calibration_arrays_exact': True, 'connectors': audits,
        'wall_seconds': time.perf_counter() - started,
    })
    print(f"Wrote {output / 'lifecycle.npz'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--cases', type=int, default=FIT_CASES)
    parser.add_argument('--augment-from', type=Path)
    parser.add_argument('--bridge-from', type=Path)
    args = parser.parse_args()
    if args.cases <= 0:
        raise ValueError('At least one fitting case is required')
    device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = SimpleNamespace(host=load_host(device), device=device,
                              mean=torch.tensor((.485, .456, .406), device=device)[:, None, None],
                              std=torch.tensor((.229, .224, .225), device=device)[:, None, None])
    env = OfficialFourRoom(fixed_distractor=True)
    physics = OpenPhysics(env)
    if args.bridge_from is not None:
        try:
            augment_bridges(env, physics, encoder, args.bridge_from, args.output)
        finally:
            env.close()
        return
    if args.augment_from is not None:
        try:
            augment_coverage(env, physics, encoder, args.augment_from, args.output)
        finally:
            env.close()
        return
    rows, sources, audits = [], [], []
    try:
        for i in range(args.cases):
            groups, probe_targets, audit = collect_case(env, physics, encoder, i, FIT_SEED + i)
            rows.extend(groups)
            sources.extend(probe_targets)
            audits.append(audit)
            print(f'Case {i + 1}/{args.cases} recorded with {len(groups)} paired groups', flush=True)
    finally:
        env.close()

    arrays = collate(rows, sources)
    args.output.mkdir(parents=True, exist_ok=True)
    temp = args.output / 'lifecycle.tmp.npz'
    np.savez_compressed(temp, **arrays)
    os.replace(temp, args.output / 'lifecycle.npz')
    write_json(args.output / 'lifecycle_cases.json', {
        'source': 'Executed FourRoom RGB with one fixed visible distractor. Frozen LeWM encodes each arrival.',
        'fit_cases': audits, 'shapes': {key: list(value.shape) for key, value in arrays.items()},
        'group_order_per_case': ['initial_cover', 'quiet_outward_cover', 'quiet_lateral_cover',
                                 'first_probe', 'second_probe', 'third_probe'],
        'physical_coordinates': 'Private recorder audit. No coordinates or lock labels in the numeric archive.',
        'independent_test': 'The fixed Stage A clean ABA/BAB lifecycle is held separately.',
    })
    print(f"Wrote {args.output / 'lifecycle.npz'}", flush=True)


if __name__ == '__main__':
    main()
