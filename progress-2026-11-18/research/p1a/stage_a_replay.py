"""Replay the Stage A FourRoom factual histories with four budget-matched forecasters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from .native_response_relations_fit import native_rollout
from .native_topology_condition import action_embeddings
from .native_topology_writein_attribution import first_block_groups, regret_parts
from .stage_a_baselines import GRUDynamics, OnlineAdapter, predictor_parameter_count
from .stage_a_fourroom import (
    ACTION_BLOCK, LOCKS, Native, OfficialFourRoom, OpenPhysics, ROOT, decision_bank, fixed_tasks, latent_goal_cost,
    write_json,
)


DIRECTORY = ROOT / 'runs/stage_a_fourroom_20261117'
GRU_CHECKPOINT = ROOT / 'runs/stage_a_20261117/gru.pt'
ARMS = ('frozen', 'adajepa', 'optta', 'gru')
FORECAST_BATCH = 96
REPLAY_SEED = 20261117


@dataclass(frozen=True)
class Bank:
    actions: torch.Tensor
    physical_costs: np.ndarray
    first_groups: np.ndarray
    group_costs: np.ndarray
    comparable_modes: tuple[int, ...]


def require_qualification(directory):
    """Fail before fitting or running the comparators if native or physical gates failed."""
    physical = json.loads((directory / 'physical.json').read_text(encoding='utf-8'))
    native = json.loads((directory / 'native.json').read_text(encoding='utf-8'))
    report = native['qualification']
    families = ('one_door', 'two_doors')
    geometry = ('solid_wall_endpoint_axes', 'false_cross_axes')
    if (native['protocol'] != physical['protocol']
            or not all(physical['qualification']['summary'][family]['passed'] for family in families)
            or not all(report['summary'][family]['passed'] for family in families)
            or not all(report['geometry'][name]['passed'] for name in geometry)
            or not all(row['matched_factual_three_frame_roots'] for row in physical['lifecycle'])
            or not all(row['distinct_routes'] and row['positive_uniform_regret'] and row['complete_two_door_routes']
                       for row in physical['forks'])):
        raise RuntimeError('FourRoom must pass the fixed Stage A native and physical gates before baseline replay')
    return physical


def _rollout(host, history, past, candidates):
    return torch.cat([native_rollout(host, history, past, chunk)
                      for chunk in candidates.split(FORECAST_BATCH, dim=1)], dim=1)


def _timed(device, fn, *args):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    value = fn(*args)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return value, time.perf_counter() - started


def _bank(host, raw, physical_costs, comparable_modes):
    indices = []
    group_costs = []
    for truth in physical_costs:
        group, minima = first_block_groups(raw, truth)
        indices.append(group)
        group_costs.append(minima)
    np.testing.assert_array_equal(*indices)
    actions = action_embeddings(host, raw)[None]
    return Bank(actions, physical_costs, group, np.stack(group_costs), comparable_modes)


def _bank_series(host, physics, task, metadata, raw_actions, raw_xy, pinned_raw, pinned_costs):
    """Replay public commands on the open proposal map and use real XY only to label outcomes."""
    fork_roots = {metadata['initial_fork_block'], *(probe['fork_block'] for probe in metadata['probes'])}
    _, modes = _labels(metadata)
    nominal = np.asarray(task.start, dtype=np.float32)
    banks = {}
    for root in range(len(raw_actions) + 1):
        if root >= 2:
            if root in fork_roots:
                np.testing.assert_array_equal(raw_xy[root * ACTION_BLOCK], task.start)
                raw, costs = pinned_raw, pinned_costs
            else:
                raw, _, costs = decision_bank(task, nominal, root, physics, raw_xy[root * ACTION_BLOCK],
                                              horizon=pinned_raw.shape[1])
            comparable = (0, 1) if root in fork_roots else (int(modes[root]),)
            banks[root] = _bank(host, raw, costs, comparable)
        if root < len(raw_actions):
            nominal = physics.rollout(nominal, raw_actions[root:root + 1][None])[0, -1]
    return banks


def _labels(metadata):
    """Associate analysis-only condition labels with observed facts and irreversible clock indices."""
    labels = [[] for _ in range(len(metadata['blocks']) + 1)]
    modes = np.full(len(labels), metadata['initial_mode'], dtype=int)
    labels[metadata['initial_fork_block']].append(('initial_fork', metadata['initial_mode']))
    for number, probe in enumerate(metadata['probes']):
        mode = probe['mode']
        prefix = f'reversal_{number}' if number else 'initial_probe'
        if probe['switch_block'] is not None:
            switch = probe['switch_block']
            modes[switch:] = mode
            labels[switch].append((f'{prefix}_unobserved', mode))
            for arrival in (1, 2):
                labels[switch + arrival].append((f'{prefix}_arrival_{arrival}', mode))
        for arrival in (1, 2):
            labels[probe['pre_block'] + arrival].append((f'{prefix}_probe_arrival_{arrival}', mode))
        labels[probe['fork_block']].append((f'{prefix}_retained_fork', mode))
    for root, block in enumerate(metadata['blocks'], 1):
        if block['event'] in ('neutral_away', 'neutral_return'):
            labels[root].append(('neutral_control', int(modes[root - 1])))
    return labels, modes


def _arrival_truth(physics, metadata, raw_xy, raw_actions):
    """Verify the actual raw transitions and mark the earliest physical mode difference."""
    _, modes = _labels(metadata)
    truth = []
    for index, commands in enumerate(raw_actions):
        start = index * ACTION_BLOCK
        paths = np.stack([physics.rollout(raw_xy[start], commands[None, None], lock)[0] for lock in LOCKS])
        raw = raw_xy[start + 1:start + ACTION_BLOCK + 1]
        np.testing.assert_array_equal(raw, paths[modes[index]])
        comparable = not physics.near < raw_xy[start, 0] < physics.far
        difference = np.flatnonzero(np.any(paths[0] != paths[1], axis=-1))
        if metadata['blocks'][index]['event'].startswith('neutral_') and (not comparable or len(difference)):
            raise ValueError('A designated quiet block has a physical mode effect')
        truth.append({'physically_neutral': not len(difference) if comparable else None,
                      'first_mode_difference_raw_step': int(start + difference[0] + 1)
                      if comparable and len(difference) else None})
    return truth


def _choice(scores, bank):
    values = scores.reshape(-1).cpu().numpy()
    if not np.isfinite(values).all():
        raise FloatingPointError('A forecast produced nonfinite candidate scores')
    selected = int(values.argmin())
    choices = np.full((len(bank.physical_costs), 1), selected)
    regrets = regret_parts(bank.physical_costs, bank.group_costs, choices)[:, 0]
    first = [float(value) if mode in bank.comparable_modes else None for mode, value in enumerate(regrets[:, 0])]
    full = [float(value) if mode in bank.comparable_modes else None for mode, value in enumerate(regrets[:, 2])]
    return {'scores': values.tolist(), 'selected': selected, 'first_block_group': int(bank.first_groups[selected]),
            'first_block_regret_by_mode': first, 'full_plan_regret_by_mode': full}


def _actual_latent_mse(issued, arrived):
    error = (issued - arrived).square().mean(dim=-1)
    return error.cpu().tolist()


@torch.no_grad()
def replay_arm(host, gru, name, episode, latents, executed, goal, banks, metadata, arrival_truth):
    """Score one prescribed episode without feeding evaluator labels to any forecaster."""
    if name not in ARMS:
        raise ValueError(f'Unknown baseline method {name}')
    dev = latents.device
    criterion = lambda trajectory, goal_latent: latent_goal_cost(host, trajectory, goal_latent)
    adapter = (OnlineAdapter(host, _rollout, criterion, method=name, replans=len(executed) - 2)
               if name in ('adajepa', 'optta') else None)
    hidden = None
    if name == 'gru':
        for init in range(2):
            hidden = gru.observe(hidden, latents[init:init + 1], executed[init:init + 1])
    if adapter:
        for init in range(2):
            adapter.add_segment(latents[None, init:init + 2], executed[None, init:init + 1])

    labels, modes = _labels(metadata)
    first_probe = metadata['probes'][0]
    anchor_obs = first_probe['pre_block']
    anchor_end = first_probe['post_block']
    anchor_bank = banks[first_probe['fork_block']]
    anchor_old_mode = metadata['initial_mode']
    pending_plan = None
    issued = None
    reference_anchor = None
    rounds = []

    def predict(root, candidates):
        if name == 'gru':
            return gru.rollout(hidden, latents[root:root + 1], candidates)
        return _rollout(host, latents[None, root - 2:root + 1], executed[None, root - 2:root], candidates)

    def rank(root, bank):
        trajectory = predict(root, bank.actions)
        return trajectory, criterion(trajectory, goal)

    try:
        for root, bank in banks.items():
            before = None
            before_seconds, update_seconds = 0., 0.
            records = []
            informative_tags = [key for key, _ in labels[root] if key != 'neutral_control']
            if root > 2:
                if labels[root] and name != 'frozen':
                    before, before_seconds = _timed(dev, rank, root, bank)
                previous, actual = latents[root - 1:root], latents[root:root + 1]
                action = executed[root - 1:root]
                if adapter:
                    records, update_seconds = _timed(dev, lambda: adapter.arrive(
                        previous, action, actual, episode=episode, replan=root - 3, plan=pending_plan,
                    ))
                if name == 'gru':
                    hidden, update_seconds = _timed(dev, gru.observe, hidden, previous, action)

            (trajectory, scores), selection_seconds = _timed(dev, rank, root, bank)
            after = _choice(scores, bank)
            row = {'root': root, 'raw_step': root * ACTION_BLOCK, 'truth_mode': int(modes[root]),
                   'phase_labels': labels[root], 'arrived_event': metadata['blocks'][root - 1]['event'],
                   'selection_candidates': bank.actions.size(1), 'selection_seconds': selection_seconds,
                   'pre_diagnostic_seconds': before_seconds, 'update_seconds': update_seconds,
                   'arrival_truth': arrival_truth[root - 1],
                   'online_update': records, 'selected_latent_trajectory': trajectory[0, after['selected']].cpu().tolist(),
                   'selection': after, 'prequential_arrival_mse': None if issued is None else
                   _actual_latent_mse(issued, latents[root])}
            if labels[root] and root > 2:
                pre_trajectory, pre_scores = before if before is not None else (trajectory, scores)
                row['before_update'] = _choice(pre_scores, bank)
                row['change'] = {
                    'forecast_rms': (pre_trajectory - trajectory).square().mean().sqrt().item(),
                    'cost_rms': (pre_scores - scores).square().mean().sqrt().item(),
                    'first_action_changed': after['first_block_group'] != row['before_update']['first_block_group'],
                }
            if informative_tags and root >= first_probe['fork_block']:
                def old_anchor():
                    control = executed[None, None, anchor_obs:anchor_end]
                    prediction = predict(anchor_obs, control)[0, 0]
                    old_truth = latents[anchor_obs + 1:anchor_end + 1]
                    _, old_cost = rank(first_probe['fork_block'], anchor_bank)
                    return _actual_latent_mse(prediction, old_truth), _choice(old_cost, anchor_bank)

                (mse, selection), anchor_seconds = _timed(dev, old_anchor)
                regret = selection['first_block_regret_by_mode'][anchor_old_mode]
                if root == first_probe['fork_block']:
                    reference_anchor = np.asarray(mse), regret
                interpretation = 'history_transfer' if name == 'gru' else 'fixed_window_weight_change'
                row['old_anchor'] = {'mode': anchor_old_mode, 'interpretation': interpretation, 'mse_by_horizon': mse,
                                     'first_block_regret': regret, 'selection': selection, 'seconds': anchor_seconds,
                                     'delta_mse_from_retained_fork': None if reference_anchor is None else
                                     (np.asarray(mse) - reference_anchor[0]).tolist(),
                                     'delta_regret_from_retained_fork': None if reference_anchor is None else
                                     regret - reference_anchor[1]}

            if root < len(executed):
                if name == 'optta':
                    pending_plan = adapter.capture(latents[None, root - 2:root + 1], executed[None, root - 2:root],
                                                   bank.actions, goal, scores, replan=root - 2)
                issued, row['issuance_seconds'] = _timed(dev, predict, root, executed[None, None, root:root + 1])
                issued = issued[0, 0, 0]
            rounds.append(row)
    finally:
        if adapter:
            adapter.reset()
    return rounds, adapter.stats if adapter else gru.stats if name == 'gru' else {}


def _episodes(physical, histories):
    for index, fork in enumerate(physical['forks']):
        for initial, suffix in enumerate(('a', 'b')):
            metadata = fork['chronology'][f'episode_{suffix}']
            order = ''.join('AB'[probe['mode']] for probe in metadata['probes'])
            if order not in ('ABA', 'BAB') or metadata['neutral_nonzero_raw_steps'] < 10:
                raise ValueError('Expected held-out two-flip histories and at least ten active neutral controls')
            yield index, fork, order, metadata, {key: histories[key][index, initial] for key in histories}


@torch.no_grad()
def _model_inputs(native, arrays):
    return (torch.stack([native.encode(rgb) for rgb in arrays['rgb']]),
            action_embeddings(native.host, arrays['raw_actions']), native.encode(arrays['goal_rgb'])[None])


def _summarize(cases):
    summary = {}
    for name in ARMS:
        by_phase = {}
        rows = [row for case in cases for row in case['arms'][name]]
        for case in cases:
            for row in case['arms'][name]:
                for phase, mode in row['phase_labels']:
                    by_phase.setdefault((case['order'], phase), []).append((row, mode))
                    by_phase.setdefault(('all', phase), []).append((row, mode))
        summaries = {}
        for (order, phase), queries in by_phase.items():
            item = {'count': len(queries)}
            for key in ('first_block_regret_by_mode', 'full_plan_regret_by_mode'):
                item[f'mean_{key}'] = float(np.mean([row['selection'][key][mode] for row, mode in queries]))
                paired = [row['selection'][key][mode] - row['before_update'][key][mode] for row, mode in queries
                          if 'before_update' in row]
                item[f'after_minus_before_{key}'] = float(np.mean(paired)) if paired else None
            changes = [row['change'] for row, _ in queries if 'change' in row]
            if changes:
                item['first_block_changes'] = sum(row['first_action_changed'] for row in changes)
                item['forecast_change_rms_mean'] = float(np.mean([row['forecast_rms'] for row in changes]))
            old = [row['old_anchor'] for row, _ in queries if 'old_anchor' in row]
            if old:
                item['old_anchor_interpretation'] = old[0]['interpretation']
                item['old_anchor_mse_by_horizon'] = np.mean([row['mse_by_horizon'] for row in old], axis=0).tolist()
                item['old_anchor_first_block_regret'] = float(np.mean([row['first_block_regret'] for row in old]))
                item['old_anchor_delta_from_retained_fork'] = np.mean(
                    [row['delta_mse_from_retained_fork'] for row in old], axis=0,
                ).tolist()
                item['old_anchor_first_block_regret_delta'] = float(np.mean(
                    [row['delta_regret_from_retained_fork'] for row in old],
                ))
            summaries.setdefault(order, {})[phase] = item
        summary[name] = {'phase_order': summaries, 'rounds': len(rows),
                         'candidate_selection_programs': sum(row['selection_candidates'] for row in rows),
                         'optta_gradient_rerolls': sum(row['selection_candidates'] for row in rows
                                                      if any(step['loss_opt'] is not None for step in row['online_update'])),
                         'optta_planning_forward_seconds': sum(step.get('planning_forward_seconds', 0.) for row in rows
                                                               for step in row['online_update']),
                         'optta_balanced_gradients_seconds': sum(step.get('balanced_gradients_seconds', 0.) for row in rows
                                                                for step in row['online_update']),
                         'selection_seconds': sum(row['selection_seconds'] for row in rows),
                         'update_seconds': sum(row['update_seconds'] for row in rows)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default=DIRECTORY, type=Path)
    parser.add_argument('--gru', default=GRU_CHECKPOINT, type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    physical = require_qualification(args.directory)
    weights = torch.load(args.gru, map_location='cpu', weights_only=True)
    training = json.loads(args.gru.with_name('gru_training.json').read_text(encoding='utf-8'))
    if training['qualifying_protocol'] != physical['protocol']:
        raise ValueError('The GRU must be trained on the replay host protocol')
    torch.set_num_threads(2)
    torch.manual_seed(REPLAY_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    native = Native(torch.device(args.device))
    capacity = predictor_parameter_count(native.host)
    if capacity != weights['target_parameters']:
        raise ValueError('The GRU checkpoint size target must match the qualified native predictor')
    gru = GRUDynamics(weights['latent_dim'], weights['action_dim'], capacity).to(native.device)
    gru.load_state_dict(weights['state_dict'], strict=True)
    gru.eval().requires_grad_(False)
    output = args.directory / 'baseline_replay'
    output.mkdir(exist_ok=False)
    env = OfficialFourRoom(fixed_distractor='fixed_distractor' in physical['protocol'])
    try:
        physics = OpenPhysics(env)
        tasks = {task.name: task for task in fixed_tasks()}
        with np.load(args.directory / 'fork_lifecycle_rgb_controls.npz', allow_pickle=False) as trace_archive, np.load(
                args.directory / 'fork_candidates.npz', allow_pickle=False) as pinned:
            histories = dict(trace_archive)
            raw_banks, physical_costs = pinned['raw_candidates'], pinned['candidate_cost_by_mode']
        with torch.no_grad():
            inputs = [(*entry, _model_inputs(native, entry[-1])) for entry in _episodes(physical, histories)]
        cases = []
        settings = {}
        with (output / 'rounds.jsonl').open('x', encoding='utf-8') as stream, torch.no_grad():
            for index, fork, order, metadata, arrays, (z, executed, goal) in inputs:
                banks = _bank_series(native.host, physics, tasks[fork['task']], metadata,
                                     arrays['raw_actions'], arrays['raw_xy'], raw_banks[index], physical_costs[index])
                truth = _arrival_truth(physics, metadata, arrays['raw_xy'], arrays['raw_actions'])
                episode = f"{fork['task']}_{order}"
                episode_index = 2 * index + metadata['initial_mode']
                dropout_seed = REPLAY_SEED + index
                record = {'episode': episode, 'seed': fork['seed'], 'order': order, 'arms': {}}
                for name in ARMS:
                    # Pair grounding dropout across the two adapting methods and both factual polarities.
                    torch.manual_seed(dropout_seed)
                    rows, settings[name] = replay_arm(native.host, gru, name, episode_index, z, executed, goal, banks,
                                                     metadata, truth)
                    record['arms'][name] = rows
                    stream.write(json.dumps({'episode': episode, 'seed': fork['seed'], 'order': order, 'arm': name,
                                             'grounding_dropout_seed': dropout_seed, 'rows': rows}, allow_nan=False) + '\n')
                    stream.flush()
                    print(episode, name, len(rows), flush=True)
                cases.append(record)
        write_json(output / 'summary.json', {'settings': settings, 'qualification_protocol': physical['protocol'],
                                            'summary': _summarize(cases), 'episodes': len(cases),
                                            'grounding_dropout_seed_rule': '20261117 + pinned fork index',
                                            'gru_training': training})
    finally:
        env.close()


if __name__ == '__main__':
    main()
