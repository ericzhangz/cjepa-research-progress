"""Matched clean FourRoom P=I oracle capacity test on the Stage A fixed menu."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from .native_topology_condition import action_embeddings, load_host
from .native_topology_writein_attribution import first_block_groups
from .stage_a_fit_gru import encode_frames
from .stage_a_fourroom import (
    ACTION_BLOCK, CENTER, DOORS, HORIZON, LOCKS, ROOT, OfficialFourRoom,
    OpenPhysics, Task, candidate_bank, fixed_tasks, latent_goal_cost, write_json,
)
from .stage_c_data import command, execute, matches
from .stage_c_native_check import native_conditioned


OUT = ROOT / 'runs/stage_c_clean_20261117'
QUAL = ROOT / 'runs/stage_a_fourroom_clean_20261117'
ARMS = ('frozen', 'offsets', 'offsets_final_two')
PHASES = ('initial_probe_retained_fork', 'reversal_1_retained_fork', 'reversal_2_retained_fork')
UPDATES, SEED = 144, 2026112900
OFFSET_LR, BLOCK_LR = .008, .0001
FIT_SOURCE_SEED, FIT_BUNDLE_SEED, CHECK_SOURCE_SEED = 2026113000, 2026113100, 2026113200
FIT_SOURCE_COUNT, FIT_BUNDLE_COUNT, CHECK_SOURCE_COUNT = 32, 24, 16
H12_BATCH, H1_BATCH = 4, 4
DECISION_LIMIT = 10.


def encoder_for(host):
    device = host.action_mean.device
    return SimpleNamespace(
        host=host, device=device,
        mean=torch.tensor((.485, .456, .406), device=device)[:, None, None],
        std=torch.tensor((.229, .224, .225), device=device)[:, None, None],
    )


def encode_panel(encoder, frames):
    frames = np.asarray(frames)
    return encode_frames(encoder, frames.reshape(-1, *frames.shape[-3:])).reshape(*frames.shape[:-3], -1)


def collect_sources(env, physics, encoder, seed, count):
    windows, actions, targets = [], [], []
    for i in range(count):
        door = ('top', 'bottom')[i % 2]
        direction = (1., -1.)[(i // 2) % 2]
        start = np.array((CENTER - direction * (9. + (i // 4) % 4),
                          DOORS[door] + np.random.default_rng(seed + i).uniform(-5., 5.)), dtype=np.float32)
        goal = start.copy()
        goal[0] = CENTER + direction * 48.
        across = command(0, direction)
        along = command(1, .4 * (-1., 1.)[(i // 4) % 2])
        prefix = np.stack((-across, across))
        programs = np.stack((np.concatenate((prefix, across[None])), np.concatenate((prefix, along[None]))))
        traces = [[execute(env, physics, start, goal, seed+i, lock, prefix[:0], program)
                   for program in programs] for lock in LOCKS]
        first = traces[0][0]
        for trace in (traces[0][1], *traces[1]):
            matches(first, trace, rows=3)
        if np.array_equal(traces[0][0].rgb[-1], traces[1][0].rgb[-1]):
            raise AssertionError('The physical source probe must separate conditions')
        np.testing.assert_array_equal(traces[0][1].rgb[-1], traces[1][1].rgb[-1])
        windows.append(first.rgb[:3])
        actions.append(programs)
        targets.append(np.stack([[trace.rgb[-1] for trace in pair] for pair in traces]))
    return {'window': encode_panel(encoder, windows), 'actions': torch.tensor(np.stack(actions)),
            'target': encode_panel(encoder, targets)}


def collect_bundles(env, physics, encoder, seed, count):
    templates = tuple(task for task in fixed_tasks() if task.family == 'two_doors' and task.name.endswith('_0'))
    windows, actions, targets, cases = [], [], [], []
    for i in range(count):
        template = templates[i % len(templates)]
        random = np.random.default_rng(seed+i)
        start = np.asarray(template.start, dtype=np.float32) + random.uniform(-6., 6., 2).astype(np.float32)
        goal = np.asarray(template.goal, dtype=np.float32) + random.uniform(-6., 6., 2).astype(np.float32)
        task = Task(template.name, template.family, tuple(start), tuple(goal), seed+i, ())
        bank = candidate_bank(task, start, 0)
        half = len(bank) // 2
        indices = np.concatenate([base + np.r_[0, random.choice(np.arange(1, half-9), 3, replace=False)]
                                  for base in (0, half)])
        programs = bank[indices]
        away = command(0, -.4 if start[0] < CENTER else .4)
        prefix = np.stack((away, -away))
        full = np.concatenate((np.broadcast_to(prefix, (len(indices), *prefix.shape)), programs), axis=1)
        traces = [[execute(env, physics, start, goal, seed+i, lock, prefix[:0], program)
                   for program in full] for lock in LOCKS]
        first = traces[0][0]
        for trace in (*traces[0][1:], *traces[1]):
            matches(first, trace, rows=3)
        np.testing.assert_allclose(first.xy[2], start, atol=1e-5, rtol=0)
        windows.append(encode_panel(encoder, first.rgb[:3]))
        actions.append(full)
        target = encode_panel(encoder, np.stack([[trace.rgb[3:] for trace in pair] for pair in traces]))
        if not (target[0] - target[1]).abs().any():
            raise AssertionError('The matched H12 bundle has no visible condition effect')
        targets.append(target)
        cases.append({'seed': seed+i, 'route_template': template.name,
                      'candidate_indices': indices.tolist(), 'start': start.tolist(), 'goal': goal.tolist()})
        print(f'Collected H12 bundle {i+1}/{count}', flush=True)
    return {'window': torch.stack(windows), 'actions': torch.tensor(np.stack(actions)),
            'target': torch.stack(targets)}, cases


def collect(host, out):
    encoder = encoder_for(host)
    env = OfficialFourRoom(fixed_distractor=True)
    physics = OpenPhysics(env)
    try:
        fit_source = collect_sources(env, physics, encoder, FIT_SOURCE_SEED, FIT_SOURCE_COUNT)
        print('Collected H1 fit sources', flush=True)
        check_source = collect_sources(env, physics, encoder, CHECK_SOURCE_SEED, CHECK_SOURCE_COUNT)
        print('Collected H1 held-out sources', flush=True)
        fit_bundle, cases = collect_bundles(env, physics, encoder, FIT_BUNDLE_SEED, FIT_BUNDLE_COUNT)
    finally:
        env.close()
    write_json(out / 'oracle_fit_cases.json', {'cases': cases, 'fit_source_seed': FIT_SOURCE_SEED,
                                              'check_source_seed': CHECK_SOURCE_SEED})
    result = {'fit_source': fit_source, 'check_source': check_source, 'fit_bundle': fit_bundle}
    torch.save(result, out / 'oracle_data.pt')
    return result


def offsets_for_mode(b, v, mode):
    return b[:, None, None, :] + mode[None, :, None, None] * v[:, None, None, :]


def rollout_identity(host, window, controls, offsets=None):
    """Roll one native window per program, simplifying Eq. 49.1 with P=I."""
    forecast = []
    for step in range(controls.size(1) - window.size(1) + 1):
        actions = controls[:, step:step + window.size(1)]
        predicted = (host.predict(window, actions) if offsets is None
                     else native_conditioned(host, window, actions, offsets))[:, -1]
        forecast.append(predicted)
        window = torch.cat((window[:, 1:], predicted[:, None]), dim=1)
    return torch.stack(forecast, dim=1)


@torch.no_grad()
def with_embeddings(host, panel):
    device = host.action_mean.device
    return {**{key: value.to(device) for key, value in panel.items() if key != 'actions'},
            'actions': action_embeddings(host, panel['actions'])}


def paired_selection(panel, rows, candidates):
    window = panel['window'][rows].repeat_interleave(2, dim=0)
    controls = panel['actions'][rows, candidates].repeat_interleave(2, dim=0)
    target = panel['target'][rows, :, candidates].flatten(0, 1)
    modes = window.new_tensor((-1., 1.)).repeat(len(rows))
    return window, controls, modes, target


def fit(host, data, arm, out):
    started = time.perf_counter()
    transformer = host.predictor.transformer
    bundle = with_embeddings(host, data['fit_bundle'])
    source = with_embeddings(host, data['fit_source'])
    b = nn.Parameter(bundle['window'].new_zeros((len(transformer.layers), bundle['actions'].size(-1))))
    v = nn.Parameter(torch.zeros_like(b))
    last_two = transformer.layers[-2:]
    groups = [{'params': [b, v], 'lr': OFFSET_LR}]
    if arm == 'offsets_final_two':
        last_two.requires_grad_(True)
        groups.append({'params': list(last_two.parameters()), 'lr': BLOCK_LR})
    expected_parameters = [id(p) for group in groups[1:] for p in group['params']]
    if set(expected_parameters) != {id(p) for p in host.parameters() if p.requires_grad} or b.numel()+v.numel() != 1536:
        raise AssertionError('The Oracle-k arms must train exactly 1536 offsets and optionally the final two blocks')
    optimizer = torch.optim.AdamW(groups)
    generator = torch.Generator().manual_seed(SEED)
    losses = []
    for iteration in range(UPDATES):
        bundle_rows = torch.randint(len(bundle['window']), (H12_BATCH,), generator=generator)
        choice = torch.randint(bundle['actions'].size(1), (H12_BATCH,), generator=generator)
        window, control, mode, target = paired_selection(bundle, bundle_rows, choice)
        prediction = rollout_identity(host, window, control, offsets_for_mode(b, v, mode))
        full_loss = (prediction-target).square().mean()
        source_rows = torch.randint(len(source['window']), (H1_BATCH,), generator=generator).repeat_interleave(2)
        event = torch.arange(2).repeat(H1_BATCH)
        window, control, mode, target = paired_selection(source, source_rows, event)
        local_prediction = rollout_identity(host, window, control, offsets_for_mode(b, v, mode))[:, 0]
        local_loss = (local_prediction-target).square().mean()
        loss = (full_loss+local_loss) / 2
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite {arm} loss at iteration {iteration}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append([float(full_loss.detach()), float(local_loss.detach())])
        if (iteration+1) % 24 == 0:
            print(f'{arm} {iteration+1}/{UPDATES}: H12={full_loss:.4f}, H1={local_loss:.4f}', flush=True)
    state = {'arm': arm, 'b': b.detach().cpu(), 'v': v.detach().cpu()}
    if arm == 'offsets_final_two':
        state['final_two_blocks'] = {name: weight.detach().cpu() for name, weight in last_two.state_dict().items()}
    torch.save(state, out / f'oracle_{arm}.pt')
    return {'trainable_parameters': sum(p.numel() for group in groups for p in group['params']),
            'loss_categories': ['H12 full rollout MSE', 'H1 source MSE'], 'losses': losses,
            'updates': UPDATES, 'wall_seconds': time.perf_counter()-started,
            'bias_norm': float(b.detach().norm()), 'mode_vector_norm': float(v.detach().norm())}


def load_arm(arm, out, device):
    host = load_host(device)
    if arm == 'frozen':
        return host, None, None
    state = torch.load(out / f'oracle_{arm}.pt', map_location=device, weights_only=True)
    if arm == 'offsets_final_two':
        host.predictor.transformer.layers[-2:].load_state_dict(state['final_two_blocks'], strict=True)
    return host, state['b'], state['v']


@torch.no_grad()
def source_check(host, data, b, v):
    panel = with_embeddings(host, data['check_source'])
    rows = torch.arange(len(panel['window'])).repeat_interleave(2)
    event = torch.arange(2).repeat(len(panel['window']))
    window, control, mode, target = paired_selection(panel, rows, event)
    zero = window.new_zeros(4, control.size(-1))
    base = rollout_identity(host, window, control)
    torch.testing.assert_close(base, rollout_identity(host, window, control, offsets_for_mode(zero, zero, mode)),
                               rtol=0, atol=0)
    forecast = base if b is None else rollout_identity(host, window, control, offsets_for_mode(b, v, mode))
    forecast = forecast[:, 0].reshape(-1, 2, 2, window.size(-1))
    target = target.reshape_as(forecast)
    pred_diff = forecast[:, 0, 1] - forecast[:, 0, 0]
    target_diff = target[:, 0, 1] - target[:, 0, 0]
    right = (forecast[:, 0] - target[:, 0]).square().sum(-1)
    wrong = (forecast[:, 0].flip(1) - target[:, 0]).square().sum(-1)
    return {'diagnostic_probe_count': len(forecast),
            'true_vs_other_branch_correct': int((right < wrong).sum()),
            'true_vs_other_branch_total': right.numel(),
            'correct_direction_probes': int(((pred_diff*target_diff).sum(-1) > 0).sum()),
            'latent_rmse_probe': float((forecast[:, 0] - target[:, 0]).square().mean().sqrt()),
            'latent_rmse_neutral': float((forecast[:, 1] - target[:, 1]).square().mean().sqrt()),
            'predicted_probe_difference_l2': float(pred_diff.norm(dim=-1).mean()),
            'target_probe_difference_l2': float(target_diff.norm(dim=-1).mean()),
            'predicted_neutral_difference_l2': float((forecast[:, 1, 1]-forecast[:, 1, 0]).norm(dim=-1).mean())}


def decision_outcome(cost, programs, score, mode):
    chosen = int(score.argmin())
    _, minimum = first_block_groups(programs, cost[mode])
    best = cost.argmin(-1)
    other = 1-mode
    return {'candidate': chosen, 'first_block_regret': float(minimum[chosen] - cost[mode].min()),
            'full_plan_regret': float(cost[mode, chosen] - cost[mode].min()),
            'true_best_route_score_margin': float(score[best[other]] - score[best[mode]])}


@torch.no_grad()
def evaluate(host, arms, qual, out):
    encoder = encoder_for(host)
    data = json.loads((qual / 'physical.json').read_text(encoding='utf-8'))
    with np.load(qual / 'fork_candidates.npz', allow_pickle=False) as archive:
        all_programs, all_costs, goals = [archive[name] for name in ('raw_candidates', 'candidate_cost_by_mode', 'goal_rgb')]
    with np.load(qual / 'fork_lifecycle_rgb_controls.npz', allow_pickle=False) as archive:
        histories, past_controls = [archive[name] for name in ('rgb', 'raw_actions')]
    device = encoder.device
    all_scores, outcomes = {arm: [] for arm in arms}, {arm: [] for arm in arms}
    models = {arm: load_arm(arm, out, device) for arm in arms}
    for i, fork in enumerate(data['forks']):
        programs, costs = all_programs[i], all_costs[i]
        if len(programs) != 384 or programs.shape[1] != HORIZON:
            raise AssertionError('Expected the Stage A fixed menu and H12')
        goal = encode_frames(encoder, goals[i:i+1]).to(device)
        blocks = [value['fork_block'] for value in fork['chronology']['episode_a']['probes']]
        for phase, t in zip(PHASES, blocks, strict=True):
            np.testing.assert_array_equal(histories[i, 0, t-2:t+1], histories[i, 1, t-2:t+1])
            np.testing.assert_array_equal(past_controls[i, 0, t-2:t], past_controls[i, 1, t-2:t])
            root = encode_frames(encoder, histories[i, 0, t-2:t+1]).to(device)
            past = past_controls[i, 0, t-2:t]
            commands = action_embeddings(host, np.concatenate((np.broadcast_to(past, (len(programs), *past.shape)),
                                                               programs), axis=1))
            for arm, (predictor, b, v) in models.items():
                scores = []
                for sign in (-1., 1.) if b is not None else (0.,):
                    blocks_scores = []
                    for chunk in commands.split(96):
                        offsets = None if b is None else b + sign*v
                        predicted = rollout_identity(predictor, root[None].expand(len(chunk), -1, -1), chunk, offsets)
                        padded = torch.cat((root[-1:].expand(len(chunk), 1, -1), predicted), dim=1)
                        blocks_scores.append(latent_goal_cost(host, padded[None], goal)[0].cpu().numpy())
                    scores.append(np.concatenate(blocks_scores))
                if b is None:
                    scores = scores * 2
                all_scores[arm].append(np.stack(scores))
                for mode in range(2):
                    outcomes[arm].append({'task': fork['task'], 'phase': phase, 'root_block': t,
                                          'true_lock': LOCKS[mode],
                                          **decision_outcome(costs, programs, scores[mode], mode)})
        print(f'Evaluated fixed holdout fork {fork["task"]}', flush=True)
    np.savez_compressed(out / 'oracle_fixed_scores.npz', **{arm: np.stack(values).reshape(4, 3, 2, 384)
                                                          for arm, values in all_scores.items()})
    result = {}
    for arm, rows in outcomes.items():
        if len(rows) != 24:
            raise AssertionError('The capacity gate requires all 24 held-out mode and phase queries')
        mean = float(np.mean([row['first_block_regret'] for row in rows]))
        result[arm] = {'queries': len(rows), 'mean_first_block_regret': mean,
                       'mean_full_plan_regret': float(np.mean([row['full_plan_regret'] for row in rows])),
                       'correct_route_direction': sum(row['true_best_route_score_margin'] > 0 for row in rows),
                       'mean_first_block_regret_by_lock': {lock: float(np.mean([row['first_block_regret'] for row in rows
                                                                              if row['true_lock'] == lock]))
                                                           for lock in LOCKS},
                       'capacity_pass': mean < DECISION_LIMIT, 'rows': rows}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--qualification', type=Path, default=QUAL)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stage', choices=('collect-fit', 'evaluate', 'all'), default='all')
    args = parser.parse_args()
    started = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    baked = args.out / 'oracle_data.pt'
    device = torch.device(args.device)
    protocol = {'projection': 'identity', 'fit_seed': SEED, 'offset_lr': OFFSET_LR, 'final_blocks_lr': BLOCK_LR,
                'updates_per_arm': UPDATES, 'matched_H12_pairs_per_update': H12_BATCH,
                'matched_H1_pairs_per_update': 2*H1_BATCH, 'optimizer': 'AdamW', 'mode_mapping': dict(zip(LOCKS, (-1, 1))),
                'strict_mean_first_regret_threshold': DECISION_LIMIT, 'fixed_holdout_root_count': 12,
                'fit_sources': FIT_SOURCE_COUNT, 'check_sources': CHECK_SOURCE_COUNT, 'fit_bundles': FIT_BUNDLE_COUNT,
                'training_data': 'frozen encodings of actual executed RGB; true mode provided for these oracle arms'}
    if args.stage != 'evaluate':
        write_json(args.out / 'oracle_protocol.json', protocol)
    host = load_host(device)
    archive = (torch.load(baked, weights_only=True)
               if args.stage == 'evaluate' or baked.is_file() else collect(host, args.out))
    fit_path = args.out / 'oracle_fit.json'
    if args.stage != 'evaluate':
        fit_rows = {}
        for arm in ARMS[1:]:
            fit_rows[arm] = fit(host, archive, arm, args.out)
            write_json(fit_path, fit_rows)
    if args.stage != 'collect-fit':
        evaluation_started = time.perf_counter()
        fit_rows = json.loads(fit_path.read_text(encoding='utf-8'))
        checks = {arm: source_check(model, archive, b, v) for arm in ARMS
                  for model, b, v in (load_arm(arm, args.out, device),)}
        decisions = evaluate(host, ARMS, args.qualification, args.out)
        write_json(args.out / 'oracle_capacity.json', {'protocol': protocol, 'source': checks,
                                                       'fit': fit_rows, 'decision': decisions,
                                                       'evaluation_wall_seconds': time.perf_counter()-evaluation_started,
                                                       'total_wall_seconds': time.perf_counter()-started})
        print(json.dumps({arm: {k: v for k, v in row.items() if k != 'rows'}
                          for arm, row in decisions.items()}), flush=True)


if __name__ == '__main__':
    main()
