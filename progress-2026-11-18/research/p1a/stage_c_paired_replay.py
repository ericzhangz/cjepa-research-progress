"""Replay the clean §49.8 factual support path without editing weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .stage_a_fourroom import ROOT, Native, OfficialFourRoom, OpenPhysics, fixed_tasks, latent_goal_cost, write_json
from .stage_a_replay import _arrival_truth, _bank_series, _choice, _episodes, _labels, _model_inputs, _timed
from .stage_c_paired import (
    conditional_step, menu_forecast, replace_support, source_evidence, two_offsets,
)


@torch.no_grad()
def mode_scores(host, window, past, bank, goal, offsets):
    parts = []
    for chunk in bank.actions[0].split(96):
        controls = torch.cat((past.expand(len(chunk), -1, -1), chunk), dim=1)
        latent = menu_forecast(host, window[None, None], controls[None, None], offsets)[0, 0].transpose(0, 1)
        latent = torch.cat((window[-1].expand(2, len(chunk), 1, -1), latent), dim=2)
        parts.append(latent_goal_cost(host, latent, goal.expand(2, -1)))
    return torch.cat(parts, dim=-1)


def sequential_arrivals(device, host, z, actions, offsets, scale_squared, *, omitted_arrivals=()):
    state = z.new_full((2,), .5)
    timeline = [state, state, state]
    omitted_state = state
    omitted_timeline = [state, state, state]
    observations, timings = [], []
    for t in range(2, len(actions)):
        forecast, issue_seconds = _timed(
            device, conditional_step, host, z[t-2:t+1].expand(2, -1, -1),
            actions[t-2:t+1].expand(2, -1, -1), offsets,
        )
        def assimilate():
            evidence = source_evidence(forecast, z[t+1], scale_squared)
            return replace_support(state, evidence), evidence

        (state, evidence), write_seconds = _timed(device, assimilate)
        timeline.append(state)
        if t+1 not in omitted_arrivals:
            omitted_state = replace_support(omitted_state, evidence)
        omitted_timeline.append(omitted_state)
        observations.append((evidence, forecast))
        timings.append((issue_seconds, write_seconds))
    e, issued = zip(*observations, strict=True)
    return torch.stack(timeline), torch.stack(e), torch.stack(issued), timings, torch.stack(omitted_timeline)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'runs/stage_c_clean_20261117')
    directory = parser.parse_args().directory
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    qualified = ROOT / 'runs/stage_a_fourroom_clean_20261117'
    device = torch.device('cuda')
    native = Native(device)
    state = torch.load(directory / 'paired_unlabeled.pt', map_location=device, weights_only=True)
    if 'final_two_blocks' in state:
        native.host.predictor.transformer.layers[-2:].load_state_dict(state['final_two_blocks'], strict=True)
    fit = json.loads((directory / 'paired_unlabeled_fit.json').read_text(encoding='utf-8'))
    offsets = two_offsets(state['b'], state['v'])
    physical = json.loads((qualified / 'physical.json').read_text(encoding='utf-8'))
    with np.load(qualified / 'fork_lifecycle_rgb_controls.npz', allow_pickle=False) as archive:
        episodes = list(_episodes(physical, dict(archive)))
    with np.load(qualified / 'fork_candidates.npz', allow_pickle=False) as archive:
        programs, costs = archive['raw_candidates'], archive['candidate_cost_by_mode']
    tasks = {task.name: task for task in fixed_tasks()}
    rows, old_anchor_checks, omitted_by_episode = [], [], []
    env = OfficialFourRoom(fixed_distractor=True)
    try:
        physics = OpenPhysics(env)
        for index, fork, order, meta, arrays in episodes:
            z, actions, goal = _model_inputs(native, arrays)
            omitted_arrivals = frozenset(t for probe in meta['probes']
                                        for t in range(probe['pre_block']+1, probe['post_block']+1))
            p, e, issued, timings, omitted = sequential_arrivals(
                device, native.host, z, actions, offsets, fit['s2_obs'], omitted_arrivals=omitted_arrivals)
            label, modes = _labels(meta)
            truth = _arrival_truth(physics, meta, arrays['raw_xy'], arrays['raw_actions'])
            banks = _bank_series(native.host, physics, tasks[fork['task']], meta, arrays['raw_actions'],
                                 arrays['raw_xy'], programs[index], costs[index])
            episode_id = f"{fork['task']}_{order}"
            omitted_by_episode.append({'episode': episode_id, 'arrival_blocks': sorted(omitted_arrivals)})
            old_anchor = meta['probes'][0]['pre_block']
            old_window = z[old_anchor-2:old_anchor+1][None, None]
            old_controls = actions[old_anchor-2:old_anchor+2][None, None, None]
            old_prediction = menu_forecast(native.host, old_window, old_controls, offsets)[0, 0, 0]
            old_truth = z[old_anchor+1:old_anchor+3]
            old_mse = (old_prediction - old_truth).square().mean(dim=-1)
            for t, bank in banks.items():
                if not label[t]:
                    continue
                scores, planning_time = _timed(device, mode_scores, native.host, z[t-2:t+1], actions[t-2:t],
                                               bank, goal, offsets)
                before = p[t-1]
                keep = p[meta['probes'][0]['fork_block']] if t >= meta['probes'][0]['fork_block'] else before
                states = torch.stack((p[t], before, p[t].new_full((2,), .5), keep,
                                      p[t].new_tensor((1., 0.)), p[t].new_tensor((0., 1.))))
                choices = {key: _choice(score, bank) for key, score in zip(
                    ('written', 'before', 'uniform', 'keep_old', 'branch_0', 'branch_1'), states @ scores, strict=True)}
                choices['omit_probe_writes'] = _choice(omitted[t] @ scores, bank)
                rows.append({'episode': episode_id, 'root': t, 'mode_eval_only': int(modes[t]),
                             'phase_labels': label[t], 'probabilities': p[t].tolist(),
                             'omit_probe_probabilities': omitted[t].tolist(),
                             'evidence': float(e[t-3]) if t>=3 else None,
                             'written_prequential_mse': (issued[t-3] - z[t]).square().mean(-1).tolist()
                               if t>=3 else None,
                             'first_mode_difference_raw_step': truth[t-1]['first_mode_difference_raw_step'],
                             'neutral': truth[t-1]['physically_neutral'],
                             'conditional_forecast_seconds': planning_time, 'choices': choices,
                             'old_anchor_mse_by_branch': old_mse.tolist(),
                             'issue_seconds': timings[t-3][0] if t>=3 else None,
                             'write_seconds': timings[t-3][1] if t>=3 else None})
            repeated = menu_forecast(native.host, old_window, old_controls, offsets)[0, 0, 0]
            torch.testing.assert_close(repeated, old_prediction, rtol=0., atol=0.)
            old_anchor_checks.append({'episode': episode_id, 'anchor': old_anchor,
                                      'max_abs_difference': float((repeated-old_prediction).abs().max())})
            print(episode_id, flush=True)
    finally:
        env.close()
    write_json(directory / 'paired_unlabeled_replay.json', {
        'rows': rows, 'arm': state['arm'], 'old_anchor_checks': old_anchor_checks,
        'omit_probe_write_arrivals': omitted_by_episode,
        'omit_probe_write_control': 'Omit only pre_block < arrival <= post_block for each designated probe. '
                                    'Keep all other arrivals and reuse identical candidate branch scores.',
    })


if __name__ == '__main__':
    main()
