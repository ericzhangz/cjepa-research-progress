"""Trace fixed-pool first-block errors through actual-window and free §49.8 prediction."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.p1a.native_topology_condition import action_embeddings, load_host
from research.p1a.stage_a_fit_gru import encode_frames
from research.p1a.stage_a_fourroom import (
    ACTION_BLOCK, LOCKS, Native, OfficialFourRoom, OpenPhysics, actual_crossings, fixed_tasks,
    latent_goal_cost, write_json,
)
from research.p1a.stage_c_paired import conditional_step, menu_forecast, two_offsets

HERE = Path(__file__).resolve().parent
QUALIFIED = HERE.parent / 'stage_a_fourroom_clean_20261117'


def scores(host, root, z, goal):
    z = z.reshape(-1, 1, *z.shape[-2:])
    root = root.reshape(1, 1, 1, -1).expand(len(z), 1, 1, -1)
    return latent_goal_cost(host, torch.cat((root,z), dim=-2), goal.expand(len(z), -1)).flatten().tolist()


def baseline_trace(host, offsets, native, window, actions, reached, controls, goal, probabilities):
    free = menu_forecast(host, window[None, None], actions[None, None, None], offsets)[0, 0, 0]
    local = conditional_step(host, reached[:, None].expand(-1, 2, -1, -1),
                             controls[:, None].expand(-1, 2, -1, -1), offsets).transpose(0, 1)
    free_cost = scores(host, window[-1], free, goal)
    local_cost = scores(host, window[-1], local, goal)
    support = window.new_tensor(probabilities)
    explanation = int(support.argmax())
    return {'probabilities': probabilities, 'chosen_explanation': explanation,
            'free_last2_score_by_branch': free_cost,
            'actual_window_last2_score_by_branch': local_cost,
            'free_mixture': float(support @ window.new_tensor(free_cost)),
            'actual_window_mixture': float(support @ window.new_tensor(local_cost)),
            'free_vs_actual_window_l2_by_horizon': (native.decode(free)-native.decode(local)).norm(dim=-1).cpu().tolist(),
            'F': native.decode(free[explanation]).cpu().tolist(),
            'L': native.decode(local[explanation]).cpu().tolist()}


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=HERE)
    parser.add_argument('--qualification', type=Path, default=QUALIFIED)
    parser.add_argument('--baseline-directory', type=Path)
    args = parser.parse_args()
    native = Native(device)
    host = native.host
    state = torch.load(args.directory / 'paired_unlabeled.pt', map_location=device, weights_only=True)
    if 'final_two_blocks' in state:
        host.predictor.transformer.layers[-2:].load_state_dict(state['final_two_blocks'], strict=True)
    offsets = two_offsets(state['b'],state['v'])
    rows = json.loads((args.directory / 'paired_unlabeled_replay.json').read_text(encoding='utf-8'))['rows']
    baseline_rows = []
    baseline_host, baseline_offsets = None, None
    if args.baseline_directory is not None:
        baseline_rows = json.loads((args.baseline_directory / 'paired_unlabeled_replay.json').read_text(encoding='utf-8'))['rows']
        baseline_state = torch.load(args.baseline_directory / 'paired_unlabeled.pt', map_location=device, weights_only=True)
        baseline_host = load_host(device)
        if 'final_two_blocks' in baseline_state:
            baseline_host.predictor.transformer.layers[-2:].load_state_dict(baseline_state['final_two_blocks'], strict=True)
        baseline_offsets = two_offsets(baseline_state['b'], baseline_state['v'])
    forks = json.loads((args.qualification / 'physical.json').read_text(encoding='utf-8'))['forks']
    tasks = {task.name: task for task in fixed_tasks()}
    env = OfficialFourRoom(fixed_distractor=True)
    physics = OpenPhysics(env)
    report = []
    try:
        with np.load(args.qualification / 'fork_candidates.npz', allow_pickle=False) as archive, np.load(
                args.qualification / 'fork_lifecycle_rgb_controls.npz', allow_pickle=False) as factual:
            for i, fork in enumerate(forks):
                task = tasks[fork['task']]
                probroot = fork['chronology']['episode_a']['probes'][0]['fork_block']
                window = torch.stack([native.encode(rgb) for rgb in factual['rgb'][i, 0, probroot-2:probroot+1]])
                past = factual['raw_actions'][i, 0, probroot-2:probroot]
                goal = native.encode(archive['goal_rgb'][i])[None]
                q = {(row['episode'],mode): row for row in rows for phase,mode in row['phase_labels']
                     if phase=='initial_probe_retained_fork'}
                for order, mode in (('ABA', 0), ('BAB', 1)):
                    row = q[(f'{task.name}_{order}',mode)]
                    p = window.new_tensor(row['probabilities'])
                    explain = int(p.argmax())
                    index_by_role = {'chosen': row['choices']['written']['selected'],
                                     'physical_best': fork['true_indices'][mode],
                                     'physical_opposite': fork['true_indices'][1-mode]}
                    if baseline_rows:
                        baseline_query = {(item['episode'], mode): item for item in baseline_rows
                                          for phase, mode in item['phase_labels']
                                          if phase == 'initial_probe_retained_fork'}
                        baseline_row = baseline_query[(row['episode'], mode)]
                        index_by_role['baseline_selected'] = baseline_row['choices']['written']['selected']
                        failures = {'opposite_0_0': 115, 'opposite_0_1': 186,
                                    'opposite_2_0': 54, 'opposite_2_1': 105}
                        index_by_role['baseline_failure_candidate'] = failures[task.name]
                    roles_by_index = {}
                    for role, idx in index_by_role.items():
                        roles_by_index.setdefault(idx, []).append(role)
                    for idx, roles in roles_by_index.items():
                        control = archive['raw_candidates'][i,idx]
                        env.ego_lock = LOCKS[mode]
                        env.reset(seed=task.seed, options={'state':task.start,'target_state':task.goal})
                        frames, positions = [], []
                        for raw in control:
                            positions.extend(env.execute(raw))
                            frames.append(env.render())
                        np.testing.assert_array_equal(positions,archive['raw_xy_by_mode'][i,mode,idx])
                        endpoint = np.asarray(positions)[ACTION_BLOCK-1::ACTION_BLOCK]
                        true = encode_frames(native,np.asarray(frames)).to(device)
                        actions = action_embeddings(host,np.concatenate((past,control)))
                        free = menu_forecast(host,window[None,None],actions[None,None,None],offsets)[0,0,0]
                        reached = torch.cat((window,true[:-1]), dim=0).unfold(0,3,1).transpose(-1,-2)
                        ac = actions.unfold(0,3,1).transpose(-1,-2)
                        l = conditional_step(host,reached[:,None].expand(-1,2,-1,-1),
                                             ac[:,None].expand(-1,2,-1,-1),offsets).transpose(0,1)
                        fcost = scores(host,window[-1],free,goal)
                        replayed = float(p @ window.new_tensor(fcost))
                        np.testing.assert_allclose(replayed,row['choices']['written']['scores'][idx],rtol=1e-4,atol=.005)
                        decoded_c, decoded_f, decoded_l = [native.decode(z).cpu().numpy().tolist() for z in
                                                            (true,free[explain],l[explain])]
                        result = {'task':task.name,'physical_mode':mode,'role':roles[0],'roles':roles,'candidate':idx,
                                       'support': p.cpu().tolist(),
                                       'chosen_explanation':explain,'is_fitting_nominal_slot':idx in (0,192),
                                       'scores':{'exact_goal':scores(host,window[-1],true[None],goal)[0],
                                                 'free_mixture':replayed,'free_branch':fcost[explain],
                                                 'actual_window_branch':scores(host,window[-1],l[explain:explain+1],goal)[0]},
                                       'free_last2_score_by_branch': fcost,
                                       'actual_window_last2_score_by_branch': scores(host,window[-1],l,goal),
                                       'free_vs_actual_window_l2_by_horizon': (native.decode(free)-native.decode(l)).norm(dim=-1).cpu().tolist(),
                                       'free_vs_true_latent_l2_by_horizon': (free-true).norm(dim=-1).cpu().tolist(),
                                       'actual_window_vs_true_latent_l2_by_horizon': (l-true).norm(dim=-1).cpu().tolist(),
                                       'physical_cost':float(archive['candidate_cost_by_mode'][i,mode,idx]),
                                       'true_crossings':actual_crossings(np.vstack((task.start,positions)),physics),
                                       'true_xy':endpoint.tolist(),'C':decoded_c,'L':decoded_l,'F':decoded_f}
                        if baseline_host is not None:
                            baseline = baseline_trace(baseline_host, baseline_offsets, native, window, actions,
                                                      reached, ac, goal, baseline_row['probabilities'])
                            np.testing.assert_allclose(baseline['free_mixture'],
                                                       baseline_row['choices']['written']['scores'][idx],
                                                       rtol=1e-4, atol=.005)
                            result['baseline_same_program'] = baseline
                        report.append(result)
    finally:
        env.close()
    compact = []
    for row in report:
        if 'chosen' not in row['roles'] and 'baseline_failure_candidate' not in row['roles']:
            continue
        explanation = row['chosen_explanation']
        difference = row['free_vs_actual_window_l2_by_horizon'][explanation]
        item = {'task': row['task'], 'physical_mode': row['physical_mode'],
                'roles': row['roles'], 'candidate': row['candidate'], 'scores': row['scores'],
                'physical_cost': row['physical_cost'], 'true_crossings': row['true_crossings'],
                'F_L_position_distance_H5_H12': [difference[4], difference[11]]}
        if 'baseline_same_program' in row:
            baseline = row['baseline_same_program']
            gap = baseline['free_vs_actual_window_l2_by_horizon'][baseline['chosen_explanation']]
            item['baseline_scores'] = {key: baseline[key] for key in ('free_mixture', 'actual_window_mixture')}
            item['baseline_F_L_position_distance_H5_H12'] = [gap[4], gap[11]]
        compact.append(item)
    write_json(args.directory / 'fixed_regret_trace.json', {
        'checkpoint_directory': str(args.directory.resolve()),
        'baseline_directory': str(args.baseline_directory.resolve()) if args.baseline_directory else None,
        'trace_scope': 'Initial retained fork; fixed existing candidate controls; both physical conditions.',
        'selected_and_baseline_failures': compact, 'rows': report})
    print(f'Traced {len(report)} frozen-weight plans with exact H1–H12 actual controls.',flush=True)


if __name__=='__main__':
    main()
