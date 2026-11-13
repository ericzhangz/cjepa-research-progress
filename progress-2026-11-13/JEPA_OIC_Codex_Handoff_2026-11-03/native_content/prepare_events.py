"""Finite native-physics development panel for history-conditioned JEPA content.

Only obs_rgb, mask, executed_actions and query_actions are deployment inputs.
Full images, future images, physical states and outcomes are offline evidence.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'oic_step1'))
from prepare_visual_contact import scene
from prepare_pusht_continuation import environment


OBS_STEPS = np.arange(0, 31, 5)
QUERY_STEPS = np.array([10, 15, 20, 30])
BACKGROUND = np.rint(255 * np.array([.485, .456, .406])).astype(np.uint8)


def public_masks(cameras, visibility='original'):
    masks = np.zeros((7, 96, 96), dtype=bool)
    for index, camera, width in [(0, 0, 19), (1, 0, 19), (2, 1, 15), (3, 2, 3)]:
        x, y = np.rint(cameras[camera] * 96 / 512).astype(int)
        radius = width // 2
        if not (radius <= x < 96 - radius and radius <= y < 96 - radius):
            raise ValueError('Public camera extends outside the original canvas')
        masks[index, y-radius:y+radius+1, x-radius:x+radius+1] = True
    masks[4:, :12] = True
    if visibility == 'minimal':
        masks[2] = masks[4]
    elif visibility != 'original':
        raise ValueError('Fresh physics panels use the original or fixed minimal sensor')
    return masks


def physical_state(env):
    return np.array([*env.agent.position, *env.agent.velocity, *env.block.position,
                     *env.block.velocity, env.block.angle, env.block.angular_velocity])


def simulate(env, reset, blocks):
    state, _ = env.reset(seed=0, options={'reset_to_state': reset})
    done = False
    terminal_step = -1
    states = [physical_state(env)]
    coverages = [env._get_coverage()]
    frames = [env._render(visualize=False)]
    terminated = [False]
    controls, was_executed, contacts, contact_points = [], [], [], []
    for nominal_step, relative in enumerate(blocks.reshape(-1, 2), start=1):
        target = np.full(2, np.nan)
        hit, points, executed = 0, 0, False
        if not done:
            target = state[:2] + 100 * relative
            state, _, done, truncated, info = env.step(target)
            if truncated:
                raise RuntimeError('Unexpected native environment truncation')
            executed = True
            hit, points = info['n_contacts'], env.n_contact_points
            if done:
                terminal_step = nominal_step
        controls.append(target)
        was_executed.append(executed)
        contacts.append(hit)
        contact_points.append(points)
        states.append(physical_state(env))
        coverages.append(env._get_coverage())
        terminated.append(done)
        if nominal_step % 5 == 0:
            frames.append(env._render(visualize=False))
    return dict(state=np.array(states), coverage=np.array(coverages), rgb=np.stack(frames),
                controls=np.array(controls), was_executed=np.array(was_executed),
                contacts=np.array(contacts), contact_points=np.array(contact_points),
                terminated=np.array(terminated), terminal_step=terminal_step)


def pixel_groups(frames, controls):
    """Exact visible evidence and public executed relative commands, within scene."""
    keys = {}
    labels = []
    for frame in frames:
        key = frame.tobytes() + controls.tobytes()
        labels.append(keys.setdefault(key, len(keys)))
    return np.array(labels, dtype=np.int16)


def conditional(values, groups):
    return np.stack([values[groups == label].mean(axis=0) for label in groups])


def generate(count, seed=11032026, first_id=100000, visibility='original'):
    angles = np.random.default_rng(seed).uniform(-np.pi, np.pi, count)
    rgb = (96, 96, 3)
    p = dict(scene_ids=np.arange(first_id, first_id+count), psi=angles,
             generation_seed=np.array(seed),
             observation_steps=OBS_STEPS, query_steps=QUERY_STEPS,
             executed_actions=np.empty((count, 6, 5, 2)),
             query_actions=np.empty((count, 4, 2, 5, 2)),
             mask=np.empty((count, 7, 96, 96), bool),
             obs_rgb=np.empty((count, 4, 7, *rgb), np.uint8),
             full_rgb=np.empty((count, 4, 7, *rgb), np.uint8),
             future_rgb=np.empty((count, 4, 4, 4, 2, *rgb), np.uint8),
             coverage=np.empty((count, 4, 4, 4)),
             reset_state=np.empty((count, 4, 5)),
             physical_state=np.empty((count, 4, 31, 10)),
             physical_coverage=np.empty((count, 4, 31)),
             absolute_control_targets=np.empty((count, 4, 30, 2)),
             actual_step=np.empty((count, 4, 30), bool),
             contacts=np.empty((count, 4, 30), np.int32),
             contact_points=np.empty((count, 4, 30), np.int32),
             terminated=np.empty((count, 4, 31), bool),
             query_terminated=np.empty((count, 4, 4), bool),
             query_terminal_step=np.full((count, 4, 4), -1, np.int16),
             query_physical_state=np.empty((count, 4, 4, 4, 11, 10)),
             query_physical_coverage=np.empty((count, 4, 4, 4, 11)),
             query_absolute_control_targets=np.empty((count, 4, 4, 4, 10, 2)),
             query_actual_step=np.empty((count, 4, 4, 4, 10), bool),
             query_contacts=np.empty((count, 4, 4, 4, 10), np.int32),
             query_contact_points=np.empty((count, 4, 4, 4, 10), np.int32),
             query_future_terminated=np.empty((count, 4, 4, 4, 11), bool))
    env = environment()
    start = time.perf_counter()
    try:
        for s, angle in enumerate(angles):
            reset, prefix, candidates, cameras = scene(angle)
            normal = np.array([-np.sin(angle), np.cos(angle)])
            extra = np.repeat(np.stack([np.zeros(2), .10*normal, .10*normal,
                                        np.zeros(2)])[:, None], 5, axis=1)
            executed = np.concatenate([prefix, extra])
            masks = public_masks(cameras, visibility)
            p['mask'][s], p['executed_actions'][s] = masks, executed
            p['query_actions'][s], p['reset_state'][s] = candidates, reset
            for w in range(4):
                base = simulate(env, reset[w], executed)
                p['full_rgb'][s, w] = base['rgb']
                p['obs_rgb'][s, w] = np.where(masks[..., None], base['rgb'], BACKGROUND)
                for target, source in [('physical_state','state'), ('physical_coverage','coverage'),
                                       ('absolute_control_targets','controls'), ('actual_step','was_executed'),
                                       ('contacts','contacts'), ('contact_points','contact_points'),
                                       ('terminated','terminated')]:
                    p[target][s, w] = base[source]
                for q, step in enumerate(QUERY_STEPS):
                    blocks = step // 5
                    p['query_terminated'][s, w, q] = base['terminated'][step]
                    if base['terminated'][step]:
                        p['query_terminal_step'][s, w, q] = base['terminal_step']
                    for a, candidate in enumerate(candidates):
                        branch = simulate(env, reset[w], np.concatenate([executed[:blocks], candidate]))
                        for key in ['state', 'terminated']:
                            if not np.array_equal(branch[key][:step+1], base[key][:step+1]):
                                raise RuntimeError(f'Query prefix changed {key} at scene {s}, world {w}')
                        # GEOS area arithmetic can differ at machine precision when native
                        # block.shapes is traversed in another order after env.reset().
                        if not np.allclose(branch['coverage'][:step+1], base['coverage'][:step+1],
                                           rtol=0, atol=1e-12):
                            raise RuntimeError('Query prefix changed physical coverage')
                        for key in ['controls', 'was_executed', 'contacts', 'contact_points']:
                            if not np.array_equal(branch[key][:step], base[key][:step], equal_nan=True):
                                raise RuntimeError(f'Query prefix changed {key}')
                        if not np.array_equal(branch['rgb'][:blocks+1], base['rgb'][:blocks+1]):
                            raise RuntimeError('Query changed an earlier rendered frame')
                        p['future_rgb'][s, w, q, a] = branch['rgb'][-2:]
                        p['coverage'][s, w, q, a] = branch['coverage'][-1]
                        for target, source in [('query_physical_state','state'),
                                               ('query_physical_coverage','coverage'),
                                               ('query_future_terminated','terminated')]:
                            p[target][s, w, q, a] = branch[source][step:]
                        for target, source in [('query_absolute_control_targets','controls'),
                                               ('query_actual_step','was_executed'),
                                               ('query_contacts','contacts'),
                                               ('query_contact_points','contact_points')]:
                            p[target][s, w, q, a] = branch[source][step:]
            print(json.dumps(dict(scene_id=int(p['scene_ids'][s]),
                                  completed=s+1, wall_seconds=time.perf_counter()-start)), flush=True)
    finally:
        env.close()
    return p


def identify(p):
    n = len(p['scene_ids'])
    groups = {name: np.empty((n, 4, 4), np.int16)
              for name in ['history', 'window3', 'latest', 'without_t15']}
    values = {name: np.empty_like(p['coverage']) for name in groups}
    summary = {}
    for q, step in enumerate(QUERY_STEPS):
        end = step // 5 + 1
        spans = dict(history=np.arange(end), window3=np.arange(max(0, end-3), end),
                     latest=np.array([end-1]), without_t15=np.array([i for i in range(end) if i != 3]))
        for s in range(n):
            for name, span in spans.items():
                group = pixel_groups(p['obs_rgb'][s][:, span], p['executed_actions'][s, :end-1])
                groups[name][s, :, q] = group
                values[name][s, :, q] = conditional(p['coverage'][s, :, q], group)
        full = values['history'][:, :, q]
        optimal = full >= full.max(-1, keepdims=True)-1e-12
        row = dict(mean_candidate_coverage_range=float(np.ptp(p['coverage'][:, :, q], axis=-1).mean()),
                   terminated_worlds=int(p['query_terminated'][:, :, q].sum()),
                   history_groups=int(sum(len(np.unique(g[:, q])) for g in groups['history'])))
        for name in ['window3', 'latest', 'without_t15']:
            weaker = values[name][:, :, q]
            weaker_opt = weaker >= weaker.max(-1, keepdims=True)-1e-12
            changed = ~(optimal & weaker_opt).any(-1)
            selected = np.take_along_axis(full, weaker.argmax(-1)[..., None], axis=-1)[..., 0]
            excess = full.max(-1)-selected
            # Conditional means are the Bayes square-loss predictors for each evidence class.
            separation = ((full-weaker)**2).mean(-1)
            row[name] = dict(action_change_mass=float(changed.mean()),
                             conditional_regret_from_dropped_evidence=float(excess.mean()),
                             conditional_action_value_mse=float(separation.mean()),
                             changed_worlds=int(changed.sum()),
                             changed_scenes=int(changed.any(-1).sum()),
                             groups=int(sum(len(np.unique(g[:, q])) for g in groups[name])))
        summary[str(step)] = row
    for name in groups:
        p['group_'+name] = groups[name]
        p['values_'+name] = values[name]
    result = dict(seed=int(p.get('generation_seed', 11032026)), scenes=n, worlds=4*n, query_results=summary,
                  final_obs_exact_same_per_scene=bool(all(np.array_equal(x, np.repeat(x[:1], 4, axis=0))
                                                         for x in p['obs_rgb'][:, :, 4:])),
                  initial_observed_worlds=int(sum(len(np.unique([x.tobytes() for x in scene_obs[:, :2]]))
                                                 for scene_obs in p['obs_rgb'])),
                  main_prefix_contact_worlds=int((p['contact_points'][:, :, :10] > 0).any(-1).sum()),
                  post_event_actual_contact_worlds=int((p['contact_points'][:, :, 15:] > 0).any(-1).sum()),
                  executed_steps=int(p['actual_step'].sum()),
                  query_executed_steps=int(p['query_actual_step'].sum()))
    return result


def visualize(p, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for s in range(min(4, len(p['scene_ids']))):
        fig, axes = plt.subplots(4, 7, figsize=(14, 8), constrained_layout=True)
        for row, (world, field) in enumerate([(0, 'full_rgb'), (0, 'obs_rgb'),
                                              (2, 'full_rgb'), (2, 'obs_rgb')]):
            for col, step in enumerate(OBS_STEPS):
                axes[row, col].imshow(p[field][s, world, col])
                axes[row, col].set_title(f'w{world} {field[:-4]} t{step}', fontsize=9)
                axes[row, col].axis('off')
        fig.suptitle(f'Scene {p["scene_ids"][s]} | visible canvas and offline full teacher')
        fig.savefig(out / f'scene_{p["scene_ids"][s]}.png', dpi=110)
        plt.close(fig)


def revise_sensor(visibility):
    """Apply a recorded development visibility change to identical physical worlds."""
    source = ROOT / 'runs/development_11032026.npz'
    with np.load(source, allow_pickle=False) as stored:
        p = {key: stored[key].copy() for key in stored.files}
    for s, angle in enumerate(p['psi']):
        _, _, _, cameras = scene(angle)
        original_masks = public_masks(cameras)
        p['mask'][s, 2] = original_masks[4]
        if visibility == 'revised':
            p['mask'][s, 3] = original_masks[2]
        p['obs_rgb'][s] = np.where(p['mask'][s, None, ..., None], p['full_rgb'][s], BACKGROUND)
    return p


def check_confirmation_ready():
    """Open confirmation only after the reviewed final checkpoints are fixed."""
    fit = ROOT / 'runs/pilot_2026-11-03'
    ready = json.loads((fit / 'CONFIRMATION_READY.json').read_text(encoding='utf-8'))
    if ready.get('cr4_passed') is not True or ready.get('training_steps') != 200:
        raise ValueError('Confirmation requires passed CR4 and fixed 200-step training')
    expected = {f'{kind}_{seed}.pt' for kind in
                ('initializer', 'conditional', 'state_aware', 'compressed', 'residual')
                for seed in (11, 22, 33)}
    hashes = ready.get('weights_sha256', {})
    if set(hashes) != expected:
        raise ValueError('Confirmation readiness must identify all 15 final checkpoints')
    for name in sorted(expected):
        path = fit / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f'Final checkpoint changed after readiness: {name}')
        training = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        trace = training.get('trace', [])
        if len(trace) != 200 or trace[-1].get('step') != 200:
            raise ValueError(f'Checkpoint lacks a completed 200-step trace: {name}')
        if not name.startswith('initializer_') and training.get('steps') != 200:
            raise ValueError(f'Writer training budget is not fixed at 200: {name}')
    protocol_path = ROOT / 'native_content/pilot_protocol.json'
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != ready.get('protocol_sha256'):
        raise ValueError('Pilot protocol changed after readiness')
    spec = json.loads(protocol_path.read_text(encoding='utf-8'))['data']['confirmation']
    if (spec['count'], spec['seed'], spec['first_scene_id']) != (48, 11034026, 102000):
        raise ValueError('Confirmation sampling differs from the fixed pilot protocol')
    return ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['development', 'train', 'confirmation'], default='development')
    parser.add_argument('--count', type=int, choices=[4, 16], default=16)
    changes = parser.add_mutually_exclusive_group()
    changes.add_argument('--revise-sensor', action='store_true')
    changes.add_argument('--minimal-sensor', action='store_true')
    args = parser.parse_args()
    out = ROOT / 'runs'
    out.mkdir(parents=True, exist_ok=True)
    name = 'development_11032026' if args.count == 16 else 'development_smoke4_11032026'
    visibility = 'revised' if args.revise_sensor else 'minimal' if args.minimal_sensor else 'original'
    readiness = None
    if args.phase in ('train', 'confirmation'):
        if args.revise_sensor or args.minimal_sensor or args.count != 16:
            raise ValueError('Train and confirmation use the fixed minimal sensor without extra selectors')
        name = 'train_11033026' if args.phase == 'train' else 'confirmation_11034026'
        visibility = 'minimal'
        if args.phase == 'confirmation':
            readiness = check_confirmation_ready()
    elif visibility != 'original':
        if args.count != 16:
            raise ValueError('Sensor changes apply to the full fixed development panel')
        name += '_' + visibility
    target = out / (name + '.npz')
    if target.exists():
        raise FileExistsError(target)
    started = time.perf_counter()
    if args.phase == 'train':
        panel = generate(96, seed=11033026, first_id=101000, visibility='minimal')
    elif args.phase == 'confirmation':
        panel = generate(48, seed=11034026, first_id=102000, visibility='minimal')
        if check_confirmation_ready() != readiness:
            raise ValueError('Confirmation readiness changed during physical generation')
    else:
        panel = revise_sensor(visibility) if visibility != 'original' else generate(args.count)
    result = identify(panel)
    result['wall_seconds_before_save'] = time.perf_counter() - started
    result['deployment_inputs'] = ['obs_rgb', 'mask', 'executed_actions', 'query_actions']
    result['offline_only'] = 'full_rgb, future_rgb, physical fields, absolute target controls, outcomes and grouping'
    result['physical_state_columns'] = ['agent_x', 'agent_y', 'agent_vx', 'agent_vy',
                                         'block_x', 'block_y', 'block_vx', 'block_vy', 'block_angle', 'block_omega']
    result['time_semantics'] = 't0 is after native reset integration; each block is 5 native control steps. Success is absorbing, skipped steps retain state and image.'
    result['scope'] = ('Fixed training set, disjoint seed and scene IDs from development. '
                       'Equal mass across four native reset worlds per scene, fixed minimal visibility, '
                       'no model-based filtering or confirmation.') if args.phase == 'train' else (
                       'Finite development only. Equal mass across four native reset worlds per scene. '
                       'No model-based filtering, training, or confirmation.')
    if args.phase == 'confirmation':
        result['scope'] = ('First fixed independent confirmation generation after CR4, 15 final checkpoints '
                           'and pilot protocol fixed. Equal four-world mass, no scene filtering.')
        result['confirmation_ready'] = readiness
    sensor_rules = {
        'original': 'Original t10 camera1 width15 and t15 camera2 width3.',
        'revised': 'First development revision changes t10 to top12 strip and t15 to camera1 width15.',
        'minimal': 'Third development version changes only t10 to top12 strip; original t15 camera2 width3 retained.'}
    result['sensor_revision'] = dict(applied=visibility != 'original', version=visibility,
        reason='Original t10 camera already separates all worlds, making t15 observation redundant.',
        rule=sensor_rules[visibility],
        selection_basis='Exact visible-evidence groups and true outcomes; no writer training or writer scores.',
        source='development_11032026_minimal.npz sensor contract' if args.phase != 'development' else
               'development_11032026.npz' if visibility != 'original' else None,
        protocol_deviation=visibility == 'minimal' and args.phase == 'development',
        deviation_reason='Root authorized correction of the prior two-variable sensor change beyond the original one-revision limit. Development only; stop further visibility search.' if visibility == 'minimal' and args.phase == 'development' else None)
    with target.open('xb') as f:
        np.savez_compressed(f, **panel)
    (out / (name + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    pictures = out / name
    pictures.mkdir(exist_ok=True)
    visualize(panel, pictures)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
