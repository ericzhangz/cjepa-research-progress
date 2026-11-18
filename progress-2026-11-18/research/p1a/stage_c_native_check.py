"""No-training native-host and anti-collapse checks for Stage C-0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .native_response_relations_fit import native_rollout
from .native_topology_condition import action_embeddings
from .stage_a_fourroom import (
    ACTION_BLOCK, GOAL_RADIUS, MAX_GEOMETRY_RATE, ROOT, Native, OfficialFourRoom, OpenPhysics,
    candidate_bank, fixed_tasks, geometry_diagnostics, protocol, write_json,
)


def canonical_window(window, projection, center):
    return center + (window - center) @ projection


def native_conditioned(host, window, controls, offsets):
    """Use the host's four blocks, with AdaLN offsets at the latest token."""
    predictor = host.predictor
    transformer = predictor.transformer
    state = transformer.input_proj(predictor.dropout(window + predictor.pos_embedding[:, :window.size(1)]))
    condition = transformer.cond_proj(controls)
    for block, offset in zip(transformer.layers, offsets, strict=True):
        altered = torch.cat((condition[:, :-1], condition[:, -1:] + offset), dim=1)
        state = block(state, altered)
    output = transformer.output_proj(transformer.norm(state))
    return host.pred_proj(output.flatten(0, 1)).unflatten(0, window.shape[:2])


def anchored_step(host, window, controls, projection, center, offsets):
    base = host.predict(window, controls)[:, -1]
    canonical = canonical_window(window, projection, center)
    reference = host.predict(canonical, controls)[:, -1]
    conditional = native_conditioned(host, canonical, controls, offsets)[:, -1]
    return base, (conditional - reference) @ projection


def rollout(host, root, controls, projection, center, offsets=None):
    """Run canonical-only F0, or Eq. 49.1 when AdaLN offsets are provided."""
    window = root.expand(len(controls), -1, -1)
    forecast = []
    for step in range(controls.size(1) - window.size(1) + 1):
        command = controls[:, step:step + window.size(1)]
        if offsets is None:
            predicted = host.predict(canonical_window(window, projection, center), command)[:, -1]
        else:
            base, revision = anchored_step(host, window, command, projection, center, offsets)
            predicted = base + revision
        forecast.append(predicted)
        window = torch.cat((window[:, 1:], predicted[:, None]), dim=1)
    return torch.stack(forecast, dim=1)


def summarize(rows):
    return dict(zip(('median', 'p90', 'maximum'), np.quantile(rows, (.5, .9, 1.)).tolist(), strict=True))


@torch.no_grad()
def factual_windows(host, archive, projection, center):
    device = projection.device
    reports = {}
    for kind in ('bg', 'lock', 'action', 'open'):
        z = torch.as_tensor(archive[f'check_{kind}_z'], device=device)
        commands = torch.as_tensor(archive[f'check_{kind}_a'], device=device)
        if z.ndim == 4:
            if commands.ndim == 4:
                commands = commands[:, None].expand(-1, z.size(1), -1, -1, -1)
            z, commands = z.flatten(0, 1), commands.flatten(0, 1)
        commands = action_embeddings(host, commands)
        raw, canonical = [], []
        for block_z, block_commands in zip(z.split(128), commands.split(128), strict=True):
            window = block_z[:, :-1]
            raw.append(host.predict(window, block_commands)[:, -1])
            shifted = canonical_window(window, projection, center)
            canonical.append(host.predict(shifted, block_commands)[:, -1])
        raw, canonical = torch.cat(raw), torch.cat(canonical)
        change = canonical - raw
        target = z[:, -1]
        reports[kind] = {
            'n_branch_windows': len(z),
            'raw_shift_l2': summarize(change.norm(dim=-1).cpu().numpy()),
            'projected_shift_l2': summarize((change @ projection).norm(dim=-1).cpu().numpy()),
            'raw_factual_rmse': float((raw - target).square().mean().sqrt()),
            'canonical_factual_rmse': float((canonical - target).square().mean().sqrt()),
            'raw_projected_factual_rmse': float(((raw - target) @ projection).square().mean().sqrt()),
            'canonical_projected_factual_rmse': float(((canonical - target) @ projection).square().mean().sqrt()),
        }
    return reports


@torch.no_grad()
def all_open_replay(native, stage_a, projection, center):
    """Reuse actual roots and the already selected H12 action sequences."""
    pinned = json.loads(stage_a.read_text(encoding='utf-8'))
    if pinned['protocol'] != protocol():
        raise ValueError('Stage A archive does not match the fixed protocol')
    rows = []
    env = OfficialFourRoom()
    try:
        physics = OpenPhysics(env)
        for task, archived in zip(fixed_tasks(), pinned['qualification']['tasks'], strict=True):
            if task.name != archived['task']:
                raise ValueError('Fixed Stage A task order changed')
            env.reset(seed=task.seed, options={'state': task.start, 'target_state': task.goal})
            nominal = np.asarray(task.start, dtype=np.float32)
            commands = [np.zeros((ACTION_BLOCK, 2), dtype=np.float32) for _ in range(2)]
            latents = [native.encode(env.render())]
            for command in commands:
                env.execute(command)
                latents.append(native.encode(env.render()))
            for decision in archived['decisions']:
                replan = decision['replan']
                plan = candidate_bank(task, nominal, replan)[decision['selected_index']]
                np.testing.assert_array_equal(plan[0], np.asarray(decision['raw_action_block'], dtype=np.float32))
                actual_xy = physics.rollout(env.xy, plan[None])[0, ACTION_BLOCK - 1::ACTION_BLOCK]
                raw_controls = np.concatenate((commands[-2:], plan))
                controls = action_embeddings(native.host, raw_controls[None])
                root = torch.stack(latents[-3:])[None]
                stock = native_rollout(native.host, root, controls[:, :2], controls[:, None, 2:])[0, 0]
                shifted = rollout(native.host, root, controls, projection, center)[0]
                origin = native.decode(root[0, -1]).cpu().numpy()
                native_xy, canonical_xy = native.decode(torch.stack((stock, shifted))).cpu().numpy()
                np.testing.assert_allclose(native_xy, decision['geometry']['predicted_xy'], rtol=1e-4, atol=.003)
                rows.append({
                    'task': task.name, 'family': task.family, 'replan': replan,
                    'candidate_index': decision['selected_index'], 'actual_xy': actual_xy.tolist(),
                    'goal': task.goal,
                    'stock': geometry_diagnostics(physics, origin, native_xy, plan),
                    'canonical': geometry_diagnostics(physics, origin, canonical_xy, plan),
                })
                path = env.execute(plan[0])
                np.testing.assert_array_equal(path, np.asarray(decision['actual_raw_xy'], dtype=np.float32))
                nominal = physics.rollout(nominal, plan[None, :1])[0, -1]
                commands.append(plan[0])
                latents.append(native.encode(env.render()))
    finally:
        env.close()

    truth = np.asarray([row['actual_xy'] for row in rows])
    goal = np.asarray([row['goal'] for row in rows])[:, None]
    true_goal = np.linalg.norm(truth - goal, axis=-1).min(axis=-1) < GOAL_RADIUS
    summary = {}
    for arm in ('stock', 'canonical'):
        predicted = np.asarray([row[arm]['predicted_xy'] for row in rows])
        error = ((predicted - truth) ** 2).sum(axis=-1)
        predicted_goal = np.linalg.norm(predicted - goal, axis=-1).min(axis=-1) < GOAL_RADIUS
        family_metrics = {}
        for family in ('one_door', 'two_doors'):
            mask = np.asarray([row['family'] == family for row in rows])
            family_metrics[family] = {
                'fixed_plans': int(mask.sum()),
                'physical_goal_reachable': int((mask & true_goal).sum()),
                'retained_goal_reach': int((mask & true_goal & predicted_goal).sum()),
                'false_goal_reach': int((mask & ~true_goal & predicted_goal).sum()),
                'position_rmse_by_horizon_px': np.sqrt(error[mask].mean(axis=0)).tolist(),
            }
        summary[arm] = {'position_rmse_by_horizon_px': np.sqrt(error.mean(axis=0)).tolist(),
                        'by_family': family_metrics}
        for indicator in ('solid_wall_endpoint_axes', 'false_cross_axes'):
            flags = np.asarray([row[arm][indicator] for row in rows]).any(-1)
            summary[arm][indicator] = {
                'count': int(flags.sum()), 'total': flags.size,
                'rate': float(flags.mean()), 'passes_stage_a_geometry_limit': bool(flags.mean() <= MAX_GEOMETRY_RATE),
            }
    return {'summary': summary, 'roots': rows}


def antisymmetric_jacobian(host, archive, projection, center, out):
    device = projection.device
    target = torch.as_tensor(archive['fit_bundle_z'], device=device)
    difference = (target[:, 0] - target[:, 1]) @ projection
    reports = []
    seed = None
    for index, difference_i in enumerate(difference):
        diff_norm = difference_i.norm()
        if not bool(diff_norm > 0):
            continue
        root = torch.as_tensor(archive['fit_bundle_root_z'][index], device=device)[None]
        past = np.broadcast_to(archive['fit_bundle_past'][index], (len(difference_i), 2, ACTION_BLOCK, 2))
        actions = np.concatenate((past, archive['fit_bundle_a'][index]), axis=1)
        controls = action_embeddings(host, actions)
        transformer = host.predictor.transformer
        condition = transformer.cond_proj(controls)
        # Each overlapping window exposes the latest action at these indices.
        condition_rms = condition[:, 2:].square().mean().sqrt()
        offsets = torch.zeros((len(transformer.layers), condition.size(-1)), device=device,
                              dtype=controls.dtype, requires_grad=True)
        base = native_rollout(host, root, controls[None, 0, :2], controls[None, :, 2:])[0]
        forecast = rollout(host, root, controls, projection, center, offsets)
        torch.testing.assert_close(forecast, base, rtol=0., atol=0.)
        contraction = (forecast.double() * difference_i.double()).sum(dim=-1).mean()
        (gradient,) = torch.autograd.grad(contraction, offsets)
        # RMS across real menu and horizon positions. No division by evidence scale.
        difference_rms = diff_norm / np.sqrt(difference_i.numel() / difference_i.size(-1))
        gradient_norm = gradient.norm()
        relative = gradient_norm / difference_rms
        if not bool(torch.isfinite(gradient).all()) or not bool(condition_rms > 0):
            raise ValueError(f'Non-finite Jacobian or zero condition scale for pair {index}')
        reports.append({
            'bundle_index': index,
            'projected_true_difference_token_rms': float(difference_rms),
            'native_condition_rms': float(condition_rms),
            'vjp_norm': float(gradient_norm),
            'vjp_over_difference_rms': float(relative),
            'vjp_per_layer_norm': gradient.norm(dim=-1).tolist(),
            'zero_offset_h12_max_abs_error': float((forecast.detach() - base).abs().max()),
        })
        if bool(relative > torch.finfo(offsets.dtype).eps):
            seed = (1e-3 * condition_rms * gradient / gradient_norm).detach()
            # The finite perturbation should reproduce the antisymmetric directional response.
            with torch.no_grad():
                positive = rollout(host, root, controls, projection, center, seed)
                negative = rollout(host, root, controls, projection, center, -seed)
                finite_difference = ((positive - negative).double() * difference_i.double()).sum(dim=-1).mean()
            expected = 2 * (gradient * seed).sum()
            reports[-1].update(seed_l2_norm=float(seed.norm()),
                               finite_antisymmetric_contraction=float(finite_difference),
                               first_order_antisymmetric_contraction=float(expected))
            np.savez_compressed(out / 'jacobian_init.npz', antisymmetric=seed.cpu().numpy(),
                                common=np.zeros_like(seed.cpu().numpy()), bundle_index=index)
            break
    return {'passed': seed is not None, 'pairs': reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage-a', type=Path, default=ROOT / 'runs/stage_a_fourroom_20261117/native.json')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    native = Native(device)
    with np.load(args.geometry) as geom:
        projection = torch.as_tensor(geom['P'], device=device, dtype=torch.float32)
        center = torch.as_tensor(geom['center'], device=device, dtype=torch.float32)
    args.out.mkdir(parents=True, exist_ok=True)
    with np.load(args.calibration) as archive:
        factual = factual_windows(native.host, archive, projection, center)
        jacobian = antisymmetric_jacobian(native.host, archive, projection, center, args.out)
    all_open = all_open_replay(native, args.stage_a, projection, center)
    write_json(args.out / 'native_check.json', {'factual_windows': factual, 'all_open_fixed_replay': all_open,
                                              'jacobian': jacobian})
    print(json.dumps({'factual_windows': factual, 'all_open': all_open['summary'], 'jacobian': jacobian}), flush=True)


if __name__ == '__main__':
    main()
