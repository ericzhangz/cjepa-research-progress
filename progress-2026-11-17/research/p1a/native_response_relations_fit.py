"""Fit the §20 relational predictor and its matched native control on the recorded PushT panel."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

if __package__:
    from . import native_response_mixture_regulation as data
    from .native_response_action_effects import load_coordinates, save_split
    from .native_response_observation import _log_density
    from .native_response_relations_intermediates import QUERY_STEP, SOURCE_STEPS, _full_sigma, _source, _source_trace
else:
    import native_response_mixture_regulation as data
    from native_response_action_effects import load_coordinates, save_split
    from native_response_observation import _log_density
    from native_response_relations_intermediates import QUERY_STEP, SOURCE_STEPS, _full_sigma, _source, _source_trace


STEPS = (QUERY_STEP,)
UPDATES = data.REANCHORED_UPDATES
SEED = data.REANCHORED_SEED
ARM_NAMES = ('relational', 'native')
OUTPUT = data.RUNS / 'native_response_relations_20261114/action_effects_20261115'


def _query_paths(model, panel, rows, relational):
    history, past, candidates = data._query_inputs(panel, rows, STEPS)
    past = model.action_encoder(past)
    actions = model.action_encoder(candidates.flatten(1, 2)).unflatten(1, candidates.shape[1:3])
    if relational:
        return model.rollout_response_modes(history, past, actions)
    return native_rollout(model, history, past, actions).unsqueeze(-2)


def native_rollout(model, history, past, actions):
    """Continue one native window per candidate with no factual observations."""
    batch, count, horizon, _ = actions.shape
    length = history.size(1)
    controls = torch.cat((past[:, None].expand(-1, count, -1, -1), actions), dim=2).flatten(0, 1)
    window = history[:, None].expand(-1, count, -1, -1).flatten(0, 1)
    trajectory = []
    for step in range(horizon):
        predicted = model.predict(window, controls[:, step:step + length])[:, -1]
        trajectory.append(predicted)
        window = torch.cat((window[:, 1:], predicted[:, None]), dim=1)
    return torch.stack(trajectory, dim=1).reshape(batch, count, horizon, -1)


def _source_loss(model, panel, rows, sigma, relational):
    if relational:
        _, log_p, sources, _ = _source_trace(model, panel, rows, sigma)
        likelihood = tuple(source['log_likelihood'] for source in sources)
        log_evidence = [
            (source['log_support_before'] + ll).logsumexp(-1)
            for source, ll in zip(sources, likelihood, strict=True)
        ]
        return -torch.stack(log_evidence).mean() / sigma.numel(), log_p, likelihood

    visible, commands = data._factual_inputs(panel, rows)
    actions = model.action_encoder(commands)
    likelihood = []
    for step in SOURCE_STEPS:
        window, controls, arrived = _source(visible, actions, step)
        issued = model.predict(window, controls)[:, -1:, :]
        likelihood.append(_log_density(arrived, issued, sigma))
    return -torch.stack(likelihood).mean() / sigma.numel(), visible.new_zeros(len(visible), 1), tuple(likelihood)


def _forecast_energy(paths, panel, rows, coordinates, scales):
    latent, xy, valid = (value.flatten(0, 3) for value in data._targets(panel, rows, STEPS))
    return data._fan_energy(paths, latent, xy, valid, coordinates, scales)


def _preflight(model, panel, sigma, coordinates, scales):
    rows = slice(0, 1)
    paths = _query_paths(model, panel, rows, True)
    with torch.no_grad():
        native = _query_paths(model, panel, rows, False)
        torch.testing.assert_close(paths[..., 0, :], native[..., 0, :], atol=2e-5, rtol=2e-5)
    obs, log_p, likelihood = _source_loss(model, panel, rows, sigma, True)
    latent, xy, valid = (value.flatten(0, 3) for value in data._targets(panel, rows, STEPS))
    energy, roots = data._fan_energy(paths[:, :, 1:], latent[:, :, 1:], xy[:, :, 1:], valid[:, :, 1:],
                                    coordinates, tuple(scale[1:] for scale in scales))
    use = data._use_loss(log_p, energy, roots)
    relation = model.response_mixture.relation_directions
    source_grads = torch.autograd.grad(use, likelihood, retain_graph=True)
    direction_grads = [
        torch.autograd.grad(ll, relation, grad_outputs=grad, retain_graph=True)[0]
        for ll, grad in zip(likelihood, source_grads, strict=True)
    ]
    diagnostic = {
        'native_h1_h2_max_abs_error': (paths[..., 0, :] - native[..., 0, :]).abs().max().item(),
        'h2_to_each_source_likelihood': [grad.norm().item() for grad in source_grads],
        'h2_via_each_source_to_relation': [grad.norm().item() for grad in direction_grads],
        'observation_nll': obs.item(),
        'h2_fan_use_loss': use.item(),
    }
    if not all(np.isfinite([obs.item(), use.item(), *diagnostic['h2_via_each_source_to_relation']])):
        raise RuntimeError(f'Nonfinite preflight: {diagnostic}')
    if not all(value > 0 for value in diagnostic['h2_via_each_source_to_relation']):
        raise RuntimeError(f'H2 did not differentiate through both real writes: {diagnostic}')
    return diagnostic


def _fit(model, panel, sigma, coordinates, scales, schedule, output, relational):
    last = model.predictor.transformer.layers[-1]
    last.requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    parameters = [value for value in model.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    device = sigma.device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with (output / 'training.csv').open('x', encoding='utf-8', newline='') as handle:
        log = csv.writer(handle)
        log.writerow(('update', 'scene_id', 'loss', 'obs_nll_per_dimension', 'fan_energy_loss', 'gradient_norm'))
        for update, scene in enumerate(schedule, 1):
            rows = slice(scene, scene + 1)
            optimizer.zero_grad(set_to_none=True)
            obs, log_p, _ = _source_loss(model, panel, rows, sigma, relational)
            paths = _query_paths(model, panel, rows, relational)
            energy, roots = _forecast_energy(paths, panel, rows, coordinates, scales)
            use = data._use_loss(log_p, energy, roots)
            loss = obs + use
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss at update {update}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            log.writerow((update, int(panel.scene_ids[scene]), loss.item(), obs.item(), use.item(), norm.item()))
            handle.flush()
            if update == 1 or update % 32 == 0 or update == len(schedule):
                arm = 'relational' if relational else 'native'
                print(f'{arm} {update}/{len(schedule)} obs={obs.item():.5f} future={use.item():.5f}', flush=True)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    fit = {'updates': len(schedule), 'seconds': time.perf_counter() - start,
           'trainable_parameters': sum(value.numel() for value in parameters)}
    if device.type == 'cuda':
        fit['peak_allocated_mib'] = torch.cuda.max_memory_allocated(device) / 2**20
    state = {'mechanism': 'relational' if relational else 'native', 'last_block': last.state_dict(),
             'pred_proj': model.pred_proj.state_dict(), 'full_sigma': sigma, 'object_coordinates': coordinates}
    if relational:
        state['response_mixture'] = model.response_mixture.state_dict()
    torch.save(state, output / 'model.pt')
    return fit


@torch.no_grad()
def _save_native(model, panel, sigma, coordinates, output):
    coordinates = tuple(value.to(sigma) for value in coordinates)
    predictions, likelihood = [], []
    for first in range(0, len(panel.scene_ids), 2):
        rows = slice(first, first + 2)
        paths = _query_paths(model, panel, rows, False)[..., 0, :].double()
        w, bias = (value.double() for value in coordinates)
        predictions.append((paths @ w + bias).cpu().numpy())
        _, _, ll = _source_loss(model, panel, rows, sigma, False)
        likelihood.append(torch.cat(ll, dim=-1).cpu().numpy())
    roots = tuple(panel.visible.shape[:3])
    np.savez_compressed(output, scene_ids=panel.scene_ids,
                        prediction_xy=np.concatenate(predictions).reshape(*roots, 4, 2, 2),
                        source_nll=np.concatenate(likelihood).reshape(*roots, len(SOURCE_STEPS)) / -sigma.numel())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    if not (args.output / 'zero_step' / 'initialization.json').is_file():
        raise FileNotFoundError('Run the step-125 zero-update measurement first')
    for arm in ARM_NAMES:
        if (args.output / arm).exists():
            raise FileExistsError(args.output / arm)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(SEED)
    train = data.load_data('train', data.ARCHIVE_DIR, args.device)
    development = data.load_data('development', data.ARCHIVE_DIR, args.device)
    if np.intersect1d(train.scene_ids, development.scene_ids).size:
        raise ValueError('Fit and development scenes must be disjoint')
    for split in (train, development):
        if not all(split.valid[..., step // 5 - 1].all() for step in SOURCE_STEPS):
            raise ValueError('Missing executed source transitions')
        for step in SOURCE_STEPS:
            if not torch.equal(split.visible[..., step // 5, :], split.full[..., step // 5, :]):
                raise ValueError(f'Expected the declared public full frame at {step}')

    coordinates = load_coordinates()
    scales = data._reanchored_scales(train, STEPS)
    schedule = np.random.default_rng(SEED).integers(len(train.scene_ids), size=UPDATES).tolist()
    model = data._native_model(args.device)
    sigma, calibration_count = _full_sigma(model, train)
    for split, panel in (('train', train), ('development', development)):
        _save_native(model, panel, sigma, coordinates, args.output / f'official_{split}.npz')
    model.enable_response_mixture(parameterization='relational', num_modes=4)
    preflight = _preflight(model, train, sigma, coordinates, scales)
    print(json.dumps(preflight, allow_nan=False), flush=True)
    fits = {}
    for arm in ARM_NAMES:
        relational = arm == 'relational'
        if not relational:
            model = data._native_model(args.device)
        destination = args.output / arm
        destination.mkdir()
        fits[arm] = _fit(model, train, sigma, coordinates, scales, schedule, destination, relational)
        for split, panel in (('train', train), ('development', development)):
            if relational:
                save_split(model, panel, sigma, coordinates, destination, split)
            else:
                _save_native(model, panel, sigma, coordinates, destination / f'{split}.npz')
        del model
    summary = {'seed': SEED, 'learning_rate': 1e-4, 'batch_scenes': 1, 'calibration_count': calibration_count,
               'fit_scene_ids': train.scene_ids.tolist(), 'development_scene_ids': development.scene_ids.tolist(),
               'scene_schedule': [int(train.scene_ids[i]) for i in schedule], 'arms': fits, 'preflight': preflight}
    (args.output / 'fit_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(fits, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
