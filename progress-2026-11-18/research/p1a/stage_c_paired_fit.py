"""Train the shared §49.8 predictor and evidence update on legal paired histories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .native_topology_condition import action_embeddings, load_host
from .stage_a_fourroom import ROOT, write_json
from .stage_c_paired import (
    conditional_step, factual_support, menu_forecast, paired_energy, source_evidence, two_offsets,
)


def dataset(archive, host, device):
    read = torch.as_tensor(archive['fit_read_t'], device=device, dtype=torch.long)
    end = torch.arange(-2, 1, device=device)[None, None] + read[..., None]
    groups = torch.arange(len(read), device=device)[:, None, None]
    prefix_z = torch.as_tensor(archive['fit_prefix_z'], device=device)
    prefix_a = action_embeddings(host, archive['fit_prefix_a'])
    windows = prefix_z[groups, 0, end]
    past = prefix_a[groups, 0, end[..., :2]]
    menu = action_embeddings(host, archive['fit_menu_a'])
    controls = torch.cat((past[:, :, None].expand(*menu.shape[:3], *past.shape[-2:]), menu), dim=-2)
    truth = torch.as_tensor(archive['fit_target_z'], device=device)
    directed = torch.as_tensor(archive['fit_directed'], device=device)
    return prefix_z, prefix_a, windows, controls, truth, directed, read


@torch.no_grad()
def fit_alignment(host, z, actions, offsets, s2_obs, connectors):
    """Name the two explanations using only predeclared fitting probes."""
    rows = []
    for connector in connectors:
        group, arrival = connector['group'], connector['first_probe_arrival']
        names = connector['physical_row_order']
        if sorted(names) != ['bottom', 'top'] or arrival < 3:
            raise ValueError('Expected an actual first probe with two named fitting histories')
        window = z[group, :, arrival-3:arrival]
        commands = actions[group, :, arrival-3:arrival]
        torch.testing.assert_close(window[0], window[1], rtol=0., atol=0.)
        torch.testing.assert_close(commands[0], commands[1], rtol=0., atol=0.)
        forecast = conditional_step(host, window[0].expand(2, -1, -1),
                                    commands[0].expand(2, -1, -1), offsets)
        evidence = source_evidence(forecast, z[group, :, arrival], s2_obs)
        by_lock = dict(zip(names, evidence.tolist(), strict=True))
        rows.append({'group': group, 'arrival': arrival, 'direction': connector['direction'],
                     'near_root': connector['read_roots'][0],
                     'evidence_by_lock': by_lock, 'delta_top_minus_bottom': by_lock['top'] - by_lock['bottom']})
    mean_delta = float(np.mean([row['delta_top_minus_bottom'] for row in rows]))
    if not np.isfinite(mean_delta):
        raise FloatingPointError('Nonfinite fit-only alignment')
    mapping = [0, 1] if mean_delta > 0 else [1, 0] if mean_delta < 0 else None
    return {'physical_to_explanation': mapping, 'frozen_before_heldout': True,
            'source': 'Eight predeclared fitting bridge first probes only; physical order bottom/top, evaluation naming only',
            'mean_delta_top_minus_bottom': mean_delta, 'fitting_probe_pairs': rows}


@torch.no_grad()
def bridge_source_audit(host, z, actions, offsets, s2_obs, connectors):
    """Track each H1 event before looking at the held-out candidate bank."""
    rows = []
    for connector in connectors:
        g = connector['group']
        for endpoint, key in enumerate(('first_probe_arrival', 'second_probe_arrival')):
            t = connector[key]
            support, evidence, pending = factual_support(
                host, z[g, :, :t+1], actions[g, :, :t], offsets, s2_obs)
            error = (pending[:, -1] - z[g, :, t, None]).square().sum(-1) / (2 * s2_obs)
            rows.append({'group': g, 'endpoint': endpoint, 'direction': connector['direction'],
                         'arrival': t, 'physical_row_order_eval_only': connector['physical_row_order'],
                         'evidence': evidence[:, -1].tolist(), 'source_energy_by_history_and_branch': error.tolist(),
                         'support_before': support[:, t-1].tolist(), 'support_after': support[:, t].tolist()})
    return rows


def fit(directory, arm, updates, device, continuation=None, bridge_assignment=None):
    torch.manual_seed(2026113700)
    host = load_host(device)
    start = time.perf_counter()
    with np.load(directory / 'lifecycle.npz', allow_pickle=False) as archive:
        z, actions, windows, controls, truth, directed, read = dataset(archive, host, device)
        source_strength = np.square(archive['source_delta']).sum(-1) * .5
    predecessor = (json.loads((continuation / 'paired_unlabeled_fit.json').read_text(encoding='utf-8'))
                   if continuation else None)
    ordinary = len(directed)
    connectors = None
    if bridge_assignment:
        audit = json.loads((directory / 'bridge_data_audit.json').read_text(encoding='utf-8-sig'))
        connectors, ordinary = audit['connectors'], audit['original_groups']
        if ordinary >= len(directed) or not directed[ordinary:].all():
            raise ValueError('Expected appended directed bridge groups')
        if predecessor and (predecessor['arm'] != arm or predecessor['fitting_groups'] != ordinary):
            raise ValueError('Bridge continuation requires the original fitted arm and groups')
        if [row['group'] for row in connectors] != list(range(ordinary, len(directed))):
            raise ValueError('Fit-only bridge metadata must cover exactly the appended groups')
    sources = None
    if bridge_assignment == 'linked_sources':
        indices = []
        for row in connectors:
            g = row['group']
            paired_arrivals = [row['first_probe_arrival'], row['second_probe_arrival']]
            for root, t in enumerate(paired_arrivals):
                if not 3 <= t <= int(read[g, root]):
                    raise ValueError('Source must arrive by its menu read')
                torch.testing.assert_close(z[g, 0, t-3:t], z[g, 1, t-3:t], rtol=0., atol=0.)
                torch.testing.assert_close(actions[g, 0, t-3:t], actions[g, 1, t-3:t], rtol=0., atol=0.)
                if torch.equal(z[g, 0, t], z[g, 1, t]):
                    raise ValueError('A paired source must contain two distinct actual arrivals')
            indices.append(paired_arrivals)
        sources = torch.tensor(indices, dtype=torch.long, device=device)
    s2_obs = float(np.quantile(source_strength, .10) / 3)
    # Slots 0 and 1 anchor calibration and initialization across coverage arms.
    nominal = truth[:, :, :, :2].contiguous()
    nominal_controls = controls[:, :, :2].contiguous()
    original_nominal = nominal[:ordinary][directed[:ordinary]]
    s2_bundle = float((original_nominal[:, :, 1] - original_nominal[:, :, 0]).square().sum(-1).mean())
    if not min(s2_bundle, s2_obs) > 0:
        raise ValueError('Both fitted scales require actual distinguishable consequences')
    layers = host.predictor.transformer.layers
    if predecessor:
        np.testing.assert_allclose([s2_obs, s2_bundle], [predecessor['s2_obs'], predecessor['s2_bundle']],
                                   rtol=1e-6, atol=0.)
        state = torch.load(continuation / 'paired_unlabeled.pt', map_location=device, weights_only=True)
        b, v = nn.Parameter(state['b']), nn.Parameter(state['v'])
        if arm == 'offsets_final_two':
            layers[-2:].load_state_dict(state['final_two_blocks'], strict=True)
        norm = predecessor['initial_vjp']
    else:
        condition = host.predictor.transformer.cond_proj(nominal_controls[directed][:1])
        b = nn.Parameter(z.new_zeros(len(layers), condition.size(-1)))
        v = nn.Parameter(torch.zeros_like(b))
        initial_group = int(torch.where(directed)[0][0])
        forecast = menu_forecast(host, windows[initial_group:initial_group+1],
                                 nominal_controls[initial_group:initial_group+1], two_offsets(b, v))
        true_difference = nominal[initial_group, :, 1] - nominal[initial_group, :, 0]
        gradient, = torch.autograd.grad((forecast[0, :, :, 1] * true_difference).sum(), v)
        norm = gradient.norm()
        if not bool(torch.isfinite(norm) & (norm > 0)):
            raise ValueError('Zero or invalid native derivative for actual H12 branch contrast')
        with torch.no_grad():
            v.copy_(gradient * (1e-3 * condition.square().mean().sqrt() / norm))
    groups = [{'params': [b, v], 'lr': .008}]
    if arm == 'offsets_final_two':
        layers[-2:].requires_grad_(True)
        groups.append({'params': layers[-2:].parameters(), 'lr': .0001})
    opt = torch.optim.AdamW(groups)
    generator = torch.Generator().manual_seed(2026113700)
    partitions = tuple(torch.where(directed[:ordinary] == active)[0] for active in (False, True))
    bridge_generator = torch.Generator().manual_seed(2026113800)
    if sources is not None:
        source_before = bridge_source_audit(host, z, actions, two_offsets(b, v), s2_obs, connectors)
    history = []
    for step in range(1, updates+1):
        selected = torch.cat([part[torch.randint(len(part), (1,), generator=generator)] for part in partitions])
        if connectors is not None:
            bridge = ordinary + torch.randint(len(directed) - ordinary, (1,), generator=bridge_generator)
            selected = torch.cat((selected, bridge.to(selected.device)))
        native_offsets = two_offsets(b, v)
        prediction = menu_forecast(host, windows[selected], controls[selected], native_offsets)
        last_read = int(read[selected].max()) + 1 if connectors else z.size(-2)
        support, evidence, issued = factual_support(host, z[selected, :, :last_read],
                                               actions[selected, :, :last_read-1], native_offsets, s2_obs)
        support = support.transpose(1, 2)[torch.arange(len(selected), device=device)[:, None], read[selected]]
        if sources is None:
            energies = paired_energy(prediction, truth[selected], support, directed[selected], s2_bundle)
        else:
            t = sources[bridge.item() - ordinary]
            predicted_h1 = issued[-1, :, t-3].transpose(0, 1)
            actual_h1 = z[selected[-1]][:, t].transpose(0, 1)
            bridge_cost = (predicted_h1 - actual_h1.unsqueeze(-2)).square().sum(-1) / (2 * s2_obs)
            energies = torch.cat((
                paired_energy(prediction[:-1], truth[selected[:-1]], support[:-1], directed[selected[:-1]], s2_bundle),
                paired_energy(prediction[-1:], truth[selected[-1:]], support[-1:], directed[selected[-1:]],
                              s2_bundle, source_cost=bridge_cost.unsqueeze(0)),
            ))
        if bridge_assignment == 'split':
            # The control sees exactly the same arrivals and menus, but may pick
            # a different whole-menu permutation on each side of the bridge.
            bridge_energy = paired_energy(prediction[-1].unsqueeze(1), truth[selected[-1]].unsqueeze(1),
                                          support[-1].unsqueeze(1), directed[selected[-1]].expand(read.size(1)),
                                          s2_bundle).mean()
            energies = torch.cat((energies[:-1], bridge_energy[None]))
        cost = energies.mean()
        if not bool(torch.isfinite(cost)):
            raise FloatingPointError(f'Nonfinite joint loss at update {step}')
        opt.zero_grad(set_to_none=True)
        cost.backward()
        opt.step()
        if step == 1 or step % 64 == 0 or step == updates:
            arrival = torch.arange(3, last_read, device=device)[None, None]
            factual_mask = arrival <= read[selected].amax(dim=-1)[:, None, None]
            peak = evidence.detach().abs()
            peak = peak.masked_fill(~factual_mask, 0).max()
            row = {'update': step, 'loss': float(cost.detach()), 'grad_b': float(b.grad.norm()),
                   'grad_v': float(v.grad.norm()), 'max_abs_factual_evidence': float(peak)}
            history.append(row)
            print(row, flush=True)
    state = {'b': b.detach().cpu(), 'v': v.detach().cpu(), 'arm': arm}
    if arm == 'offsets_final_two':
        state['final_two_blocks'] = {key: val.cpu() for key, val in layers[-2:].state_dict().items()}
    if connectors is not None:
        write_json(directory / 'fit_alignment.json',
                   fit_alignment(host, z, actions, two_offsets(b, v), s2_obs, connectors))
    if sources is not None:
        write_json(directory / 'fit_source_audit.json',
                   {'before_training': source_before, 'after_training': bridge_source_audit(
                       host, z, actions, two_offsets(b, v), s2_obs, connectors)})
    torch.save(state, directory / 'paired_unlabeled.pt')
    write_json(directory / 'paired_unlabeled_fit.json', {
        'arm': arm, 'fit_seed': 2026113700, 's2_obs': s2_obs, 's2_bundle': s2_bundle,
        'initial_vjp': float(norm), 'fitting_groups': len(directed), 'updates': updates,
        'training_candidates_per_root': controls.size(2), 'calibration_candidates_per_root': nominal_controls.size(2),
        'bridge_groups': len(directed) - ordinary, 'bridge_assignment': bridge_assignment,
        'continuation': str(continuation.resolve()) if continuation else None,
        'checkpoints': history, 'wall_seconds': time.perf_counter()-start,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=ROOT / 'runs/stage_c_clean_20261117')
    parser.add_argument('--arm', choices=('offsets', 'offsets_final_two'), required=True)
    parser.add_argument('--updates', type=int, default=1280)
    parser.add_argument('--continuation', type=Path)
    parser.add_argument('--bridge-assignment', choices=('linked', 'split', 'linked_sources'))
    args = parser.parse_args()
    if bool(args.continuation) != (args.bridge_assignment in ('linked', 'split')):
        parser.error('Only linked/split use --continuation; linked_sources restarts from the official host')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    fit(args.directory, args.arm, args.updates, torch.device('cuda'), args.continuation, args.bridge_assignment)


if __name__ == '__main__':
    main()
