"""Stage C-0 geometry and H1 evidence calibration with a frozen LeWM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .native_topology_condition import action_embeddings, load_host
from .stage_a_fourroom import ROOT, write_json


def second_moment(deltas):
    rows = deltas.reshape(-1, deltas.shape[-1]).astype(np.float64)
    return rows.T @ rows / len(rows)


@torch.inference_mode()
def secants(host, archive, split):
    def read(name):
        return archive[f'{split}_{name}'].astype(np.float64)

    lock, lock_alt = read('lock_z'), read('lock_alt_z')
    actions = read('lock_a')
    windows = np.stack((lock, lock_alt), axis=1)
    x = torch.as_tensor(windows.reshape(-1, 4, lock.shape[-1]), dtype=torch.float32,
                        device=host.action_mean.device)
    controls = action_embeddings(host, np.repeat(actions, 4, axis=0).astype(np.float32))
    issued = torch.cat([host.predict(z[:, :-1], a)[:, -1] for z, a in
                        zip(x.split(64), controls.split(64), strict=True)])
    residuals = windows[..., -1, :] - issued.cpu().numpy().reshape(windows.shape[:-2] + (-1,))
    action = read('action_z')
    bundle = read('bundle_z')
    return {
        'window': lock_alt[..., :3, :] - lock[..., :3, :],
        'innovation': residuals[:, 1] - residuals[:, 0],
        'lock_h1': lock[:, 1, -1] - lock[:, 0, -1],
        'action_h1': action[:, 1, -1] - action[:, 0, -1],
        'lock_long': bundle[:, 1, :, 1:] - bundle[:, 0, :, 1:],
        'action_long': bundle[:, :, 1, 1:] - bundle[:, :, 0, 1:],
    }


def projector(nuisance, action, generalized):
    if not generalized:
        values, vectors = np.linalg.eigh(nuisance - action)
        floor = np.finfo(values.dtype).eps * len(values) * abs(values).max()
        retained = vectors[:, values <= floor]
    else:
        n = len(nuisance)
        noise = nuisance / np.trace(nuisance) + (1e-6 / n) * np.eye(n)
        values, vectors = np.linalg.eigh(noise)
        inverse_sqrt = (vectors * values**-.5) @ vectors.T
        values, vectors = np.linalg.eigh(inverse_sqrt @ (action / np.trace(action)) @ inverse_sqrt)
        retained = np.linalg.qr(inverse_sqrt @ vectors[:, values > 1], mode='reduced')[0]
    return retained @ retained.T, retained.shape[1]


def retention(deltas, projection):
    original = np.linalg.norm(deltas, axis=-1).ravel()
    projected = np.linalg.norm(deltas @ projection, axis=-1).ravel()
    ratios = projected[original > 0] / original[original > 0]
    if not len(ratios):
        raise ValueError('Zero energy in a required calibration comparison')
    return {
        'rms': float(np.linalg.norm(projected) / np.linalg.norm(original)),
        'p50': float(np.quantile(ratios, .50)),
        'p90': float(np.quantile(ratios, .90)),
    }


def distribution(values):
    return {name: float(value) for name, value in zip(
        ('min', 'p10', 'median', 'p90', 'max'), np.quantile(values, (0, .10, .50, .90, 1)), strict=True
    )} | {'mean': float(np.mean(values))}


def evidence_terms(rows, p):
    d = rows['lock_h1'] @ p
    m = np.sum(d**2, axis=-1) / 2
    b = np.abs(np.einsum('nd,nwd->nw', d, rows['innovation'] @ p)).max(axis=1)
    return m, b


def assess(p, rank, splits, action, h1):
    m0, b0 = evidence_terms(splits['fit'], p)
    s_obs = float(np.quantile(m0 - b0, .10) / 3)
    s_bundle = float(np.trace(p @ action @ p))
    s_h1 = float(np.trace(p @ h1 @ p))
    split_reports = {}
    for name, rows in splits.items():
        m, b = evidence_terms(rows, p)
        split_reports[name] = {
            'retention': {kind: retention(delta, p) for kind, delta in rows.items()},
            'lock_half_squared': distribution(m),
            'background_abs_inner_product': distribution(b),
            'old_bundle_trace_ideal': distribution(m / s_bundle) if s_bundle > 0 else None,
            'h1_trace_ideal': distribution(m / s_h1) if s_h1 > 0 else None,
            's_obs_ideal': distribution(m / s_obs) if s_obs > 0 else None,
            's_obs_conservative': distribution((m - b) / s_obs) if s_obs > 0 else None,
            's_obs_background': distribution(b / s_obs) if s_obs > 0 else None,
            'min_s_obs_for_background_p90_le_0_1': float(np.quantile(b, .90) / .10),
            'max_s_obs_for_conservative_p10_ge_2': float(np.quantile(m - b, .10) / 2),
            'evidence_rows': [
                {'row': i, 'ideal_numerator': float(signal), 'background_numerator': float(noise)}
                for i, (signal, noise) in enumerate(zip(m, b, strict=True))
            ],
        }
    check = split_reports['check']
    q = check['retention']
    geometry = q['window']['rms'] <= .25 and q['innovation']['rms'] <= .25 and q['lock_h1']['rms'] >= .60
    evidence = (s_obs > 0 and all(report['s_obs_background']['p90'] <= .10
                                  for report in split_reports.values())
                and check['s_obs_conservative']['p10'] >= 2.)
    return {
        'rank': rank,
        'orthonormal_projector_max_abs_error': float(np.max(np.abs(p @ p - p))),
        'projection_passed': geometry, 'evidence_passed': evidence,
        's2_bundle': s_bundle, 's2_obs': s_obs, 's2_h1_trace': s_h1,
        'splits': split_reports,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/stage_c_20261117')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    host = load_host(torch.device(args.device))
    with np.load(args.out / 'calibration.npz') as archive:
        center = archive['fit_bg_z'][..., :3, :].mean(axis=(0, 1, 2), dtype=np.float64)
        splits = {name: secants(host, archive, name) for name in ('fit', 'check')}
    fit = splits['fit']
    nuisance = .5 * (second_moment(fit['window']) + second_moment(fit['innovation']))
    h1 = .5 * (second_moment(fit['lock_h1']) + second_moment(fit['action_h1']))
    long = .5 * (second_moment(fit['lock_long']) + second_moment(fit['action_long']))
    action = .5 * (h1 + long)
    traces = {name: float(np.trace(covariance)) for name, covariance in
              [('nuisance', nuisance), ('action', action), ('h1', h1), ('h2_to_h12', long)]}
    if min(traces.values()) <= 0:
        raise ValueError('A required paired covariance has no energy')
    reports = {}
    for name, generalized in [('spectral_subtraction', False), ('normalized_generalized_ratio', True)]:
        p, rank = projector(nuisance, action, generalized)
        current = assess(p, rank, splits, action, h1)
        reports[name] = current
        np.savez_compressed(args.out / f'{name}.npz', P=p, center=center)
        if current['projection_passed']:
            break
    write_json(args.out / 'geometry_checks.json', {
        'active_diagnostic_projection': name, 'traces': traces, 'projections': reports,
    })
    np.savez_compressed(args.out / 'geometry.npz', P=p, center=center,
                        s2_obs=current['s2_obs'], s2_bundle=current['s2_bundle'],
                        projection_passed=current['projection_passed'], evidence_passed=current['evidence_passed'])
    for method, report in reports.items():
        print(method, 'rank', report['rank'], 'gates', report['projection_passed'], report['evidence_passed'],
              'retention', {k: round(v['rms'], 4) for k, v in report['splits']['check']['retention'].items()},
              's2_obs', report['s2_obs'], flush=True)


if __name__ == '__main__':
    main()
