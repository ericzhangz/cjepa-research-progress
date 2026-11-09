"""Reference kernels for the P1A diagnostic experiment, not a new JEPA method.

Only the local integration adapter must supply the exact checkpoint, image
transform and saved action normalizer. No simulator or model is bundled here.
"""
from __future__ import annotations
from collections.abc import Callable
import numpy as np
import torch
from torch import Tensor


def build_action_bank(seed: int, horizon: int = 23, block_size: int = 5) -> tuple[np.ndarray, list[str]]:
    """32 fixed open-loop candidates; shape (32, H, block_size, 2), RAW actions.

    9 constant controls: stop + 8 unit directions.
    16 turns: each unit direction, then +/- pi/2 after exactly two model blocks.
    7 wandering controls: theta_0~U(-pi,pi); theta_j=theta_(j-1)+N(0,(pi/4)^2).
    A control is repeated for all five environment steps in its block.
    No state, goal, reward, expert future or world-model prediction is used.
    """
    if horizon < 4 or block_size < 1:
        raise ValueError('horizon must be >=4 and block_size >=1')
    rng = np.random.default_rng(seed)
    angles = np.arange(8, dtype=np.float64) * np.pi / 4
    seq = [np.zeros((horizon, 2), dtype=np.float64)]
    names = ['stop']
    for i, theta in enumerate(angles):
        v = np.array([np.cos(theta), np.sin(theta)])
        seq.append(np.repeat(v[None], horizon, axis=0)); names.append(f'constant_{i}')
    for i, theta in enumerate(angles):
        for sign in (-1, 1):
            th = np.full(horizon, theta); th[2:] += sign*np.pi/2
            seq.append(np.stack((np.cos(th), np.sin(th)), axis=-1))
            names.append(f'turn_{i}_{sign:+d}')
    for i in range(7):
        th = np.empty(horizon); th[0] = rng.uniform(-np.pi, np.pi)
        th[1:] = th[0] + np.cumsum(rng.normal(0, np.pi/4, horizon-1))
        seq.append(np.stack((np.cos(th), np.sin(th)), axis=-1)); names.append(f'wander_{i}')
    blocks = np.repeat(np.stack(seq)[:, :, None, :], block_size, axis=2).astype(np.float32)
    if not np.isfinite(blocks).all() or np.abs(blocks).max() > 1.000001:
        raise AssertionError('invalid actions')
    return blocks, names


@torch.no_grad()
def latent_rollout(
    predict: Callable[[Tensor, Tensor], Tensor],
    action_embed: Callable[[Tensor], Tensor],
    z_history: Tensor,
    past_action_blocks: Tensor,
    future_action_blocks: Tensor,
    history_size: int,
) -> Tensor:
    """Explicit sliding-context native rollout, returns future latents only.

    z_history: [B,T,D]; past_action_blocks: [B,T-1,A]; future: [B,H,A].
    All action blocks are ALREADY normalized by the checkpoint's normalizer,
    in its exact original coordinate ordering. This function does not normalize.
    predict accepts matching [B,L,D] / embedded [B,L,Ae] and returns [B,L,D].
    Original modules must already be eval() and frozen. This does not perform
    training, state smoothing, parameter adaptation or a new memory update.
    """
    if z_history.ndim != 3 or past_action_blocks.ndim != 3 or future_action_blocks.ndim != 3:
        raise ValueError('all inputs must be 3D tensors')
    B, T, D = z_history.shape
    if history_size < 1 or T < 1 or future_action_blocks.shape[1] < 1:
        raise ValueError('empty context/future or invalid history size')
    if past_action_blocks.shape[:2] != (B,T-1) or future_action_blocks.shape[0] != B:
        raise ValueError('history/action alignment mismatch')
    if past_action_blocks.shape[-1] != future_action_blocks.shape[-1]:
        raise ValueError('action dimension mismatch')
    z = z_history.clone(); actions = past_action_blocks.clone(); outputs = []
    for j in range(future_action_blocks.shape[1]):
        actions = torch.cat([actions, future_action_blocks[:,j:j+1]], dim=1)
        length = min(history_size, z.shape[1])
        pred = predict(z[:,-length:], action_embed(actions[:,-length:]))
        if pred.ndim != 3 or pred.shape != (B,length,D):
            raise ValueError(f'predict output shape {pred.shape} != {(B,length,D)}')
        nxt = pred[:,-1:]
        if not torch.isfinite(nxt).all():
            raise FloatingPointError('non-finite native prediction')
        outputs.append(nxt); z = torch.cat([z,nxt], dim=1)
    return torch.cat(outputs, dim=1)


def selection_metrics(cost: np.ndarray, distances: np.ndarray, hit_any: np.ndarray,
                      *, score_rtol: float = 1e-6, score_atol: float = 1e-8) -> dict:
    """One root / horizon / scorer; smaller cost and distance are better.

    Selection uses the expected outcome of uniform tie breaking among score
    minimizers. Labels are not used to resolve ties. Non-finite prediction costs
    trigger an explicit uniform-candidate fallback, rather than silent exclusion.
    Distances must be finite; an invalid physical label is a data error.
    """
    c=np.asarray(cost,dtype=np.float64); d=np.asarray(distances,dtype=np.float64)
    hit=np.asarray(hit_any,dtype=bool)
    if c.ndim!=1 or c.size<2 or d.shape!=c.shape or hit.shape!=c.shape:
        raise ValueError('expected matching candidate vectors')
    if not np.isfinite(d).all():
        raise ValueError('non-finite physical labels; audit geometry instead of replacing them')
    invalid=not np.isfinite(c).all()
    selected=np.ones(c.size,dtype=bool) if invalid else np.isclose(c,c.min(),rtol=score_rtol,atol=score_atol)
    return {
        'selected_indices': np.flatnonzero(selected).tolist(),
        'invalid_cost_uniform_fallback': invalid,
        'selection_regret': float(d[selected].mean()-d.min()),
        'uniform_regret': float(d.mean()-d.min()),
        'selected_hit_probability': float(hit[selected].mean()),
        'uniform_hit_probability': float(hit.mean()),
        'candidate_set_has_hit': bool(hit.any()),
        'candidate_distance_span': float(d.max()-d.min()),
    }


def pair_concordance(cost: np.ndarray, distance: np.ndarray, min_gap: float = 1e-6) -> dict:
    """Association across different ACTIONS at one root and one endpoint time."""
    c=np.asarray(cost,dtype=np.float64); d=np.asarray(distance,dtype=np.float64)
    if c.ndim!=1 or c.shape!=d.shape or min_gap<0 or not np.isfinite(d).all():
        raise ValueError('invalid concordance inputs')
    if not np.isfinite(c).all():
        return {'value':None,'pairs':0,'invalid_cost':True}
    i,j=np.triu_indices(len(c),k=1); dd=d[i]-d[j]; dc=c[i]-c[j]
    use=np.abs(dd)>min_gap
    tie=np.isclose(c[i],c[j],rtol=1e-6,atol=1e-8)
    credit=np.where(tie,0.5,(dc*dd>0).astype(float))
    return {'value':float(credit[use].mean()) if use.any() else None,
            'pairs':int(use.sum()), 'invalid_cost':False}


def root_paired_bootstrap(differences: np.ndarray, seed: int = 20261106,
                          draws: int = 10000) -> dict:
    """Input has ONE scalar per independent root, not per candidate/variant/frame."""
    x=np.asarray(differences,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or draws<1:
        raise ValueError('need >=2 finite root-level differences')
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),size=(draws,len(x)))].mean(axis=1)
    return {'estimate':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'roots':int(len(x)),'draws':int(draws),'seed':int(seed)}
