"""Executed fixed single-pass fit and trajectory/episode objectives.

Historical FourRoom acquisition and driver omitted; retained calculations unchanged.
"""

import torch
from sklearn.covariance import LedoitWolf

from .prbd import (_log_mass, assimilate, assimilate_continuous, continuous_quadrature, issue, quadrature)


def windows(sequence):
    return sequence.unfold(0, 3, 1).transpose(1, 2).contiguous()



@torch.no_grad()
def calibrate(host, episodes):
    residuals = []
    for _, z, a, _ in episodes:
        nominal = host.predict(windows(z[:-1]), windows(a))[:, -1]
        residuals.append(z[3:] - nominal)
    residual = torch.cat(residuals).double()
    estimate = LedoitWolf().fit(residual.numpy())
    noise = torch.linalg.cholesky(torch.from_numpy(estimate.covariance_))
    return noise, {
        "residual_count": len(residual), "dimension": residual.shape[-1],
        "shrinkage": float(estimate.shrinkage_),
        "calibration_mean_residual_l2": residual.mean(0).norm().item(),
        "calibration_mean_squared_residual_l2": residual.square().sum(-1).mean().item(),
        "centered_noise_trace": float(estimate.covariance_.trace()),
    }



def trajectory_log_predictive(paths, weights, observations, noise, nodes, masses,
                              *, geometry="original"):
    """One normalized joint density for a fixed prototype over an issued path.

    All H geometries are signed from the entire K-path family and the same
    root weights before any target is scored.  The per-horizon densities are
    normalized in observation space.  Their product is mixed over k exactly
    once; future arrivals never change the issued paths or public weights.
    The first arrival is returned for the separate, causal factual filter.
    """
    if paths.ndim != 3 or observations.shape != paths.shape[1:]:
        raise ValueError("Expected paths [K,H,D] and actual targets [H,D]")
    if paths.size(1) < 1:
        raise ValueError("A trajectory needs at least one target")
    identity = torch.eye(paths.size(0), dtype=torch.float64, device=paths.device)
    signed = [issue(responses, weights, identity, noise)
              for responses in paths.unbind(dim=1)]
    arrivals = [assimilate(issued, target, nodes, masses, geometry=geometry)
                for issued, target in zip(signed, observations, strict=True)]
    branch_joint = torch.stack([arrival.log_likelihood for arrival in arrivals]).sum(0)
    joint = torch.logsumexp(_log_mass(signed[0].prior) + branch_joint, dim=0)
    return joint, arrivals[0]



def trajectory_paths(model, z, a, horizon):
    """Issue each actual root's own K free paths, including terminal roots.

    Roots sharing a length are batched only for the native forward.  At root
    t, controls are a[t-2:t+H], and no z after t enters a predicted window.
    Terminal roots use min(horizon, remaining); no padding or root selection.
    """
    roots = len(a) - 2
    if len(z) != len(a) + 1 or roots < 1:
        raise ValueError("Every factual root needs three states and a successor")
    result = [None] * roots
    lengths = [min(horizon, len(a) - t) for t in range(2, len(a))]
    for length in sorted(set(lengths), reverse=True):
        selected = [root for root, value in enumerate(lengths) if value == length]
        window = torch.stack([z[root:root + 3] for root in selected])
        controls = torch.stack([a[root:root + 2 + length] for root in selected])
        paths = model.rollout(window, controls)
        for root, path in zip(selected, paths.unbind(), strict=True):
            result[root] = path
    return result



def trajectory_episode_loss(model, z, a, noise, transition, nodes, masses,
                            *, horizon=4, geometry="original"):
    """Overlapping conditional suffix criterion, normalized per target step.

    This is a composite training criterion, not a claim that overlapping
    suffixes are independent observations.  The public factual filter writes
    only the first actual arrival at each root and keeps its gradient history.
    Fixed-k suffixes currently require identity transition (constant condition).
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not torch.equal(transition, torch.eye(len(transition), dtype=transition.dtype,
                                             device=transition.device)):
        raise ValueError("Fixed-prototype trajectory training requires identity transition")
    paths = trajectory_paths(model, z, a, horizon)
    posterior = transition.new_full((len(transition),), 1 / len(transition))
    losses = []
    for root, path in enumerate(paths):
        actual = z[root + 3:root + 3 + path.size(1)]
        log_joint, first_arrival = trajectory_log_predictive(
            path, posterior, actual, noise, nodes, masses, geometry=geometry)
        losses.append(-log_joint)
        posterior = first_arrival.posterior
    targets = sum(path.size(1) for path in paths)
    loss = torch.stack(losses).sum() / targets
    return loss, {"transitions": len(paths), "scored_target_steps": targets,
                  "root_horizon_counts": {str(length): sum(path.size(1) == length for path in paths)
                                           for length in range(1, horizon + 1)}}



def partition_paths(model, z, a, horizon):
    """Issue a disjoint cover of z[3:T+1] using actual block-root windows.

    Each free block is signed using its actual three-state root only. Later
    actual states are targets; they never replace an imagined state inside a
    block. The next block refreshes from its own already-observed root.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if z.ndim != 2 or a.ndim != 2 or len(z) != len(a) + 1 or len(a) < 3:
        raise ValueError("Expected one usable actual episode z[T+1,D], a[T,A], T>=3")
    if not torch.isfinite(z).all() or not torch.isfinite(a).all():
        raise ValueError("Actual episode encodings and controls must be finite")
    roots = list(range(2, len(a), horizon))
    lengths = [min(horizon, len(a) - root) for root in roots]
    result = [None] * len(roots)
    for length in sorted(set(lengths), reverse=True):
        selected = [i for i, value in enumerate(lengths) if value == length]
        window = torch.stack([z[roots[i] - 2:roots[i] + 1] for i in selected])
        controls = torch.stack([a[roots[i] - 2:roots[i] + length] for i in selected])
        predicted = model.rollout(window, controls)
        for index, path in zip(selected, predicted.unbind(), strict=True):
            if path.ndim != 3 or path.shape[1:] != (length, z.size(1)):
                raise ValueError("Native free paths must have shape [K,H,D]")
            result[index] = (roots[index], path)
    return result



def partition_episode_log_predictive(model, z, a, noise, transition, nodes, masses,
                                     *, horizon):
    """One proper episode density: one k for every block and every target.

    Uniform-reference response geometry is independent of the filtering prior.
    Within each block all means are free predictions from its observed root.
    Products are first formed separately for each k over the entire episode;
    exactly one final logsumexp mixes these episode densities with uniform w0.
    A partition contains each usable target once, including the terminal tail.
    """
    count = len(transition)
    identity = torch.eye(count, dtype=transition.dtype, device=transition.device)
    if not torch.equal(transition, identity):
        raise ValueError("Episode-static prototype likelihood requires identity transition")
    uniform = transition.new_full((count,), 1 / count)
    factors, coverage, block_lengths, roots = [], [], [], []
    for root, path in partition_paths(model, z, a, horizon):
        if path.size(0) != count:
            raise ValueError("Native path and transition prototype counts disagree")
        roots.append(root)
        block_lengths.append(path.size(1))
        for step, response in enumerate(path.unbind(1), start=1):
            target = root + step
            signed = issue(response, uniform, transition, noise, reference="uniform")
            arrival = assimilate_continuous(signed, z[target], nodes, masses)
            factors.append(arrival.log_likelihood)
            coverage.append(target)
    if coverage != list(range(3, len(z))):
        raise RuntimeError("A partition must score each usable actual target exactly once")
    branch_joint = torch.stack(factors).sum(0)
    joint = torch.logsumexp(_log_mass(uniform) + branch_joint, 0)
    return joint, {"roots": roots, "block_lengths": block_lengths,
                   "target_indices": coverage, "scored_target_steps": len(coverage),
                   "prototype_episode_log_likelihood": branch_joint.detach().tolist()}



def consistency_episode_loss(model, z, a, noise, transition, nodes=None, masses=None,
                             *, dual=True):
    """Equal composite of two normalized episode log densities, per target.

    H1 and contiguous H<=4 each define a separate proper partition density.
    Their mean log score is an explicit composite criterion, not the log of
    their product presented as one normalized trajectory density. No targets
    are reused inside either density and no public belief is written here.
    """
    if (nodes is None) != (masses is None):
        raise ValueError("Supply both continuous integration tensors, or neither")
    if nodes is None:
        nodes, masses = continuous_quadrature()
    partitions, scores = {}, []
    for horizon in ((1, 4) if dual else (1,)):
        log_joint, details = partition_episode_log_predictive(
            model, z, a, noise, transition, nodes, masses, horizon=horizon)
        partitions[f"H{horizon}"] = details
        scores.append(log_joint / details["scored_target_steps"])
    loss = -torch.stack(scores).mean()
    return loss, {"transitions": len(a) - 2, "scored_target_steps": len(a) - 2,
                  "density_factor_evaluations": sum(
                      part["scored_target_steps"] for part in partitions.values()),
                  "public_belief_writes_in_objective": 0, "partitions": partitions}



def fit(model, episodes, noise, transition, nodes, masses, *, objective="causal",
        horizon=1, geometry="original"):
    if objective not in ("causal", "trajectory", "episode_h1", "episode_dual"):
        raise ValueError("Unknown factual-sequence fitting objective")
    if objective == "causal" and horizon != 1:
        raise ValueError("The causal one-step objective requires horizon=1")
    if objective.startswith("episode_") and geometry != "continuous_bounded_uniform":
        raise ValueError("Episode objectives require continuous_bounded_uniform observation law")
    optimizer = torch.optim.Adam([model.offsets], lr=0.001)
    records = []
    for eid, z, a, _ in episodes:
        if objective.startswith("episode_"):
            loss, counts = consistency_episode_loss(model, z, a, noise, transition, nodes, masses,
                                                    dual=objective == "episode_dual")
        elif objective == "trajectory":
            loss, counts = trajectory_episode_loss(model, z, a, noise, transition, nodes, masses,
                                                   horizon=horizon, geometry=geometry)
        else:
            responses = model(windows(z[:-1]), windows(a))
            posterior = transition.new_full((len(transition),), 1 / len(transition))
            losses = []
            for response, actual in zip(responses, z[3:], strict=True):
                signed = issue(response, posterior, transition, noise)
                # Preserve the legacy call and numerical path by default.
                arrival = (assimilate(signed, actual, nodes, masses) if geometry == "original"
                           else assimilate(signed, actual, nodes, masses, geometry=geometry))
                losses.append(-arrival.log_predictive)
                posterior = arrival.posterior
            loss = torch.stack(losses).mean()
            counts = {"transitions": len(losses)}
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = model.offsets.grad
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Invalid single-trajectory loss or gradient in episode {eid}")
        records.append({"episode": eid, **counts, "nll": loss.item(),
                        "gradient_norm_by_prototype": gradient.flatten(1).norm(dim=1).tolist()})
        optimizer.step()
        if not torch.isfinite(model.offsets).all():
            raise RuntimeError(f"Nonfinite native offsets after episode {eid}")
        if objective.startswith("episode_") and (len(records) % 8 == 0 or len(records) == len(episodes)):
            print(f"{objective}: {len(records)}/{len(episodes)} episode updates; "
                  f"last episode={eid}, nll={loss.item():.6f}", flush=True)
    return records
