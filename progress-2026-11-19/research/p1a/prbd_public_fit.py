"""Shared fixed single-pass fitting used by the TwoRoom speed probe.

Extracted from the original public-prefix fitter; its FourRoom data and
execution driver are excluded. The three numerical definitions are unchanged.
"""

import torch
from sklearn.covariance import LedoitWolf

from .prbd import assimilate, issue


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


def fit(model, episodes, noise, transition, nodes, masses):
    optimizer = torch.optim.Adam([model.offsets], lr=0.001)
    records = []
    for eid, z, a, _ in episodes:
        responses = model(windows(z[:-1]), windows(a))
        posterior = transition.new_full((len(transition),), 1 / len(transition))
        losses = []
        for response, actual in zip(responses, z[3:], strict=True):
            signed = issue(response, posterior, transition, noise)
            arrival = assimilate(signed, actual, nodes, masses)
            losses.append(-arrival.log_predictive)
            posterior = arrival.posterior
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = model.offsets.grad
        if not torch.isfinite(loss) or not torch.isfinite(gradient).all():
            raise RuntimeError(f"Invalid single-trajectory loss or gradient in episode {eid}")
        records.append({"episode": eid, "transitions": len(losses), "nll": loss.item(),
                        "gradient_norm_by_prototype": gradient.flatten(1).norm(dim=1).tolist()})
        optimizer.step()
    return records
