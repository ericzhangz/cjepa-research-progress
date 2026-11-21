"""Phase 2 fixed forecasts. Inputs to the predictor are public; scoring lives separately."""
import argparse
from pathlib import Path
from time import perf_counter

import torch

from .prbd import continuous_quadrature
from .prbd_paths import output_path
from .prbd_phase2 import OUT, assimilate_step, load_phase2_checkpoint


def tv(x, y):
    return (x-y).abs().sum(-1) * .5


def transport(w, transition):
    prior = w @ transition
    return prior / prior.sum(-1, keepdim=True)


def settings(cal):
    nodes, masses = continuous_quadrature()
    return dict(transition=cal["transition"], deadzone=cal["deadzone"],
                scales=nodes.to(cal["transition"]), masses=masses.to(cal["transition"]))


@torch.no_grad()
def static_forecast(model, cal, batch):
    device = model.offsets.device
    # Forecast function has no access to any true future token after the read root.
    z = batch["z"][..., :7, :].to(device)
    a = batch["a"].to(device)
    n = len(z)
    window = z[..., 4:7, :].reshape(-1, 3, 192)
    commands = a[..., 4:10, :].reshape(-1, 6, 192)
    paths = model.rollout(window, commands).double().reshape(n, 2, 2, 4, 4, 192)
    args = settings(cal)
    w = cal["transition"].new_full((n*2*2, 4), .25)
    states = {key: w for key in ("first", "full", "omit_first", "uniform")}
    records = []
    for t in range(2, 6):
        means = model(z[..., t-2:t+1, :].reshape(-1, 3, 192), a[..., t-2:t+1, :].reshape(-1, 3, 192))
        step = assimilate_step(means, states["full"], z[..., t+1, :].reshape(-1, 192), cal["noise_cholesky"], **args)
        raw = (step.prior.log() + step.relative_log_likelihood).softmax(-1)
        records.append({k: v.reshape(n, 2, 2, *v.shape[1:]).cpu() for k, v in {
            "w":states["full"], "prior":step.prior, "posterior":step.posterior,
            "raw_posterior":raw, "separation":step.separation, "silent":step.silent,
            "relative_logs":step.relative_log_likelihood}.items()})
        for name in states:
            if name == "full":
                states[name] = step.posterior
                continue
            prior = transport(states[name], cal["transition"])
            if (name == "first" and t == 2) or (name == "omit_first" and t > 2):
                states[name] = torch.where(step.silent[:, None], prior,
                                          (prior.log() + step.relative_log_likelihood).softmax(-1))
            else:
                states[name] = prior
    forecasts = {}
    for name, state in states.items():
        weights = transport(state, cal["transition"]).reshape(n, 2, 2, 4)
        forecasts[name] = (weights[..., None, None] * paths).sum(3).cpu()
    forecasts["nominal"] = paths[:, :, :, 0].cpu()
    return {"seed":batch["seed"], "predictions":forecasts, "records":records}


@torch.no_grad()
def switch_forecast(model, cal, batch):
    z, a = (batch[k].flatten(0, 1).to(model.offsets.device) for k in ("z", "a"))
    lengths = batch["lengths"].flatten().to(z.device)
    w = cal["transition"].new_full((len(a), 4), .25)
    result = w.new_full(a.shape, float("nan"))
    uniform = torch.full_like(result, float("nan"))
    states = w.new_full((*a.shape[:2], 4), float("nan"))
    kwargs = settings(cal)
    for t in range(2, a.size(1)):
        valid = lengths > t
        if not valid.any():
            break
        means = model(z[valid, t-2:t+1], a[valid, t-2:t+1])
        prior = transport(w[valid], cal["transition"])
        result[valid, t] = (prior[..., None] * means).sum(1)
        uniform[valid, t] = means.double().mean(1)
        step = assimilate_step(means, w[valid], z[valid, t+1], cal["noise_cholesky"], **kwargs)
        w[valid] = step.posterior
        states[valid, t] = step.posterior
    n = len(batch["seed"])
    return {"seed":batch["seed"], "predictions":result.reshape(n, 2, *a.shape[1:]).cpu(),
            "uniform":uniform.reshape(n, 2, *a.shape[1:]).cpu(),
            "posteriors":states.reshape(n, 2, a.size(1), 4).cpu()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cal = load_phase2_checkpoint(args.checkpoint, args.device)
    for name, fn in (("static", static_forecast), ("switch", switch_forecast)):
        batch = torch.load(OUT / f"{name}.pt", map_location="cpu", weights_only=True)
        start = perf_counter()
        result = fn(model, cal, batch)
        result["elapsed_seconds"] = perf_counter()-start
        torch.save(result, output_path(OUT / f"pred_{name}_prbd.pt"))
        if name == "static":
            initial, initial_cal = load_phase2_checkpoint(args.checkpoint.parent / "initial.pt", args.device)
            if initial_cal["deadzone"] != cal["deadzone"] or not torch.equal(initial_cal["noise_cholesky"], cal["noise_cholesky"]):
                raise ValueError("Calibration differs between initialization and final weights")
            result = fn(initial, initial_cal, batch)
            torch.save(dict(seed=result["seed"], records=result["records"]), output_path(OUT / "initial_static_writes.pt"))


if __name__ == "__main__":
    main()
