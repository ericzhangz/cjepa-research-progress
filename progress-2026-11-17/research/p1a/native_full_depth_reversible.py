"""Fit and evaluate the same all-block conditioner on full-view physical regime pairs."""

import json
import sys

import numpy as np
import torch

from . import native_response_mixture_regulation as data
from .native_full_depth_conditioning import (
    FullDepthConditioner, advance_log_support, conditioned_rollout, write_observation,
)
from .native_response_mixture_pilot import fit_coordinates


ASSETS = data.RUNS / "native_reversible_pusht_20261115"
OUTPUT = data.RUNS / "native_full_depth_reversible_20261115"
UPDATES = 320
SWITCH = 0.05
SEED = 2615


def encode_rgb(model, rgb):
    sys.path.insert(0, str(data.ROOT / "oic_step1"))
    from run_native_pusht_host import encode
    return encode(model, rgb)


def action_embeddings(model, commands):
    mean, scale = data._action_moments(commands)
    normalized = ((commands - mean) / scale).flatten(-2)
    shape = normalized.shape[:-1]
    return model.action_encoder(normalized.reshape(-1, shape[-1], 10)).reshape(*shape, -1)


@torch.no_grad()
def panel(model, part, device):
    with np.load(ASSETS / f"{part}.npz", allow_pickle=False) as archive:
        rgb = archive["rgb"]
        np.testing.assert_array_equal(rgb[:, 0, :, 0], rgb[:, 1, :, 0])
        names = archive["candidate"].tolist()
        commands = torch.as_tensor(archive["actions"], dtype=torch.float32, device=device)
        print(f"Encoding {part} {rgb.shape[:-3]} on {device}", flush=True)
        return {
            "z": encode_rgb(model, rgb), "a": action_embeddings(model, commands),
            "xy": torch.as_tensor(archive["state"][..., 2:4], dtype=torch.float32, device=device),
            "scene_ids": archive["scene_id"].tolist(), "names": names,
        }


def offsets(adapter, batch, *, frozen=False):
    dev = next(adapter.parameters()).device
    code = torch.tensor([[1.0, -1.0], [1.0, 1.0]], device=dev)
    if frozen:
        code = torch.zeros_like(code)
    return adapter(code)[None].expand(batch, -1, -1, -1)


def block_windows(latents, commands, issued_at, hold_embedding):
    latents = torch.cat((latents[..., :1, :].expand(*latents.shape[:-2], 2, latents.size(-1)), latents), dim=-2)
    filler = hold_embedding.expand(*commands.shape[:-2], 2, commands.size(-1))
    past = torch.cat((filler, commands), dim=-2)
    return latents[..., issued_at:issued_at + 3, :], past[..., issued_at:issued_at + 2, :]


def forecast(model, adapter, values, rows, start, horizon, hold_embedding):
    z, a = values["z"][rows], values["a"][rows]
    history, past = block_windows(z, a, start, hold_embedding)
    past = past[:, None].expand(-1, 2, -1, -1, -1)
    plans = a[:, None, :, start:start + horizon].expand(-1, 2, -1, -1, -1)
    history, past, plans = [x.flatten(0, 2) for x in (history, past, plans)]
    prediction = conditioned_rollout(model, history, past, plans[:, None], offsets(adapter, len(history)))[:, 0]
    return prediction.unflatten(0, z.shape[:3])


def select_true(prediction):
    return torch.stack((prediction[:, 0, :, 0], prediction[:, 1, :, 1]), dim=1)


def factual_error(predicted, truth):
    return (predicted - truth).square().sum(-1).mean(dim=(0, 1, 2)).sqrt().tolist()


@torch.no_grad()
def summarize(model, adapter, values, hold, w, b, scale):
    query = forecast(model, adapter, values, slice(None), 0, 4, hold)
    xy_modes = query @ w + b
    truth = values["xy"][..., 1:, :]
    actual = select_true(xy_modes)
    mean = xy_modes.mean(dim=3)
    target_difference = truth[:, 1] - truth[:, 0]
    prediction_difference = actual[:, 1] - actual[:, 0]
    aligned = (target_difference * prediction_difference).sum(-1).mean(dim=(0, 1))
    denominator = target_difference.square().sum(-1).mean(dim=(0, 1))
    root = values["z"][:, :, :, 0]
    prior = root.new_full((*root.shape[:-1], 2), -np.log(2))
    observed = values["z"][..., 1, :] @ w + b
    log_support, ll = write_observation(prior, xy_modes[..., 0, :], observed, scale)
    label = torch.arange(2, device=root.device)[None, :, None].expand_as(prior[..., 0])
    proper = -log_support.gather(-1, label[..., None]).mean()
    frozen = forecast_native(model, values, hold) @ w + b
    return {
        "scenes": values["scene_ids"], "candidates": values["names"],
        "true_mode_endpoint_rmse": factual_error(actual, truth),
        "uniform_support_endpoint_rmse": factual_error(mean, truth),
        "frozen_endpoint_rmse": factual_error(frozen, truth),
        "true_mode_directed_fraction": (aligned / denominator.clamp_min(1e-10)).tolist(),
        "true_mode_gap_rmse": (prediction_difference - target_difference).square().sum(-1).mean(dim=(0, 1)).sqrt().tolist(),
        "true_mode_gap_rms": denominator.sqrt().tolist(),
        "source_labels_nll_nats": proper.item(),
        "source_label_accuracy_by_candidate": (log_support.argmax(-1) == label).float().mean(dim=(0, 1)).tolist(),
        "source_abs_log_odds_by_candidate": (ll[..., 1] - ll[..., 0]).abs().mean(dim=(0, 1)).tolist(),
    }


@torch.no_grad()
def forecast_native(model, values, hold):
    from .native_response_relations_fit import native_rollout
    z, a = values["z"], values["a"]
    history, past = block_windows(z, a, 0, hold)
    past = past[:, None].expand(-1, 2, -1, -1, -1)
    plans = a[:, None].expand(-1, 2, -1, -1, -1)
    predicted = native_rollout(model, history.flatten(0, 2), past.flatten(0, 2), plans.flatten(0, 2)[:, None])
    return predicted[:, 0].unflatten(0, z.shape[:3])


@torch.no_grad()
def lifecycle(model, adapter, filename, hold, w, b, scale, device):
    with np.load(filename, allow_pickle=False) as archive:
        frames = encode_rgb(model, archive["rgb"])
        commands = action_embeddings(model, torch.as_tensor(archive["action"], dtype=torch.float32, device=device))
        phases, labels = archive["phase"].tolist(), archive["mode"].tolist()
        records = []
        log_mass = frames.new_full((1, 2), -np.log(2))
        for step, (phase, label) in enumerate(zip(phases, labels, strict=True)):
            history, past = block_windows(frames[None], commands[None], step, hold)
            issued = conditioned_rollout(model, history, past, commands[None, None, step:step + 1],
                                         offsets(adapter, 1))[0, 0, :, 0] @ w + b
            observed = frames[step + 1] @ w + b
            prior = advance_log_support(log_mass, SWITCH)
            log_mass, ll = write_observation(prior, issued[None], observed[None], scale)
            records.append({"step": step + 1, "phase": phase, "offline_label": label,
                            "prior": prior.exp()[0].tolist(), "support": log_mass.exp()[0].tolist(),
                            "log_likelihood_difference": (ll[0, 1] - ll[0, 0]).item(),
                            "issued_xy": issued.tolist(), "arrived_xy": observed.tolist(),
                            "readout_gap": (issued[1] - issued[0]).norm().item()})
    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = data._native_model(device)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    adapter = FullDepthConditioner(model, 2).to(device)
    fitting = panel(model, "fit", device)
    print("Full-view fit encodings ready", flush=True)
    count = len(fitting["scene_ids"])
    cut = count - max(2, count // 5)
    w, b = fit_coordinates(fitting["z"][:cut], fitting["xy"][:cut])
    hold = action_embeddings(model, torch.zeros((1, 5, 2), device=device))[0]
    lat_scale = fitting["z"][:cut].std().clamp_min(0.01)
    xy_scale = (fitting["xy"][:cut, 1] - fitting["xy"][:cut, 0]).square().mean().sqrt().clamp_min(1)
    opt = torch.optim.AdamW(adapter.parameters(), lr=0.002, weight_decay=0.01)
    losses = []
    for step in range(UPDATES):
        rows = rng.choice(cut, 2, replace=False).tolist()
        start = step % 3
        prediction = select_true(forecast(model, adapter, fitting, rows, start, 2, hold))
        latent = fitting["z"][rows, ..., start + 1:start + 3, :]
        xy = fitting["xy"][rows, ..., start + 1:start + 3, :]
        positions = prediction @ w + b
        absolute = ((positions - xy) / xy_scale).square().mean()
        paired = (((positions[:, 1] - positions[:, 0]) - (xy[:, 1] - xy[:, 0])) / xy_scale).square().mean()
        appearance = ((prediction - latent) / lat_scale).square().mean()
        loss = 0.25 * absolute + paired + 0.5 * appearance
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
        opt.step()
        if (step + 1) % 40 == 0:
            losses.append([step + 1, loss.item()])
            print(losses[-1], flush=True)

    with torch.no_grad():
        residuals = []
        for start in range(4):
            estimated = select_true(forecast(model, adapter, fitting, slice(cut, None), start, 1, hold)) @ w + b
            measured = fitting["z"][cut:, ..., start + 1:start + 2, :] @ w + b
            residuals.append(estimated - measured)
        scale = torch.cat(residuals, dim=-2).flatten(0, -2).square().mean(dim=0).sqrt().clamp_min(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "training_summary.json").open("x", encoding="utf-8") as stream:
        json.dump({"seed": SEED, "updates": UPDATES, "training_scenes": fitting["scene_ids"][:cut],
                   "calibration_scenes": fitting["scene_ids"][cut:], "losses": losses,
                   "scale": scale.tolist(), "switch_probability": SWITCH}, stream, indent=2)
    torch.save({"adapter": adapter.state_dict(), "w": w.cpu(), "b": b.cpu(), "scale": scale.cpu()},
               OUTPUT / "conditioner.pt")

    answers = {}
    for part in ("development", "heldout"):
        if (ASSETS / f"{part}.npz").exists():
            answers[part] = summarize(model, adapter, panel(model, part, device), hold, w, b, scale)
    for file in sorted(ASSETS.glob("*lifecycle*.npz")):
        answers[file.stem] = lifecycle(model, adapter, file, hold, w, b, scale, device)
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(answers, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
