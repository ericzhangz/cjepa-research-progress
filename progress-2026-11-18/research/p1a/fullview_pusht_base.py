"""Compare frozen and H1–H4 latent-fitted native LeWM in full-view PushT."""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .native_full_depth_reversible import encode_rgb
from .native_full_depth_task_geometry import RADIUS, chart_pose, pose_chart
from .native_response_mixture_regulation import _action_moments, _native_model
from .native_response_relations_fit import native_rollout
from .native_reversible_pusht import ACTION_BLOCK, ACTION_SCALE, CANDIDATES, IMAGE_SIZE, PushTEnv, Vec2d, candidate_actions


OUTPUT = Path(__file__).resolve().parents[2] / "runs/fullview_pusht_base_20261115"
FIT_IDS = np.arange(57000, 57192)
HELDOUT_IDS = np.arange(58000, 58064)
SEED = 261115
HISTORY = 3
HORIZON = 4
PULSE = (0, 5)
HOLD = CANDIDATES.index("hold")
FIT_EPOCHS = 40
FIT_BATCH = 32
FIT_LR = 1e-4
READOUT_EPOCHS = 80
READOUT_BATCH = 128
READOUT_WIDTH = 128
READOUT_RIDGE = 1e-3
BOOTSTRAPS = 2000


def protocol():
    return {
        "fit_scene_ids": FIT_IDS.tolist(), "heldout_scene_ids": HELDOUT_IDS.tolist(), "seed": SEED,
        "environment": "Unmodified gym-pusht 0.1.8 physics, fixed native goal, full 96×96 RGB",
        "reset": "Object origin [184,328]×[172,294], angle uniform [-pi,pi]; agent left of stem in the object frame",
        "prefix": "Two real five-control blocks; hold, then advance toward stem at relative magnitude 0.035",
        "candidates": CANDIDATES, "pulse_indices": PULSE, "hold_index": HOLD,
        "actions": "Existing PushT menu, rotated into each scene's object reset frame; scene multiplier uniform [0.85,1.15]; shared across branches",
        "action_transform": "Native relative control: current agent position + 100 × command, clipped to workspace",
        "history": HISTORY, "horizons_blocks": list(range(1, HORIZON + 1)), "block_controls": ACTION_BLOCK,
        "world_model_fit": {"epochs": FIT_EPOCHS, "batch": FIT_BATCH, "lr": FIT_LR, "optimizer": "AdamW",
                            "weight_decay": 1e-4, "loss": "Equal H1–H4 mean squared native latent error",
                            "trainable_modules": ["action_encoder", "predictor", "pred_proj"],
                            "eval_mode_during_fit": True, "checkpoint_choice": "last epoch"},
        "readout_fit": {"ridge": READOUT_RIDGE, "width": READOUT_WIDTH, "epochs": READOUT_EPOCHS,
                        "batch": READOUT_BATCH, "mlp_lr": 1e-3, "mlp_optimizer": "AdamW",
                        "supervision": "Frozen real full-RGB latents and physical rigid chart, fit scenes only"},
        "diagnostic_thresholds": {"h4_f_pulse_xy_signed_gain": 0.70, "h4_c_object_xy_rmse_world_px": 5},
        "bootstrap": {"resamples": BOOTSTRAPS, "unit": "heldout scene"},
    }


def scene_input(seed):
    rng = np.random.default_rng(seed)
    center = rng.uniform([184, 172], [328, 294])
    angle = rng.uniform(-np.pi, np.pi)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, s], [-s, c]])
    agent = center + np.array([-rng.uniform(44, 54), rng.uniform(43, 57)]) @ rotation
    # Native reset writes the origin before the angle, then rotates around the body's CoG.
    # Supply the pre-rotation origin so the physical start has the sampled visible pose.
    center_of_gravity = np.array([0.0, 45.0])
    initial = np.r_[agent, center - center_of_gravity + center_of_gravity @ rotation, angle]
    prefix = np.zeros((HISTORY - 1, ACTION_BLOCK, 2), dtype=np.float32)
    prefix[1] = np.array([0.035, 0.]) @ rotation
    plans = (candidate_actions() @ rotation * rng.uniform(0.85, 1.15)).astype(np.float32)
    np.testing.assert_array_equal(plans[PULSE[0], 1:], plans[PULSE[1], 1:])
    if np.array_equal(plans[PULSE[0], 0], plans[PULSE[1], 0]):
        raise AssertionError("The first controls must differ for the pulse pair")
    return initial, center, prefix, plans


def execute_block(env, commands):
    for command in commands:
        target = np.clip(np.asarray(env.agent.position) + ACTION_SCALE * command, 0, 512)
        _, _, _, truncated, _ = env.step(Vec2d(*target))
        if truncated:
            raise RuntimeError("Native PushT truncated a fixed-length action branch")
    return env.get_obs(), env._render(visualize=False)


def collect_scene(env, seed):
    initial, center, prefix, plans = scene_input(seed)
    roots = []
    states, images = [], []
    for plan in plans:
        start, _ = env.reset(seed=0, options={"reset_to_state": initial})
        np.testing.assert_allclose(start[2:4], center, rtol=0, atol=1e-10)
        root = [(start, env._render(visualize=False))]
        root.extend(execute_block(env, block) for block in prefix)
        if roots:
            for actual, expected in zip(root, roots, strict=True):
                np.testing.assert_array_equal(actual[0], expected[0])
                np.testing.assert_array_equal(actual[1], expected[1])
        else:
            roots = root
        trajectory = [execute_block(env, block) for block in plan]
        states.append(np.stack([item[0] for item in trajectory]))
        images.append(np.stack([item[1] for item in trajectory]))
    return {
        "initial_state": initial, "prefix_actions": prefix, "actions": plans,
        "root_state": np.stack([item[0] for item in roots]), "future_state": np.stack(states),
        "root_rgb": np.stack([item[1] for item in roots]), "future_rgb": np.stack(images),
    }


@torch.no_grad()
def collect(model, ids, name):
    env = PushTEnv(obs_type="state", render_mode="rgb_array", observation_width=IMAGE_SIZE,
                  observation_height=IMAGE_SIZE)
    rows = []
    try:
        for offset in range(0, len(ids), 8):
            scenes = [collect_scene(env, int(seed)) for seed in ids[offset:offset + 8]]
            chunk = {key: np.stack([row[key] for row in scenes]) for key in scenes[0]}
            chunk["root_z"] = encode_rgb(model, chunk.pop("root_rgb")).cpu().numpy()
            chunk["future_z"] = encode_rgb(model, chunk.pop("future_rgb")).cpu().numpy()
            rows.append(chunk)
            if len(rows) % 4 == 0 or offset + 8 >= len(ids):
                print(f"{name}: collected and encoded {min(offset + 8, len(ids))}/{len(ids)} scenes", flush=True)
    finally:
        env.close()
    return {"scene_ids": ids, **{key: np.concatenate([row[key] for row in rows]) for key in rows[0]}}


def batch_inputs(panel, device):
    root = torch.as_tensor(panel["root_z"], device=device)
    target = torch.as_tensor(panel["future_z"], device=device)
    candidates = target.size(1)
    prefix = np.broadcast_to(panel["prefix_actions"][:, None], (*target.shape[:2], HISTORY - 1, ACTION_BLOCK, 2))
    actions = torch.as_tensor(np.concatenate((prefix, panel["actions"]), axis=2), device=device)
    mean, std = _action_moments(actions)
    normalized = ((actions - mean) / std).flatten(-2)
    return root[:, None].expand(-1, candidates, -1, -1).flatten(0, 1), target.flatten(0, 1), normalized.flatten(0, 1)


def free_rollout(model, root, embedded_actions):
    return native_rollout(model, root, embedded_actions[:, :HISTORY - 1], embedded_actions[:, None, HISTORY - 1:])[:, 0]


def fit_model(model, panel, device, output):
    root, target, controls = batch_inputs(panel, device)
    parts = {name: getattr(model, name) for name in ("action_encoder", "predictor", "pred_proj")}
    parameters = [p for module in parts.values() for p in module.requires_grad_(True).parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=FIT_LR, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(SEED)
    with (output / "world_model_training.csv").open("x", newline="", encoding="utf-8") as stream:
        trace = csv.writer(stream)
        trace.writerow(["epoch", "step", "gradient_norm", "latent_mse_h1", "latent_mse_h2", "latent_mse_h3", "latent_mse_h4"])
        step = 0
        for epoch in range(1, FIT_EPOCHS + 1):
            batches = torch.randperm(len(root), generator=rng, device=device).split(FIT_BATCH)
            records = []
            for rows in batches:
                optimizer.zero_grad(set_to_none=True)
                prediction = free_rollout(model, root[rows], model.action_encoder(controls[rows]))
                mse = (prediction - target[rows].detach()).square().mean(dim=(0, 2))
                mse.mean().backward()
                norm = torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in parameters]))
                data = torch.cat((norm[None], mse.detach())).cpu().numpy()
                if not np.isfinite(data).all():
                    raise FloatingPointError(f"Nonfinite world-model fit at step {step}")
                optimizer.step()
                step += 1
                trace.writerow([epoch, step, *data])
                records.append(data)
            stream.flush()
            if epoch == 1 or epoch % 5 == 0:
                print(f"World model epoch {epoch}/{FIT_EPOCHS}: grad and H1–H4 {np.mean(records, axis=0)}", flush=True)
    torch.save({name: part.state_dict() for name, part in parts.items()}, output / "world_model.pt")
    model.requires_grad_(False)


def fit_readouts(panel, device, output):
    root, future = (torch.as_tensor(panel[key], device=device).flatten(0, -2) for key in ("root_z", "future_z"))
    z = torch.cat((root, future))
    root, future = (torch.as_tensor(panel[key], dtype=z.dtype, device=device).flatten(0, -2)
                    for key in ("root_state", "future_state"))
    y = pose_chart(torch.cat((root, future))[:, 2:])
    z_mean, z_std = z.mean(0), z.std(0).clamp_min(1e-6)
    y_mean, y_std = y.mean(0), y.std(0).clamp_min(1e-6)
    x, labels = (z - z_mean) / z_std, (y - y_mean) / y_std
    gram = x.T.double() @ x.double() / len(x)
    gram.diagonal().add_(READOUT_RIDGE)
    weight = torch.linalg.solve(gram, x.T.double() @ labels.double() / len(x)).float()
    torch.manual_seed(SEED)
    mlp = nn.Sequential(nn.Linear(x.size(-1), READOUT_WIDTH), nn.GELU(), nn.Linear(READOUT_WIDTH, y.size(-1))).to(device)
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = torch.Generator(device=device).manual_seed(SEED + 1)
    for _ in range(READOUT_EPOCHS):
        for rows in torch.randperm(len(x), generator=rng, device=device).split(READOUT_BATCH):
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(mlp(x[rows]), labels[rows]).backward()
            optimizer.step()
    torch.save(dict(z_mean=z_mean, z_std=z_std, y_mean=y_mean, y_std=y_std,
                    ridge_weight=weight, mlp=mlp.state_dict()), output / "readouts.pt")

    def decode(z, method):
        normalized = (z - z_mean) / z_std
        raw = ((normalized @ weight) if method == "linear" else mlp(normalized)) * y_std + y_mean
        return torch.cat((raw[..., :2], RADIUS * F.normalize(raw[..., 2:], dim=-1)), dim=-1)

    return decode


@torch.no_grad()
def evaluate(model, panel, device):
    root, future, controls = batch_inputs(panel, device)
    free, real_window = [], []
    for z, y, a in zip(root.split(64), future.split(64), controls.split(64), strict=True):
        embedded = model.action_encoder(a)
        forecast = free_rollout(model, z, embedded)
        actual = torch.cat((z, y), dim=1)
        teacher = torch.stack([model.predict(actual[:, h:h + HISTORY], embedded[:, h:h + HISTORY])[:, -1]
                               for h in range(HORIZON)], dim=1)
        torch.testing.assert_close(forecast[:, 0], teacher[:, 0], rtol=0, atol=0)
        if not torch.isfinite(forecast).all() or not torch.isfinite(teacher).all():
            raise FloatingPointError("Nonfinite heldout native forecasts")
        free.append(forecast)
        real_window.append(teacher)
    shape = panel["future_z"].shape
    return {"F": torch.cat(free).reshape(shape), "L": torch.cat(real_window).reshape(shape), "C": future.reshape(shape)}


def residual_terms(values):
    return {"C_minus_Y": values["C"] - values["Y"], "L_minus_C": values["L"] - values["C"],
            "F_minus_L": values["F"] - values["L"]}


def contrast_metrics(values, draws):
    y = values["Y"]
    target_energy = np.square(y).sum(-1)
    target_by_scene = target_energy.reshape(len(y), -1, HORIZON).sum(axis=1)
    denominator = target_by_scene.sum(axis=0)
    if np.any(denominator <= 1e-12):
        raise ValueError("No physical action-effect energy at one or more horizons")
    den_samples = target_by_scene[draws].sum(axis=1)
    results = {"Y_rms": np.sqrt(target_energy.reshape(-1, HORIZON).mean(axis=0)).tolist()}
    for name, value in (values | residual_terms(values)).items():
        products = (value * y).sum(axis=-1).reshape(len(y), -1, HORIZON).sum(axis=1)
        gain = products.sum(axis=0) / denominator
        gain_draws = products[draws].sum(axis=1) / den_samples
        orthogonal = value - y * gain[None, :, None]
        results[name] = {
            "signed_gain": gain.tolist(), "signed_gain_ci95": np.quantile(gain_draws, [0.025, 0.975], axis=0).T.tolist(),
            "orthogonal_rms": np.sqrt(np.square(orthogonal).sum(-1).reshape(-1, HORIZON).mean(axis=0)).tolist(),
        }
    return results


def xy_metrics(values, draws):
    parts = residual_terms(values)
    residual = np.abs(sum(parts.values()) - (values["F"] - values["Y"])).max()
    if residual > 1e-10:
        raise AssertionError(f"Physical error decomposition did not close: {residual}")
    relative = {key: np.delete(value, HOLD, axis=1) - value[:, HOLD:HOLD + 1] for key, value in values.items()}
    pulse = {key: value[:, PULSE[0]] - value[:, PULSE[1]] for key, value in values.items()}
    return {
        "endpoint_rmse": {name: np.sqrt(np.square(error).sum(-1).mean(axis=(0, 1))).tolist()
                          for name, error in (parts | {key: values[key] - values["Y"] for key in ("C", "L", "F")}).items()},
        "relative_to_hold": contrast_metrics(relative, draws), "first_block_pulse": contrast_metrics(pulse, draws),
        "max_decomposition_residual": float(residual),
    }, parts


@torch.no_grad()
def summarize(panel, latents, decode):
    y = panel["future_state"][..., 2:].astype(np.float64)
    random = np.random.default_rng(SEED + 2)
    draws = random.integers(len(y), size=(BOOTSTRAPS, len(y)))
    result, arrays = {}, {}
    for arm, vectors in latents.items():
        arrays.update({f"{arm}_{key}_latent": value.cpu().numpy() for key, value in vectors.items()})
        result[arm] = {"latent_mse": {key: (vectors[key] - vectors["C"]).square().mean(dim=(0, 1, 3)).tolist()
                                      for key in ("F", "L")}}
        for readout in ("linear", "mlp"):
            chart = {key: decode(value, readout) for key, value in vectors.items()}
            chart["Y"] = pose_chart(torch.as_tensor(y, device=vectors["C"].device))
            pose = {key: chart_pose(vector).double().cpu().numpy() for key, vector in chart.items()}
            pose["Y"] = y
            xy = {key: value[..., :2] for key, value in pose.items()}
            xy_result, xy_parts = xy_metrics(xy, draws)
            geometry, _ = xy_metrics({key: value.double().cpu().numpy() for key, value in chart.items()}, draws)
            angle = {key: (pose[key][..., 2] - y[..., 2] + np.pi) % (2 * np.pi) - np.pi for key in ("C", "L", "F")}
            result[arm][readout] = {"object_xy_world_px": xy_result, "rigid_material_world_px": geometry,
                                   "angle_rmse_deg": {key: np.rad2deg(np.sqrt(np.square(error).mean(axis=(0, 1)))).tolist()
                                                      for key, error in angle.items()}}
            arrays.update({f"{arm}_{readout}_{key}_pose": value for key, value in pose.items()})
            arrays.update({f"{arm}_{readout}_{key}_xy": value for key, value in xy_parts.items()})
    return result, arrays


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output / "PROTOCOL.json"
    if lock.exists():
        if json.loads(lock.read_text(encoding="utf-8")) != json.loads(json.dumps(protocol())):
            raise RuntimeError("Run configuration differs from the saved protocol")
    else:
        write_json(lock, protocol())
    start = time.perf_counter()
    device = torch.device(args.device)
    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = _native_model(device)
    torch.testing.assert_close(model.predictor.pos_embedding.size(1), HISTORY)
    fit = collect(model, FIT_IDS, "fit")
    np.savez_compressed(args.output / "fit_data.npz", **fit)
    decode = fit_readouts(fit, device, args.output)
    encoder = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
               if name.startswith(("encoder.", "projector."))}
    baseline = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items() if name not in encoder}
    fit_model(model, fit, device, args.output)
    for name, expected in encoder.items():
        torch.testing.assert_close(model.state_dict()[name].cpu(), expected, rtol=0, atol=0)
    heldout = collect(model, HELDOUT_IDS, "heldout")
    fitted = evaluate(model, heldout, device)
    model.load_state_dict(encoder | baseline, strict=True)
    frozen = evaluate(model, heldout, device)
    torch.testing.assert_close(fitted["C"], frozen["C"], rtol=0, atol=0)
    summary, records = summarize(heldout, {"frozen": frozen, "fitted": fitted}, decode)
    np.savez_compressed(args.output / "heldout_branches.npz", **heldout, **records)
    gradients = np.loadtxt(args.output / "world_model_training.csv", delimiter=",", skiprows=1)[:, 2]
    summary["validation"] = {"fit_holdout_scene_overlap": int(len(np.intersect1d(FIT_IDS, HELDOUT_IDS))),
                             "exact_shared_root_for_all_candidates": True, "exact_pulse_suffix": True,
                             "encoder_and_projector_frozen": True, "h1_max_F_L_latent_residual": 0.0,
                             "fit_steps": len(gradients), "gradient_norm_min_median_max": np.quantile(gradients, [0, .5, 1]).tolist(),
                             "device": str(device), "elapsed_seconds": time.perf_counter() - start}
    write_json(args.output / "summary.json", summary)
    print(f"Completed heldout analysis in {summary['validation']['elapsed_seconds']:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()
