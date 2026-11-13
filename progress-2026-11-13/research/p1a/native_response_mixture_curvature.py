"""Measure the local split directions of the trained single-unit PushT pilot."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from native_response_mixture_pilot import (
    DEVELOPMENT, NORMALIZATION, QUERY_STEPS, ROOT, TRAIN, forecasts, future_inputs,
    issue_event, load_panel, subsequent_roots,
)


def center(values):
    return values - values.mean(dim=1, keepdim=True)


def capture(model, windows, embedding):
    inputs = model.predictor(windows, embedding, response_inputs=True)
    _, normalized, condition, _, _ = inputs
    features = torch.cat((normalized, condition, torch.ones_like(normalized[:, :1])), dim=-1)
    return inputs, features.double()


def measure_pairs(model, panel, object_coordinates, batch_size):
    transformer = model.predictor.transformer
    native_mlp = transformer.layers[-1].mlp.net
    norm, projection, head = [deepcopy(m).double() for m in
                              (transformer.norm, transformer.output_proj, model.pred_proj)]

    def tail(inputs):
        h, _, _, gate, preactivation = [x.double() for x in inputs]
        u = preactivation[:, :1]
        direction = gate * native_mlp[4].weight[:, 0].double()

        def output(offset):
            shifted = h + direction * (native_mlp[2](u + offset) - native_mlp[2](u))
            return head(projection(norm(shifted)))

        return torch.autograd.functional.jvp(output, torch.zeros_like(u), torch.ones_like(u))

    y = panel.arrived_pusher.cpu().numpy().astype(bool)
    scene, pair = np.nonzero(panel.same_prefix & (y[:, 0::2] != y[:, 1::2]))
    positive = np.where(y[scene, 2 * pair], 2 * pair, 2 * pair + 1)
    negative = 4 * pair + 1 - positive
    records = {}
    model.requires_grad_(False)
    with torch.no_grad():
        for start in range(0, len(scene), batch_size):
            batch = slice(start, start + batch_size)
            rows, pos, neg = scene[batch], positive[batch], negative[batch]
            plus, minus = panel.visible[rows, pos], panel.visible[rows, neg]
            if not torch.equal(plus[:, :3], minus[:, :3]) or not torch.equal(plus[:, 4:7], minus[:, 4:7]):
                raise ValueError("Expected identical issuance and query windows")
            commands = panel.executed[rows]
            embedding = model.action_encoder(commands)
            source, x = capture(model, plus[:, :3], embedding[:, :3])
            mean, slope = tail(source)
            w, bias = [p.double() for p in model.event_coordinates]
            xy = (mean @ w + bias) * (96 / 512)
            relative = xy[:, None] - panel.patch[rows].double()
            nearest, indices = relative.norm(dim=-1).topk(2, largest=False)
            radial = relative[torch.arange(len(rows), device=xy.device), indices[:, 0]]
            margin = nearest[:, 1] - nearest[:, 0]
            smooth = (margin > 1e-8) & (nearest[:, 0] > 1e-8)
            alpha = -(radial * (slope @ w) * (96 / 512)).sum(-1) / nearest[:, 0]
            event_probability = (15 * 96 / 512 - nearest[:, 0]).sigmoid()
            op = model.response_mixture
            rho0 = op.readout(x[:, :-1].float())[:, 1].double().sigmoid()
            retention = (1 - rho0)[:, None].expand(-1, 2)
            windows = torch.stack((plus, minus), dim=1).flatten(0, 1)
            for t in (3, 4, 5):
                _, features = capture(model, windows[:, t-2:t+1],
                                      embedding[:, t-2:t+1].repeat_interleave(2, dim=0))
                rho = op.readout(features[:, :-1].float())[:, 1].double().sigmoid()
                retention = retention * (1 - rho.reshape(-1, 2))
            kappa = 0.5 * alpha * (retention[:, 0] * (1 - event_probability)
                                   + retention[:, 1] * event_probability)
            n, k = panel.candidates[rows].shape[:2]
            past = commands[:, None, 4:6].expand(-1, k, -1, -1)
            control = torch.cat((past, panel.candidates[rows, :, :1]), dim=2).flatten(0, 1)
            queried, query_x = capture(model, plus[:, 4:7].repeat_interleave(k, dim=0),
                                       model.action_encoder(control))
            _, query_slope = tail(queried)
            query_slope = query_slope.reshape(n, k, -1)
            step = QUERY_STEPS.index(30)

            def difference(values):
                return center((values[rows, pos, step, :, 0] - values[rows, neg, step, :, 0]).double())

            values = {
                "scene_id": panel.scene_id[rows], "positive_world": pos, "negative_world": neg,
                "x": x, "query_x": query_x.reshape(n, k, -1), "kappa": kappa,
                "smooth": smooth, "alpha": alpha, "event_probability": event_probability,
                "retention": retention, "nearest_margin": margin, "latent_slope": query_slope,
                "latent_target": difference(panel.future_full),
                "physical_slope": query_slope @ object_coordinates[0].double(),
                "physical_target": difference(panel.object_xy),
            }
            for key, value in values.items():
                records.setdefault(key, []).append(torch.as_tensor(value).cpu())
    return {key: torch.cat(parts) for key, parts in records.items()}


def curvature_inputs(records, kind):
    valid = records["smooth"]
    x, query_x, kappa = [records[key][valid] for key in ("x", "query_x", "kappa")]
    slope, target = [records[f"{kind}_{key}"][valid] for key in ("slope", "target")]
    inner = (slope * target).mean(dim=-1)
    v = (query_x * inner[..., None]).mean(dim=1)
    return x, kappa[:, None] * v


def summarize(records, directions):
    valid = records["smooth"]
    summary = {
        "pairs": len(valid), "smooth_pairs": int(valid.sum()),
        "source_derivative_rms": records["alpha"][valid].square().mean().sqrt().item(),
    }
    for kind in ("latent", "physical"):
        x, weighted_v = curvature_inputs(records, kind)
        slope, target = [records[f"{kind}_{key}"][valid] for key in ("slope", "target")]
        k = slope.size(1)
        per_candidate = torch.eye(k, dtype=slope.dtype)[None, :, None, :] * slope[..., None]
        basis = center(per_candidate).flatten(1, 2)
        target = target.flatten(1)
        projection = (basis @ (torch.linalg.pinv(basis, rtol=1e-10) @ target[..., None])).squeeze(-1)
        summary[kind] = {
            "relaxed_tangent_target_energy_fraction": (projection.square().sum() / target.square().sum()).item(),
            "target_rms_per_coordinate": target.square().mean().sqrt().item(),
            "native_offset_derivative_rms_per_coordinate": slope.square().mean().sqrt().item(),
        }
        for label, direction in directions.items():
            per_pair = -(x @ direction) * (weighted_v @ direction)
            values = torch.full(valid.shape, torch.nan, dtype=per_pair.dtype)
            values[valid] = per_pair
            records[f"{kind}_curvature_{label}"] = values
            ids, inverse = torch.unique(records["scene_id"][valid], return_inverse=True)
            totals = torch.zeros(len(ids), dtype=per_pair.dtype).index_add_(0, inverse, per_pair)
            summary[kind][label] = {
                "mean_curvature": per_pair.mean().item(), "negative_pairs": int((per_pair < 0).sum()),
                "scenes": len(ids), "negative_scenes": int((totals < 0).sum()),
            }
    return summary


def full_objective_curvature(model, panel, direction, batch_size):
    """Central difference of directional gradients at fixed 0.01 native offset RMS."""
    weight = model.response_mixture.readout.weight
    bias = model.response_mixture.readout.bias
    device_direction = direction.to(weight)
    grad_signs = []
    weight.requires_grad_(True)
    bias.requires_grad_(True)
    saved_weight, saved_bias = weight[0].detach().clone(), bias[0].detach().clone()
    try:
        for sign in (-1, 1):
            with torch.no_grad():
                weight[0].copy_(sign * 0.01 * device_direction[:-1])
                bias[0].copy_(sign * 0.01 * device_direction[-1])
            combined = []
            for start in range(0, len(panel.scene_id), batch_size):
                rows = slice(start, start + batch_size)
                visible, commands, issued, _, _, written, _, evidence = issue_event(model, panel, rows)
                roots = subsequent_roots(model, visible, commands, written)
                roots[10] = issued.log_weights
                future = []
                for step in (10, 15, 30):
                    prediction = forecasts(model, *future_inputs(panel, rows, visible, commands, step), roots[step])
                    truth = panel.future_full[rows, :, QUERY_STEPS.index(step)].flatten(0, 1)
                    future.append(F.mse_loss(prediction, truth))
                slopes = []
                for index, loss in enumerate((-evidence.mean(), torch.stack(future).mean())):
                    dw, db = torch.autograd.grad(loss, (weight, bias), retain_graph=index == 0)
                    slopes.append((dw[0] @ device_direction[:-1] + db[0] * device_direction[-1]).item())
                combined.append(slopes)
            grad_signs.append(np.average(combined, weights=[min(batch_size, len(panel.scene_id) - i)
                                                          for i in range(0, len(panel.scene_id), batch_size)], axis=0))
    finally:
        with torch.no_grad():
            weight[0].copy_(saved_weight)
            bias[0].copy_(saved_bias)
        weight.requires_grad_(False)
        bias.requires_grad_(False)
    event, future = (grad_signs[1] - grad_signs[0]) / 0.02
    return {"event": float(event), "future_mse": float(future), "joint": float(event + future)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "research_sources_2026-10-22/lewm_assets"))
    from prepare_assets import load_model

    with NORMALIZATION.open(encoding="utf-8") as stream:
        moments = json.load(stream)
    normalizer = [torch.tensor(moments[key], device=args.device) for key in ("mean", "std")]
    panels = {name: load_panel(spec, args.device, normalizer) for name, spec in
              (("train", TRAIN), ("historical_development", DEVELOPMENT))}
    model = load_model(args.device).eval()
    model.enable_response_mixture()
    checkpoint = torch.load(ROOT / "runs/native_response_mixture_20261113/pilot.pt",
                            map_location=args.device, weights_only=True)
    model.predictor.transformer.layers[-1].load_state_dict(checkpoint["last_block"])
    model.pred_proj.load_state_dict(checkpoint["pred_proj"])
    model.response_mixture.load_state_dict(checkpoint["response_mixture"])
    model.event_coordinates = checkpoint["event_coordinates"]
    records = {name: measure_pairs(model, panel, checkpoint["object_coordinates"], args.batch_size)
               for name, panel in panels.items()}
    train = records["train"]
    x, weighted_v = curvature_inputs(train, "latent")
    hessian = -(x.T @ weighted_v + weighted_v.T @ x) / (2 * len(x))
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    seed = torch.zeros_like(eigenvectors[:, 0])
    seed[-1] = 1
    readout = model.response_mixture.readout
    fitted = torch.cat((readout.weight[0], readout.bias[:1])).detach().cpu().double()
    normalization_features = torch.cat((train["x"], train["query_x"].flatten(0, 1)))
    directions = {label: d / (normalization_features @ d).square().mean().sqrt()
                  for label, d in (("bias_seed", seed), ("train_latent_min_eigen", eigenvectors[:, 0]),
                                   ("fitted_row", fitted))}
    summary = {name: summarize(rows, directions) for name, rows in records.items()}
    summary["train_extreme_latent_curvature_unit_parameter_norm"] = eigenvalues[[0, -1]].tolist()
    print(json.dumps(summary, indent=2), flush=True)
    for name, panel in panels.items():
        summary[name]["original_objective_directional_curvature_at_zero"] = {}
        for label in ("bias_seed", "train_latent_min_eigen"):
            result = full_objective_curvature(model, panel, directions[label], args.batch_size)
            summary[name]["original_objective_directional_curvature_at_zero"][label] = result
            print(name, label, result, flush=True)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, rows in records.items():
        np.savez_compressed(args.output / f"{name}_pairs.npz", **{key: value.numpy() for key, value in rows.items()})
    np.savez_compressed(args.output / "directions.npz", **{label: d.numpy() for label, d in directions.items()})


if __name__ == "__main__":
    main()
