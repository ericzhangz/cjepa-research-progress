"""Measure the fixed-relation H1 physical ceiling on recorded PushT mirror pairs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import native_response_mixture_regulation as data
from .native_response_relations_intermediates import HOLD_CANDIDATE, QUERY_STEP
from .native_response_strength_bounds import response_bounds


RUN = data.RUNS / "native_response_relations_20261114/action_effects_20261115"
CHECKPOINT = RUN / "relational/model.pt"
OUTPUT = RUN / "strength_ceiling"
HORIZON_INDEX = 0
CAPACITY_FIELDS = (
    "max_dual_norm",
    "max_activation_delta_norm",
    "max_gated_residual_norm",
)


def _mirror_query(panel):
    rows = slice(None)
    layout = panel.visible.shape[:3]
    history, past, candidates = data._query_inputs(panel, rows, (QUERY_STEP,))
    for value in (history, past, candidates):
        negative, positive = value.unflatten(0, layout).unbind(2)
        if not torch.equal(negative, positive):
            raise ValueError("Each mirror pair must expose identical public windows and commands")
    mean, std = data._action_moments(candidates)
    hold = ((candidates.new_zeros(5, 2) - mean) / std).flatten()
    hold_commands = candidates[:, HOLD_CANDIDATE, HORIZON_INDEX]
    if not torch.equal(hold_commands, hold.expand_as(hold_commands)):
        raise ValueError("Expected the actual zero-command reference at candidate 3")

    _, endpoints, valid = data._targets(panel, rows, (QUERY_STEP,))
    truth = endpoints[..., 0, :, HORIZON_INDEX, :].double()
    if not valid[..., 0, :, HORIZON_INDEX].all() or not torch.isfinite(truth).all():
        raise ValueError("Expected a full fan of executed finite H1 endpoints for every mirror history")
    return history[::2], past[::2], candidates[::2], truth


def _spectral_norm(matrix):
    return torch.linalg.eigvalsh(matrix @ matrix.mT)[..., -1].clamp_min(0).sqrt()


def _native_gains(model, h, gate, coordinates):
    transformer = model.predictor.transformer
    w2 = transformer.layers[-1].mlp.net[4].weight
    relation_basis = torch.linalg.qr(model.response_mixture.relation_directions.T).Q
    weight, bias = coordinates
    with torch.enable_grad():
        residual = h.detach().requires_grad_(True)
        native = model.pred_proj(transformer.output_proj(transformer.norm(residual))) @ weight + bias
        jacobian = torch.stack([
            torch.autograd.grad(native[:, axis].sum(), residual, retain_graph=axis + 1 < native.shape[-1])[0]
            for axis in range(native.shape[-1])
        ], dim=1)
    activation_jacobian = (jacobian * gate[:, None, :]) @ w2
    unrestricted = _spectral_norm(activation_jacobian)
    restricted = _spectral_norm(activation_jacobian @ relation_basis)
    if (unrestricted <= 0).any() or (restricted > unrestricted * (1 + 1e-12)).any():
        raise RuntimeError("Expected positive unrestricted gain and a contractive relation-space restriction")
    return native.detach(), unrestricted, restricted, restricted / unrestricted


def _measure(model, query, coordinates):
    history, past, candidates, truth = query
    pair_count, fan, horizon = candidates.shape[:3]
    past = model.action_encoder(past).double()
    actions = model.action_encoder(candidates.flatten(1, 2)).unflatten(1, (fan, horizon)).double()
    root = history.double()[:, None].expand(-1, fan, -1, -1).flatten(0, 1)
    controls = torch.cat((past[:, None].expand(-1, fan, -1, -1), actions[:, :, :1]), dim=2).flatten(0, 1)
    inputs, reference_inputs = model._response_inputs(root, controls)
    if reference_inputs is not None:
        raise RuntimeError("The fixed relational path must use a single native pass")
    h, _, _, gate, preactivation = inputs
    native_xy, gain_all, gain_relations, gain_ratio = _native_gains(model, h, gate, coordinates)
    w, bias = coordinates
    torch.testing.assert_close(native_xy, model.predict(root, controls)[:, -1] @ w + bias,
                               rtol=1e-10, atol=1e-9)

    fields = ("y_lower", "y_upper", "y_native", *CAPACITY_FIELDS)
    measurements = [response_bounds(model, x, g, u, coordinates) for x, g, u in zip(h, gate, preactivation, strict=True)]
    values = {
        field: torch.stack([torch.as_tensor(item[field], device=h.device, dtype=h.dtype) for item in measurements])
        for field in fields
    }
    torch.testing.assert_close(values["y_native"], native_xy, rtol=1e-10, atol=1e-9)
    lower, upper = values["y_lower"], values["y_upper"]
    modes, _, _, _ = model.response_mixture.native_modes(
        inputs, reference_inputs, model.predictor.transformer, model.pred_proj,
    )
    mode_xy = modes @ w + bias
    if ((mode_xy < lower[:, None] - 1e-9) | (mode_xy > upper[:, None] + 1e-9)).any():
        raise RuntimeError("A checkpoint mode fell outside the output bound")
    values.update(activation_gain_all=gain_all, activation_gain_relations=gain_relations,
                  activation_gain_ratio=gain_ratio)
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise RuntimeError("The fixed-relation ceiling must be finite on the recorded queries")
    if len(h) != pair_count * fan:
        raise RuntimeError("Expected one native H1 response per mirror pair and candidate")
    return {key: value.unflatten(0, (*truth.shape[:2], fan)) for key, value in values.items()}


def _contrast_metrics(truth, half_width):
    # Each mirror contrast is enclosed in a box symmetric about zero.
    distance = (truth.abs() - half_width).clamp_min(0).norm(dim=-1)
    excluded = distance > 0
    has_effect = truth.ne(0).any(-1)
    count = int(excluded.sum())
    nonzero = int(has_effect.sum())
    summary = {
        "true_contrast_rms": truth.square().sum(-1).mean().sqrt().item(),
        "reachable_contrast_upper_rms": half_width.square().sum(-1).mean().sqrt().item(),
        "contrast_rmse_lower_bound": distance.square().mean().sqrt().item(),
        "excluded_target_count": count,
        "target_count": excluded.numel(),
        "excluded_target_fraction": count / excluded.numel(),
        "nonzero_target_count": nonzero,
        "excluded_nonzero_target_fraction": count / nonzero if nonzero else None,
    }
    return summary, distance, excluded


def _summarize(measured, true_action, true_hold):
    width = measured["y_upper"] - measured["y_lower"]
    if (width < 0).any():
        raise RuntimeError("Encountered an inverted physical output interval")
    hold_width = width[..., HOLD_CANDIDATE, :]
    action_width = width[..., :HOLD_CANDIDATE, :] + hold_width[..., None, :]
    action, action_distance, action_excluded = _contrast_metrics(true_action, action_width)
    hold, hold_distance, hold_excluded = _contrast_metrics(true_hold, hold_width)
    action["excluded_mirror_pair_count"] = int(action_excluded.any(-1).sum())
    arrays = dict(action_contrast_box_half_width=action_width, hold_contrast_box_half_width=hold_width,
                  action_contrast_error_lower_bound=action_distance, hold_contrast_error_lower_bound=hold_distance,
                  action_contrast_excluded=action_excluded, hold_contrast_excluded=hold_excluded)
    summary = {
        "mirror_action": action,
        "mirror_hold": hold,
        "native_capacity": {
            key: {"rms": measured[key].square().mean().sqrt().item(), "max": measured[key].max().item()}
            for key in CAPACITY_FIELDS
        },
        "local_activation_gain": {
            key: {"min": measured[key].min().item(), "median": measured[key].quantile(0.5).item(),
                  "max": measured[key].max().item()}
            for key in ("activation_gain_all", "activation_gain_relations", "activation_gain_ratio")
        },
    }
    return summary, arrays


@torch.no_grad()
def run(device):
    torch.set_float32_matmul_precision("highest")
    panels = {split: data.load_data(split, data.ARCHIVE_DIR, device) for split in ("train", "development")}
    if np.intersect1d(panels["train"].scene_ids, panels["development"].scene_ids).size:
        raise ValueError("The fit and historical development scenes must be disjoint")
    queries = {split: _mirror_query(panel) for split, panel in panels.items()}
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=True)
    if checkpoint["mechanism"] != "relational":
        raise ValueError("Expected the fitted relational checkpoint")
    coordinates = tuple(value.double() for value in checkpoint["object_coordinates"])
    model = data._native_model(device)
    op = model.enable_response_mixture(parameterization="relational", num_modes=4)
    model.requires_grad_(False).double()
    model.action_encoder.float()

    summary = {
        "query_step": QUERY_STEP,
        "horizon_environment_steps": 5,
        "hold_candidate": HOLD_CANDIDATE,
        "optimization_steps": 0,
        "device": str(device),
        "predictor_dtype": "float64",
        "action_encoder_dtype": "float32",
        "bound_relaxation": "Independent strengths and coordinate intervals per endpoint, arbitrary support",
        "axes": {"truth_xy": ["scene", "world", "program", "candidate", "xy"],
                 "candidate_outputs": ["scene", "world", "candidate", "xy"],
                 "candidate_scalar_fields": ["scene", "world", "candidate"],
                 "action_contrast": ["scene", "world", "nonhold_candidate", "xy"],
                 "hold_contrast": ["scene", "world", "xy"]},
    }
    records = {}
    for split, (_, _, candidates, truth) in queries.items():
        negative, positive = truth.unbind(2)
        displacement = positive - negative
        true_hold = displacement[..., HOLD_CANDIDATE, :]
        true_action = displacement[..., :HOLD_CANDIDATE, :] - true_hold[..., None, :]
        scene_ids = panels[split].scene_ids
        summary[split] = {"scene_ids": scene_ids.tolist(), "mirror_pairs": int(np.prod(truth.shape[:2]))}
        records[split] = dict(scene_ids=scene_ids, truth_xy=truth, true_action_contrast=true_action,
                              true_hold_contrast=true_hold, candidate_normalized_commands=(
                                  candidates[..., HORIZON_INDEX, :].unflatten(0, truth.shape[:2])))

    for arm in ("official_initialization", "fitted"):
        if arm == "fitted":
            model.predictor.transformer.layers[-1].load_state_dict(checkpoint["last_block"])
            model.pred_proj.load_state_dict(checkpoint["pred_proj"])
            op.load_state_dict(checkpoint["response_mixture"])
        for split, query in queries.items():
            measured = _measure(model, query, coordinates)
            record = records[split]
            summary[split][arm], distances = _summarize(
                measured, record["true_action_contrast"], record["true_hold_contrast"],
            )
            record.update({f"{arm}_{name}": value for name, value in (measured | distances).items()})
            print(f"{arm} {split}: {summary[split]['mirror_pairs']} H1 mirror pairs", flush=True)
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=torch.device("cuda"), type=torch.device)
    parser.add_argument("--output", default=OUTPUT, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    summary, records = run(args.device)
    encoded = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    args.output.mkdir(parents=True, exist_ok=False)
    for split, record in records.items():
        np.savez_compressed(args.output / f"{split}.npz", **{
            key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in record.items()
        })
    (args.output / "summary.json").write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
