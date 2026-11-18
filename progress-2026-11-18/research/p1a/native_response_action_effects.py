"""Measure the §18 physical prediction family and §20 paired write effects at step 125."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if __package__:
    from . import native_response_mixture_regulation as data
    from .native_response_relations_intermediates import HOLD_CANDIDATE, QUERY_STEP, SOURCE_STEPS, _full_sigma, _source_trace
else:
    import native_response_mixture_regulation as data
    from native_response_relations_intermediates import HOLD_CANDIDATE, QUERY_STEP, SOURCE_STEPS, _full_sigma, _source_trace


ARMS = ("none", "105_only", "110_only", "both")
HORIZONS = (5, 10)
OUTPUT = data.RUNS / "native_response_relations_20261114/action_effects_20261115/zero_step"


def load_coordinates():
    """Load only the fixed offline physical readout from the historical checkpoint."""
    return torch.load(data.WARM_START, map_location="cpu", weights_only=True)["object_coordinates"]


def _support_arms(op, prior, posterior, sources):
    from module import IssuedResponse

    first, latest = sources
    issue = IssuedResponse(op.prepare_state(prior, len(prior)), latest["means"])
    _, latest_only, _ = op.condition(issue, latest["log_likelihood"])
    arms = torch.stack((prior, first["log_support_after"], latest_only, posterior), dim=1)
    # Independent additive log-odds identity for both real write schedules.
    log_p = prior.double()
    ll_first, ll_latest = (source["log_likelihood"].double() for source in sources)
    expected = torch.stack((log_p, log_p + ll_first, log_p + ll_latest, log_p + ll_first + ll_latest), dim=1)
    torch.testing.assert_close(arms.double().softmax(-1), expected.softmax(-1), atol=2e-6, rtol=0)
    return arms


def _relation_trace(model, last_action, trace):
    op = model.response_mixture
    transformer = model.predictor.transformer
    last_block = transformer.layers[-1]
    # The native final-block gate depends only on the projected action condition.
    gate = last_block.adaLN_modulation(transformer.cond_proj(last_action)).chunk(6, dim=-1)[-1]
    correction = gate[..., None, :] * F.linear(trace["activation_delta"], last_block.mlp.net[4].weight)
    load, dual = trace["relation_load"], trace["relation_dual"]
    torch.testing.assert_close(dual, load * op.mode_strengths(), atol=2e-5, rtol=2e-5)
    drive = F.gelu(trace["preactivation"]) @ op.relation_directions.T - op.relation_thresholds
    if drive.ndim == load.ndim - 1:
        drive = drive.unsqueeze(-2).expand_as(load)
    return dict(relation_drive=drive, relation_load=load, relation_dual=dual,
                gate_correction_rms=correction.square().mean(-1).sqrt())


@torch.no_grad()
def collect_split(model, panel, sigma, coordinates, *, batch_size=2):
    """Return raw measurements with leading scene, world and program axes."""
    op = model.response_mixture
    if op.parameterization != "relational" or op.num_modes != 4:
        raise ValueError("Expected the four-mode exact relational pathway")
    weights, bias = (value.to(device=panel.visible.device, dtype=torch.float64) for value in coordinates)
    _, truth, valid = data._targets(panel, slice(None), (QUERY_STEP,))
    truth, valid = truth.squeeze(-4).double(), valid.squeeze(-3)
    raw = {}
    for start in range(0, len(panel.scene_ids), batch_size):
        rows = slice(start, start + batch_size)
        prior, posterior, sources, hold = _source_trace(model, panel, rows, sigma)
        log_support = _support_arms(op, prior, posterior, sources)
        support = log_support.double().softmax(-1)
        history, past, commands = data._query_inputs(panel, rows, (QUERY_STEP,))
        if not torch.equal(commands[:, HOLD_CANDIDATE], hold.expand_as(commands[:, HOLD_CANDIDATE])):
            raise ValueError("Expected the independent hold branch at candidate 3")
        if not torch.equal(commands[:, :, 0], commands[:, :, 1]):
            raise ValueError("Expected H2 to repeat its candidate's first physical command block")
        actions = model.action_encoder(commands.flatten(1, 2)).unflatten(1, commands.shape[1:3])
        trace = {}
        modes = model.rollout_response_modes(history, model.action_encoder(past), actions, trace=trace)
        mode_xy = modes.double() @ weights + bias
        mode_effect = mode_xy - mode_xy[:, HOLD_CANDIDATE:HOLD_CANDIDATE + 1]
        prediction = torch.einsum("bak,bchkx->bchax", support, mode_xy)
        effect = prediction - prediction[:, HOLD_CANDIDATE:HOLD_CANDIDATE + 1]
        weighted_effect = torch.einsum("bak,bchkx->bchax", support, mode_effect)
        torch.testing.assert_close(effect, weighted_effect, atol=2e-10, rtol=0)
        support_delta = support[:, -1:, :] - support
        for output, table in ((prediction, mode_xy), (effect, mode_effect)):
            projected = torch.einsum("bak,bchkx->bchax", support_delta, table)
            direct = output[..., -1:, :] - output
            torch.testing.assert_close(direct, projected, atol=2e-10, rtol=0)

        source_traces = [_relation_trace(model, model.action_encoder(source["normalized_commands"])[:, -1],
                                         source) for source in sources]
        fields = {
            "log_support": log_support, "support": support, "mode_xy": mode_xy,
            "effects_by_mode_xy": mode_effect, "prediction_xy": prediction, "effects_xy": weighted_effect,
            "source_log_likelihood": torch.stack([source["log_likelihood"] for source in sources], dim=1),
            "source_normalized_command": torch.stack([source["normalized_commands"][:, -1] for source in sources], dim=1),
            **{f"query_{key}": value for key, value in _relation_trace(model, actions, trace).items()},
            **{f"source_{key}": torch.stack([entry[key] for entry in source_traces], dim=1)
               for key in source_traces[0]},
        }
        for key, value in fields.items():
            array = value.unflatten(0, panel.visible[rows].shape[:3]).cpu().numpy()
            raw.setdefault(key, []).append(array)

    truth_effect = truth - truth[..., HOLD_CANDIDATE:HOLD_CANDIDATE + 1, :, :]
    pair_valid = valid & valid[..., HOLD_CANDIDATE:HOLD_CANDIDATE + 1, :]
    arrays = {key: np.concatenate(parts) for key, parts in raw.items()}
    arrays.update(scene_ids=panel.scene_ids, truth_xy=truth.cpu().numpy(), valid=valid.cpu().numpy(),
                  truth_effects_xy=truth_effect.cpu().numpy(), effect_valid=pair_valid.cpu().numpy(),
                  inactive=(pair_valid & truth_effect.eq(0).all(-1)).cpu().numpy(), arm_names=np.asarray(ARMS),
                  candidate_normalized_commands=panel.candidates.cpu().numpy(),
                  source_steps=np.asarray(SOURCE_STEPS), horizon_steps=np.asarray(HORIZONS))
    if not all(np.isfinite(array).all() for key, array in arrays.items() if key != "arm_names"):
        raise ValueError("Found nonfinite values in the zero-step records")
    return arrays


def _rmse_and_pairs(predictions, truth, valid, scene_ids):
    squared = np.square(predictions - truth[..., None, :]).sum(-1)
    count = valid.reshape(len(valid), -1).sum(-1)
    per_scene = (squared * valid[..., None]).reshape(len(valid), -1, len(ARMS)).sum(1)
    active = count > 0
    if not active.any():
        return {"valid_endpoints": 0, "scene_count": 0}
    scene_rmse = np.sqrt(per_scene[active] / count[active, None])
    differences = scene_rmse[:, -1:] - scene_rmse
    bootstrap_indices = np.random.default_rng(1611).integers(active.sum(), size=(4000, active.sum()))
    bootstrap = differences[bootstrap_indices].mean(1)
    return {
        "valid_endpoints": int(count.sum()), "scene_count": int(active.sum()),
        "rmse": dict(zip(ARMS, np.sqrt(per_scene.sum(0) / count.sum()).tolist(), strict=True)),
        "paired_both_minus": {
            arm: {"mean_scene_rmse_difference": float(differences[:, index].mean()),
                  "scene_bootstrap_ci95": np.percentile(bootstrap[:, index], [2.5, 97.5]).tolist(),
                  "scenes_improved": int((differences[:, index] < 0).sum()),
                  "per_scene_rmse_difference": dict(zip(map(str, scene_ids[active]), differences[:, index].tolist(),
                                                        strict=True))}
            for index, arm in enumerate(ARMS[:-1])
        },
    }


def _norm_rms(values):
    return float(np.sqrt(np.square(values).sum(-1).mean())) if len(values) else None


def _native_trace_summary(arrays, prefix, index):
    axis = 3 if prefix == "source" else 4
    traces = {key: np.take(arrays[f"{prefix}_{key}"], index, axis=axis)[..., 1:, :]
              for key in ("relation_drive", "relation_load", "relation_dual")}
    traces["gate_correction_rms"] = np.take(arrays[f"{prefix}_gate_correction_rms"], index, axis=axis)[..., 1:]
    levels = {f"{key}_rms": float(np.sqrt(np.square(values.astype(np.float64)).mean()))
              for key, values in traces.items()}
    drive, dual = traces["relation_drive"], traces["relation_dual"]
    levels["relation_slots"] = dual.size
    levels["positive_duals"] = int((dual > 0).sum())
    levels["newly_loaded_at_solution"] = int(((drive <= 0) & (dual > 0)).sum())
    return levels


def summarize_split(arrays):
    scene_ids = arrays["scene_ids"]
    source = {}
    for index, step in enumerate(SOURCE_STEPS):
        ll = arrays["source_log_likelihood"][..., index, :]
        before, after = (0, 1) if index == 0 else (1, 3)
        tv = 0.5 * np.abs(arrays["support"][..., before, :] - arrays["support"][..., after, :]).sum(-1)
        source[str(step)] = {"actual_arrivals": ll[..., 0].size, "max_mode_log_likelihood_span": float(np.ptp(ll, -1).max()),
                             "mean_support_tv": float(tv.mean()), "max_support_tv": float(tv.max()),
                             **_native_trace_summary(arrays, "source", index)}
    horizons = {}
    for index, step in enumerate(HORIZONS):
        truth = arrays["truth_xy"][..., index, :]
        prediction = arrays["prediction_xy"][..., index, :, :]
        valid = arrays["valid"][..., index]
        h = HOLD_CANDIDATE
        absolute = _rmse_and_pairs(prediction, truth, valid, scene_ids)
        hold = _rmse_and_pairs(prediction[..., h, :, :], truth[..., h, :], valid[..., h], scene_ids)
        effect = arrays["effects_xy"][..., :h, index, :, :]
        observed = arrays["truth_effects_xy"][..., :h, index, :]
        effect_valid = arrays["effect_valid"][..., :h, index]
        inactive = arrays["inactive"][..., :h, index]
        mode_effects = arrays["effects_by_mode_xy"][..., :h, index, :, :]
        bounds = (mode_effects.min(-2), mode_effects.max(-2))
        within = ((observed >= bounds[0]) & (observed <= bounds[1])).all(-1)
        groups = {}
        for name, mask in (("all", effect_valid), ("inactive", inactive), ("active", effect_valid & ~inactive)):
            groups[name] = {
                **_rmse_and_pairs(effect, observed, mask, scene_ids),
                "true_action_effect_rms": _norm_rms(observed[mask]),
                "mode_action_effect_coordinate_span_rms": _norm_rms(np.ptp(mode_effects, -2)[mask]),
                "endpoints_within_mode_coordinate_envelope": int((within & mask).sum()),
                "both_minus_support_readout_rms": {
                    arm: _norm_rms((effect[..., -1, :] - effect[..., a, :])[mask])
                    for a, arm in enumerate(ARMS[:-1])
                },
            }
        hold_values = prediction[..., h, :, :]
        horizons[f"H{index + 1}"] = {
            "environment_steps": step, "absolute_all_candidates": absolute, "absolute_hold": hold,
            "hold_mode_coordinate_span_rms": _norm_rms(np.ptp(arrays["mode_xy"][..., h, index, :, :], -2)[valid[..., h]]),
            "hold_both_minus_support_readout_rms": {
                arm: _norm_rms((hold_values[..., -1, :] - hold_values[..., a, :])[valid[..., h]])
                for a, arm in enumerate(ARMS[:-1])
            },
            "held_referenced_action_effect": groups,
            **_native_trace_summary(arrays, "query", index),
        }
    return {
        "query_step": QUERY_STEP, "scenes": scene_ids.tolist(), "roots": int(np.prod(arrays["valid"].shape[:3])),
        "source": source, "horizons": horizons,
        "arms": dict(zip(ARMS, ([], [105], [110], [105, 110]), strict=True)),
        "true_inactivity": "Both archived object XY endpoint components equal the valid same-root hold endpoint.",
        "metrics": "Map-unit Euclidean RMSE. Bootstrap resamples scene RMSE differences, keeping all roots within scenes.",
        "support_check": "For each arm, direct difference from both writes matches the contrast of support on one fixed mode table.",
        "capacity": "The coordinate envelope is a necessary condition only; a fan shares one support over candidates and horizons.",
    }


def save_split(model, panel, sigma, coordinates, output, split, *, batch_size=2):
    arrays = collect_split(model, panel, sigma, coordinates, batch_size=batch_size)
    summary = summarize_split(arrays)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / f"{split}_action_effects.npz", **arrays)
    (output / f"{split}_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", type=torch.device)
    parser.add_argument("--output", default=OUTPUT, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_float32_matmul_precision("highest")
    model = data._native_model(args.device)
    model.enable_response_mixture(parameterization="relational", num_modes=4)
    coordinates = load_coordinates()
    train = data.load_data("train", data.ARCHIVE_DIR, args.device)
    sigma, count = _full_sigma(model, train)
    for split, panel in (("train", train), ("development", data.load_data("development", data.ARCHIVE_DIR, args.device))):
        summary, _ = save_split(model, panel, sigma, coordinates, args.output, split)
        print(f"{split}: {len(summary['scenes'])} scenes, {summary['roots']} roots", flush=True)
    metadata = {"optimization_steps": 0, "relation_seed": 1714, "full_sigma_calibration_arrivals": count,
                "full_sigma": sigma.cpu().tolist(), "device": str(args.device)}
    (args.output / "initialization.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
