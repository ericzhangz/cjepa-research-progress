"""Inspect the untrained native activation relations on recorded official PushT inputs."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

if __package__:
    from . import native_response_mixture_regulation as data
    from .native_response_observation import _log_density, _residual_scale
else:
    import native_response_mixture_regulation as data
    from native_response_observation import _log_density, _residual_scale


SOURCE_STEPS = (105, 110)
QUERY_STEP = 125
SCENE_ID = 101000
HOLD_CANDIDATE = 3
RELATION_FIELDS = ("preactivation", "activation_delta", "relation_load", "relation_dual")
OUTPUT = data.RUNS / "native_response_relations_20261114"


def _source(visible, actions, step):
    end = step // 5
    return visible[:, end - 3:end], actions[:, end - 3:end], visible[:, end]


@torch.no_grad()
def _full_sigma(model, panel):
    residuals = []
    for start in range(0, len(panel.scene_ids), 2):
        rows = slice(start, start + 2)
        visible, commands = data._factual_inputs(panel, rows)
        actions = model.action_encoder(commands)
        valid = panel.valid[rows].flatten(0, 2)
        for step in SOURCE_STEPS:
            active = valid[:, step // 5 - 1]
            window, controls, target = _source(visible[active], actions[active], step)
            prediction = model.predict(window, controls)[:, -1]
            residuals.append(target - prediction)
    residuals = torch.cat(residuals)
    if len(residuals) != 256 or not torch.isfinite(residuals).all():
        raise RuntimeError("Expected 256 executed train FULL residuals for fixed scale calibration")
    return _residual_scale(residuals), len(residuals)


def _mode_span(value, axis):
    return (value.amax(axis) - value.amin(axis)).abs().max().item()


def _rms(value):
    return value.square().mean().sqrt().item()


def _relation_summary(activation, load, dual, op):
    with torch.no_grad():
        v = torch.nn.functional.gelu(activation)
        raw_load = v @ op.relation_directions.T - op.relation_thresholds
        coupled_on = (raw_load <= 0) & (dual > 0)
        coupled_off = (raw_load > 0) & (dual == 0)
        return {
            "non_native_relation_slots": dual[..., 1:, :].numel(),
            "non_native_positive_duals": int((dual[..., 1:, :] > 0).sum()),
            "non_native_modes_with_two_positive_duals": int((dual[..., 1:, :] > 0).all(-1).sum()),
            "non_native_newly_loaded_at_solution": int(coupled_on[..., 1:, :].sum()),
            "non_native_unloaded_at_solution": int(coupled_off[..., 1:, :].sum()),
            "max_dual_load_consistency_residual": (dual - load * op.mode_strengths()).abs().max().item(),
        }


def _source_trace(model, panel, rows, sigma):
    op = model.response_mixture
    visible, commands = data._factual_inputs(panel, rows)
    actions = model.action_encoder(commands)
    prior = op.initial_state(len(visible))
    log_support = prior
    source_fields = []
    action_mean, action_std = data._action_moments(commands)
    hold = ((commands.new_zeros(5, 2) - action_mean) / action_std).flatten()
    for step in SOURCE_STEPS:
        end = step // 5
        if not panel.valid[rows, ..., end - 1].all():
            raise RuntimeError(f"Missing actual execution for departure at {step - 5}")
        window, controls, target = _source(visible, actions, step)
        if not torch.equal(target, panel.full[rows, ..., end, :].flatten(0, 2)):
            raise RuntimeError(f"Expected the scheduled public FULL reveal at {step}")
        _, advanced, issued = model.predict_response(window, controls, log_support, factual=True)
        if not torch.equal(advanced, log_support):
            raise RuntimeError("Issued predictions changed public support before the real arrival")
        likelihood = _log_density(target, issued.means, sigma)
        _, written, _ = op.condition(issued, likelihood)
        with torch.no_grad():
            native = model.predict(window, controls)[:, -1]
            torch.testing.assert_close(issued.means[:, 0], native, atol=2e-5, rtol=2e-5)
        source_fields.append({
            "window": window, "normalized_commands": commands[:, end - 3:end],
            "public_target": target, "means": issued.means, "native_means": native,
            "log_likelihood": likelihood, "log_support_before": log_support, "log_support_after": written,
            **{key: getattr(issued, key) for key in RELATION_FIELDS},
        })
        log_support = written
    if not torch.equal(source_fields[0]["normalized_commands"][:, -1], hold.expand(len(visible), -1)):
        raise RuntimeError("Expected an exact raw zero command before the step-105 reveal")
    if (source_fields[1]["normalized_commands"][:, -1] == hold).all(-1).any():
        raise RuntimeError("Expected a nonzero actually executed step-110 probe")
    return prior, log_support, source_fields, hold


def _h2_gradients(modes, log_support, sources, panel, rows, op):
    index = data.QUERY_STEPS.index(QUERY_STEP)
    train_latent = panel.future[..., index, :, 1, :]
    train_physical = panel.object_xy[..., index, :, 1, :]
    train_valid = panel.query_valid[..., index, :, 1]
    scales = tuple(target[train_valid].var(dim=0, unbiased=False)[None]
                   for target in (train_latent, train_physical))
    if not all(torch.isfinite(scale).all() and (scale > 0).all() for scale in scales):
        raise RuntimeError("The fixed H2 target scales must be positive and finite")
    latent = train_latent[rows].flatten(0, 2).unsqueeze(-2)
    physical = train_physical[rows].flatten(0, 2).unsqueeze(-2)
    valid = train_valid[rows].flatten(0, 2).unsqueeze(-1)
    # Extract the existing offline readout only. The model above uses official weights.
    coordinates = torch.load(data.WARM_START, map_location="cpu", weights_only=True)["object_coordinates"]
    energy, active = data._fan_energy(modes[:, :, 1:2], latent, physical, valid, coordinates, scales)
    loss = data._use_loss(log_support, energy, active)
    parameters = {"directions": op.relation_directions, "raw_strengths": op.relation_raw_strengths}
    likelihoods = tuple(source["log_likelihood"] for source in sources)
    means = tuple(source["means"] for source in sources)
    likelihood_gradients = torch.autograd.grad(loss, likelihoods, retain_graph=True)
    mean_gradients = torch.autograd.grad(loss, means, retain_graph=True)

    def norms(gradients):
        return {key: gradient.norm().item() for key, gradient in zip(parameters, gradients, strict=True)}

    source_results = []
    for likelihood, likelihood_gradient, mean_gradient in zip(likelihoods, likelihood_gradients, mean_gradients,
                                                               strict=True):
        upstream = torch.autograd.grad(likelihood, tuple(parameters.values()), grad_outputs=likelihood_gradient,
                                       retain_graph=True)
        source_results.append({"to_log_likelihood": likelihood_gradient.norm().item(),
                               "to_issued_means": mean_gradient.norm().item(),
                               "via_this_likelihood": norms(upstream)})
    total = torch.autograd.grad(loss, tuple(parameters.values()), retain_graph=True)
    future_only_loss = data._use_loss(log_support.detach(), energy, active)
    future_only = torch.autograd.grad(future_only_loss, tuple(parameters.values()))
    return {
        "valid_endpoints": int(valid.sum()),
        "active_fans": int(active.sum()),
        "total_parameter_gradient_norm": norms(total),
        "future_path_only_gradient_norm": norms(future_only),
        "source_gradient_norms": dict(zip(map(str, SOURCE_STEPS), source_results, strict=True)),
    }, energy, likelihood_gradients


def _run(device):
    torch.set_float32_matmul_precision("highest")
    panel = data.load_data("train", data.ARCHIVE_DIR, device)
    model = data._native_model(device)
    op = model.enable_response_mixture(parameterization="relational", num_modes=4)
    sigma, calibration_count = _full_sigma(model, panel)
    row = np.flatnonzero(panel.scene_ids == SCENE_ID).item()
    rows = slice(row, row + 1)
    prior, log_support, sources, hold = _source_trace(model, panel, rows, sigma)
    history, history_commands, candidates = data._query_inputs(panel, rows, (QUERY_STEP,))
    trace = {}
    actions = model.action_encoder(candidates.flatten(1, 2)).unflatten(1, candidates.shape[1:3])
    modes = model.rollout_response_modes(history, model.action_encoder(history_commands),
                                         actions, trace=trace)
    if not torch.equal(candidates[:, HOLD_CANDIDATE], hold.expand_as(candidates[:, HOLD_CANDIDATE])):
        raise RuntimeError("Expected candidate 3 to use only exact raw zero commands")
    if not torch.equal(candidates[:, :, 0], candidates[:, :, 1]):
        raise RuntimeError("Expected repeated commands in the existing H2 archive")
    expected_probe = candidates[:, 0, 0]
    if not torch.equal(expected_probe, sources[1]["normalized_commands"][:, -1]):
        raise RuntimeError("Expected the actual step-110 probe to match candidate 0's first command")

    gradients, energy, likelihood_gradients = _h2_gradients(modes, log_support, sources, panel, rows, op)
    with torch.no_grad():
        # Diagnostic accumulation in float64 prevents float32 rounding of the
        # common mode component from dominating tiny support interventions.
        mode_values = modes.double()
        kept_support = log_support.double().softmax(-1)
        initial_support = prior.double().softmax(-1)
        kept = torch.einsum("bk,bshkd->bshd", kept_support, mode_values)
        initial = torch.einsum("bk,bshkd->bshd", initial_support, mode_values)
        effect = torch.einsum("bk,bshkd->bshd", kept_support - initial_support,
                              mode_values - mode_values[..., :1, :])
        source_summaries = {}
        for step, source in zip(SOURCE_STEPS, sources, strict=True):
            source_summaries[str(step)] = {
                "issued_from_step": step - 5,
                "executed_tracks": len(source["means"]),
                "max_native_mode_0_error": (source["means"][:, 0] - source["native_means"]).abs().max().item(),
                "max_mode_prediction_span": _mode_span(source["means"], -2),
                "max_mode_log_likelihood_span": _mode_span(source["log_likelihood"], -1),
                "max_support_tv": (0.5 * (source["log_support_after"].exp() -
                                          source["log_support_before"].exp()).abs().sum(-1)).max().item(),
                **_relation_summary(source["preactivation"][:, None], source["relation_load"],
                                    source["relation_dual"], op),
            }

        summary = {
            "scope": "Untrained official PushT zero-update computation and gradient check on train data.",
            "optimization_steps": 0,
            "device": str(device),
            "inputs": {
                "archive": data.ARCHIVE_DIR.as_posix(),
                "official_host": (data.ROOT / "research_sources_2026-10-22/lewm_assets/model/weights.pt").as_posix(),
                "split": "train",
                "checked_scene": SCENE_ID,
                "histories": len(history),
                "actual_source_steps": list(SOURCE_STEPS),
                "other_arrivals": "Public frames still enter causal windows; no likelihood writes at other steps.",
                "observation_model": "Shared fixed diagonal Gaussian, identity observation map on scheduled FULL.",
                "offline_h2_target": "Independent candidate executions, full-encoding and object XY objectives.",
                "offline_xy_readout": data.WARM_START.as_posix(),
                "detail": "Only the offline readout is extracted from pilot.pt; no trained response weights are loaded.",
            },
            "initialization": {
                "modes": op.num_modes,
                "relations": op.relation_directions.shape[0],
                "seed": 1714,
                "directions": "CPU Gaussian normalized rowwise at initialization",
                "direction_row_norms": op.relation_directions.norm(dim=1).tolist(),
                "direction_cosine": torch.nn.functional.cosine_similarity(
                    op.relation_directions[0], op.relation_directions[1], dim=0,
                ).item(),
                "threshold_initial_value": 0,
                "raw_non_native_strength_distribution": "-1 + 0.1 * standard normal",
                "native_mode_strength": 0,
                "non_native_strength_range": [op.mode_strengths()[1:].min().item(), op.mode_strengths()[1:].max().item()],
            },
            "full_scale_calibration": {"actual_transitions": calibration_count, "train_scene_count": len(panel.scene_ids),
                                       "min_sigma": sigma.min().item(), "max_sigma": sigma.max().item()},
            "source_checks": source_summaries,
            "query_checks": {
                "root_step": QUERY_STEP,
                "candidate_count_per_history": candidates.size(1),
                "horizon_in_environment_steps": [5, 10],
                "hold_candidate_index": HOLD_CANDIDATE,
                "candidate_h2_commands": "Repeats of the first command only.",
                "support_readout_accumulation_dtype": "float64, diagnostic only",
                "public_root_equal_within_all_mirrored_pairs": bool(torch.equal(history[0::2], history[1::2])),
                "root_history_actions_equal_within_all_mirrored_pairs": bool(torch.equal(history_commands[0::2],
                                                                                         history_commands[1::2])),
                "mean_prediction_kept_minus_initial_rms_by_horizon": [_rms(effect[:, :, h])
                                                                      for h in range(modes.size(2))],
                "hold_mean_kept_minus_initial_rms_by_horizon": [_rms(effect[:, HOLD_CANDIDATE, h])
                                                               for h in range(modes.size(2))],
                "hold_absolute_prediction_mode_span_by_horizon": [_mode_span(modes[:, HOLD_CANDIDATE, h], -2)
                                                                   for h in range(modes.size(2))],
                "hold_kept_mean_mirror_difference_rms_by_horizon": [
                    _rms(kept[0::2, HOLD_CANDIDATE, h] - kept[1::2, HOLD_CANDIDATE, h])
                    for h in range(modes.size(2))
                ],
                **_relation_summary(trace["preactivation"], trace["relation_load"], trace["relation_dual"], op),
            },
            "h2_gradient_checks": gradients,
        }
    tensors = {
        "full_sigma": sigma, "query_window": history, "query_history_normalized_commands": history_commands,
        "query_candidate_normalized_commands": candidates, "query_mode_means": modes,
        "query_kept_log_support": log_support, "query_initial_log_support": prior,
        "query_kept_mean": kept, "query_initial_mean": initial, "query_h2_group_energy": energy,
        "query_kept_minus_initial_mean": effect,
        **{f"query_{key}": trace[key] for key in RELATION_FIELDS},
    }
    for step, source, gradient in zip(SOURCE_STEPS, sources, likelihood_gradients, strict=True):
        tensors.update({f"source_{step}_{key}": value for key, value in source.items()})
        tensors[f"source_{step}_h2_log_likelihood_gradient"] = gradient
    arrays = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise RuntimeError("Found a nonfinite source, relation, support or H2 gradient trace")
    return summary, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", type=torch.device)
    parser.add_argument("--output", default=OUTPUT, type=Path)
    args = parser.parse_args()
    summary, arrays = _run(args.device)
    content = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "native_intermediates.npz", **arrays)
    (args.output / "native_intermediate_summary.json").write_text(content, encoding="utf-8")
    print(content, end="", flush=True)


if __name__ == "__main__":
    main()
