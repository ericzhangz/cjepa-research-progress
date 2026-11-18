"""Replay §36 with a single scalar visible measurement after the initial RGB frame."""

import json

import numpy as np
import torch

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import (
    conditioned_rollout, edge_posterior, historical_support, inherit_window, next_edge_ancestry,
)
from .native_full_depth_first_effect import branch_offsets
from .native_full_depth_reversible import SWITCH, action_embeddings, block_windows, encode_rgb
from .native_full_depth_task_geometry import chart_at_latent, chart_pose
from .native_pusht_visible_measurement import visible_t_chroma
from .native_visible_source import FITTED, PAIRS


OUTPUT = FITTED.parent / "native_history_partial_arrivals_20261115"
HORIZONS = [0, 1, 3]
HOLD = 3


def image_x(rgb):
    gray, valid = visible_t_chroma(rgb)
    weights = np.where(valid, gray, 0)
    mass = weights.sum(axis=(-2, -1))
    if (mass <= 0).any():
        raise ValueError("The declared scalar sensor requires visible T pixels")
    return (weights * (np.arange(96) + 0.5)).sum(axis=(-2, -1)) / mass * (512 / 96) - 0.5


def query_bank(windows, log_weights, transition):
    if log_weights.ndim == 2:
        _, ancestry = historical_support(log_weights, transition)
        return windows, ancestry
    return windows.flatten(1, 2), next_edge_ancestry(log_weights, transition)


def partial_measurement(issued, measurement, axis, offset, noise):
    innovation = measurement - (issued @ axis + offset)
    log_likelihood = -0.5 * (innovation / noise).square()
    revised = issued + innovation[..., None] * (axis / axis.square().sum())
    return revised, log_likelihood


def initial_edge_bank(initial):
    windows = initial[:, None, None, None].expand(-1, 2, 2, 3, -1)
    weights = torch.eye(2, dtype=initial.dtype, device=initial.device)[None].log() - np.log(2)
    return windows, weights.expand(len(initial), -1, -1)


def edge_observation(forecast, windows, weights, transition, past, action, measure, *,
                     keep_evidence=True, mean_input=False):
    """Issue the one real command before consuming its observation and shifting the bank."""
    batch = len(windows)
    grouped = windows.transpose(1, 2).flatten(0, 1)
    log_group = weights.transpose(1, 2)
    mass = log_group.logsumexp(-1, keepdim=True)
    conditional = (log_group - torch.where(mass.isfinite(), mass, 0)).exp()
    ancestry = conditional.flatten(0, 1)[:, None].expand(-1, 2, -1)
    older = inherit_window(grouped, ancestry)
    past = past.repeat_interleave(2, dim=0)
    action = action.repeat_interleave(2, dim=0)
    forecasts = forecast(older, past, action) if mean_input else forecast(grouped, past, action, ancestry)
    issued = forecasts[:, 0, :, 0].unflatten(0, (batch, 2))
    revised, likelihood = measure(issued)
    prior, post = edge_posterior(weights, transition, likelihood)
    if not keep_evidence:
        post = post.logsumexp(dim=1, keepdim=True) + (prior - prior.logsumexp(dim=1, keepdim=True))
    windows = torch.cat((older.unflatten(0, (batch, 2))[..., 1:, :], revised.unsqueeze(-2)), dim=-2)
    return windows, post, prior, likelihood, issued


def readout_metrics(prediction, truth):
    d = prediction - truth
    relative = (prediction - prediction[:, HOLD:HOLD + 1]) - (truth - truth[:, HOLD:HOLD + 1])
    rel = relative[:, [0, 1, 2, 4, 5]]
    return {
        "absolute_xy_rmse": d.square().sum(-1).mean(dim=(0, 1)).sqrt().tolist(),
        "relative_xy_rmse": rel.square().sum(-1).mean(dim=(0, 1)).sqrt().tolist(),
        "per_scene_absolute_xy_mse": d.square().sum(-1).mean(dim=1).tolist(),
        "per_scene_relative_xy_mse": rel.square().sum(-1).mean(dim=1).tolist(),
    }


@torch.inference_mode()
def run(device):
    model, adapter, gate, hold, *_ = substep.load_core(device)
    fitted = torch.load(FITTED / "forecast_only.pt", map_location=device, weights_only=True)
    adapter.load_state_dict(fitted["adapter"])
    w, b = (fitted["chart"][k] for k in ("weight", "bias"))
    axis = w[:, 0]
    transition = hold.new_tensor([[1 - SWITCH, SWITCH], [SWITCH, 1 - SWITCH]])

    def forecast(windows, past, actions, ancestry=None, trace=None):
        return conditioned_rollout(model, windows, past, actions, branch_offsets(adapter, len(windows)),
                                   first_effect=gate, ancestry=ancestry, trace=trace)

    with np.load(substep.PAIRS / "calibration.npz", allow_pickle=False) as cal:
        rgb = cal["rgb"]
        cal_scenes = cal["scene_id"].tolist()
        z = encode_rgb(model, rgb)
        actions = action_embeddings(model, hold.new_tensor(cal["actions"]))
        cal_error = []
        for tick in range(actions.size(2)):
            history, past, cmd = substep.root_controls({"z": z, "actions": hold.new_tensor(cal["actions"])},
                                                        actions, slice(None), tick, hold)
            command = action_embeddings(model, cmd)
            forecasted = forecast(history, past, command[:, None, None])[:, 0, :, 0]
            labels = torch.arange(2, device=device)[None, :, None].expand(z.shape[:3]).reshape(-1)
            matched = forecasted[torch.arange(len(labels), device=device), labels]
            observed = hold.new_tensor(image_x(rgb[..., tick + 1, :, :, :])).reshape(-1)
            cal_error.append(observed - (matched @ axis + b[0]))
    cal_error = torch.cat(cal_error)
    drift = cal_error.mean()
    noise = cal_error.std(unbiased=False)
    if noise <= 0:
        raise ValueError("Calibration must supply a positive forecast residual variance")

    def observe(z, y):
        return partial_measurement(z, y, axis, b[0] + drift, noise)

    with np.load(PAIRS / "fresh_switches.npz", allow_pickle=False) as actual, np.load(
        PAIRS / "post_switch_candidates.npz", allow_pickle=False
    ) as fans:
        scenes = actual["scene_id"].tolist()
        phases = actual["phase"].tolist()
        modes = actual["mode"].tolist()
        if scenes != fans["scene_id"].tolist() or set(scenes) & set(cal_scenes):
            raise ValueError("The past trajectory, candidates and separate calibration must align")
        ticks = [phases.index(name) for name in fans["phase"].tolist()]
        observed_modes = [modes[tick] for tick in ticks]
        if observed_modes != fans["mode"].tolist() or fans["future_mode"].tolist() != [0, 1]:
            raise ValueError("Each fixed fan must have matching actual and counterfactual mode labels")
        rgb = actual["rgb"]
        initial = encode_rgb(model, rgb[:, 0])
        y = hold.new_tensor(image_x(rgb[:, 1:]))
        commands = action_embeddings(model, hold.new_tensor(actual["actions"]))
        plans = action_embeddings(model, hold.new_tensor(fans["actions"]))[None].expand(len(scenes), -1, -1, -1)
        physical_xy = hold.new_tensor(fans["state"][..., HORIZONS, 2:4])
    batch = len(scenes)
    beginning = initial[:, None, None].expand(-1, 2, 3, -1)
    equal = hold.new_full((batch, 2), -np.log(2))
    start = initial_edge_bank(initial)
    states = {"premerge": (beginning, equal),
              "postmerge": start, "no_ancestry_evidence": start}
    past = torch.cat((hold.expand(batch, 2, -1), commands), dim=1)
    trajectory, queries = [], {}
    for tick in range(commands.size(1)):
        actions = commands[:, None, tick:tick + 1]
        executed_past = past[:, tick:tick + 2]
        traces = {}
        fixed_root_omission = None
        for name, (windows, weights) in states.items():
            if name == "premerge":
                prior, ancestry = historical_support(weights, transition)
                issued = forecast(windows, executed_past, actions, ancestry)[:, 0, :, 0]
                revised, ll = observe(issued, y[:, tick, None])
                post = (prior + ll).log_softmax(-1)
                older = inherit_window(windows, ancestry)
                windows = torch.cat((older[..., 1:, :], revised.unsqueeze(-2)), dim=-2)
                path = None
            else:
                windows, post, prior, ll, _ = edge_observation(
                    forecast, windows, weights, transition, executed_past, actions,
                    lambda issued: observe(issued, y[:, tick, None, None]), keep_evidence=name == "postmerge",
                )
                if name == "postmerge":
                    fixed_root_omission = post.logsumexp(dim=1, keepdim=True) + (
                        prior - prior.logsumexp(dim=1, keepdim=True)
                    )
                conditional_before = (prior - prior.logsumexp(dim=1, keepdim=True)).exp()
                conditional_after = (post - post.logsumexp(dim=1, keepdim=True)).exp()
                path = {
                    "conditional_parent_before": conditional_before.tolist(),
                    "conditional_parent_after": conditional_after.tolist(),
                    "conditional_parent_update_rms": (conditional_after - conditional_before).square().mean().sqrt().item(),
                    "same_destination_parent_log_likelihood_gap_rms": ll.diff(dim=1).square().mean().sqrt().item(),
                }
            states[name] = windows, post
            mode_log = post if post.ndim == 2 else post.logsumexp(dim=1)
            traces[name] = {"posterior_mode_one": mode_log[:, 1].exp().tolist(),
                            "source_log_likelihood": ll.tolist(), "ancestry": path}
        trajectory.append({"tick": tick, "phase": phases[tick], "observed_x": y[:, tick].tolist(), "arms": traces})
        if tick not in ticks:
            continue
        query = {}
        mode = modes[tick]  # for scoring only
        q = ticks.index(tick)
        truth = physical_xy[:, q, mode]
        for name, (windows, log_p) in states.items():
            bank, ancestry = query_bank(windows, log_p, transition)

            def evaluate(pred):
                chart = chart_at_latent(pred[:, :, mode, HORIZONS], w, b)
                xy = chart_pose(chart)[..., :2]
                return readout_metrics(xy, truth), xy

            query_past = past[:, tick + 1:tick + 3]
            inherited = forecast(bank, query_past, plans, ancestry)
            query[name], xy = evaluate(inherited)
            if name == "postmerge":
                ancestry_without_this_update = next_edge_ancestry(fixed_root_omission, transition)
                omit = forecast(bank, query_past, plans, ancestry_without_this_update)
                current_chart = chart_at_latent(inherited[:, :, mode, HORIZONS], w, b)
                omitted_chart = chart_at_latent(omit[:, :, mode, HORIZONS], w, b)
                chart_delta = current_chart - omitted_chart
                query["fixed_root_ancestry_write"] = {
                    "chart_rms_by_horizon_and_coordinate": chart_delta.square().mean(dim=(0, 1)).sqrt().tolist(),
                    "absolute_and_relative_without_last_ancestry_write": evaluate(omit)[0],
                    "per_scene_candidate_chart_delta": chart_delta.tolist(),
                }
                query["postmerge_input_mean"], mixed_xy = evaluate(
                    forecast(inherit_window(bank, ancestry), query_past, plans))
                expanded = torch.stack([forecast(bank[:, i], query_past, plans) for i in range(bank.size(1))], dim=-2)
                expanded_xy = chart_pose(chart_at_latent(expanded[:, :, mode, HORIZONS], w, b))[..., :2]
                mixed_path_xy = torch.einsum("bi,bchid->bchd", ancestry[:, mode], expanded_xy)
                query["postmerge_path_mean"] = readout_metrics(mixed_path_xy, truth)
                query["attention_location_xy_rms"] = (xy - mixed_xy).square().mean(dim=(0, 1, 3)).sqrt().tolist()
        # All comparisons are evaluated after the partial replays have issued their predictions from the same root.
        full = encode_rgb(model, rgb[:, tick - 1:tick + 2])
        query["complete_window_native"] = evaluate(forecast(full, query_past, plans))[0]
        queries[phases[tick]] = query
    return {"scene_ids": scenes, "calibration_scene_ids": cal_scenes,
            "sensor_drift": drift.item(), "sensor_std": noise.item(), "sensor": "T visible centroid X",
            "horizon_raw_steps": [5, 10, 20], "trajectory": trajectory, "queries": queries}


if __name__ == "__main__":
    result = run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "stage.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(OUTPUT / "stage.json", flush=True)
