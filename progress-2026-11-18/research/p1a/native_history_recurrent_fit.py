"""Fixed-budget fit of the chronological partial-arrival operator in §37."""

import argparse
import copy
import json
import time

import numpy as np
import torch

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import conditioned_rollout, inherit_window, next_edge_ancestry
from .native_full_depth_first_effect import branch_offsets
from .native_full_depth_reversible import SWITCH, action_embeddings, encode_rgb
from .native_full_depth_task_geometry import chart_at_latent, chart_pose, pose_chart
from .native_history_partial_arrivals import (
    FITTED, HOLD, HORIZONS, OUTPUT as PARTIAL, PAIRS as DEVELOPMENT, edge_observation,
    image_x, initial_edge_bank, partial_measurement, query_bank, readout_metrics,
)


OUTPUT = FITTED.parent / "native_history_recurrent_fit_20261115"
UPDATES = 48
SEED = 3715
ACTIONS = [0, 1, 2, 4, 5]
ARMS = ("joint", "no_ancestry_evidence", "input_mean")


@torch.no_grad()
def load_panel(model, hold, folder, drift):
    with np.load(folder / "fresh_switches.npz", allow_pickle=False) as actual, np.load(
        folder / "post_switch_candidates.npz", allow_pickle=False
    ) as fan:
        scenes = actual["scene_id"].tolist()
        modes = actual["mode"].tolist()
        phase = actual["phase"].tolist()
        roots = [phase.index(name) for name in fan["phase"].tolist()]
        if (scenes != fan["scene_id"].tolist() or fan["future_mode"].tolist() != [0, 1] or
                fan["mode"].tolist() != [modes[tick] for tick in roots] or
                np.any(fan["actions"][HOLD])):
            raise ValueError("Expected matched factual regimes and candidate roots")
        rgb = actual["rgb"]
        paired = hold.new_tensor(actual["path_state"][..., -1, 2:5])
        paired_y = hold.new_tensor(image_x(actual["path_rgb"][..., -1, :, :, :]))
        truth = hold.new_tensor(np.take(fan["state"], HORIZONS, axis=-2)[..., 2:5])
        source_chart = pose_chart(paired)
        source_chart[..., 0] = paired_y - drift
        executed = action_embeddings(model, hold.new_tensor(actual["actions"]))
        candidates = action_embeddings(model, hold.new_tensor(fan["actions"]))
        return {
            "scene_ids": scenes, "mode": modes, "phase": phase, "roots": roots,
            "initial": encode_rgb(model, rgb[:, 0]),
            "measurement": hold.new_tensor(image_x(rgb[:, 1:])),
            "past": torch.cat((hold.expand(len(scenes), 2, -1), executed), dim=1),
            "action": executed, "candidates": candidates[None].expand(len(scenes), -1, -1, -1),
            "source": source_chart, "future": pose_chart(truth), "xy": truth[..., :2],
        }


def replay(forecast, panel, rows, transition, weight, bias, drift, noise, arm):
    windows, log_p = initial_edge_bank(panel["initial"][rows])
    past = panel["past"][rows]
    action = panel["action"][rows]
    target = panel["measurement"][rows]
    candidate = panel["candidates"][rows]
    label = panel["mode"]
    roots = {}
    source, support, nll = [], [], []
    for tick in range(action.size(1)):
        def observe(issued):
            return partial_measurement(issued, target[:, tick, None, None], weight[:, 0], bias[0] + drift, noise)

        windows, log_p, prior, likelihood, issued = edge_observation(
            forecast, windows, log_p, transition, past[:, tick:tick + 2], action[:, None, tick:tick + 1], observe,
            keep_evidence=arm != "no_ancestry_evidence", mean_input=arm == "input_mean",
        )
        # Only the supervised source tensor uses the true predecessor. The replay never uses it.
        source.append(chart_at_latent(issued[:, label[max(0, tick - 1)]], weight, bias))
        nll.append(-(prior + likelihood).flatten(1).logsumexp(-1))
        support.append(log_p.logsumexp(dim=1).exp()[:, 1])
        if tick in panel["roots"]:
            roots[tick] = windows, log_p, prior

    def predict(tick, windows, log_weights):
        bank, ancestry = query_bank(windows, log_weights, transition)
        query_past = past[:, tick + 1:tick + 3]
        if arm == "input_mean":
            prediction = forecast(inherit_window(bank, ancestry), query_past, candidate)
        else:
            prediction = forecast(bank, query_past, candidate, ancestry)
        return chart_at_latent(prediction[..., HORIZONS, :].transpose(1, 2), weight, bias)

    future = [predict(tick, *roots[tick][:2]) for tick in panel["roots"]]
    return torch.stack(source, 1), torch.stack(future, 1), torch.stack(support, 1), torch.stack(nll, 1), roots, predict


def fit_loss(source, future, panel, rows, scale):
    # Keep hold absolute error and the error of each independent non-hold contrast.
    e = (future - panel["future"][rows]) / scale
    absolute = e[..., HOLD, :, :].square().mean()
    relative = (e[..., ACTIONS, :, :] - e[..., HOLD:HOLD + 1, :, :]).square().mean()
    paired = ((source - panel["source"][rows]) / scale).square().mean()
    return paired, absolute, relative


@torch.no_grad()
def attribute(forecast, panel, config):
    """Separate movement of ancestry weights from the action effects they can read."""
    _, future, _, _, roots, predict = replay(forecast, panel, slice(None), *config, "joint")
    transition = config[0]
    result = {}
    for q, tick in enumerate(panel["roots"]):
        windows, post, prior = roots[tick]
        omit = post.logsumexp(dim=1, keepdim=True) + (prior - prior.logsumexp(dim=1, keepdim=True))
        mode = panel["mode"][tick]
        before = next_edge_ancestry(omit, transition)[:, mode]
        after = next_edge_ancestry(post, transition)[:, mode]
        corner = []
        for index in range(4):
            unit = torch.full_like(post.flatten(1), -torch.inf)
            unit[:, index] = 0
            endpoint = predict(tick, windows, unit.unflatten(-1, (2, 2)))
            corner.append(endpoint[:, mode, ..., 1])
        edge_y = torch.stack(corner, 1)  # scene, edge, action, horizon
        relative = edge_y[:, :, ACTIONS] - edge_y[:, :, HOLD:HOLD + 1]
        span = relative.amax(dim=1) - relative.amin(dim=1)
        effect = future[:, q, mode, ..., 1] - predict(tick, windows, omit)[:, mode, ..., 1]
        effect_relative = effect[:, ACTIONS] - effect[:, HOLD:HOLD + 1]
        true = panel["future"][:, q, :, :, :, 1]
        true_k = true.diff(dim=1).squeeze(1)
        pred_k = future[:, q, ..., 1].diff(dim=1).squeeze(1)
        target = true_k[:, ACTIONS] - true_k[:, HOLD:HOLD + 1]
        predicted = pred_k[:, ACTIONS] - pred_k[:, HOLD:HOLD + 1]
        result[panel["phase"][tick]] = {
            "query_ancestry_total_variation_mean": (before - after).abs().sum(-1).mean().item() / 2,
            "prior_ancestry_largest_weight": before.max(-1).values.mean().item(),
            "relative_y_per_edge_span_rms": span.square().mean(dim=(0, 1)).sqrt().tolist(),
            "relative_y_effect_of_last_write_rms": effect_relative.square().mean(dim=(0, 1)).sqrt().tolist(),
            "target_relative_mode_gap_rms": target.square().mean(dim=(0, 1)).sqrt().tolist(),
            "prediction_relative_mode_gap_rms": predicted.square().mean(dim=(0, 1)).sqrt().tolist(),
            "relative_mode_gap_signed_gain": ((target * predicted).sum(dim=(0, 1)) /
                                               target.square().sum(dim=(0, 1)).clamp_min(1e-8)).tolist(),
        }
    return result


@torch.no_grad()
def evaluate(forecast, panel, config, arm):
    source, future, support, nll, roots, predict = replay(forecast, panel, slice(None), *config, arm)
    labels = support.new_tensor(panel["mode"])
    result = {
        "sensor_nll": nll.mean(dim=0).tolist(),
        "posterior_mode_one": support.tolist(),
        "brier_by_tick": (support - labels).square().mean(dim=0).tolist(),
        "correct_by_tick": ((support >= 0.5) == labels.bool()).sum(dim=0).tolist(),
        "paired_source_rmse_by_tick_and_chart": (source - panel["source"]).square().mean(dim=(0, 2)).sqrt().tolist(),
        "queries": {},
    }
    for q, tick in enumerate(panel["roots"]):
        mode = panel["mode"][tick]
        true_xy = panel["xy"][:, q, mode]
        chart = future[:, q, mode]
        predicted = chart_pose(chart)[..., :2]
        query = readout_metrics(predicted, true_xy)
        query["hold_y_chart_rmse"] = (chart[:, HOLD, :, 1] - panel["future"][:, q, mode, HOLD, :, 1]
                                     ).square().mean(dim=0).sqrt().tolist()
        true_gap = panel["future"][:, q].diff(dim=1).squeeze(1)
        forecast_gap = future[:, q].diff(dim=1).squeeze(1)
        relative_true_gap = true_gap[:, ACTIONS] - true_gap[:, HOLD:HOLD + 1]
        relative_forecast_gap = forecast_gap[:, ACTIONS] - forecast_gap[:, HOLD:HOLD + 1]
        query["relative_y_mode_gap_rmse"] = (relative_forecast_gap - relative_true_gap
                                            ).square().mean(dim=(0, 1)).sqrt()[:, 1].tolist()
        query["relative_y_mode_gap_target_rms"] = relative_true_gap.square().mean(dim=(0, 1)).sqrt()[:, 1].tolist()
        if arm != "no_ancestry_evidence":
            windows, post, prior = roots[tick]
            omit = post.logsumexp(dim=1, keepdim=True) + (prior - prior.logsumexp(dim=1, keepdim=True))
            without = chart_pose(predict(tick, windows, omit)[:, mode])[..., :2]
            query["without_last_ancestry"] = readout_metrics(without, true_xy)
            query["last_ancestry_xy_shift_rms"] = (predicted - without).square().mean(dim=(0, 1, 3)).sqrt().tolist()
        result["queries"][panel["phase"][tick]] = query
    return result


def run(device, *, attribution=False):
    torch.manual_seed(SEED)
    model, adapter, gate, hold, *_ = substep.load_core(device)
    checkpoint = torch.load(FITTED / "forecast_only.pt", map_location=device, weights_only=True)
    original_adapter = checkpoint["adapter"]
    original_head = copy.deepcopy(model.pred_proj.state_dict())
    weight, bias = (checkpoint["chart"][key] for key in ("weight", "bias"))
    calibration = json.loads((PARTIAL / "stage.json").read_text(encoding="utf-8"))
    drift = hold.new_tensor(calibration["sensor_drift"])
    noise = hold.new_tensor(calibration["sensor_std"])
    transition = hold.new_tensor([[1 - SWITCH, SWITCH], [SWITCH, 1 - SWITCH]])
    config = (transition, weight, bias, drift, noise)

    def forecast(windows, past, actions, ancestry=None):
        return conditioned_rollout(model, windows, past, actions, branch_offsets(adapter, len(windows)),
                                   first_effect=gate, ancestry=ancestry)

    train = load_panel(model, hold, FITTED.parent / "native_paired_effect_pairs_20261115", drift)
    dev = load_panel(model, hold, DEVELOPMENT, drift)
    extra = load_panel(model, hold, FITTED.parent / "native_task_geometry_pairs_20261115", drift)
    panels = {"fit": train, "development": dev, "corroboration": extra}
    scenes = {name: value["scene_ids"] for name, value in panels.items()}
    groups = list(scenes.values()) + [calibration["calibration_scene_ids"]]
    if len(set().union(*groups)) != sum(map(len, groups)):
        raise ValueError("Fit, calibration and both development panels must be scene-disjoint")
    adapter.load_state_dict(original_adapter)
    if attribution:
        result = {"frozen": {name: attribute(forecast, value, config) for name, value in panels.items()}}
        learned = torch.load(OUTPUT / "joint.pt", map_location=device, weights_only=True)
        adapter.load_state_dict(learned["adapter"])
        model.pred_proj.load_state_dict(learned["pred_proj"])
        result["joint"] = {name: attribute(forecast, value, config) for name, value in panels.items()}
        return result
    scale = (train["future"][..., ACTIONS, :, :] - train["future"][..., HOLD:HOLD + 1, :, :]
             ).square().mean().sqrt().clamp_min(1.0)
    rng = np.random.default_rng(SEED)
    schedule = [rng.choice(len(train["scene_ids"]), 2, replace=False).tolist() for _ in range(UPDATES)]
    result = {"scenes": scenes, "steps": UPDATES, "chart_scale": scale.item(), "arms": {}}
    result["arms"]["frozen"] = {name: evaluate(forecast, value, config, "joint")
                                for name, value in panels.items() if name != "fit"}
    expected = calibration["queries"]
    for key, row in result["arms"]["frozen"]["development"]["queries"].items():
        torch.testing.assert_close(torch.tensor(row["absolute_xy_rmse"]),
                                   torch.tensor(expected[key]["postmerge"]["absolute_xy_rmse"]),
                                   atol=5e-4, rtol=2e-5)
    parameters = list(adapter.requires_grad_(True).parameters()) + list(model.pred_proj.requires_grad_(True).parameters())
    result["trainable_parameters"] = sum(p.numel() for p in parameters)
    print("Frozen §36 numerical parity. Start matched fit:", result["trainable_parameters"], "parameters", flush=True)
    for arm in ARMS:
        adapter.load_state_dict(original_adapter)
        model.pred_proj.load_state_dict(original_head)
        optimizer = torch.optim.AdamW(parameters, lr=0.0005)
        loss_trace = []
        started = time.perf_counter()
        for i, rows in enumerate(schedule):
            source, future, *_ = replay(forecast, train, rows, *config, arm)
            parts = fit_loss(source, future, train, rows, scale)
            loss = sum(parts)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
            optimizer.step()
            entry = {"step": i + 1, "source": parts[0].item(), "absolute": parts[1].item(),
                     "relative": parts[2].item(), "gradient_norm": norm.item()}
            loss_trace.append(entry)
            if i == 0 or (i + 1) % 12 == 0:
                print(arm, entry, flush=True)
        elapsed = time.perf_counter() - started
        arm_result = {name: evaluate(forecast, value, config, arm) for name, value in panels.items()}
        arm_result.update(seconds=elapsed, loss=loss_trace)
        result["arms"][arm] = arm_result
        with (OUTPUT / f"{arm}.pt").open("xb") as stream:
            torch.save({"adapter": adapter.state_dict(), "pred_proj": model.pred_proj.state_dict()}, stream)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribute", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    result = run(torch.device("cuda" if torch.cuda.is_available() else "cpu"), attribution=args.attribute)
    output = OUTPUT / ("attribution.json" if args.attribute else "stage.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(output, flush=True)
