"""Measure and fit a true-history contrast at a fixed current physical mode."""

import argparse
import copy
import json
import time

import numpy as np
import torch

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import conditioned_rollout, inherit_window
from .native_full_depth_first_effect import branch_offsets
from .native_full_depth_reversible import SWITCH, action_embeddings, encode_rgb
from .native_full_depth_task_geometry import chart_at_latent, chart_pose, pose_chart
from .native_history_crossed_physics import OUTPUT as PHYSICS
from .native_history_partial_arrivals import (
    FITTED, HOLD, HORIZONS, OUTPUT as SCALAR, edge_observation, image_x, initial_edge_bank,
    partial_measurement, query_bank,
)
from .native_history_recurrent_fit import OUTPUT as OLD_FIT
from .native_reversible_pusht import ReversiblePushT, make_scene


OUTPUT = PHYSICS.parent / "native_history_matched_fit_20261115"
NON_HOLD = [0, 1, 2, 4, 5]
UPDATES = 48
SEED = 3815


@torch.no_grad()
def load_panel(model, hold):
    with np.load(PHYSICS / "physical.npz", allow_pickle=False) as raw:
        observed = raw["observed_x"]
        np.testing.assert_array_equal(observed[..., :6], np.broadcast_to(observed[:, :1, :1, :6], (len(observed), 2, 2, 6)))
        commands = action_embeddings(model, hold.new_tensor(raw["commands"])[None])[0]
        plans = action_embeddings(model, hold.new_tensor(raw["plans"]))
        truth = hold.new_tensor(np.take(raw["future_state"], HORIZONS, axis=-2)[..., 2:5])
        return {
            "ids": raw["scene_ids"].tolist(), "initial": encode_rgb(model, raw["initial_rgb"]),
            "measured": hold.new_tensor(observed), "past": torch.cat((hold.expand(2, -1), commands)),
            "action": commands, "plans": plans, "xy": truth[..., :2], "chart": pose_chart(truth),
        }


def trace(forecast, panel, rows, config, *, reader=None):
    transition, weight, bias, drift, noise = config
    measured = panel["measured"][rows].flatten(0, 2)
    scenes = len(panel["initial"][rows])
    batch = measured.size(0)
    windows, log_p = initial_edge_bank(panel["initial"][rows].repeat_interleave(4, dim=0))
    past = panel["past"][None].expand(batch, -1, -1)
    actions = panel["action"][None, None].expand(batch, -1, -1, -1)
    issued_source = []
    for tick in range(actions.size(2)):
        def observe(issued):
            return partial_measurement(issued, measured[:, tick, None, None], weight[:, 0], bias[0] + drift, noise)

        windows, log_p, prior, _, issued = edge_observation(
            forecast, windows, log_p, transition, past[:, tick:tick + 2],
            actions[:, :, tick:tick + 1], observe,
        )
        issued_mean = issued @ weight[:, 0] + bias[0] + drift
        issued_source.append(issued_mean.unflatten(0, (scenes, 2, 2)))

    plans = panel["plans"][None].expand(batch, -1, -1, -1)
    edges = torch.arange(2, device=measured.device)
    physical_mode = edges.repeat(scenes * 2)
    index = torch.arange(batch, device=measured.device)
    current_mask = edges[None] == physical_mode[:, None]

    reader = forecast if reader is None else reader

    def read(log_weights, *, location="attention"):
        bank, ancestry = query_bank(windows, log_weights, transition)
        query_past = past[:, -2:]

        def to_chart(native):
            selected = native[index, :, physical_mode][:, :, HORIZONS]
            return chart_at_latent(selected, weight, bias)

        if location == "expanded":
            by_edge = torch.stack([to_chart(reader(window, query_past, plans)) for window in bank.unbind(dim=1)], dim=1)
            mixed = torch.einsum("bi,biahd->bahd", ancestry[index, physical_mode], by_edge)
            return mixed.unflatten(0, (scenes, 2, 2))
        if location == "input":
            native = reader(inherit_window(bank, ancestry), query_past, plans)
        else:
            native = reader(bank, query_past, plans, ancestry)
        return to_chart(native).unflatten(0, (scenes, 2, 2))

    hard_current = log_p.masked_fill(~current_mask[:, None], -torch.inf)
    # Select physical parent/current only in the source tensor passed to the loss.
    j, k = edges[:, None], edges[None]
    issued_mu = issued_source[-1]
    issued_at_labels = issued_mu[:, j, k, j, k]
    return read(log_p), issued_at_labels, (issued_source, windows, prior, log_p, hard_current, read)


def hold_and_relative(chart):
    hold = chart[..., HOLD, :, :]
    relative = chart[..., NON_HOLD, :, :] - hold.unsqueeze(-3)
    return hold, relative


def pairing(prediction, truth):
    # Feature axis is last. The two physical histories are at axis one.
    gap = prediction.diff(dim=1).squeeze(1)
    target = truth.diff(dim=1).squeeze(1)
    axes = (0,) + tuple(range(2, gap.ndim - 2))
    cross = (gap * target).sum(-1).mean(dim=axes)
    target_power = target.square().sum(-1).mean(dim=axes)
    forecast_power = gap.square().sum(-1).mean(dim=axes)
    center = prediction.mean(1, keepdim=True).expand_as(prediction)
    return {
        "truth_gap_rms": target_power.sqrt().tolist(), "predicted_gap_rms": forecast_power.sqrt().tolist(),
        "signed_gain": (cross / target_power.clamp_min(1e-8)).tolist(),
        "correct_mse_reduction_vs_swap": cross.tolist(),
        "correct_mse_reduction_vs_pool": ((2 * cross - forecast_power) / 4).tolist(),
        "pair_gap_rmse": (gap - target).square().sum(-1).mean(dim=axes).sqrt().tolist(),
        "scene_physical_truth": truth.tolist(), "scene_predictions": prediction.tolist(),
        "scene_endpoint_mse": (prediction - truth).square().sum(-1).tolist(),
        "scene_swapped_endpoint_mse": (prediction.flip(1) - truth).square().sum(-1).tolist(),
        "scene_pooled_endpoint_mse": (center - truth).square().sum(-1).tolist(),
    }


@torch.no_grad()
def evaluate(forecast, panel, rows, config, *, detailed=False):
    prediction, source, (issued, windows, prior, post, known, read) = trace(forecast, panel, rows, config)
    target = panel["measured"][rows, ..., -1]
    j, k = torch.arange(2, device=source.device)[:, None], torch.arange(2, device=source.device)[None]
    source_prior = prior.unflatten(0, (-1, 2, 2)).softmax(-2)
    predicted_sensor = (source_prior * issued[-1]).sum(-2)[:, j, k, k]
    result = {
        "source": pairing(predicted_sensor[..., None, None], target[..., None, None]),
        "source_true_edge_offline": pairing(source[..., None, None], target[..., None, None]),
        "modes": {},
    }
    truth = hold_and_relative(panel["xy"][rows])
    arms = {"ordinary": prediction, "known_current": read(known)}
    if detailed:
        before = post.logsumexp(1, keepdim=True) + (prior - prior.logsumexp(1, keepdim=True))
        arms.update(omit_last=read(before), native_input_mean=read(known, location="input"),
                    expanded_native=read(known, location="expanded"))
        # Compare two physical arrivals against each same already-issued prediction bank.
        mu_gap = issued[-1].diff(dim=-2).squeeze(-2)[:, j, k, k]
        dy = (target[:, 1] - target[:, 0])[:, None, :]
        evidence = mu_gap * dy / config[-1].square()
        tv = (post.softmax(1) - prior.softmax(1)).abs().sum(1) / 2
        result["source_attribution"] = {
            "within_bank_mu_gap": mu_gap.tolist(),
            "log_odds_gap_across_two_measurements": evidence.tolist(),
            "positive_odds_response_fraction": (evidence > 0).float().mean(dim=(0, 1)).tolist(),
            "last_history_weight_tv": tv.tolist(),
        }
    for name, values in arms.items():
        hold, relative = hold_and_relative(chart_pose(values)[..., :2])
        result["modes"][name] = {"hold": pairing(hold, truth[0]), "relative": pairing(relative, truth[1])}
    return result


def pair_loss(predict, target, scale, gap_scale, sign):
    center = (predict.mean(dim=1) - target.mean(dim=1)) / scale
    gap = (predict.diff(dim=1) - sign * target.diff(dim=1)) / gap_scale
    return center.square().mean() + gap.square().mean()


@torch.no_grad()
def complete_observations(model):
    """Reproduce the same physical roots and retain their last three real RGB frames."""
    with np.load(PHYSICS / "physical.npz", allow_pickle=False) as raw:
        scenes = raw["scene_ids"]
        commands, common = raw["commands"], raw["common_modes"]
        roots, sensor, initial = raw["root_state"], raw["observed_x"], raw["initial_rgb"]
    windows = np.empty((*roots.shape[:-1], 3, 96, 96, 3), dtype=np.uint8)
    env = ReversiblePushT()
    try:
        for row, seed in enumerate(scenes):
            scene = make_scene(int(seed))
            for j, k in np.ndindex((2, 2)):
                np.testing.assert_array_equal(env.start(scene, int(common[0])), initial[row])
                frames = []
                for mode, command in zip([*map(int, common), j, k], commands):
                    env.set_regime(mode)
                    frames.append(env.execute(command))
                np.testing.assert_array_equal(env.get_obs(), roots[row, j, k])
                np.testing.assert_allclose(image_x(np.stack(frames)), sensor[row, j, k], rtol=0, atol=1e-8)
                windows[row, j, k] = frames[-3:]
    finally:
        env.close()
    return encode_rgb(model, windows)


@torch.no_grad()
def complete_comparison(forecast, panel, config, windows):
    _, weight, bias, _, _ = config
    histories = windows.flatten(0, 2)
    batch = len(histories)
    past = panel["past"][-2:][None].expand(batch, -1, -1)
    plans = panel["plans"][None].expand(batch, -1, -1, -1)
    issued = forecast(histories, past, plans)
    index = torch.arange(batch, device=histories.device)
    physical = index % 2
    chart = chart_at_latent(issued[index, :, physical][:, :, HORIZONS], weight, bias)
    predicted = chart_pose(chart).unflatten(0, (-1, 2, 2))[..., :2]
    truth = hold_and_relative(panel["xy"])
    hold, relative = hold_and_relative(predicted)
    return {"hold": pairing(hold, truth[0]), "relative": pairing(relative, truth[1])}


def run(device, *, diagnose, full_view=False):
    torch.manual_seed(SEED)
    model, adapter, gate, hold, *_ = substep.load_core(device)
    fitted = torch.load(FITTED / "forecast_only.pt", map_location=device, weights_only=True)
    adapter.load_state_dict(fitted["adapter"])
    original_head = copy.deepcopy(model.pred_proj.state_dict())
    axis, bias = (fitted["chart"][key] for key in ("weight", "bias"))
    calibration = json.loads((SCALAR / "stage.json").read_text(encoding="utf-8"))
    transition = hold.new_tensor([[1 - SWITCH, SWITCH], [SWITCH, 1 - SWITCH]])
    config = (transition, axis, bias, hold.new_tensor(calibration["sensor_drift"]), hold.new_tensor(calibration["sensor_std"]))
    panel = load_panel(model, hold)
    if panel["ids"] != list(range(56000, 56016)):
        raise ValueError("Expected the predeclared crossed 16-scene panel")

    def forecast(windows, past, action, ancestry=None):
        return conditioned_rollout(model, windows, past, action, branch_offsets(adapter, len(windows)),
                                   first_effect=gate, ancestry=ancestry)

    if full_view:
        windows = complete_observations(model)

        def unconditioned(windows, past, action):
            offsets = torch.zeros_like(branch_offsets(adapter, len(windows)))
            return conditioned_rollout(model, windows, past, action, offsets)

        result = {"unconditioned": complete_comparison(unconditioned, panel, config, windows),
                  "frozen": complete_comparison(forecast, panel, config, windows)}
        trained = torch.load(OUTPUT / "correct.pt", map_location=device, weights_only=True)
        adapter.load_state_dict(trained["adapter"])
        model.pred_proj.load_state_dict(trained["pred_proj"])
        result["paired_fit"] = complete_comparison(forecast, panel, config, windows)
        return result

    if diagnose:
        results = {"frozen": evaluate(forecast, panel, slice(None), config, detailed=True)}
        old = torch.load(OLD_FIT / "joint.pt", map_location=device, weights_only=True)
        adapter.load_state_dict(old["adapter"])
        model.pred_proj.load_state_dict(old["pred_proj"])
        results["old_fit"] = evaluate(forecast, panel, slice(None), config, detailed=True)
        return results

    hold_truth, rel_truth = hold_and_relative(panel["chart"])
    source_truth = panel["measured"][..., -1]
    scale = hold.new_tensor(json.loads((OLD_FIT / "stage.json").read_text(encoding="utf-8"))["chart_scale"])

    def target_scale(value, dims):
        return value[:12].diff(dim=1).square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1.0)

    scales = (target_scale(source_truth, (0, 1, 2)), target_scale(hold_truth, (0, 1, 2, 4)),
              target_scale(rel_truth, (0, 1, 2, 3, 5)))
    parameters = list(adapter.requires_grad_(True).parameters()) + list(model.pred_proj.requires_grad_(True).parameters())
    rng = np.random.default_rng(SEED)
    schedule = [rng.choice(12, 2, replace=False).tolist() for _ in range(UPDATES)]
    result = {"fit": panel["ids"][:12], "heldout": panel["ids"][12:], "arms": {}}
    for arm, sign in (("correct", 1), ("swapped", -1), ("shared", 0)):
        adapter.load_state_dict(fitted["adapter"])
        model.pred_proj.load_state_dict(original_head)
        optimizer = torch.optim.AdamW(parameters, lr=0.0005)
        started = time.perf_counter()
        history = []
        for step, rows in enumerate(schedule, 1):
            prediction, source, _ = trace(forecast, panel, rows, config)
            a, b = hold_and_relative(prediction)
            parts = (pair_loss(source, source_truth[rows], scale, scales[0], 1),
                     pair_loss(a, hold_truth[rows], scale, scales[1], sign),
                     pair_loss(b, rel_truth[rows], scale, scales[2], sign))
            optimizer.zero_grad(set_to_none=True)
            sum(parts).backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, error_if_nonfinite=True)
            optimizer.step()
            record = [step, *(x.item() for x in parts), norm.item()]
            history.append(record)
            if step == 1 or step % 12 == 0:
                print(arm, record, flush=True)
        elapsed = time.perf_counter() - started
        result["arms"][arm] = {"seconds": elapsed, "loss": history,
                               "fit": evaluate(forecast, panel, slice(0, 12), config),
                               "heldout": evaluate(forecast, panel, slice(12, None), config, detailed=True)}
        with (OUTPUT / f"{arm}.pt").open("xb") as stream:
            torch.save({"adapter": adapter.state_dict(), "pred_proj": model.pred_proj.state_dict()}, stream)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--diagnose", action="store_true")
    options.add_argument("--full-view", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / ("full_view.json" if args.full_view else "frozen.json" if args.diagnose else "fit.json")
    if path.exists():
        raise FileExistsError(path)
    result = run(torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 diagnose=args.diagnose, full_view=args.full_view)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(path, flush=True)
