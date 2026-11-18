"""Read a single rigid-object chart from legal observations and native forecasts."""

import argparse
import json
import math

import numpy as np
import shapely
import torch
from torch.nn import functional as F

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import advance_log_support
from .native_full_depth_first_effect import rollout
from .native_full_depth_reversible import SWITCH, action_embeddings, block_windows, encode_rgb
from .native_full_depth_substep_candidates import candidate_metrics
from .native_response_mixture_pilot import fit_coordinates


ROOT = substep.OUTPUT.parent
PAIRS = ROOT / "native_task_geometry_pairs_20261115"
OUTPUT = ROOT / "native_full_depth_task_geometry_20261115"
HORIZONS = [0, 1, 3]
# Area weights 4:3, rectangle centroids 15 and 75, and the
# corresponding mean squared distances about the rectangle centroids.
CENTROID_Y = 285.0 / 7.0
RADIUS = math.sqrt((4 * (1275 + (15 - CENTROID_Y) ** 2) + 3 * (750 + (75 - CENTROID_Y) ** 2)) / 7)
OUTLINE = np.asarray([[-60, 0], [60, 0], [60, 30], [15, 30], [15, 120],
                      [-15, 120], [-15, 30], [-60, 30]], dtype=np.float64)
AREA = 6300


def pose_chart(pose):
    """Area-mean squared material-point displacement is squared chart distance."""
    x, y, theta = pose.unbind(dim=-1)
    sin, cos = theta.sin(), theta.cos()
    return torch.stack((x - CENTROID_Y * sin, y + CENTROID_Y * cos, RADIUS * cos, RADIUS * sin), dim=-1)


def chart_pose(chart):
    x, y, cos, sin = chart.unbind(dim=-1)
    angle = torch.atan2(sin, cos)
    return torch.stack((x + CENTROID_Y * angle.sin(), y - CENTROID_Y * angle.cos(), angle), dim=-1)


def chart_at_latent(latent, weight, bias):
    raw = latent @ weight + bias
    return torch.cat((raw[..., :2], RADIUS * F.normalize(raw[..., 2:], dim=-1)), dim=-1)


def polygons(pose):
    value = np.asarray(pose, dtype=np.float64)
    angle = value[..., 2, None]
    cos, sin = np.cos(angle), np.sin(angle)
    u, v = OUTLINE.T
    x = value[..., 0, None] + cos * u - sin * v
    y = value[..., 1, None] + sin * u + cos * v
    return shapely.polygons(np.stack((x, y), axis=-1))


def coverage(pose, goal):
    """Use exact fractional intersection, with broadcastable pose and goal arrays."""
    return shapely.area(shapely.intersection(polygons(pose), polygons(goal))) / AREA


def shape_cost(chart, goal):
    pose = chart_pose(chart).cpu().numpy()
    goal_pose = chart_pose(goal).cpu().numpy()
    return torch.as_tensor(1 - coverage(pose, goal_pose), device=chart.device)


def scene_support(factors):
    """Consume separately issued likelihoods, one real block and phase at a time."""
    batch = factors.size(0)
    log_support = factors.new_full((batch, 2), -math.log(2))
    priors, history = [], []
    for tick in factors.unbind(dim=1):
        log_support = advance_log_support(log_support, SWITCH)
        priors.append(log_support.exp()[:, 1])
        arrivals = []
        for arrival in tick.unbind(dim=-1):
            log_support = (log_support + arrival).log_softmax(-1)
            arrivals.append(log_support.exp()[:, 1])
        history.append(torch.stack(arrivals, dim=-1))
    return torch.stack(priors, dim=1), torch.stack(history, dim=1)


def sensor_scores(issued, measured, modes, mean, covariance, active, *, path_scale=None):
    factual = torch.stack([measured[:, tick, mode] for tick, mode in enumerate(modes)], dim=1)
    result = {}
    arms = [("endpoint", substep.endpoint_evidence, 1.0), ("path", substep.chronological_evidence, 1.0)]
    if path_scale is not None:
        arms.append(("path_temperature", substep.chronological_evidence, path_scale))
    for name, factor, scale in arms:
        left = scale * factor(issued, measured[:, :, 0], mean, covariance)
        right = scale * factor(issued, measured[:, :, 1], mean, covariance)
        middle, directed, nll = substep.nll_pairs(left, right)
        prior, posterior = scene_support(scale * factor(issued, factual, mean, covariance))
        result[name] = {
            "prior": prior, "posterior": posterior, "middle": middle, "directed": directed, "nll": nll,
            "active_nll": nll[active].mean(), "inactive_mean_evidence": middle[~active].abs().mean(),
        }
    return result


def source_predictions(model, adapter, gate, hold, prefix, encoded, raw_actions):
    embedded = action_embeddings(model, raw_actions)
    predictions = []
    for tick in range(raw_actions.size(1)):
        history, past = block_windows(encoded, embedded, tick, hold)
        predictions.append(torch.stack([
            substep.predict_step(model, adapter, gate, prefix, history, past, raw_actions[:, tick], step)
            for step in range(1, 6)
        ], dim=-2))
    return torch.stack(predictions, dim=1), embedded


def contrast(predicted, actual):
    target = actual.diff(dim=2).squeeze(dim=2)
    estimate = predicted.diff(dim=2).squeeze(dim=2)
    lhs, rhs = torch.triu_indices(target.size(2), target.size(2), offset=1, device=target.device)
    truth = target[:, :, lhs] - target[:, :, rhs]
    effect = estimate[:, :, lhs] - estimate[:, :, rhs]
    significant = truth.abs() > 1e-4
    agreement = (1 + (truth * effect).sign()) / 2
    best = predicted.argmin(dim=3)
    direction = (truth * effect).sum(dim=(0, 1, 2)) / truth.square().sum(dim=(0, 1, 2))
    return {"true_nonzero_pairs": significant.sum(dim=(0, 1, 2)),
            "true_contrast_rms": truth.square().mean(dim=(0, 1, 2)).sqrt(),
            "signed_fraction": direction,
            "sign_agreement": (agreement * significant).sum(dim=(0, 1, 2)) / significant.sum(dim=(0, 1, 2)),
            "different_best_candidate": (best[:, :, 0] != best[:, :, 1]).sum(dim=(0, 1)),
            "per_root_true_contrast_rms": truth.square().mean(dim=2).sqrt()}


def rank_arms(costs, supports, truth, geometric_xy, rgb_xy, physical_xy, true_xy):
    def score(cost, p, xy):
        p = torch.stack((1 - p, p), dim=-1)[:, :, :, None, None]
        outcome = (cost * p).sum(dim=2)
        return candidate_metrics(outcome, truth, (xy * p[..., None]).sum(dim=2), true_xy)

    arms = {
        "native_l2_xy_path": score(costs["native_l2"], supports["xy_path"], geometric_xy),
        "shape_xy_path": score(costs["shape"], supports["xy_path"], geometric_xy),
        "shape_uniform": score(costs["shape"], supports["uniform"], geometric_xy),
        "shape_chart_endpoint": score(costs["shape"], supports["chart_endpoint"], geometric_xy),
        "shape_chart_path": score(costs["shape"], supports["chart_path"], geometric_xy),
        "shape_skip_current": score(costs["shape"], supports["skip"], geometric_xy),
        "shape_known_mode": score(costs["shape"], supports["known_mode"], geometric_xy),
        "actual_rgb_native_l2": score(costs["actual_rgb_l2"], supports["known_mode"], rgb_xy),
        "actual_rgb_shape_known_mode": score(costs["actual_rgb_shape"], supports["known_mode"], rgb_xy),
        "actual_rgb_shape_chart_path": score(costs["actual_rgb_shape"], supports["chart_path"], rgb_xy),
        "actual_geometry_uniform": score(costs["actual_shape"], supports["uniform"], physical_xy),
        "actual_geometry_chart_path": score(costs["actual_shape"], supports["chart_path"], physical_xy),
        "actual_geometry_known_mode": score(costs["actual_shape"], supports["known_mode"], physical_xy),
    }
    if "chart_temperature" in supports:
        arms["shape_chart_temperature"] = score(costs["shape"], supports["chart_temperature"], geometric_xy)
    return arms


def block_reversals(sensors, modes, active, ticks):
    chosen = (sensors["posterior"][:, ticks, -1] >= 0.5)
    truth = torch.tensor(modes, device=chosen.device)[ticks]
    informative = active[:, ticks]
    correct = chosen == truth
    return {"informative_by_reverse": informative.sum(0), "correct_by_reverse": (correct & informative).sum(0),
            "both_informative": informative.all(-1).sum(),
            "both_informative_and_correct": (correct & informative).all(-1).sum()}


def json_array(value):
    if isinstance(value, torch.Tensor):
        return value.tolist()
    raise TypeError(f"Unexpected result type: {type(value)}")


@torch.inference_mode()
def run(device, output):
    from .native_reversible_pusht import make_scene

    model, adapter, gate, hold, w, b, _ = substep.load_core(device)
    prefix = substep.NativeActionPrefixes(hold.numel()).to(device).eval()
    old = torch.load(substep.OUTPUT / "prefix.pt", map_location=device, weights_only=True)
    prefix.load_state_dict(old["prefix"])
    print("Reading actual fitting latents", flush=True)
    with np.load(substep.PAIRS / "fit.npz", allow_pickle=False) as file:
        z = encode_rgb(model, file["path_rgb"]).flatten(0, -2)
        p = torch.tensor(file["path_state"][..., 2:5].copy(), device=device).flatten(0, -2)
        goal = encode_rgb(model, file["goal_rgb"])
        goal_pose = torch.tensor(np.stack([make_scene(int(seed)).goal for seed in file["scene_id"]]),
                                 dtype=p.dtype, device=device)
        fit_ids = file["scene_id"].tolist()
    weight, bias = fit_coordinates(torch.cat((z, goal)), pose_chart(torch.cat((p, goal_pose))))

    def readout(z):
        return chart_at_latent(z, weight, bias)

    print("Calibrating chronological shape observations", flush=True)
    calibration = substep.panel(model, "calibration")
    drift, covariance, residual_count = substep.calibrate_residuals(
        model, adapter, gate, hold, prefix, calibration, lambda observed, predicted: readout(observed) - readout(predicted),
    )
    result = evaluate(model, adapter, gate, hold, prefix, readout, drift, covariance, w, b, old, PAIRS)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.json").open("x", encoding="utf-8") as file:
        json.dump({"fit_scenes": fit_ids, "calibration_scenes": calibration["scene"],
                   "calibration_residuals": residual_count, **result}, file, default=json_array, indent=2, allow_nan=False)
    torch.save({"weight": weight.cpu(), "bias": bias.cpu(), "drift": drift.cpu(),
                "covariance": covariance.cpu()}, output / "coordinates.pt")


def paired_chart_error(predicted, target, mode):
    error = predicted - target
    common = error.mean(dim=mode).square().sum(dim=-1)
    contrast = (0.5 * error.diff(dim=mode).squeeze(mode)).square().sum(dim=-1)
    return common, contrast


@torch.inference_mode()
def evaluate(model, adapter, gate, hold, prefix, readout, drift, covariance, w, b, old, pairs, *, path_scale=None):
    device = hold.device
    with np.load(pairs / "fresh_switches.npz", allow_pickle=False) as data:
        ids = data["scene_id"].tolist()
        modes = data["mode"].tolist()
        phases = data["phase"].tolist()
        actual = encode_rgb(model, data["rgb"])
        paired = encode_rgb(model, data["path_rgb"])
        actions = torch.as_tensor(data["actions"], dtype=torch.float32, device=device)
        physical_source = torch.as_tensor(data["path_state"][..., 2:5], device=device)
    print("Scoring all actual arrivals and paired future forecasts", flush=True)
    issued, embedded = source_predictions(model, adapter, gate, hold, prefix, actual, actions)
    active = physical_source[..., :2].diff(dim=2).squeeze(2).norm(dim=-1).amax(dim=-1) >= 1
    chart_source = readout(paired)
    modes_path = readout(issued)
    s_chart = sensor_scores(modes_path, chart_source, modes, drift, covariance, active, path_scale=path_scale)
    s_xy = sensor_scores(issued @ w + b, paired @ w + b, modes, old["drift"], old["covariance"], active)
    with np.load(pairs / "post_switch_candidates.npz", allow_pickle=False) as data:
        tick = [phases.index(x) for x in data["phase"].tolist()]
        if ids != data["scene_id"].tolist() or [modes[t] for t in tick] != data["mode"].tolist():
            raise ValueError("Candidate panel is not aligned with the actual two-switch history")
        if data["future_mode"].tolist() != [0, 1]:
            raise ValueError("Candidate modes must be in the model's native order")
        plans = action_embeddings(model, torch.as_tensor(data["actions"], dtype=torch.float32, device=device))
        targets = encode_rgb(model, data["goal_rgb"])
        goals_physical = data["goal_pose"]
        true_pose = data["state"][..., HORIZONS, 2:5]
        true_coverage = data["coverage"][..., HORIZONS]
        future_rgb = encode_rgb(model, data["rgb"][..., HORIZONS, :, :, :])
    recalculated = coverage(true_pose, goals_physical[:, None, None, None, None])
    geometry_error = float(np.max(np.abs(recalculated - true_coverage)))
    if geometry_error > 1e-5:
        raise ValueError(f"Coverage does not match the native shape evaluator: {geometry_error}")
    forecast_rows, reference_rows = [], []
    for t in tick:
        root, past = block_windows(actual, embedded, t + 1, hold)
        route = {}
        pred = rollout(model, adapter, gate, root, past, plans[None].expand(len(ids), -1, -1, -1), compact=True, trace=route)
        forecast_rows.append(pred[..., HORIZONS, :].transpose(1, 2))
        reference_rows.append(route["rows"])
    native = torch.stack(forecast_rows, dim=1)
    future, rgb_chart, goal_chart = readout(native), readout(future_rgb), readout(targets)
    expanded_goal = goal_chart[:, None, None, None, None]
    target_latent = targets[:, None, None, None, None]
    actual_coverage = torch.as_tensor(true_coverage, dtype=torch.double, device=device)
    physical = torch.as_tensor(true_pose, device=device)
    full_xy = physical[..., :2]
    task = {
        "native_l2": (native - target_latent).square().sum(dim=-1).double(),
        "shape": shape_cost(future, expanded_goal),
        "actual_rgb_l2": (future_rgb - target_latent).square().sum(dim=-1).double(),
        "actual_rgb_shape": shape_cost(rgb_chart, expanded_goal),
        "actual_shape": 1 - actual_coverage,
    }
    chart_path = s_chart["path"]["posterior"][:, tick, -1]
    supports = {"xy_path": s_xy["path"]["posterior"][:, tick, -1], "chart_path": chart_path,
                "chart_endpoint": s_chart["endpoint"]["posterior"][:, tick, -1],
                "skip": s_chart["path"]["prior"][:, tick], "uniform": torch.full_like(chart_path, 0.5),
                "known_mode": chart_path.new_tensor([modes[t] for t in tick]).expand_as(chart_path)}
    if path_scale is not None:
        supports["chart_temperature"] = s_chart["path_temperature"]["posterior"][:, tick, -1]
    factual_coverage = torch.stack([actual_coverage[:, row, modes[t]] for row, t in enumerate(tick)], dim=1)
    factual_xy = torch.stack([full_xy[:, row, modes[t]] for row, t in enumerate(tick)], dim=1)
    arms = rank_arms(task, supports, factual_coverage, chart_pose(future)[..., :2],
                     chart_pose(rgb_chart)[..., :2], full_xy, factual_xy)

    def by_source(group):
        return {name: {**values, **block_reversals(values, modes, active, tick)}
                for name, values in group.items()}

    def rmse(estimate, target, dims):
        return (estimate - target).square().sum(-1).mean(dim=dims).sqrt()

    genuine = pose_chart(physical)
    actual_source = pose_chart(physical_source)
    source_common, source_gap = paired_chart_error(modes_path, actual_source, 2)
    future_common, future_gap = paired_chart_error(future, genuine, 2)
    return {"scenes": ids, "phases": phases, "modes": modes,
                   "raw_horizons": [5, 10, 20], "true_geometry_max_coverage_error": geometry_error,
                   "active": active, "reference_rows": reference_rows, "support": supports,
                   "source": {"old_xy": by_source(s_xy), "chart": by_source(s_chart)},
                   "coordinates": {
                       "source_actual_chart_rmse": rmse(chart_source, actual_source, (0, 1, 2)),
                       "source_native_chart_rmse": rmse(modes_path, actual_source, (0, 1, 2)),
                       "future_actual_chart_rmse": rmse(rgb_chart, genuine, (0, 1, 2, 3)),
                       "future_native_chart_rmse": rmse(future, genuine, (0, 1, 2, 3)),
                       "goal_chart_rmse": rmse(goal_chart, pose_chart(torch.as_tensor(goals_physical, device=device)), 0),
                       "source_midpoint_sq_per_scene": source_common.mean(dim=1),
                       "source_half_gap_sq_per_scene": source_gap.mean(dim=1),
                       "future_midpoint_sq_per_scene": future_common.mean(dim=(1, 2)),
                       "future_half_gap_sq_per_scene": future_gap.mean(dim=(1, 2)),
                   },
                   "truth_coverage": actual_coverage, "costs": task, "arms": arms,
                   "physical_contrast": {name: contrast(cost, task["actual_shape"]) for name, cost in task.items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=type(OUTPUT), default=OUTPUT)
    args = parser.parse_args()
    run(torch.device("cuda" if torch.cuda.is_available() else "cpu"), args.output)
