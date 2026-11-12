"""Finite shared-boundary training and two-arrival replay on frozen PushT LeWM."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

from native_gate_bound_paired_data import BRANCHES, encode, load_paired
from native_gate_bound_pusht import ASSETS, GRID_COUNT, MIN_SLOPE, NORMALIZATION, capture, decode, physical_errors, tail
from query_feedback_learning import write_csv, write_json


UPDATES = 64
LR = .01
CALIBRATION = Path("runs/native_gate_bound_20261112/pusht_curve/calibration.npz")


def prepare(bank, calibration, device):
    def tensor(x):
        return torch.as_tensor(x, dtype=torch.float32, device=device)

    fitted = {name: tensor(value) for name, value in calibration.items()}
    case = {name: tensor(bank[name]) for name in ("h", "m", "native", "true_latent", "true_physical", "normal")}
    case["covector"] = case["normal"] @ fitted["w"].T
    case["s"] = (case["h"] * case["covector"][:, None]).sum(-1) + (case["normal"] @ fitted["intercept"])[:, None]
    case["physical_standard"] = (case["true_physical"] - fitted["mean"]) / fitted["scale"]
    return case, fitted


def normalized(direction):
    return direction / direction.abs().max()


def gate_geometry(case, direction):
    d = case["m"] * normalized(direction)
    beta = (d * case["covector"][:, None]).sum(-1)
    return d, beta


def minimum_gate_geometry(case):
    product = case["m"] * case["covector"][:, None]
    return case["m"] * product.sign(), product.abs().sum(-1)


def fact_grid(s, beta, cap, *, nonnegative=False):
    s, beta = s[:, :2], beta[:, :2]
    span = torch.where(beta.abs() >= MIN_SLOPE, cap * beta.abs(), 0.)
    low = (s - span).amax(-1)
    high = s.amax(-1)
    if nonnegative:
        low, high = low.clamp_min(0.), high.clamp_min(0.)
    lattice = torch.linspace(0., 1., GRID_COUNT, device=s.device)
    points = torch.lerp(low[:, None], high[:, None], lattice[None])
    points = torch.cat((points, s.amin(-1).clamp(min=low)[:, None]), dim=-1)
    # The largest finite knot is unregulated only for the facts, not a new command.
    return torch.cat((points.sort(dim=-1).values, points.new_full((len(s), 1), torch.inf)), dim=-1)


def bounded_coefficients(s, beta, bounds, cap):
    exposure = (s - bounds).clamp_min(0.)
    active = exposure > 0.
    stable = beta.abs() >= MIN_SLOPE
    alpha = -exposure / torch.where(stable, beta, 1.)
    valid = ~active | (stable & (alpha.abs() <= cap + 2e-6))
    apply = active & valid
    return torch.where(apply, alpha, 0.), valid, apply


def forecast(host, case, fitted, geom, bounds, slots):
    d, beta = geom
    alpha, valid, apply = bounded_coefficients(case["s"][:, slots, None], beta[:, slots, None],
                                               bounds[:, None], fitted["cap"])
    native = case["native"][:, slots, None].expand(*apply.shape, -1)
    if bool(apply.any()):
        hidden = (case["h"][:, slots, None] + d[:, slots, None] * alpha[..., None])[apply]
        output = native.clone()
        output[apply] = torch.cat([tail(host, chunk) for chunk in hidden.split(2048)], dim=0)
    else:
        output = native
    return output, valid, apply, alpha


@torch.no_grad()
def predict_relative_four(host, z, action, covector, bound, cap):
    """Modify the native last block once. The fourth request is the zero reference."""
    roots = len(bound)
    block = host.predictor.transformer.layers[-1]
    captured = {}

    def keep_mlp(_, args, output):
        captured["m"] = output[:, -1].reshape(roots, 4, -1)

    def apply_bound(_, args, output):
        hidden = output[:, -1].reshape(roots, 4, -1)
        s = ((hidden - hidden[:, -1:]) * covector[:, None]).sum(-1)
        d, beta = minimum_gate_geometry(dict(m=captured["m"], covector=covector))
        alpha, valid, applied = bounded_coefficients(s, beta, bound[:, None], cap)
        modified = torch.where(applied[..., None], hidden + d * alpha[..., None], hidden)
        captured.update(valid=valid, applied=applied)
        return torch.cat((output[:, :-1], modified.flatten(0, 1)[:, None]), dim=1)

    with ExitStack() as hooks:
        hooks.callback(block.mlp.register_forward_hook(keep_mlp).remove)
        hooks.callback(block.register_forward_hook(apply_bound).remove)
        output = host.predict(z, action)[:, -1].reshape(roots, 4, -1)
    return output, captured["valid"], captured["applied"]


def source_observation(case, fitted):
    """The writer receives only the two executed requests' already arrived encodings."""
    reached = case["true_latent"][:, :2] @ fitted["readout"][:, :2]
    return ((reached[:, 1] - reached[:, 0]) * case["normal"]).sum(-1)


def arrived_facts(case):
    return {**{name: case[name][:, :2] for name in ("h", "m", "native", "true_latent", "s")},
            "normal": case["normal"], "covector": case["covector"]}


def source_residual(case, fitted, prediction, observed):
    change = (prediction[:, 1] - prediction[:, 0]) @ fitted["readout"][:, :2]
    return (change * case["normal"][:, None]).sum(-1) - observed[:, None]


def query_centered(case, fitted, prediction):
    read = prediction @ fitted["readout"] + fitted["offset"]
    error = read - case["physical_standard"][:, 2:, None]
    return error - error.mean(1, keepdim=True)


def fit_direction(host, case, fitted):
    direction = fitted["u"].clone().requires_grad_()
    optimizer = torch.optim.Adam([direction], lr=LR)
    observed = source_observation(case, fitted)
    logs = []
    for step in range(1, UPDATES + 1):
        with torch.no_grad():
            geom = gate_geometry(case, direction)
            grid = fact_grid(case["s"], geom[1], fitted["cap"])
            predictions, valid, _, _ = forecast(host, case, fitted, geom, grid, slice(None))
            source = source_residual(case, fitted, predictions[:, :2], observed).square()
            query = query_centered(case, fitted, predictions[:, 2:]).square().mean((1, 3))
            scores = (source + query).masked_fill(~valid.all(1), torch.inf)
            # The largest minimizer requests the least native-coordinate correction.
            selected = scores.shape[1] - 1 - scores.flip(1).argmin(-1)
            bounds = grid.gather(1, selected[:, None])
        optimizer.zero_grad(set_to_none=True)
        geom = gate_geometry(case, direction)
        predictions, valid, active, _ = forecast(host, case, fitted, geom, bounds, slice(None))
        if not bool(valid.all()):
            raise RuntimeError("Offline chosen bound is outside the declared shared native domain")
        source_loss = source_residual(case, fitted, predictions[:, :2], observed).square().mean()
        query_loss = query_centered(case, fitted, predictions[:, 2:]).square().mean()
        loss = source_loss + query_loss
        if loss.requires_grad:
            loss.backward()
            grad = float(direction.grad.norm())
            if not np.isfinite(grad):
                raise RuntimeError("Nonfinite shared direction gradient")
            optimizer.step()
        else:
            grad = 0.
        logs.append(dict(step=step, source_mse=float(source_loss.detach()), query_mse=float(query_loss.detach()),
                         gradient_norm=grad, active_fit_roots=int(active.any(1).sum())))
    return normalized(direction.detach()), logs


@torch.no_grad()
def arrival_residuals(host, case, fitted, geom, grid):
    predicted, valid, _, _ = forecast(host, case, fitted, geom, grid, slice(None))
    residual = (case["true_latent"][:, :, None] - predicted) @ fitted["readout"][:, :2]
    return (residual * case["normal"][:, None, None]).sum(-1), valid


def select_boundary(grid, low, high, valid, tolerance):
    error = high - low
    compatible = (error <= tolerance) & valid
    unbounded = compatible[:, -1]
    finite = ~unbounded & compatible.any(-1)
    bound = grid.masked_fill(~compatible, -torch.inf).amax(-1)
    bound = torch.where(finite, bound, torch.inf)
    return dict(bound=bound, grid=grid, low=low, high=high, valid=valid, error=error, compatible=compatible,
                finite=finite, unbounded=unbounded, best_error=error.masked_fill(~valid, torch.inf).amin(-1),
                native_error=error[:, -1])


@torch.no_grad()
def form_boundary(host, case, fitted, geom, tolerance, *, nonnegative=False):
    if case["true_latent"].shape[1] != 2 or case["h"].shape[1] != 2:
        raise ValueError("The boundary writer accepts exactly two arrived facts")
    grid = fact_grid(case["s"], geom[1], fitted["cap"], nonnegative=nonnegative)
    residual, valid = arrival_residuals(host, case, fitted, geom, grid)
    return select_boundary(grid, residual.amin(1), residual.amax(1), valid.all(1), tolerance)


@torch.no_grad()
def extend_boundary(host, case, fitted, geom, state, tolerance):
    if case["true_latent"].shape[1] != 1 or case["h"].shape[1] != 1:
        raise ValueError("The boundary update accepts one newly arrived fact")
    residual, valid = arrival_residuals(host, case, fitted, geom, state["grid"])
    return select_boundary(state["grid"], torch.minimum(state["low"], residual[:, 0]),
                           torch.maximum(state["high"], residual[:, 0]), state["valid"] & valid[:, 0], tolerance)


@torch.no_grad()
def fixed_boundary(host, case, fitted, geom, *, nonnegative=False):
    s = case["s"][:, :2]
    low, high = (0., s.max().clamp_min(0.)) if nonnegative else (s.min(), s.max())
    grid = torch.linspace(low, high, GRID_COUNT, device=s.device)
    grid = torch.cat((grid, grid.new_full((1,), torch.inf)))
    prediction, _, _, _ = forecast(host, case, fitted, geom, grid[None].expand(len(s), -1), slice(2, None))
    score = query_centered(case, fitted, prediction).square().mean((0, 1, 3))
    best = len(grid) - 1 - score.flip(0).argmin()
    return grid[best], grid, score


def array(value):
    return value.detach().cpu().numpy()


@torch.no_grad()
def evaluate(host, bank, case, fitted, calibration, trained, fixed, tolerance, *, relative=False):
    forecasts = {"native": case["native"][:, 2:]}
    traces, query_stats = {}, {}
    facts = arrived_facts(case)
    geom = minimum_gate_geometry(case) if relative else gate_geometry(case, trained)
    writers = {"relative_facts": geom} if relative else {"initial_facts": gate_geometry(case, fitted["u"]),
                                                          "trained_facts": geom}
    for name, geometry in writers.items():
        factual_geometry = tuple(value[:, :2] for value in geometry)
        state = form_boundary(host, facts, fitted, factual_geometry, tolerance, nonnegative=relative)
        forecasts[name], valid, active, coeff = forecast(host, case, fitted, geometry, state["bound"][:, None], slice(2, None))
        forecasts[name] = forecasts[name][:, :, 0]
        traces.update({f"{name}_{k}": array(v) for k, v in state.items()})
        traces[f"{name}_valid"], traces[f"{name}_alpha"] = array(valid[:, :, 0]), array(coeff[:, :, 0])
        query_stats[name] = dict(finite_roots=int(state["finite"].sum()), unbounded_roots=int(state["unbounded"].sum()),
                                 empty_roots=int((~state["finite"] & ~state["unbounded"]).sum()),
                                 out_of_domain=int((~valid).sum()), applied=int(active.sum()))

    forecasts["fixed"], valid, active, coeff = forecast(host, case, fitted, geom, fixed.expand(len(case["s"]), 1), slice(2, None))
    forecasts["fixed"] = forecasts["fixed"][:, :, 0]
    query_stats["fixed"] = dict(out_of_domain=int((~valid).sum()), applied=int(active.sum()))
    traces["fixed_valid"], traces["fixed_alpha"] = array(valid[:, :, 0]), array(coeff[:, :, 0])

    grid = fact_grid(case["s"], geom[1], fitted["cap"], nonnegative=relative)
    predictions, valid, active, coeff = forecast(host, case, fitted, geom, grid, slice(2, None))
    centered = query_centered(case, fitted, predictions)[..., :2]
    error = (centered * case["normal"][:, None, None]).sum(-1).abs().mean(1)
    error = error.masked_fill(~valid.all(1), torch.inf)
    choice = error.shape[1] - 1 - error.flip(1).argmin(-1)
    pick = choice[:, None, None, None].expand(-1, 4, 1, predictions.shape[-1])
    forecasts["oracle_shared"] = predictions.gather(2, pick)[:, :, 0]
    traces["oracle_bound"] = array(grid.gather(1, choice[:, None])[:, 0])
    traces["oracle_alpha"] = array(coeff.gather(2, choice[:, None, None].expand(-1, 4, 1))[:, :, 0])
    query_stats["oracle_shared"] = dict(applied=int(active.gather(2, choice[:, None, None].expand(-1, 4, 1)).sum()))

    if relative:
        for prediction in forecasts.values():
            torch.testing.assert_close(prediction[:, -1], forecasts["native"][:, -1], atol=0, rtol=0)
    rows, outputs, summary, floor = score_queries(bank, calibration, forecasts, query_stats)
    traces.update(outputs)
    return rows, traces, summary, floor


def score_queries(bank, calibration, forecasts, query_stats):
    physical = bank["true_physical"][:, 2:]
    normal = bank["normal"][:, None]
    rows, traces, summary, native_metrics = [], {}, {}, None
    query_contacts = bank["contacts"][:, 2:]
    mixed = query_contacts[:, 1] != query_contacts[:, 2]
    for mode, predictions in forecasts.items():
        z = array(predictions)
        traces[f"{mode}_latent"] = z
        estimate = decode(z, calibration)
        error = estimate - physical
        centered = error - error.mean(1, keepdims=True)
        angle_delta = np.arctan2(estimate[..., 5], estimate[..., 4]) - np.arctan2(physical[..., 5], physical[..., 4])
        wrapped = np.arctan2(np.sin(angle_delta), np.cos(angle_delta))
        centered_angle = wrapped - wrapped.mean(1, keepdims=True)
        result = dict(**physical_errors(estimate, physical, normal),
                      centered_normal_abs=np.abs((centered[..., :2] * normal).sum(-1)),
                      centered_object=np.linalg.norm(centered[..., 2:4], axis=-1),
                      centered_angle=np.abs(centered_angle),
                      latent_mse=np.square(z - bank["true_latent"][:, 2:]).mean(-1))
        if native_metrics is None:
            native_metrics = result
        gains = (native_metrics["centered_normal_abs"] - result["centered_normal_abs"]).reshape(-1, 4, 4).mean((1, 2))
        mixed_delta = np.abs(((error[:, 1, :2] - error[:, 2, :2]) * normal[:, 0]).sum(-1))
        summary[mode] = dict(**{k: float(v.mean()) for k, v in result.items()},
                             improved_scenes=int((gains > 0).sum()), median_scene_gain=float(np.median(gains)),
                             equal_turn_mixed_contact_normal_error=float(mixed_delta[mixed].mean()) if mixed.any() else None,
                             **query_stats.get(mode, {}))
        for i, scene in enumerate(bank["scene_id"]):
            for j in range(4):
                rows.append(dict(mode=mode, scene=int(scene), world=int(bank["world_id"][i]), candidate=j,
                                 contact=int(query_contacts[i, j]), **{k: float(v[i, j]) for k, v in result.items()}))
    floor = physical_errors(decode(bank["true_latent"][:, 2:], calibration), physical, normal)
    return rows, traces, summary, {k: float(v.mean()) for k, v in floor.items()}


def load_host():
    sys.path.insert(0, str(ASSETS))
    from prepare_assets import load_model

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    host = load_model(device).eval().requires_grad_(False)
    stats = json.loads(NORMALIZATION.read_text())
    moments = [torch.tensor(stats[key], device=device, dtype=torch.float32) for key in ("mean", "std")]
    return host, moments, device


@torch.inference_mode()
def second_fact_zero_reference(host, moments, split, scene_ids, world_ids):
    start = 101000 if split == "train" else 107000
    rows = 4 * (scene_ids - start) + world_ids
    with np.load(BRANCHES / split / "branches.npz", allow_pickle=False) as bank:
        np.testing.assert_array_equal(scene_ids, bank["scene_ids"][rows])
        np.testing.assert_array_equal(world_ids, bank["world_ids"][rows])
        pixels = bank["context_rgb"][rows, 1:3]
        earlier = bank["past_actions"][rows, 1]
    history = encode(host, pixels)
    return capture(host, history, np.stack((earlier, np.zeros_like(earlier)), axis=1), moments)["h"]


def transferable_domain(s, span, relative):
    left = s - span
    fact_lower = left[:, :2].max(-1)
    if relative:
        fact_lower = np.maximum(fact_lower, 0.)
    fact_upper = s[:, :2].max(-1)
    query_lower = left[:, 2:].max(-1)
    query_upper = s[:, 2:].max(-1)
    joint = np.maximum(0., np.minimum(fact_upper, query_upper) - np.maximum(fact_lower, query_lower))
    individual = np.maximum(0., np.minimum(fact_upper[:, None], s[:, 2:]) -
                            np.maximum(fact_lower[:, None], left[:, 2:]))
    return fact_lower, fact_upper, query_lower, query_upper, joint, individual


@torch.inference_mode()
def reference_domain(args):
    for name in ("reference_domains.csv", "reference_domains.json", "reference_coordinates.npz"):
        if (args.output / name).exists():
            raise FileExistsError(args.output / name)
    with np.load(args.calibration, allow_pickle=False) as prior:
        calibration = {key: prior[key] for key in ("w", "intercept", "cap", "u")}
    with np.load(args.output / "curves_and_outputs.npz", allow_pickle=False) as saved:
        sources = {key: saved[key] for key in ("scene_id", "world_id", "h", "m", "normal", "trained_u")}
    host, moments, _ = load_host()
    h_reference = second_fact_zero_reference(host, moments, "test", sources["scene_id"], sources["world_id"])
    covector = sources["normal"] @ calibration["w"].T
    offset = sources["normal"] @ calibration["intercept"]
    absolute = (sources["h"] * covector[:, None]).sum(-1) + offset[:, None]
    reference = (h_reference * covector).sum(-1) + offset
    relative = absolute - np.stack((absolute[:, 0], reference, *([absolute[:, 5]] * 4)), axis=1)
    m_w = sources["m"] * covector[:, None]
    spans = {"initial": np.abs(m_w @ calibration["u"]), "trained": np.abs(m_w @ sources["trained_u"]),
             "any_direction_upper_bound": np.abs(m_w).sum(-1)}
    rows, summary = [], {}
    for name, s in (("absolute", absolute), ("relative_to_native_zero", relative)):
        for direction, slope in spans.items():
            span = float(calibration["cap"]) * np.where(slope >= MIN_SLOPE, slope, 0.)
            f_lo, f_hi, q_lo, q_hi, joint, individual = transferable_domain(s, span, name != "absolute")
            usable = joint > 1e-6
            summary[f"{name}_{direction}"] = dict(all_queries=int(usable.sum()),
                 at_least_one_query=int((individual > 1e-6).any(-1).sum()), individually_usable=int((individual > 1e-6).sum()),
                 positive_joint_width_median=float(np.median(joint[usable])) if usable.any() else None,
                 query_lower_minus_fact_upper_quantiles=np.quantile(q_lo-f_hi, [.1, .5, .9]).tolist())
            for i, scene in enumerate(sources["scene_id"]):
                rows.append(dict(coordinate=name, direction=direction, scene=int(scene), world=int(sources["world_id"][i]),
                                 factual_lower=float(f_lo[i]), factual_upper=float(f_hi[i]), query_lower=float(q_lo[i]),
                                 query_upper=float(q_hi[i]), all_query_width=float(joint[i]),
                                 individually_usable=int((individual[i] > 1e-6).sum())))
    write_csv(args.output / "reference_domains.csv", rows)
    write_json(args.output / "reference_domains.json", summary)
    np.savez_compressed(args.output / "reference_coordinates.npz", absolute=absolute, relative=relative,
                        reference_fact=h_reference, scene_id=sources["scene_id"], world_id=sources["world_id"])
    print(json.dumps(summary, indent=2), flush=True)


@torch.inference_mode()
def minimum_step_check(args):
    for name in ("minimum_step.json", "minimum_step.npz"):
        if (args.output / name).exists():
            raise FileExistsError(args.output / name)
    with np.load(args.calibration, allow_pickle=False) as prior:
        calibration = {key: prior[key] for key in ("w", "cap", "readout", "offset", "mean", "scale")}
    with np.load(args.output / "curves_and_outputs.npz", allow_pickle=False) as saved:
        source = {key: saved[key] for key in ("h", "m", "native", "normal", "scene_id", "world_id")}
    with np.load(args.output / "reference_coordinates.npz", allow_pickle=False) as coord:
        np.testing.assert_array_equal(coord["scene_id"], source["scene_id"])
        np.testing.assert_array_equal(coord["world_id"], source["world_id"])
        relative = coord["relative"]
    host, _, device = load_host()
    case = {k: torch.as_tensor(source[k], device=device, dtype=torch.float32) for k in ("h", "m", "native")}
    case["s"] = torch.as_tensor(relative, device=device, dtype=torch.float32)
    case["covector"] = torch.as_tensor(source["normal"] @ calibration["w"].T, device=device, dtype=torch.float32)
    geom = minimum_gate_geometry(case)
    spans = float(calibration["cap"]) * np.where(array(geom[1]) >= MIN_SLOPE, array(geom[1]), 0.)
    fact_lo, fact_hi, query_lo, query_hi, width, _ = transferable_domain(relative, spans, True)
    chosen = width > 1e-6
    midpoint = (np.maximum(fact_lo, query_lo) + np.minimum(fact_hi, query_hi)) / 2
    bound = torch.as_tensor(np.where(chosen, midpoint, np.inf), device=device, dtype=torch.float32)
    fitted = {"cap": torch.as_tensor(calibration["cap"], device=device, dtype=torch.float32)}
    pred, valid, active, alpha = forecast(host, case, fitted, geom, bound[:, None], slice(None))
    active, alpha, pred = active[..., 0], alpha[..., 0], pred[:, :, 0]
    if not bool(valid.all()) or bool(active[:, -1].any()):
        raise RuntimeError("The common geometric domain or zero-command invariant failed")
    actual_s = case["s"] + (geom[0] * alpha[..., None] * case["covector"][:, None]).sum(-1)
    torch.testing.assert_close(actual_s, torch.minimum(case["s"], bound[:, None]), atol=2e-6, rtol=0)
    torch.testing.assert_close(pred[~active], case["native"][~active], atol=0, rtol=0)
    prediction = array(pred)
    change = decode(prediction, calibration) - decode(source["native"], calibration)
    normal_motion = (change[:, 2:, :2] * source["normal"][:, None]).sum(-1)
    object_motion = np.linalg.norm(change[:, 2:, 2:4], axis=-1)
    active_query = array(active[:, 2:])
    summary = dict(selected_roots=int(chosen.sum()), applied_queries=int(active_query.sum()),
                   applied_facts=int(active[:, :2].sum()), maximum_gate_step=float(alpha.abs().max()),
                   max_hidden_coordinate_error=float((actual_s-torch.minimum(case["s"],bound[:,None])).abs().max()),
                   sign_matched=int((normal_motion[active_query] < 0).sum()),
                   predicted_normal_delta_quantiles=np.quantile(normal_motion[active_query],[.1,.5,.9]).tolist(),
                   predicted_object_delta_quantiles=np.quantile(object_motion[active_query],[.1,.5,.9]).tolist())
    write_json(args.output / "minimum_step.json", summary)
    np.savez_compressed(args.output / "minimum_step.npz", bound=array(bound), alpha=array(alpha), active=array(active),
                        latent=prediction, coordinate=array(actual_s), decoded_change=change,
                        scene_id=source["scene_id"], world_id=source["world_id"])
    print(json.dumps(summary, indent=2), flush=True)


def run(args):
    host, moments, device = load_host()
    with np.load(args.calibration, allow_pickle=False) as prior:
        calibration = dict(prior)
    started = perf_counter()
    training = load_paired(host, moments, "fit")
    fit_case, fitted = prepare(training, calibration, device)
    tolerance_data = load_paired(host, moments, "calibrate")
    tune_case, _ = prepare(tolerance_data, calibration, device)
    tolerance = torch.quantile(source_residual(tune_case, fitted, tune_case["native"][:, :, None],
                                               source_observation(tune_case, fitted)).abs().flatten(), .90)
    print(f"Fixed factual contrast tolerance: {float(tolerance):.6g} from {len(tolerance_data['scene_id'])} roots", flush=True)
    train_started = perf_counter()
    trained, logs = fit_direction(host, fit_case, fitted)
    fit_seconds = perf_counter() - train_started
    fixed, fixed_grid, fixed_loss = fixed_boundary(host, fit_case, fitted, gate_geometry(fit_case, trained))
    print(f"Completed {len(logs)} finite-fit updates in {fit_seconds:.2f}s", flush=True)
    evaluation = load_paired(host, moments, "evaluate")
    case, _ = prepare(evaluation, calibration, device)
    rows, traces, summary, floor = evaluate(host, evaluation, case, fitted, calibration, trained, fixed, tolerance)
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "train_steps.csv", logs)
    write_csv(args.output / "requests.csv", rows)
    write_json(args.output / "summary.json", dict(tolerance=float(tolerance), tolerance_roots=len(tolerance_data["scene_id"]),
                fixed_bound=float(fixed) if torch.isfinite(fixed) else "unbounded", train_scenes=len(np.unique(training["scene_id"])),
                evaluation_scenes=len(np.unique(evaluation["scene_id"])), candidates_per_root=4,
                fixed_fit_updates=len(logs), optimizer_steps_with_gradient=sum(log["gradient_norm"] > 0 for log in logs),
                fit_seconds=fit_seconds, elapsed_seconds=perf_counter()-started, readout_floor=floor, modes=summary))
    np.savez_compressed(args.output / "curves_and_outputs.npz", **traces, **evaluation, trained_u=array(trained),
                        fixed_grid=array(fixed_grid), fixed_train_loss=array(fixed_loss))
    print(json.dumps(summary, indent=2), flush=True)


def scene_comparisons(rows, *, method="relative_facts"):
    data = {}
    keys = ("centered_normal_abs", "object", "angle")
    rng = np.random.default_rng(20261112)
    for row in rows:
        data.setdefault(row["mode"], []).append([row[key] for key in keys])
    causal = np.asarray(data[method]).reshape(-1, 16, len(keys)).mean(1)
    resamples = rng.integers(0, len(causal), (4000, len(causal)))
    results = {}
    for reference in ("native", "fixed"):
        baseline = np.asarray(data[reference]).reshape(causal.shape[0], 16, len(keys)).mean(1)
        gain = baseline - causal
        intervals = np.quantile(gain[resamples].mean(1), [.025, .975], axis=0)
        results[reference] = {key: dict(mean=float(gain[:, i].mean()), median=float(np.median(gain[:, i])),
                                       lower=float(intervals[0, i]), upper=float(intervals[1, i]),
                                       improved=int((gain[:, i] > 0.).sum()), worsened=int((gain[:, i] < 0.).sum()))
                              for i, key in enumerate(keys)}
    return results


@torch.inference_mode()
def relative_replay(args):
    destination = args.output / "relative_replay"
    if destination.exists():
        raise FileExistsError(destination)
    with np.load(args.calibration, allow_pickle=False) as prior:
        calibration = dict(prior)
    with np.load(args.output / "curves_and_outputs.npz", allow_pickle=False) as prior:
        names = ("h", "m", "native", "true_latent", "true_physical", "normal", "scene_id", "world_id", "contacts")
        evaluation = {name: prior[name] for name in names}
    with np.load(args.output / "reference_coordinates.npz", allow_pickle=False) as coordinates:
        for key in ("scene_id", "world_id"):
            np.testing.assert_array_equal(coordinates[key], evaluation[key])
        relative = coordinates["relative"]
    tolerance = json.loads((args.output / "summary.json").read_text())["tolerance"]
    host, moments, device = load_host()
    fit_bank = load_paired(host, moments, "fit")
    fit_case, fitted = prepare(fit_bank, calibration, device)
    zero_h = second_fact_zero_reference(host, moments, "train", fit_bank["scene_id"], fit_bank["world_id"])
    zero_h = torch.as_tensor(zero_h, device=device)
    zero_s = (zero_h * fit_case["covector"]).sum(-1) + (fit_case["normal"] @ fitted["intercept"])
    s = fit_case["s"]
    fit_case["s"] = s - torch.cat((s[:, :1], zero_s[:, None], s[:, -1:].expand(-1, 4)), dim=1)
    fixed, fixed_grid, fixed_loss = fixed_boundary(host, fit_case, fitted, minimum_gate_geometry(fit_case), nonnegative=True)

    case, _ = prepare(evaluation, calibration, device)
    case["s"] = torch.as_tensor(relative, device=device, dtype=torch.float32)
    rows, traces, summary, floor = evaluate(host, evaluation, case, fitted, calibration, None, fixed, tolerance, relative=True)
    destination.mkdir(parents=True, exist_ok=False)
    write_csv(destination / "requests.csv", rows)
    write_json(destination / "summary.json", dict(tolerance=tolerance, fixed_bound=float(fixed) if torch.isfinite(fixed) else "unbounded",
                fit_roots=len(fit_bank["scene_id"]), evaluation_roots=len(evaluation["scene_id"]), modes=summary,
                paired_scene_gains=scene_comparisons(rows), readout_floor=floor))
    np.savez_compressed(destination / "facts_and_outputs.npz", **traces, s=relative, fixed_grid=array(fixed_grid),
                        fixed_fit_loss=array(fixed_loss), scene_id=evaluation["scene_id"], world_id=evaluation["world_id"])
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=CALIBRATION)
    parser.add_argument("--output", type=Path, default=Path("runs/native_gate_bound_20261112/pusht_paired"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reference-domain", action="store_true")
    mode.add_argument("--minimum-step-check", action="store_true")
    mode.add_argument("--relative-replay", action="store_true")
    args = parser.parse_args()
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        if args.reference_domain:
            reference_domain(args)
        elif args.minimum_step_check:
            minimum_step_check(args)
        elif args.relative_replay:
            relative_replay(args)
        else:
            run(args)
