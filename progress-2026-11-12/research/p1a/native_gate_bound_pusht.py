"""Calibrate and measure a single finite curve at PushT LeWM's final gate."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np
import torch

from query_feedback_learning import write_csv, write_json


WORKSPACE = Path(__file__).resolve().parents[2]
ARCHIVE = WORKSPACE / "JEPA_OIC_Codex_Handoff_2026-11-03/runs"
ASSETS = WORKSPACE / "research_sources_2026-10-22/lewm_assets"
NORMALIZATION = WORKSPACE / "oic_step1/runs/pusht_continuation_2026-10-31/action_normalization.json"
GRID_COUNT = 33
MIN_SLOPE = 1e-4


def physical_labels(states):
    block = states[..., 4:6]
    angle = states[..., 8]
    return np.concatenate((states[..., :2] - block, block, np.stack((np.cos(angle), np.sin(angle)), -1)), -1)


def panel(name, cache_name, limit):
    with np.load(ARCHIVE / f"{name}.npz", allow_pickle=False) as p, np.load(
        ARCHIVE / f"{cache_name}.npz", allow_pickle=False
    ) as cached:
        np.testing.assert_array_equal(p["scene_ids"], cached["scene_ids"])
        np.testing.assert_array_equal(p["query_steps"], [10, 15, 20, 30])
        np.testing.assert_array_equal(p["query_steps"], cached["query_steps"])
        z = cached["full_latents"][:limit]
        actions = p["executed_actions"][:limit]
        candidates = p["query_actions"][:limit, :, 0]
        truth = cached["future_latents"][:limit, :, 1, :, 0]
        factual_physical = p["physical_state"][:limit]
        fork = p["query_physical_state"][:limit, :, 1]
        np.testing.assert_array_equal(fork[..., 0, :], np.broadcast_to(factual_physical[:, :, None, 15], fork[..., 0, :].shape))
        if not (p["actual_step"][:limit, :, :15].all() and p["query_actual_step"][:limit, :, 1, :, :5].all()):
            raise ValueError("The fixed early factual and candidate blocks must all execute")
        contact = np.any(p["query_contacts"][:limit, :, 1, :, :5] > 0, axis=-1)
        factual_contact = p["contacts"][:limit, :, 5:15].reshape(limit, 4, 2, 5).any(-1)
        ids = p["scene_ids"][:limit]
    approach = actions[:, 1, 0]
    np.testing.assert_array_equal(actions[:, 1], np.broadcast_to(approach[:, None], actions[:, 1].shape))
    normal = approach / np.linalg.norm(approach, axis=-1, keepdims=True)
    return dict(ids=ids, z=z, action=actions, candidate=candidates, truth=truth, normal=normal,
                physical=physical_labels(fork[..., 5, :]), contact=contact, factual_contact=factual_contact,
                factual_physical=physical_labels(factual_physical[:, :, [10, 15]]))


def tail(model, h):
    transformer = model.predictor.transformer
    return model.pred_proj(transformer.output_proj(transformer.norm(h)))


@torch.no_grad()
def capture(model, states, actions, moments):
    device = next(model.parameters()).device
    z = torch.as_tensor(states, device=device, dtype=torch.float32)
    raw = torch.as_tensor(actions, device=device, dtype=torch.float32)
    normalized = ((raw - moments[0]) / moments[1]).flatten(-2)
    block = model.predictor.transformer.layers[-1]
    record = {}

    def keep(name, value):
        if name in record:
            raise RuntimeError(f"Repeated final block value: {name}")
        record[name] = value[:, -1].clone()

    with ExitStack() as stack:
        stack.callback(block.norm2.register_forward_pre_hook(lambda _, args: keep("x", args[0])).remove)
        stack.callback(block.adaLN_modulation.register_forward_hook(
            lambda _, args, out: keep("g", out.chunk(6, dim=-1)[-1])).remove)
        stack.callback(block.mlp.register_forward_hook(lambda _, args, out: keep("m", out)).remove)
        stack.callback(block.register_forward_hook(lambda _, args, out: keep("h", out)).remove)
        record["native"] = model.predict(z, model.action_encoder(normalized))[:, -1].clone()
    torch.testing.assert_close(record["x"] + record["g"] * record["m"], record["h"], atol=2e-5, rtol=0)
    torch.testing.assert_close(tail(model, record["h"]), record["native"], atol=2e-5, rtol=0)
    return {key: val.cpu().numpy() for key, val in record.items()}


def capture_panel(model, bank, moments):
    scenes, worlds = bank["z"].shape[:2]
    count = scenes * worlds
    z = bank["z"].reshape(count, 7, 192)
    a = np.repeat(bank["action"], worlds, axis=0)
    c = np.repeat(bank["candidate"], worlds, axis=0)
    near = capture(model, z[:, :2], a[:, :2], moments)
    far = capture(model, z[:, :3], a[:, :3], moments)
    first = np.repeat(a[:, None, 1:3], 4, axis=1)
    forks = capture(model, np.repeat(z[:, 1:4], 4, axis=0),
                    np.concatenate((first, c[:, :, None]), axis=2).reshape(-1, 3, 5, 2), moments)
    return {key: np.concatenate((near[key][:, None], far[key][:, None], forks[key].reshape(count, 4, 192)), axis=1)
            for key in near}


def ridge(x, y, *, bias):
    x, y = x.astype(np.float64), y.astype(np.float64)
    offset = x.mean(0) if bias else 0
    y_offset = y.mean(0) if bias else 0
    centered = x - offset
    gram = centered.T @ centered
    reg = 0.01 * np.trace(gram) / len(gram)
    weight = np.linalg.solve(gram + reg * np.eye(len(gram)), centered.T @ (y - y_offset))
    return weight, y_offset - offset @ weight if bias else np.zeros(y.shape[1])


def truth(bank):
    count = len(bank["ids"]) * 4
    latent = np.concatenate((bank["z"][:, :, 2:4], bank["truth"]), axis=2).reshape(count, 6, 192)
    phys = np.concatenate((bank["factual_physical"], bank["physical"]), axis=2).reshape(count, 6, 6)
    return latent, phys


def fit(model, bank, samples):
    target, measured = truth(bank)
    target, measured = target.reshape(-1, 192), measured.reshape(-1, 6)
    mean = measured.mean(0)
    group_scales = [np.sqrt(np.mean(measured[:, i:i+2].var(0))) for i in (0, 2, 4)]
    scale = np.repeat(group_scales, 2)
    readout, offset = ridge(target, (measured - mean) / scale, bias=True)

    h = samples["h"].reshape(-1, 192)
    native = samples["native"].reshape(-1, 192)
    centered = h - h.mean(axis=-1, keepdims=True)
    w, intercept = ridge(centered, native @ readout[:, :2] + offset[:2], bias=True)
    w -= w.mean(axis=0)
    normal = np.repeat(bank["normal"], 4 * 6, axis=0)
    tangent = np.stack((-normal[:, 1], normal[:, 0]), axis=-1)
    device = next(model.parameters()).device
    h_t = torch.as_tensor(h, device=device).requires_grad_()
    with torch.enable_grad():
        decoded = tail(model, h_t) @ h_t.new_tensor(readout)
        jac = np.stack([torch.autograd.grad(decoded[:, i].sum(), h_t, retain_graph=i < 5)[0].cpu().numpy()
                        for i in range(6)], axis=1)
    rows = np.concatenate(((normal @ w.T)[:, None], np.einsum("bi,bid->bd", normal, jac[:, :2])[:, None],
                           np.einsum("bi,bid->bd", tangent, jac[:, :2])[:, None], jac[:, 2:]), axis=1)
    design = rows * samples["m"].reshape(-1, 1, 192)
    desired = np.broadcast_to([1., 1., 0., 0., 0., 0., 0.], design.shape[:2])
    u, _ = ridge(design.reshape(-1, 192), desired.reshape(-1, 1), bias=False)
    u = u[:, 0] / np.max(np.abs(u))
    cap = float(np.sqrt(np.square(samples["g"]).mean()))
    fact_residual = np.sqrt(np.square(samples["native"][:, :2] - target.reshape(-1, 6, 192)[:, :2]).mean(-1))
    tolerance = float(np.quantile(fact_residual[~bank["factual_contact"].reshape(-1, 2)], .9))
    return dict(readout=readout, offset=offset, mean=mean, scale=scale, w=w, intercept=intercept, u=u,
                cap=np.array(cap), tolerance=np.array(tolerance))


@torch.no_grad()
def curves(model, bank, samples, calibrated):
    device = next(model.parameters()).device
    normal = np.repeat(bank["normal"], 4, axis=0)
    covector = normal @ calibrated["w"].T
    s = (samples["h"] * covector[:, None]).sum(-1) + (normal @ calibrated["intercept"])[:, None]
    delta = samples["m"] * calibrated["u"]
    slope = (delta * covector[:, None]).sum(-1)
    alpha = -np.sign(slope[..., None]) * calibrated["cap"] * np.linspace(0., 1., GRID_COUNT)
    alpha[np.abs(slope) < MIN_SLOPE] = 0.
    coordinate = s[..., None] + slope[..., None] * alpha
    hidden = samples["h"].reshape(-1, 192)
    direction = delta.reshape(-1, 192)
    flat_alpha = alpha.reshape(-1, GRID_COUNT)
    forecast = []
    for start in range(0, len(hidden), 128):
        span = slice(start, start + 128)
        base = torch.as_tensor(hidden[span], device=device)
        d = base.new_tensor(direction[span])
        a = base.new_tensor(flat_alpha[span])
        modified = base[:, None] + d[:, None] * a[..., None]
        forecast.append(tail(model, modified.flatten(0, 1)).reshape(-1, GRID_COUNT, 192).cpu().numpy())
    latent = np.concatenate(forecast).reshape(*s.shape, GRID_COUNT, 192)
    np.testing.assert_allclose(latent[:, :, 0], samples["native"], atol=2e-5, rtol=0)
    latent[:, :, 0] = samples["native"]
    return dict(latent=latent, s=s, slope=slope, alpha=alpha, coordinate=coordinate)


def decode(z, fitted):
    return (z @ fitted["readout"] + fitted["offset"]) * fitted["scale"] + fitted["mean"]


def physical_errors(estimate, target, normal):
    offset = estimate - target
    rel, obj = offset[..., :2], offset[..., 2:4]
    angular = np.arctan2(estimate[..., 5], estimate[..., 4]) - np.arctan2(target[..., 5], target[..., 4])
    return dict(normal_abs=np.abs((rel * normal).sum(-1)), pusher=np.linalg.norm(rel + obj, axis=-1),
                object=np.linalg.norm(obj, axis=-1), angle=np.abs(np.arctan2(np.sin(angular), np.cos(angular))))


def summarize(bank, traces, fitted):
    target, measured = truth(bank)
    error = np.square(traces["latent"] - target[:, :, None]).mean(-1)
    relative = np.repeat(bank["normal"], 4, axis=0)[:, None]
    curve_phys = decode(traces["latent"], fitted)
    components = physical_errors(curve_phys, measured[:, :, None], relative[:, :, None])
    best_latent = error.argmin(-1)
    best_normal = components["normal_abs"].argmin(-1)
    rows, per_mode = [], {}
    for mode, slots in dict(native=np.zeros_like(best_latent), oracle_latent=best_latent, oracle_normal=best_normal).items():
        ii, jj = np.indices(slots.shape)
        measured_error = {key: arr[ii, jj, slots] for key, arr in components.items()}
        chosen_error = error[ii, jj, slots]
        rel = (curve_phys[ii, jj, slots, :2] - measured[..., :2])[:, 2:]
        object_delta = (curve_phys[ii, jj, slots, 2:4] - measured[..., 2:4])[:, 2:]
        centered = rel - rel.mean(1, keepdims=True)
        centered_obj = object_delta - object_delta.mean(1, keepdims=True)
        center_normal = np.abs((centered * relative).sum(-1))
        center_object = np.linalg.norm(centered_obj, axis=-1)
        metrics = dict(latent_mse=chosen_error, **measured_error)
        per_mode[mode] = {
            **{key: float(value[:, 2:].mean()) for key, value in metrics.items()},
            "centered_normal_abs": float(center_normal.mean()), "centered_object": float(center_object.mean()),
            "moves": int((slots[:, 2:] > 0).sum()), "at_limit": int((slots[:, 2:] == GRID_COUNT - 1).sum()),
        }
        for k, scene in enumerate(np.repeat(bank["ids"], 4)):
            for j in range(6):
                rows.append(dict(mode=mode, scene=int(scene), world=k % 4, slot=j,
                                 alpha=float(traces["alpha"][k, j, slots[k, j]]),
                                 slope=float(traces["slope"][k, j]),
                                 coordinate=float(traces["coordinate"][k, j, slots[k, j]]),
                                 **{name: float(value[k, j]) for name, value in metrics.items()}))
    floor = physical_errors(decode(target, fitted), measured, relative)
    info = {
        "curve": per_mode,
        "candidate_requests": int(error[:, 2:].shape[0] * 4),
        "slope_invalid": int((np.abs(traces["slope"][:, 2:]) < MIN_SLOPE).sum()),
        "normal_tail_secant_quantiles": np.quantile(
            (((curve_phys[:, 2:, -1, :2] - curve_phys[:, 2:, 0, :2]) * relative).sum(-1)
             / np.minimum(traces["coordinate"][:, 2:, -1] - traces["coordinate"][:, 2:, 0], -1e-12))
            [np.abs(traces["slope"][:, 2:]) >= MIN_SLOPE], [.1, .5, .9]).tolist(),
        "readout_floor": {name: float(value[:, 2:].mean()) for name, value in floor.items()},
    }
    return rows, info, error


def factual_envelope(coordinates, rms):
    """Return the finite knots of the worst residual of both arrived facts."""
    axes = [np.unique(x, return_index=True) for x in coordinates]
    grids = [(x, y[indices]) for (x, indices), y in zip(axes, rms, strict=True)]
    lower = max(x[0] for x, _ in grids)
    knots = np.unique(np.concatenate(([lower], *(x[x > lower] for x, _ in grids))))

    def values(points):
        return np.stack([np.interp(points, x, y) for x, y in grids])

    delta = np.diff(values(knots), axis=0)[0]
    for_crossing = delta[:-1] * delta[1:] < 0
    left, right = knots[:-1][for_crossing], knots[1:][for_crossing]
    start, end = delta[:-1][for_crossing], delta[1:][for_crossing]
    crossings = left + (right - left) * start / (start - end)
    knots = np.sort(np.r_[knots, crossings])
    return knots, values(knots).max(axis=0)


def audit_arrivals(output):
    """Use only the first two actual arrivals, independent of the candidate labels."""
    with np.load(output / "calibration.npz", allow_pickle=False) as fit_values:
        tolerance = float(fit_values["tolerance"])
    with np.load(output / "curve.npz", allow_pickle=False) as trace:
        coordinates = trace["coordinate"][:, :2]
        latent = trace["latent"][:, :2]
        actual = trace["true_latent"][:, :2]
        rms = np.sqrt(np.square(latent - actual[:, :, None]).mean(axis=-1))
        ids = np.repeat(trace["scene_id"], 4)
        contact = trace["factual_contact"].reshape(-1, 2)
    rows = []
    for index, scene in enumerate(ids):
        before, observed = rms[index, :, 0], rms[index].min(axis=-1)
        knots, errors = factual_envelope(coordinates[index], rms[index])
        minimum, unlimited = float(errors.min()), float(errors[-1])
        if unlimited <= tolerance:
            status = "unbounded"
            upper_bound = None
        elif minimum > tolerance:
            status = "no_common_bound" if np.all(observed <= tolerance) else "curve_mismatch"
            upper_bound = None
        else:
            status = "finite"
            j = np.flatnonzero(errors <= tolerance)[-1]
            upper_bound = float(knots[j] + (knots[j + 1] - knots[j]) *
                                (tolerance - errors[j]) / (errors[j + 1] - errors[j]))
        rows.append(dict(scene=int(scene), world=index % 4,
                         contact_5_10=bool(contact[index, 0]), contact_10_15=bool(contact[index, 1]),
                         native_5_10=float(before[0]), native_10_15=float(before[1]),
                         best_curve_5_10=float(observed[0]), best_curve_10_15=float(observed[1]),
                         common_minimum_rms=minimum, unbounded_rms=unlimited,
                         finite_support_tolerance_width=unlimited - minimum, status=status, upper_bound=upper_bound))
    best = rms.min(axis=-1)
    widths = np.array([row["finite_support_tolerance_width"] for row in rows])
    statuses = ("unbounded", "no_common_bound", "curve_mismatch", "finite")
    audit = {
        "tolerance": tolerance,
        "actual_roots": len(rows),
        "single_arrival_coverage": (best <= tolerance).sum(axis=0).tolist(),
        "single_arrival_finite": ((best <= tolerance) & (rms[:, :, 0] > tolerance)).sum(axis=0).tolist(),
        "single_arrival_contact_counts": contact.sum(axis=0).tolist(),
        "minimum_curve_rms_median": np.median(best, axis=0).tolist(),
        "common_state_counts": {status: sum(row["status"] == status for row in rows) for status in statuses},
        "finite_support_tolerance_width_quantiles": np.quantile(widths, [0, .25, .5, .75, .9, 1.]).tolist(),
        "exact_positive_finite_support_windows_above_1e_7": int((widths > 1e-7).sum()),
    }
    for name in ("arrivals.csv", "arrivals.json"):
        if (output / name).exists():
            raise FileExistsError(output / name)
    write_csv(output / "arrivals.csv", rows)
    write_json(output / "arrivals.json", audit)
    print(json.dumps(audit, indent=2))


def run(output):
    sys.path.insert(0, str(ASSETS))
    from prepare_assets import load_model

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    host = load_model("cuda").eval().requires_grad_(False)
    norm = json.loads(NORMALIZATION.read_text())
    moments = [torch.tensor(norm[key], device="cuda", dtype=torch.float32) for key in ("mean", "std")]
    train = panel("train_11033026", "train_11033026_native96_latents", 24)
    fit_samples = capture_panel(host, train, moments)
    calibrated = fit(host, train, fit_samples)
    seen = panel("confirmation_11034026", "confirmation_11034026_native48_latents", 48)
    held_samples = capture_panel(host, seen, moments)
    trace = curves(host, seen, held_samples, calibrated)
    rows, summary, error = summarize(seen, trace, calibrated)
    summary.update(cap=float(calibrated["cap"]), tolerance=float(calibrated["tolerance"]))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "requests.csv", rows)
    write_json(output / "summary.json", summary)
    np.savez_compressed(output / "calibration.npz", **calibrated)
    np.savez_compressed(output / "curve.npz", **trace, latent_mse=error, scene_id=seen["ids"],
                        contact=seen["contact"], factual_contact=seen["factual_contact"],
                        normal=seen["normal"], true_latent=truth(seen)[0], true_physical=truth(seen)[1], **held_samples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/native_gate_bound_20261112/pusht_curve"))
    parser.add_argument("--audit-arrivals", action="store_true")
    args = parser.parse_args()
    if args.audit_arrivals:
        audit_arrivals(args.output)
    else:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            run(args.output)
