"""Test native FourRoom mode forecasts and causal full-RGB support updates."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .fullview_rooms_base import original_probe
from .native_full_depth_conditioning import (
    FirstEffectReadout, FullDepthConditioner, advance_log_support, conditioned_rollout, write_observation,
)
from .native_response_relations_fit import native_rollout
from .physical_topology_gate import VisibleMapCost, make_scene


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/native_topology_condition_20261115"
STOCK = ROOT / "runs/fullview_rooms_base_20261115/fourroom/summary.json"
TRAIN = range(2026111600, 2026111680)
CALIBRATE = range(2026111680, 2026111696)
HOLDOUT = range(2026111800, 2026111832)
BATCH, EPOCHS, SEED, SWITCH = 16, 40, 411115, .01
SCALE_FLOOR = 1e-3


class FourRoomHost(nn.Module):
    """Expose the checkpoint's existing projected modules through the native LeWM interface."""

    def __init__(self, level, moments):
        super().__init__()
        self.level = level
        self.register_buffer("action_mean", torch.as_tensor(moments["mean"], dtype=torch.float64).reshape(2))
        self.register_buffer("action_std", torch.as_tensor(moments["std"], dtype=torch.float64).reshape(2))

    @property
    def predictor(self):
        return self.level.predictor.predictor

    @property
    def pred_proj(self):
        return self.level.predictor.projector

    @property
    def encoder(self):
        return self.level.encoder

    @property
    def projector(self):
        return self.encoder.projector

    def encode(self, info):
        native = self.level.encode(info)
        native["emb"] = native["embed_0"]
        if "action_0" in native:
            native["act_emb"] = native["action_0"]
        return native

    def predict(self, history, embedded_actions):
        return self.level.predict(history, embedded_actions)


def load_host(device):
    assets = json.loads(STOCK.read_text(encoding="utf-8"))
    level = torch.load(assets["checkpoint"], map_location="cpu", weights_only=False).get_level(1)
    level.interpolate_pos_encoding = True
    if (not isinstance(level.action_encoder, nn.Identity) or getattr(level, "action_queue_size", 0) != 0
            or level.predictor.predictor.pos_embedding.shape != (1, 3, 192)
            or len(level.predictor.predictor.transformer.layers) != 4):
        raise ValueError("Expected the official flat FourRoom level and three-token predictor")
    moments = torch.load(assets["normalizer"], map_location="cpu", weights_only=False)["stats"]["action"]
    return FourRoomHost(level, moments).to(device).eval().requires_grad_(False)


def load_readout(device, run=None):
    if run is None:
        probe, mean, std = original_probe(ROOT / "research/phase00c/probe_heads.pt")
    else:
        state = torch.load(Path(run) / "readout_diagnostic_fit80/layout_readout.pt", map_location="cpu", weights_only=True)
        probe = nn.Sequential(nn.Linear(192, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 2))
        probe.load_state_dict(state["state_dict"], strict=True)
        probe.eval().requires_grad_(False)
        mean, std = state["mean"], state["std"]
    probe = probe.to(device)
    mean, std = (tensor.to(device=device, dtype=torch.float32) for tensor in (mean, std))

    def decode(z):
        return probe(z) * std + mean

    return decode


def action_embeddings(host, raw):
    """Normalize each primitive axis before five-control temporal flattening."""
    raw = torch.as_tensor(raw, device=host.action_mean.device)
    if raw.shape[-2:] != (5, 2):
        raise ValueError("Expected five primitive FourRoom actions per model block")
    values = ((raw - host.action_mean) / host.action_std).float().flatten(-2)
    leading = values.shape[:-1]
    sequences = values.reshape(-1, leading[-1] if leading else 1, 10)
    encoded = host.level.action_embed(host.level.action_encoder(sequences))
    return encoded.reshape(*leading, encoded.shape[-1])


def forecast(host, conditioner, gate, history, past, raw_plans, *, compact=False, trace=None, force=None):
    controls = torch.cat((past[:, None].expand(-1, raw_plans.size(1), -1, -1, -1), raw_plans), dim=2)
    embedded = action_embeddings(host, controls)
    history_actions, future = embedded[:, 0, :2], embedded[:, :, 2:]
    if conditioner is None:
        return native_rollout(host, history, history_actions, future).unsqueeze(2)
    code = history.new_tensor([[1., -1.], [1., 1.]])
    offsets = conditioner(code)[None].expand(len(history), -1, -1, -1)
    return conditioned_rollout(host, history, history_actions, future, offsets, first_effect=gate,
                               compact=compact, trace=trace, forced_exposure=force)


def load_trained(run, arm, device):
    host = load_host(device)
    if arm == "stock":
        return host, None, None
    if arm not in ("native", "conditioned"):
        raise ValueError(f"Unknown trained arm: {arm}")
    state = torch.load(Path(run) / f"{arm}.pt", map_location=device, weights_only=True)
    host.level.predictor.load_state_dict(state["predictor"], strict=True)
    host.level.action_embed.load_state_dict(state["action_embed"], strict=True)
    if arm == "native":
        return host, None, None
    conditioner = FullDepthConditioner(host, 2).to(device).eval()
    gate = FirstEffectReadout(host).to(device).eval()
    conditioner.load_state_dict(state["conditioner"], strict=True)
    gate.load_state_dict(state["gate"], strict=True)
    gate.forced_exposure = state.get("forced_exposure")
    return host, conditioner.requires_grad_(False), gate.requires_grad_(False)


def read_panel(path, seeds, device):
    with np.load(path, allow_pickle=False) as data:
        available = data["seed"]
        rows, = np.where(np.isin(available, list(seeds)))
        np.testing.assert_array_equal(available[rows], list(seeds))
        out = {key: torch.as_tensor(data[key][rows], device=device) for key in (
            "seed", "root_z", "root_action", "action", "future_z", "future_xy", "life_action", "life_z",
            "life_cf_z", "life_xy", "life_mode", "life_valid", "life_event", "life_informative",
        )}
    return out


def fan(panel):
    history, past, plan, z = (panel[key].flatten(0, 1) for key in ("root_z", "root_action", "action", "future_z"))
    y = panel["future_xy"].flatten(0, 1)[..., -1, :]
    relevance = (y[:, :, 0, 0] != y[:, :, 1, 0]).any(dim=-1)
    return history, past, plan, z, y, relevance


def sources(panel):
    rows, initial, time = panel["life_valid"].nonzero(as_tuple=True)
    keep = time >= 2
    rows, initial, time = rows[keep], initial[keep], time[keep]
    prev = time[:, None] + time.new_tensor([-2, -1, 0])
    history = panel["life_z"][rows[:, None], initial[:, None], prev]
    past = panel["life_action"][rows[:, None], initial[:, None], prev[:, :-1]]
    action = panel["life_action"][rows, initial, time][:, None, None]
    target = panel["life_cf_z"][rows, initial, time][:, None, :, None]
    label = panel["life_informative"][rows, initial, time]
    return (history, past, action, target, label), (rows, initial, time)


@torch.no_grad()
def check_native(host, panel):
    history, past, plan, _, _, _ = fan(panel)
    history, past, plan = history[:2], past[:2], plan[:2, :1]
    cond = FullDepthConditioner(host, 2).to(history.device).eval()
    gate = FirstEffectReadout(host).to(history.device).eval()
    embed = action_embeddings(host, torch.cat((past, plan[:, 0]), dim=1))
    window = history
    reference = []
    for h in range(plan.size(2)):
        new = host.level.predict(window, embed[:, h:h + 3])[:, -1]
        reference.append(new)
        window = torch.cat((window[:, 1:], new[:, None]), dim=1)
    expected = torch.stack(reference, dim=1)[:, None, None]
    stock = forecast(host, None, None, history, past, plan)
    torch.testing.assert_close(stock, expected, rtol=0, atol=0)
    trace = {}
    dormant = forecast(host, cond, gate, history, past, plan, compact=True, trace=trace, force=0.)
    torch.testing.assert_close(dormant, expected.expand_as(dormant), rtol=0, atol=0)
    if trace["rows"] != [(2, 2)] * plan.size(2):
        raise AssertionError("Forced common windows did not take one row per pair")
    cond.layers.weight.normal_(std=.1)
    active = forecast(host, cond, gate, history, past, plan, compact=True, force=1.)
    if not (active[:, :, 0] != active[:, :, 1]).any():
        raise AssertionError("Nonzero mode offsets did not reach the native suffix")
    return {"stock_and_zero_offset_parity_max_abs": 0., "forced_shared_rows": trace["rows"],
            "forced_active_mode_max_abs": float((active[:, :, 0] - active[:, :, 1]).abs().max())}


def fit(arm, panel, directory, device):
    torch.manual_seed(SEED)
    host = load_host(device)
    conditioner = FullDepthConditioner(host, 2).to(device).eval() if arm == "conditioned" else None
    gate = FirstEffectReadout(host).to(device).eval() if conditioner is not None else None
    modules = {"predictor": host.level.predictor, "action_embed": host.level.action_embed}
    if conditioner is not None:
        modules.update(conditioner=conditioner, gate=gate)
    parameters = [parameter for part in modules.values() for parameter in part.requires_grad_(True).parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    history, past, plan, z, _, relevance = fan(panel)
    (source_history, source_past, source_action, source_z, source_relevance), _ = sources(panel)
    source_groups = tuple((source_relevance == value).nonzero().flatten() for value in (False, True))
    if not min(map(len, source_groups)):
        raise ValueError("Need informative and uninformative factual training windows")
    rng = torch.Generator(device=device).manual_seed(SEED)
    with (directory / f"{arm}_training.csv").open("x", encoding="utf-8", newline="") as file:
        trace = csv.writer(file)
        trace.writerow(["epoch", "gradient_l2", "h1_mse", "h2_mse", "h3_mse", "h4_mse", "source_mse", "gate_bce"])
        for epoch in range(1, EPOCHS + 1):
            aggregate = []
            for b in torch.randperm(len(history), generator=rng, device=device).split(BATCH):
                s = torch.cat([group[torch.randint(len(group), (BATCH // 2,), generator=rng, device=device)]
                               for group in source_groups])
                main, local = {}, {}
                predicted = forecast(host, conditioner, gate, history[b], past[b], plan[b], trace=main)
                issued = forecast(host, conditioner, gate, source_history[s], source_past[s], source_action[s], trace=local)
                full_mse = (predicted - z[b]).square().mean(dim=(0, 1, 2, 4))
                local_mse = (issued - source_z[s]).square().mean()
                bce = full_mse.new_zeros(())
                if gate is not None:
                    fan_bce = F.binary_cross_entropy_with_logits(main["logits"][:, :, 0, 0], relevance[b].float())
                    local_bce = F.binary_cross_entropy_with_logits(local["logits"][:, 0, 0, 0], source_relevance[s].float())
                    bce = .5 * (fan_bce + local_bce)
                loss = .5 * (full_mse.mean() + local_mse) + bce
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
                record = torch.cat((norm[None], full_mse, local_mse[None], bce[None])).detach().cpu().numpy()
                if not np.isfinite(record).all():
                    raise FloatingPointError(f"Nonfinite training trace for {arm} at epoch {epoch}")
                optimizer.step()
                trace.writerow([epoch, *record])
                aggregate.append(record)
            file.flush()
            if epoch == 1 or epoch % 5 == 0:
                print(f"{arm} epoch {epoch}/{EPOCHS}: {np.mean(aggregate, axis=0)}", flush=True)
    torch.save({name: module.state_dict() for name, module in modules.items()}, directory / f"{arm}.pt")


def endpoint_metrics(values, y):
    targets = np.diff(y, axis=3).squeeze(3).reshape(-1, 4, 2)
    power = np.square(targets).sum((0, 2))
    c, l, f = (values[k].astype(np.float64) for k in ("C", "L", "F"))
    parts = {"C-Y": c - y, "L-C": l - c, "F-L": f - l}
    closure = float(np.abs(sum(parts.values()) - (f - y)).max())
    if closure > 1e-10:
        raise AssertionError("The physical decomposition must close per heldout branch")

    def rms(array):
        return np.sqrt(np.square(array.reshape(-1, 4, 2)).sum(-1).mean(0)).tolist()

    def gap(array):
        return np.diff(array, axis=3).squeeze(3).reshape(-1, 4, 2)

    def gain(array):
        return [float(num / den) if den > 0 else None
                for num, den in zip((gap(array) * targets).sum((0, 2)), power, strict=True)]

    return {"truth_gap_rms": rms(targets), "absolute_rmse": {k: rms(v - y) for k, v in values.items()},
            "mode_gap_rmse": {k: rms(gap(v) - targets) for k, v in values.items()},
            "mode_gap_gain": {k: gain(v) for k, v in (values | parts).items()}, "decomposition_max_residual": closure}


def gate_counts(relevant, predicted):
    relevant, predicted = np.broadcast_arrays(relevant, predicted)
    eligible = int(relevant.sum())
    n_total = int(relevant.size)
    return {"total": n_total, "active": eligible, "inactive": n_total - eligible,
            "false_negative": int((~predicted & relevant).sum()), "false_positive": int((predicted & ~relevant).sum())}


def evaluate_fan(host, conditioner, gate, panel, local_labels):
    history, past, actions, target, physical, relevance = fan(panel)
    steps = target.size(-2)
    free_path, teacher_path, local_gates = [], [], []
    compaction = np.zeros((steps, 2), dtype=np.int64)
    dense_max = 0.0
    for start in range(0, len(history), 16):
        sl = slice(start, start + 16)
        h, p, a, z = (value[sl] for value in (history, past, actions, target))
        route = {}
        free = forecast(host, conditioner, gate, h, p, a, compact=True, trace=route)
        if gate is not None:
            dense = forecast(host, conditioner, gate, h, p, a, compact=False)
            torch.testing.assert_close(free, dense, rtol=2e-4, atol=2e-5)
            dense_max = max(dense_max, float((free - dense).abs().max()))
            compaction += np.asarray(route["rows"])
        free_path.append(free.expand_as(z))
        b, c, m, _, d = z.shape
        ctrl = torch.cat((p[:, None].expand(-1, c, -1, -1, -1), a), dim=2)
        true_ctrl = ctrl[:, :, None].expand(-1, -1, m, -1, -1, -1).flatten(0, 2)
        root = h[:, None, None].expand(-1, c, m, -1, -1)
        real = torch.cat((root, z), dim=-2).flatten(0, 2)
        mode = torch.arange(m, device=z.device).repeat(b * c)
        index = torch.arange(len(real), device=z.device)
        teacher, exposures = [], []
        for k in range(steps):
            diag = {}
            one = forecast(host, conditioner, gate, real[:, k:k + 3], true_ctrl[:, k:k + 2],
                           true_ctrl[:, None, k + 2:k + 3], compact=True, trace=diag)
            selected = one[index, 0, mode if conditioner is not None else torch.zeros_like(mode), 0]
            teacher.append(selected.reshape(b, c, m, d))
            if gate is not None:
                exposures.append(diag["exposure"][:, 0, 0, 0].reshape(b, c, m))
        teacher_path.append(torch.stack(teacher, dim=-2))
        if exposures:
            local_gates.append(torch.stack(exposures, dim=-1))
    free, teacher = torch.cat(free_path), torch.cat(teacher_path)
    torch.testing.assert_close(free[..., 0, :], teacher[..., 0, :], rtol=2e-4, atol=2e-5)
    shape = (*panel["root_z"].shape[:2], *target.shape[1:])
    states = {key: value.reshape(shape) for key, value in (("C", target), ("L", teacher), ("F", free))}
    measured = {"latent_mse": {k: (v - target).square().mean(dim=(0, 1, 2, 4)).tolist()
                               for k, v in (("F", free), ("L", teacher))},
                "free_teacher_h1_max_abs": float((free[..., 0, :] - teacher[..., 0, :]).abs().max())}
    arrays = {}
    if local_gates:
        exposure = torch.cat(local_gates).reshape(*shape[:-1]).cpu().numpy()
        actual = local_labels >= 0
        observed_h1 = actual[..., 0, 0].reshape(relevance.shape)
        np.testing.assert_array_equal(relevance.cpu().numpy(), observed_h1)
        measured.update(local_gate_by_h=[gate_counts(actual[..., k], exposure[..., k] > 0) for k in range(steps)],
                        compact_rows=compaction.tolist(), dense_rows=[[2 * len(history) * target.size(1)] * 2] * steps,
                        compact_dense_max_abs=dense_max)
        arrays["teacher_exposure"] = exposure
    return states, measured, arrays, physical.reshape(*shape[:-1], 2).cpu().numpy().astype(np.float64)


def oracle_values(ids, truth, stock, native, conditional):
    y = truth.shape
    costs = {k: np.empty(y[:4], dtype=np.float32)
             for k in ("physical", "stock", "native", "true", "uniform", "wrong")}
    for index, scene_id in enumerate(ids):
        scene = make_scene(int(scene_id))
        for root in range(y[1]):
            cost = VisibleMapCost(scene, scene.southeast if root in (0, 2) else scene.northwest)
            for name, positions in (("physical", truth), ("stock", stock), ("native", native), ("true", conditional)):
                costs[name][index, root] = cost(positions[index, root, ..., -1, :])
    costs["uniform"][:] = costs["true"].mean(-1, keepdims=True)
    costs["wrong"][:] = costs["true"][..., ::-1]
    physical = costs["physical"]
    best = physical.min(axis=2)
    measured, arrays = {}, dict(costs)
    for name, forecast_cost in costs.items():
        choices = forecast_cost.argmin(axis=2)
        chosen = np.take_along_axis(physical, choices[:, :, None], axis=2).squeeze(2)
        regret = chosen - best
        arrays[f"{name}_choice"] = choices
        arrays[f"{name}_regret"] = regret
        measured[name] = {"mean_regret_by_root": regret.mean(axis=(0, 2)).tolist(),
                          "navigation_mean_regret": float(regret[:, 2:].mean()),
                          "zero_regret_fraction_by_root": (regret <= 1e-6).mean(axis=(0, 2)).tolist()}
    return measured, arrays


def predict_sources(host, conditioner, gate, panel):
    parts, indices = sources(panel)
    h, p, a, target, _ = parts
    issued, gates = [], []
    for start in range(0, len(h), 128):
        sl = slice(start, start + 128)
        trace = {}
        result = forecast(host, conditioner, gate, h[sl], p[sl], a[sl], compact=True, trace=trace)
        issued.append(result[:, 0, :, 0])
        gates.append(trace["exposure"][:, 0, 0, 0])
    return torch.cat(issued), torch.cat(gates), indices


def lifecycle(host, conditioner, gate, calibration, heldout, directory):
    fitted, _, index = predict_sources(host, conditioner, gate, calibration)
    real_mode = calibration["life_mode"][index].long()
    observed = calibration["life_z"][index[0], index[1], index[2] + 1]
    residual = observed - fitted[torch.arange(len(fitted), device=fitted.device), real_mode]
    scale = residual.square().mean(0).sqrt().clamp_min(SCALE_FLOOR)
    torch.save(scale.cpu(), directory / "likelihood_scale.pt")

    issued, exposure, index = predict_sources(host, conditioner, gate, heldout)
    rows, initial, time = index
    width = heldout["life_valid"].size(-1)
    frames = heldout["life_z"]
    labels, events, eligible = (heldout[key] for key in ("life_mode", "life_event", "life_informative"))
    if (events[..., :2] > 0).any():
        raise AssertionError("Physical event occurs before the three-frame contract")
    arrived = frames[rows, initial, time + 1]
    log_init = issued.new_full((len(frames), 2, 2), -float(np.log(2)))
    post, omitted = log_init.clone(), log_init.clone()
    shape = (*labels.shape, 2)
    timeline = {k: torch.full(shape, torch.nan, device=post.device) for k in ("prior", "post", "omit", "log_likelihood")}
    predictions = torch.full((*shape, issued.size(-1)), torch.nan, device=post.device)
    gates = torch.full(labels.shape, torch.nan, device=post.device)
    predictions[index], gates[index] = issued, exposure
    for t in range(2, width):
        selected = time == t
        a, b = rows[selected], initial[selected]
        if not len(a):
            continue
        prior = advance_log_support(post[a, b], SWITCH)
        from_omit = advance_log_support(omitted[a, b], SWITCH)
        post[a, b], likelihood = write_observation(prior, issued[selected], arrived[selected], scale)
        include = events[a, b, t] != 3
        omitted[a, b] = (from_omit + likelihood * include[:, None]).log_softmax(-1)
        timeline["prior"][a, b, t], timeline["post"][a, b, t] = prior, post[a, b]
        timeline["omit"][a, b, t], timeline["log_likelihood"][a, b, t] = omitted[a, b], likelihood

    def mode_metrics(support, correct):
        nll = -support.gather(-1, correct[:, None]).squeeze(-1)
        belief = (-nll).exp()
        return {"true_probability_mean": float(belief.mean()), "correct_over_half": int((belief > .5).sum()),
                "mean_nll": float(nll.mean()), "events": len(belief)}

    actual_mode = labels[index].long()
    actual_ll = timeline["log_likelihood"][index]
    event_report = {}
    for code, name in ((1, "first"), (2, "reentry"), (3, "reversed")):
        pick = events[index] == code
        event_report[name] = {k: mode_metrics(timeline[k][index][pick], actual_mode[pick]) for k in ("prior", "post", "omit")}
        event_report[name]["uniform"] = mode_metrics(torch.full_like(actual_ll[pick], -float(np.log(2))), actual_mode[pick])
        oracle = torch.full_like(actual_ll[pick], -torch.inf)
        oracle.scatter_(-1, actual_mode[pick, None], 0.)
        event_report[name]["true"] = mode_metrics(oracle, actual_mode[pick])
        directional = actual_ll[pick].gather(-1, actual_mode[pick, None]).squeeze(-1) - actual_ll[pick].gather(
            -1, 1 - actual_mode[pick, None]).squeeze(-1)
        event_report[name]["correct_likelihood_order"] = int((directional > 0).sum())
        event_report[name]["mean_signed_log_likelihood_gap"] = float(directional.mean())
    active = exposure > 0
    shared = ~active
    if not torch.equal(issued[shared, 0], issued[shared, 1]):
        raise AssertionError("Compacted shared factual suffixes must issue the same latent")
    np.testing.assert_array_equal(actual_ll[shared, 0].cpu().numpy(), actual_ll[shared, 1].cpu().numpy())
    likelihood_delta = actual_ll.diff(dim=-1).abs().squeeze(-1)
    quiet_truth = ~eligible[index]
    summary = {"calibration_windows": len(residual), "calibration_scale_min_med_max": scale.quantile(
        scale.new_tensor([0, .5, 1])).tolist(), "events": event_report,
        "gate": gate_counts(eligible[index].cpu().numpy(), active.cpu().numpy()),
        "quiet_truth_mean_abs_log_likelihood_gap": float(likelihood_delta[quiet_truth].mean()),
        "exact_shared_count": int(shared.sum())}
    arrays = {k: v.cpu().numpy() for k, v in timeline.items()}
    arrays.update(issued=predictions.cpu().numpy(), exposure=gates.cpu().numpy(),
                  **{k: heldout[k].cpu().numpy() for k in (
                      "seed", "life_z", "life_mode", "life_valid", "life_event", "life_informative", "life_action", "life_xy",
                  )})
    return summary, arrays


@torch.no_grad()
def evaluate(directory, device):
    holdout = read_panel(directory / "data_holdout.npz", HOLDOUT, device)
    calibration = read_panel(directory / "data_train.npz", CALIBRATE, device)
    with np.load(directory / "data_holdout_gate.npz", allow_pickle=False) as gate_labels:
        local = gate_labels["first_local_divergence_raw"]
    reports, arrays, trajectories = {}, {}, {}
    readouts = {"primary": load_readout(device, directory), "stock_p0c": load_readout(device)}
    for arm in ("stock", "native", "conditioned"):
        print(f"Evaluating holdout fan: {arm}", flush=True)
        modules = load_trained(directory, arm, device)
        states, summary, observed, physical = evaluate_fan(*modules, holdout, local)
        arrays.update({f"{arm}_{k}_latent": value.cpu().numpy() for k, value in states.items()})
        arrays.update({f"{arm}_{k}": value for k, value in observed.items()})
        for name, readout in readouts.items():
            decoded = {key: readout(z).double().cpu().numpy() for key, z in states.items()}
            summary[name] = {"all": endpoint_metrics(decoded, physical),
                             "contact": endpoint_metrics({k: v[:, :2] for k, v in decoded.items()}, physical[:, :2]),
                             "navigation": endpoint_metrics({k: v[:, 2:] for k, v in decoded.items()}, physical[:, 2:])}
            if name == "primary":
                trajectories[arm] = decoded["F"]
                arrays.update({f"{arm}_{k}_xy": v for k, v in decoded.items()})
        reports[arm] = summary
        if arm != "stock":
            trace = np.loadtxt(directory / f"{arm}_training.csv", delimiter=",", skiprows=1)
            reports[arm]["training_steps"] = len(trace)
            reports[arm]["gradient_l2_min_med_max"] = np.quantile(trace[:, 1], [0, .5, 1]).tolist()
    ids = holdout["seed"].cpu().numpy()
    reports["oracle"], ar = oracle_values(ids, physical, *(trajectories[key] for key in ("stock", "native", "conditioned")))
    arrays.update(ar, physical=physical, scene_ids=ids, candidate_actions=holdout["action"].cpu().numpy(),
                  root_z=holdout["root_z"].cpu().numpy(), root_actions=holdout["root_action"].cpu().numpy())
    np.savez_compressed(directory / "oracle_branches.npz", **arrays)
    reports["lifecycle"], lifecycle_arrays = lifecycle(*modules, calibration, holdout, directory)
    np.savez_compressed(directory / "lifecycle.npz", **lifecycle_arrays)
    with (directory / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, allow_nan=False)
    print("Oracle and lifecycle archives saved", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--phase", choices=("fit", "evaluate"), required=True)
    cli.add_argument("--run", type=Path, default=RUN)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    if args.phase == "fit":
        panel = read_panel(args.run / "data_train.npz", TRAIN, device)
        verification = check_native(load_host(device), panel)
        with (args.run / "interface.json").open("x", encoding="utf-8") as f:
            json.dump(verification, f, indent=2, allow_nan=False)
        print(f"Verified native interface: {verification}", flush=True)
        for arm in ("conditioned", "native"):
            fit(arm, panel, args.run, device)
    else:
        evaluate(args.run, device)


if __name__ == "__main__":
    main()
