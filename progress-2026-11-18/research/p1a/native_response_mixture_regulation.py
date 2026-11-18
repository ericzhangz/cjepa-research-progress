"""Fit native response regulation on the recorded PushT writes and candidate fans."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import torch

if __package__:
    from .native_response_mixture_pilot import NORMALIZATION, ROOT, forecasts
    from .native_response_observation import FULL, PATCH, NativeResponseObservation
else:
    from native_response_mixture_pilot import NORMALIZATION, ROOT, forecasts
    from native_response_observation import FULL, PATCH, NativeResponseObservation


RUNS = ROOT / "runs"
ARCHIVE_DIR = RUNS / "native_response_mixture_counterevidence_20261113"
OBSERVATION = RUNS / "native_response_observation_20261113/observation.pt"
WARM_START = RUNS / "native_response_mixture_physical_20261113/pilot.pt"
QUERY_STEPS = (30, 45, 60, 105, 110, 125)
CUE_STEPS = {15: PATCH, 105: FULL, 110: FULL}
FIRST_SOURCE = 2
UPDATES = 512
BATCH_SIZE = 2
SEED = 1611
REANCHORED_SEED = 1714
REANCHORED_UPDATES = 128
REANCHORED_BATCH_SIZE = 1
REANCHORED_LOSS_STEPS = (110, 125)
MODES = 4


@dataclass
class RegulationData:
    scene_ids: np.ndarray
    visible: torch.Tensor
    full: torch.Tensor
    future: torch.Tensor
    valid: torch.Tensor
    query_valid: torch.Tensor
    commands: torch.Tensor
    candidates: torch.Tensor
    object_xy: torch.Tensor
    patch_center: torch.Tensor


def _native_model(device):
    sys.path.insert(0, str(ROOT / "research_sources_2026-10-22/lewm_assets"))
    from prepare_assets import load_model

    return load_model(device).eval().requires_grad_(False)


def _action_moments(like):
    moments = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
    return [like.new_tensor(moments[key]) for key in ("mean", "std")]


@torch.inference_mode()
def encode_data(split, archive_dir, device):
    archive_dir = Path(archive_dir)
    output = archive_dir / f"{split}_latents.npz"
    if output.exists():
        raise FileExistsError(output)
    model = _native_model(device)
    sys.path.insert(0, str(ROOT / "oic_step1"))
    from run_native_pusht_host import encode

    with np.load(archive_dir / f"{split}_physics.npz", allow_pickle=False) as raw:
        ids = raw["scene_ids"]
        raw_public = raw["obs_rgb"]
        raw_full = raw["full_rgb"]
        observation_steps = raw["observation_steps"]
        np.testing.assert_array_equal(observation_steps, np.arange(0, 126, 5))
        np.testing.assert_array_equal(raw["cue_observation_steps"], tuple(CUE_STEPS))
        np.testing.assert_array_equal(raw["query_steps"], QUERY_STEPS)
        print(f"Encoding {split} factual full frames", flush=True)
        full = encode(model, raw_full).cpu().numpy()
        public = np.empty_like(full)
        full_columns = [step // 5 for step, kind in CUE_STEPS.items() if kind == FULL]
        np.testing.assert_array_equal(raw_public[:, :, :, full_columns], raw_full[:, :, :, full_columns])
        public[:, :, :, full_columns] = full[:, :, :, full_columns]
        partial_columns = [0, 1, 2, 3]
        public[:, :, :, partial_columns] = encode(model, raw_public[:, :, :, partial_columns]).cpu().numpy()
        strip_columns = [i for i in range(4, len(observation_steps)) if i not in full_columns]
        strip = raw_public[0, 0, 0, strip_columns[0]]
        strips = raw_public[:, :, :, strip_columns]
        changed = (strips != strip).any(axis=(-1, -2, -3))
        encoded_strips = np.broadcast_to(encode(model, strip[None]).cpu().numpy()[0],
                                         (*changed.shape, full.shape[-1])).copy()
        if changed.any():
            encoded_strips[changed] = encode(model, strips[changed]).cpu().numpy()
        public[:, :, :, strip_columns] = encoded_strips
        print(f"Encoded {int(changed.sum())} factual nonblank strip windows for {split}", flush=True)
        print(f"Encoding {split} independent branch frames", flush=True)
        future = encode(model, raw["future_rgb"]).cpu().numpy()
    if not all(np.isfinite(values).all() for values in (full, public, future)):
        raise ValueError("Encountered a nonfinite frozen LeWM encoding")
    temp_path = output.with_suffix(".partial.npz")
    np.savez_compressed(temp_path, scene_ids=ids, full=full, visible=public, future=future)
    temp_path.replace(output)
    print(f"Saved {output}", flush=True)


def load_data(split, archive_dir, device):
    archive_dir = Path(archive_dir)

    def tensor(array):
        return torch.as_tensor(array, device=device)

    with np.load(archive_dir / f"{split}_physics.npz", allow_pickle=False) as raw, np.load(
        archive_dir / f"{split}_latents.npz", allow_pickle=False,
    ) as encoded:
        ids = raw["scene_ids"]
        np.testing.assert_array_equal(ids, encoded["scene_ids"])
        np.testing.assert_array_equal(raw["query_steps"], QUERY_STEPS)
        visible, full, future = [tensor(encoded[key]) for key in ("visible", "full", "future")]
        scene, world, program, time, size = visible.shape
        if (world, program, time, size) != (4, 2, 26, 192) or full.shape != visible.shape:
            raise ValueError("Expected four physical worlds and two programs with aligned public/full encodings")
        if future.shape != (scene, world, program, len(QUERY_STEPS), 4, 2, size):
            raise ValueError("Unexpected independent branch encoding dimensions")
        executed = raw["actual_step"].reshape(scene, world, program, time - 1, 5)
        valid = executed.all(-1)
        if (valid[..., 1:] & ~valid[..., :-1]).any():
            raise ValueError("A terminated prefix cannot resume on later factual blocks")
        raw_query_valid = raw["query_actual_step"]
        query_valid = np.stack((raw_query_valid[..., :5].all(-1), raw_query_valid.all(-1)), axis=-1)
        query_valid &= np.take(valid, np.asarray(QUERY_STEPS) // 5 - 1, axis=-1)[..., None, None]
        mean, std = _action_moments(visible)

        def commands(key):
            return ((tensor(raw[key]).to(visible) - mean) / std).flatten(-2)

        object_xy = tensor(raw["query_physical_state"][..., [5, 10], 4:6].copy()).to(visible)
        return RegulationData(ids, visible, full, future, tensor(valid), tensor(query_valid),
                              commands("executed_actions"), commands("query_actions"), object_xy,
                              tensor(raw["patch"].mean(axis=1)).to(visible))


def make_model(device, observation_path=OBSERVATION, warm_start=WARM_START, trained=None, *,
               mechanism="legacy"):
    model = _native_model(device)
    if mechanism == "reanchored":
        return _make_reanchored_model(model, observation_path, warm_start, trained, device)
    op = model.enable_response_mixture(gated=True, transition_timing="before")
    model.observation_model = NativeResponseObservation.load(observation_path, device)
    checkpoint = torch.load(warm_start if trained is None else trained, map_location=device, weights_only=True)
    model.predictor.transformer.layers[-1].load_state_dict(checkpoint["last_block"])
    model.predictor.transformer.layers[-1].requires_grad_(True)
    model.pred_proj.load_state_dict(checkpoint["pred_proj"])
    model.pred_proj.requires_grad_(True)
    if trained is None:
        old = checkpoint["response_mixture"]
        op.initialize_gated_direction(old["readout.weight"][0], old["readout.bias"][0])
    else:
        op.load_state_dict(checkpoint["response_mixture"])
        model.observation_model.load_state_dict(checkpoint["observation_model"])
    model.eval()
    return model, checkpoint["object_coordinates"]


def _make_reanchored_model(model, observation_path, physical_readout, trained, device):
    reference = next(model.action_encoder.parameters())
    mean, std = _action_moments(reference)
    raw_hold = reference.new_zeros(1, 1, 5, 2)
    normalized_hold = ((raw_hold - mean) / std).flatten(-2)
    with torch.no_grad():
        hold_embedding = model.action_encoder(normalized_hold).reshape(-1)
    op = model.enable_response_mixture(parameterization="reanchored", num_modes=MODES,
                                       reference_action_embedding=hold_embedding)
    model.register_buffer("normalized_response_hold", normalized_hold.reshape(-1), persistent=False)
    model.predictor.transformer.layers[-1].requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    model.observation_model = NativeResponseObservation.load(observation_path, device).requires_grad_(False)
    checkpoint = torch.load(physical_readout if trained is None else trained, map_location=device, weights_only=True)
    if trained is not None:
        if checkpoint.get("mechanism") != "reanchored":
            raise ValueError("Expected a §17 reanchored-regulation checkpoint")
        model.predictor.transformer.layers[-1].load_state_dict(checkpoint["last_block"])
        model.pred_proj.load_state_dict(checkpoint["pred_proj"])
        op.load_state_dict(checkpoint["response_mixture"])
        model.observation_model.load_state_dict(checkpoint["observation_model"])
    return model.eval(), checkpoint["object_coordinates"]


def _factual_inputs(panel, rows):
    visible = panel.visible[rows]
    commands = panel.commands[rows, None].expand(*visible.shape[:3], *panel.commands.shape[-2:])
    return visible.flatten(0, 2), commands.flatten(0, 2)


def _issue_all(model, visible, commands, valid):
    windows = visible[:, :-1].unfold(1, 3, 1).transpose(-1, -2)
    controls = model.action_encoder(commands).unfold(1, 3, 1).transpose(-1, -2)
    valid = valid[:, FIRST_SOURCE:]
    # Means and transition logits depend on the window/action, independently of the belief.
    # The actual pre-emission belief is applied sequentially in trace_factual.
    _, _, issued = model.predict_response(windows[valid], controls[valid],
                                          model.response_mixture.initial_state(int(valid.sum())), factual=True)

    def scatter(value):
        result = value.new_zeros((*valid.shape, *value.shape[1:]))
        result[valid] = value
        return result

    return issued._make(scatter(value) if value is not None else None for value in issued)


@torch.no_grad()
def calibrate_full(model, panel):
    valid = panel.valid.flatten(0, 2)
    visible, commands = _factual_inputs(panel, slice(None))
    issued = _issue_all(model, visible, commands, valid)
    target_columns = [step // 5 for step, kind in CUE_STEPS.items() if kind == FULL]
    source_columns = [column - 1 for column in target_columns]
    residual = panel.full.flatten(0, 2)[:, target_columns] - issued.means[:, np.asarray(source_columns) - FIRST_SOURCE, 1]
    actual = residual[valid[:, source_columns]]
    if len(actual) == 0:
        raise ValueError("No fully executed train factual blocks for full-view calibration")
    model.observation_model.set_full_reference_sigma(actual.square().mean(0).sqrt())
    return len(actual)


def trace_factual(model, panel, rows=slice(None), *, skip_steps=(), trace=False):
    """Commit each actual public arrival to the exact native pre-emission prior."""
    op = model.response_mixture
    visible, commands = _factual_inputs(panel, rows)
    public_valid = panel.valid[rows].flatten(0, 2)[:, FIRST_SOURCE:]
    issued = _issue_all(model, visible, commands, panel.valid[rows].flatten(0, 2))
    n, times, _ = issued.log_weights.shape
    ll = visible.new_zeros(n, times, 3)
    patch_center = panel.patch_center[rows].repeat_interleave(panel.visible.shape[1] * panel.visible.shape[2], dim=0)
    for step, channel in CUE_STEPS.items():
        arrival = step // 5
        index = arrival - FIRST_SOURCE - 1
        keep = public_valid[:, index]
        if keep.any() and step not in skip_steps:
            ll[keep, index] = model.observation_model(issued.means[keep, index], visible[keep, arrival], channel,
                                                    patch_center[keep] if channel == PATCH else None)

    state = op.initial_state(n)
    history = [state]
    pre_emission, evidences = [], []
    for index in range(times):
        keep = public_valid[:, index]
        emission = op.advance(state, issued.reset_logit[:, index])
        arrival = 5 * (FIRST_SOURCE + index + 1)
        if arrival in CUE_STEPS and arrival not in skip_steps:
            record = issued._replace(log_weights=emission)
            _, posterior, evidence = op.condition(record, ll[:, index])
        else:
            # An unscored observation still supplies the next real window.
            posterior, evidence = emission, torch.zeros_like(emission[:, 0])
        state = torch.where(keep[:, None], posterior, state)
        history.append(state)
        pre_emission.append(torch.where(keep[:, None], emission, state))
        evidences.append(torch.where(keep, evidence, 0))

    states = torch.stack(history, dim=1)
    evidence = torch.stack(evidences, dim=1)
    cue_indexes = [step // 5 - FIRST_SOURCE - 1 for step in CUE_STEPS if step not in skip_steps]
    selected = public_valid[:, cue_indexes]
    nll = -evidence[:, cue_indexes][selected] / visible.size(-1)
    result = {"states": states, "nll": nll, "log_likelihood": ll, "reset_logit": issued.reset_logit}
    if trace:
        result.update(emission=torch.stack(pre_emission, dim=1), issued_means=issued.means,
                      relevance=issued.relevance, offset=issued.offset, valid=public_valid, evidence=evidence)
    return result


def _query_inputs(panel, rows, steps):
    visible, commands = _factual_inputs(panel, rows)
    positions = [step // 5 for step in steps]
    windows = torch.stack([visible[:, pos - 2:pos + 1] for pos in positions], dim=1).flatten(0, 1)
    past = torch.stack([commands[:, pos - 2:pos] for pos in positions], dim=1).flatten(0, 1)
    candidates = panel.candidates[rows, None, None, None].expand(
        *panel.visible[rows].shape[:3], len(positions), *panel.candidates.shape[-3:],
    ).flatten(0, 3)
    return windows, past, candidates


def _query_roots(states, steps):
    return states[:, [step // 5 - FIRST_SOURCE for step in steps]]


def _queries(model, panel, rows, states, *, trace=False):
    windows, past, candidates = _query_inputs(panel, rows, QUERY_STEPS)
    roots = _query_roots(states, QUERY_STEPS).flatten(0, 1)
    fields = {} if trace else None
    predicted = forecasts(model, windows, past, candidates, roots, trace=fields)
    if trace:
        fields["h1_modes"] = fields["h1_modes"].reshape(*panel.future[rows].shape[:5], 3, predicted.shape[-1])
    return predicted.reshape_as(panel.future[rows]), fields


@torch.no_grad()
def evaluate(model, panel, rows=slice(None), *, skip_steps=(), prior_only=False):
    factual = trace_factual(model, panel, rows, skip_steps=skip_steps, trace=True)
    states = factual["states"]
    if prior_only:
        states = model.response_mixture.initial_state(len(states))[:, None].expand_as(states)
    predictions, query_trace = _queries(model, panel, rows, states, trace=True)
    return {"predictions": predictions, **factual, **query_trace}


def centered(values, valid):
    weights = valid[..., None]
    mean = (values * weights).sum(dim=-3, keepdim=True) / weights.sum(dim=-3, keepdim=True).clamp_min(1)
    return (values - mean) * weights


def _mean_by_horizon(values, valid):
    # Dimensions are scene, world, program, query, candidate, horizon, coordinate.
    totals = (values * valid[..., None]).sum(dim=(0, 1, 2, 3, 4, 6))
    counts = valid.sum(dim=(0, 1, 2, 3, 4)) * values.size(-1)
    return totals / counts.clamp_min(1)


def _loss_scales(panel):
    latent_mean = (panel.future * panel.query_valid[..., None]).sum(dim=(0, 1, 2, 3, 4))
    latent_mean = latent_mean / panel.query_valid.sum(dim=(0, 1, 2, 3, 4))[:, None]
    latent_var = _mean_by_horizon((panel.future - latent_mean).square(), panel.query_valid)
    physics_valid = panel.query_valid & (panel.query_valid.sum(dim=-2, keepdim=True) > 1)
    physics_var = _mean_by_horizon(centered(panel.object_xy, physics_valid).square(), physics_valid)
    if not all(torch.isfinite(scale).all() and (scale > 0).all() for scale in (latent_var, physics_var)):
        raise ValueError("Training requires positive target scales and independent outcomes at both horizons")
    return latent_var, physics_var


def _losses(panel, rows, predicted, nll, object_coordinates, scales):
    valid = panel.query_valid[rows]
    latent = _mean_by_horizon((predicted - panel.future[rows]).square(), valid) / scales[0]
    w = object_coordinates[0].to(predicted)
    physical_error = predicted @ w - panel.object_xy[rows]
    physics_valid = valid & (valid.sum(dim=-2, keepdim=True) > 1)
    physical = _mean_by_horizon(centered(physical_error, physics_valid).square(), physics_valid) / scales[1]
    evidence = nll.sum() / max(1, nll.numel())
    return evidence, latent, physical


def _response_controls(model, commands):
    encoded = model.action_encoder(commands)
    hold = (commands == model.normalized_response_hold).all(-1, keepdim=True)
    return torch.where(hold, model.response_mixture.reference_action_embedding, encoded)


def _reanchored_facts(model, panel, rows, *, curve_anchor=True, trace=False):
    """Prepare causal native predictions from each factual window before scoring its arrival."""
    visible, commands = _factual_inputs(panel, rows)
    arrivals = np.arange(5 * (FIRST_SOURCE + 1), 5 * visible.size(1), 5)
    valid = panel.valid[rows].flatten(0, 2)[:, FIRST_SOURCE:]
    window = visible[:, :-1].unfold(1, 3, 1).transpose(-1, -2)
    actions = _response_controls(model, commands).unfold(1, 3, 1).transpose(-1, -2)
    cue_indexes = {step: int(np.searchsorted(arrivals, step)) for step in CUE_STEPS}
    scored = torch.zeros_like(valid)
    scored[:, list(cue_indexes.values())] = valid[:, list(cue_indexes.values())]
    # Unscored frames always refresh the public window. Their unused issue predictions
    # are needed for the saved trace but have no path to the fitting objective.
    issued_at = valid if trace else scored
    op = model.response_mixture
    _, _, issue = model.predict_response(window[issued_at], actions[issued_at],
                                         op.initial_state(window[issued_at].size(0)),
                                         factual=True, curve_anchor=curve_anchor)

    def scatter(value):
        result = value.new_zeros((*valid.shape, *value.shape[1:]))
        result[issued_at] = value
        return result

    means = scatter(issue.means)
    log_likelihood = means.new_zeros((*valid.shape, means.size(-2)))
    patch_center = panel.patch_center[rows].repeat_interleave(panel.visible.shape[1] * panel.visible.shape[2], dim=0)
    for step, channel in CUE_STEPS.items():
        index = cue_indexes[step]
        keep = scored[:, index]
        if keep.any():
            center = patch_center[keep] if channel == PATCH else None
            log_likelihood[keep, index] = model.observation_model(means[keep, index], visible[keep, step // 5],
                                                                 channel, center)

    facts = {"issued_template": issue, "log_likelihood": log_likelihood, "issued_means": means,
             "valid": valid, "scored": scored, "held": (commands[:, FIRST_SOURCE:] ==
                                                         model.normalized_response_hold).all(-1)}
    if trace:
        facts.update(source_preactivation=scatter(issue.preactivation),
                     source_reference_preactivation=scatter(issue.reference_preactivation),
                     source_activation_delta=scatter(issue.activation_delta))
    return facts


def _reanchored_support(model, facts, *, skip_steps=(), prior_only=False):
    """Commit each scored actual arrival once, leaving all other support unchanged."""
    op = model.response_mixture
    ll = facts["log_likelihood"]
    state = op.initial_state(len(ll))
    used = facts["scored"].clone()
    if prior_only:
        used.zero_()
    else:
        for step in skip_steps:
            used[:, step // 5 - FIRST_SOURCE - 1] = False
    states, evidences = [state], []
    for index in range(ll.size(1)):
        write = used[:, index]
        if (5 * (FIRST_SOURCE + index + 1)) in CUE_STEPS:
            issue = facts["issued_template"]._replace(log_weights=state, means=facts["issued_means"][:, index])
            _, updated, evidence = op.condition(issue, ll[:, index])
            state = torch.where(write[:, None], updated, state)
        else:
            evidence = state.new_zeros(len(ll))
        evidences.append(torch.where(write, evidence, 0))
        states.append(state)

    evidence = torch.stack(evidences, dim=1)
    states = torch.stack(states, dim=1)
    return {"states": states, "emission": states[:, :-1], "evidence": evidence, "used": used,
            "nll": -evidence[used] / facts["issued_means"].size(-1)}


def _reanchored_modes(model, panel, rows, steps, *, curve_anchor=True, trace=False):
    windows, past, candidates = _query_inputs(panel, rows, steps)
    batch, fan, horizon, action_size = candidates.shape
    fields = {} if trace else None
    actions = _response_controls(model, candidates.reshape(batch, fan * horizon, action_size))
    modes = model.rollout_response_modes(windows, _response_controls(model, past),
                                         actions.reshape(batch, fan, horizon, -1),
                                         trace=fields, curve_anchor=curve_anchor)
    shape = (*panel.visible[rows].shape[:3], len(steps), fan, horizon, modes.size(-2))
    modes = modes.reshape(*shape, modes.size(-1))
    if trace:
        fields = {f"read_{key}": value.reshape(*shape, value.size(-1)) for key, value in fields.items()}
    return modes, fields


def _reanchored_read(modes, states, steps):
    log_p = _query_roots(states, steps).reshape(*modes.shape[:4], modes.size(-2))
    mean = (modes * log_p.exp()[..., None, None, :, None]).sum(-2)
    return mean, log_p


def _targets(panel, rows, steps):
    indexes = [QUERY_STEPS.index(step) for step in steps]
    return (panel.future[rows][..., indexes, :, :, :], panel.object_xy[rows][..., indexes, :, :, :],
            panel.query_valid[rows][..., indexes, :, :])


def _reanchored_scales(panel, steps):
    latent, physical, valid = _targets(panel, slice(None), steps)
    variances = []
    for targets in (latent, physical):
        variances.append(torch.stack([targets[..., h, :][valid[..., h]].var(0, unbiased=False)
                                      for h in range(valid.size(-1))]))
    if not all(torch.isfinite(scale).all() and (scale > 0).all() for scale in variances):
        raise ValueError("Training requires finite positive latent and physical variance for every coordinate")
    return tuple(variances)


def _fan_energy(modes, latent, physical, valid, coordinates, scales):
    """Reduce candidate endpoints within each fixed mode before mixing explanations."""
    w, bias = [value.to(modes) for value in coordinates]
    latent_cost = (modes - latent[..., None, :]).square() / scales[0][:, None, :]
    physical_cost = (modes @ w + bias - physical[..., None, :]).square() / scales[1][:, None, :]
    endpoint_cost = 0.5 * (latent_cost.mean(-1) + physical_cost.mean(-1))
    counts = valid.sum(dim=(-2, -1))
    energy = (endpoint_cost * valid[..., None]).sum(dim=(-3, -2)) / counts[..., None].clamp_min(1)
    return energy, counts > 0


def _use_loss(log_p, energy, valid_roots):
    return -(log_p - energy).logsumexp(-1)[valid_roots].mean()


@torch.no_grad()
def _calibrate_reanchored(model, panel):
    """Calibrate the FULL observation covariance on official mode-0 train residuals."""
    residual = []
    for start in range(0, len(panel.scene_ids), BATCH_SIZE):
        rows = slice(start, start + BATCH_SIZE)
        visible, commands = _factual_inputs(panel, rows)
        actions = _response_controls(model, commands)
        valid = panel.valid[rows].flatten(0, 2)
        for step, kind in CUE_STEPS.items():
            if kind == FULL:
                arrival = step // 5
                active = valid[:, arrival - 1]
                if active.any():
                    prediction = model.predict(visible[active, arrival - 3:arrival],
                                               actions[active, arrival - 3:arrival])[:, -1]
                    residual.append(visible[active, arrival] - prediction)
    if not residual:
        raise ValueError("No fully executed train factual blocks for full-view calibration")
    actual = torch.cat(residual)
    model.observation_model.set_full_reference_sigma(actual.square().mean(0).sqrt())
    return len(actual)


def _reanchored_gradient_check(model, panel, coordinates, scales, steps, *, curve_anchor=True):
    rows = slice(0, 1)
    facts = _reanchored_facts(model, panel, rows, curve_anchor=curve_anchor)
    support = _reanchored_support(model, facts)
    modes, _ = _reanchored_modes(model, panel, rows, steps, curve_anchor=curve_anchor)
    _, log_p = _reanchored_read(modes, support["states"], steps)
    latent, physical, valid = _targets(panel, rows, steps)
    energy, active = _fan_energy(modes, latent, physical, valid, coordinates, scales)
    use = _use_loss(log_p, energy, active)
    p = log_p.detach().double().requires_grad_()
    with torch.no_grad():
        p.copy_(p.log_softmax(-1))
    e = energy.detach().double().requires_grad_()
    dp, de = torch.autograd.grad(_use_loss(p.log_softmax(-1), e, active), (p, e))
    responsibility = (p - e).softmax(-1)
    torch.testing.assert_close(dp[active] * active.sum(), (p.exp() - responsibility)[active])
    torch.testing.assert_close(de[active] * active.sum(), responsibility[active])
    direction = torch.linspace(-1, 1, p.numel(), device=p.device, dtype=p.dtype).reshape_as(p)
    epsilon = 1e-5
    forward = _use_loss((p + epsilon * direction).log_softmax(-1), e, active)
    backward = _use_loss((p - epsilon * direction).log_softmax(-1), e, active)
    finite_diff = (forward - backward) / (2 * epsilon)
    torch.testing.assert_close((dp * direction).sum(), finite_diff, atol=1e-8, rtol=1e-6)

    h2_valid = valid.clone()
    h2_valid[..., :-1] = False
    h2_energy, h2_active = _fan_energy(modes, latent, physical, h2_valid, coordinates, scales)
    h2_use = _use_loss(log_p, h2_energy, h2_active)
    grad_ll, grad_source, grad_modes = torch.autograd.grad(
        h2_use, (facts["log_likelihood"], facts["issued_means"], modes), retain_graph=True,
    )
    offsets = model.response_mixture.response_offsets
    cue_checks = {}
    for step in CUE_STEPS:
        index = step // 5 - FIRST_SOURCE - 1
        likelihood = facts["log_likelihood"][:, index]
        means = facts["issued_means"][:, index]
        movement = support["states"][:, index + 1] - support["states"][:, index]
        source_effect = (grad_ll[:, index].detach() * likelihood).sum()
        through_source = torch.autograd.grad(source_effect, offsets, retain_graph=True)[0]
        cue_checks[str(step)] = {
            "anchor_command_tracks": facts["held"][:, index].sum().item(),
            "actual_scored_tracks": facts["scored"][:, index].sum().item(),
            "max_mode_prediction_difference": (means - means[:, :1]).abs().max().item(),
            "max_log_likelihood_span": (likelihood - likelihood[:, :1]).abs().max().item(),
            "max_log_support_change": movement.abs().max().item(),
            "h2_source_likelihood_to_offsets_grad_norm": through_source.norm().item(),
        }
        if curve_anchor and facts["held"][:, index].all():
            torch.testing.assert_close(means, means[:, :1].expand_as(means), atol=0, rtol=0)
            torch.testing.assert_close(likelihood, likelihood[:, :1].expand_as(likelihood), atol=0, rtol=0)
            torch.testing.assert_close(movement, torch.zeros_like(movement), atol=0, rtol=0)
            torch.testing.assert_close(through_source, torch.zeros_like(through_source), atol=0, rtol=0)
    grad_b = torch.autograd.grad(support["nll"].mean() + use, offsets)[0]
    late = 110 // 5 - FIRST_SOURCE - 1
    results = {"fan_objective_finite_difference_error": ((dp * direction).sum() - finite_diff).abs().item(),
               "h2_to_step110_log_likelihood": grad_ll[:, late].norm().item(),
               "h2_to_step110_issued_means": grad_source[:, late].norm().item(),
               "h2_to_native_predictions": grad_modes[..., -1, :, :].norm().item(),
               "total_loss_to_full_width_offsets": grad_b.norm().item()}
    if not all(np.isfinite(value) for value in results.values()):
        raise RuntimeError(f"Nonfinite gradient on a supervised path: {results}")
    for key in ("h2_to_native_predictions", "total_loss_to_full_width_offsets"):
        if results[key] <= 0:
            raise RuntimeError(f"Missing gradient on a supervised path: {results}")
    _, _, candidates = _query_inputs(panel, rows, steps)
    hold_through_horizon = (candidates == model.normalized_response_hold).all(-1).all(-1)
    held_modes = modes.flatten(0, 3)[hold_through_horizon]
    hold_difference = (held_modes - held_modes[..., :1, :]).abs().max().item()
    if curve_anchor and hold_difference != 0:
        raise RuntimeError(f"The hold-only candidate diverged across modes by {hold_difference}")
    results.update(cues=cue_checks, hold_only_query_fans=int(hold_through_horizon.sum().item()),
                   hold_only_query_max_mode_difference=hold_difference,
                   h2_to_predictions_by_mode=grad_modes[..., -1, :, :].reshape(-1, MODES, modes.size(-1))
                   .square().sum(dim=(0, 2)).sqrt().tolist())
    print(json.dumps(results, indent=2, allow_nan=False), flush=True)
    return results


def _fit_reanchored(model, panel, coordinates, output, args):
    if output.exists():
        raise FileExistsError(output)
    steps = tuple(args.query_loss_steps)
    curve_anchor = args.curve_anchor == "reanchored"
    calibrated = _calibrate_reanchored(model, panel)
    scales = _reanchored_scales(panel, steps)
    gradient_check = _reanchored_gradient_check(model, panel, coordinates, scales, steps, curve_anchor=curve_anchor)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    generator = np.random.default_rng(args.seed)
    logs = []
    for step in range(1, args.updates + 1):
        rows = generator.choice(len(panel.scene_ids), args.batch_size, replace=False)
        facts = _reanchored_facts(model, panel, rows, curve_anchor=curve_anchor)
        support = _reanchored_support(model, facts)
        modes, _ = _reanchored_modes(model, panel, rows, steps, curve_anchor=curve_anchor)
        _, log_p = _reanchored_read(modes, support["states"], steps)
        energy, valid = _fan_energy(modes, *_targets(panel, rows, steps), coordinates, scales)
        obs = support["nll"].mean()
        use = _use_loss(log_p, energy, valid)
        loss = obs + use
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite §17 objective at update {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        logs.append([step, loss.item(), obs.item(), use.item(), gradient.item()])
        if step == 1 or step % 32 == 0 or step == args.updates:
            print(f"Update {step:03d}/{args.updates}: obs={obs.item():.6g} use={use.item():.6g} "
                  f"grad={gradient.item():.6g}", flush=True)

    output.mkdir(parents=True)
    torch.save({"mechanism": "reanchored", "curve_anchor": args.curve_anchor,
                "last_block": model.predictor.transformer.layers[-1].state_dict(),
                "pred_proj": model.pred_proj.state_dict(), "response_mixture": model.response_mixture.state_dict(),
                "observation_model": model.observation_model.state_dict(), "object_coordinates": coordinates},
               output / "model.pt")
    np.savetxt(output / "training_steps.csv", np.asarray(logs), delimiter=",", comments="",
               header="update,loss,observation_nll_per_coordinate,fan_use_loss,gradient_norm")
    report = {"mechanism": "reanchored", "curve_anchor": args.curve_anchor,
              "base_checkpoint": "official LeWM PushT", "modes": MODES, "offset_initialization_std": 0.1,
              "offset_initialization_seed": REANCHORED_SEED,
              "physical_readout_checkpoint": str(WARM_START), "observation_checkpoint": str(OBSERVATION),
              "updates": args.updates, "batch_size": args.batch_size, "seed": args.seed, "learning_rate": 1e-4,
              "gradient_clip": 1.0, "full_view_sigma_factual_transitions": calibrated, "gradient_check": gradient_check,
              "latent_target_variance_by_horizon_and_coordinate": scales[0].tolist(),
              "physical_target_variance_by_horizon_and_coordinate": scales[1].tolist(),
              "scored_arrivals": list(CUE_STEPS), "query_loss_steps": list(steps),
              "unscored_arrival_policy": "public frame refreshes the next window; no likelihood or support change",
              "train_scene_ids": panel.scene_ids.tolist(), "invalid_factual_blocks": (~panel.valid).sum().item(),
              "invalid_query_endpoints": (~panel.query_valid).sum().item()}
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _gradient_check(model, panel, object_coordinates, scales):
    rows = np.arange(BATCH_SIZE)
    trace = trace_factual(model, panel, rows, trace=True)
    prediction, _ = _queries(model, panel, rows, trace["states"])
    evidence, latent, physical = _losses(panel, rows, prediction, trace["nll"], object_coordinates, scales)
    future_h2 = latent[1] + physical[1]
    grad_ll, grad_reset = torch.autograd.grad(future_h2, (trace["log_likelihood"], trace["reset_logit"]),
                                             retain_graph=True)
    source = torch.autograd.grad(evidence, model.observation_model.patch_precision_coordinate, retain_graph=True)[0]
    direct = torch.autograd.grad(physical.mean(), prediction)[0]
    patch_index = 15 // 5 - FIRST_SOURCE - 1
    values = {
        "h2_to_step15_log_likelihood": grad_ll[:, patch_index].norm().item(),
        "h2_to_factual_reset_after_step15": grad_reset[:, patch_index + 1:].norm().item(),
        "evidence_to_patch_precision": source.norm().item(),
        "physical_to_native_predictions": direct.norm().item(),
    }
    if not all(np.isfinite(value) and value > 0 for value in values.values()):
        raise RuntimeError(f"Missing or nonfinite gradient on a supervised path: {values}")
    values["serial_factual_max_log_state_error"] = _check_factual_parity(model, panel, trace, rows)
    print(f"Gradient path check: {values}", flush=True)
    return values


@torch.no_grad()
def _check_factual_parity(model, panel, trace, rows):
    op = model.response_mixture
    visible, commands = _factual_inputs(panel, rows)
    valid = panel.valid[rows].flatten(0, 2)
    controls = model.action_encoder(commands[:1])
    state = op.initial_state(1)
    serial = [state]
    for source in range(FIRST_SOURCE, visible.shape[1] - 1):
        if valid[0, source]:
            _, state, issue = model.predict_response(visible[:1, source - 2:source + 1],
                                                     controls[:, source - 2:source + 1], state, factual=True)
            channel = CUE_STEPS.get(5 * (source + 1))
            if channel is not None:
                center = panel.patch_center[rows[:1]] if channel == PATCH else None
                ll = model.observation_model(issue.means, visible[:1, source + 1], channel, center)
                _, state, _ = op.condition(issue, ll)
        serial.append(state)
    reference = torch.stack(serial, dim=1)
    batched = trace["states"][:1]
    torch.testing.assert_close(batched, reference, atol=2e-3, rtol=2e-4)
    return (batched - reference).abs().max().item()


def _save_trace(model, split, panel, output):
    # Bounded scene batches preserve the exact same computations as fitting.
    parts = [evaluate(model, panel, slice(start, start + BATCH_SIZE))
             for start in range(0, len(panel.scene_ids), BATCH_SIZE)]
    arrays = {key: torch.cat([part[key] for part in parts]).cpu().numpy() for key in parts[0]}
    np.savez_compressed(output / f"{split}_intermediates.npz", scene_ids=panel.scene_ids,
                        query_steps=QUERY_STEPS, factual_arrivals=np.arange(15, 126, 5),
                        scored_arrivals=list(CUE_STEPS), **arrays)


def train(model, train_panel, development, object_coordinates, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    calibrated = calibrate_full(model, train_panel)
    scales = _loss_scales(train_panel)
    gradient_check = _gradient_check(model, train_panel, object_coordinates, scales)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    generator = np.random.default_rng(SEED)
    logs = []
    for step in range(1, UPDATES + 1):
        rows = generator.choice(len(train_panel.scene_ids), BATCH_SIZE, replace=False)
        factual = trace_factual(model, train_panel, rows)
        prediction, _ = _queries(model, train_panel, rows, factual["states"])
        evidence, latent, physical = _losses(train_panel, rows, prediction, factual["nll"], object_coordinates, scales)
        loss = evidence + latent.mean() + physical.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0,
                                                 error_if_nonfinite=True)
        optimizer.step()
        logs.append([loss.item(), evidence.item(), *latent.tolist(), *physical.tolist(), gradient.item()])
        if step == 1 or step % 32 == 0:
            print(f"Update {step:03d}/{UPDATES}: {logs[-1]}", flush=True)

    output.mkdir(parents=True)
    torch.save({"last_block": model.predictor.transformer.layers[-1].state_dict(),
                "pred_proj": model.pred_proj.state_dict(), "response_mixture": model.response_mixture.state_dict(),
                "observation_model": model.observation_model.state_dict(), "object_coordinates": object_coordinates},
               output / "model.pt")
    np.savetxt(output / "training_steps.csv", np.asarray(logs), delimiter=",", comments="",
               header="loss,observation_nll,latent_h1,latent_h2,physical_h1,physical_h2,gradient_norm")
    report = {"updates": UPDATES, "batch_size": BATCH_SIZE, "seed": SEED, "learning_rate": 1e-4,
              "full_view_sigma_factual_transitions": calibrated, "gradient_check": gradient_check,
              "latent_target_variance": scales[0].tolist(), "centered_physical_target_variance": scales[1].tolist(),
              "scored_arrivals": list(CUE_STEPS), "unscored_arrival_policy": "actual frame in window; transition only"}
    for split, panel in (("train", train_panel), ("development", development)):
        _save_trace(model, split, panel, output)
        report[split] = {"scenes": len(panel.scene_ids), "invalid_factual_blocks": (~panel.valid).sum().item(),
                         "invalid_query_endpoints": (~panel.query_valid).sum().item()}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("encode", "check", "fit"))
    parser.add_argument("--mechanism", choices=("legacy", "reanchored"), default="legacy")
    parser.add_argument("--split", choices=("train", "development"))
    parser.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=REANCHORED_UPDATES)
    parser.add_argument("--batch-size", type=int, default=REANCHORED_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=REANCHORED_SEED)
    parser.add_argument("--query-loss-steps", type=int, nargs="+", choices=QUERY_STEPS, default=REANCHORED_LOSS_STEPS)
    parser.add_argument("--curve-anchor", choices=("reanchored", "unanchored"), default="reanchored")
    args = parser.parse_args()
    torch.manual_seed(args.seed if args.mechanism == "reanchored" else SEED)
    if args.action == "encode":
        if args.split is None:
            parser.error("encode needs --split train or development")
        encode_data(args.split, args.archive_dir, args.device)
        return
    if args.action == "fit" and args.output is None:
        parser.error("fit needs --output")
    if args.action == "check" and args.mechanism != "reanchored":
        parser.error("check needs --mechanism reanchored")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    train_panel = load_data("train", args.archive_dir, args.device)
    development = load_data("development", args.archive_dir, args.device)
    if np.intersect1d(train_panel.scene_ids, development.scene_ids).size:
        raise ValueError("Train and development physics panels must be disjoint")
    model, coordinates = make_model(args.device, mechanism=args.mechanism)
    if args.mechanism == "reanchored":
        if args.updates < 1 or not 1 <= args.batch_size <= len(train_panel.scene_ids):
            parser.error("Use a positive update budget and a batch no larger than the train scene panel")
        if len(set(args.query_loss_steps)) != len(args.query_loss_steps):
            parser.error("Specify each query loss step once")
        if args.action == "check":
            calibrated = _calibrate_reanchored(model, train_panel)
            print(f"Calibrated FULL on {calibrated} actual train transitions", flush=True)
            results = _reanchored_gradient_check(model, train_panel, coordinates,
                                                 _reanchored_scales(train_panel, args.query_loss_steps),
                                                 args.query_loss_steps, curve_anchor=args.curve_anchor == "reanchored")
            if args.output is not None:
                report = {"mechanism": args.mechanism, "curve_anchor": args.curve_anchor, "device": args.device,
                          "optimization_steps": 0, "shared_host_initialization": "official LeWM PushT",
                          "nonzero_mode_offsets": {"initialization": "fixed-seed Gaussian", "std": 0.1,
                                                   "seed": REANCHORED_SEED},
                          "full_observation_calibration": {"source": "train split only", "actual_transitions": calibrated},
                          "gradient_check_scene_ids": train_panel.scene_ids[:1].tolist(),
                          "query_steps": list(args.query_loss_steps), "checks": results}
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
            return
        return _fit_reanchored(model, train_panel, coordinates, args.output, args)
    train(model, train_panel, development, coordinates, args.output)


if __name__ == "__main__":
    main()
