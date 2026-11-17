"""One PushT development fit of §16 using the arrived public patch at step 15."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "JEPA_OIC_Codex_Handoff_2026-11-03/runs"
NORMALIZATION = ROOT / "oic_step1/runs/pusht_continuation_2026-10-31/action_normalization.json"
TRAIN = ("train_11033026", 96)
DEVELOPMENT = ("confirmation_11034026", 48)
QUERY_STEPS = [10, 15, 20, 30]
SENSOR_RADIUS = 15 * 96 / 512


@dataclass
class Panel:
    scene_id: np.ndarray
    visible: torch.Tensor
    factual_full: torch.Tensor
    future_full: torch.Tensor
    executed: torch.Tensor
    candidates: torch.Tensor
    patch: torch.Tensor
    arrived_pusher: torch.Tensor
    agent_xy: torch.Tensor
    object_xy: torch.Tensor
    same_prefix: np.ndarray
    forward_axis: torch.Tensor


def load_panel(spec, device, normalizer):
    name, count = spec
    cached = f"{name}_native{count}_latents.npz"
    with np.load(ARCHIVE / f"{name}.npz", allow_pickle=False) as data, np.load(
        ARCHIVE / cached, allow_pickle=False
    ) as enc:
        ids = data["scene_ids"]
        np.testing.assert_array_equal(ids, enc["scene_ids"])
        np.testing.assert_array_equal(data["observation_steps"], np.arange(0, 31, 5))
        np.testing.assert_array_equal(data["query_steps"], QUERY_STEPS)
        np.testing.assert_array_equal(data["query_steps"], enc["query_steps"])
        if not data["actual_step"].all() or not data["query_actual_step"].all():
            raise ValueError("Only actually executed observations can enter this pilot")
        obs, full, future = [enc[k] for k in ("obs_latents", "full_latents", "future_latents")]
        np.testing.assert_array_equal(obs.shape, (count, 4, 7, 192))
        np.testing.assert_array_equal(full.shape, obs.shape)
        np.testing.assert_array_equal(future.shape, (count, 4, 4, 4, 2, 192))
        np.testing.assert_array_equal(obs[:, :, [2, 4, 5, 6]],
                                      np.broadcast_to(obs[0, 0, 2], (count, 4, 4, 192)))
        executed, candidates = data["executed_actions"], data["query_actions"]
        np.testing.assert_array_equal(executed[:, 2], candidates[:, 3, 0])
        np.testing.assert_allclose(full[:, :, 3], future[:, :, 0, 3, 0], atol=2e-5, rtol=2e-5)
        physical = data["physical_state"]
        forks = data["query_physical_state"]
        origin = physical[:, :, QUERY_STEPS, None]
        np.testing.assert_array_equal(forks[..., 0, :], np.broadcast_to(origin, forks[..., 0, :].shape))
        mask = data["mask"][:, 3]
        scene, rows, cols = np.nonzero(mask)
        np.testing.assert_array_equal(scene, np.repeat(np.arange(count), 9))
        patch = np.stack((cols, rows), axis=-1).reshape(count, 9, 2)
        frame = data["obs_rgb"][:, :, 3].astype(np.int16)
        seen = (((frame[..., 2] - frame[..., 0]) > 115) & mask[:, None]).any(axis=(-2, -1))
        same_prefix = (obs[:, [0, 2], :3] == obs[:, [1, 3], :3]).all(axis=(-1, -2))

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=device)

        def commands(raw):
            mean, std = normalizer
            return ((tensor(raw) - mean) / std).flatten(-2)

        return Panel(ids, tensor(obs), tensor(full), tensor(future), commands(executed),
                     commands(candidates), tensor(patch), tensor(seen), tensor(physical[:, :, 15, :2]),
                     tensor(np.take(forks[..., 4:6], [5, 10], axis=-2)), same_prefix,
                     F.normalize(tensor(candidates[:, 0, 0, 0]), dim=-1))


def fit_coordinates(x, y):
    x, y = x.flatten(0, -2).double(), y.flatten(0, -2).double()
    xm, ym = x.mean(0), y.mean(0)
    xc = x - xm
    gram = xc.T @ xc
    gram.diagonal().add_(0.01 * gram.diagonal().mean())
    weight = torch.linalg.solve(gram, xc.T @ (y - ym))
    return weight.float(), (ym - xm @ weight).float()


def sensor_logits(full, patch, coordinates):
    w, b = coordinates
    xy = (full @ w + b) * (96 / 512)
    distance = (xy[..., None, :] - patch[:, None, None]).norm(dim=-1).amin(dim=-1)
    return SENSOR_RADIUS - distance


def repeat_worlds(panel, name, rows):
    value = getattr(panel, name)[rows]
    return value[:, None].expand(-1, 4, *value.shape[1:]).flatten(0, 1)


def issue_event(model, panel, rows):
    """Read only the three visible frames available before the step-15 arrival."""
    visible = panel.visible[rows].flatten(0, 1)
    commands = repeat_worlds(panel, "executed", rows)
    op = model.response_mixture
    log_weights = op.initial_state(len(visible))
    _, skipped, issued = model.predict_response(
        visible[:, :3], model.action_encoder(commands[:, :3]), log_weights, factual=True)
    patch = repeat_worlds(panel, "patch", rows)
    logits = sensor_logits(issued.means[:, None], patch, model.event_coordinates)[:, 0]
    event = panel.arrived_pusher[rows].flatten()[:, None].expand_as(logits)
    ll = -F.binary_cross_entropy_with_logits(logits, event, reduction="none")
    posterior, written, evidence = op.condition(issued, ll)
    return visible, commands, issued, ll, skipped, written, posterior, evidence


def subsequent_roots(model, visible, commands, state, *, survival=None):
    roots = {15: state}
    for t in (3, 4, 5):
        _, state, issued = model.predict_response(
            visible[:, t-2:t+1], model.action_encoder(commands[:, t-2:t+1]), state,
            factual=survival is not None)
        if survival is not None:
            survival.append(torch.sigmoid(-issued.reset_logit.detach().double()))
    roots[30] = state
    return roots


def forecasts(model, history, past, candidates, state, *, trace=None):
    count, forks, horizon = candidates.shape[:3]
    windows = history.repeat_interleave(forks, dim=0)
    past = past[:, None].expand(-1, forks, -1, -1)
    controls = torch.cat((past, candidates), dim=2).flatten(0, 1)
    embedding = model.action_encoder(controls)
    belief = state.repeat_interleave(forks, dim=0)
    outputs = []
    for t in range(horizon):
        predicted, belief, issued = model.predict_response(
            windows, embedding[:, t:t+3], belief, factual=trace is not None and t == 0)
        if issued is not None:
            trace["h1_modes"] = issued.means.reshape(count, forks, 3, -1)
        windows = torch.cat((windows[:, 1:], predicted[:, None]), dim=1)
        outputs.append(predicted.reshape(count, forks, -1))
    return torch.stack(outputs, dim=2)


def future_inputs(panel, rows, visible, commands, step):
    t = step // 5
    return visible[:, t-2:t+1], commands[:, t-2:t], repeat_worlds(panel, "candidates", rows)


def matched_worlds(panel, rows):
    visible = panel.visible[rows]
    labels = panel.arrived_pusher[rows].bool()
    distinct = labels[:, 0::2] != labels[:, 1::2]
    same_query = (visible[:, 0::2, 4:7] == visible[:, 1::2, 4:7]).flatten(2).all(dim=-1)
    legal = torch.as_tensor(panel.same_prefix[rows], device=visible.device) & distinct
    if (legal & ~same_query).any():
        raise ValueError("The opposite arrivals must share the step-30 query window")
    scene, pair = legal.nonzero(as_tuple=True)
    positive = 2 * pair + (~labels[scene, 2 * pair]).long()
    negative = 4 * pair + 1 - positive
    return scene, positive, negative


def centered(values):
    return values - values.mean(1, keepdim=True)


def paired_futures(panel, rows, pairs):
    scene, positive, negative = pairs
    truth = panel.future_full[rows, :, QUERY_STEPS.index(30), :, 0].double()
    return centered(truth[scene, positive] - truth[scene, negative])


def paired_positions(panel, rows, pairs):
    scene, positive, negative = pairs
    truth = panel.object_xy[rows, :, QUERY_STEPS.index(30), :, 0].double()
    return truth[scene[:, None], torch.stack((positive, negative), dim=1)]


def paired_modes(model, panel, rows, visible, commands, state, pairs):
    scene, positive, _ = pairs
    source_row = 4 * scene + positive
    trace = {}
    forecasts(model, visible[source_row, 4:7], commands[source_row, 4:6],
              panel.candidates[rows][scene, :, :1], state[source_row], trace=trace)
    return trace["h1_modes"].double()


def contrast(delta, modes):
    return centered((delta[:, None, :, None] * (modes - modes[:, :, 1:2])).sum(dim=2))


def source_targets(panel, rows, pairs, coordinates):
    scene, positive, negative = pairs
    actual = panel.factual_full[rows][scene[:, None], torch.stack((negative, positive), dim=1), 3]
    endpoints = sensor_logits(actual[:, None], panel.patch[rows][scene], coordinates)[:, 0].double()
    prob = endpoints.sigmoid()
    modes = torch.stack((prob[:, 0], prob.mean(1), prob[:, 1]), dim=-1)
    return endpoints[:, 1] - endpoints[:, 0], modes


def offline_states(mode_prob, prior, retained):
    """Two supervision states, ordered by observed arrival 1 then 0."""
    prior = prior.double().softmax(dim=-1).unsqueeze(-2)
    joint = torch.stack((mode_prob, 1 - mode_prob), dim=1) * prior
    posterior = joint / joint.sum(dim=-1, keepdim=True)
    return prior + retained[..., None] * (posterior - prior)


def paired_loss_terms(model, panel, rows, visible, commands, state, issued, survival,
                      physical_weight=None):
    pairs = matched_worlds(panel, rows)
    scene, positive, negative = pairs
    if not len(scene):
        # The ordinary future loss still uses these scenes.
        zero = state.sum() * 0
        return (zero,) if survival is None else (zero, zero)
    modes = paired_modes(model, panel, rows, visible, commands, state, pairs)
    if survival is None:
        probs = state.unflatten(0, (-1, 4)).double().exp()
        prediction = contrast(probs[scene, positive] - probs[scene, negative], modes)
        return (0.25 * F.mse_loss(prediction, paired_futures(panel, rows, pairs)),)

    gap, mode_prob = source_targets(panel, rows, pairs, model.event_coordinates)
    logits = sensor_logits(issued.means[4 * scene + positive, None], panel.patch[rows][scene],
                           model.event_coordinates)[:, 0].double()
    source_loss = 0.25 * F.mse_loss(logits[:, -1] - logits[:, 0], gap)
    if physical_weight is not None:
        probs = state.detach().unflatten(0, (-1, 4)).double().softmax(dim=-1)
        world = torch.stack((positive, negative), dim=1)
        prediction = torch.einsum("pyk,pakd,dc->pyac", probs[scene[:, None], world],
                                  modes, physical_weight)
        error = prediction - paired_positions(panel, rows, pairs)
        return source_loss, (error - error.mean(dim=2, keepdim=True)).square().mean()

    retained = torch.stack(survival).prod(0).view(-1, 4)
    retained = retained[scene[:, None], torch.stack((positive, negative), dim=1)]
    teacher = offline_states(mode_prob, model.response_mixture.log_prior, retained)
    prediction = contrast(teacher[:, 0] - teacher[:, 1], modes)
    return source_loss, 0.25 * F.mse_loss(prediction, paired_futures(panel, rows, pairs))


def fit(model, panel, updates, batch_size, seed, *, paired=False, direct_scale=None,
        physical_weight=None):
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    generator = np.random.default_rng(seed)
    log = []
    for i in range(updates):
        rows = generator.choice(len(panel.scene_id), batch_size, replace=False)
        visible, commands, issued, ll, skipped, written, posterior, evidence = issue_event(model, panel, rows)
        survival = [torch.sigmoid(-issued.reset_logit.detach().double())] if direct_scale is not None else None
        states = subsequent_roots(model, visible, commands, written, survival=survival)
        future_losses = []
        states[10] = issued.log_weights
        for step in (10, 15, 30):
            predictions = forecasts(model, *future_inputs(panel, rows, visible, commands, step), states[step])
            targets = panel.future_full[rows, :, QUERY_STEPS.index(step)].flatten(0, 1)
            future_losses.append(F.mse_loss(predictions, targets))
        prediction_loss = torch.stack(future_losses).mean()
        event_loss = -evidence.mean()
        loss = event_loss + prediction_loss
        optimizer.zero_grad(set_to_none=True)
        if paired or direct_scale is not None:
            terms = paired_loss_terms(model, panel, rows, visible, commands, states[30], issued,
                                      survival, physical_weight)
            extra = terms[0] if direct_scale is None else sum(term / scale for term, scale in zip(terms, direct_scale))
            loss.backward(retain_graph=True)
            split = model.response_mixture.readout
            with torch.no_grad():
                split.weight.grad[0].zero_()
                split.bias.grad[0].zero_()
            extra.backward()
            split_grad = torch.cat((split.weight.grad[0], split.bias.grad[:1])).norm()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0,
                                      error_if_nonfinite=True)
        optimizer.step()
        log.append([event_loss.item(), prediction_loss.item()])
        if paired or direct_scale is not None:
            log[-1].extend([term.item() for term in terms])
            log[-1].append(split_grad.item())
        if i == 0 or (i + 1) % 16 == 0 or i + 1 == updates:
            message = f"update {i+1:03d}: event {log[-1][0]:.5f}, future {log[-1][1]:.5f}"
            if paired or direct_scale is not None:
                message += f", supervision {log[-1][2:-1]}, split grad {log[-1][-1]:.4g}"
            print(message, flush=True)
    return np.asarray(log)


@torch.no_grad()
def evaluate(model, panel, object_coordinates, batch_size, *, direct=False):
    result = {}
    for i in range(0, len(panel.scene_id), batch_size):
        rows = slice(i, i + batch_size)
        visible, commands, issued, ll, skipped, written, posterior, evidence = issue_event(model, panel, rows)
        op = model.response_mixture
        survival = [torch.sigmoid(-issued.reset_logit.double())] if direct else None
        old = subsequent_roots(model, visible, commands, skipped)
        new = subsequent_roots(model, visible, commands, written, survival=survival)
        fields = dict(event_ll=ll, event_posterior=posterior.exp(), event_pre=issued.log_weights.exp(),
                      event_reset=issued.reset_logit.sigmoid(), event_nll=-evidence,
                      issued_latents=issued.means)
        if direct:
            fields["retained30"] = torch.stack(survival).prod(0)
            fields["event_logits"] = sensor_logits(issued.means[:, None], repeat_worlds(panel, "patch", rows),
                                                   model.event_coordinates)[:, 0]
        for step in (15, 30):
            args = future_inputs(panel, rows, visible, commands, step)
            trace = {}
            fields[f"skipped_p{step}"] = old[step].exp()
            fields[f"written_p{step}"] = new[step].exp()
            fields[f"written_z{step}"] = forecasts(model, *args, new[step], trace=trace)
            fields[f"skipped_z{step}"] = forecasts(model, *args, old[step])
            fields[f"h1_modes{step}"] = trace["h1_modes"]
            mode_delta = ((new[step].exp() - old[step].exp())[:, None, :, None]
                          * trace["h1_modes"]).sum(dim=2)
            torch.testing.assert_close(fields[f"written_z{step}"][:, :, 0] -
                                       fields[f"skipped_z{step}"][:, :, 0],
                                       mode_delta, atol=5e-6, rtol=1e-4)
            if step == 15:
                likelihood = (ll - ll.amax(-1, keepdim=True)).exp()
                weight = issued.log_weights.exp()
                mean_likelihood = (weight * likelihood).sum(-1)
                g = trace["h1_modes"] - trace["h1_modes"][:, 3:4]
                centered = g - (weight[:, None, :, None] * g).sum(2, keepdim=True)
                covariance = ((weight * likelihood)[:, None, :, None] * centered).sum(2)
                effect = covariance * (1 - issued.reset_logit.sigmoid())[:, None, None]
                effect = effect / mean_likelihood[:, None, None]
                torch.testing.assert_close(mode_delta - mode_delta[:, 3:4], effect,
                                           atol=5e-6, rtol=1e-4)
                fields["covariance_action_effect15"] = effect
        for name, value in fields.items():
            result.setdefault(name, []).append(value.unflatten(0, (-1, 4)).cpu().numpy())
    result = {name: np.concatenate(parts) for name, parts in result.items()}
    metrics = {}
    w, b = (x.cpu().numpy() for x in object_coordinates)
    for step in (15, 30):
        idx = QUERY_STEPS.index(step)
        truth = panel.object_xy[:, :, idx].cpu().numpy()
        targets = panel.future_full[:, :, idx].cpu().numpy()
        result[f"true_xy{step}"] = truth
        probe_error = (targets @ w + b - truth)
        metrics[str(step)] = {"target_readout_rmse": np.sqrt(np.square(probe_error).sum(-1).mean((0, 1, 2))).tolist()}
        errors = {}
        for arm in ("written", "skipped"):
            z = result[f"{arm}_z{step}"]
            error = z @ w + b - truth
            centered_error = error - error.mean(axis=2, keepdims=True)
            errors[arm] = np.square(centered_error).sum(-1).mean((1, 2))
            metrics[str(step)][arm] = {
                "latent_mse": np.square(z - targets).mean((0, 1, 2, 4)).tolist(),
                "absolute_rmse": np.sqrt(np.square(error).sum(-1).mean((0, 1, 2))).tolist(),
                "action_difference_rmse": np.sqrt(errors[arm].mean(0)).tolist(),
            }
        paired = errors["written"] - errors["skipped"]
        result[f"paired_action_squared_error{step}"] = paired
        rng = np.random.default_rng(1611)
        resamples = rng.integers(0, len(paired), (5000, len(paired)))
        metrics[str(step)]["mean_paired_action_squared_error_change"] = paired.mean(0).tolist()
        metrics[str(step)]["paired_95pct_interval"] = np.quantile(
            paired[resamples].mean(axis=1), [.025, .975], axis=0).T.tolist()
        metrics[str(step)]["improved_scene_count"] = (paired < 0).sum(0).tolist()
    metrics["patch_nll"] = float(result["event_nll"].mean())
    metrics["patch_constant_prior_nll"] = float(np.log(2))
    metrics["posterior_mean_total_variation_t15"] = float(
        (0.5 * np.abs(result["written_p15"] - result["skipped_p15"]).sum(-1)).mean())
    metrics["posterior_mean_total_variation_t30"] = float(
        (0.5 * np.abs(result["written_p30"] - result["skipped_p30"]).sum(-1)).mean())
    return metrics, result


def h1_write_effect(records, step, coordinates):
    """Measure the H1 action effect via mode differences before adding the base forecast."""
    w, bias = coordinates
    modes = records[f"h1_modes{step}"].astype(np.float64)
    state = {}
    for label in ("written", "skipped"):
        weights = records[f"{label}_p{step}"].astype(np.float64)
        state[label] = weights / weights.sum(-1, keepdims=True)
    delta = state["written"] - state["skipped"]
    shift = ((delta[:, :, None, :, None] * (modes - modes[:, :, :, 1:2])).sum(3)) @ w
    baseline = (state["skipped"][:, :, None, :, None] * modes).sum(3) @ w + bias
    error = baseline - records[f"true_xy{step}"][:, :, :, 0]
    centered_shift = shift - shift.mean(2, keepdims=True)
    centered_error = error - error.mean(2, keepdims=True)
    per_scene = ((2 * centered_error + centered_shift) * centered_shift).sum(-1).mean((1, 2))
    return centered_error, centered_shift, per_scene


def paired_diagnostics(panel, records, object_coordinates):
    scene, positive, negative = [x.cpu().numpy() for x in matched_worlds(panel, slice(None))]
    pair_rows = scene[:, None], np.stack((positive, negative), axis=1)
    metric = {}
    values = {"pair_scene_id": panel.scene_id[scene], "pair_worlds": pair_rows[1]}
    w, bias = [x.cpu().numpy().astype(np.float64) for x in object_coordinates]
    direction = panel.forward_axis.cpu().numpy()
    first, second = np.triu_indices(4, 1)

    def center(array):
        return array - array.mean(axis=-3, keepdims=True)

    def ranking(xy):
        scores = np.einsum("swahd,sd->swah", xy, direction)
        differences = scores[:, :, first] - scores[:, :, second]
        signs = np.where(np.abs(differences) > 1e-6, np.sign(differences), 0)
        return scores, signs

    def interval(values):
        rng = np.random.default_rng(1611)
        offsets = rng.integers(0, len(values), (5000, len(values)))
        return np.quantile(values[offsets].mean(1), (.025, .975), axis=0).T.tolist()

    for step in (15, 30):
        truth = records[f"true_xy{step}"].astype(np.float64)
        true_score, true_sign = ranking(truth)
        valid = true_sign != 0
        reversal = (true_sign[pair_rows][:, 0] * true_sign[pair_rows][:, 1]) < 0
        values[f"true_forward_scores{step}"] = true_score
        values[f"true_forward_signs{step}"] = true_sign
        values[f"true_reversal{step}"] = reversal
        metric[str(step)] = {
            "non_tied_true_comparisons": valid.sum((0, 1, 2)).tolist(),
            "true_reversal_count": reversal.sum((0, 1)).tolist(),
        }
        concordance = {}
        for label in ("written", "skipped"):
            xy = records[f"{label}_z{step}"].astype(np.float64) @ w + bias
            score, sign = ranking(xy)
            correct = sign == true_sign
            credit = (correct + 0.5 * (sign == 0)) * valid
            concordance[label] = credit.sum((1, 2)) / valid.sum((1, 2)).clip(min=1)
            both_right = correct[pair_rows].all(axis=1) & reversal
            picked = score.argmax(axis=2)
            realized = np.take_along_axis(true_score, picked[:, :, None], axis=2).squeeze(2)
            regret = true_score.max(axis=2) - realized
            metric[str(step)][label] = {
                "forward_pairwise_concordance": (credit.sum((0, 1, 2)) / valid.sum((0, 1, 2))).tolist(),
                "forward_regret": regret.mean((0, 1)).tolist(),
                "strict_reversals_both_correct": both_right.sum((0, 1)).tolist(),
            }
            values[f"{label}_forward_scores{step}"] = score
            values[f"{label}_forward_signs{step}"] = sign
            values[f"{label}_reversal_correct{step}"] = both_right
        rank_gain = concordance["written"] - concordance["skipped"]
        metric[str(step)]["scene_ranking_gain"] = rank_gain.mean(0).tolist()
        metric[str(step)]["scene_ranking_gain_95pct"] = interval(rank_gain)
        values[f"scene_ranking_gain{step}"] = rank_gain
        base, shift, scene_change = h1_write_effect(records, step, (w, bias))
        metric[str(step)]["h1_mode_decomposed"] = {
            "centered_physical_effect_rms": float(np.sqrt(np.square(shift).sum(-1).mean())),
            "centered_squared_error_change": float(scene_change.mean()),
            "centered_squared_error_change_95pct": interval(scene_change),
            "improved_scenes": int((scene_change < 0).sum()),
        }
        values[f"h1_centered_physical_shift{step}"] = shift
        values[f"h1_mode_decomposed_scene_error_change{step}"] = scene_change
        if step == 30:
            contributions = {}
            for name, reducer in (("midpoint", lambda x: x.mean(axis=1)),
                                  ("contrast", lambda x: (x[:, 0] - x[:, 1]) / 2)):
                a, b = (reducer(x[pair_rows]) for x in (base, shift))
                term = ((2 * a + b) * b).sum(-1).mean(1)
                values[f"h1_centered_error_change_{name}30"] = term
                contributions[name] = float(term.mean())
            metric["30"]["matched_h1_centered_squared_error_change"] = contributions
            actual = center(truth[scene, positive] - truth[scene, negative])[:, :, 0]
            physical = shift[scene, positive] - shift[scene, negative]
            magnitude = np.sqrt(np.square(physical).sum(-1).mean())
            metric["30"]["matched_h1_physical_effect_rms"] = float(magnitude)
            metric["30"]["matched_h1_true_physical_difference_rms"] = float(np.sqrt(np.square(actual).sum(-1).mean()))
            values["decomposed_h1_matched_physical_effect30"] = physical
            mode_xy = records["h1_modes30"][scene, positive].astype(np.float64) @ w
            mode_scores = np.einsum("pakd,pd->pak", mode_xy, direction[scene])
            mode_differences = mode_scores[:, first] - mode_scores[:, second]
            can_reverse = (mode_differences.min(-1) < -1e-6) & (mode_differences.max(-1) > 1e-6)
            metric["30"]["h1_true_reversals_reachable_by_any_weights"] = int((can_reverse & reversal[:, :, 0]).sum())
            values["h1_mode_forward_score_differences30"] = mode_differences
    return metric, values


@torch.no_grad()
def direct_diagnostics(panel, records, coordinates, object_coordinates):
    device = panel.visible.device
    rows = slice(None)
    pairs = matched_worlds(panel, rows)
    scene, positive, negative = pairs
    indices = scene[:, None], torch.stack((positive, negative), dim=1)
    gap, teacher_prob = source_targets(panel, rows, pairs, coordinates)
    retained = torch.as_tensor(records["retained30"], device=device)[indices]
    prior = torch.as_tensor(records["event_pre"], device=device)[scene, positive].double().log()
    teacher = offline_states(teacher_prob, prior, retained)
    deploy = torch.as_tensor(records["written_p30"], device=device)[indices].double()
    deploy = deploy / deploy.sum(dim=-1, keepdim=True)
    logits = torch.as_tensor(records["event_logits"], device=device)[scene, positive].double()
    modes = torch.as_tensor(records["h1_modes30"], device=device)[scene, positive].double()
    target = paired_futures(panel, rows, pairs)
    deltas = [state[:, 0] - state[:, 1] for state in (teacher, deploy)]
    teacher_response, deploy_response = [contrast(d, modes) for d in deltas]
    gap_error = logits[:, 2] - logits[:, 0] - gap
    object_weight = object_coordinates[0].double()
    position = torch.einsum("pyk,pakd,dc->pyac", teacher, modes, object_weight)
    physical_target = panel.object_xy[indices[0], indices[1], QUERY_STEPS.index(30), :, 0].double()
    physical_pair_target = centered(physical_target[:, 0] - physical_target[:, 1])
    axis = panel.forward_axis[scene].double()
    candidate_axes = torch.triu_indices(4, 4, offset=1, device=device)

    def signs(xy):
        score = torch.einsum("pyac,pc->pya", xy, axis)
        d = score[:, :, candidate_axes[0]] - score[:, :, candidate_axes[1]]
        return torch.where(d.abs() > 1e-6, d.sign(), 0.0)

    actual_signs, teacher_signs = signs(physical_target), signs(position)
    reversals = actual_signs[:, 0] * actual_signs[:, 1] < 0

    def rms(x):
        return x.square().sum(-1).mean().sqrt().item()

    def mse(x):
        return x.square().mean().item()

    teacher_error, deploy_error = [response - target for response in (teacher_response, deploy_response)]
    result = {
        "pairs": len(scene),
        "source_target_gap_mean": gap.mean().item(),
        "source_predicted_signed_gap_mean": (logits[:, 2] - logits[:, 0]).mean().item(),
        "source_gap_rmse": gap_error.square().mean().sqrt().item(),
        "source_gap_positive_count": int((logits[:, 2] > logits[:, 0]).sum()),
        "mean_teacher_odd_weight": (deltas[0][:, 2] - deltas[0][:, 0]).mean().item(),
        "mean_deploy_odd_weight": (deltas[1][:, 2] - deltas[1][:, 0]).mean().item(),
        "mean_deploy_even_weight_abs": deltas[1][:, 1].abs().mean().item(),
        "latent_baseline_mse": mse(target),
        "latent_teacher_contrast_mse": mse(teacher_error),
        "latent_deploy_contrast_mse": mse(deploy_error),
        "physical_target_rms": rms(physical_pair_target),
        "physical_teacher_response_rms": rms(teacher_response @ object_weight),
        "physical_deploy_response_rms": rms(deploy_response @ object_weight),
        "physical_teacher_contrast_error_rms": rms(teacher_response @ object_weight - physical_pair_target),
        "physical_deploy_contrast_error_rms": rms(deploy_response @ object_weight - physical_pair_target),
        "teacher_both_correct_reversals": int(((actual_signs == teacher_signs).all(dim=1) & reversals).sum()),
        "true_reversal_count": int(reversals.sum()),
    }
    trace = {"offline_source_probabilities": teacher_prob, "matched_deploy_source_logits": logits,
             "offline_p30": teacher, "matched_deploy_p30": deploy, "offline_latent_contrast30": teacher_response,
             "matched_deploy_latent_contrast30": deploy_response, "matched_target_latent_contrast30": target,
             "offline_forward_signs30": teacher_signs}
    return result, {key: value.cpu().numpy() for key, value in trace.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1611)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--paired-from", type=Path, help="Resume the single-unit fit with paired supervision")
    resume.add_argument("--direct-from", type=Path, help="Resume with separate §16.15 source and use supervision")
    resume.add_argument("--physical-from", type=Path, help="Resume with §16.17 actual-write physical supervision")
    args = parser.parse_args()
    if args.updates < 1 or not 1 <= args.batch_size <= 48:
        parser.error("Use a positive fit budget and a batch of 1 through 48 scenes")
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    sys.path.insert(0, str(ROOT / "research_sources_2026-10-22/lewm_assets"))
    from prepare_assets import load_model

    with NORMALIZATION.open(encoding="utf-8") as stream:
        moments = json.load(stream)
    normalizer = [torch.tensor(moments[k], device=args.device) for k in ("mean", "std")]
    train, dev = [load_panel(spec, args.device, normalizer) for spec in (TRAIN, DEVELOPMENT)]
    if np.intersect1d(train.scene_id, dev.scene_id).size:
        raise ValueError("The scene splits must be disjoint")
    model = load_model(args.device).eval().requires_grad_(False)
    model.enable_response_mixture()
    model.predictor.transformer.layers[-1].requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    initial = args.paired_from or args.direct_from or args.physical_from
    direct = bool(args.direct_from or args.physical_from)
    direct_scale = None
    physical_weight = None
    if initial:
        checkpoint = torch.load(initial / "pilot.pt", map_location=args.device, weights_only=True)
        model.predictor.transformer.layers[-1].load_state_dict(checkpoint["last_block"])
        model.pred_proj.load_state_dict(checkpoint["pred_proj"])
        model.response_mixture.load_state_dict(checkpoint["response_mixture"])
        model.event_coordinates = checkpoint["event_coordinates"]
        object_coordinates = checkpoint["object_coordinates"]
        split = model.response_mixture.readout
        with torch.no_grad():
            if direct:
                all_rows = slice(None)
                pairs = matched_worlds(train, all_rows)
                gap, _ = source_targets(train, all_rows, pairs, model.event_coordinates)
                if args.physical_from:
                    physical_weight = object_coordinates[0].double()
                    truth = paired_positions(train, all_rows, pairs)
                    target_loss_scale = (truth - truth.mean(dim=2, keepdim=True)).square().mean().item()
                else:
                    split.weight[0].zero_()
                    split.bias[0].zero_()
                    target = paired_futures(train, all_rows, pairs)
                    target_loss_scale = 0.25 * target.square().mean().item()
                direct_scale = (0.25 * gap.square().mean().item(), target_loss_scale)
                if not all(np.isfinite(scale) and scale > 0 for scale in direct_scale):
                    raise ValueError("Direct supervision needs two positive training-set loss scales")
            else:
                direction_file = ROOT / "runs/native_response_mixture_curvature_20261113/directions.npz"
                with np.load(direction_file, allow_pickle=False) as directions:
                    d = torch.as_tensor(directions["train_latent_min_eigen"], device=args.device,
                                        dtype=split.weight.dtype)
                split.weight[0].copy_(0.01 * d[:-1])
                split.bias[0].copy_(0.01 * d[-1])
    else:
        model.event_coordinates = fit_coordinates(train.factual_full[:, :, 3], train.agent_xy)
        object_coordinates = fit_coordinates(train.future_full, train.object_xy)
    with torch.no_grad():
        before, initial_records = evaluate(model, dev, object_coordinates, args.batch_size, direct=direct)
    if initial:
        before["fixed_candidate_diagnostics"], _ = paired_diagnostics(dev, initial_records, object_coordinates)
    if direct:
        before["offline_and_deploy"], _ = direct_diagnostics(dev, initial_records, model.event_coordinates,
                                                             object_coordinates)
    calibration = {}
    for name, panel in (("train", train), ("historical_development", dev)):
        logits = sensor_logits(panel.factual_full[:, :, 3, None], panel.patch, model.event_coordinates)[:, :, 0]
        calibration[name] = {
            "scenes": len(panel.scene_id), "matched_prefix_pairs": int(panel.same_prefix.sum()),
            "true_full_latent_patch_accuracy": ((logits > 0) == panel.arrived_pusher.bool()).float().mean().item(),
        }
    print("observation alignment", json.dumps(calibration), flush=True)
    if args.physical_from:
        before_train, rows = evaluate(model, train, object_coordinates, args.batch_size, direct=True)
        before_train["fixed_candidate_diagnostics"], _ = paired_diagnostics(train, rows, object_coordinates)
        before_train["offline_and_deploy"], _ = direct_diagnostics(train, rows, model.event_coordinates,
                                                                   object_coordinates)
    loss = fit(model, train, args.updates, args.batch_size, args.seed, paired=bool(args.paired_from),
               direct_scale=direct_scale, physical_weight=physical_weight)
    model.requires_grad_(False)
    metrics, records = evaluate(model, dev, object_coordinates, args.batch_size, direct=direct)
    if initial:
        metrics["fixed_candidate_diagnostics"], paired_records = paired_diagnostics(dev, records, object_coordinates)
        records.update(paired_records)
    if direct:
        metrics["offline_and_deploy"], direct_records = direct_diagnostics(dev, records, model.event_coordinates,
                                                                          object_coordinates)
        records.update(direct_records)
    if args.physical_from:
        after_train, train_records = evaluate(model, train, object_coordinates, args.batch_size, direct=True)
        after_train["fixed_candidate_diagnostics"], rows = paired_diagnostics(train, train_records, object_coordinates)
        train_records.update(rows)
        after_train["offline_and_deploy"], rows = direct_diagnostics(train, train_records, model.event_coordinates,
                                                                     object_coordinates)
        train_records.update(rows)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "intermediates.npz", loss=loss, scene_id=dev.scene_id,
                        same_prefix=dev.same_prefix, arrived_pusher=dev.arrived_pusher.cpu().numpy(),
                        **records)
    if args.physical_from:
        np.savez_compressed(args.output / "train_intermediates.npz", scene_id=train.scene_id, **train_records)
    torch.save({
        "last_block": model.predictor.transformer.layers[-1].state_dict(),
        "pred_proj": model.pred_proj.state_dict(),
        "response_mixture": model.response_mixture.state_dict(),
        "event_coordinates": model.event_coordinates,
        "object_coordinates": object_coordinates,
    }, args.output / "pilot.pt")
    summary = {"updates": args.updates, "batch_scenes": args.batch_size, "seed": args.seed,
               "learning_rate": 1e-4, "device": args.device, "observation_alignment": calibration,
               "before_training": before, "metrics": metrics}
    if args.paired_from:
        summary["loss_columns"] = ["event_nll", "future_mse", "matched_loss", "matched_split_gradient_norm"]
    if direct:
        use = "actual_write_centered_position_mse" if args.physical_from else "use_loss"
        summary["loss_columns"] = ["event_nll", "future_mse", "source_gap_loss", use, "direct_split_gradient_norm"]
        summary["direct_zero_response_train_loss"] = direct_scale
    if args.physical_from:
        summary["fit_before_training"] = before_train
        summary["fit_after_training"] = after_train
    if initial:
        summary["initialized_from"] = str(initial)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
