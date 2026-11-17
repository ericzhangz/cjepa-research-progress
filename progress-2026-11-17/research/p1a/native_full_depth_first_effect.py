"""Train and evaluate one native first-effect readout on matched physical branches."""

import argparse
import json

import numpy as np
import torch
from torch.nn import functional as F

from . import native_response_mixture_regulation as data
from .native_full_depth_conditioning import (
    FirstEffectReadout, FullDepthConditioner, advance_log_support, conditioned_rollout, write_observation,
)
from .native_full_depth_reversible import ASSETS, OUTPUT as MODEL_RUN, SWITCH, action_embeddings, block_windows, encode_rgb
from .native_full_depth_source_evidence import PAIRS, decompose_switches, mode_order


FIRST_PAIRS = data.RUNS / "native_full_depth_first_effect_pairs_20261115"
OUTPUT = data.RUNS / "native_full_depth_first_effect_20261115"
SEED = 2815
UPDATES = 320


def from_array(value, like):
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


@torch.no_grad()
def fit_examples(model):
    with np.load(FIRST_PAIRS / "fit.npz", allow_pickle=False) as raw:
        rgb = encode_rgb(model, raw["rgb"])
        other = encode_rgb(model, raw["counterfactual_rgb"])
        actual = rgb[..., 1:, :]
        labels = torch.arange(2, device=rgb.device)[None, :, None, None].expand(actual.shape[:-1])
        action = action_embeddings(model, from_array(raw["actions"], rgb))
        xy = from_array(raw["state"][..., 2:4], rgb)
        other_xy = from_array(raw["counterfactual_state"][..., 2:4], rgb)
        return dict(z=rgb, a=action, paired_z=mode_order(actual, other, labels), xy=xy,
                    paired_xy=mode_order(xy[..., 1:, :], other_xy, labels),
                    active=torch.as_tensor(raw["physical_activation"], device=rgb.device),
                    scenes=raw["scene_id"].tolist())


def branch_offsets(adapter, n):
    code = next(adapter.parameters()).new_tensor([[1., -1.], [1., 1.]])
    return adapter(code)[None].expand(n, -1, -1, -1)


def rollout(model, adapter, readout, history, past, actions, *, arm="first_effect", trace=None, compact=False):
    return conditioned_rollout(model, history, past, actions, branch_offsets(adapter, len(history)),
                               first_effect=None if arm == "previous" else readout,
                               compact=compact, trace=trace, forced_exposure=1.0 if arm == "forced" else None)


def decay_free_loss(predicted, target, latents, target_latents, xy_scale, latent_scale):
    midpoint = ((predicted.mean(dim=-3) - target.mean(dim=-3)) / xy_scale).square().mean()
    gap = ((predicted.diff(dim=-3) - target.diff(dim=-3)) / xy_scale).square().mean()
    appearance = ((latents - target_latents) / latent_scale).square().mean()
    return 0.5 * midpoint + gap + 0.5 * appearance


def optimize(model, adapter, readout, fit, hold, w, b):
    values = list(adapter.parameters()) + list(readout.parameters())
    opt = torch.optim.AdamW(values, lr=0.002, weight_decay=0.01)
    rng = np.random.default_rng(SEED)
    lat_scale = fit["z"].std().clamp_min(0.01)
    xy_scale = fit["paired_xy"].diff(dim=-2).square().mean().sqrt().clamp_min(1)
    trace_log = []
    for step in range(UPDATES):
        rows = rng.choice(len(fit["scenes"]), 2, replace=False).tolist()
        t = step % 4
        z, actions = fit["z"][rows], fit["a"][rows]
        history, past = block_windows(z, actions, t, hold)
        past = past[:, None].expand(-1, 2, -1, -1, -1)
        controls = actions[:, None, :, t:t + 1].expand(-1, 2, -1, -1, -1)
        trace = {}
        h1 = rollout(model, adapter, readout, history.flatten(0, 2), past.flatten(0, 2),
                     controls.flatten(0, 2)[:, None], trace=trace)[:, 0].unflatten(0, z.shape[:3])
        issued_logits = trace["logits"][:, 0, 0, 0].unflatten(0, z.shape[:3])
        target_z = fit["paired_z"][rows, ..., t, :, :].unsqueeze(-2)
        target_xy = fit["paired_xy"][rows, ..., t, :, :].unsqueeze(-2)
        gate = F.binary_cross_entropy_with_logits(issued_logits, fit["active"][rows, ..., t].to(issued_logits))
        paired = decay_free_loss(h1 @ w + b, target_xy, h1, target_z, xy_scale, lat_scale)

        initial = z[:, 0, :, 0, :].flatten(0, 1)[:, None, :].expand(-1, 3, -1)
        h2_past = hold.expand(len(initial), 2, -1)
        h2_actions = actions[:, :, :2, :].flatten(0, 1)[:, None]
        h2 = rollout(model, adapter, readout, initial, h2_past, h2_actions)[:, 0].unflatten(0, (len(rows), -1))
        true_z = z[:, :, :, 1:3].transpose(1, 2)
        true_xy = fit["xy"][rows][:, :, :, 1:3].transpose(1, 2)
        native_multistep = decay_free_loss(h2 @ w + b, true_xy, h2, true_z, xy_scale, lat_scale)
        loss = paired + 0.5 * native_multistep + gate
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(values, 5.0)
        opt.step()
        if (step + 1) % 40 == 0:
            trace_log.append([step + 1, paired.item(), native_multistep.item(), gate.item()])
            print(trace_log[-1], flush=True)
    return trace_log


def rmse(x, axes):
    return x.square().sum(dim=-1).mean(dim=axes).sqrt().tolist()


@torch.no_grad()
def heldout_fan(model, adapter, old_adapter, readout, hold, w, b):
    with np.load(ASSETS / "heldout.npz", allow_pickle=False) as raw:
        root = encode_rgb(model, raw["rgb"][:, 0, 0, 0])
        actions = action_embeddings(model, from_array(raw["actions"], root))
        truth = from_array(raw["state"][:, :, :, 1:, 2:4], root).transpose(1, 2)
        scenes = raw["scene_id"].tolist()
    history = root[:, None].expand(-1, 3, -1)
    past = hold.expand(len(root), 2, -1)
    result = {}
    for arm in ("previous", "forced", "first_effect"):
        chosen = old_adapter if arm == "previous" else adapter
        trace = {}
        predictions = rollout(model, chosen, readout, history, past, actions, arm=arm, trace=trace,
                              compact=arm != "previous")
        xyz = predictions @ w + b
        physical_gap = truth.diff(dim=2).squeeze(2)
        native_gap = xyz.diff(dim=2).squeeze(2)
        inner = (physical_gap * native_gap).sum(dim=-1).mean(dim=(0, 1))
        energy = physical_gap.square().sum(-1).mean(dim=(0, 1))
        result[arm] = {"endpoint_rmse": rmse(xyz - truth, (0, 1, 2)),
                       "paired_gap_rmse": rmse(native_gap - physical_gap, (0, 1)),
                       "directed_fraction": (inner / energy).tolist(), "exact_requested_rows": trace.get("rows")}
        if arm == "first_effect":
            dense = rollout(model, chosen, readout, history, past, actions)
            torch.testing.assert_close(dense, predictions, rtol=2e-4, atol=2e-5)
            result[arm]["dense_comparison_max_abs"] = (dense - predictions).abs().max().item()
            result[arm]["zero_both_by_horizon"] = (trace["exposure"] == 0).all(dim=2).sum(dim=(0, 1)).tolist()
    return {"scene_ids": scenes, "root_candidate_queries": int(actions.size(0) * actions.size(1)), "arms": result}


@torch.no_grad()
def heldout_transitions(model, adapter, old_adapter, readout, hold, w, b, scale, *, fresh=False):
    path = FIRST_PAIRS / "fresh_switches.npz" if fresh else PAIRS / "heldout_switches.npz"
    with np.load(path, allow_pickle=False) as raw:
        rgb, other = encode_rgb(model, raw["rgb"]), encode_rgb(model, raw["counterfactual_rgb"])
        actions = action_embeddings(model, from_array(raw["actions"], rgb))
        physical_gap = np.linalg.norm(raw["state"][:, 1:, 2:4] - raw["counterfactual_state"][..., 2:4], axis=-1)
        if fresh:
            activation = raw["physical_activation"]
        else:
            with np.load(FIRST_PAIRS / "evaluation_labels.npz", allow_pickle=False) as measures:
                np.testing.assert_array_equal(raw["scene_id"], measures["heldout_scene_id"])
                np.testing.assert_array_equal(raw["phase"], measures["heldout_phase"])
                np.testing.assert_array_equal(raw["mode"], measures["heldout_mode"])
                activation = measures["heldout_physical_activation"]
        physical = torch.as_tensor(activation, device=rgb.device)
        scenes, phases, modes = raw["scene_id"].tolist(), raw["phase"].tolist(), raw["mode"].tolist()
    actual, alternative = rgb[:, 1:] @ w + b, other @ w + b
    pair = mode_order(actual, alternative, torch.tensor(modes, device=rgb.device).expand(len(scenes), -1))
    calibrated_truth = 0.5 * ((pair.diff(dim=2).squeeze(2) / scale).square().sum(dim=-1))
    scores = {}
    for arm in ("previous", "forced", "first_effect"):
        chosen = old_adapter if arm == "previous" else adapter
        state = actual.new_full((len(scenes), 2), -np.log(2))
        centers, gaps, priors, posterior, exposures, evidence, logit_values, issued_values, requests = (
            [], [], [], [], [], [], [], [], [],
        )
        for t in range(len(phases)):
            history, past = block_windows(rgb, actions, t, hold)
            trace = {}
            issued = rollout(model, chosen, readout, history, past, actions[:, None, t:t + 1],
                             arm=arm, trace=trace, compact=arm != "previous")[:, 0, :, 0] @ w + b
            prior = advance_log_support(state, SWITCH)
            state, ll = write_observation(prior, issued, actual[:, t], scale)
            _, left = write_observation(prior, issued, pair[:, t, 0], scale)
            _, right = write_observation(prior, issued, pair[:, t, 1], scale)
            c, d = left.diff(dim=-1).flatten(), right.diff(dim=-1).flatten()
            active = trace["exposure"][:, 0, 0, 0] if trace else torch.ones_like(c)
            estimate = trace["logits"][:, 0, 0, 0] if trace else torch.ones_like(c)
            if arm == "first_effect":
                torch.testing.assert_close(ll.diff(dim=-1)[active == 0], torch.zeros_like(ll.diff(dim=-1)[active == 0]),
                                           rtol=0, atol=0)
            centers.append((c + d) * 0.5)
            gaps.append((d - c) * 0.5)
            priors.append(prior.diff(dim=-1).flatten())
            posterior.append(state.exp()[:, 1])
            exposures.append(active)
            evidence.append(ll.diff(dim=-1).flatten())
            logit_values.append(estimate)
            issued_values.append(issued)
            requests.append(trace.get("rows"))
        values = {name: torch.stack(value, dim=1) for name, value in (
            ("midpoint", centers), ("oriented_half_gap", gaps), ("prior_log_odds", priors), ("posterior_braked", posterior),
            ("exposure", exposures), ("evidence", evidence), ("logit", logit_values), ("issued_xy", issued_values),
        )}
        on = values["exposure"] > 0
        center, gap = values["midpoint"], values["oriented_half_gap"]
        nll = 0.5 * (F.softplus(center - gap) + F.softplus(-center - gap))
        scores[arm] = {"branch_readout_requests": requests,
                       "physical_false_negative": int((~on & physical).sum()), "physical_false_positive": int((on & ~physical).sum()),
                       "physical_active_count": int(physical.sum()), "physical_inactive_count": int((~physical).sum()),
                       "inactive_abs_evidence_mean": values["evidence"][~physical].abs().mean().item(),
                       "paired_nll_on_physical_active": nll[physical].mean().item(),
                       "paired_nll_all": nll.mean().item(),
                       "per_scene": {str(scene): {name: value[i].tolist() for name, value in values.items()}
                                     for i, scene in enumerate(scenes)}}
    # Counterfactual endpoints are an offline source-information reference only.
    prior = actual.new_full((len(scenes), 2), -np.log(2))
    oracle_prior, oracle_support = [], []
    for t, label in enumerate(modes):
        prior = advance_log_support(prior, SWITCH)
        oracle_prior.append(prior.diff(dim=-1).squeeze(-1))
        increment = (2 * label - 1) * calibrated_truth[:, t]
        prior = (prior + torch.stack((torch.zeros_like(increment), increment), dim=-1)).log_softmax(dim=-1)
        oracle_support.append(prior.exp()[:, 1])
    priors, posterior = torch.stack(oracle_prior, dim=1), torch.stack(oracle_support, dim=1)
    scores["exact_measured_endpoints_offline"] = {"per_scene": {
        str(scene): {"midpoint": torch.zeros_like(calibrated_truth[i]).tolist(),
                     "oriented_half_gap": calibrated_truth[i].tolist(), "prior_log_odds": priors[i].tolist(),
                     "posterior_braked": posterior[i].tolist()}
        for i, scene in enumerate(scenes)
    }}
    result = {"scenes": scenes, "phases": phases, "true_modes": modes,
              "physical_gap_by_scene": dict(zip(map(str, scenes), physical_gap.tolist(), strict=True)),
              "physical_activation_by_scene": dict(zip(map(str, scenes), physical.tolist(), strict=True)),
              "oracle_measured_gap_ideal_evidence_by_scene": dict(zip(map(str, scenes), calibrated_truth.tolist(), strict=True)),
              "scores": scores}
    return result, decompose_switches(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT, type=type(OUTPUT))
    parser.add_argument("--fresh-only", action="store_true")
    args = parser.parse_args()
    output = args.output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = data._native_model(device)
    params = torch.load(MODEL_RUN / "conditioner.pt", map_location=device, weights_only=True)
    adapter = FullDepthConditioner(model, 2).to(device)
    old_adapter = FullDepthConditioner(model, 2).to(device)
    for instance in (adapter, old_adapter):
        instance.load_state_dict(params["adapter"])
    gate = FirstEffectReadout(model).to(device)
    hold = action_embeddings(model, torch.zeros((1, 5, 2), device=device))[0]
    w, b, scale = (params[k] for k in ("w", "b", "scale"))
    if args.fresh_only:
        fixed = torch.load(output / "conditioner.pt", map_location=device, weights_only=True)
        adapter.load_state_dict(fixed["adapter"])
        gate.load_state_dict(fixed["readout"])
        switches, attribution = heldout_transitions(model, adapter, old_adapter, gate, hold, w, b, scale, fresh=True)
        with (output / "fresh_summary.json").open("x", encoding="utf-8") as stream:
            json.dump({"switches": switches, "attribution": attribution}, stream, indent=2, allow_nan=False)
        return
    fit = fit_examples(model)
    print(f"Exact H1 source pairs encoded for {len(fit['scenes'])} scenes", flush=True)
    history = optimize(model, adapter, gate, fit, hold, w, b)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": adapter.state_dict(), "readout": gate.state_dict()}, output / "conditioner.pt")
    fan = heldout_fan(model, adapter, old_adapter, gate, hold, w, b)
    switches, attribution = heldout_transitions(model, adapter, old_adapter, gate, hold, w, b, scale)
    result = {"training_scene_ids": fit["scenes"], "updates": UPDATES, "training_trace": history,
              "fan": fan, "switches": switches, "attribution": attribution}
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
