"""Compare paired and factual source scoring on the fixed reversible native predictor."""

import json
from dataclasses import dataclass

import numpy as np
import torch

from . import native_response_mixture_regulation as data
from .native_full_depth_conditioning import FullDepthConditioner, advance_log_support, conditioned_rollout
from .native_full_depth_reversible import ASSETS, OUTPUT as MODEL_RUN, SWITCH, action_embeddings, block_windows, encode_rgb, forecast, offsets
from .native_response_mixture_pilot import fit_coordinates


PAIRS = data.RUNS / "native_full_depth_source_pairs_20261115"
OUTPUT = data.RUNS / "native_full_depth_source_evidence_20261115"
MEANINGFUL_GAP = 1.0


def pusher_pixels(rgb):
    """Raw public blue-circle centroid in rendered 512-unit image coordinates."""
    blue = rgb[..., 2].astype(np.int16) - rgb[..., 0] > 115
    count = blue.sum(axis=(-2, -1))
    if np.any(count == 0):
        raise ValueError("The public pusher silhouette must be visible for this score.")
    rows, columns = rgb.shape[-3:-1]
    x = (blue.sum(axis=-2) @ (np.arange(columns) + 0.5)) / count * (512 / columns)
    y = (blue.sum(axis=-1) @ (np.arange(rows) + 0.5)) / count * (512 / rows)
    return np.stack((x, y), axis=-1)


@dataclass(frozen=True)
class SourceScore:
    drift: torch.Tensor
    covariance: torch.Tensor
    only_object: bool = False
    common_pusher: bool = False

    def log_evidence(self, issued, arrived):
        means = issued + self.drift
        if self.common_pusher:
            push = means[..., :2].mean(dim=-2, keepdim=True).expand_as(means[..., :2])
            means = torch.cat((push, means[..., 2:]), dim=-1)
        if self.only_object:
            residual = arrived[..., None, 2:] - means[..., 2:]
            precision = torch.linalg.inv(self.covariance[2:, 2:])
            return -0.5 * ((residual @ precision) * residual).sum(dim=-1)
        residual = arrived[..., None, :] - means
        pusher, block = residual[..., :2], residual[..., 2:]
        pusher_precision = torch.linalg.inv(self.covariance[:2, :2])
        conditional_map = self.covariance[2:, :2] @ pusher_precision
        conditional_covariance = self.covariance[2:, 2:] - conditional_map @ self.covariance[:2, 2:]
        conditional = block - pusher @ conditional_map.T
        log_pusher = -0.5 * ((pusher @ pusher_precision) * pusher).sum(dim=-1)
        log_block = -0.5 * ((conditional @ torch.linalg.inv(conditional_covariance)) * conditional).sum(dim=-1)
        return log_pusher + log_block


def source_scores(issued, observed_pair, object_scale):
    residual = (observed_pair - issued).reshape(-1, 2, 4)
    midpoint = residual.mean(dim=1)
    half_difference = 0.5 * (residual[:, 1] - residual[:, 0])
    drift = midpoint.mean(dim=0)
    centered = midpoint - drift
    common = centered.T @ centered / len(centered)
    difference = half_difference.T @ half_difference / len(half_difference)
    floor = torch.eye(4, dtype=issued.dtype, device=issued.device)
    pooled = common + difference + floor
    flat = (residual - drift).reshape(-1, 4)
    torch.testing.assert_close(pooled, flat.T @ flat / len(flat) + floor)
    paired = common + difference.diag().diag() + floor
    marginal = torch.block_diag(floor[:2, :2], object_scale.double().square().diag())
    return {
        "old_object": SourceScore(torch.zeros_like(drift), marginal, only_object=True),
        "factual_joint": SourceScore(drift, pooled),
        "paired_joint": SourceScore(drift, paired),
        "paired_object": SourceScore(drift, paired, only_object=True),
        "paired_anchor": SourceScore(drift, paired, common_pusher=True),
    }, {"midpoint_covariance": common.tolist(), "contrast_second_moment": difference.tolist(), "drift": drift.tolist()}


def mode_order(actual, other, factual_label):
    zero = torch.where(factual_label[..., None] == 0, actual, other)
    one = torch.where(factual_label[..., None] == 0, other, actual)
    return torch.stack((zero, one), dim=-2)


def observed(rgb, z, w, b):
    pixel_xy = torch.as_tensor(pusher_pixels(rgb), dtype=z.dtype, device=z.device)
    return torch.cat((pixel_xy, z @ w[:, 2:] + b[2:]), dim=-1).double()


def load_frozen(device):
    model = data._native_model(device)
    params = torch.load(MODEL_RUN / "conditioner.pt", map_location=device, weights_only=True)
    adapter = FullDepthConditioner(model, 2).to(device).eval()
    adapter.load_state_dict(params["adapter"])
    hold = action_embeddings(model, torch.zeros((1, 5, 2), device=device))[0]
    train_scenes = json.loads((MODEL_RUN / "training_summary.json").read_text(encoding="utf-8"))["training_scenes"]
    with np.load(ASSETS / "fit.npz", allow_pickle=False) as archive:
        use = np.isin(archive["scene_id"], train_scenes)
        if sorted(archive["scene_id"][use].tolist()) != sorted(train_scenes):
            raise ValueError("Expected the fixed fitting scenes from the fitted conditioner.")
        rgb = archive["rgb"][use]
    encoded = encode_rgb(model, rgb)
    pusher_w, pusher_b = fit_coordinates(encoded, torch.as_tensor(pusher_pixels(rgb), dtype=encoded.dtype, device=device))
    w = torch.cat((pusher_w, params["w"]), dim=1)
    b = torch.cat((pusher_b, params["b"]), dim=0)
    return model, adapter, hold, w, b, params["scale"]


def calibrate(model, adapter, hold, w, b, scale):
    with np.load(PAIRS / "calibration.npz", allow_pickle=False) as archive:
        rgb, alt = archive["rgb"], archive["counterfactual_rgb"]
        z = encode_rgb(model, rgb)
        alt_z = encode_rgb(model, alt)
        actions = torch.as_tensor(archive["actions"], device=z.device, dtype=z.dtype)
        panel = {"z": z, "a": action_embeddings(model, actions)}
        issued = torch.stack([forecast(model, adapter, panel, slice(None), t, 1, hold).squeeze(-2)
                              for t in range(actions.size(2))], dim=3)
        issued = (issued @ w + b).double()
        factual = observed(rgb[..., 1:, :, :, :], z[..., 1:, :], w, b)
        counterpart = observed(alt, alt_z, w, b)
        labels = torch.arange(2, device=z.device)[None, :, None, None].expand_as(factual[..., 0])
        pair = mode_order(factual, counterpart, labels)
        physical = archive["state"][..., 1:, :2] - archive["counterfactual_state"][..., :2]
        scene_ids = archive["scene_id"].tolist()
    scores, metrics = source_scores(issued, pair, scale)
    metrics.update({"scene_ids": scene_ids, "pairs": int(labels.numel()),
                    "physical_pusher_max_abs_contrast": float(np.abs(physical).max()),
                    "pixel_pusher_max_abs_contrast": (pair[..., 1, :2] - pair[..., 0, :2]).abs().max().item(),
                    "native_pusher_max_abs_contrast": (issued[..., 1, :2] - issued[..., 0, :2]).abs().max().item()})
    for name, score in scores.items():
        if not score.common_pusher and not score.only_object:
            diff = pair[..., 0, None, :] - issued - score.drift
            dense = -0.5 * ((diff @ torch.linalg.inv(score.covariance)) * diff).sum(dim=-1)
            torch.testing.assert_close(dense, score.log_evidence(issued, pair[..., 0, :]))
    metrics["score_models"] = {name: {"covariance": score.covariance.tolist()} for name, score in scores.items()}
    return scores, metrics


def evaluate(model, adapter, hold, w, b, scores, archive_path, *, single=False, reference=None):
    with np.load(archive_path, allow_pickle=False) as archive:
        rgb = archive["rgb"][None] if single else archive["rgb"]
        alt = archive["opposite_rgb"][None] if single else archive["counterfactual_rgb"]
        true_state = archive["state"][None, 1:] if single else archive["state"][:, 1:]
        alternate = archive["opposite_state"][None] if single else archive["counterfactual_state"]
        commands = archive["action"][None] if single else archive["actions"]
        phase, mode = archive["phase"].tolist(), archive["mode"].tolist()
        ids = ["historical_lifecycle"] if single else archive["scene_id"].tolist()
    z, alt_z = encode_rgb(model, rgb), encode_rgb(model, alt)
    actions = action_embeddings(model, torch.as_tensor(commands, device=z.device, dtype=z.dtype))
    labels = torch.as_tensor(mode, device=z.device).expand(z.size(0), -1)
    branch_offsets = offsets(adapter, z.size(0))
    native_means = []
    for t in range(actions.size(1)):
        history, past = block_windows(z, actions, t, hold)
        issued = conditioned_rollout(model, history, past, actions[:, None, t:t + 1], branch_offsets)[:, 0, :, 0]
        native_means.append(issued @ w + b)
    issued = torch.stack(native_means, dim=1).double()
    actual = observed(rgb[:, 1:], z[:, 1:], w, b)
    opposite = observed(alt, alt_z, w, b)
    pair = mode_order(actual, opposite, labels)
    if reference is not None:
        expected = torch.tensor([row["log_likelihood_difference"] for row in reference], device=z.device, dtype=torch.float64)
        old = scores["old_object"].log_evidence(issued, actual).diff(dim=-1).flatten()
        torch.testing.assert_close(old, expected, atol=2e-4, rtol=1e-4)
    physical_gap = torch.as_tensor(np.linalg.norm(true_state[..., 2:4] - alternate[..., 2:4], axis=-1), device=z.device)
    active = physical_gap >= MEANINGFUL_GAP
    result = {}
    for name, model_score in scores.items():
        log_at_zero = model_score.log_evidence(issued, pair[..., 0, :])
        log_at_one = model_score.log_evidence(issued, pair[..., 1, :])
        evidence_zero = log_at_zero.diff(dim=-1).squeeze(-1)
        evidence_one = log_at_one.diff(dim=-1).squeeze(-1)
        center = 0.5 * (evidence_one + evidence_zero)
        oriented = 0.5 * (evidence_one - evidence_zero)
        log_likelihood = torch.where(labels[..., None] == 1, log_at_one, log_at_zero)
        requested = model_score.log_evidence(issued, actual)
        torch.testing.assert_close(requested, log_likelihood)
        per_pair_nll = 0.5 * (torch.nn.functional.softplus(evidence_zero) + torch.nn.functional.softplus(-evidence_one))
        both_correct = oriented > center.abs()
        state = log_likelihood.new_full((len(ids), 2), -np.log(2))
        priors, posterior = [], []
        for t in range(len(phase)):
            prior = advance_log_support(state, SWITCH)
            state = (prior + log_likelihood[:, t]).log_softmax(-1)
            priors.append(prior.diff(dim=-1).flatten())
            posterior.append(state.exp()[:, 1])
        p = torch.stack(posterior, dim=-1)
        priors = torch.stack(priors, dim=-1)
        matched = (p > 0.5) == labels.bool()
        result[name] = {
            "paired_nll_all": per_pair_nll.mean().item(),
            "paired_nll_active": per_pair_nll[active].mean().item(),
            "both_evidence_signs_correct_active": int((both_correct & active).sum()),
            "active_events": int(active.sum()),
            "midpoint_abs_active": center[active].abs().mean().item(),
            "aligned_half_gap_active": oriented[active].mean().item(),
            "neutral_abs_evidence": log_likelihood.diff(dim=-1).squeeze(-1)[~active].abs().mean().item(),
            "correct_by_phase": matched.sum(0).tolist(),
            "correct_active_by_phase": (matched & active).sum(0).tolist(),
            "per_scene": {str(scene): {"midpoint": center[i].tolist(), "oriented_half_gap": oriented[i].tolist(),
                                       "prior_log_odds": priors[i].tolist(), "posterior_braked": p[i].tolist(),
                                       "paired_nll": per_pair_nll[i].tolist()}
                          for i, scene in enumerate(ids)},
        }
    return {"scenes": ids, "phases": phase, "true_modes": mode, "active_events_by_phase": active.sum(0).tolist(),
            "physical_gap_by_scene": dict(zip(map(str, ids), physical_gap.tolist(), strict=True)),
            "physical_pusher_max_abs_contrast": float(np.abs(true_state[..., :2] - alternate[..., :2]).max()),
            "pixel_pusher_max_abs_contrast": (pair[..., 1, :2] - pair[..., 0, :2]).abs().max().item(),
            "scores": result}


def decompose_switches(evaluated):
    """Separate actual prediction changes from correct labels inherited by chance."""
    modes = torch.tensor(evaluated["true_modes"], dtype=torch.long)
    switches = (torch.diff(modes) != 0).nonzero().flatten() + 1
    if len(switches) != 2:
        raise ValueError("Expected the fixed two-reversal protocol.")
    signed = 2 * modes - 1
    ids = list(map(str, evaluated["scenes"]))
    physical = torch.tensor([evaluated["physical_gap_by_scene"][scene] for scene in ids])
    active = physical[:, switches] >= MEANINGFUL_GAP

    def changes(p):
        correct = (p - 0.5) * signed > 0
        actual_changes = correct[:, switches - 1] & correct[:, switches]
        return {
            "active_reversals": active.sum(dim=0).tolist(),
            "correct_label_after": (correct[:, switches] & active).sum(dim=0).tolist(),
            "correct_changes_active": (actual_changes & active).sum(dim=0).tolist(),
            "both_active_scenes": int(active.all(dim=1).sum()),
            "both_changes_active": int((actual_changes & active).all(dim=1).sum()),
            "paired_changes_by_scene": dict(zip(ids, (actual_changes & active).tolist(), strict=True)),
        }

    out = {}
    for name, summary in evaluated["scores"].items():
        records = [summary["per_scene"][i] for i in ids]
        actual = torch.tensor([record["posterior_braked"] for record in records], dtype=torch.float64)
        contrast = torch.tensor([record["oriented_half_gap"] for record in records], dtype=torch.float64)
        midpoint = torch.tensor([record["midpoint"] for record in records], dtype=torch.float64)
        prior = torch.tensor([record["prior_log_odds"] for record in records], dtype=torch.float64)
        state = contrast.new_full((len(ids), 2), -np.log(2))
        oracle = []
        for t in range(len(modes)):
            advance = advance_log_support(state, SWITCH)
            evidence = signed[t] * contrast[:, t]
            state = (advance + torch.stack((torch.zeros_like(evidence), evidence), dim=-1)).log_softmax(dim=-1)
            oracle.append(state[:, 1].exp())
        at_switch = contrast[:, switches]
        needs = (-prior[:, switches] * signed[switches]).clamp_min(0)
        took_correct_path = (actual - 0.5) * signed > 0
        changed_and_aligned = took_correct_path[:, switches - 1] & took_correct_path[:, switches]
        changed_and_aligned &= active & (at_switch > 0)
        out[name] = {
            "actual": changes(actual),
            "actual_correct_changes_with_oriented_evidence": changed_and_aligned.sum(dim=0).tolist(),
            "actual_both_changes_with_oriented_evidence": int(changed_and_aligned.all(dim=1).sum()),
            "offline_remove_paired_midpoint_recursively": changes(torch.stack(oracle, dim=-1)),
            "can_clear_actual_prior_using_only_true_difference":
                ((at_switch > needs) & active).sum(dim=0).tolist(),
            "mean_true_difference_active": [at_switch[:, t][active[:, t]].mean().item() for t in range(2)],
            "mean_true_signed_midpoint_active": [
                (midpoint[:, at] * signed[at])[active[:, t]].mean().item() for t, at in enumerate(switches)
            ],
        }
    return out


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frozen = load_frozen(device)
    scores, calibration = calibrate(*frozen)
    reference = json.loads((MODEL_RUN / "summary.json").read_text(encoding="utf-8"))["lifecycle"]
    result = {"calibration": calibration,
              "lifecycle": evaluate(*frozen[:-1], scores, ASSETS / "lifecycle.npz", single=True, reference=reference),
              "heldout_switches": evaluate(*frozen[:-1], scores, PAIRS / "heldout_switches.npz")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    with (OUTPUT / "switch_attribution.json").open("x", encoding="utf-8") as stream:
        json.dump(decompose_switches(result["heldout_switches"]), stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
