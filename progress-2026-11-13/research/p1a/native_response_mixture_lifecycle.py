"""Replay the recorded native PushT retreat, hold and return with fixed §16 weights."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from native_response_mixture_pilot import NORMALIZATION, ROOT, forecasts, issue_event


@torch.inference_mode()
def replay(physics, checkpoint_path, device):
    sys.path[:0] = [str(ROOT / "research_sources_2026-10-22/lewm_assets"), str(ROOT / "oic_step1")]
    from prepare_assets import load_model
    from run_native_pusht_host import encode

    model = load_model(device).eval().requires_grad_(False)
    model.enable_response_mixture().requires_grad_(False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.predictor.transformer.layers[-1].load_state_dict(checkpoint["last_block"])
    model.pred_proj.load_state_dict(checkpoint["pred_proj"])
    model.response_mixture.load_state_dict(checkpoint["response_mixture"])
    model.event_coordinates = checkpoint["event_coordinates"]

    with np.load(physics, allow_pickle=False) as data:
        scene = data["scene_ids"]
        world_count = data["obs_rgb"].shape[1]
        observations = data["observation_steps"]
        roots = data["query_steps"]
        np.testing.assert_array_equal(observations, np.arange(0, 61, 5))
        np.testing.assert_array_equal(roots, [30, 45, 60])
        visible = encode(model, data["obs_rgb"])
        event = torch.as_tensor(data["arrived_pusher"], device=device, dtype=visible.dtype)
        masks = data["mask"][:, 3]
        patch_scene, y, x = np.nonzero(masks)
        np.testing.assert_array_equal(patch_scene, np.repeat(np.arange(len(scene)), 9))
        patch = visible.new_tensor(np.stack((x, y), axis=-1).reshape(len(scene), 9, 2))
        moments = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
        mean, std = [visible.new_tensor(moments[key]) for key in ("mean", "std")]

        def actions(key):
            return ((visible.new_tensor(data[key]) - mean) / std).flatten(-2)

        executed, candidates = [actions(key) for key in ("executed_actions", "query_actions")]
        panel = SimpleNamespace(visible=visible, executed=executed, patch=patch, arrived_pusher=event)
        v, c, issue, ll, skip, write, *_ = issue_event(model, panel, slice(None))
        results = {"scene_id": scene, "roots": roots, "issued_modes": issue.means.cpu().numpy(),
                   "source_ll": ll.unflatten(0, (-1, world_count)).cpu().numpy(),
                   "event": event.cpu().numpy(), "source_window": visible[:, :, :3].cpu().numpy()}
        metadata = ("physical_state", "query_physical_state", "query_actual_step", "actual_step",
                    "query_contacts", "contacts", "query_actions", "query_future_terminated")
        results.update({key: data[key] for key in metadata})
        controls = candidates[:, None].expand(-1, world_count, -1, -1, -1).flatten(0, 1)
        prior, carried, history = skip, write, [write]
        for t in range(3, 12):
            _, write, _ = model.predict_response(v[:, t-2:t+1], model.action_encoder(c[:, t-2:t+1]), write)
            history.append(write)
        results["native_p15_to60"] = torch.stack(history, dim=1).unflatten(0, (-1, world_count)).exp().cpu().numpy()
        results["prior"] = prior.unflatten(0, (-1, world_count)).exp().cpu().numpy()
        for root in roots.tolist():
            pos = root // 5
            trace = {}
            for label, belief in (("skip", prior), ("write", history[pos-3]), ("carry", carried)):
                value = forecasts(model, v[:, pos-2:pos+1], c[:, pos-2:pos], controls, belief,
                                  trace=trace if label == "write" else None)
                results[f"{label}_z{root}"] = value.unflatten(0, (-1, world_count)).cpu().numpy()
            results[f"h1_modes{root}"] = trace["h1_modes"].unflatten(0, (-1, world_count)).cpu().numpy()
    return results, [v.cpu().double().numpy() for v in checkpoint["object_coordinates"]]


def summarize(data, coordinates):
    w, bias = coordinates
    pair_match = (data["source_window"][:, 0::2] == data["source_window"][:, 1::2]).all((-1, -2))
    opposite = data["event"][:, 0::2] != data["event"][:, 1::2]
    scenes, pair = np.nonzero(pair_match & opposite)
    pos = 2 * pair + (data["event"][scenes, 2 * pair] == 0)
    neg = 4 * pair + 1 - pos
    data["pair_scene_id"], data["pair_worlds"] = data["scene_id"][scenes], np.stack((pos, neg), axis=1)
    forward = data["query_actions"][:, 0, 0, 0]
    forward = forward / np.linalg.norm(forward, axis=-1, keepdims=True)
    first, second = np.triu_indices(4, 1)
    results = {}
    for q, root in enumerate(data["roots"]):
        truth = np.take(data["query_physical_state"][:, :, q, :, :, 4:6], [5, 10], axis=3)
        start = data["physical_state"][:, :, root, 4:6]
        valid = np.stack((data["query_actual_step"][:, :, q, :, :5].all(-1),
                          data["query_actual_step"][:, :, q].all(-1)), axis=-1)
        all_executed = valid.all(axis=2)
        no_contact = ~(data["query_contacts"][:, :, q, :, :5] > 0).any(axis=-1)
        h1_same_root_no_contact = (no_contact & valid[..., 0]).all(axis=-1)
        world_still = (np.linalg.norm(truth[..., 0, :] - start[:, :, None], axis=-1) < 1e-8)
        dormant = (no_contact & valid[..., 0] & world_still).all(axis=-1)
        valid_h1_root = all_executed[..., 0]
        # Condition the paired truth measure on fully executed actions in both worlds.
        matched_valid = all_executed[scenes, pos] & all_executed[scenes, neg]
        diff = truth[scenes, pos] - truth[scenes, neg]
        diff -= diff.mean(axis=1, keepdims=True)
        mean_diff_sq = (diff**2).sum(-1).mean(axis=1)
        tv = 0.5 * np.abs(data["native_p15_to60"][:, :, root//5-3] - data["prior"]).sum(-1)
        true_score = np.einsum("swahd,sd->swah", truth, forward)
        true_gap = true_score[:, :, first] - true_score[:, :, second]
        signs = np.where(np.abs(true_gap) > 1e-6, np.sign(true_gap), 0)
        valid_pairs = valid[:, :, first] & valid[:, :, second]
        scored = (signs != 0) & valid_pairs
        reversal = (signs[scenes, pos] * signs[scenes, neg] < 0) & valid_pairs[scenes, pos] & valid_pairs[scenes, neg]
        arms = {}
        skip = data[f"skip_z{root}"].astype(np.float64) @ w
        for name in ("skip", "write", "carry"):
            pred = data[f"{name}_z{root}"].astype(np.float64) @ w
            centered = pred - pred.mean(axis=2, keepdims=True)
            true_center = truth - truth.mean(axis=2, keepdims=True)
            centered_sq = (centered - true_center)**2
            sq = centered_sq.sum(-1).mean(axis=2)
            shift = pred - skip
            shift -= shift.mean(axis=2, keepdims=True)
            score = np.einsum("swahd,sd->swah", pred + bias, forward)
            gap = score[:, :, first] - score[:, :, second]
            predicted_sign = np.where(np.abs(gap) > 1e-6, np.sign(gap), 0)
            correct = predicted_sign == signs
            both = correct[scenes, pos] & correct[scenes, neg] & reversal
            credit = (correct + 0.5 * (predicted_sign == 0)) * scored
            comparisons = scored.sum(axis=(0, 1, 2))
            arms[name] = {
                "centered_xy_rmse": [float(np.sqrt(sq[..., h][all_executed[..., h]].mean()))
                                     if all_executed[..., h].any() else None for h in range(2)],
                "rank_concordance": [float(credit[..., h].sum() / comparisons[h]) if comparisons[h] else None
                                     for h in range(2)],
                "both_correct_reversals": both.sum(axis=(0, 1)).tolist(),
                "centered_shift_rms": np.sqrt((shift**2).sum(-1).mean((0, 1, 2))).tolist(),
                "dormant_h1_centered_shift_rms": float(np.sqrt((shift[..., 0, :]**2).sum(-1)[dormant].mean()))
                if dormant.any() else None,
            }
            data[f"{name}_centered_sq_by_world{root}"] = sq
            data[f"{name}_restore_shift{root}"] = shift
            data[f"{name}_both_correct_reversals{root}"] = both
        data[f"no_movement_no_contact_h1_{root}"] = dormant
        modes = data[f"h1_modes{root}"].astype(np.float64) @ w
        modes -= modes.mean(axis=2, keepdims=True)
        diameter = np.linalg.norm(modes[..., :, None, :] - modes[..., None, :, :], axis=-1).max((-1, -2))
        data[f"mode_diameter{root}"] = diameter
        results[str(root)] = {
            "complete_roots": all_executed.sum(axis=(0, 1)).tolist(),
            "all_candidates_h1_no_contact_roots": int(h1_same_root_no_contact.sum()),
            "all_candidates_h1_no_contact_and_still_roots": int(dormant.sum()),
            "true_qualified_pair_contrast_rms": [float(np.sqrt(mean_diff_sq[..., h][matched_valid[..., h]].mean()))
                                                if matched_valid[..., h].any() else None for h in range(2)],
            "true_reversals": reversal.sum(axis=(0, 1)).tolist(),
            "h1_centered_mode_diameter": float(diameter.mean()),
            "h1_centered_mode_diameter_at_dormant_roots": float(diameter[dormant].mean()) if dormant.any() else None,
            "mean_actual_h1_object_movement_rms": float(np.sqrt(((truth[..., 0, :] - start[:, :, None])**2)
                                                                  .sum(-1)[valid_h1_root].mean()))
            if valid_h1_root.any() else None,
            "native_tv_from_prior": float(tv.mean()),
            "arms": arms,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.physics.parent
    if (root / "prediction_summary.json").exists():
        raise FileExistsError(root / "prediction_summary.json")
    torch.set_num_threads(2)
    records, coordinates = replay(args.physics, args.checkpoint, args.device)
    result = summarize(records, coordinates)
    np.savez_compressed(root / "prediction_intermediates.npz", **records)
    (root / "prediction_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
