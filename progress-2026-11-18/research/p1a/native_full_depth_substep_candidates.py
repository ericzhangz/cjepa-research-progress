"""Attribute downstream candidate rankings to the actual temporal source updates."""

import argparse
import json

import numpy as np
import torch

from .native_full_depth_first_effect import rollout
from .native_full_depth_reversible import action_embeddings, block_windows, encode_rgb
from .native_full_depth_substep import OUTPUT, PAIRS, load_core


def candidate_metrics(goal_score, coverage, xy, true_xy):
    # Axes before the coordinates: scene, physical query, candidate, horizon.
    rank = torch.triu_indices(goal_score.size(2), goal_score.size(2), offset=1, device=goal_score.device)
    lhs, rhs = rank
    truth_pair = coverage[:, :, lhs] - coverage[:, :, rhs]
    predicted_pair = goal_score[:, :, rhs] - goal_score[:, :, lhs]
    valid = truth_pair.abs() > 1e-4
    direction = (truth_pair * predicted_pair).sign()
    agreement = (direction + 1) * 0.5
    selected = goal_score.argmin(dim=2, keepdim=True)
    attained = coverage.gather(dim=2, index=selected).squeeze(2)
    regret = coverage.max(dim=2).values - attained
    absolute = (xy - true_xy).square().sum(-1)
    relative = ((xy - xy[:, :, 3:4]) - (true_xy - true_xy[:, :, 3:4])).square().sum(-1)
    return {"rank_agreement": ((agreement * valid).sum(dim=(0, 1, 2)) / valid.sum(dim=(0, 1, 2))).tolist(),
            "pair_counts": valid.sum(dim=(0, 1, 2)).tolist(), "regret": regret.mean(dim=(0, 1)).tolist(),
            "absolute_xy_rmse": absolute.mean(dim=(0, 1, 2)).sqrt().tolist(),
            "hold_relative_rmse": relative[:, :, [0, 1, 2, 4, 5]].mean(dim=(0, 1, 2)).sqrt().tolist(),
            "selected_candidate": selected.squeeze(2).tolist(), "per_root_regret": regret.tolist(),
            "per_root_rank_agreement": ((agreement * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1)).tolist(),
            "per_root_pair_counts": valid.sum(dim=2).tolist()}


def diagnose_readout(model, device):
    previous = json.loads((OUTPUT / "candidate_attribution.json").read_text(encoding="utf-8"))
    per_mode = torch.as_tensor(previous["per_mode_native_latent_cost"], device=device, dtype=torch.double)
    truth = torch.as_tensor(previous["truth_coverage"], device=device, dtype=torch.double)
    indices = torch.triu_indices(per_mode.size(2), per_mode.size(2), offset=1, device=device)
    differences = per_mode[:, :, indices[0]] - per_mode[:, :, indices[1]]
    overturn = (differences[:, :, :, 0] * differences[:, :, :, 1]) < 0
    non_ties = (truth[:, :, indices[0]] - truth[:, :, indices[1]]).abs() > 1e-4
    choices = per_mode.argmin(dim=2)
    prior = torch.tensor(previous["arms"]["matched_endpoint"]["selected_candidate"])
    temporal = torch.tensor(previous["arms"]["native_path"]["selected_candidate"])
    horizon = [0, 1, 3]
    with np.load(PAIRS / "post_switch_candidates.npz", allow_pickle=False) as forks:
        np.testing.assert_array_equal(forks["scene_id"], previous["scene_ids"])
        labels = encode_rgb(model, forks["rgb"][..., horizon, :, :, :])
        targets = encode_rgb(model, forks["goal_rgb"])
        correct_xy = torch.as_tensor(forks["state"][..., 2:4], device=device).double()[..., horizon, :]
    perfect_latent = (labels - targets[:, None, None, None, :]).square().sum(-1).double()
    exact_cost = candidate_metrics(perfect_latent, truth, correct_xy, correct_xy)
    return {"horizon_raw_steps": previous["horizon_raw_steps"],
            "non_tied_physical_pairs_switchable_by_support": (overturn & non_ties).sum(dim=(0, 1, 2)).tolist(),
            "total_non_tied_physical_pairs": non_ties.sum(dim=(0, 1, 2)).tolist(),
            "roots_different_native_best_by_perfect_mode": (choices[:, :, 0] != choices[:, :, 1]).sum(dim=(0, 1)).tolist(),
            "roots": len(previous["scene_ids"]) * len(previous["physical_query_modes"]),
            "roots_endpoint_and_temporal_select_different_candidate": (prior != temporal).sum(dim=(0, 1)).tolist(),
            "actual_endpoint_rgb_native_goal": {k: exact_cost[k] for k in ("rank_agreement", "regret", "per_root_regret", "per_root_rank_agreement")}}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnose-readout", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, adapter, gate, hold, w, b, _ = load_core(device)
    if args.diagnose_readout:
        result = diagnose_readout(model, device)
        with (OUTPUT / "readout_attribution.json").open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
        return
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))["scores"]
    horizon = [0, 1, 3]
    with np.load(PAIRS / "fresh_switches.npz", allow_pickle=False) as actual, np.load(
        PAIRS / "post_switch_candidates.npz", allow_pickle=False,
    ) as forks:
        scenes, phases, modes = forks["scene_id"].tolist(), forks["phase"].tolist(), forks["mode"].tolist()
        if scenes != actual["scene_id"].tolist() or scenes != summary["scenes"]:
            raise ValueError("Candidate roots must match the actual arrived histories.")
        at = [summary["phases"].index(phase) for phase in phases]
        if [summary["true_modes"][step] for step in at] != modes:
            raise ValueError("Candidate forks must use the regime whose actual consequence just arrived.")
        z = encode_rgb(model, actual["rgb"])
        executed = action_embeddings(model, torch.as_tensor(actual["actions"], device=device).float())
        commands = torch.as_tensor(forks["actions"], device=device).float()
        if commands.ndim == 4:
            commands = commands[None].expand(len(scenes), -1, -1, -1, -1)
        plans = action_embeddings(model, commands)
        goal = encode_rgb(model, forks["goal_rgb"])
        truth = torch.as_tensor(forks["coverage"], device=device).double()[..., horizon]
        truth_xy = torch.as_tensor(forks["state"][..., 2:4], device=device).double()[..., horizon, :]
    endpoints = []
    for observed in at:
        root, commands = block_windows(z, executed, observed + 1, hold)
        endpoints.append(rollout(model, adapter, gate, root, commands, plans, compact=True)[..., horizon, :])
    predictions = torch.stack(endpoints, dim=1)
    native_latent_cost = (predictions - goal[:, None, None, None, None, :]).square().sum(-1).double()
    positions = (predictions @ w + b).double()
    posteriors = {}
    for name in ("previous_endpoint", "matched_endpoint", "native_path"):
        per_scene = summary["scores"][name]["per_scene"]
        posteriors[name] = [[per_scene[str(scene)]["posterior_braked"][observed] for observed in at] for scene in scenes]
    per_scene = summary["scores"]["native_path"]["per_scene"]
    posteriors["same_window_omit_current_write"] = torch.tensor(
        [[per_scene[str(scene)]["prior_log_odds"][observed] for observed in at] for scene in scenes], dtype=torch.double,
    ).sigmoid().tolist()
    posteriors["oracle_known_regime"] = [modes for _ in scenes]
    arms = {}
    for name, support in posteriors.items():
        p = torch.tensor(support, device=device, dtype=torch.double)
        p = torch.stack((1 - p, p), dim=-1)
        goal_score = (native_latent_cost * p[:, :, None, :, None]).sum(dim=3)
        forecast_xy = (positions * p[:, :, None, :, None, None]).sum(dim=3)
        arms[name] = candidate_metrics(goal_score, truth, forecast_xy, truth_xy)
    result = {"scene_ids": scenes, "physical_query_labels": phases, "physical_query_modes": modes,
              "horizon_raw_steps": [5, 10, 20], "arms": arms, "arrived_support": posteriors,
              "truth_coverage": truth.tolist(), "per_mode_native_latent_cost": native_latent_cost.tolist()}
    with (OUTPUT / "candidate_attribution.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
