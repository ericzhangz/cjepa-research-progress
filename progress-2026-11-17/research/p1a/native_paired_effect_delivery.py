"""Replay both directions of a real PushT reversal and then score the same legal candidates."""

import json

import numpy as np
import torch

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import advance_log_support
from .native_full_depth_first_effect import rollout
from .native_full_depth_paired_effect import INITIAL, OUTPUT, ObservableChart
from .native_full_depth_reversible import SWITCH, action_embeddings, block_windows, encode_rgb
from .native_full_depth_task_geometry import chart_pose, json_array, pose_chart, scene_support, shape_cost, source_predictions


DELIVERY = OUTPUT.parent / "native_reversible_delivery_20261115" / "physical_paths.npz"


def physical_choice(costs, support, actual_coverage):
    mass = torch.stack((1 - support, support), dim=-1)
    queue = (costs * mass[:, None]).sum(dim=-1)
    selected = queue.argmin(dim=-1, keepdim=True)
    picked = actual_coverage.gather(-1, selected).squeeze(-1)
    regret = actual_coverage.max(dim=-1).values - picked
    lhs, rhs = torch.triu_indices(costs.size(1), costs.size(1), offset=1, device=costs.device)
    truth_diff = actual_coverage[:, lhs] - actual_coverage[:, rhs]
    significant = truth_diff.abs() > 0.01
    predicted_diff = queue[:, rhs] - queue[:, lhs]
    agree = 0.5 * (1 + (truth_diff * predicted_diff).sign())
    return {"selected": selected.squeeze(-1), "attained": picked, "regret": regret,
            "significant_pairs": significant.sum(dim=-1), "agreement_credit": (agree * significant).sum(dim=-1)}


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, adapter, gate, hold, *_ = substep.load_core(device)
    prefix = substep.NativeActionPrefixes(hold.numel()).to(device).eval()
    old = torch.load(substep.OUTPUT / "prefix.pt", map_location=device, weights_only=True)
    initial = torch.load(INITIAL / "coordinates.pt", map_location=device, weights_only=True)
    chart = ObservableChart(initial).eval()
    with np.load(DELIVERY, allow_pickle=False) as raw:
        scenes = raw["scene_id"].tolist()
        modes = torch.tensor(raw["actual_mode"][:, -1], dtype=torch.long, device=device)
        names = raw["candidate"].tolist()
        coarse = encode_rgb(model, raw["actual_rgb"][:, ::5])
        arrived = encode_rgb(model, raw["actual_rgb"][:, 1:]).unflatten(1, (-1, 5))
        goals = encode_rgb(model, raw["goal_rgb"])
        actual_commands = torch.tensor(raw["prefix_action"], dtype=torch.float32, device=device)
        actual_commands = actual_commands[None].expand(len(scenes), -1, -1, -1)
        actions = torch.tensor(raw["candidate_action"], dtype=torch.float32, device=device)
        candidates = action_embeddings(model, actions[:, None])[None].expand(len(scenes), -1, -1, -1)
        coverage = torch.tensor(raw["candidate_coverage"][..., -1], device=device)
        actual = coverage[torch.arange(len(scenes), device=device), modes]
        true_pose = torch.tensor(raw["candidate_state"][..., -1, 2:5], device=device).transpose(1, 2)
        true_goal = torch.tensor(raw["goal_pose"], device=device)
    methods = {}
    for arm in ("frozen", "forecast_only", "finite_effect"):
        if arm == "frozen":
            bundle = initial
            prefix.load_state_dict(old["prefix"])
        else:
            bundle = torch.load(OUTPUT / f"{arm}.pt", map_location=device, weights_only=True)
            adapter.load_state_dict(bundle["adapter"])
            prefix.load_state_dict(bundle["prefix"])
            chart.load_state_dict(bundle["chart"])
        latent, embedded = source_predictions(model, adapter, gate, hold, prefix, coarse, actual_commands)
        issued = chart(latent)
        factors = substep.chronological_evidence(issued, chart(arrived), bundle["drift"], bundle["covariance"])
        prior, posterior = scene_support(factors)
        root, past = block_windows(coarse, embedded, embedded.size(1), hold)
        prediction = chart(rollout(model, adapter, gate, root, past, candidates, compact=True)[..., 0, :])
        goal = chart(goals)
        cost = shape_cost(prediction, goal[:, None, None])
        # Apply the same prediction-time switching prior to both state
        # interventions. Neither reads the alternative actual arrivals.
        before_after = torch.stack((prior[:, -1], posterior[:, -1, -1]), dim=-1)
        supports = advance_log_support(torch.stack((1 - before_after, before_after), dim=-1).log(), SWITCH).exp()[..., 1]
        methods[arm] = {
            "support_prior": prior, "support_chronological": posterior, "log_likelihood": factors,
            "shape_cost": cost, "absolute_shape_error": (prediction - pose_chart(true_pose)).square().sum(-1).sqrt(),
            "goal_pose": chart_pose(goal),
            "goal_shape_error": (goal - pose_chart(true_goal)).square().sum(-1).sqrt(),
            "decisions": {
                "prior": physical_choice(cost, supports[:, 0], actual),
                "path": physical_choice(cost, supports[:, 1], actual),
                "uniform": physical_choice(cost, torch.full_like(supports[:, 1], 0.5), actual),
                "known_mode": physical_choice(cost, modes.double(), actual),
            },
        }
    with (OUTPUT / "delivery_decisions.json").open("x", encoding="utf-8") as file:
        json.dump({"scenes": scenes, "actual_mode": modes, "candidate_names": names,
                   "actual_coverage": actual, "arms": methods}, file, default=json_array, allow_nan=False, indent=2)


if __name__ == "__main__":
    main()
