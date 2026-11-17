"""Fixed-weight intermediate measurements for §35 on archived PushT queries."""

import json

import numpy as np
import torch

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import conditioned_rollout, historical_support, inherit_window, write_observation
from .native_full_depth_first_effect import branch_offsets
from .native_full_depth_paired_effect import ObservableChart
from .native_full_depth_reversible import SWITCH, action_embeddings, block_windows, encode_rgb
from .native_full_depth_task_geometry import chart_pose
from .native_visible_source import FITTED, OUTPUT as SOURCE, PAIRS


OUTPUT = SOURCE.parent / "native_history_inheritance_20261115"
HORIZONS = [0, 1, 3]


def rms(delta):
    return delta.double().square().mean(dim=tuple(range(delta.ndim - 2)) + (-1,)).sqrt().tolist()


def report_pair(first, second, chart):
    delta = first - second
    xy_delta = chart_pose(chart(first))[..., :2] - chart_pose(chart(second))[..., :2]
    return {
        "latent_rms": rms(delta),
        "xy_rms": rms(xy_delta),
        "per_scene_candidate_mode_xy_distance": xy_delta.norm(dim=-1).tolist(),
    }


def analytic_lifecycle():
    """Independent scalar sensor arrivals and two candidate native attention rows."""
    log_support = torch.tensor([0.5, 0.5], dtype=torch.double).log()
    transition = log_support.new_tensor([[0.9, 0.1], [0.1, 0.9]])
    weights, messages = [], []
    # Equal-variance means +/-1 give the likelihood ratio exp(2*y).
    issued = log_support.new_tensor([[-1.0], [1.0]])
    for ratio in (9.0, 1.0, 1.0 / 162.0):
        measurement = log_support.new_tensor([0.5 * np.log(ratio)])
        log_support, _ = write_observation(log_support, issued, measurement, 1.0)
        _, ancestry = historical_support(log_support, transition)
        beta = ancestry[1, 1]
        weights.append(log_support[1].exp().item())
        messages.append([float(beta - 0.25), float(beta - 0.75)])
    np.testing.assert_allclose(weights, [0.9, 0.9, 1 / 19])
    np.testing.assert_allclose(messages[1], messages[0])
    assert messages[0][1] > 0 and messages[2][1] < 0
    return {"posterior_mode_one": weights, "messages": messages}


def replay_full_arrivals(forecast, parents, prior, transition, start, coarse, executed, hold, chart, bundle):
    """Issue first, score each actual endpoint once, and replace its fully observed token."""
    posterior = prior
    drift = bundle["drift"].reshape(-1, 4)[-1]
    covariance = bundle["covariance"][-4:, -4:]
    whiten = torch.linalg.cholesky(covariance)
    stages = []
    for tick in range(start, executed.size(1)):
        common, past = block_windows(coarse, executed, tick, hold)
        log_prior, ancestry = historical_support(posterior, transition)
        trace = {}
        issued = forecast(parents, past, executed[:, None, tick:tick + 1], ancestry, trace)[:, 0, :, 0]
        arrived = coarse[:, tick + 1]
        residual = chart(arrived)[:, None] - chart(issued) - drift
        white = torch.linalg.solve_triangular(whiten, residual.to(whiten).unsqueeze(-1), upper=False)
        log_evidence = -0.5 * white.squeeze(-1).square().sum(-1)
        posterior = (log_prior + log_evidence.to(log_prior)).log_softmax(dim=-1)
        native = forecast(common, past, executed[:, None, tick:tick + 1])[:, 0, :, 0]
        contrast = trace["parent_attention_residual"].diff(dim=1)
        stages.append({
            "issued_at_block": tick,
            "prior_mode_one": log_prior[:, 1].exp().tolist(),
            "actual_arrival_mode_one": posterior[:, 1].exp().tolist(),
            "log_likelihood_ratio": log_evidence.diff(dim=-1).squeeze(-1).tolist(),
            "parent_attention_gap_rms": contrast.square().mean().sqrt().item(),
            "issued_minus_public_native_rms": (issued - native).square().mean().sqrt().item(),
        })
        earlier = inherit_window(parents, ancestry)[:, :, 1:]
        parents = torch.cat((earlier, arrived[:, None, None].expand(-1, 2, 1, -1)), dim=2)
    return stages


@torch.inference_mode()
def run(device):
    lifecycle = analytic_lifecycle()
    model, adapter, gate, hold, *_ = substep.load_core(device)
    bundle = torch.load(FITTED / "forecast_only.pt", map_location=device, weights_only=True)
    adapter.load_state_dict(bundle["adapter"])
    chart = ObservableChart(bundle["chart"]).eval()
    source = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    evidence = source["source"]["camera_path"]
    with np.load(PAIRS / "fresh_switches.npz", allow_pickle=False) as actual, np.load(
        PAIRS / "post_switch_candidates.npz", allow_pickle=False
    ) as forks:
        scenes = actual["scene_id"].tolist()
        phases = actual["phase"].tolist()
        ticks = [phases.index(name) for name in forks["phase"].tolist()]
        if scenes != source["scene_ids"] or scenes != forks["scene_id"].tolist() or phases != source["phases"]:
            raise ValueError("Source observations and fixed candidate roots must align")
        coarse = encode_rgb(model, actual["rgb"])
        executed = action_embeddings(model, hold.new_tensor(actual["actions"]))
        plans = action_embeddings(model, hold.new_tensor(forks["actions"]))[None].expand(len(scenes), -1, -1, -1)
    offsets = branch_offsets(adapter, len(scenes))
    old_weights = hold.new_tensor(evidence["prior"])
    arrived_weights = hold.new_tensor(evidence["posterior"])[..., -1]
    transition = hold.new_tensor([[1 - SWITCH, SWITCH], [SWITCH, 1 - SWITCH]])
    identity = torch.eye(2, dtype=hold.dtype, device=device).expand(len(scenes), -1, -1)

    def forecast(history, past, actions, ancestry=None, trace=None):
        return conditioned_rollout(model, history, past, actions, offsets, first_effect=gate,
                                   ancestry=ancestry, trace=trace)[..., HORIZONS if actions.size(2) == 4 else [0], :]

    def controlled_parents(at, root):
        # Replace only the middle token by two forecasts issued before its timestamp.
        older, older_past = block_windows(coarse, executed, at - 2, hold)
        earlier = forecast(older, older_past, executed[:, None, at - 2:at - 1])
        parents = root[:, None].expand(-1, 2, -1, -1).clone()
        parents[:, :, 1] = earlier[:, 0, :, 0]
        return parents

    result = {"scenes": scenes, "horizon_raw_steps": [5, 10, 20], "analytic": lifecycle, "queries": {}}
    for tick in ticks:
        root, past = block_windows(coarse, executed, tick + 1, hold)
        parents = controlled_parents(tick + 1, root)
        before, after = (torch.stack((1 - p[:, tick], p[:, tick]), -1).log() for p in (old_weights, arrived_weights))
        _, before_ancestry = historical_support(before, transition)
        _, after_ancestry = historical_support(after, transition)
        trace = {}
        read = forecast(parents, past, plans, after_ancestry, trace=trace)
        omit = forecast(parents, past, plans, before_ancestry)
        independent = forecast(parents, past, plans)
        collapse = forecast(inherit_window(parents, after_ancestry), past, plans)
        every_path = torch.stack([forecast(parents[:, i], past, plans) for i in range(2)], dim=-2)
        expanded = torch.einsum("bji,bcjhid->bcjhd", after_ancestry, every_path)
        one_hot = forecast(parents, past, plans, identity)
        same_parents = forecast(root, past, plans, after_ancestry)
        same_native = forecast(root, past, plans)
        torch.testing.assert_close(one_hot, independent, rtol=2e-4, atol=3e-5)
        torch.testing.assert_close(same_parents, same_native, rtol=2e-4, atol=3e-5)
        residual = trace["parent_attention_residual"].unflatten(0, (len(scenes), plans.size(1)))
        inherited = trace["inherited_attention_residual"].unflatten(0, (len(scenes), plans.size(1)))
        parent_difference = residual[:, :, 1] - residual[:, :, 0]
        delta = after_ancestry - before_ancestry
        write_effect = delta[:, None, :, 1, None, None] * parent_difference[:, :, None]
        result["queries"][phases[tick]] = {
            "actual_root": tick + 1,
            "parent_history_difference_rms": (parents[:, 1] - parents[:, 0]).square().mean().sqrt().item(),
            "support_before_and_after": torch.stack((old_weights[:, tick], arrived_weights[:, tick]), -1).tolist(),
            "ancestry_after": after_ancestry.tolist(),
            "parent_attention_residual_difference_rms": parent_difference.square().mean().sqrt().item(),
            "inherited_attention_residual_rms": inherited.square().mean().sqrt().item(),
            "fact_write_attention_residual_rms": write_effect.square().mean().sqrt().item(),
            "one_hot_native_max_abs": (one_hot - independent).abs().max().item(),
            "identical_history_native_max_abs": (same_parents - same_native).abs().max().item(),
            "write_minus_skip": report_pair(read, omit, chart),
            "attention_minus_window_mean": report_pair(read, collapse, chart),
            "attention_minus_independent": report_pair(read, independent, chart),
            "attention_minus_expanded_path_mean": report_pair(read, expanded, chart),
        }
    start = ticks[0] - 1
    root, _ = block_windows(coarse, executed, start, hold)
    parents = controlled_parents(start, root)
    p = arrived_weights[:, start - 1]
    result["full_rgb_replay"] = replay_full_arrivals(
        forecast, parents, torch.stack((1 - p, p), -1).log(), transition,
        start, coarse, executed, hold, chart, bundle,
    )
    return result


if __name__ == "__main__":
    result = run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT / "stage.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(OUTPUT / "stage.json", flush=True)
