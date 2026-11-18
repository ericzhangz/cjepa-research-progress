"""Trace §47's source supervision and the unchanged archived neutral command."""

import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.p1a.native_topology_condition import ROOT, RUN, load_readout, load_trained
from research.p1a.native_topology_prefix_fit import (
    SOURCE_MARGIN, prefix_source_indices, read_live_panels, source_hinge,
)
from research.p1a.native_topology_xy_evidence import archived_spike, paired_directions, student_t
from research.p1a.physical_topology_gate import CENTER, HALF_WALL, RADIUS


ACCUM = ROOT / "runs/native_topology_accumulated_20261116/corrected"
BEFORE = ROOT / "runs/native_topology_live_20261116/arm2"
OUTPUT = Path(__file__).with_suffix(".json")


def whiten(x, factor):
    return torch.linalg.solve_triangular(factor, x.double().reshape(-1, 2).T, upper=False).T.reshape_as(x)


def score(arrival, issued, factor):
    residual = whiten(arrival[:, :, None] - issued[:, None], factor)
    return paired_directions(student_t(-0.5 * residual.square().sum(-1)))


def dist_stats(value):
    return {"mean": float(value.mean()), "q10_median_q90": np.quantile(value, [.1, .5, .9]).tolist()} if len(value) else None


def source_stats(directory, decode, factor):
    path = directory / "data_train_live.npz"
    _, panel, labels = read_live_panels(path, "cpu", accumulated=directory != BEFORE)
    _, _, action, target, raw_effect, physical = panel
    _, _, _, physics, static = labels
    physical = physical[:, 0, :, 0]
    end_effect = (physical[:, 0] != physical[:, 1]).any(-1)
    assert (raw_effect | ~end_effect).all()
    with torch.no_grad():
        arrived = decode(target[:, 0, :, 0]).double()
    d = whiten(arrived[:, 1] - arrived[:, 0], factor).norm(dim=-1)
    ideal = score(arrived, arrived, factor)
    torch.testing.assert_close(ideal[:, 0], ideal[:, 1], rtol=0, atol=0)
    exact_limit = ideal[:, 0]
    arbitrary_limit = 5 * torch.asinh(d / (2 * np.sqrt(3)))
    assert (arbitrary_limit + 1e-12 >= exact_limit).all()
    coordinate_error = (arrived - physical).norm(dim=-1)
    stat = {"raw_neutral_static": int(static.sum()),
            "raw_neutral_other": int((~raw_effect & ~static).sum()),
            "raw_effect_converged": int((raw_effect & ~end_effect).sum()),
            "endpoint_effect": int(end_effect.sum()),
            "endpoint_matching_exact_appearance_disagreement": int((~end_effect & (d != 0)).sum()),
            "decoded_factual_vs_physical_L2_px_all_branches": dist_stats(coordinate_error.numpy().ravel())}
    per_batch = [4, 4, 8]
    stat["draws_per_update_static_other_neutral_raw_effect"] = per_batch
    stat["expected_endpoint_effect_draws_per_update"] = 8 * float(end_effect.sum() / raw_effect.sum())
    stat["expected_uniform_draws_per_row_per_300_steps_static_other_neutral_raw_effect"] = [
        300 * count / int(population.sum()) for count, population in zip(
            per_batch, (static, ~raw_effect & ~static, raw_effect), strict=True)]

    issued = arrived.clone().requires_grad_(True)
    old_hinge = source_hinge(issued, arrived, end_effect, factor)
    old_gradient = torch.autograd.grad(old_hinge, issued)[0]
    proposed_score = score(arrived, issued, factor)
    capped_margin = ideal.clamp_max(SOURCE_MARGIN)
    fit_identity = (capped_margin - proposed_score).relu().mean()
    capped_gradient = torch.autograd.grad(fit_identity, issued, retain_graph=True)[0]
    evidence_match = (ideal - proposed_score).square().mean()
    matching_gradient = torch.autograd.grad(evidence_match, issued)[0]
    assert capped_gradient.eq(0).all() and matching_gradient.eq(0).all()
    # A descent step for the old hinge is an outward mode deformation at the exact pair.
    relative_descent = old_gradient[:, 0] - old_gradient[:, 1]
    delta = arrived[:, 1] - arrived[:, 0]
    outward = (relative_descent * delta).sum(-1)
    assert (outward >= 0).all()
    torch.testing.assert_close(old_gradient.sum(1), torch.zeros_like(old_gradient[:, 0]), rtol=0, atol=1e-15)
    stat["old_source_hinge_at_exact_decoded_centers_nat"] = float(old_hinge.detach())
    stat["old_source_hinge_at_exact_centers_outward_rows"] = int((outward > 0).sum())
    stat["old_source_hinge_at_exact_centers_outward_gradient_L2_per_row_before_informative_mean"] = dist_stats(
        (old_gradient.flatten(1).norm(dim=-1)[outward > 0] * end_effect.sum()).numpy())
    stat["old_source_hinge_at_exact_centers_active_magnitude_by_affected_rows_nat"] = dist_stats(
        (SOURCE_MARGIN - exact_limit[outward > 0]).numpy())
    stat["same_minimum_gradient_at_exact_centers_capped_hinge_and_squared_evidence_match"] = True

    # A like-kind source uses shared motion up to the left wall, hits it, and moves tangentially.
    with np.load(path, allow_pickle=False) as data:
        l, r, c, origin, h = indices = prefix_source_indices(data)
        raw = data["local_xy"][indices]
        start = data["future_xy"][l, r, c, origin, h - 1, -1]
        face = CENTER - HALF_WALL - RADIUS
        intent = action[:, 0, 0].sum(1).numpy()
        left_slide = ((~raw_effect & physics["blocked"][:, 0, 0]).numpy()
                      & (start[:, 0] < face) & (raw[:, 0, -1, 0] <= face)
                      & (intent[:, 0] > 0) & (raw[:, 0, -1, 0] > start[:, 0])
                      & (raw[:, 0, -1, 1] != start[:, 1])
                      & (raw[:, 0, -1, 0] < start[:, 0] + 2.5 * intent[:, 0] - 1e-5))
        stat["like_kind_left_wall_slide_neutral_sources"] = int(left_slide.sum())
        stat["expected_like_kind_left_wall_slide_draws_per_300_steps"] = (
            1200 * int(left_slide.sum()) / stat["raw_neutral_other"])
        cohort = data["mining_round"][l, r, c] if "mining_round" in data else np.zeros(len(l), dtype=int)
        stat["by_mining_round"] = {}
        for key in np.unique(cohort):
            take = torch.from_numpy(cohort == key)
            informative = take & end_effect
            stat["by_mining_round"][str(key)] = {
                "sources": int(take.sum()),
                "endpoint_effect": int(informative.sum()),
                "neutral_left_wall_slide": int(left_slide[take.numpy()].sum()),
                "endpoint_effect_below_3_at_truth": int((informative & (exact_limit < SOURCE_MARGIN)).sum()),
                "endpoint_effect_below_3_at_any_centers": int((informative & (arbitrary_limit < SOURCE_MARGIN)).sum()),
                "truthful_lr_nat": dist_stats(exact_limit[informative].numpy()),
            }
    stat["endpoint_effect_below_3_at_truth"] = int((end_effect & (exact_limit < SOURCE_MARGIN)).sum())
    stat["endpoint_effect_below_3_at_any_centers"] = int((end_effect & (arbitrary_limit < SOURCE_MARGIN)).sum())
    stat["truthful_true_center_lr_nat"] = dist_stats(exact_limit[end_effect].numpy())
    return stat


@torch.no_grad()
def spike_stats(directory, decode, factor, latent_scale):
    host, conditioner, gate = load_trained(directory, "conditioned", "cpu")
    case = archived_spike((host, conditioner, gate), decode, latent_scale, factor, "cpu")
    assert gate.forced_exposure == 1
    issued = torch.tensor(case["forecast_xy_mode_0_1"], dtype=torch.float64)
    arrived = torch.tensor(case["arrived_xy"], dtype=torch.float64)
    midpoint = issued.mean(0)
    difference = issued[0] - issued[1]
    white_delta = whiten(difference, factor)
    white_error = whiten(arrived - midpoint, factor)
    projected = white_error @ white_delta
    norm = white_delta.norm()
    distances = whiten(arrived[None] - issued, factor).square().sum(-1)
    torch.testing.assert_close(distances[0] - distances[1], -2 * projected, rtol=1e-12, atol=1e-12)
    case["geometry"] = dict(
        midpoint_xy=midpoint.tolist(), mode0_minus_mode1_xy=difference.tolist(),
        midpoint_error_y_minus_m_xy=(arrived - midpoint).tolist(),
        mode_gap_px=float(difference.norm()), covariance_scaled_mode_gap=float(norm),
        covariance_scaled_midpoint_error=float(white_error.norm()),
        covariance_scaled_midpoint_error_along_gap=float(projected/norm),
        covariance_scaled_error_dot_gap=float(projected),
        covariance_scaled_residual_sq=distances.tolist(),
        likelihood_gap_nat=float(student_t(-.5 * distances).diff().neg()[0]),
        # The first block uses the mean offset. Remaining blocks propagate the mode gap.
        conditioner_half_offset_gap_L2_by_layer=(conditioner(torch.tensor([[1., -1.], [1., 1.]])).diff(dim=0)[0]/2).norm(dim=-1).tolist(),
    )
    return case


def main():
    torch.set_num_threads(3)
    factor = torch.load(ACCUM / "evidence/xy_cholesky.pt", map_location="cpu", weights_only=True)
    latent_scale = torch.load(RUN / "likelihood_scale.pt", map_location="cpu", weights_only=True)
    decode = load_readout("cpu", RUN)
    report = {"training_pool": {}, "fixed_archived_command": {}}
    for directory in (BEFORE, *(ACCUM / f"round{i}" for i in range(1, 4))):
        report["training_pool"][directory.name] = source_stats(directory, decode, factor)
        report["fixed_archived_command"][directory.name] = spike_stats(directory, decode, factor, latent_scale)
        print(directory.name, json.dumps(report["fixed_archived_command"][directory.name]["geometry"]), flush=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
