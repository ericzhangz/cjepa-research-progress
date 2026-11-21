"""Score the Phase 2 repair on the archived panel and decompose its cross-action contrast."""
import json

import torch

from .prbd_paths import REPAIR
from .prbd_phase2 import OUT
from .prbd_phase2_score import BASELINES, contrast, load, save, static, switches


def response_geometry(package, batch, selected):
    if not torch.equal(package["seed"], batch["seed"]):
        raise ValueError("Response and observation seeds differ")
    indices = [batch["seed"].tolist().index(seed) for seed in selected]
    paths = package["family"][indices].double()
    if not torch.isfinite(paths).all():
        raise ValueError("Nonfinite signed response")
    family_error = (paths[:, 0] - paths[:, 1]).abs().max().item()
    torch.testing.assert_close(paths[:, 0], paths[:, 1], atol=1e-5, rtol=1e-5)
    # Two actions from precisely the same window, for each prototype and horizon.
    responses = paths[:, 0, 0] - paths[:, 0, 1]
    truth = contrast(batch["z"][indices, :, :, 7:11].double())
    energy = truth.square().sum((0, 2))
    if (energy <= 0).any():
        raise ValueError("Undefined normalized true direction")
    align = (responses * truth[:, None]).sum(-1)
    # Two independent choices of support, one per actual history. This bound is
    # per root and horizon and uses true futures only after forecasting.
    ceiling = 100 * (align.amax(1) - align.amin(1)).sum(0) / energy
    common = align.sum(0)
    common_ceiling = 100 * (common.amax(0) - common.amin(0)) / energy
    rows = {}
    uniform = contrast(package["predictions"]["uniform"][indices])
    for name, priors in package["read_priors"].items():
        w = priors[indices]
        replay_error = (w[:, :, 0] - w[:, :, 1]).abs().max().item()
        torch.testing.assert_close(w[:, :, 0], w[:, :, 1], atol=1e-10, rtol=1e-9)
        dw = w[:, 0, 0] - w[:, 1, 0]
        reconstruction = (dw[:, :, None, None] * responses).sum(1)
        observed = contrast(package["predictions"][name][indices]) - uniform
        torch.testing.assert_close(reconstruction, observed, atol=1e-5, rtol=1e-5)
        inner = (reconstruction * truth).sum((0, 2))
        output_energy = reconstruction.square().sum((0, 2))
        rows[name] = dict(
            mean_tv_histories=float(0.5 * dw.abs().sum(1).mean()),
            mean_read_weights_slow_fast=w[:, :, 0].mean(0).tolist(),
            max_prior_difference_repeated_actions=replay_error,
            max_contrast_reconstruction_error=float((observed-reconstruction).abs().max()),
            attained_directed_percent=(100 * inner / energy).tolist(),
            output_energy_percent=(100 * output_energy / energy).tolist(),
            global_alignment_cosine=[float(dot/(a*b).sqrt()) if a>0 else None
                                     for dot,a,b in zip(inner,output_energy,energy)],
        )
    return dict(n=len(indices), max_family_difference_repeated_histories=family_error,
                hindsight_simplex_directed_percent=ceiling.tolist(),
                hindsight_common_slow_fast_pair_directed_percent=common_ceiling.tolist(),
                hindsight_common_slow_fast_pair_by_h=list(zip(common.argmax(0).tolist(),common.argmin(0).tolist())),
                actual=rows)


def admission(static_results, switch_results):
    h = static_results["physical_nonzero_panel"]["horizon"]
    w = static_results["probes"]["trained"]["matched"]
    sw = switch_results["full"]["summary"]
    fast = sw["to_fast"]
    return dict(
        quiet=w["statistics"]["write"]["p95"] <= .01 and w["over_half"] == 0,
        cross_action=all(h[i][arm]["directed_percent"] > 0 and h[i][arm]["sse_gain_percent"] > 0
                         and 2*h[i][arm]["improved"] > static_results["physical_nonzero"]
                         for i in ("1", "4") for arm in ("first", "full")),
        both_switches=all(sw[d]["opportunities"] > 0 and 2*sw[d]["recovered"] >= sw[d]["opportunities"]
                          for d in ("to_fast", "to_slow")),
        fast_lag=fast["median_lag"] is not None and fast["median_lag"] <= 2,
        matched_controls=all(
            h["4"][m]["prbd_gain_over_method"]["ci95"][0] > 0
            and all(switch_results[m]["summary"][d]["prbd_gain_over_method"]["ci95"][0] > 0
                    for d in ("to_fast", "to_slow"))
            for m in BASELINES),
    )


def main():
    torch.set_num_threads(3)
    pred = REPAIR / "unlabeled"
    static_results = static(OUT, pred, REPAIR, inspect_mass=True)
    switch_results = switches(OUT, pred, REPAIR)
    roots = json.loads((REPAIR / "static_physical_nonzero_roots.json").read_text(encoding="utf-8"))
    geometry = response_geometry(load(pred/"pred_static_prbd.pt"), load(OUT/"static.pt"), roots["1"]["seed"])
    save("response_geometry.json", geometry, REPAIR)
    save("admission.json", admission(static_results, switch_results), REPAIR)


if __name__ == "__main__":
    main()
