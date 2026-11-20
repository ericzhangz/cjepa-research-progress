"""Read finite and integrated single-source weights through the same saved paths."""

from .prbd_paths import EVIDENCE, SPEED_DATA, output_path

import json
from pathlib import Path

import numpy as np
import torch


OUT = EVIDENCE


def compare(saved, rows):
    by_key = {(r["seed"], r["speed"], r["geometry"]): r for r in rows}
    out = {}
    for geometry in ("original", "bounded"):
        out[geometry] = {}
        for weight in ("finite32_posterior", "continuous_posterior"):
            dots, truth_sq = [], []
            for c in saved["cases"]:
                if not c["matched"]:
                    continue
                slow, fast = [np.asarray(by_key[c["seed"], speed, geometry][weight])
                              for speed in (2.5, 7.5)]
                ds, df = c["move_minus_stop"].double().numpy()
                predicted = np.einsum("k,khd->hd", slow, ds) - np.einsum("k,khd->hd", fast, df)
                target = c["true_interaction"].double().numpy()
                dots.append((predicted*target).sum(axis=1))
                truth_sq.append((target**2).sum(axis=1))
            out[geometry][weight] = (100*np.sum(dots, axis=0)/np.sum(truth_sq, axis=0)).tolist()
    return out


def main():
    reference = json.loads((OUT/"p0_observed_sources.json").read_text())["checkpoints"]
    result = {"scope": "source-only signed recovery in percent, H1 through H4; models trained using Q32"}
    for key, tensor_name in (("p12_frozen", "p12"), ("p3_bounded_d", "p3_bounded_d_evaluation")):
        tensors = torch.load(OUT/f"{tensor_name}_tensors.pt", weights_only=True, map_location="cpu")
        result[key] = compare(tensors, reference[key]["rows"])

    with np.load(SPEED_DATA/"data_fit.npz", allow_pickle=False) as data:
        a = data["actions"]
        valid = data["block_valid"].copy()
        valid[:, :2] = False
        result["fit_action_coverage"] = {"scored_blocks": int(valid.sum()),
            "five_zero_commands": int((np.all(a==0, axis=(-1,-2)) & valid).sum())}
    timeline = json.loads((OUT/"p3_comparison.json").read_text())["bounded_d_posterior_timelines"]
    stops = [e for c in timeline for history in c["histories"] for e in history["arrivals"]
             if e["raw_action_all_zero"]]
    assert all(e["physical_displacement_zero"] and e["latent_change_exact_zero"] for e in stops)
    tv = np.array([e["all_write_update_TV"] for e in stops])
    result["bounded_d_stationary_arrival_TV"] = {
        "count": len(tv), "nonzero": int(np.count_nonzero(tv)), "median": float(np.median(tv)),
        "min": float(tv.min()), "max": float(tv.max())}
    (output_path(OUT/"source_reference_readout.json")).write_text(
        json.dumps(result, allow_nan=False, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
