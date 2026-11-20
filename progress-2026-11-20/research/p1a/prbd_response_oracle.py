"""Nearest attainable slow-minus-fast action effect for archived PRBD branches."""

from .prbd_paths import output_path

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import torch


def nearest(family, truth):
    # Two simplexes, one per history. Within a history p is shared by move,
    # stop, and (for the stacked problem) by all selected horizons.
    slow, fast = family
    k = len(slow)
    design = np.concatenate((slow.reshape(k, -1), -fast.reshape(k, -1)), axis=0)
    target = truth.ravel()
    scale = max(1., target @ target, np.max(np.sum(design**2, axis=1)))
    uniform = np.full(2*k, 1/k)
    sum_rows = np.repeat(np.eye(2), k, axis=1)

    def objective(p):
        residual = p @ design - target
        return (residual @ residual)/scale, 2*(design @ residual)/scale

    result = minimize(objective, uniform, method="SLSQP", jac=True,
                      bounds=[(0., 1.)] * (2*k),
                      constraints=[{"type": "eq", "fun": lambda p: sum_rows @ p-1,
                                    "jac": lambda p: sum_rows}],
                      options={"ftol": 1e-14, "maxiter": 300})
    if not result.success or np.max(np.abs(sum_rows @ result.x-1)) > 1e-8:
        raise ValueError(result.message)
    error, grad = objective(result.x)
    # Convex first-order bound, true optimum lies in [error-gap, error].
    gap = grad @ result.x - grad[:k].min() - grad[k:].min()
    if gap > 2e-6:
        raise ArithmeticError(f"Oracle accuracy insufficient: gap / scale {gap}")
    zero_error = float(target @ target)
    return {"slow": result.x[:k].tolist(), "fast": result.x[k:].tolist(),
            "truth_sse": zero_error, "fit_sse": float(error*scale),
            "optimality_gap_bound": max(0., float(gap*scale))}


def run(inputs):
    cases = torch.load(inputs, map_location="cpu", weights_only=True)["cases"]
    rows = []
    for c in cases:
        if not c["matched"]:
            continue
        family = c["move_minus_stop"].double().numpy()
        truth = c["true_interaction"].double().numpy()
        row = {"seed": c["seed"], "by_horizon": {
            str(h+1): nearest(family[:, :, h], truth[h]) for h in range(len(truth))}}
        row["shared_over_horizons"] = nearest(family, truth)
        rows.append(row)

    def summary(records):
        total = sum(v["truth_sse"] for v in records)
        return {"seeds": len(records), "percent_best_squared_error_gain_vs_zero":
                100*(1-sum(v["fit_sse"] for v in records)/total),
                "numeric_upper_gap_percentage_points":
                100*sum(v["optimality_gap_bound"] for v in records)/total}

    return {"panel": "matched only; ex-post two owned supports per reset, no truth in deployment",
            "horizons": {h: summary([r["by_horizon"][h] for r in rows])
                         for h in rows[0]["by_horizon"]},
            "one_support_pair_across_all_horizons": summary([r["shared_over_horizons"] for r in rows]),
            "per_seed": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensors", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    output_path(args.out).write_text(json.dumps(run(args.tensors), allow_nan=False, indent=2) + "\n", encoding="utf-8")
