"""Read-only Phase 1 diagnostics on the two stored G3 response families."""

import json

import torch
from .prbd_paths import ROOT, output_path


ARCHIVE = ROOT / "runs/prbd_consistency_20261120"
OUT = ROOT / "runs/prbd_phase1_20261121"


def planning_read(seed, history, speed, weights, futures, threshold):
    highest, index = weights.max(0)
    kept = 1 if highest >= threshold else len(weights)
    observed_l2, error_bound = 0., 0.
    if kept == 1:
        paths = futures[speed].permute(0, 2, 1, 3)  # program,H,K,D
        displacements = paths - paths[:, :, index:index + 1]
        observed = torch.einsum("phkd,k->phd", displacements, weights).norm(dim=-1)
        bound = (1 - highest) * displacements.norm(dim=-1).amax(-1)
        if (observed > bound + 1e-10).any():
            raise AssertionError("Discarded-mass bound violated")
        observed_l2, error_bound = float(observed.max()), float(bound.max())
    return dict(seed=seed, history=history, speed_index=speed, retained=kept,
                max_weight=float(highest), max_latent_l2=observed_l2, max_l2_bound=error_bound)


def evaluate(arm, student_results):
    name = f"g3_episode_{arm}_measure"
    saved = json.loads((ARCHIVE / f"{name}.json").read_text(encoding="utf-8"))
    eligible = {s["seed"] for s in saved["seeds"] if s["horizons"]["1"]["physical_nonzero"]}
    cases = torch.load(ARCHIVE / f"{name}_tensors.pt", weights_only=True)["cases"]
    threshold = .95
    rows = []
    prune, student_prune = [], []
    student_by_seed = {r["seed"]: r for r in student_results["rows"]}
    for case in cases:
        seed = case["seed"]
        if seed not in eligible:
            continue
        actual = case["true_future"].double()
        d = (actual[0, 0] - actual[0, 1]) - (actual[1, 0] - actual[1, 1])
        futures = case["future_responses"].double()
        if not torch.equal(futures[0], futures[1]):
            raise ValueError("The oracle bound requires the same native future family")
        family = futures[0, 0] - futures[0, 1]  # K,H,D
        projections = (family * d).sum(-1)
        width = projections.amax(0) - projections.amin(0)
        rows.append(dict(seed=seed, max_numerator=width.tolist(),
                         denominator=d.square().sum(-1).tolist()))
        for history in ("single_source", "all_arrivals", "omit_source"):
            for speed, weights in enumerate(case["supports"][history].double()):
                prune.append(planning_read(seed, history, speed, weights, futures, threshold))
                student_weights = student_by_seed[seed]["supports"]["candidate_shared_student_t_nu3"][history][speed]
                student_prune.append(planning_read(seed, history, speed,
                    torch.tensor(student_weights, dtype=torch.float64), futures, threshold))
    if len(rows) != len(eligible) or len(rows) != 39:
        raise ValueError("Unexpected archive coverage")
    pooled = [100 * sum(r["max_numerator"][h] for r in rows) /
              sum(r["denominator"][h] for r in rows) for h in range(4)]
    return dict(arm=arm, roots=len(rows), oracle_latent_signed_percent=pooled,
                oracle_constraints="Independent probability vector per speed, common across both candidates; "
                "future truth available for each root and horizon. Bound is exact only for this checkpoint.",
                threshold=threshold, conditional_planning_reads=len(prune),
                reads_using_one_branch=sum(r["retained"] == 1 for r in prune),
                max_observed_support=max(r["max_weight"] for r in prune),
                rows=rows, pruning=prune, student_pruning=student_prune,
                student_reads_using_one_branch=sum(r["retained"] == 1 for r in student_prune),
                student_max_error=max(r["max_latent_l2"] for r in student_prune),
                student_max_bound=max(r["max_l2_bound"] for r in student_prune))


if __name__ == "__main__":
    laws = json.loads((OUT / "shared_student_t_phase1.json").read_text(encoding="utf-8"))
    result = {arm: evaluate(arm, laws["historical_replay"]["episode_" + arm]) for arm in ("dual", "h1")}
    output_path(OUT / "archive.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for arm, values in result.items():
        print(arm, values["oracle_latent_signed_percent"],
              values["reads_using_one_branch"], "/", values["conditional_planning_reads"])
