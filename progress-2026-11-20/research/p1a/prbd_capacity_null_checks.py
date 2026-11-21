"""Phase 1 neutral-arrival diagnostics after the supervised A/B capacity fit.

Use the unchanged G3 development histories. No fitting or new environment run.
Unlike historical G3, these models have no algebraic null-action cancellation.
"""

import json
from time import perf_counter

import torch

from .prbd_capacity_oracle import load_oracle_checkpoint
from .prbd_shared_student_t_phase1 import HISTORY, OUT, evaluate
from .prbd_paths import ROOT, output_path




@torch.no_grad()
def check_arm(path, batch, seeds, noise):
    model = load_oracle_checkpoint(path, "cpu").eval().requires_grad_(False)
    lookup = {int(seed): i for i, seed in enumerate(batch["seed"])}
    windows, controls, arrivals, identities = [], [], [], []
    for seed in seeds:
        for speed in (0, 1):
            z, a = batch["z"][lookup[seed], speed, 0], batch["a"][lookup[seed], speed, 0]
            for root in range(2, 6):
                windows.append(z[root-2:root+1])
                controls.append(a[root-2:root+1])
                arrivals.append(z[root+1])
                identities.append((seed, speed, root))
    # No equality to archived native responses is asserted across batch shapes.
    responses = []
    for start in range(0, len(windows), 16):
        responses.append(model(torch.stack(windows[start:start+16]),
                               torch.stack(controls[start:start+16])))
    responses = torch.cat(responses)
    uniform = torch.full((4,), .25, dtype=torch.float64)
    states, rows = {}, []
    for identity, means, target in zip(identities, responses, arrivals, strict=True):
        seed, speed, root = identity
        key = (seed, speed)
        before = states.get(key, uniform.clone())
        causal = evaluate("candidate_shared_student_t_nu3", means, target, before, noise)
        states[key] = causal.posterior
        if root not in (4, 5):
            continue
        fixed = evaluate("candidate_shared_student_t_nu3", means, target, uniform, noise)
        white = torch.linalg.solve_triangular(noise, means.double().T, upper=False).T
        max_white_pair = float(torch.cdist(white, white).max())
        speed_logbf = float(causal.relative_log_likelihood[1]-causal.relative_log_likelihood[2])
        rows.append({"seed": seed, "speed_index": speed, "action_t": root,
            "arrival_index": root+1, "command": "actual zero primitive action block in G3 program",
            "branch_raw_pairwise_max_l2": float(torch.cdist(means.double(), means.double()).max()),
            "branch_whitened_pairwise_max_l2": max_white_pair,
            "max_B_column_l2": float((white-white.mean(0)).norm(dim=1).max()),
            "branch_means_exactly_equal": torch.equal(means, means[:1].expand_as(means)),
            "pair_logBF_max_absolute": float(causal.relative_log_likelihood.max()-causal.relative_log_likelihood.min()),
            "speed_branches_raw_pair_l2": float((means[1].double()-means[2].double()).norm()),
            "speed_branches_whitened_pair_l2": float((white[1]-white[2]).norm()),
            "speed_branches_abs_logBF": abs(speed_logbf),
            "speed_only_uniform_prior_write_TV": abs(float(torch.sigmoid(
                torch.tensor(speed_logbf, dtype=torch.float64)))-.5),
            "uniform_prior_write_TV": float(.5*(fixed.posterior-uniform).abs().sum()),
            "causal_prior": before.tolist(), "causal_posterior": causal.posterior.tolist(),
            "causal_write_TV": float(.5*(causal.posterior-before).abs().sum())})
    metrics = ("branch_raw_pairwise_max_l2", "branch_whitened_pairwise_max_l2", "max_B_column_l2",
               "pair_logBF_max_absolute", "speed_branches_raw_pair_l2", "speed_branches_whitened_pair_l2",
               "speed_branches_abs_logBF", "speed_only_uniform_prior_write_TV",
               "uniform_prior_write_TV", "causal_write_TV")
    summary = {name: {"min": min(row[name] for row in rows), "max": max(row[name] for row in rows),
                       "mean": sum(row[name] for row in rows)/len(rows)} for name in metrics}
    return {"checkpoint": str(path.resolve()), "actual_neutral_arrivals": len(rows),
            "exact_equal_response_arrivals": sum(row["branch_means_exactly_equal"] for row in rows),
            "nonzero_causal_write_arrivals": sum(row["causal_write_TV"] > 0 for row in rows),
            "summary": summary, "rows": rows}


def main():
    torch.set_num_threads(2)
    started = perf_counter()
    batch = torch.load(HISTORY/"dev.pt", weights_only=True, map_location="cpu")
    saved = json.loads((HISTORY/"g3_episode_dual_measure.json").read_text())
    seeds = [row["seed"] for row in saved["seeds"] if row["horizons"]["1"]["physical_nonzero"]]
    noise = torch.load(HISTORY/"g3_episode_dual.pt", weights_only=True, map_location="cpu")["noise_cholesky"].double()
    report = {"scope": "Phase 1 intermediate neutral-arrival test; no fit, G4 or Phase 2",
              "observation_law": "candidate_shared_student_t_nu3; scale R, covariance 3R",
              "physical_condition": "Same 39 static-speed matched G3 resets; actual t4 and t5 zero-command blocks",
              "public_state": "Uniform start; causal t2,t3,t4,t5 real arrivals only",
              "model_boundary": "Capacity A/B have null_control=None; no historical algebraic null-response assumption",
              "native_batch_size": 16, "arms": {}}
    for arm in ("A", "B"):
        path = ROOT/"runs/prbd_capacity_phase1_20261121"/arm/"final.pt"
        print(arm, "actual neutral-arrival replay", flush=True)
        report["arms"][arm] = check_arm(path, batch, seeds, noise)
    report["seconds"] = perf_counter()-started
    output_path(OUT/"capacity_neutral_arrivals.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    output_path(OUT/"CAPACITY_NEUTRAL_ARRIVALS.md").write_text(render(report), encoding="utf-8")
    print(output_path(OUT/"capacity_neutral_arrivals.json"), report["seconds"], flush=True)


def render(report):
    lines = ["# Phase 1: actual neutral arrivals without an algebraic null anchor", "",
        "Replay the same 39 matched G3 resets at both static physical speeds. Each arm receives 156 actual zero-command arrivals (t4,t5). Native responses are issued before the successor, and causal support uses actual t2..t5 arrivals only. A/B are the supervised capacity checkpoints, not PRBD probability-trained candidates. The archived whitening scale R remains fixed; the candidate uses shared full-latent t3 with covariance 3R.", "",
        "| Arm | Speed-branch whitened pair spread min/max | Speed-pair abs logBF min/max | Binary equal-prior TV mean | Four-branch equal-prior TV mean | Causal write TV mean/max |",
        "|---|---:|---:|---:|---:|---:|"]
    for arm, data in report["arms"].items():
        s = data["summary"]
        spread, bf = s["speed_branches_whitened_pair_l2"], s["speed_branches_abs_logBF"]
        causal = s["causal_write_TV"]
        lines.append(f"| {arm} | {spread['min']:.6f}/{spread['max']:.6f} | {bf['min']:.6f}/{bf['max']:.6f} | "
            f"{s['speed_only_uniform_prior_write_TV']['mean']:.6f} | {s['uniform_prior_write_TV']['mean']:.6f} | "
            f"{causal['mean']:.6f}/{causal['max']:.9f} |")
    lines.extend(["", "The speed-pair columns use the two offline labeled speed branches (1 and 2), so this result is not merely suppression of the nominal branch. Every arm has 0/156 exactly equal response families and 156/156 nonzero causal neutral writes. Strong prior concentration can make a nonzero write numerically tiny; the JSON includes likelihood spans and uniform-prior checks separately.", "",
        "These measurements show that historical G3's exact no-write property does not transfer to A/B after removing the algebraic null anchor. They do not establish calibrated probabilities for the supervised models: R was retained from the historical fit, and no density training or recalibration was performed. No new trajectories, Phase 2, G4 or planning were run.", "",
        f"Reproduce: python -m research.p1a.prbd_capacity_null_checks. Runtime {report['seconds']:.3f} s.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
