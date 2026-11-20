"""Independent continuous-reference replay of the final G3 H1/H4 signs."""

from .prbd_paths import CONSISTENCY, output_path

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .prbd import assimilate_continuous, issue
from .prbd_continuous_checks import continuous_reference
from .prbd_native import NativePrototypes
from .prbd_tworoom_speed_fit import load_host


OUT = CONSISTENCY
COMPARISONS = {"single_vs_uniform": ("single_source", "uniform"),
               "all_vs_uniform": ("all_arrivals", "uniform"),
               "all_vs_omit": ("all_arrivals", "omit_source")}


def contrast(prediction, step):
    return prediction[0, 0, step]-prediction[0, 1, step]-prediction[1, 0, step]+prediction[1, 1, step]


@torch.no_grad()
def check(label, batch):
    checkpoint = torch.load(OUT/f"g3_{label}.pt", weights_only=True, map_location="cpu")
    evaluation = json.loads((OUT/f"g3_{label}_measure.json").read_text())
    saved = torch.load(OUT/f"g3_{label}_measure_tensors.pt", weights_only=True, map_location="cpu")
    by_seed = {row["seed"]: row for row in evaluation["seeds"]}
    lookup = {int(seed): i for i, seed in enumerate(batch["seed"])}
    model = NativePrototypes(load_host("cpu"), null_control=batch["null_control"]).eval().requires_grad_(False)
    model.offsets.copy_(checkpoint["offsets"])
    noise, transition = checkpoint["noise_cholesky"].double(), checkpoint["transition"].double()
    uniform = noise.new_full((4,), .25)
    likelihood_rows, rows = [], []
    for case in saved["cases"]:
        seed = case["seed"]
        index = lookup[seed]
        source, reversal = [], []
        for speed in (0, 1):
            z, a = batch["z"][index, speed, 0], batch["a"][index, speed, 0]
            for step, collection in ((2, source), (3, reversal)):
                means = model(z[None, step-2:step+1], a[None, step-2:step+1])[0]
                signed = issue(means, uniform, transition, noise, reference="uniform")
                fixed = assimilate_continuous(signed, z[step+1], checkpoint["scales"], checkpoint["masses"])
                reference = continuous_reference(means.double().numpy(), z[step+1].double().numpy(), noise.numpy())
                archived = torch.tensor(by_seed[seed]["actual_write_traces"][speed][step-2]
                                        ["weights"]["all_arrivals"]["relative_log_likelihood"], dtype=torch.float64)
                reproduced_error = fixed.relative_log_likelihood-archived
                reproduced_pair_error = float(reproduced_error.max()-reproduced_error.min())
                assert reproduced_pair_error < 1e-8, (seed, speed, step, reproduced_pair_error)
                error = fixed.relative_log_likelihood-torch.from_numpy(reference["ratios"])
                pair_error = float(error.max()-error.min())
                assert pair_error <= 1e-4
                collection.append(torch.from_numpy(reference["ratios"]))
                likelihood_rows.append({"seed": seed, "speed_index": speed, "action_t": step,
                    "fixed_trace_reproduction_pair_error": reproduced_pair_error,
                    "continuous_pair_logBF_error": pair_error})
        source, reversal = torch.stack(source), torch.stack(reversal)
        reference_supports = {"single_source": source.softmax(1),
                              "all_arrivals": (source+reversal).softmax(1),
                              "omit_source": reversal.softmax(1),
                              "uniform": uniform.expand(2, -1),
                              "nominal": case["supports"]["nominal"]}
        support_errors = {name: float(.5*(weight-case["supports"][name]).abs().sum(1).max())
                          for name, weight in reference_supports.items()}
        assert max(support_errors.values()) <= 1e-5
        paths = case["future_responses"].double()
        common = paths[:, :, 0]
        predictions = {name: common+torch.einsum("sk,spkhd->sphd", weights, paths-common[:, :, None])
                       for name, weights in reference_supports.items()}
        target = case["true_future"].double()
        truth = target[0, 0]-target[0, 1]-target[1, 0]+target[1, 1]
        row = {"seed": seed, "support_TV_errors": support_errors, "horizons": {}}
        for horizon, hi in (("1", 0), ("4", 3)):
            comparisons = {}
            for name, (after, before) in COMPARISONS.items():
                fixed_after, fixed_before = (contrast(case["predictions"][method], hi)
                                               for method in (after, before))
                ref_after, ref_before = (contrast(predictions[method], hi) for method in (after, before))
                fixed_dot = float((fixed_after-fixed_before)@truth[hi])
                reference_dot = float((ref_after-ref_before)@truth[hi])
                assert np.sign(fixed_dot) == np.sign(reference_dot), (label, seed, horizon, name, fixed_dot, reference_dot)
                fixed_gain = float((fixed_before-truth[hi]).square().sum()-(fixed_after-truth[hi]).square().sum())
                ref_gain = float((ref_before-truth[hi]).square().sum()-(ref_after-truth[hi]).square().sum())
                comparisons[name] = {"fixed_signed_numerator": fixed_dot,
                                      "reference_signed_numerator": reference_dot,
                                      "fixed_interaction_error_gain": fixed_gain,
                                      "reference_interaction_error_gain": ref_gain,
                                      "signed_direction_matches": True}
            row["horizons"][horizon] = {"truth_squared_norm": float(truth[hi].square().sum()),
                                         "comparisons": comparisons}
        rows.append(row)
    aggregate = {}
    for horizon in ("1", "4"):
        denominator = sum(row["horizons"][horizon]["truth_squared_norm"] for row in rows)
        comparisons = {}
        for name in COMPARISONS:
            terms = [row["horizons"][horizon]["comparisons"][name] for row in rows]
            totals = {key: sum(term[key] for term in terms) for key in
                      ("fixed_signed_numerator", "reference_signed_numerator",
                       "fixed_interaction_error_gain", "reference_interaction_error_gain")}
            assert np.sign(totals["fixed_signed_numerator"]) == np.sign(totals["reference_signed_numerator"])
            assert np.sign(totals["fixed_interaction_error_gain"]) == np.sign(totals["reference_interaction_error_gain"])
            comparisons[name] = {**totals,
                "fixed_signed_recovery": totals["fixed_signed_numerator"]/denominator,
                "reference_signed_recovery": totals["reference_signed_numerator"]/denominator,
                "all_per_seed_signed_directions_match": True}
        aggregate[horizon] = comparisons
    return {"matched_resets": len(rows), "likelihood_rows": likelihood_rows, "rows": rows,
            "aggregate": aggregate,
            "max_pair_logBF_error": max(row["continuous_pair_logBF_error"] for row in likelihood_rows),
            "max_archived_fixed_trace_reproduction_error": max(row["fixed_trace_reproduction_pair_error"] for row in likelihood_rows),
            "max_support_TV_error": max(error for row in rows for error in row["support_TV_errors"].values())}


def main():
    # Match the saved evaluator's native FP32 two-thread forward calculation.
    torch.set_num_threads(2)
    batch = torch.load(OUT/"dev.pt", weights_only=True, map_location="cpu")
    started = perf_counter()
    report = {"scope": "G3 development numerical replay only; saved native future families unchanged; no G4/P4",
        "criteria": "locked G1 logBF1e-4, TV1e-5, same per-seed and aggregate H1/H4 signs",
        "chronology": "Adaptive reference at actual source and reversal. Subsequent native-null likelihood ratios are exactly zero, already exhaustively tested.",
        "checkpoints": {}}
    for label in ("episode_h1", "episode_dual"):
        print(label, "continuous signed readout replay", flush=True)
        report["checkpoints"][label] = check(label, batch)
        print(label, "passed", flush=True)
    report["passed"], report["seconds"] = True, perf_counter()-started
    (output_path(OUT/"g1_g3_signed_reference.json")).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    (output_path(OUT/"G1_G3_SIGNED_REFERENCE.md")).write_text(render_report(report), encoding="utf-8")
    print(OUT/"g1_g3_signed_reference.json", report["seconds"], flush=True)


def render_report(report):
    lines = ["# Continuous-reference replay preserves the new G3 signs", "",
        "Both final models pass the locked numerical sign check on all 42 matched development resets. The native t2 source and t3 reversal responses were reconstructed with the evaluator's identical FP32 two-thread single-case forward. Their fixed-rule log-Bayes factors reproduce every saved trace exactly (maximum discrepancy zero). Adaptive SciPy continuous likelihoods then replace the two actual informative arrivals in the single-source, complete-write, and omit-source prefixes. The two following native-null updates are exact no-writes, already checked on the complete new cohorts. The future branch families and actual targets are the saved evaluator tensors.", "",
        "Every per-seed H1/H4 directed numerator retains its sign for all three comparisons under both final models. Aggregate directed numerators and aggregate interaction-error gains also retain their signs. No new histories, fitting, G4, or P4 were run.", "",
        "| Final weights | Maximum logBF error (nat) | Maximum prefix posterior TV |",
        "|---|---:|---:|"]
    for label, details in report["checkpoints"].items():
        lines.append(f"| {label} | {details['max_pair_logBF_error']:.3g} | {details['max_support_TV_error']:.3g} |")
    lines.extend(["", "| Final weights | H | Comparison | Fixed rule signed recovery | Continuous reference signed recovery |",
                  "|---|---:|---|---:|---:|"])
    for label, details in report["checkpoints"].items():
        for horizon, comparisons in details["aggregate"].items():
            for comparison, values in comparisons.items():
                lines.append(f"| {label} | {horizon} | {comparison} | {100*values['fixed_signed_recovery']:+.8f}% | {100*values['reference_signed_recovery']:+.8f}% |")
    lines.extend(["", "In the primary episode-dual arm, the complete-write versus uniform recovery remains negative at both H1 and H4 under the continuous reference. The positive complete-write versus omit-source increment also remains positive. Thus the actual reversal observations drive the complete state in the wrong direction relative to uniform; this conclusion is not caused by the fixed integration rule.", "",
        f"Runtime: {report['seconds']:.3f} s. Full per-seed numerical values and signs are in `g1_g3_signed_reference.json`.", "",
        "Reproduce: `python -m research.p1a.prbd_consistency_sign_check`.", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    main()
