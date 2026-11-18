"""Summarize the frozen Stage A replay using actual candidate scores and physical costs."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
QUAL = ROOT / "runs/stage_a_fourroom_20261117"
REPLAY = QUAL / "baseline_replay"
OUT = Path(__file__).with_name("replay_checks.json")
ARMS = ("frozen", "adajepa", "optta", "gru")
ZERO = 1e-5


def mean(vals):
    return float(np.mean(vals)) if vals else None


def regrets(rows, key="first_block_regret_by_mode"):
    return [row["selection"][key][mode] for row, mode in rows]


def phase(rows, label):
    return [(row, mode) for row in rows for name, mode in row["phase_labels"] if name == label]


def phase_summary(rows):
    observed = regrets(rows)
    if any(value is None for value in observed):
        raise ValueError("Factual mode must have a defined physical cost")
    returns = {"count": len(rows), "exact_best_first_block": sum(v <= ZERO for v in observed),
               "mean_first_regret": mean(observed), "mean_plan_regret": mean(regrets(rows, "full_plan_regret_by_mode"))}
    returns["mean_factual_arrival_latent_mse"] = mean([row["prequential_arrival_mse"] for row, _ in rows
                                                       if row["prequential_arrival_mse"] is not None])
    before = [(row["before_update"]["first_block_regret_by_mode"][mode], selected)
              for (row, mode), selected in zip(rows, observed, strict=True) if "before_update" in row]
    returns["fixed_latest_window_comparison"] = {
        "count": len(before), "better": sum(b < a - ZERO for a, b in before),
        "worse": sum(b > a + ZERO for a, b in before), "mean_after_minus_before_regret": mean([b-a for a,b in before]),
    }
    return returns


def main():
    protocol = json.loads((QUAL / "physical.json").read_text(encoding="utf-8"))
    trace = [json.loads(line) for line in (REPLAY / "rounds.jsonl").open(encoding="utf-8") if line.strip()]
    assert len(trace) == 32, f"Expected 8 episodes with 4 arms, got {len(trace)} rows"
    forks = {row["task"]: row for row in protocol["forks"]}
    arms = {}
    for arm in ARMS:
        episodes = {record["episode"]: record for record in trace if record["arm"] == arm}
        assert len(episodes) == 8
        joint = [round_ for episode in episodes.values() for round_ in episode["rows"]]
        assert len(joint) == 248 and all(row["selection_candidates"] == 384 for row in joint)
        quiet = phase(joint, "neutral_control")
        retained = [pair for r in joint for pair in ((r, m) for key, m in r["phase_labels"] if key.endswith("_retained_fork"))]
        assert len(retained) == 24
        orders = {order: [r for ep in episodes.values() if ep["order"] == order for r in ep["rows"]]
                  for order in ("ABA", "BAB")}
        paired = []
        for task, fork in forks.items():
            a, b = (episodes[f"{task}_{order}"]["rows"] for order in ("ABA", "BAB"))
            k0, k1 = fork["true_indices"]
            for phase_index, name in enumerate(("initial_probe_retained_fork", "reversal_1_retained_fork", "reversal_2_retained_fork")):
                (left, mode_a), = phase(a, name)
                (right, mode_b), = phase(b, name)
                assert left["root"] == right["root"] and mode_a != mode_b
                if arm == "frozen":
                    np.testing.assert_allclose(left["selection"]["scores"], right["selection"]["scores"], atol=1e-6, rtol=0)
                gaps = [row["selection"]["scores"][k1] - row["selection"]["scores"][k0] for row in (left, right)]
                correct = [float((1 - 2 * m) * gap) > ZERO for m, gap in zip((mode_a, mode_b), gaps, strict=True)]
                own = [left["selection"]["first_block_regret_by_mode"][mode_a],
                       right["selection"]["first_block_regret_by_mode"][mode_b]]
                swapped = [right["selection"]["first_block_regret_by_mode"][mode_a],
                           left["selection"]["first_block_regret_by_mode"][mode_b]]
                own_gain = [other - self for other, self in zip(swapped, own, strict=True)]
                paired.append({"own_history_regret": own, "swapped_history_regret": swapped,
                               "own_history_regret_gain": own_gain, "fork": task, "phase": phase_index, "models_selected_distinct_first_blocks":
                               left["selection"]["first_block_group"] != right["selection"]["first_block_group"],
                               "oracle_two_route_predicted_gaps": gaps, "correct_direction": correct,
                               "both_route_directions_correct": all(correct)})
        arms[arm] = {
            "neutral": phase_summary(quiet),
            "retained": phase_summary(retained),
            "retained_by_unseen_order": {order: phase_summary(
                [(r, m) for r in rows for label, m in r["phase_labels"] if label.endswith("_retained_fork")])
                for order, rows in orders.items()},
            "reversal_probe_arrivals": {f"reversal_{rev}_probe_arrival_{arr}": phase_summary(
                phase(joint, f"reversal_{rev}_probe_arrival_{arr}")) for rev in (1, 2) for arr in (1, 2)},
            "retained_complementary_pairs": paired,
            "own_history_by_fork_phase": {
                str(f): {"mean_gain": mean(gains), "better": sum(v > ZERO for v in gains),
                         "worse": sum(v < -ZERO for v in gains), "count": len(gains)}
                for f in range(3)
                for gains in [[g for row in paired if row["phase"] == f for g in row["own_history_regret_gain"]]]
            },
            "retained_pairs_both_route_directions_correct": sum(row["both_route_directions_correct"] for row in paired),
            "retained_pairs_different_first_block": sum(row["models_selected_distinct_first_blocks"] for row in paired),
        }
    OUT.write_text(json.dumps(arms, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
