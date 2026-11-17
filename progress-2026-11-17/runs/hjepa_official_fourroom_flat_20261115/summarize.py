"""Verify and export the author evaluator's per-task results."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="Require all 50 tasks and save the table")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    rows = []
    for index in range(50):
        path = root / "chunks" / f"tasks_{index:03}-{index + 1:03}.json"
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        success, = data["episode_successes"]
        steps, = data["steps_to_success"]
        wall_to_success, = data["wall_clock_to_success"]
        assert type(success) is bool
        assert math.isfinite(steps) == success
        assert math.isfinite(wall_to_success) == success
        assert not success or 1 <= steps <= 210
        assert data["success_rate"] == 100 * success
        assert data["loaded_eval_original_lengths"] == [76]
        assert data["seeds"] is None
        solves = data["solve_records"]
        assert solves[0]["horizon"] == 23 and solves[-1]["horizon"] == 1
        assert len(solves) == 22
        assert all(r["level"] == 1 and r["num_samples"] == 200 for r in solves)
        assert all(len(r["n_iters"]) == 1 and 30 <= r["n_iters"][0] <= 90 for r in solves)
        rows.append({
            "task_index": index,
            "planner_seed": 42 + index,
            "success": success,
            "steps_to_success": int(steps) if success else None,
            "wall_seconds_to_success": wall_to_success if success else None,
            "wall_seconds_eval": data["evaluation_time"],
            "planning_calls": len(solves),
            "optimizer_steps_total": sum(r["n_iters"][0] for r in solves),
        })
    successful_steps = [r["steps_to_success"] for r in rows if r["success"]]
    summary = {"completed": len(rows), "successes": len(successful_steps)}
    if rows:
        elapsed = [row["wall_seconds_eval"] for row in rows]
        summary["median_eval_seconds_per_task"] = statistics.median(elapsed)
        summary["sum_eval_seconds"] = sum(elapsed)
    if len(rows) == 50:
        summary["success_rate_percent"] = 100 * len(successful_steps) / 50
        summary["mean_steps_to_success"] = statistics.mean(successful_steps) if successful_steps else None
    print(json.dumps(summary))
    if args.write:
        assert len(rows) == 50, f"Only {len(rows)} official task files are complete"
        with (root / "tasks.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
