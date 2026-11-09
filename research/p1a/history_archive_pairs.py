"""Find FourRoom archive pairs with different executed histories and the same raw suffix."""

import argparse
import json
from pathlib import Path

import numpy as np


def main(args):
    with np.load(args.source / "physical.npz", allow_pickle=False) as archive:
        data = {key: archive[key] for key in (
            "root_id", "candidate_id", "raw_actions", "prefix_absolute_step", "absolute_step",
            "agent_xy", "distractor_xy", "closed_door_idx", "de", "dd", "hit",
        )}
    roots = [json.loads(line) for line in (args.source / "roots.jsonl").read_text(encoding="utf-8").splitlines()]
    np.testing.assert_array_equal(data["root_id"], [root["root_id"] for root in roots])
    actions = data["raw_actions"]
    n_roots, n_candidates, n_blocks, block_size, _ = actions.shape
    first = data["prefix_absolute_step"][-1]
    rendered_steps = first + block_size * np.arange(1, n_blocks + 1)
    # No query at the shared root prefix or after exhausting the action bank.
    event_steps = rendered_steps[:-1]
    step_index = np.searchsorted(data["absolute_step"], event_steps)
    np.testing.assert_array_equal(data["absolute_step"][step_index], event_steps)
    left, right = np.triu_indices(n_candidates, k=1)
    same_block = np.all(actions[:, left] == actions[:, right], axis=(-2, -1))
    different_past = ~np.logical_and.accumulate(same_block, axis=-1)[..., :-1]
    suffix_from_block = np.logical_and.accumulate(same_block[..., ::-1], axis=-1)[..., ::-1]
    same_suffix = suffix_from_block[..., 1:]
    pairs = []
    for i, root in enumerate(roots):
        with np.load(args.observations / f"{root['root_id']}.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["root_id"], root["root_id"])
            np.testing.assert_array_equal(archive["candidate_id"], data["candidate_id"])
            np.testing.assert_array_equal(archive["absolute_step"], rendered_steps)
            rgb = archive["rgb"]
        for pair, event in zip(*np.where(different_past[i] & same_suffix[i])):
            a, b, t = left[pair], right[pair], step_index[event]
            equal = {
                f"same_current_{name}": bool(np.array_equal(data[name][i, a, t], data[name][i, b, t]))
                for name in ("agent_xy", "distractor_xy", "closed_door_idx")
            }
            equal.update({
                f"same_future_{name}": bool(np.array_equal(data[name][i, a, t + 1:], data[name][i, b, t + 1:]))
                for name in ("agent_xy", "distractor_xy", "closed_door_idx", "de", "dd", "hit")
            })
            pairs.append({
                "root_id": root["root_id"], "candidate_a": int(data["candidate_id"][a]),
                "candidate_b": int(data["candidate_id"][b]), "absolute_step": int(event_steps[event]),
                "same_current_rgb": bool(np.array_equal(rgb[a, event], rgb[b, event])),
                "different_past_agent_xy": bool(not np.array_equal(data["agent_xy"][i, a, :t], data["agent_xy"][i, b, :t])),
                "different_available_past_rgb": any(not np.array_equal(rgb[a, j], rgb[b, j]) for j in range(event)),
                **equal,
            })
    columns = tuple(pairs[0])
    table = {key: np.array([pair[key] for pair in pairs]) for key in columns}
    same_rgb = table["same_current_rgb"]
    same_xy = table["same_current_agent_xy"]
    future_fields = [key for key in columns if key.startswith("same_future_")]
    panel = {
        "physical_archive": str((args.source / "physical.npz").resolve()),
        "root_world_configs": str((args.source / "roots.jsonl").resolve()),
        "rendered_archive": str(args.observations.resolve()),
        "roots": n_roots, "candidates_per_root": n_candidates,
        "original_prefix_end": int(first), "raw_steps_per_model_step": block_size,
        "query_absolute_steps": event_steps.tolist(), "final_absolute_step": int(rendered_steps[-1]),
        "pair_unit": "one unordered pair of candidates within the same root at the same absolute step",
        "pair_definition": "at least one different executed raw ego action after the shared prefix; all unexecuted raw controls equal element by element",
        "same_world_parameters_by_construction": ["reset seed and initial conditions", "full environment configuration", "goal position"],
        "same_distractor_coordinates_across_all_root_candidates_and_raw_steps": bool(np.all(data["distractor_xy"] == data["distractor_xy"][:, :1])),
        "same_door_index_across_all_root_candidates_and_raw_steps": bool(np.all(data["closed_door_idx"] == data["closed_door_idx"][:, :1])),
        "agent_speed_world_configs": sorted({root["world_config"]["agent_speed"] for root in roots}),
        "pairs_with_different_past_and_same_suffix": len(pairs),
        "same_current_agent_xy_pairs": int(same_xy.sum()),
        "same_current_rgb_pairs": int(same_rgb.sum()),
        "same_current_rgb_but_different_agent_xy": int((same_rgb & ~same_xy).sum()),
        "same_current_rgb_and_different_past_agent_xy_pairs": int((same_rgb & table["different_past_agent_xy"]).sum()),
        "same_current_rgb_and_different_available_past_rgb_pairs": int((same_rgb & table["different_available_past_rgb"]).sum()),
        "same_current_rgb_and_different_future_counts": {key.removeprefix("same_future_"): int((same_rgb & ~table[key]).sum()) for key in future_fields},
        "same_current_rgb_pairs_by_step": {str(step): int((same_rgb & (table["absolute_step"] == step)).sum()) for step in event_steps},
        "limits": [
            "This panel covers only the archived P1A assessment roots and original fixed action bank; pairs share roots and trajectories.",
            "The public ego transition uses the current position, ordered actions and fixed speed and wall/door geometry. It has no actuator inertia or state dependent on previous ego commands.",
            "Within each root and time, candidates share a distractor trajectory from the same reset and seed; this panel cannot test histories that change hidden distractor heading or dynamics.",
            "Past RGB comparisons use only available dense model-step frames; historical action comparisons use every ordered raw control after the shared prefix.",
        ],
        "pair_arrays": "archive_pair_indices.npz",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "archive_pair_indices.npz", **table)
    (args.output / "archive_pairs.json").write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in panel.items() if key.startswith("same_current")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
