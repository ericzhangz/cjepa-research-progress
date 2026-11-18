"""Decompose saved PushT step-125 prediction errors by paired history."""

import json
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = RUN_DIR.with_name("native_response_mixture_counterevidence_20261113")
QUERY_STEP = 125


def error_terms(predicted, observed):
    # Axes: scene, physical world, mirror program, candidate, horizon, XY.
    error = predicted - observed
    total = np.square(error).sum(axis=-1).mean(axis=(1, 2, 3))
    midpoint = np.square(error.mean(axis=2)).sum(axis=-1).mean(axis=(1, 2))
    history_gap = (0.25 * np.square(error[:, :, 1] - error[:, :, 0])).sum(axis=-1)
    history_gap = history_gap.mean(axis=(1, 2))
    swapped = np.square(predicted[:, :, ::-1] - observed).sum(axis=-1).mean(axis=(1, 2, 3))
    prediction_delta = predicted[:, :, 1] - predicted[:, :, 0]
    observed_delta = observed[:, :, 1] - observed[:, :, 0]
    alignment = (prediction_delta * observed_delta).sum(axis=-1).mean(axis=(1, 2))
    np.testing.assert_allclose(total, midpoint + history_gap, atol=1e-10, rtol=0)
    np.testing.assert_allclose(swapped - total, alignment, atol=1e-10, rtol=0)
    return dict(total=total, midpoint=midpoint, history_gap=history_gap, swap_minus_correct=alignment)


def summarize(predicted, observed, scene_ids):
    write = error_terms(predicted["write"], observed)
    skip = error_terms(predicted["skip_late"], observed)
    changes = {key: write[key] - skip[key] for key in ("total", "midpoint", "history_gap")}
    return {
        "write_rmse": np.sqrt(write["total"].mean(axis=0)).tolist(),
        "skip_late_rmse": np.sqrt(skip["total"].mean(axis=0)).tolist(),
        "write_minus_skip_late_mean_squared": {key: value.mean(axis=0).tolist() for key, value in changes.items()},
        "scene_count_improved": {key: (value < 0).sum(axis=0).tolist() for key, value in changes.items()},
        "swap_minus_correct_mean_squared": {
            "write": write["swap_minus_correct"].mean(axis=0).tolist(),
            "skip_late": skip["swap_minus_correct"].mean(axis=0).tolist(),
        },
        "per_scene": {
            str(scene_id): {
                "write": {key: value[row].tolist() for key, value in write.items()},
                "skip_late": {key: value[row].tolist() for key, value in skip.items()},
            }
            for row, scene_id in enumerate(scene_ids)
        },
    }


def analyze_split(split):
    with (
        np.load(RUN_DIR / f"{split}_paired_intermediates.npz", allow_pickle=False) as paired,
        np.load(ARCHIVE_DIR / f"{split}_latents.npz", allow_pickle=False) as encoded,
        np.load(ARCHIVE_DIR / f"{split}_physics.npz", allow_pickle=False) as physics,
    ):
        scene_ids = paired["scene_id"]
        query = np.flatnonzero(paired["query_steps"] == QUERY_STEP).item()
        np.testing.assert_array_equal(encoded["scene_ids"], scene_ids)
        np.testing.assert_array_equal(physics["scene_ids"], scene_ids)
        np.testing.assert_array_equal(physics["query_steps"][-1], QUERY_STEP)
        np.testing.assert_array_equal(physics["observation_steps"][-3:], [115, 120, 125])
        np.testing.assert_array_equal(physics["horizon_steps"], [5, 10])
        visible = encoded["visible"]
        frames = physics["obs_rgb"]
        commands = physics["executed_actions"]
        np.testing.assert_array_equal(visible[:, :, 0, -3:], visible[:, :, 1, -3:])
        np.testing.assert_array_equal(frames[:, :, 0, -3:], frames[:, :, 1, -3:])
        np.testing.assert_array_equal(commands[:, 0, -2:], commands[:, 1, -2:])
        np.testing.assert_array_equal(physics["query_actions"][:, 3], 0)
        if not paired["valid"][:, :, :, query].all():
            raise ValueError("The comparison requires all actual candidate futures.")

        truth = paired["truth_xy"][:, :, :, query].astype(np.float64)
        predictions = {
            arm: paired[f"{arm}_xy"][:, :, :, query].astype(np.float64)
            for arm in ("write", "skip_late")
        }

    def center_candidates(values):
        return values - values.mean(axis=3, keepdims=True)

    def relative_to_hold(values):
        return values[:, :, :, :3] - values[:, :, :, 3:4]

    def absolute_hold(values):
        return values[:, :, :, 3:4]

    return {
        "scenes": len(scene_ids),
        "mirror_pairs": truth.shape[0] * truth.shape[1],
        "metrics": {
            name: summarize({arm: transform(values) for arm, values in predictions.items()}, transform(truth), scene_ids)
            for name, transform in (
                ("candidate_centered", center_candidates),
                ("relative_to_hold", relative_to_hold),
                ("absolute_hold", absolute_hold),
            )
        },
    }


if __name__ == "__main__":
    print(json.dumps({
        "query_step": QUERY_STEP,
        "horizon_environment_steps": [5, 10],
        "splits": {split: analyze_split(split) for split in ("train", "development")},
    }, indent=2, allow_nan=False))