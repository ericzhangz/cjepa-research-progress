"""Run the fixed-host full-observation carry check from §25.3."""

import json

import numpy as np
import torch

from . import native_response_mixture_regulation as data
from .native_response_relations_fit import native_rollout


ANCHOR = 110
QUERY = 125
HOLD = 3
MODEL_STEP = 5
OUTPUT = data.RUNS / "native_response_observation_carry_20261115"


def norm_squared(value):
    return np.square(value).sum(axis=-1)


def contrast(value):
    # scene, physical world, mirrored execution program, ...
    return value[:, :, 1] - value[:, :, 0]


def measure(truth, prediction):
    dims = tuple(range(truth.ndim - 2))
    truth_difference = contrast(truth)
    response_difference = contrast(prediction)
    pair_dims = tuple(range(truth_difference.ndim - 2))
    pair_error = norm_squared(response_difference - truth_difference)
    alignment = (response_difference * truth_difference).sum(axis=-1)
    paired_errors = norm_squared(prediction - truth)
    swapped = norm_squared(prediction[:, :, ::-1] - truth)
    np.testing.assert_allclose((swapped - paired_errors).mean(axis=dims), alignment.mean(axis=pair_dims))
    return {
        "endpoint_rmse": np.sqrt(paired_errors.mean(axis=dims)).tolist(),
        "true_history_difference_rms": np.sqrt(norm_squared(truth_difference).mean(axis=pair_dims)).tolist(),
        "predicted_history_difference_rms": np.sqrt(norm_squared(response_difference).mean(axis=pair_dims)).tolist(),
        "history_difference_rmse": np.sqrt(pair_error.mean(axis=pair_dims)).tolist(),
        "swap_minus_correct_endpoint_mse": alignment.mean(axis=pair_dims).tolist(),
        "per_scene_history_difference_mse": pair_error.mean(axis=tuple(range(1, pair_error.ndim - 1))).tolist(),
        "per_scene_swap_minus_correct_mse": alignment.mean(axis=tuple(range(1, alignment.ndim - 1))).tolist(),
    }


@torch.inference_mode()
def run(device):
    panel = data.load_data("development", data.ARCHIVE_DIR, device)
    model = data._native_model(device)
    anchor = ANCHOR // MODEL_STEP
    end = QUERY // MODEL_STEP
    query_index = data.QUERY_STEPS.index(QUERY)
    if not (panel.valid[..., anchor:end].all() and panel.query_valid[..., query_index, :, :].all()):
        raise ValueError("The fixed check requires complete factual and candidate executions.")
    torch.testing.assert_close(panel.visible[..., anchor - 1:anchor + 1, :],
                               panel.full[..., anchor - 1:anchor + 1, :], rtol=0, atol=0)
    torch.testing.assert_close(panel.visible[:, :, 0, end - 2:end + 1],
                               panel.visible[:, :, 1, end - 2:end + 1], rtol=0, atol=0)
    torch.testing.assert_close(panel.commands[:, 0, end - 2:end], panel.commands[:, 1, end - 2:end], rtol=0, atol=0)
    mean, std = data._action_moments(panel.visible)
    normalized_hold = ((torch.zeros_like(mean) - mean) / std).expand(5, -1).flatten()
    recorded_hold = panel.candidates[:, HOLD, :, :]
    torch.testing.assert_close(recorded_hold, normalized_hold.expand_as(recorded_hold), rtol=0, atol=0)

    # Only these observations and the executed command blocks enter the carry path.
    # Future truth is read for scoring after prediction.
    anchor_outputs = []
    query_outputs = {"carry_110": [], "public_125": []}
    for start in range(0, len(panel.scene_ids), 2):
        rows = slice(start, start + 2)
        visible, commands = data._factual_inputs(panel, rows)
        controls = model.action_encoder(commands)
        context = visible[:, anchor - 2:anchor + 1]
        carried = native_rollout(model, context, controls[:, anchor - 2:anchor], controls[:, None, anchor:end])[:, 0]
        public_window, past, candidates = data._query_inputs(panel, rows, (QUERY,))
        actions = model.action_encoder(candidates.flatten(1, 2)).unflatten(1, candidates.shape[1:3])
        past = model.action_encoder(past)
        anchor_outputs.append(carried)
        query_outputs["carry_110"].append(native_rollout(model, carried, past, actions))
        query_outputs["public_125"].append(native_rollout(model, public_window, past, actions))

    prefix_shape = panel.visible.shape[:3]
    predicted_frames = torch.cat(anchor_outputs).unflatten(0, prefix_shape)
    frame_truth = panel.full[..., anchor + 1:end + 1, :]
    frame_error = (predicted_frames - frame_truth).double().square().mean(dim=-1)
    frame_difference_error = (predicted_frames.diff(dim=2) - frame_truth.diff(dim=2)).double().square().mean(dim=-1)

    weight, offset = torch.load(data.WARM_START, map_location=device, weights_only=True)["object_coordinates"]

    def read_object(values):
        return (values.double() @ weight.double() + offset.double()).cpu().numpy()

    truth = panel.object_xy[..., query_index, :, :, :].double().cpu().numpy()
    physical_outputs = {key: read_object(torch.cat(values).unflatten(0, prefix_shape))
                        for key, values in query_outputs.items()}
    physical_outputs["encoded_true_endpoints"] = read_object(panel.future[..., query_index, :, :, :])

    def relative_to_hold(positions):
        return positions[..., :HOLD, :, :] - positions[..., HOLD:HOLD + 1, :, :]

    return {
        "scope": "Fixed official host and 16 historical development scenes; no fitting or partial-view writes.",
        "scene_ids": panel.scene_ids.tolist(),
        "anchor_step": ANCHOR,
        "query_step": QUERY,
        "horizon_environment_steps": [5, 10],
        "carried_observation_steps": list(range(ANCHOR + MODEL_STEP, QUERY + MODEL_STEP, MODEL_STEP)),
        "carried_latent_rmse": frame_error.mean(dim=(0, 1, 2)).sqrt().cpu().tolist(),
        "carried_latent_history_difference_rmse": frame_difference_error.mean(dim=(0, 1, 2)).sqrt().cpu().tolist(),
        "metrics": {key: {
            "absolute_hold": measure(truth[..., HOLD, :, :], predicted[..., HOLD, :, :]),
            "relative_to_hold": measure(relative_to_hold(truth), relative_to_hold(predicted)),
        } for key, predicted in physical_outputs.items()},
    }


if __name__ == "__main__":
    output = run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(OUTPUT / "summary.json")
