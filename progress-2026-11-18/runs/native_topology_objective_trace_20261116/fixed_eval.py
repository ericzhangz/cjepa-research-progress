"""Evaluate H5 native prediction and first actions on the two archived CEM populations."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.p1a.native_topology_condition import ROOT, RUN, forecast, load_readout, load_trained
from research.p1a.native_topology_paired_data import clipping
from research.p1a.native_topology_paired_eval import first_command_report
from research.p1a.native_topology_prefix_eval import SEEDS, ranking_report, reduce_ranks
from research.p1a.physical_topology_gate import CENTER, ITERATIONS, SAMPLES, PhysicalRollout, VisibleMapCost, make_scene, room


LIVE = ROOT / "runs/native_topology_live_20261116"
CORRECTED = ROOT / "runs/native_topology_accumulated_20261116/corrected"
BANKS = {"original_CEM": LIVE / "live_eval_arm2/live_bank.npz",
         "corrected_CEM": CORRECTED / "live_eval/live_bank.npz"}


def load_bank(path):
    with np.load(path, allow_pickle=False) as entries:
        data = {key: entries[key] for key in ("seed", "root_xy", "root_z", "root_action", "action", "future_xy", "proposal_xy")}
    np.testing.assert_array_equal(data["seed"], SEEDS)
    if data["action"].shape != (len(SEEDS), 2, 4 + ITERATIONS * SAMPLES, 5, 5, 2):
        raise ValueError(f"Expected two navigation roots and the archived H5 budget in {path}")
    for key in ("action", "future_xy", "proposal_xy"):
        data[key] = data[key][:, :, 4:]
    return data


@torch.inference_mode()
def predict(model, decode, bank, device):
    _, _, gate = model
    if gate.forced_exposure != 1.0:
        raise ValueError("Expected the forced r=1 Arm 2 path")
    shape = bank["future_xy"].shape
    result = np.empty(shape[:-2] + (2,), dtype=np.float32)
    for i in range(len(bank["seed"])):
        for root in range(2):
            history, past, action = (torch.as_tensor(bank[key][i, root], device=device)[None]
                                     for key in ("root_z", "root_action", "action"))
            for start in range(0, action.shape[1], SAMPLES):
                slc = slice(start, start + SAMPLES)
                rolled = forecast(*model, history, past, action[:, slc], compact=True)
                xy = decode(rolled[0])
                xy[..., -1, :] = decode(rolled[0, ..., -1, :])
                result[i, root, slc] = xy.cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite fixed-bank output")
    return result


def progress(truth, predicted, direction, mask):
    assert truth.shape == predicted.shape and mask.shape == truth.shape[:-1]
    true = (truth * direction).sum(axis=-1)[mask]
    estimate = (predicted * direction).sum(axis=-1)[mask]
    return {"n": int(mask.sum()), "true_px": float(true.mean()) if len(true) else None,
            "predicted_px": float(estimate.mean()) if len(true) else None,
            "forecast_minus_physical_px": float((estimate - true).mean()) if len(true) else None}


def report(bank, predicted):
    actual_raw = bank["future_xy"]
    actual = actual_raw[..., -1, :]
    start = bank["root_xy"][..., -1, :]
    goals = np.broadcast_to(np.array([2, 0]), start.shape[:2])
    actual_cost, pred_cost = [np.empty(actual.shape[:-1], dtype=np.float32) for _ in range(2)]
    start_cost = np.empty(start.shape[:2], dtype=np.float32)
    common_blocked = np.empty(actual.shape[:3], dtype=bool)
    for i, seed in enumerate(bank["seed"]):
        scene = make_scene(int(seed))
        for j, goal in enumerate((scene.southeast, scene.northwest)):
            geometry = VisibleMapCost(scene, goal)
            actual_cost[i, j], pred_cost[i, j] = geometry(actual[i, j]), geometry(predicted[i, j])
            start_cost[i, j] = geometry(start[i, j])
            raw, action = actual_raw[i, j], bank["action"][i, j]
            hits = clipping(start[i, j], action, raw)
            common_blocked[i, j] = (raw[:, 0] == raw[:, 1]).all(axis=(1, 2, 3)) & hits[:, 0, -5:].any(axis=-1)

    terminal, forecasted = actual[..., -1, :], predicted[..., -1, :]
    final_true_cost, final_pred_cost = actual_cost[..., -1], pred_cost[..., -1]
    goal_true = room(terminal) == goals[..., None, None]
    goal_hat = room(forecasted) == goals[..., None, None]
    rank = ranking_report(final_pred_cost, final_true_cost, forecasted, terminal, goals)
    selected = rank["selected_index"][..., None, :]
    selected_actual_goal = np.take_along_axis(goal_true, selected, axis=2).squeeze(axis=2)
    selected_forecast_goal = np.take_along_axis(goal_hat, selected, axis=2).squeeze(axis=2)
    by_first = first_command_report(bank["action"], final_pred_cost, final_true_cost)

    starts = np.broadcast_to(start[..., None, None, None, :], actual[..., :1, :].shape)
    prior = np.concatenate((starts, actual[..., :-1, :]), axis=-2)
    actual_move, estimate_move = actual - starts, predicted - starts
    magnitude = np.linalg.norm(actual_move, axis=-1, keepdims=True)
    travel = np.divide(actual_move, magnitude, out=np.zeros_like(actual_move), where=magnitude > 0)
    room_change = (room(actual_raw) != room(starts)[..., None]).any(axis=(-1, -2))
    legal_route = room_change & (final_true_cost < start_cost[..., None, None])
    by_h = [progress(actual_move[..., h, :], estimate_move[..., h, :], travel[..., h, :],
                     legal_route & (magnitude[..., h, 0] > 0)) for h in range(actual.shape[-2])]
    face = PhysicalRollout(make_scene(int(bank["seed"][0]))).near
    face_far = 2 * CENTER - face
    low_cross = (prior >= CENTER) & (actual <= face)
    high_cross = (prior < CENTER) & (actual >= face_far)
    passed = low_cross | high_cross
    predicted_passage = low_cross & (predicted <= face) | high_cross & (predicted >= face_far)
    intent = bank["action"][..., -1, :, :].sum(axis=-2)
    length = np.linalg.norm(intent, axis=-1, keepdims=True)
    direction = np.divide(intent, length, out=np.zeros_like(intent), where=length > 0)
    common = progress(terminal - prior[..., -1, :], forecasted - prior[..., -1, :], direction[..., None, :],
                      np.broadcast_to(common_blocked[..., None], terminal.shape[:-1]))

    return {"goal_room": {"all_branches": int(goal_true.size), "actual_goal_available_queries": int(goal_true.any(axis=2).sum()),
                          "actual_goal_branches": int(goal_true.sum()), "predicted_goal_branches": int(goal_hat.sum()),
                          "true_positive_branches": int((goal_true & goal_hat).sum()),
                          "false_positive_branches": int((~goal_true & goal_hat).sum()),
                          "false_negative_branches": int((goal_true & ~goal_hat).sum()),
                          "selected_actual_goal": int(selected_actual_goal.sum()),
                          "selected_forecast_goal": int(selected_forecast_goal.sum()),
                          "selected_false_positive": int((~selected_actual_goal & selected_forecast_goal).sum()),
                          "selected_false_negative": int((selected_actual_goal & ~selected_forecast_goal).sum())},
            "passage": {"actual_directed_crossings_by_h": passed.sum(axis=(0, 1, 2, 3, 5)).tolist(),
                        "correctly_predicted_crossings_by_h": predicted_passage.sum(axis=(0, 1, 2, 3, 5)).tolist(),
                        "missed_crossings_by_h": (passed & ~predicted_passage).sum(axis=(0, 1, 2, 3, 5)).tolist()},
            "legal_route_displacement_by_h": by_h, "common_blocked_H5_projection": common,
            "ranking": reduce_ranks(rank), "first_command": by_first}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", metavar="NAME=DIR", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arrays", type=Path, help="Store the complete fixed H5 predictions")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoints = {}
    for value in args.checkpoint:
        name, sep, directory = value.partition("=")
        if not (name and sep and directory) or name in checkpoints or not (Path(directory) / "conditioned.pt").is_file():
            parser.error(f"Expected a distinct checkpoint NAME=DIR containing conditioned.pt, got {value}")
        checkpoints[name] = Path(directory)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    decode = load_readout(device, RUN)
    banks = {name: load_bank(path) for name, path in BANKS.items()}
    for key in ("seed", "root_xy", "root_z", "root_action"):
        np.testing.assert_array_equal(banks["original_CEM"][key], banks["corrected_CEM"][key])
    results = {"banks": {name: str(path.resolve()) for name, path in BANKS.items()},
               "checkpoints": {name: str(path.resolve()) for name, path in checkpoints.items()},
               "horizon": 5, "queries_per_bank": 64, "CEM_candidates_per_query": ITERATIONS * SAMPLES, "results": {}}
    arrays = []
    for name, checkpoint in checkpoints.items():
        model = load_trained(checkpoint, "conditioned", device)
        results["results"][name] = {}
        arms = []
        for bank_name, bank in banks.items():
            print(f"{name} on {bank_name}", flush=True)
            predicted = predict(model, decode, bank, device)
            matches = ((bank_name == "corrected_CEM" and checkpoint.resolve() == (CORRECTED / "round3").resolve())
                       or (bank_name == "original_CEM" and checkpoint.resolve() == (LIVE / "arm2").resolve()))
            if matches:
                np.testing.assert_allclose(predicted[..., -1, :], bank["proposal_xy"], rtol=0, atol=2e-4)
            results["results"][name][bank_name] = report(bank, predicted)
            if args.arrays:
                arms.append(predicted)
        del model
        if args.arrays:
            arrays.append(np.stack(arms))
    if args.arrays:
        np.savez_compressed(args.arrays, F=np.stack(arrays), checkpoint=np.asarray(list(checkpoints)),
                            bank=np.asarray(list(banks)), seed=SEEDS, root=["northwest", "southeast"],
                            candidate=np.arange(ITERATIONS * SAMPLES), mode=[0, 1], horizon=np.arange(1, 6), coordinate=["x", "y"])
        results["forecast_arrays"] = str(args.arrays.resolve())
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
