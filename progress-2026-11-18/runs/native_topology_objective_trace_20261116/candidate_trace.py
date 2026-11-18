"""Read-only H5 forecast attribution on the two archived candidates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from research.p1a.native_topology_condition import ROOT, RUN, forecast, load_readout, load_trained
from research.p1a.native_topology_mpc import TorchVisibleMapCost
from research.p1a.native_topology_prefix_data import encode_groups, replay_bank
from research.p1a.native_topology_prefix_fit import physical_labels, physical_loss, rank_hinge, wall_intrusion
from research.p1a.physical_topology_gate import CENTER, SPEED, PhysicalRollout, ReversibleFourRoom, VisibleMapCost, make_scene, room


LIVE = ROOT / "runs/native_topology_live_20261116"
CORRECTED = ROOT / "runs/native_topology_accumulated_20261116/corrected"
CHECKPOINTS = {"initial": LIVE / "arm2", "step_900": CORRECTED / "round3"}
SEED, ROOT_INDEX, IDS = 2026112003, 1, (380, 231)


def load_archive():
    path = LIVE / "live_eval_arm2/live_bank.npz"
    with np.load(path, allow_pickle=False) as data:
        i, = np.flatnonzero(data["seed"] == SEED)
        index = np.asarray(IDS) + 4
        one_root = {key: data[key][i, ROOT_INDEX] for key in ("root_xy", "root_z", "root_action")}
        one_root.update({key: data[key][i, ROOT_INDEX, index] for key in ("action", "future_xy", "proposal_xy")})
        return one_root


def native_paths(model, history, past, action, true_latents):
    """Use the normal entry point for free, truthful, and hybrid histories."""
    _, _, gate = model
    if gate.forced_exposure != 1.0:
        raise AssertionError("Expected the archived r=1 deployment path")
    free = forecast(*model, history[None], past[None], action[None], compact=True)[0]
    branches, modes, horizon, _ = true_latents.shape
    hshape = (branches, modes, len(history), history.shape[-1])
    ashape = (branches, len(past), *past.shape[-2:])
    truth_history = torch.cat((history.expand(hshape), true_latents), dim=-2)
    free_history = torch.cat((history.expand(hshape), free), dim=-2)
    controls = torch.cat((past.expand(ashape), action), dim=1)
    origins = torch.arange(modes).expand(branches, modes).reshape(-1)

    def suffix(windows, start):
        selected = forecast(*model, windows.flatten(0, 1),
                            controls[:, None, start:start + 2].expand(-1, modes, -1, -1, -1).flatten(0, 1),
                            action[:, None, None, start:].expand(-1, modes, -1, -1, -1, -1).flatten(0, 1),
                            compact=True)
        return selected[torch.arange(len(origins)), 0, origins].unflatten(0, (branches, modes))

    true_suffix = [suffix(truth_history[..., start:start + 3, :], start) for start in range(horizon)]
    local = torch.stack([prediction[..., 0, :] for prediction in true_suffix], dim=-2)
    torch.testing.assert_close(free[..., 0, :], local[..., 0, :], atol=3e-5, rtol=0)
    # H3 has two predicted slots. Restore each separately to localize exposure to prefix errors.
    restore_h3 = {}
    for slot in (1, 2):
        h3_window = free_history[..., 2:5, :].clone()
        h3_window[..., slot, :] = truth_history[..., 2 + slot, :]
        restore_h3[f"restore_H{slot}_only"] = suffix(h3_window, 2)[..., 0, :]
    return free, local, true_suffix, restore_h3


def gradients(free, truth, fact, free_latent, physical, geometry, gain):
    """Output-space derivatives for the archived two-candidate contrast."""
    raw = fact["future_xy"][None]
    before = np.concatenate((np.broadcast_to(fact["root_xy"][-1], raw[..., :1, -1, :].shape),
                             raw[..., :-1, -1, :]), axis=-2)
    changed = (raw[:, :, 0] != raw[:, :, 1]).any(axis=(-1, -2))
    neutral = ~np.maximum.accumulate(changed, axis=-1)
    centers = np.array([[physical.top, physical.bottom, physical.left, physical.right]], dtype=np.float32)
    labels, _ = physical_labels(raw, before, fact["action"][None], neutral,
                                  (physical.min_xy, physical.max_xy), centers, torch.device("cpu"), walls=True)
    valid, rows = torch.ones((1, 2), dtype=torch.bool), torch.zeros(1, dtype=torch.int64)
    labels["valid"] = valid
    hypothesis = free.detach()[None].requires_grad_(True)
    y = torch.as_tensor(truth)[None]
    rmse, _, overshoot, invariant = physical_loss(hypothesis, free_latent[None], y, labels, rows, gain)
    wall = wall_intrusion(hypothesis, labels["wall_side"], valid)
    backward = {}
    gains = (("coordinate", rmse / gain), ("wall", 10 * wall / gain), ("positive_forward", overshoot / gain),
             ("neutral_coordinate", invariant))
    for name, value in gains:
        backward[name] = {"scaled_value": value.item(), "scaled_descent_xy": -torch.autograd.grad(value, hypothesis)[0][0]}

    terminal = torch.stack((free[0, :, -1], free[0, :, -1], free[0, :, -1], free[1, :, -1]))[None]
    terminal = terminal.detach().requires_grad_(True)
    true = y[:, [0, 0, 0, 1], :, -1]
    score = TorchVisibleMapCost(geometry, "cpu")
    eligibility = torch.zeros_like(true[..., 0], dtype=torch.bool)
    # Both archived modes meet the physical gap and distinct first-command/arrival checks.
    assert (fact["action"][0, 0] != fact["action"][1, 0]).any()
    assert ((truth[0, :, 0] != truth[1, :, 0]).any(axis=-1)).all()
    true_cost = score(true)
    eligibility[:, 0] = true_cost[:, 0] > true_cost[:, 3] + 5
    rlabels = {"rank": eligibility, "maps": {0: score}}
    masked, correct, total = rank_hinge(terminal, true, rows, rlabels)
    cost = score(terminal)
    unmasked = (cost[:, 3] - cost[:, 0] + 5).relu()
    backward["ranking"] = {"eligible_mode": eligibility[0, 0], "factual_room_pairs": correct.item(),
                          "eligible_pairs": total.item(), "hinge_px_per_mode": unmasked[0],
                          "exact_cost_grad_xy": torch.autograd.grad(cost.sum(), terminal, retain_graph=True)[0][0, [0, 3]],
                          "unmasked_descent_xy": -torch.autograd.grad(unmasked.sum(), terminal, retain_graph=True)[0][0, [0, 3]],
                          "actual_descent_xy": -torch.autograd.grad(masked / gain**.5, terminal)[0][0, [0, 3]]}
    return {"wall_side": labels["wall_side"][0], "common_blocked": labels["blocked"][0], **backward}


def analogous_training_cases(physical):
    """Count SE H3 passages through the right opening and factual-source draws."""
    paths = {"initial": CHECKPOINTS["initial"], **{f"round{i}": CORRECTED / f"round{i}" for i in (1, 2, 3)}}
    reports = {}
    for name, path in paths.items():
        with np.load(path / "data_train_live.npz", allow_pickle=False) as data:
            raw, action = data["future_xy"], data["action"]
            root = data["root_xy"][..., -1, :]
            valid = data["candidate_valid"] if "candidate_valid" in data else np.ones(action.shape[:3], dtype=bool)
            start = np.broadcast_to(root[..., None, None, None, :], raw[..., :1, -1, :].shape)
            before = np.concatenate((start, raw[..., :-1, -1, :]), axis=-2)
            predecessor = np.concatenate((before[..., None, :], raw[..., :-1, :]), axis=-2)
            end = raw[..., -1, :]
            free_passage = ((before >= CENTER).all(axis=-1) & (end[..., 0] >= CENTER) & (end[..., 1] <= physical.near)
                            & valid[..., None, None])
            expected = np.clip(predecessor + SPEED * action[:, :, :, None], physical.min_xy, physical.max_xy)
            free_clipped_on_arrival_side = ((np.abs(raw[..., 1] - expected[..., 1]) > 1e-5)
                                            & (predecessor[..., 1] < CENTER)).any(axis=-1)

            local, local_valid = data["local_xy"], data["local_valid"]
            local_valid[..., 0] = False
            changed = (local[..., 0, :, :] != local[..., 1, :, :]).any(axis=(-2, -1))
            static = (local == before[..., None, None, :]).all(axis=(-3, -2, -1))
            neutral_active = local_valid & ~changed & ~static
            end = local[..., -1, :]
            local_passage = ((before >= CENTER).all(axis=-1)[..., None] & (end[..., 0] >= CENTER)
                             & (end[..., 1] <= physical.near) & neutral_active[..., None])
            predecessor = np.concatenate((np.broadcast_to(before[..., None, None, :], local[..., :1, :].shape),
                                          local[..., :-1, :]), axis=-2)
            expected = np.clip(predecessor + SPEED * action[..., None, :, None, :, :], physical.min_xy, physical.max_xy)
            local_clipped_on_arrival_side = ((np.abs(local[..., 1] - expected[..., 1]) > 1e-5)
                                             & (predecessor[..., 1] < CENTER)).any(axis=-1)

            index = (slice(None), 3, slice(None), slice(None), 2)
            passage_rows = local_passage[index].any(axis=-1)
            constrained_rows = (local_passage & local_clipped_on_arrival_side)[index].any(axis=-1)
            draws = 4 * 300 / neutral_active.sum()
            reports[name] = {"controls": int(valid.sum()), "SE_branches": 2 * int(valid[:, 3].sum()),
                             "free_SE_H3_passage_branches": int(free_passage[index].sum()),
                             "free_SE_H3_passage_branches_with_arrival_clip": int((free_passage & free_clipped_on_arrival_side)[index].sum()),
                             "nonstatic_neutral_source_rows": int(neutral_active.sum()),
                             "source_SE_H3_passage_rows": int(passage_rows.sum()),
                             "source_SE_H3_passage_rows_with_arrival_clip": int(constrained_rows.sum()),
                             "expected_source_SE_H3_passage_rows_in_300_updates": draws * int(passage_rows.sum()),
                             "expected_source_SE_H3_passage_rows_with_arrival_clip_in_300_updates": draws * int(constrained_rows.sum())}
    return reports


def serialize(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def evaluate(checkpoints):
    torch.set_num_threads(1)
    saved = load_archive()
    scene = make_scene(SEED)
    physical = PhysicalRollout(scene)
    geometry = VisibleMapCost(scene, scene.northwest)
    env = ReversibleFourRoom(scene)
    decode = load_readout("cpu", RUN)
    truth = saved["future_xy"][..., -1, :]
    np.testing.assert_array_equal(replay_bank(physical, saved["root_xy"][-1], saved["action"]), saved["future_xy"])
    history, past, actions = (torch.as_tensor(saved[key]) for key in ("root_z", "root_action", "action"))
    published = json.loads((CORRECTED / "cand380.json").read_text(encoding="utf-8"))
    gain = torch.load(CHECKPOINTS["step_900"] / "conditioned.pt", map_location="cpu",
                      weights_only=True)["xy_readout_gain_px2"]
    results = {"seed": SEED, "candidate_ids": IDS, "archive_slots": [i + 4 for i in IDS],
               "checkpoints": {name: str(path.resolve()) for name, path in checkpoints.items()},
               "xy_axes": ["candidate", "factual mode", "H1-H5", "coordinate"],
               "loss_gain_px2_per_latent_unit2": gain, "root_xy": saved["root_xy"],
               "goal_xy": scene.northwest, "commands": saved["action"], "Y": truth,
               "physical_primitives": saved["future_xy"], "Y_room": room(truth), "Y_cost": geometry(truth),
               "analogous_training_cases": analogous_training_cases(physical)}
    try:
        env.begin(scene.northwest, scene.southeast, 0)
        for name, path in checkpoints.items():
            print(f"Tracing {name}", flush=True)
            model = load_trained(path, "conditioned", "cpu")
            with torch.no_grad():
                if "C" not in results:
                    (latents,), rendered = encode_groups(env, model[0], (truth,), batch=16,
                                                        known=(saved["root_xy"], saved["root_z"]))
                    factual = torch.as_tensor(latents)
                    results["rendered_frames"] = rendered
                    results["C"] = decode(factual)
                    results["C-Y"] = results["C"].numpy() - truth
                    results["C_room"] = room(results["C"].numpy())
                free_latent, local_latent, suffix, restore = native_paths(model, history, past, actions, factual)
                free, local = map(decode, (free_latent, local_latent))
                published_arm = next((arm for arm, directory in CHECKPOINTS.items() if directory.resolve() == path.resolve()), None)
                if published_arm is not None:
                    np.testing.assert_allclose(free.numpy(), published[published_arm]["xy"], atol=2e-4, rtol=0)
                if published_arm == "initial":
                    np.testing.assert_allclose(free[..., -1, :].numpy(), saved["proposal_xy"], atol=2e-4, rtol=0)
                c = results["C"].numpy().astype(np.float64)
                f, l, y = (x.astype(np.float64) for x in (free.numpy(), local.numpy(), truth))
                np.testing.assert_allclose(f-y, (c-y) + (l-c) + (f-l), atol=1e-10, rtol=0)
                results[name] = {"F": f, "L": l, "C-Y": c-y, "L-C": l-c, "F-L": f-l,
                                 "F_room": room(f), "L_room": room(l),
                                 "F_cost": geometry(f), "L_cost": geometry(l),
                                 "recurrent_latent_RMS_F_minus_L": (free_latent - local_latent).square().mean(-1).sqrt(),
                                 "true_prefix_local_latent_RMS_L_minus_C": (local_latent - factual).square().mean(-1).sqrt(),
                                 "free_after_truth_cut": {str(cut): decode(prediction) for cut, prediction in enumerate(suffix[1:], 1)},
                                 "H3_single_slot_repairs": {key: decode(value) for key, value in restore.items()}}
            results[name]["gradients"] = gradients(free, truth, saved, free_latent, physical, geometry, gain)
    finally:
        env.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", metavar="NAME=DIR", action="append")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_suffix(".json"))
    args = parser.parse_args()
    checkpoints = CHECKPOINTS
    if args.checkpoint:
        checkpoints = {}
        for value in args.checkpoint:
            name, sep, directory = value.partition("=")
            if not (name and sep and directory) or name in checkpoints or not (Path(directory) / "conditioned.pt").is_file():
                parser.error(f"Expected a distinct checkpoint NAME=DIR containing conditioned.pt, got {value}")
            checkpoints[name] = Path(directory)
    results = evaluate(checkpoints)
    args.output.write_text(json.dumps(results, default=serialize, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
