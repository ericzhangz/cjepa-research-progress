"""Matched-current-frame FourRoom histories and a frozen information ladder."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from p1a_reference import latent_rollout
from readout_feedback import ReadoutFeedback


def collect(config_path, output):
    import gymnasium as gym
    import collect_branches  # Registers the same FourRoom environment as P1A.

    root = json.loads(config_path.read_text(encoding="utf-8").splitlines()[0])
    config = root["world_config"]
    rng = np.random.default_rng(8110)
    plan = np.repeat(np.array([[0., 0.], [0., 0.], [.35, .10], [.10, .35]], np.float32)[:, None], 5, axis=1)
    grid = np.array([(x, y) for x in range(44, 66) for y in range(44, 66)], np.float32)
    rng.shuffle(grid)
    frames, paths, jumps, seeds, anchors = [], [], [], [], []
    torch.set_num_threads(1)
    for center in grid:
        seed = int(rng.integers(0, 1_000_000_000))
        axis = int(rng.integers(0, 2))
        directions = np.eye(2, dtype=np.float32)[axis] * rng.choice([-1, 1])
        states = []
        try:
            for unit in (directions, -directions):
                initial = center - unit * 10 * config["distractor_speed"]
                sim = gym.make("swm/FourRoomDistractors-v0", max_episode_steps=400, **config)
                states.append(sim)
                slots = np.full((sim.unwrapped.num_distractors, 2), np.nan, np.float32)
                slots[0] = initial
                sim.reset(seed=seed, options={"state": [154, 155], "target_state": [178, 178], "distractor_xy": slots})
                sim.unwrapped._distractor_controller.heading[0] = np.arctan2(unit[1], unit[0])
            rgb = [[sim.render()] for sim in states]
            xy = [[sim.unwrapped._get_info()["all_agent_xy"][:2].copy()] for sim in states]
            switches = np.zeros((2, 4), dtype=bool)
            for i in range(4):
                if i == 2 and (switches[:, :2].any() or not np.array_equal(rgb[0][-1], rgb[1][-1])
                               or not np.array_equal(xy[0][-1], xy[1][-1])):
                    break
                for side, sim in enumerate(states):
                    controller = sim.unwrapped._distractor_controller
                    for action in plan[i]:
                        heading = float(controller.heading[0])
                        _, _, terminated, truncated, info = sim.step(action)
                        if terminated or truncated:
                            raise RuntimeError("Unexpected terminal state in the matched suffix.")
                        turn = float(controller.heading[0]) - heading
                        switches[side, i] |= abs(np.arctan2(np.sin(turn), np.cos(turn))) > 1e-4
                    rgb[side].append(sim.render())
                    xy[side].append(info["all_agent_xy"][:2].copy())
            if len(rgb[0]) != 5:
                continue
            frames.append(np.stack(rgb))
            paths.append(np.stack(xy))
            jumps.append(switches)
            seeds.append(seed)
            anchors.append(center)
        finally:
            for sim in states:
                sim.close()
        if len(frames) == 96:
            break
    if len(frames) != 96:
        raise RuntimeError(f"Collected only {len(frames)} exact pairs.")
    with output.open("xb") as stream:
        np.savez_compressed(stream, rgb=frames, xy=paths, teleport=jumps, seeds=seeds, anchor=anchors,
                            actions=plan, absolute_step=[0, 5, 10, 15, 20])
    print(f"Collected {len(frames)} exact pairs: {output}", flush=True)


def contrast(value):
    return value[:, 0] - value[:, 1]


def fit_readout(feature, target, split):
    # A fixed linear diagnostic, fit on training pairs only. No intercept for contrasts.
    x = feature[:split].astype(np.float64)
    gram = x @ x.T
    scale = np.trace(gram) / len(gram)
    return feature @ x.T @ np.linalg.solve(gram + np.eye(len(gram)) * (0.01 * scale), target[:split])


def load_host(args):
    sys.path[:0] = [str(args.model_dir / "upstream/h_jepa"), str(args.model_dir / "probe_deps"), str(args.model_dir / "deps")]
    from host_adapter import FrozenHost

    return FrozenHost(args.model_dir / "fourroom_lewm_object.ckpt", args.model_dir / "normalizer.pt", args.probe)


@torch.inference_mode()
def diagnose(args):
    host = load_host(args)
    from host_adapter import Timings

    model = host.model
    operator = ReadoutFeedback(model)
    with np.load(args.out / "pairs.npz", allow_pickle=False) as data:
        rgb = data["rgb"]
        xy = data["xy"]
        teleports = data["teleport"][:, :, 2:]
        raw_actions = data["actions"]
    timing = Timings()
    with timing.stage("matched_observations"):
        z = host.encode(rgb, timing)
    # Layout is pair, side, model time. The same action suffix is copied to every side.
    total = len(z) * 2
    z = z.flatten(0, 1)
    actions = torch.as_tensor(host.normalize(raw_actions), dtype=z.dtype, device=z.device)[None].expand(total, -1, -1)
    embedded = model.action_embed(model.action_encoder(actions))
    observed = z[:, :3]
    native, hidden, _ = operator.sequence(z[:, 0], observed, embedded, None, "full", 2)
    native3 = latent_rollout(model.predict, model.action_embed, observed, actions[:, :2], actions[:, 2:], 3)
    with np.load(args.previous, allow_pickle=False) as stored:
        basis = torch.as_tensor(stored["low_rank_kernel_basis"], device=z.device)
        gain = torch.as_tensor(stored["low_rank_kernel_gains"][0, 0], device=z.device)
    feedback, compact, _ = operator.sequence(z[:, 0], observed, embedded, gain, "low_rank", 2, basis=basis)
    erased, _, _ = operator.sequence(z[:, 0], observed, embedded, gain, "low_rank", 2, "erase", basis)
    # Arrange sides across the batch so the existing half-batch swap exchanges each exact pair.
    order = torch.arange(total, device=z.device).reshape(-1, 2).T.flatten()
    swapped, _, _ = operator.sequence(z[order, 0], observed[order], embedded[order], gain, "low_rank", 2, "swap", basis)
    swapped = swapped[order.argsort()]

    def paired(t):
        return t.cpu().numpy().reshape(len(rgb), 2, *t.shape[1:])

    # The t10 carry was written by the previous real interval, from t5 and its executed action.
    features = {"past_latent": contrast(paired(z[:, 1])), "native_activation": contrast(paired(hidden[:, 1])),
                "native_null": contrast(paired(operator.kernel(hidden[:, 1]))),
                "low_rank_carry": contrast(paired(compact[:, 1]))}
    z_pair = paired(z)
    actual = contrast(z_pair[:, :, 3:])
    xy_delta = contrast(xy[:, :, 3:, 1])
    # Pixel-only diagnostic: blue minus red isolates the past distractor in this matched room.
    blue = np.maximum(rgb[:, :, 1:3, :, :, 2].astype(np.int16) - rgb[:, :, 1:3, :, :, 0], 0)
    coordinate = np.arange(224, dtype=np.float64)
    mass = blue.sum((-1, -2))
    centroid = np.stack(((blue.sum(-2) @ coordinate) / mass, (blue.sum(-1) @ coordinate) / mass), -1)
    motion = contrast(centroid[:, :, 1] - centroid[:, :, 0])
    arrays = dict(z=z_pair, xy=xy, truth_delta=actual, physical_delta=xy_delta, pixel_motion_delta=motion, teleport=teleports)
    split = 72
    hold = np.arange(split, len(rgb))
    clean = hold[~teleports[hold].any((1, 2))]
    labels = {"latent": actual[:, 0], "motion": motion, "physical": xy_delta[:, 0]}
    summary = {"pairs": len(rgb), "train_pairs": split, "held_pairs": len(hold), "held_clean_pairs": len(clean),
               "held_teleported_pairs": int(len(hold) - len(clean)),
               "max_current_latent_pair_delta": float(np.abs(contrast(z_pair[:, :, 2])).max()),
               "max_future_agent_pair_delta": float(np.abs(contrast(xy[:, :, 3:, 0])).max()),
               "pixel_motion_error_clean": float(np.square(motion[clean] - xy_delta[clean, 0]).mean()),
               "physical_null_mse_clean": float(np.square(xy_delta[clean, 0]).mean()),
               "readability": {}, "rollout": {}}
    for name, feature in features.items():
        row = summary["readability"][name] = {}
        arrays[name] = feature
        for target_name, target in labels.items():
            prediction = fit_readout(feature, target, split)
            arrays[f"{name}_read_{target_name}"] = prediction
            row[target_name + "_held_clean_explained"] = float(1 - np.square(prediction[clean] - target[clean]).sum() / np.square(target[clean]).sum())
    for name, predictions in (("native", native[:, 2:]), ("context3", native3), ("feedback", feedback[:, 2:]),
                              ("erased", erased[:, 2:]), ("swapped", swapped[:, 2:])):
        p = paired(predictions)
        d = contrast(p)
        arrays[name] = p
        summary["rollout"][name] = {
            "held_latent_mse": np.square(p[hold] - z_pair[hold, :, 3:]).mean((0, 1, 3)).tolist(),
            "held_clean_pair_contrast_mse": np.square(d[clean] - actual[clean]).mean((0, 2)).tolist(),
            "held_clean_pair_contrast_rms": np.sqrt(np.square(d[clean]).mean((0, 2))).tolist()}
    with (args.out / "diagnosis.npz").open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    with (args.out / "diagnosis.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


def ridge_solve(design, target):
    x = design.reshape(-1, design.shape[-1]).contiguous()
    gram = x.T @ x
    gram.diagonal().add_(0.01 * gram.diagonal().mean())
    return torch.linalg.solve(gram, x.T @ target.reshape(-1))


@torch.inference_mode()
def route(args):
    """Replace the tied diagonal read by one fitted native tangent read."""
    host = load_host(args)
    operator = ReadoutFeedback(host.model)
    device = host.device
    with np.load(args.out / "pairs.npz", allow_pickle=False) as data:
        raw = data["actions"]
    with np.load(args.out / "diagnosis.npz", allow_pickle=False) as data:
        z = torch.as_tensor(data["z"], device=device)
        teleported = data["teleport"][:, :, 0].any(1)
    with np.load(args.previous, allow_pickle=False) as data:
        basis = torch.as_tensor(data["low_rank_kernel_basis"], device=device)
    split = 72
    n = len(z)
    action = torch.as_tensor(host.normalize(raw), dtype=z.dtype, device=device)[None]
    action = host.model.action_embed(host.model.action_encoder(action))[0]
    old_q = operator.preactivation(z[:, :, 1].reshape(-1, z.shape[-1]), action[1].expand(n * 2, -1))
    memory = (operator.activation(old_q) @ basis).reshape(n, 2, -1)
    center = memory[:split].mean((0, 1))
    scale = memory[:split].std((0, 1)).clamp_min(1e-5)
    memory = (memory - center) / scale
    dm = contrast(memory)

    # q, and hence the native Jacobian, is shared by the two histories of a pair.
    q = operator.preactivation(z[:, 0, 2], action[2].expand(n, -1))
    derivative = 0.5 * (1 + torch.erf(q / np.sqrt(2))) + q * torch.exp(-q.square() / 2) / np.sqrt(2 * np.pi)
    base = operator.activation(q)
    truth = z[:, :, 3]
    dy = contrast(truth)
    native = operator.readout(base)[:, None].expand_as(truth)

    def jacobian(directions):
        return operator.readout.weight @ (derivative[:, :, None] * directions[None])

    if args.fit_endpoints:
        train_memory = memory[:split].flatten(0, 1)
        train_target = (truth - native)[:split].flatten(0, 1)
        take = torch.arange(split, device=device).repeat_interleave(2)
    else:
        train_memory, train_target = dm[:split], dy[:split]
        take = torch.arange(split, device=device)
    tied_design = jacobian(basis)[take] * train_memory[:, None]
    tied = ridge_solve(tied_design, train_target)
    read = operator.row_basis
    jac = jacobian(read)
    design = (jac[take, :, :, None] * train_memory[:, None, None, :]).flatten(-2)
    coefficients = ridge_solve(design, train_target).reshape(read.shape[1], basis.shape[1])
    del design
    gram = train_memory.T @ train_memory
    gram.diagonal().add_(0.01 * gram.diagonal().mean())
    direct = torch.linalg.solve(gram, train_memory.T @ train_target)

    def tangent(directions):
        return operator.readout(base[:, None] + derivative[:, None] * directions)

    predictions = {"native": native, "tied_refit": tangent((memory * tied) @ basis.T),
                   "row_read": tangent((memory @ coefficients.T) @ read.T),
                   "direct_latent_read": native + memory @ direct}
    predictions["row_read_swapped"] = predictions["row_read"].flip(1)
    hold = np.arange(split, n)
    subsets = {"held": hold, "no_teleport_h1": hold[~teleported[hold]], "teleport_h1": hold[teleported[hold]]}
    report = {"train_pairs": split, "held_pairs": n - split, "retained_width": basis.shape[1],
              "read_parameters": coefficients.numel(),
              "fit": "ridge on H1 endpoint residuals" if args.fit_endpoints else "ridge on true paired H1 latent differences",
              "subsets": {name: len(idx) for name, idx in subsets.items()}, "methods": {}}
    for name, pred in predictions.items():
        report["methods"][name] = {subset: {"latent_mse": float((pred[idx] - truth[idx]).square().mean()),
                                                   "contrast_mse": float((contrast(pred[idx]) - dy[idx]).square().mean())}
                                     for subset, idx in subsets.items() if len(idx)}
    name = "route_anchored" if args.fit_endpoints else "route"
    with (args.out / f"{name}.npz").open("xb") as stream:
        np.savez_compressed(stream, **{k: v.cpu().numpy() for k, v in predictions.items()}, truth=truth.cpu().numpy(),
                            memory=memory.cpu().numpy(), write_basis=basis.cpu().numpy(), center=center.cpu().numpy(),
                            scale=scale.cpu().numpy(), read_basis=read.cpu().numpy(), coefficients=coefficients.cpu().numpy())
    with (args.out / f"{name}.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("collect", "diagnose", "route"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=Path, default=Path("runs/p1a_v1/development/roots.jsonl"))
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--probe", type=Path, default=Path("research/phase00c/probe_heads.pt"))
    parser.add_argument("--previous", type=Path, default=Path("runs/readout_lowrank_20261108/fourroom/trace.npz"))
    parser.add_argument("--fit-endpoints", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if args.stage == "collect":
        collect(args.roots, args.out / "pairs.npz")
    elif args.stage == "diagnose":
        diagnose(args)
    else:
        route(args)
