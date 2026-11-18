"""Matched offline H5 continuation of the existing FourRoom conditioned predictor."""

import argparse
import csv
import math
from time import perf_counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .native_topology_condition import RUN as PARENT_RUN
from .native_topology_condition import SEED, fan, forecast, load_readout, load_trained, read_panel, sources
from .physical_topology_gate import (CENTER, DOOR_CLEARANCE, DOOR_SIZE, HALF_WALL, RADIUS, SPEED,
                                    PhysicalRollout, VisibleMapCost, make_scene)


RUN = PARENT_RUN.with_name("native_topology_prefix_20261115")
SYMMETRIC_RUN = PARENT_RUN.with_name("native_topology_symmetric_20261116")
FIT_SEEDS = tuple(range(2026111600, 2026111632))
EPOCHS, BATCH = 20, 16
ARMS = ("old", "query", "prefix")
SYMMETRIC_ARMS = ("arm1", "arm2")
LIVE_UPDATES, CROSSING_WEIGHT = 400, 5.
ACCUMULATED_UPDATES, WALL_WEIGHT, RANK_MARGIN, SOURCE_MARGIN = 300, 10., 5., 3.


def read_fan(path, device):
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["seed"], FIT_SEEDS)
        panel = {key: torch.as_tensor(data[key], device=device) for key in (
            "root_z", "root_action", "action", "future_z", "future_xy",
        )}
    if panel["future_z"].shape != (len(FIT_SEEDS), 4, 4, 2, 5, 192):
        raise ValueError("Expected four groups of four paired H5 candidates per fit layout")
    return fan(panel)


def prefix_source_arrays(data, indices, *, with_xy):
    layout, root, candidate, origin, h = indices
    root_z = data["root_z"][layout, root]
    future_z = data["future_z"][layout, root, candidate, origin]
    root_action = data["root_action"][layout, root]
    commands = data["action"][layout, root, candidate]
    z_path = np.concatenate((root_z, future_z), axis=1)
    action_path = np.concatenate((root_action, commands), axis=1)
    idx = np.arange(len(h))
    history = z_path[idx[:, None], h[:, None] + np.arange(3)]
    past = action_path[idx[:, None], h[:, None] + np.arange(2)]
    target = data["local_z"][indices]
    y = data["local_xy"][indices][:, :, -1]
    label = (y[:, 0] != y[:, 1]).any(axis=-1)
    # The actual outgoing branch is one half of the local paired target.
    np.testing.assert_array_equal(y[idx, origin], data["future_xy"][layout, root, candidate, origin, h, -1])
    np.testing.assert_array_equal(target[idx, origin], future_z[idx, h])
    action = commands[idx, h, None, None]
    arrays = (history, past, action, target[:, None, :, None], label)
    if with_xy:
        arrays += (y[:, None, :, None],)
    return arrays


def prefix_source_indices(data):
    *indices, h = np.nonzero(data["local_valid"][..., 1:])
    return (*indices, h + 1)


def read_prefix_sources(path, device, *, with_xy=False):
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["seed"], FIT_SEEDS)
        arrays = prefix_source_arrays(data, prefix_source_indices(data), with_xy=with_xy)
    return tuple(torch.as_tensor(array, device=device) for array in arrays)


def physical_labels(raw, before, commands, neutral, bounds, centers, device, *, walls=False,
                    factual_passages=False):
    # raw [batch,candidate,mode,horizon,primitive,xy]; before is the true block start.
    stayed = ((raw >= CENTER) == (before[..., None, :] >= CENTER)).all(axis=-2)
    raw_before = np.concatenate((before[..., None, :], raw[..., :-1, :]), axis=-2)
    unconstrained = np.clip(raw_before + SPEED * commands[:, :, None], *bounds)
    corrected = np.abs(raw - unconstrained) > 1e-5
    clipped = corrected.any(axis=(-2, -1))
    intent = commands.sum(axis=-2)
    norm = np.linalg.norm(intent, axis=-1, keepdims=True)
    direction = np.divide(intent, norm, out=np.zeros_like(intent), where=norm > 0)
    blocked = neutral & clipped.all(axis=2) & (norm[..., 0] > 0)
    static = blocked & (raw == before[..., None, :]).all(axis=(2, 4, 5))
    labels = dict(stayed=stayed, direction=direction, neutral=neutral, blocked=blocked, centers=centers)
    if walls:
        near, far = CENTER - HALF_WALL - RADIUS, CENTER + HALF_WALL + RADIUS
        end = raw[..., -1, :]
        # A block can hit a wall and then legally enter the aperture on the same axis.
        stayed_left = (corrected & (raw_before < CENTER)).any(axis=-2) & (end <= near)
        stayed_right = (corrected & (raw_before >= CENTER)).any(axis=-2) & (end >= far)
        if factual_passages:
            # A completed passage supervises the destination side, using only offline facts.
            stayed_left |= (before >= CENTER) & (end <= near)
            stayed_right |= (before < CENTER) & (end >= far)
        labels["wall_side"] = stayed_left.astype(np.int8) - stayed_right.astype(np.int8)
    return {key: torch.as_tensor(array, device=device) for key, array in labels.items()}, static


def read_live_panels(path, device, *, accumulated=False, factual_targets=False):
    from .native_topology_paired_data import LIVE_SEEDS

    with np.load(path, allow_pickle=False) as file:
        keys = (
            "seed", "root_xy", "root_z", "root_action", "action", "future_z", "future_xy", "local_z",
            "local_xy", "local_valid", "first_branch_divergence_raw", "first_local_divergence_raw",
        )
        if accumulated:
            keys += ("candidate_valid", "rank_eligible")
        data = {key: file[key] for key in keys}
    np.testing.assert_array_equal(data["seed"], LIVE_SEEDS)
    n_layout = len(LIVE_SEEDS)
    candidates, horizon = data["action"].shape[2:4]
    if (data["future_z"].shape != (n_layout, 4, candidates, 2, horizon, 192)
            or (not accumulated and (candidates, horizon) != (4, 5))
            or (accumulated and horizon not in (5, 6))):
        raise ValueError("Expected four roots per layout and paired targets for each candidate")
    scenes = [make_scene(seed) for seed in LIVE_SEEDS]
    models = [PhysicalRollout(scene) for scene in scenes]
    centers = np.asarray([[p.top, p.bottom, p.left, p.right] for p in models], dtype=np.float32)
    bounds = models[0].min_xy, models[0].max_xy
    tensor = lambda x: torch.as_tensor(x, device=device)
    main = fan({key: tensor(data[key]) for key in ("root_z", "root_action", "action", "future_z", "future_xy")})
    raw = data["future_xy"].reshape(-1, candidates, 2, horizon, 5, 2)
    starts = data["root_xy"][..., -1, :].reshape(-1, 1, 1, 1, 2)
    before = np.concatenate((np.broadcast_to(starts, raw[..., :1, -1, :].shape), raw[..., :-1, -1, :]), axis=-2)
    changed = (raw[:, :, 0] != raw[:, :, 1]).any(axis=-1)
    raw_onset = changed.reshape(*changed.shape[:2], -1)
    onset = np.where(raw_onset.any(axis=-1), raw_onset.argmax(axis=-1), -1)
    flags = data["candidate_valid"].reshape(onset.shape) if accumulated else np.ones_like(onset, dtype=bool)
    np.testing.assert_array_equal(onset[flags], data["first_branch_divergence_raw"].reshape(onset.shape)[flags])
    block_changed = changed.any(axis=-1)
    first = (np.arange(raw.shape[-3]) == onset[..., None] // raw.shape[-2])
    before_split = np.concatenate((np.ones_like(first[..., :1]), ~(np.maximum.accumulate(block_changed[..., :-1], axis=-1))), axis=-1)
    before_split &= flags[..., None]
    root_physics, _ = physical_labels(raw, before, data["action"].reshape(-1, candidates, horizon, 5, 2),
                                     before_split & ~block_changed, bounds, np.repeat(centers, 4, axis=0), device,
                                     walls=accumulated, factual_passages=factual_targets)
    if accumulated:
        from .native_topology_mpc import TorchVisibleMapCost

        exact_maps, rank = {}, data["rank_eligible"].reshape(-1, candidates, 2)
        if (rank & ~flags[..., None]).any() or data["local_valid"][~data["candidate_valid"]].any():
            raise ValueError("A padded plan cannot provide a ranking pair or factual source")
        for i, scene in enumerate(scenes):
            for root, goal in ((2, scene.southeast), (3, scene.northwest)):
                j = 4 * i + root
                geometry = VisibleMapCost(scene, goal)
                costs = geometry(raw[j, :, :, -1, -1])
                if not flags[j, 3] or (rank[j] & ~(costs > costs[3] + RANK_MARGIN)).any():
                    raise ValueError(f"Expected a verified same-mode rank margin at root {j}")
                exact_maps[j] = TorchVisibleMapCost(geometry, device)
        if rank.reshape(n_layout, 4, candidates, 2)[:, :2].any():
            raise ValueError("Contact roots do not provide navigation ranking pairs")
        root_physics.update(valid=tensor(flags), rank=tensor(rank), maps=exact_maps)

    indices = prefix_source_indices(data)
    layout, root, candidate, origin, h = indices
    source = prefix_source_arrays(data, indices, with_xy=True)
    local_raw = data["local_xy"][indices]
    local_changed = (local_raw[:, 0] != local_raw[:, 1]).any(axis=-1)
    local_first = local_changed.any(axis=-1)
    np.testing.assert_array_equal(np.where(local_first, local_changed.argmax(axis=-1), -1),
                                  data["first_local_divergence_raw"][indices])
    actual = data["future_xy"][layout, root, candidate, origin]
    np.testing.assert_array_equal(local_raw[np.arange(len(h)), origin], actual[np.arange(len(h)), h])
    origin_before = actual[np.arange(len(h)), h - 1, -1, :]
    local_before = np.broadcast_to(origin_before[:, None, None, None, :], (len(h), 1, 2, 1, 2))
    source_physics, static = physical_labels(local_raw[:, None, :, None], local_before, source[2],
                                             ~local_first[:, None, None], bounds, centers[layout], device,
                                             walls=factual_targets, factual_passages=factual_targets)
    source = tuple(tensor(x) for x in (*source[:4], local_first, source[5]))
    return main, source, (tensor(before_split), tensor(first), root_physics, source_physics, tensor(static[:, 0, 0]))


def physical_loss(predicted, latent, truth, labels, rows, gain):
    """Convert one-sided pixel and neutral mode-gap errors to the existing latent loss scale."""
    stayed, direction, neutral, blocked, centers = (labels[k][rows] for k in (
        "stayed", "direction", "neutral", "blocked", "centers"))
    error = predicted - truth
    wrong_side = stayed & ((predicted >= CENTER) != (truth >= CENTER))
    x, y = predicted.unbind(dim=-1)
    vertical = (y - centers[:, None, :2, None]).abs()
    left, right = centers[:, None, None, None, 2], centers[:, None, None, None, 3]
    horizontal = torch.minimum((x - left).abs(), (x - right).abs())
    half_strip, aperture = HALF_WALL + RADIUS, DOOR_SIZE + DOOR_CLEARANCE
    in_wall = (((x - CENTER).abs() < half_strip) & (vertical > aperture)
               | ((y - CENTER).abs() < half_strip) & (horizontal > aperture))
    weight = torch.where(wrong_side.any(dim=-1) | in_wall, CROSSING_WEIGHT, 1.)
    mse = error.square().mean(dim=-1)
    # Both deltas begin at the same factual position, so it cancels in the excess projection.
    overshoot = ((error * direction[:, :, None]).sum(dim=-1).relu().square() * blocked[:, :, None])
    overshoot = overshoot.sum() / (2 * blocked.sum()).clamp_min(1)
    gap = (latent[:, :, 1] - latent[:, :, 0]).square().mean(dim=-1)
    xy_gap = (predicted[:, :, 1] - predicted[:, :, 0]).square().mean(dim=-1)
    invariant = ((gap + xy_gap / gain) * neutral).sum() / neutral.sum().clamp_min(1)
    if "valid" in labels:
        valid = labels["valid"][rows, :, None, None]
        count = valid.sum() * mse.shape[-1] * mse.shape[-2]
        return (mse * valid).sum() / count, (mse * weight * valid).sum() / count, overshoot, invariant
    return mse.mean(), (mse * weight).mean(), overshoot, invariant


def wall_intrusion(predicted, wall_side, valid):
    """Positive distance past the factual collision face, per axis, mode and block."""
    face = CENTER - wall_side * (HALF_WALL + RADIUS)
    intrusion = ((predicted - face) * wall_side).relu().square()
    contact = (wall_side != 0) & valid[..., None, None, None]
    return (intrusion * contact).sum() / contact.sum().clamp_min(1)


def rank_hinge(terminal, truth, rows, labels):
    """Exact deployment cost, with rank gradients only within the factual rooms."""
    mask = labels["rank"][rows]
    cost = torch.stack([labels["maps"][index](terminal[k]) if index in labels["maps"]
                        else torch.zeros_like(terminal[k, ..., 0]) for k, index in enumerate(rows.tolist())])
    hinge = (cost[:, 3:4] - cost + RANK_MARGIN).relu()
    correct = ((terminal >= CENTER) == (truth >= CENTER)).all(-1)
    valid_gradient = correct & correct[:, 3:4]
    rank = torch.where(valid_gradient, hinge, hinge.detach())
    return (rank * mask).sum() / mask.sum().clamp_min(1), (mask & valid_gradient).sum(), mask.sum()


def source_hinge(issued_xy, arrived_xy, informative, factor, *, factual_targets=False):
    """Score the same issued pair against both possible factual arrivals."""
    from .native_topology_xy_evidence import paired_directions, student_t

    arrived = arrived_xy.to(factor)
    residual = arrived[:, :, None] - issued_xy[:, None].to(factor)
    whitened = torch.linalg.solve_triangular(factor, residual.reshape(-1, 2).T, upper=False)
    gaussian = -0.5 * whitened.square().sum(0).reshape(-1, 2, 2)
    delta = paired_directions(student_t(gaussian))
    if factual_targets:
        difference = arrived[:, 1] - arrived[:, 0]
        separation = torch.linalg.solve_triangular(factor, difference.T, upper=False).square().sum(0)
        target = (-student_t(-.5 * separation)).clamp_max(SOURCE_MARGIN).detach()
        # Equal factual arrivals give target zero and opposite LR signs: the sum is |LR|.
        loss = (target[:, None] - delta).relu()
    else:
        loss = (SOURCE_MARGIN - delta).relu() * informative[:, None]
    # Preserve the per-informative-example coefficient of the frozen training protocol.
    return loss.sum() / (2 * informative.sum()).clamp_min(1)


def readout_gain(decode, history):
    """Mean square pixel response to unit isotropic latent noise at factual roots."""
    z = history.detach().requires_grad_(True)
    y = decode(z)
    gains = [torch.autograd.grad(y[..., axis].sum(), z, retain_graph=axis == 0)[0].square().sum(-1).mean()
             for axis in range(2)]
    gain = torch.stack(gains).mean().detach()
    if not (gain.isfinite() & (gain > 0)):
        raise ValueError("The fixed XY readout must have positive finite sensitivity")
    return gain


def fit(arm, run, main_panel, source_panel, device, *, start_run=PARENT_RUN, fixed_xy=None,
        live=None, resume=False, accumulated_round=None, xy_factor=None, objective_pilot=None):
    if objective_pilot not in (None, "control", "factual"):
        raise ValueError("Objective pilot must be control or factual")
    if objective_pilot is not None and (accumulated_round != 1 or live is None or arm != "arm2"):
        raise ValueError("The fixed 64-update objective pilot extends the accumulated Arm 2 path")
    factual_targets = objective_pilot == "factual"
    started = perf_counter()
    torch.manual_seed(SEED)
    learned_gate = arm == "arm1" if live is not None else fixed_xy is None
    directory = run / arm
    if accumulated_round is not None:
        directory = run / f"round{accumulated_round}"
        if accumulated_round > 1:
            start_run = run / f"round{accumulated_round - 1}"
    elif live is not None:
        directory.mkdir(exist_ok=True)
        phase_one = directory / "round1"
        if resume:
            start_run = phase_one
        else:
            directory = phase_one
    host, conditioner, gate = load_trained(start_run, "conditioned", device)
    modules = {"predictor": host.level.predictor, "action_embed": host.level.action_embed,
               "conditioner": conditioner, "gate": gate}
    gate.forced_exposure = None if learned_gate else 1.0
    parameters = [p for name, module in modules.items() if name != "gate" or learned_gate
                  for p in module.requires_grad_(True).parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, weight_decay=1e-4)
    history, past, plans, target, target_xy, root_labels = main_panel
    shist, spast, saction, starget, source_labels = source_panel[:5]
    if fixed_xy is not None:
        source_xy, = source_panel[5:]
        coordinate_gain = readout_gain(fixed_xy, history)
        print(f"{arm}: fixed readout XY gain {coordinate_gain.item():.8g} px^2 / latent unit^2", flush=True)
    strata = (~source_labels, source_labels)
    counts = (BATCH // 2, BATCH // 2)
    if live is not None:
        shared, first, root_physics, source_physics, static = live
        strata = (static, ~source_labels & ~static, source_labels)
        counts = (BATCH // 4, BATCH // 4, BATCH // 2)
        print(f"{arm}: shared-before steps {shared.sum((0, 1)).tolist()}, first effects {first.sum((0, 1)).tolist()}; "
              f"common blocked free/source {root_physics['blocked'].sum().item()}/{source_physics['blocked'].sum().item()}",
              flush=True)
    groups = tuple(mask.nonzero().flatten() for mask in strata)
    if any(group.numel() == 0 for group in groups):
        raise ValueError("Every required factual source stratum must contain samples")
    root_rng = torch.Generator(device=device).manual_seed(SEED)
    source_rng = torch.Generator(device=device).manual_seed(SEED)
    directory.mkdir(exist_ok=True)
    print(f"{arm}: {len(history)} roots, {'/'.join(str(len(group)) for group in groups)} "
          f"{'static/other neutral/informative' if live is not None else 'neutral/informative'} sources", flush=True)
    update, epochs, stop = 0, EPOCHS, 160
    if live is not None:
        steps = ACCUMULATED_UPDATES if accumulated_round is not None else LIVE_UPDATES
        if objective_pilot is not None:
            steps = 64
        expected = steps * (accumulated_round - 1) if accumulated_round is not None else steps * int(resume)
        if expected:
            state = torch.load(start_run / "fit_state.pt", map_location="cpu", weights_only=True)
            if state["update"] != expected or state["arm"] != arm:
                raise ValueError(f"Expected this arm's step-{expected} optimizer and sampler state")
            torch.testing.assert_close(coordinate_gain.cpu(), state["coordinate_gain"], rtol=0, atol=0)
            optimizer.load_state_dict(state["optimizer"])
            root_rng.set_state(state["root_rng"])
            source_rng.set_state(state["source_rng"])
            update = state["update"]
        epochs = math.ceil(steps / math.ceil(len(history) / BATCH))
        stop = update + steps
    with (directory / "training.csv").open("x", encoding="utf-8", newline="") as stream:
        rows = csv.writer(stream)
        columns = ["epoch", "update", "loss", "grad_norm", *(f"h{i + 1}" for i in range(plans.shape[-3])),
                   "source_mse", "root_bce", "source_bce"]
        if fixed_xy is not None:
            columns += ["root_xy_mse_px2", "source_xy_mse_px2"]
        if live is not None:
            columns += ["root_weighted_xy_mse_px2", "source_weighted_xy_mse_px2", "root_positive_forward_px2",
                        "source_positive_forward_px2", "root_neutral_invariant", "source_neutral_invariant"]
        if accumulated_round is not None:
            columns += ["wall_intrusion_px2", "rank_hinge_px", "rank_same_room_pairs", "rank_pairs", "source_hinge_nat"]
        rows.writerow(columns)
        for epoch in range(1, epochs + 1):
            totals = []
            for b in torch.randperm(len(history), generator=root_rng, device=device).split(BATCH):
                s = torch.cat([group[torch.randint(len(group), (count,), generator=source_rng, device=device)]
                               for group, count in zip(groups, counts, strict=True)])
                main_trace, local_trace = {}, {}
                predicted = forecast(host, conditioner, gate, history[b], past[b], plans[b], trace=main_trace)
                issued = forecast(host, conditioner, gate, shist[s], spast[s], saction[s], trace=local_trace)
                error = (predicted - target[b]).square()
                if accumulated_round is None:
                    per_h = error.mean(dim=(0, 1, 2, 4))
                else:
                    valid = root_physics["valid"][b]
                    per_h = (error.mean(dim=(2, 4)) * valid[..., None]).sum(dim=(0, 1)) / valid.sum()
                mse = (issued - starget[s]).square().mean()
                root_bce = source_bce = mse.new_zeros(())
                if learned_gate:
                    if live is None:
                        root_bce = F.binary_cross_entropy_with_logits(main_trace["logits"][:, :, 0, 0], root_labels[b].float())
                    else:
                        eligible = shared[b]
                        root_bce = F.binary_cross_entropy_with_logits(main_trace["logits"][:, :, 0][eligible], first[b][eligible].float())
                    source_bce = F.binary_cross_entropy_with_logits(local_trace["logits"][:, 0, 0, 0], source_labels[s].float())
                elif not totals and epoch == 1:
                    if not (main_trace["exposure"] == 1).all() or not (local_trace["exposure"] == 1).all():
                        raise RuntimeError("Forced exposure must apply to every rollout and factual source")
                extras = ()
                auxiliary = root_bce + source_bce
                if fixed_xy is not None:
                    pxy, sxy = fixed_xy(predicted), fixed_xy(issued)
                    if live is None:
                        root_xy = F.mse_loss(pxy, target_xy[b].to(pxy.dtype))
                        local_xy = F.mse_loss(sxy, source_xy[s].to(sxy.dtype))
                        auxiliary = auxiliary + (root_xy + local_xy) / coordinate_gain
                        extras = (root_xy, local_xy)
                    else:
                        rmse, rw, ro, ri = physical_loss(pxy, predicted, target_xy[b], root_physics, b, coordinate_gain)
                        smse, sw, so, si = physical_loss(sxy, issued, source_xy[s], source_physics, s, coordinate_gain)
                        extras = (rmse, smse, rw, sw, ro, so, ri, si)
                        if accumulated_round is None:
                            auxiliary = auxiliary + (rw + sw + ro + so) / coordinate_gain + ri + si
                        else:
                            wall = wall_intrusion(pxy, root_physics["wall_side"][b], valid)
                            if factual_targets:
                                wall = wall + wall_intrusion(sxy, source_physics["wall_side"][s],
                                                             torch.ones_like(source_labels[s, None]))
                            rank, active, pairs = rank_hinge(pxy[..., -1, :], target_xy[b, ..., -1, :], b, root_physics)
                            with torch.no_grad():
                                source_arrived = fixed_xy(starget[s])[:, 0, :, 0]
                            end_effect = (source_xy[s, 0, 0, 0] != source_xy[s, 0, 1, 0]).any(-1)
                            evidence = source_hinge(sxy[:, 0, :, 0], source_arrived, source_labels[s] & end_effect,
                                                    xy_factor, factual_targets=factual_targets)
                            auxiliary = auxiliary + (rmse + smse + ro + so + WALL_WEIGHT * wall) / coordinate_gain
                            auxiliary = auxiliary + ri + si + rank / coordinate_gain.sqrt() + evidence / SOURCE_MARGIN
                            extras += (wall, rank, active, pairs, evidence)
                loss = .5 * (per_h.mean() + mse + auxiliary)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
                record = torch.cat((loss[None], grad[None], per_h, mse[None], root_bce[None], source_bce[None],
                                    *(item[None] for item in extras)))
                record = record.detach().cpu().numpy()
                if not np.isfinite(record).all():
                    raise FloatingPointError(f"Nonfinite loss or gradient at {arm} update {update + 1}")
                optimizer.step()
                update += 1
                totals.append(record)
                rows.writerow([epoch, update, *record.tolist()])
                if live is not None and update == stop:
                    break
            stream.flush()
            if epoch == 1 or epoch % 5 == 0 or (live is not None and update == stop):
                print(f"{arm} {epoch}/{epochs} (update {update}): {np.mean(totals, axis=0)}", flush=True)
            if live is not None and update == stop:
                break
    if update != stop:
        raise RuntimeError(f"Expected {stop} updates, got {update}")
    state = {name: module.state_dict() for name, module in modules.items()}
    if fixed_xy is not None:
        state.update(forced_exposure=gate.forced_exposure, xy_readout_gain_px2=float(coordinate_gain))
    torch.save(state, directory / "conditioned.pt")
    if live is not None and (not resume or accumulated_round is not None):
        torch.save(dict(arm=arm, update=update, optimizer=optimizer.state_dict(), root_rng=root_rng.get_state(),
                        source_rng=source_rng.get_state(), coordinate_gain=coordinate_gain.cpu()), directory / "fit_state.pt")
    if live is not None:
        print(f"{arm}: saved update {update} to {directory}, elapsed {perf_counter() - started:.1f} seconds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", nargs="+", choices=ARMS + SYMMETRIC_ARMS, default=ARMS)
    parser.add_argument("--live-data", type=Path, help="128-layout live panel for the next 400 updates")
    parser.add_argument("--live-resume", action="store_true", help="Continue one arm from its round-one fit state")
    parser.add_argument("--cumulative-round", type=int, choices=(1, 2, 3), help="300 updates using the accumulated Arm2 panel")
    parser.add_argument("--start-checkpoint", type=Path, help="Frozen Arm2 checkpoint directory at the first cumulative round")
    parser.add_argument("--xy-cholesky", type=Path, help="The single calibrated XY Cholesky factor for all cumulative rounds")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if args.live_resume and (args.live_data is None or len(args.arms) != 1):
        parser.error("Specify one arm and its refreshed --live-data when resuming")
    if args.cumulative_round is not None:
        if (args.arms != ["arm2"] or args.live_resume or not (args.run and args.live_data and args.xy_cholesky)
                or (args.cumulative_round == 1) != (args.start_checkpoint is not None)):
            parser.error("Cumulative fitting requires --arms arm2, --run, --live-data and --xy-cholesky, plus --start-checkpoint in round 1")
        panel, prefix, live = read_live_panels(args.live_data, device, accumulated=True)
        with np.load(args.live_data, allow_pickle=False) as data:
            np.testing.assert_array_equal(data["cumulative_round"], args.cumulative_round)
        factor = torch.load(args.xy_cholesky, map_location=device, weights_only=True)
        if (factor.shape != (2, 2) or not factor.isfinite().all() or factor[0, 1] != 0
                or not (factor.diag() > 0).all()):
            raise ValueError("Expected a finite positive lower Cholesky factor")
        fit("arm2", args.run, panel, prefix, device, start_run=args.start_checkpoint,
            fixed_xy=load_readout(device, PARENT_RUN), live=live, accumulated_round=args.cumulative_round,
            xy_factor=factor)
        return
    if args.start_checkpoint is not None or args.xy_cholesky is not None:
        parser.error("--start-checkpoint and --xy-cholesky are for cumulative fitting")
    if args.live_data is not None:
        from .native_topology_paired_data import LIVE_OUTPUT

        if any(arm not in SYMMETRIC_ARMS for arm in args.arms):
            parser.error("Live continuation expects arm1 and/or arm2")
        run = args.run or LIVE_OUTPUT
        run.mkdir(parents=True, exist_ok=True)
        panel, prefix, live = read_live_panels(args.live_data, device)
        decode = load_readout(device, PARENT_RUN)
        for arm in args.arms:
            fit(arm, run, panel, prefix, device, start_run=RUN / "prefix", fixed_xy=decode,
                live=live, resume=args.live_resume)
        return
    if any(arm in SYMMETRIC_ARMS for arm in args.arms):
        if any(arm not in SYMMETRIC_ARMS for arm in args.arms):
            parser.error("Symmetric and legacy arms use different data files")
        run = args.run or SYMMETRIC_RUN
        path = run / "data_train_symmetric.npz"
        panel, sources_xy = read_fan(path, device), read_prefix_sources(path, device, with_xy=True)
        for arm in args.arms:
            fit(arm, run, panel, sources_xy, device, start_run=RUN / "prefix",
                fixed_xy=load_readout(device, PARENT_RUN) if arm == "arm2" else None)
        return
    args.run = args.run or RUN
    script = sources(read_panel(PARENT_RUN / "data_train.npz", FIT_SEEDS, device))[0]
    original = read_fan(args.run / "data_train_original.npz", device)
    coverage_file = args.run / "data_train_coverage.npz"
    coverage = read_fan(coverage_file, device)
    for i in (0, 1):
        torch.testing.assert_close(original[i], coverage[i], atol=0, rtol=0)
    prefix = read_prefix_sources(coverage_file, device)
    for arm in args.arms:
        fit(arm, args.run, original if arm == "old" else coverage, prefix if arm == "prefix" else script, device)


if __name__ == "__main__":
    main()
