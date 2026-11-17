"""Zero-update ranking and room-gradient diagnosis on the four-slot fit archive."""
from itertools import combinations
import json

import numpy as np
import torch
import torch.nn.functional as F

from .native_topology_mpc import TorchVisibleMapCost
from .native_topology_paired_data import BAD_ROLE, LEGAL_ROLE, clipping
from .native_topology_prefix_fit import FIT_SEEDS
from .native_topology_writein_replay import CHECKPOINTS, REFERENCE, RUN, forecast, load_readout, load_trained
from .physical_topology_gate import CENTER, VisibleMapCost, make_scene, room


OUTPUT = RUN / "writein_attribution_20261116/decision_gradient"
SCALE = 2 * CENTER
PAIR_DELTA = 0.01  # px, central difference away from score discontinuities
PARAM_DELTA = 0.001  # L2 step in a unit parameter direction, never saved
STRATA = ("common_blocked_detour", "swap_silent", "swap_mixed", "swap_exposed", "false_goal_room")


def payload(archive, i, root, device):
    return tuple(torch.as_tensor(archive[k][i, root][None], device=device)
                 for k in ("root_z", "root_action", "action"))


@torch.no_grad()
def select(archive, model, decode, device):
    coverage = {key: 0 for key in ("slot_pairs", "mode_reversals", "strict_reversals", "centerline_pairs", *STRATA)}
    cases = {}
    for i, raw_seed in enumerate(archive["seed"]):
        scene = make_scene(int(raw_seed))
        for root, goal in ((2, scene.southeast), (3, scene.northwest)):
            start = archive["root_xy"][i, root, -1]
            actions, xy, role = (archive[key][i, root] for key in ("action", "future_xy", "selection_role"))
            ends, first = xy[:, :, -1, -1], xy[:, :, 0, -1]
            score = VisibleMapCost(scene, goal)
            q = score(ends)
            trace = {}
            z = forecast(*model, *payload(archive, i, root, device), compact=False, trace=trace)
            pred = decode(z[0, :, :, -1]).cpu().numpy()
            active = (trace["exposure"][0] > 0).cpu().numpy().any(axis=(1, 2))
            false_goal = (room(pred) == room(goal)) & (room(ends) != room(goal))
            hits = clipping(start, actions, xy).any(axis=2)
            blocked = hits & (room(ends) == room(start))
            clean_detour = (~hits) & (room(ends) != room(start))
            for a, b in combinations(range(4), 2):
                coverage["slot_pairs"] += 1
                diff = q[b].astype(np.float64) - q[a]
                swaps = diff[0] * diff[1] < 0
                coverage["mode_reversals"] += int(swaps)
                first_diff = np.any(first[a] != first[b], axis=-1)
                commands_differ = not np.array_equal(actions[a, 0], actions[b, 0])
                off_line = (ends[[a, b]] != CENTER).all(axis=(0, 2))
                coverage["centerline_pairs"] += int(not off_line.all())
                relevant = (diff != 0) & first_diff & off_line & commands_differ
                coverage["strict_reversals"] += int(swaps and relevant.all())
                classes = []
                if swaps and relevant.all():
                    swap_class = ("swap_silent", "swap_mixed", "swap_exposed")[int(active[[a, b]].sum())]
                    classes.append((swap_class, (0, 1)))
                if relevant.all() and diff[0] * diff[1] > 0 and {int(role[a]), int(role[b])} == {BAD_ROLE, LEGAL_ROLE}:
                    good, bad = (a, b) if diff[0] > 0 else (b, a)
                    if blocked[bad].all() and clean_detour[good].all():
                        classes.append(("common_blocked_detour", (0, 1)))
                wrong_modes = np.flatnonzero(relevant & false_goal[[a, b]].any(axis=0))
                if len(wrong_modes):
                    classes.append(("false_goal_room", (int(wrong_modes[0]),)))
                for label, modes in classes:
                    coverage[label] += 1
                    if label not in cases:
                        cases[label] = (i, root, a, b, modes)
    return coverage, cases


def objectives(xy, truth, q, geometry):
    hard = geometry(xy)
    orientation = (q[1] - q[0]).sign()
    priority = F.softplus((0.02 + orientation * (hard[0] - hard[1]) / SCALE) / 0.05)
    e = (xy - truth).norm(dim=-1)
    clearance = (truth - CENTER).abs()
    side = (truth - CENTER).sign()
    wrong_side = (clearance / 2 - side * (xy - CENTER)).clamp_min(0).square().sum(dim=(0, 2))
    gap = (e.sum(dim=0) - (q[1] - q[0]).abs() / 2).clamp_min(0).square()
    return torch.stack((priority, (wrong_side + gap) / SCALE**2))


def flat_norm(values):
    return torch.linalg.vector_norm(torch.stack([value.norm() for value in values]))


def module_norms(groups, sizes, gradients):
    norms, pos = {}, 0
    for name, count in zip(groups, sizes, strict=True):
        present = [g for g in gradients[pos:pos + count] if g is not None]
        norms[name] = float(flat_norm(present)) if present else 0.
        pos += count
    return norms


def list_cpu(value):
    return value.detach().cpu().tolist()


def pair_diagnosis(archive, case, model, decode, groups, device):
    i, root, a, b, modes = case
    slots = (a, b)
    seed = int(archive["seed"][i])
    scene = make_scene(seed)
    goal = (scene.southeast, scene.northwest)[root - 2]
    physical_score = VisibleMapCost(scene, goal)
    geom = TorchVisibleMapCost(physical_score, device)
    h, past, plans = payload(archive, i, root, device)
    targets = archive["future_xy"][i, root, list(slots)]
    y = targets[:, list(modes), -1, -1]
    tq = physical_score(y)
    truth, q = (torch.as_tensor(x, device=device) for x in (y, tq))
    terms = ["hard_softplus", "physical_margin"]
    params = tuple(p for module in groups.values() for p in module.parameters() if p.requires_grad)
    sizes = [sum(p.requires_grad for p in module.parameters()) for module in groups.values()]
    entry = {"seed": seed, "root_slot": root, "slots": slots, "modes": modes,
             "root_xy": archive["root_xy"][i, root, -1].tolist(), "goal_xy": goal.tolist(),
             "first_raw_block": archive["action"][i, root, list(slots), 0].tolist(),
             "true_first_xy": targets[:, list(modes), 0, -1].tolist(),
             "true_terminal_xy": y.tolist(), "true_terminal_cost": tq.tolist()}
    offsets, logit_inputs = [], []
    collect = lambda pool: (lambda _m, _i, output: pool.append(output))
    handles = (model[1].register_forward_hook(collect(offsets)),
               model[2].register_forward_hook(collect(logit_inputs)))
    trace = {}
    try:
        z = forecast(*model, h, past, plans, compact=False, trace=trace)
    finally:
        for handle in handles:
            handle.remove()
    assert len(offsets) == 1 and len(logit_inputs) == plans.shape[2]
    terminals = decode(z[0, list(slots), :, -1])[:, list(modes)]
    with torch.no_grad():
        torch.testing.assert_close(geom(terminals), torch.as_tensor(physical_score(terminals.cpu().numpy()), device=device),
                                   rtol=1e-6, atol=1e-5)
    logits = trace["logits"][0, list(slots)][:, list(modes)]
    exposure = trace["exposure"][0, list(slots)][:, list(modes)]
    mode_gaps = (decode(z[0, list(slots), 1]) - decode(z[0, list(slots), 0])).norm(dim=-1)
    pred_room = (terminals.detach() >= CENTER)
    wrong_coord = pred_room != (truth >= CENTER)
    offset_jacobian = []
    for s in range(len(slots)):
        by_mode = []
        for m in range(len(modes)):
            rows = [torch.autograd.grad(terminals[s, m, j], offsets[0], retain_graph=True)[0] for j in range(2)]
            by_mode.append(float(flat_norm([g[1] - g[0] for g in rows])))
        offset_jacobian.append(by_mode)
    entry.update(pred_terminal_xy=list_cpu(terminals), pred_terminal_cost=list_cpu(geom(terminals)),
                 wrong_coordinates=list_cpu(wrong_coord), gate_logit=list_cpu(logits),
                 exposure=list_cpu(exposure), decoded_mode_gap_by_horizon=list_cpu(mode_gaps),
                 terminal_jacobian_signed_offsets=offset_jacobian, terms={})

    with torch.no_grad():
        reference = tuple(p.detach().clone() for p in params)

    def checked_trial(direction, sign):
        with torch.no_grad():
            for p, base, d in zip(params, reference, direction, strict=True):
                p.copy_(base + sign * PARAM_DELTA * d)
            trace2 = {}
            zz = forecast(*model, h, past, plans, compact=False, trace=trace2)
            xy = decode(zz[0, list(slots), :, -1])[:, list(modes)]
            return (objectives(xy, truth, q, geom), geom(xy), xy,
                    int(((xy >= CENTER) != pred_room).sum()),
                    int(((trace2["exposure"][0] > 0) != (trace["exposure"][0] > 0)).sum()))

    pair_obj = objectives(terminals, truth, q, geom)
    z_target = torch.as_tensor(archive["future_z"][i, root, list(slots)], device=device)
    selected_latents = z[0, list(slots)][:, list(modes)]
    latent_obj = (selected_latents - z_target[:, list(modes)]).square().mean()
    labels = (targets[:, 0, 0, -1] != targets[:, 1, 0, -1]).any(axis=-1)
    bce = F.binary_cross_entropy_with_logits(trace["logits"][0, list(slots), 0, 0],
                                              torch.as_tensor(labels, dtype=logits.dtype, device=device))
    for key, obj in (("original_h1_h5_mse", latent_obj), ("original_h1_gate_bce", bce)):
        gg = torch.autograd.grad(obj, params, retain_graph=True, allow_unused=True)
        entry[key] = dict(value=float(obj.detach()), gradient_norm_by_module=module_norms(groups, sizes, gg))
    parameter_steps = []
    for name, values in zip(terms, pair_obj, strict=True):
        grad = torch.autograd.grad(values.mean(), (terminals, offsets[0], *logit_inputs, *params), retain_graph=True)
        grad_xy, off = grad[:2]
        gate_grad = torch.stack([g.reshape(4, 2)[list(slots)][:, list(modes)] for g in grad[2:-len(params)]], dim=-1)
        param_grad = grad[-len(params):]
        assert bool(torch.isfinite(flat_norm(param_grad))) and bool(torch.isfinite(grad_xy).all())
        assert not bool(gate_grad[exposure == 0].any())
        norm = flat_norm(param_grad)
        if float(norm) > 0:
            parameter_steps.append((name, float(norm), tuple(-g / norm for g in param_grad)))
        coord_checks = []
        with torch.no_grad():
            for s, m, axis in wrong_coord.nonzero().tolist():
                d = torch.zeros_like(terminals)
                d[s, m, axis] = (truth[s, m, axis] - terminals[s, m, axis]).sign()
                xy_p, xy_n = terminals + PAIR_DELTA * d, terminals - PAIR_DELTA * d
                index = terms.index(name)
                derivative = ((objectives(xy_p, truth, q, geom)[index].mean()
                              - objectives(xy_n, truth, q, geom)[index].mean()) / (2 * PAIR_DELTA))
                coord_checks.append({"slot": slots[s], "mode": modes[m], "axis": axis,
                                     "into_true_side": int(d[s, m, axis]), "ad_derivative": float((grad_xy * d).sum()),
                                     "fd_derivative": float(derivative),
                                     "room_changes": int((((xy_p >= CENTER) != pred_room) |
                                                          ((xy_n >= CENTER) != pred_room)).sum())})
        entry["terms"][name] = dict(loss_by_mode=list_cpu(values), grad_xy=list_cpu(grad_xy),
                                     common_offset_gradient_norm=float((off[1] + off[0]).norm()),
                                     signed_offset_gradient_norm=float((off[1] - off[0]).norm()),
                                     grad_gate_logit=list_cpu(gate_grad), gradient_norm_by_module=module_norms(groups, sizes, param_grad),
                                     wrong_coordinate_steps=coord_checks)
    for name, norm, direction in parameter_steps:
        try:
            plus, minus = (checked_trial(direction, sign) for sign in (1, -1))
        finally:
            with torch.no_grad():
                for p, base in zip(params, reference, strict=True):
                    p.copy_(base)
        assert all(torch.equal(p, base) for p, base in zip(params, reference, strict=True))
        pos = terms.index(name)
        fd = float((plus[0][pos].mean() - minus[0][pos].mean()) / (2 * PARAM_DELTA))
        entry["terms"][name]["parameter_step"] = {
            "ad_derivative": -norm, "fd_derivative": fd, "plus_loss": float(plus[0][pos].mean()),
            "minus_loss": float(minus[0][pos].mean()), "plus_terminal_cost": list_cpu(plus[1]),
            "minus_terminal_cost": list_cpu(minus[1]), "plus_terminal_xy": list_cpu(plus[2]),
            "minus_terminal_xy": list_cpu(minus[2]), "room_changes_pm": [plus[3], minus[3]],
            "gate_changes_pm": [plus[4], minus[4]],
        }
    return entry


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    model = load_trained(CHECKPOINTS["paired"], "conditioned", device)
    decode = load_readout(device, REFERENCE)
    with np.load(RUN / "data_train_paired.npz", allow_pickle=False) as src:
        archive = {k: src[k] for k in ("seed", "root_xy", "root_z", "root_action", "action",
                                       "future_xy", "future_z", "selection_role")}
    np.testing.assert_array_equal(archive["seed"], FIT_SEEDS)
    coverage, cases = select(archive, model, decode, device)
    print(coverage, flush=True)
    host, conditioner, gate = model
    groups = dict(predictor=host.level.predictor, action_embed=host.level.action_embed,
                  conditioner=conditioner, gate=gate)
    for module in groups.values():
        module.requires_grad_(True)
    report = {}
    for label in STRATA:
        if label in cases:
            report[label] = pair_diagnosis(archive, cases[label], model, decode, groups, device)
            print(label, report[label]["seed"], report[label]["slots"], flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    output = dict(selection="first lexicographic case by fit seed, root 2 then 3, slot combinations, mode 0 then 1",
                  coverage=coverage, missing=[label for label in STRATA if label not in cases],
                  parameter_step_l2=PARAM_DELTA, coordinate_step_px=PAIR_DELTA, cases=report)
    (OUTPUT / "summary.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
