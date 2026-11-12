"""First staged test of displacement transfer through the native prediction head."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from native_gate_bound_paired import CALIBRATION, load_host
from native_gate_bound_sequence_data import sequence_bank
from native_response import ContactTransferEvidence, ContactTransferPort
from query_feedback_learning import write_csv, write_json


def analytic():
    dtype = torch.float64
    rotation = torch.tensor([[0., -1.], [1., 0.]], dtype=dtype)
    eye = torch.eye(2, dtype=dtype)
    zero = ContactTransferEvidence.empty(rotation)
    time = torch.arange(3, dtype=dtype)[:, None]
    drift = torch.tensor([7., -9.], dtype=dtype) + time * torch.tensor([5., 8.], dtype=dtype)
    dither = torch.tensor([1., -2., 1.], dtype=dtype)[:, None]

    def burst(axis, transfer, quadratic=0.):
        pusher = dither * axis
        payload = drift + pusher @ transfer.T + quadratic * time.square()
        return torch.cat((pusher, payload), dim=-1)

    first = zero.observe_burst(burst(eye[0], rotation))
    full = first.observe_burst(burst(eye[1], rotation))
    torch.testing.assert_close(first.relation()[1], eye * eye[0], atol=0, rtol=0)
    torch.testing.assert_close(full.relation()[0], rotation, atol=0, rtol=0)
    torch.testing.assert_close(full.relation()[2], torch.tensor(0., dtype=dtype), atol=0, rtol=0)
    other = rotation.clone()
    other[:, 1] = 3.
    torch.testing.assert_close(burst(eye[0], rotation), burst(eye[0], other), atol=0, rtol=0)
    curved = zero.observe_burst(burst(eye[0], rotation, quadratic=3.)).relation()[0]
    torch.testing.assert_close(curved[:, 0], rotation[:, 0] + 1., atol=0, rtol=0)

    class AnalyticHost(nn.Module):
        def __init__(self):
            super().__init__()
            self.pred_proj = nn.Module()
            self.pred_proj.net = nn.Sequential(nn.Linear(6, 6, bias=False, dtype=dtype))
            with torch.no_grad():
                self.pred_proj.net[0].weight.copy_(torch.eye(6, dtype=dtype))

        def predict(self, states, actions):
            q = torch.cat((actions, actions, states[..., -2:]), dim=-1)
            return self.pred_proj.net(q.flatten(0, 1)).reshape_as(q)

    model = AnalyticHost().eval()
    port = ContactTransferPort(model, torch.eye(6, dtype=dtype), eye[0] * 0.)
    z = torch.arange(6, dtype=dtype).reshape(1, 1, 6)
    actions = torch.cat((eye, eye[:1] * 0.), dim=0).reshape(1, 3, 1, 2)
    reference = (actions[:, :, -1] == 0.).all(dim=-1)
    native, _ = port.predict(z, actions, zero.relation(), at_reference=reference)
    partial, partial_trace = port.predict(z, actions, first.relation(), at_reference=reference, trace=True)
    regulated, _ = port.predict(z, actions, full.relation(), at_reference=reference)
    repeat, _ = port.predict(z, actions, full.relation(), at_reference=reference)
    released, _ = port.predict(z, actions, full.clear().relation(), at_reference=reference)
    no_coupling = full.clear().observe_burst(burst(eye[0], eye * 0.))
    after_release, _ = port.predict(z, actions, no_coupling.relation(), at_reference=reference)
    conflicting = full.observe_burst(burst(eye[0], eye * 0.)).relation()[2]
    assert partial_trace["support"].tolist() == [[True, False, False]]
    torch.testing.assert_close(partial[0, 0, 0, 2:4], rotation[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(partial[:, 1:], native[:, 1:], atol=0, rtol=0)
    torch.testing.assert_close(regulated[0, :2, 0, 2:4], rotation.T, atol=0, rtol=0)
    torch.testing.assert_close(regulated[..., :2], native[..., :2], atol=0, rtol=0)
    torch.testing.assert_close(regulated[..., 4:], native[..., 4:], atol=0, rtol=0)
    torch.testing.assert_close(repeat, regulated, atol=0, rtol=0)
    torch.testing.assert_close(released, native, atol=0, rtol=0)
    torch.testing.assert_close(after_release[0, 0, 0, 2:4], eye[0] * 0., atol=0, rtol=0)
    assert full.count == 2 and conflicting > 0.
    return zero, first, full, dict(transfer=rotation.tolist(), first_support=partial_trace["support"].tolist(),
                                   quadratic_bias=(curved[:, 0]-rotation[:, 0]).tolist(),
                                   old_new_mixture_sse=float(conflicting), release_restores_native=True,
                                   new_arrivals_form_zero_relation=True,
                                   first_burst=burst(eye[0], rotation).tolist(),
                                   second_burst=burst(eye[1], rotation).tolist())


def physical_relation(bank, destination):
    rows = []
    for root, labels in enumerate(bank["physical"]):
        object_motion = labels[..., 2:4].astype(np.float64)
        pusher = labels[..., :2].astype(np.float64) + object_motion
        engaged = bank["query_contacts"][root, :3].all()
        for anchor, name in ((3, "hold"), (0, "straight")):
            others = [i for i in range(4) if i != anchor]
            p = (pusher[others] - pusher[anchor]).transpose(1, 2, 0)
            o = (object_motion[others] - object_motion[anchor]).transpose(1, 2, 0)
            both_p = np.concatenate(tuple(p), axis=-1)
            joint = np.concatenate(tuple(o), axis=-1) @ np.linalg.pinv(both_p)
            local = o @ np.linalg.pinv(p)
            for horizon in range(2):
                s = np.linalg.svd(p[horizon], compute_uv=False)
                for index, candidate in enumerate(others):
                    u, v = p[horizon, :, index], o[horizon, :, index]
                    rows.append(dict(scene=int(bank["scene_id"][root]), world=int(bank["world_id"][root]),
                                     anchor=name, horizon=horizon+1, candidate=candidate,
                                     all_nonzero_contact=bool(engaged), pusher_x=float(u[0]), pusher_y=float(u[1]),
                                     object_x=float(v[0]), object_y=float(v[1]), object_effect=float(np.linalg.norm(v)),
                                     min_singular=float(s[-1]),
                                     same_horizon_error=float(np.linalg.norm(local[horizon] @ u-v)),
                                     joint_fit_error=float(np.linalg.norm(joint @ u-v)),
                                     one_step_fit_error=float(np.linalg.norm(local[0] @ u-v))))
    summary = {}
    for name in ("hold", "straight"):
        for scope in ("all", "all_nonzero_contact"):
            for horizon in (1, 2):
                subset = [row for row in rows if row["anchor"] == name and row["horizon"] == horizon
                          and (scope == "all" or row["all_nonzero_contact"])]
                keys = ("object_effect", "same_horizon_error", "joint_fit_error", "one_step_fit_error", "min_singular")
                summary[f"{name}_{scope}_h{horizon}"] = dict(roots=len(subset)//3, **{
                    key: float(np.mean([row[key] for row in subset])) for key in keys
                })
    write_csv(destination / "physical_relation.csv", rows)
    write_json(destination / "physical_relation.json", summary)


@torch.inference_mode()
def run_native(evidence, destination):
    host, moments, device = load_host()
    bank = sequence_bank()
    roots = np.arange(0, 32, 4)

    def tensor(array):
        return torch.as_tensor(array, device=device, dtype=torch.float32)

    def embed(raw):
        command = ((raw - moments[0]) / moments[1]).flatten(-2)
        encoded = host.action_encoder(command.reshape(-1, *command.shape[-2:]))
        return encoded.reshape(*command.shape[:-1], encoded.shape[-1])

    with np.load(CALIBRATION, allow_pickle=False) as fit:
        readout = tensor((fit["readout"] * fit["scale"]).T)
    readout[:2] += readout[2:4]
    r = len(roots)
    z = tensor(bank["history"][roots, 2:5])
    past = tensor(bank["actions"][roots, 2:4])
    queries = tensor(bank["queries"][roots])
    ref1 = (queries[:, :, 0].flatten(-2) == 0.).all(dim=-1)
    ref2 = (queries[:, :, 1].flatten(-2) == 0.).all(dim=-1).reshape(4*r, 1)
    reference = embed(queries.new_zeros((1, 1, 5, 2)))[0, 0]
    actions1 = embed(torch.cat((past[:, None].expand(-1, 4, -1, -1, -1), queries[:, :, :1]), dim=2))
    port = ContactTransferPort(host, readout, reference)
    actions2 = embed(torch.stack((past[:, None, 1].expand(-1, 4, -1, -1), queries[:, :, 0], queries[:, :, 1]), dim=2))
    actions2 = actions2.reshape(4*r, 1, 3, -1)
    rows = []
    summary = {}

    def log_stage(name, horizon, output, trace, relation):
        effect = (output[..., -1, :] - trace["reference"][:, None]).double() @ readout.double().T
        change = (output[..., -1, :] - trace["native"][..., -1, :]).double() @ readout.double().T
        torch.testing.assert_close(output[..., :-1, :], trace["native"][..., :-1, :], atol=2e-6, rtol=0)
        torch.testing.assert_close(change[..., [0, 1, 4, 5]], torch.zeros_like(change[..., [0, 1, 4, 5]]),
                                   atol=2e-4, rtol=0)
        supported = trace["support"]
        anchor = ref1 if horizon == 1 else ref2
        assert not (supported & anchor).any()
        if name == "two_axes":
            assert torch.equal(supported, ~anchor)
        if supported.any():
            target = trace["effect"][..., :2] @ relation[0].T
            torch.testing.assert_close(effect[..., 2:4][supported], target[supported], atol=2e-4, rtol=0)
        torch.testing.assert_close(change[~supported], torch.zeros_like(change[~supported]), atol=2e-4, rtol=0)
        summary[f"{name}_h{horizon}"] = dict(supported=int(supported.sum()), requests=4*r,
                                              max_pusher_change=float(change[..., :2].abs().max()),
                                              max_angle_change=float(change[..., 4:].abs().max()),
                                              max_object_change=float(change[..., 2:4].abs().max()))
        before = trace["effect"].reshape(r, 4, 6).tolist()
        after = effect.reshape(r, 4, 6).tolist()
        endpoints = output[..., -1, :].reshape(r, 4, -1).double()
        program = ((endpoints - endpoints[:, -1:]) @ readout.double().T).tolist()
        active = supported.reshape(r, 4).tolist()
        for i, root in enumerate(roots):
            for j in range(4):
                row = dict(arm=name, horizon=horizon, scene=int(bank["scene_id"][root]),
                           world=int(bank["world_id"][root]), candidate=j, supported=active[i][j])
                for k, label in enumerate(("p_x", "p_y", "o_x", "o_y", "cos", "sin")):
                    row[f"native_{label}"] = before[i][j][k]
                    row[f"routed_{label}"] = after[i][j][k]
                    row[f"program_{label}"] = program[i][j][k]
                rows.append(row)

    relations = {name: tuple(value.to(device) for value in state.relation())
                 for name, state in zip(("clear", "one_axis", "two_axes", "released"), (*evidence, evidence[-1].clear()))}
    for name, relation in relations.items():
        out1, trace1 = port.predict(z, actions1, relation, at_reference=ref1, trace=True)
        log_stage(name, 1, out1, trace1, relation)
        z2 = torch.cat((z[:, None, 1:].expand(-1, 4, -1, -1), out1[:, :, -1:]), dim=2).flatten(0, 1)
        out2, trace2 = port.predict(z2, actions2, relation, at_reference=ref2, trace=True)
        log_stage(name, 2, out2, trace2, relation)
        if name == "clear":
            clear1, clear2 = out1, out2
        if name == "released":
            torch.testing.assert_close(out1, clear1, atol=0, rtol=0)
            torch.testing.assert_close(out2, clear2, atol=0, rtol=0)

    direct = host.predict(z.repeat_interleave(4, dim=0), actions1.flatten(0, 1)).reshape_as(clear1)
    torch.testing.assert_close(clear1, direct, atol=2e-6, rtol=0)
    summary["empty_native_max_abs"] = float((clear1-direct).abs().max())
    summary["readout_singular_values"] = torch.linalg.svdvals(port.head_readout).tolist()
    summary["lift_max_singular_value"] = float(torch.linalg.svdvals(port.lift).max())

    timing = []
    baseline = lambda: host.predict(z[:1].repeat_interleave(4, dim=0), actions1[:1].flatten(0, 1))
    regulated = lambda: port.predict(z[:1], actions1[:1], relations["two_axes"], at_reference=ref1[:1])
    for _ in range(10):
        baseline()
        regulated()
    for index in range(30):
        entry = {"trial": index}
        arms = [("native", baseline), ("regulated", regulated)]
        for name, arm in arms[::1 if index % 2 == 0 else -1]:
            torch.cuda.synchronize(device)
            start = perf_counter()
            arm()
            torch.cuda.synchronize(device)
            entry[name] = (perf_counter()-start) * 1000
        timing.append(entry)
    summary["native_ms_median"] = float(np.median([row["native"] for row in timing]))
    summary["regulated_ms_median"] = float(np.median([row["regulated"] for row in timing]))
    summary["paired_overhead_ms_median"] = float(np.median([row["regulated"]-row["native"] for row in timing]))
    write_csv(destination / "native_responses.csv", rows)
    write_csv(destination / "timing.csv", timing)
    write_json(destination / "native.json", summary)
    physical_relation(bank, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    zero, first, full, results = analytic()
    write_json(args.destination / "analytic.json", results)
    if args.native:
        run_native((zero, first, full), args.destination)


if __name__ == "__main__":
    main()
