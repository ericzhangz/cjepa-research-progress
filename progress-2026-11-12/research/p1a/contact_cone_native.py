"""Causal contact-face displacement cone through the shared native LeWM head."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from contact_face_data import (
    COLLISION_SLOP, FACE_LENGTHS, FACE_NORMALS, FACE_SEGMENTS, FACE_TANGENTS,
    PUSHER_RADIUS, contact_face_bank, observed_face,
)
from contact_face_stage import form, frame_series
from native_gate_bound_paired import CALIBRATION, load_host
from native_gate_bound_pusht import decode
from native_gate_bound_sequence_data import sequence_bank
from native_response import ContactTransferPort, project_swept_cone
from query_feedback_learning import write_csv, write_json


PITCH = 512. / 96.


def rotate(vector, angle):
    x, y = vector.unbind(-1)
    c, s = angle.cos(), angle.sin()
    return torch.stack((c*x-s*y, s*x+c*y), dim=-1)


class ConeRequest:
    def __init__(self, center, source_angle, face, supported, commands, geometry, *, sweep, require_isolation, previous=None):
        self.center = center
        self.source_angle = source_angle
        self.face = face
        self.supported = supported
        self.commands = commands
        self.starts, self.tangents, self.normals, self.lengths = geometry
        self.sweep = sweep
        self.require_isolation = require_isolation
        self.previous = source_angle if previous is None else previous

    def __call__(self, pose):
        push, origin, direction = pose.split(2, dim=-1)
        angle = torch.atan2(direction[..., 1], direction[..., 0])
        turn = angle - self.previous
        self.angle = self.previous + torch.atan2(turn.sin(), turn.cos())
        outward = self.normals[self.face.clamp_min(0)]
        closing = (self.commands*rotate(outward, self.previous)).sum(-1) < 0.
        local = rotate(push-origin, -self.angle)[..., None, :] - self.starts
        u = (local*self.tangents).sum(-1)
        n = (local*self.normals).sum(-1)
        distance = torch.hypot(n, u-u.clamp(min=torch.zeros_like(self.lengths), max=self.lengths))
        nearest, nearest_id = distance.min(-1)
        # Ambiguous vertices are excluded at the declared native precision.
        unique = (distance == nearest[..., None]).sum(-1) == 1
        if self.require_isolation:
            unique &= (distance <= PUSHER_RADIUS+PITCH).sum(-1) == 1
        idx = self.face.clamp_min(0)[..., None]
        face_s = u.gather(-1, idx).squeeze(-1)
        face_n = n.gather(-1, idx).squeeze(-1)
        on_face = ((nearest_id == self.face) & unique & (face_s > 0.) & (face_s < self.lengths[self.face.clamp_min(0)])
                   & (face_n > 0.) & (face_n <= PUSHER_RADIUS+PITCH))
        low = torch.minimum(torch.minimum(self.source_angle, self.previous), self.angle)
        high = torch.maximum(torch.maximum(self.source_angle, self.previous), self.angle)
        if not self.sweep:
            low = high = self.source_angle
        self.admitted = (self.supported & closing & on_face & ((high-low) < torch.pi)
                         & (direction.norm(dim=-1) > torch.finfo(direction.dtype).eps))
        origin_offset = 45.*torch.stack((-self.angle.sin(), self.angle.cos()), dim=-1)
        displacement = origin+origin_offset-self.center
        projected = project_swept_cone(displacement, -rotate(outward, (low+high)/2), ((high-low)/2).clamp_max(torch.pi/2))
        self.shift = torch.where(self.admitted[..., None], projected-displacement, 0.)
        self.nominal = pose
        return self.shift


@torch.inference_mode()
def run(destination, *, require_isolation):
    host, moments, device = load_host()
    bank = sequence_bank()
    arrived, commands, truth = contact_face_bank()
    np.testing.assert_array_equal(bank['queries'], commands)
    roots, count = commands.shape[:2]

    def tensor(x):
        return torch.as_tensor(np.array(x), dtype=torch.float64, device=device)

    def embed(raw):
        raw = tensor(raw).float()
        command = ((raw-moments[0])/moments[1]).flatten(-2)
        result = host.action_encoder(command.reshape(-1, *command.shape[-2:]))
        return result.reshape(*command.shape[:-1], result.shape[-1])

    with np.load(CALIBRATION, allow_pickle=False) as fit:
        decoded = decode(bank['history'], fit)
        readout = tensor((fit['readout']*fit['scale']).T)
        offset = tensor(fit['offset']*fit['scale']+fit['mean'])
    readout[:2] += readout[2:4]
    offset[:2] += offset[2:4]
    visual = dict(pusher=decoded[..., :2]+decoded[..., 2:4], origin=decoded[..., 2:4],
                  angle=np.arctan2(decoded[..., 5], decoded[..., 4]))
    exact = {k: arrived[k][:, ::5] for k in visual}
    p0 = tensor(bank['history'][:, 2:5]).float()
    past = bank['actions'][:, None, 2:4].repeat(count, axis=1)
    acts1 = embed(np.concatenate((past, commands[:, :, :1]), axis=2))
    acts2 = embed(np.concatenate((past[:, :, 1:], commands), axis=2)).reshape(roots*count, 1, 3, -1)
    port = ContactTransferPort(host, readout, acts1[0, -1, -1], physical_offset=offset)
    geometry = tuple(tensor(x) for x in (FACE_SEGMENTS[:, 0], FACE_TANGENTS, FACE_NORMALS, FACE_LENGTHS))
    raw = tensor(commands.sum(-2))
    all_pose = {key: truth[key] for key in visual}
    actual_face = observed_face(**all_pose)
    isolated_face = observed_face(**all_pose, require_isolation=True)
    records = []
    summaries = {}
    baselines = {}
    scene_draws = np.random.default_rng(20261112).integers(0, roots//4, size=(4096, roots//4))

    def make_state(poses, tolerance, persistent):
        face, s, _ = frame_series(poses, tolerance, require_isolation=require_isolation)
        evidence, _ = form(poses, face, s, 1)
        valid = (face[:, -1] >= 0) & ((evidence.transitions.numpy() > 0) | (not persistent))
        theta = tensor(poses['angle'][:, -1])
        center = tensor(poses['origin'][:, -1])+45.*torch.stack((-theta.sin(), theta.cos()), dim=-1)
        return center, theta, torch.as_tensor(face[:, -1].copy(), device=device), torch.as_tensor(valid, device=device)

    arms = [('clear', visual, PITCH, False, True), ('visual_current', visual, PITCH, False, True),
            ('visual_ray', visual, PITCH, True, False), ('visual_sweep', visual, PITCH, True, True),
            ('exact_current', exact, COLLISION_SLOP, False, True),
            ('exact_ray', exact, COLLISION_SLOP, True, False), ('exact_sweep', exact, COLLISION_SLOP, True, True)]
    for name, poses, tolerance, persistent, sweep in arms:
        center, angle, face, active = make_state(poses, tolerance, persistent)
        if name == 'clear':
            active = torch.zeros_like(active)
        center, angle, face, active = (x[:, None].expand(roots, count, *x.shape[1:]) for x in (center, angle, face, active))
        issue = ConeRequest(center, angle, face, active, raw[:, :, 0], geometry,
                            sweep=sweep, require_isolation=require_isolation)
        y1 = port.predict_object_pose(p0, acts1, issue)
        next_states = torch.cat((p0[:, None, 1:].expand(-1, count, -1, -1), y1[:, :, -1:]), dim=2).flatten(0, 1)
        final = ConeRequest(center.flatten(0, 1)[:, None], angle.flatten()[:, None], face.flatten()[:, None],
                            issue.admitted.flatten()[:, None], raw[:, :, 1].reshape(-1, 1, 2), geometry,
                            sweep=sweep, require_isolation=require_isolation, previous=issue.angle.flatten()[:, None])
        y2 = port.predict_object_pose(next_states, acts2, final)
        if name == 'clear':
            direct1 = host.predict(p0.repeat_interleave(count, 0), acts1.flatten(0, 1)).reshape_as(y1)
            direct2 = host.predict(next_states, acts2.flatten(0, 1)).reshape_as(y2)
            torch.testing.assert_close(y1, direct1, atol=0, rtol=0)
            torch.testing.assert_close(y2, direct2, atol=0, rtol=0)
        for h, output, stage in ((1, y1, issue), (2, y2, final)):
            prediction = output[..., -1, :].double() @ readout.T + offset
            delta = prediction-stage.nominal
            torch.testing.assert_close(delta[..., [0, 1, 4, 5]], torch.zeros_like(delta[..., [0, 1, 4, 5]]),
                                       atol=3e-4, rtol=0)
            torch.testing.assert_close(delta[..., 2:4], stage.shift, atol=3e-4, rtol=0)
            predicted = prediction.reshape(roots, count, 6).cpu().numpy()
            observed = truth['origin'][..., 5*h, :]
            error = np.linalg.norm(predicted[..., 2:4]-observed, axis=-1)
            contrast = predicted[..., 2:4]-predicted[:, -1:, 2:4]
            physical_contrast = observed-observed[:, -1:]
            contrast_error = np.linalg.norm(contrast-physical_contrast, axis=-1)
            if name == 'clear':
                baselines[h] = (error, contrast_error)
            base_error, base_contrast = baselines[h]
            admitted = stage.admitted.reshape(roots, count).cpu().numpy()
            applied = stage.shift.norm(dim=-1).reshape(roots, count).cpu().numpy() > 0.
            true_same = ((actual_face[..., 0] >= 0) & (actual_face[..., :5*h+1] == actual_face[..., :1]).all(-1)
                         & truth['contacts'][..., :5*h].all(-1))
            true_isolated = true_same & (isolated_face[..., :5*h+1] == actual_face[..., :1]).all(-1)
            scopes = dict(all=np.ones_like(applied), changed=applied, same_face=true_same, isolated=true_isolated)
            summaries[f'{name}_h{h}'] = dict(admitted=int(admitted.sum()), changed_requests=int(applied.sum()),
                                            changed_and_true_face=int((applied & true_same).sum()), **{
                key: dict(n=int(mask.sum()), error=float(error[mask].mean()) if mask.any() else None,
                          error_gain=float((base_error-error)[mask].mean()) if mask.any() else None,
                          contrast_gain=float((base_contrast-contrast_error)[mask].mean()) if mask.any() else None)
                for key, mask in scopes.items()
            })
            for metric, gain in (('position', base_error-error), ('contrast', base_contrast-contrast_error)):
                scene_gain = gain.reshape(roots//4, 4, count).mean((1, 2))
                distribution = scene_gain[scene_draws].mean(1)
                summaries[f'{name}_h{h}'][f'{metric}_gain_ci95'] = np.quantile(distribution, [.025, .975]).tolist()
            for i, j in np.ndindex(error.shape):
                records.append(dict(arm=name, horizon=h, scene=int(arrived['scene_id'][i]), world=int(arrived['world_id'][i]),
                                    candidate=j, issue_face=int(face[i, j]), admitted=bool(admitted[i, j]),
                                    applied=bool(applied[i, j]), truth_same_face=bool(true_same[i, j]),
                                    truth_isolated=bool(true_isolated[i, j]),
                                    predicted_x=float(predicted[i, j, 2]), predicted_y=float(predicted[i, j, 3]),
                                    actual_x=float(observed[i, j, 0]), actual_y=float(observed[i, j, 1]),
                                    error=float(error[i, j]), baseline_error=float(base_error[i, j]),
                                    contrast_error=float(contrast_error[i, j]), baseline_contrast_error=float(base_contrast[i, j])))
        if name == 'exact_sweep':
            timed_roots = issue.admitted.any(dim=-1).nonzero()
            if len(timed_roots):
                k = int(timed_roots[0, 0])
                request = ConeRequest(center[k:k+1], angle[k:k+1], face[k:k+1], active[k:k+1],
                                      raw[k:k+1, :, 0], geometry, sweep=True, require_isolation=require_isolation)
                operations = [('native', lambda: host.predict(p0[k:k+1].repeat_interleave(count, 0), acts1[k])),
                              ('cone', lambda: port.predict_object_pose(p0[k:k+1], acts1[k:k+1], request))]
                for _ in range(10):
                    for _, operation in operations:
                        operation()
                trials = []
                for t in range(30):
                    entry = dict(trial=t)
                    for label, operation in operations[::1 if t % 2 == 0 else -1]:
                        torch.cuda.synchronize(device)
                        start = perf_counter()
                        operation()
                        torch.cuda.synchronize(device)
                        entry[label] = (perf_counter()-start)*1000.
                    trials.append(entry)
                write_csv(destination / 'cone_native_timing.csv', trials)
                summaries['timing'] = dict(root=k, paired_median_ms=float(np.median([e['cone']-e['native'] for e in trials])))
    write_csv(destination / 'cone_native_queries.csv', records)
    write_json(destination / 'cone_native.json', summaries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    parser.add_argument('--require-isolation', action='store_true')
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    run(args.destination, require_isolation=args.require_isolation)
