"""First physical and visual stages of the contact-face slip relation."""

import argparse
from pathlib import Path

import numpy as np
import torch

from contact_face_data import (
    COLLISION_SLOP, FACE_NORMALS, FACE_TANGENTS, PUSHER_RADIUS, contact_face_bank,
    face_coordinates, observed_face,
)
from native_gate_bound_paired import CALIBRATION
from native_gate_bound_pusht import decode
from native_gate_bound_sequence_data import sequence_bank
from native_response import FaceSlipEvidence, project_swept_cone
from query_feedback_learning import write_csv, write_json


def frame_series(poses, tolerance, *, require_isolation=False):
    face = observed_face(**poses, tolerance=tolerance, require_isolation=require_isolation)
    gap, tangent, _ = face_coordinates(**poses)
    index = np.maximum(face, 0)[..., None]
    s = np.take_along_axis(tangent, index, axis=-1)[..., 0]
    g = np.take_along_axis(gap, index, axis=-1)[..., 0]
    return face, s, g


def form(poses, face, tangent_position, stride):
    def frame(i):
        return tuple(torch.as_tensor(x[:, i].copy()) for x in (face, poses['pusher'], poses['angle'], tangent_position))

    tangents = torch.tensor(FACE_TANGENTS, dtype=torch.float64)
    history = FaceSlipEvidence.start(*frame(0))
    for i in range(stride, face.shape[1], stride):
        history = history.observe(*frame(i), tangents, PUSHER_RADIUS)
    latest = FaceSlipEvidence.start(*frame(face.shape[1]-stride-1))
    latest = latest.observe(*frame(face.shape[1]-1), tangents, PUSHER_RADIUS)
    return history, latest


def analytic():
    face = torch.tensor([0])
    pusher = torch.tensor([[0., 0.]], dtype=torch.float64)
    angle = torch.tensor([0.], dtype=pusher.dtype)
    initial_s = torch.tensor([10.], dtype=pusher.dtype)
    tangents = torch.tensor([[0., 1.]], dtype=pusher.dtype)
    drive = torch.tensor([[0., 1.]], dtype=pusher.dtype)
    quarter = angle + .25
    rolling = FaceSlipEvidence.start(face, pusher, angle, initial_s)
    rolling = rolling.observe(face, pusher+drive, quarter, initial_s-PUSHER_RADIUS*.25,
                              tangents, PUSHER_RADIUS)
    torch.testing.assert_close(rolling.estimate()[0], angle, atol=1e-14, rtol=0)
    matched = []
    for slope in (0., 1.):
        state = FaceSlipEvidence.start(face, pusher-drive, angle, initial_s-slope)
        state = state.observe(face, pusher, angle, initial_s, tangents, PUSHER_RADIUS)
        state = state.observe(face, pusher, angle, initial_s, tangents, PUSHER_RADIUS)
        beta, supported, _ = state.estimate()
        assert bool(supported.item()) and beta.item() == slope
        target = initial_s - PUSHER_RADIUS*.25 + beta
        matched.append(float(target.item()))
    assert matched[1] - matched[0] == 1.
    released = state.observe(face-1, pusher, angle, initial_s, tangents, PUSHER_RADIUS)
    assert not bool(released.estimate()[1].item()) and released.transitions.item() == 0
    vector = torch.tensor([[1., 1.], [-2., 0.], [1., .1]], dtype=pusher.dtype)
    axis = torch.tensor([1., 0.], dtype=pusher.dtype)
    half = torch.tensor([torch.pi/6], dtype=pusher.dtype)
    swept = project_swept_cone(vector, axis, half)
    expected = torch.stack(((vector[0] @ torch.cat((half.cos(), half.sin()))) * torch.cat((half.cos(), half.sin())),
                            vector[1]*0., vector[2]))
    torch.testing.assert_close(swept, expected, atol=1e-14, rtol=0)
    torch.testing.assert_close(project_swept_cone(swept, axis, half), swept, atol=1e-14, rtol=0)
    truth = vector[2]
    assert (swept[0]-truth).square().sum() < (vector[0]-truth).square().sum()
    return dict(rotation_corrected_slip=float(rolling.estimate()[0].item()), matched_history_targets=matched,
                candidate_tangent_difference=matched[1]-matched[0], released_clears_relation=True)


def physical(destination, arrived, truth):
    actual = {key: arrived[key] for key in ('pusher', 'origin', 'angle')}
    outcomes = {key: truth[key] for key in ('pusher', 'origin', 'angle')}
    face, s, gap = frame_series(actual, COLLISION_SLOP)
    future_face, future_s, future_gap = frame_series(outcomes, COLLISION_SLOP)
    data = []
    states = []
    summary = {}
    for stride in (1, 5):
        persistent, latest = form(actual, face, s, stride)
        beta, supported, residual = (x.numpy() for x in persistent.estimate())
        last, last_supported, _ = (x.numpy() for x in latest.estimate())
        endpoint = face[:, -1]
        tangent = FACE_TANGENTS[np.maximum(endpoint, 0)]
        for i, current_face in enumerate(endpoint):
            states.append(dict(scene=int(arrived['scene_id'][i]), world=int(arrived['world_id'][i]),
                               stride=stride, face=int(current_face), gap=float(gap[i, -1]),
                               transitions=int(persistent.transitions[i]), drive_energy=float(persistent.drive_energy[i]),
                               supported=bool(supported[i]), beta=float(beta[i]), residual=float(residual[i]),
                               latest_supported=bool(last_supported[i]), latest_beta=float(last[i])))
        for steps in (5, 10):
            angles = outcomes['angle'][..., :steps+1:stride]
            pushers = outcomes['pusher'][..., :steps+1:stride, :]
            cos, sin = np.cos(angles[..., :-1]), np.sin(angles[..., :-1])
            tx, ty = tangent[:, None, None, 0], tangent[:, None, None, 1]
            world_tangent = np.stack((cos*tx-sin*ty, sin*tx+cos*ty), axis=-1)
            drive = np.sum(np.diff(pushers, axis=-2)*world_tangent, axis=(-2, -1))
            angle_change = np.diff(angles, axis=-1)
            turn = np.arctan2(np.sin(angle_change), np.cos(angle_change)).sum(-1)
            same_face = (endpoint[:, None] >= 0) & (future_face[..., :steps+1] == endpoint[:, None, None]).all(-1)
            contacts = truth['contacts'][..., :steps].all(-1)
            selected = same_face & contacts
            measured_slip = future_s[..., steps] - s[:, None, -1] + PUSHER_RADIUS*turn
            errors = dict(persistent=np.abs(beta[:, None]*drive-measured_slip),
                          latest=np.abs(last[:, None]*drive-measured_slip), rolling=np.abs(measured_slip),
                          unit=np.abs(drive-measured_slip))
            paired = selected & supported[:, None] & last_supported[:, None]
            eligible = selected & supported[:, None]
            summary[f'stride_{stride}_h{steps//5}'] = dict(
                all_branches=int(drive.size), same_face=int(same_face.sum()), eligible=int(eligible.sum()),
                same_face_with_contacts=int(selected.sum()), paired=int(paired.sum()),
                distinct_paired_scenes=len(set(arrived['scene_id'][paired.any(-1)])),
                persistent_only={k: float(v[eligible].mean()) if eligible.any() else None for k,v in errors.items()
                                 if k != 'latest'},
                paired_errors={k: float(v[paired].mean()) if paired.any() else None for k,v in errors.items()},
            )
            for root, branch in np.ndindex(drive.shape):
                row = dict(scene=int(arrived['scene_id'][root]), world=int(arrived['world_id'][root]),
                           candidate=branch, stride=stride, horizon=steps//5, issue_face=int(endpoint[root]),
                           endpoint_face=int(future_face[root, branch, steps]),
                           endpoint_gap=float(future_gap[root, branch, steps]),
                           same_face=bool(same_face[root, branch]), all_ticks_contact=bool(contacts[root, branch]),
                           eligible=bool(eligible[root, branch]), paired=bool(paired[root, branch]),
                           drive=float(drive[root, branch]), turn=float(turn[root, branch]),
                           measured_slip=float(measured_slip[root, branch]) if same_face[root, branch] else None)
                row.update({f'{k}_error': float(v[root, branch]) if paired[root, branch] else None
                            for k,v in errors.items()})
                data.append(row)
    write_csv(destination / 'physical_evidence.csv', states)
    write_csv(destination / 'physical_queries.csv', data)
    write_json(destination / 'physical.json', summary)


def swept_cone_physics(destination, arrived, truth):
    poses = {key: truth[key] for key in ('pusher', 'origin', 'angle')}
    face, _, _ = frame_series(poses, COLLISION_SLOP)
    isolated = observed_face(**poses, require_isolation=True)
    issue = face[..., 0]
    body_normal = torch.tensor(-FACE_NORMALS[np.maximum(issue, 0)])
    angle = torch.as_tensor(np.unwrap(poses['angle'], axis=-1))
    offset = 45. * torch.stack((-angle.sin(), angle.cos()), dim=-1)
    center = torch.as_tensor(poses['origin']) + offset
    rows = []
    summary = {}
    for steps in (5, 10):
        first, last = angle[..., 0], angle[..., steps]
        lowest = angle[..., :steps+1].amin(-1)
        highest = angle[..., :steps+1].amax(-1)
        displacement = center[..., steps, :] - center[..., 0, :]
        selected = ((issue >= 0) & (face[..., :steps+1] == issue[..., None]).all(-1)
                    & truth['contacts'][..., :steps].all(-1) & ((highest-lowest).numpy() < torch.pi))
        isolated_contact = selected & (isolated[..., :steps+1] == issue[..., None]).all(-1)
        errors = {}
        for name, lo, hi in (('source_ray', first, first), ('endpoints', torch.minimum(first, last), torch.maximum(first, last)),
                             ('observed_extrema', lowest, highest)):
            mid = (lo+hi)/2
            half = (hi-lo)/2
            nx, ny = body_normal.unbind(-1)
            axis = torch.stack((mid.cos()*nx-mid.sin()*ny, mid.sin()*nx+mid.cos()*ny), dim=-1)
            nearest = project_swept_cone(displacement, axis, half.clamp_max(torch.pi/2))
            errors[name] = (displacement-nearest).norm(dim=-1).numpy()
        span = (highest-lowest).numpy()
        endpoint_span = (first-last).abs().numpy()
        for root, branch in np.ndindex(selected.shape):
            row = dict(scene=int(arrived['scene_id'][root]), world=int(arrived['world_id'][root]),
                       candidate=branch, horizon=steps//5, issue_face=int(issue[root, branch]),
                       same_face_with_contacts=bool(selected[root, branch]),
                       isolated_contact=bool(isolated_contact[root, branch]),
                       span=float(span[root, branch]), endpoint_span=float(endpoint_span[root, branch]),
                       com_dx=float(displacement[root, branch, 0]), com_dy=float(displacement[root, branch, 1]))
            row.update({f'{key}_distance': float(value[root, branch]) if selected[root, branch] else None
                        for key, value in errors.items()})
            rows.append(row)
        summary[f'h{steps//5}'] = dict(branches=int(selected.sum()),
                                      scenes=len(set(arrived['scene_id'][selected.any(-1)])),
                                      mean_com_displacement=float(displacement.norm(dim=-1).numpy()[selected].mean()),
                                      **{k: dict(mean=float(v[selected].mean()), median=float(np.median(v[selected])),
                                                 p90=float(np.quantile(v[selected], .9)), max=float(v[selected].max()))
                                         for k,v in errors.items()})
        summary[f'h{steps//5}_isolated'] = dict(branches=int(isolated_contact.sum()),
            scenes=len(set(arrived['scene_id'][isolated_contact.any(-1)])), **{
                k: dict(mean=float(v[isolated_contact].mean()), max=float(v[isolated_contact].max()))
                for k,v in errors.items()
            })
    write_csv(destination / 'cone_physical_queries.csv', rows)
    write_json(destination / 'cone_physical.json', summary)


def visual(destination, arrived):
    cache = sequence_bank()
    np.testing.assert_array_equal(cache['scene_id'], arrived['scene_id'])
    np.testing.assert_array_equal(cache['world_id'], arrived['world_id'])
    with np.load(CALIBRATION, allow_pickle=False) as fit:
        decoded = decode(cache['history'], fit)
    vision = dict(pusher=decoded[..., :2]+decoded[..., 2:4], origin=decoded[..., 2:4],
                  angle=np.arctan2(decoded[..., 5], decoded[..., 4]))
    physical_pose = {key: arrived[key][:, ::5] for key in vision}
    actual_face, actual_s, _ = frame_series(physical_pose, COLLISION_SLOP)
    pitch = 512./96.
    vision_face, vision_s, _ = frame_series(vision, pitch)
    state, _ = form(vision, vision_face, vision_s, 1)
    _, identified, _ = state.estimate()
    named = actual_face >= 0
    correct = (vision_face == actual_face) & named
    overlap = named & (vision_face >= 0)
    turn = vision['angle'] - physical_pose['angle']
    angle_error = np.abs(np.arctan2(np.sin(turn), np.cos(turn)))
    rows = []
    for root, i in np.ndindex(actual_face.shape):
        rows.append(dict(scene=int(arrived['scene_id'][root]), world=int(arrived['world_id'][root]), tick=5*i,
                         physical_face=int(actual_face[root, i]), visual_face=int(vision_face[root, i]),
                         physical_s=float(actual_s[root, i]), visual_s=float(vision_s[root, i]),
                         angle_error=float(angle_error[root, i])))
    write_csv(destination / 'visual_faces.csv', rows)
    write_json(destination / 'visual.json', dict(frames=int(named.size), physical_faces=int(named.sum()),
               detected_faces=int((vision_face >= 0).sum()), both_have_face=int(overlap.sum()), correct=int(correct.sum()),
               t20_physical=int(named[:, -1].sum()), t20_detected=int((vision_face[:, -1] >= 0).sum()),
               t20_correct=int(correct[:, -1].sum()), t20_slip_identified=int(identified.sum()),
               t20_correct_with_slip=int((identified.numpy() & correct[:, -1]).sum()),
               angle_median_radians=float(np.median(angle_error))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    write_json(args.destination / 'analytic.json', analytic())
    arrived, _, truth = contact_face_bank()
    physical(args.destination, arrived, truth)
    swept_cone_physics(args.destination, arrived, truth)
    visual(args.destination, arrived)


if __name__ == '__main__':
    main()
