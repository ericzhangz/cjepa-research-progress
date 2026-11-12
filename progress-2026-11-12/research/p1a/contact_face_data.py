"""Exact PushT poses through t20 and separate t20–30 scoring trajectories."""

from pathlib import Path

import numpy as np


ARCHIVE = (Path(__file__).resolve().parents[2] / "JEPA_OIC_Codex_Handoff_2026-11-03"
           / "runs" / "confirmation_11034026.npz")
PUSHER_RADIUS = 15.0
COLLISION_SLOP = float(np.float32(0.1))  # Default of the gym-pusht 0.1.8 Pymunk space.

# Outer boundary of the two zero-radius polygons in PushTEnv.add_tee (scale=30).
# Positions are relative to block.position, the local origin of both polygons.
FACE_NAMES = ("cap_top", "cap_right", "right_shoulder", "stem_right", "stem_tip",
              "stem_left", "left_shoulder", "cap_left")
_VERTICES = np.array([[-60., 0.], [60., 0.], [60., 30.], [15., 30.],
                      [15., 120.], [-15., 120.], [-15., 30.], [-60., 30.]])
FACE_SEGMENTS = np.stack((_VERTICES, np.roll(_VERTICES, -1, axis=0)), axis=1)
_EDGES = FACE_SEGMENTS[:, 1] - FACE_SEGMENTS[:, 0]
FACE_LENGTHS = np.linalg.norm(_EDGES, axis=-1)
FACE_TANGENTS = _EDGES / FACE_LENGTHS[:, None]
FACE_NORMALS = np.stack((FACE_TANGENTS[:, 1], -FACE_TANGENTS[:, 0]), axis=-1)
for _geometry in (FACE_SEGMENTS, FACE_LENGTHS, FACE_TANGENTS, FACE_NORMALS):
    _geometry.setflags(write=False)


def face_coordinates(pusher, origin, angle):
    """Return outward disk gaps, tangent coordinates, and strict face interiors."""
    x, y = np.moveaxis(np.asarray(pusher) - np.asarray(origin), -1, 0)
    c, s = np.cos(angle), np.sin(angle)
    local_pusher = np.stack((c * x + s * y, -s * x + c * y), axis=-1)
    offset = local_pusher[..., None, :] - FACE_SEGMENTS[:, 0]
    gap = np.sum(offset * FACE_NORMALS, axis=-1) - PUSHER_RADIUS
    tangent = np.sum(offset * FACE_TANGENTS, axis=-1)
    interior = (tangent > 0.) & (tangent < FACE_LENGTHS)
    return gap, tangent, interior


def observed_face(pusher, origin, angle, tolerance=COLLISION_SLOP, *, require_isolation=False):
    """Return the unique nearest outward face within clearance tolerance, or -1."""
    gap, tangent, interior = face_coordinates(pusher, origin, angle)
    normal = gap + PUSHER_RADIUS
    overhang = tangent - np.clip(tangent, 0., FACE_LENGTHS)
    distance = np.hypot(normal, overhang)
    nearest = distance.min(axis=-1, keepdims=True)
    roundoff = 8 * np.finfo(np.float64).eps * FACE_LENGTHS.max()
    tied = np.abs(distance - nearest) <= roundoff
    visible = tied & interior & (normal > 0.) & (gap <= tolerance)
    unique = (tied.sum(axis=-1) == 1) & visible.any(axis=-1)
    if require_isolation:
        unique &= ((distance <= PUSHER_RADIUS+tolerance).sum(axis=-1) == 1)
    return np.where(unique, visible.argmax(axis=-1), -1)


def _poses(state):
    # Columns 0:2 are agent.position, 4:6 block.position and 8 block.angle.
    return dict(pusher=state[..., :2].copy(), origin=state[..., 4:6].copy(),
                angle=state[..., 8].copy())


def contact_face_bank():
    """Return scene-major arrivals, declared queries, and offline scoring truth.

    Arrivals cover t0–20. Each query supplies two blocks of five normalized
    relative controls. Truth includes the shared t20 start and t21–30 outcomes.
    The ten truth contact flags describe the simulator substeps within each
    transition, while the geometric observations describe individual arrivals.
    """
    scenes, worlds = np.arange(102000, 102048), 4
    roots = len(scenes) * worlds
    query_slot = 2
    with np.load(ARCHIVE, allow_pickle=False) as panel:
        np.testing.assert_array_equal(panel["scene_ids"], scenes)
        np.testing.assert_array_equal(panel["observation_steps"], np.arange(0, 31, 5))
        np.testing.assert_array_equal(panel["query_steps"], [10, 15, 20, 30])
        factual = panel["physical_state"][:, :, :21]
        branches = panel["query_physical_state"][:, :, query_slot]
        actions = panel["executed_actions"][:, :4]
        commands = panel["query_actions"]
        contacts = panel["query_contacts"][:, :, query_slot] > 0

        np.testing.assert_array_equal(factual.shape, (len(scenes), worlds, 21, 10))
        np.testing.assert_array_equal(branches.shape, (len(scenes), worlds, 4, 11, 10))
        np.testing.assert_array_equal(actions.shape, (len(scenes), 4, 5, 2))
        np.testing.assert_array_equal(commands.shape, (len(scenes), 4, 2, 5, 2))
        np.testing.assert_array_equal(contacts.shape, (len(scenes), worlds, 4, 10))
        if not (panel["actual_step"][:, :, :20].all()
                and panel["query_actual_step"][:, :, query_slot].all()
                and not panel["terminated"][:, :, :21].any()
                and not panel["query_future_terminated"][:, :, query_slot].any()):
            raise ValueError("The t20 prefix and all continuations must execute fully")

        initial = branches[..., 0, :]
        np.testing.assert_array_equal(initial, np.broadcast_to(factual[:, :, None, 20], initial.shape))
        np.testing.assert_array_equal(panel["absolute_control_targets"][:, :, :20],
                                      factual[:, :, :20, :2] + 100 * actions.reshape(len(scenes), 1, 20, 2))
        np.testing.assert_array_equal(panel["query_absolute_control_targets"][:, :, query_slot],
                                      branches[..., :10, :2] + 100 * commands.reshape(len(scenes), 1, 4, 10, 2))

    arrived = _poses(factual.reshape(roots, 21, 10))
    arrived.update(actions=np.repeat(actions, worlds, axis=0), scene_id=np.repeat(scenes, worlds),
                   world_id=np.tile(np.arange(worlds), len(scenes)))
    truth = _poses(branches.reshape(roots, 4, 11, 10))
    truth["contacts"] = contacts.reshape(roots, 4, 10)
    return arrived, np.repeat(commands, worlds, axis=0), truth
