"""Load the two arrived PushT contexts and their independently executed forks."""

import numpy as np
import torch

from native_gate_bound_paired_data import BRANCHES, encode
from native_gate_bound_pusht import physical_labels


ACTUAL_FIRST = 1  # The declared +60 degree move from t30 to t35.


@torch.inference_mode()
def cycle_bank(model, split):
    """Return legal issue-time histories and separate offline branch outcomes."""
    if split not in ("train", "test"):
        raise ValueError(f"Unknown PushT cycle split: {split}")
    if model.training:
        raise ValueError("The PushT encoder must be in eval mode")

    start, archived_scenes, selected_scenes = (101000, 96, 16) if split == "train" else (107000, 48, 48)
    roots = 4 * selected_scenes
    choices = np.arange(4)
    at_30 = 4 * choices
    at_35 = 4 * ACTUAL_FIRST + choices
    with np.load(BRANCHES / split / "branches.npz", allow_pickle=False) as archive:
        scene_id = archive["scene_ids"]
        world_id = archive["world_ids"]
        np.testing.assert_array_equal(scene_id, np.repeat(np.arange(start, start + archived_scenes), 4))
        np.testing.assert_array_equal(world_id, np.tile(np.arange(4), archived_scenes))
        np.testing.assert_array_equal(archive["action_pair_indices"], list(np.ndindex(4, 4)))

        prefix = archive["executed_prefix"][:roots]
        prefix_state = archive["prefix_physical_state"][:roots]
        context_rgb = archive["context_rgb"][:roots]
        past = archive["past_actions"][:roots]
        candidates = archive["candidate_actions"][:roots]
        executed = archive["actual_step"][:roots]
        if not (archive["prefix_actual_step"][:roots].all() and executed[:, at_30, :5].all()
                and executed[:, at_35].all()):
            raise ValueError("The selected PushT prefixes and candidate transitions must execute fully")

        first_actions = candidates[:, at_30, 0]
        second_actions = candidates[:, at_35, 1]
        actual_first = first_actions[:, ACTUAL_FIRST]
        np.testing.assert_array_equal(second_actions, first_actions)
        np.testing.assert_array_equal(past, prefix[:, 4:6])
        np.testing.assert_array_equal(past[:, 1], np.zeros_like(past[:, 1]))
        np.testing.assert_array_equal(candidates[:, at_35, 0],
                                      np.broadcast_to(actual_first[:, None], first_actions.shape))

        approach = prefix[:, 1, :1]
        np.testing.assert_array_equal(prefix[:, 1], np.broadcast_to(approach, prefix[:, 1].shape))
        length = np.linalg.norm(approach[:, 0], axis=-1, keepdims=True)
        np.testing.assert_allclose(length, .40, atol=1e-12, rtol=0)
        normal = approach[:, 0] / length
        tangent = np.stack((-normal[:, 1], normal[:, 0]), axis=-1)
        turn = np.sqrt(.75) * tangent
        primitives = .22 * np.stack((normal, .5 * normal + turn, .5 * normal - turn,
                                     np.zeros_like(normal)), axis=1)
        np.testing.assert_allclose(first_actions, np.broadcast_to(primitives[:, :, None], first_actions.shape),
                                   atol=1e-12, rtol=0)

        physical = archive["future_physical_state"][:roots]
        endpoints = archive["future_state"][:roots]
        controls = archive["absolute_control_targets"][:roots]
        contacts = archive["contacts"][:roots]
        contact_points = archive["contact_points"][:roots]
        terminated = archive["terminated"][:roots]
        rgb = archive["future_rgb"]
        rgb_35 = rgb[:roots, at_30, 0]
        rgb_40 = rgb[:roots, at_35, 1]

        # Each t30 request uses the archive's shared RGB and physical prefix.
        for branches in (at_30, at_35):
            initial = physical[:, branches, 0]
            np.testing.assert_array_equal(initial, np.broadcast_to(prefix_state[:, None, 30], initial.shape))
        np.testing.assert_array_equal(physical[:, at_30, 5], endpoints[:, at_30, 0])
        np.testing.assert_array_equal(physical[:, at_35, 10], endpoints[:, at_35, 1])
        np.testing.assert_array_equal(controls[:, at_30, :5],
                                      physical[:, at_30, :5, :2] + 100 * first_actions)
        np.testing.assert_array_equal(controls[:, at_35, 5:],
                                      physical[:, at_35, 5:10, :2] + 100 * second_actions)

        # All t35 requests share the outcome of the one designated t30 action.
        shared = (
            rgb[:roots, at_35, 0], endpoints[:, at_35, 0], physical[:, at_35, :6],
            controls[:, at_35, :5], contacts[:, at_35, :5], contact_points[:, at_35, :5],
            executed[:, at_35, :5], terminated[:, at_35, :6],
        )
        for values in shared:
            np.testing.assert_array_equal(values, np.broadcast_to(values[:, :1], values.shape))
        del rgb, shared

        true_physical = physical_labels(np.stack((endpoints[:, at_30, 0], endpoints[:, at_35, 1]), axis=1))
        branch_contacts = np.stack(((contacts[:, at_30, :5] > 0).any(-1),
                                    (contacts[:, at_35, 5:] > 0).any(-1)), axis=1)

    past_35 = np.stack((past[:, 1], actual_first), axis=1)
    history_actions = np.stack((past, past_35), axis=1)
    repeated = np.broadcast_to(history_actions[:, :, None], (roots, 2, 4, 2, 5, 2))
    query_actions = np.stack((first_actions, second_actions), axis=1)
    commands = np.concatenate((repeated, query_actions[:, :, :, None]), axis=3)

    # The only t35 frame entering the second issue history is the designated arrival.
    arrived = encode(model, np.concatenate((context_rgb, rgb_35[:, ACTUAL_FIRST:ACTUAL_FIRST + 1]), axis=1))
    history = torch.stack((arrived[:, :3], arrived[:, 1:]), dim=1)
    other = choices[choices != ACTUAL_FIRST]
    truth_35 = torch.empty_like(arrived)
    truth_35[:, ACTUAL_FIRST] = arrived[:, 3]
    truth_35[:, other] = encode(model, rgb_35[:, other])
    truth_latent = torch.stack((truth_35, encode(model, rgb_40)), dim=1)
    return dict(history=history.float().cpu().numpy(), commands=commands,
                truth_latent=truth_latent.float().cpu().numpy(), true_physical=true_physical,
                contacts=branch_contacts, normal=normal, scene_id=scene_id[:roots], world_id=world_id[:roots])
