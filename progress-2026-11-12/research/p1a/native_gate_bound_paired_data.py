"""Capture two executed PushT transitions and their same-prefix continuation bank."""

import sys

import numpy as np
import torch

from native_gate_bound_pusht import ARCHIVE, WORKSPACE, capture, physical_labels


BRANCHES = WORKSPACE / "OIC_representation_handoff_20261104/runs/msp_control_20261105"
FIRST = slice(4, 8)  # The fixed +60-degree first primitive, followed by each second primitive.
CAPTURE_KEYS = ("h", "m", "g", "native")

sys.path.insert(0, str(WORKSPACE / "oic_step1"))
from run_native_pusht_host import encode


@torch.inference_mode()
def load_paired(model, moments, split):
    """Return [root, slot, feature] traces, with factual arrivals t30 and t35 first."""
    if split not in ("fit", "calibrate", "evaluate"):
        raise ValueError(f"Unknown paired PushT split: {split}")
    evaluate = split == "evaluate"
    calibrate = split == "calibrate"
    start, count = (107000, 48) if evaluate else (101000, 96)
    path = BRANCHES / ("test" if evaluate else "train") / "branches.npz"
    with np.load(path, allow_pickle=False) as bank:
        scenes, worlds = bank["scene_ids"], bank["world_ids"]
        np.testing.assert_array_equal(scenes, np.repeat(np.arange(start, start + count), 4))
        np.testing.assert_array_equal(worlds, np.tile(np.arange(4), count))
        np.testing.assert_array_equal(bank["action_pair_indices"], list(np.ndindex(4, 4)))

        prefix_actions = bank["executed_prefix"]
        prefix_executed = bank["prefix_actual_step"]
        prefix_state = bank["prefix_physical_state"]
        context = bank["context_rgb"]
        past = bank["past_actions"]
        all_commands = bank["candidate_actions"]
        executed = bank["actual_step"][:, FIRST]
        contact_counts = bank["contacts"][:, FIRST, :5]
        first_contact = (contact_counts[:, 0] > 0).any(axis=-1)

        if evaluate:
            stop_contact = np.full(len(scenes), -1, dtype=np.int8)
            rows = np.arange(len(scenes))
        else:
            with np.load(ARCHIVE / "train_11033026.npz", allow_pickle=False) as original:
                np.testing.assert_array_equal(original["scene_ids"], scenes[::4])
                np.testing.assert_array_equal(original["observation_steps"], np.arange(0, 31, 5))
                np.testing.assert_array_equal(original["reset_state"].reshape(-1, 5), bank["reset_state"])
                np.testing.assert_array_equal(np.repeat(original["executed_actions"], 4, axis=0), prefix_actions)
                np.testing.assert_array_equal(original["actual_step"].reshape(-1, 30), prefix_executed)
                np.testing.assert_array_equal(original["physical_state"].reshape(-1, 31, 10), prefix_state)
                original_context = original["full_rgb"][:, :, 4:].reshape(context.shape)
                np.testing.assert_array_equal(original_context, context)
                stop_contact = (original["contacts"].reshape(-1, 30)[:, 25:30] > 0).any(axis=-1)
            if calibrate:
                complete = prefix_executed[:, 25:30].all(axis=-1) & executed[:, 0, :5].all(axis=-1)
                rows = np.flatnonzero((scenes >= start + 16) & complete & ~stop_contact & ~first_contact)
            else:
                rows = np.arange(16 * 4)

        if not len(rows) or not prefix_executed[rows].all():
            raise ValueError("Paired PushT requires a nonempty, fully executed real prefix")
        required = executed[rows, :, :5] if calibrate else executed[rows]
        if not required.all():
            raise ValueError("Paired PushT requires the requested fixed branches to execute fully")

        past = past[rows]
        np.testing.assert_array_equal(past, prefix_actions[rows, 4:6])
        np.testing.assert_array_equal(past[:, 1], np.zeros_like(past[:, 1]))
        approach = prefix_actions[rows, 1]
        np.testing.assert_array_equal(approach, np.broadcast_to(approach[:, :1], approach.shape))
        magnitude = np.linalg.norm(approach[:, 0], axis=-1, keepdims=True)
        np.testing.assert_allclose(magnitude, .40, rtol=0, atol=1e-12)
        normal = approach[:, 0] / magnitude
        tangent = np.stack((-normal[:, 1], normal[:, 0]), axis=-1)
        plus_turn = .22 * (.5 * normal + np.sqrt(.75) * tangent)
        commands = all_commands[rows, FIRST]
        first = commands[:, 0, 0]
        np.testing.assert_allclose(first, np.broadcast_to(plus_turn[:, None], first.shape), rtol=0, atol=1e-12)
        np.testing.assert_array_equal(commands[:, :, 0], np.broadcast_to(first[:, None], commands[:, :, 0].shape))
        np.testing.assert_array_equal(commands[:, :, 1], all_commands[rows, ::4, 0])

        history = encode(model, context[rows, :2])
        fact_one = capture(model, history, past, moments)
        z30 = encode(model, context[rows, 2])
        history = torch.stack((history[:, 1], z30), dim=1)
        commands_to_35 = np.stack((past[:, 1], first), axis=1)
        fact_two = capture(model, history, commands_to_35, moments)

        rgb35 = bank["future_rgb"][rows, FIRST, 0]
        physical = bank["future_physical_state"][rows, FIRST, :6]
        controls = bank["absolute_control_targets"][rows, FIRST, :5]
        endpoint35 = bank["future_state"][rows, FIRST, 0]
        for values in (rgb35, physical, controls, executed[rows, :, :5],
                       contact_counts[rows], bank["contact_points"][rows, FIRST, :5],
                       bank["terminated"][rows, FIRST, :6]):
            np.testing.assert_array_equal(values, np.broadcast_to(values[:, :1], values.shape))
        state30 = prefix_state[rows, 30]
        np.testing.assert_array_equal(physical[:, :, 0], np.broadcast_to(state30[:, None], physical[:, :, 0].shape))
        np.testing.assert_array_equal(controls, physical[:, :, :5, :2] + 100 * commands[:, :, 0])
        np.testing.assert_array_equal(physical[:, :, 5], endpoint35)

        z35 = encode(model, rgb35[:, 0])
        truth = torch.stack((z30, z35), dim=1)
        true_state = np.stack((state30, endpoint35[:, 0]), axis=1)
        contacts = np.stack((stop_contact[rows], first_contact[rows]), axis=1)
        traces = {key: np.stack((fact_one[key], fact_two[key]), axis=1) for key in CAPTURE_KEYS}
        if not calibrate:
            query_history = torch.cat((history, z35[:, None]), dim=1).repeat_interleave(4, dim=0)
            stops = np.broadcast_to(past[:, None, 1], commands[:, :, 0].shape)
            query_commands = np.concatenate((stops[:, :, None], commands), axis=2).reshape(-1, 3, 5, 2)
            query = capture(model, query_history, query_commands, moments)
            for key in CAPTURE_KEYS:
                traces[key] = np.concatenate((traces[key], query[key].reshape(len(rows), 4, 192)), axis=1)

            # The read uses only the real history through t35 and the declared candidate command.
            rgb40 = bank["future_rgb"][rows, FIRST, 1]
            truth = torch.cat((truth, encode(model, rgb40)), dim=1)
            physical = bank["future_physical_state"][rows, FIRST, 5:]
            controls = bank["absolute_control_targets"][rows, FIRST, 5:]
            endpoint40 = bank["future_state"][rows, FIRST, 1]
            np.testing.assert_array_equal(physical[:, :, 5], endpoint40)
            np.testing.assert_array_equal(controls, physical[:, :, :5, :2] + 100 * commands[:, :, 1])
            true_state = np.concatenate((true_state, endpoint40), axis=1)
            second_contact = (bank["contacts"][rows, FIRST, 5:] > 0).any(axis=-1)
            contacts = np.concatenate((contacts, second_contact), axis=1)

    return dict(**traces, true_latent=truth.cpu().numpy(), true_physical=physical_labels(true_state),
                normal=normal, scene_id=scenes[rows], world_id=worlds[rows], contacts=contacts)
