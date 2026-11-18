"""Load the fixed PushT factual prefix through t20 and its two-step continuations."""

import numpy as np

from native_gate_bound_pusht import ARCHIVE, physical_labels


def sequence_bank():
    """Return scene-major roots with arrived latents and separate future targets."""
    name = "confirmation_11034026"
    scenes = np.arange(102000, 102048)
    worlds = 4
    roots = len(scenes) * worlds
    query_slot = 2  # t20 in the archived query order.
    with np.load(ARCHIVE / f"{name}.npz", allow_pickle=False) as panel, np.load(
        ARCHIVE / f"{name}_native48_latents.npz", allow_pickle=False
    ) as cache:
        np.testing.assert_array_equal(panel["scene_ids"], scenes)
        np.testing.assert_array_equal(cache["scene_ids"], scenes)
        np.testing.assert_array_equal(panel["observation_steps"], np.arange(0, 31, 5))
        np.testing.assert_array_equal(panel["query_steps"], [10, 15, 20, 30])
        np.testing.assert_array_equal(cache["query_steps"], panel["query_steps"])

        history = cache["full_latents"][:, :, :5]
        actions = panel["executed_actions"][:, :4]
        queries = panel["query_actions"]
        targets = cache["future_latents"][:, :, query_slot]
        factual_states = panel["physical_state"][:, :, :21]
        branch_states = panel["query_physical_state"][:, :, query_slot]
        factual_contacts = panel["contacts"][:, :, 5:20]
        branch_contacts = panel["query_contacts"][:, :, query_slot]
        if not (
            panel["actual_step"][:, :, :20].all()
            and panel["query_actual_step"][:, :, query_slot].all()
            and not panel["terminated"][:, :, :21].any()
            and not panel["query_future_terminated"][:, :, query_slot].any()
        ):
            raise ValueError("The fixed PushT prefix and all t20 continuations must execute fully")

        np.testing.assert_array_equal(history.shape, (len(scenes), worlds, 5, 192))
        np.testing.assert_array_equal(targets.shape, (len(scenes), worlds, 4, 2, 192))
        np.testing.assert_array_equal(actions.shape, (len(scenes), 4, 5, 2))
        np.testing.assert_array_equal(queries.shape, (len(scenes), 4, 2, 5, 2))
        np.testing.assert_array_equal(
            branch_states[..., 0, :],
            np.broadcast_to(factual_states[:, :, None, 20, :], branch_states[..., 0, :].shape),
        )
        np.testing.assert_array_equal(
            panel["absolute_control_targets"][:, :, :20],
            factual_states[:, :, :20, :2] + 100 * actions.reshape(len(scenes), 1, 20, 2),
        )
        np.testing.assert_array_equal(
            panel["query_absolute_control_targets"][:, :, query_slot],
            branch_states[..., :10, :2] + 100 * queries.reshape(len(scenes), 1, 4, 10, 2),
        )

    np.testing.assert_array_equal(actions, np.broadcast_to(actions[:, :, :1], actions.shape))
    np.testing.assert_array_equal(actions[:, [0, 2]], 0.0)
    approach = actions[:, 1, 0]
    magnitude = np.linalg.norm(approach, axis=-1, keepdims=True)
    np.testing.assert_allclose(magnitude, 0.40, rtol=0, atol=1e-12)
    normal = approach / magnitude
    tangent = np.stack((-normal[:, 1], normal[:, 0]), axis=-1)
    np.testing.assert_allclose(actions[:, 3, 0], 0.10 * tangent, rtol=0, atol=1e-12)

    turn = np.sqrt(0.75) * tangent
    primitives = 0.22 * np.stack(
        (normal, 0.5 * normal + turn, 0.5 * normal - turn, np.zeros_like(normal)), axis=1
    )
    np.testing.assert_allclose(queries[:, :, 0, 0], primitives, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(queries, np.broadcast_to(queries[:, :, :1, :1], queries.shape))

    return dict(
        history=history.reshape(roots, 5, 192),
        actions=np.repeat(actions, worlds, axis=0),
        queries=np.repeat(queries, worlds, axis=0),
        targets=targets.reshape(roots, 4, 2, 192),
        physical=physical_labels(branch_states[..., [5, 10], :]).reshape(roots, 4, 2, 6),
        normal=np.repeat(normal, worlds, axis=0),
        scene_id=np.repeat(scenes, worlds),
        world_id=np.tile(np.arange(worlds), len(scenes)),
        fact_contacts=(factual_contacts.reshape(roots, 3, 5) > 0).any(axis=-1),
        query_contacts=(branch_contacts.reshape(roots, 4, 2, 5) > 0).any(axis=-1),
        fact_true_physical=physical_labels(factual_states[:, :, [10, 15, 20]]).reshape(roots, 3, 6),
    )
