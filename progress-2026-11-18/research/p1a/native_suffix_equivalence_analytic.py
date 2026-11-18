"""Executable finite examples for the three-token intervention and real refresh."""

import json

import numpy as np


def three_token_case(square_readout):
    original = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.int64)
    paths = {}
    for name, factors in (("native", (-1, 2, 1, 1)), ("cut", (1, 2, 1, 1))):
        window = original.copy()
        appended = []
        for factor in factors:
            oldest = window[0, 0]
            p, q = window[-1]
            readout = oldest**2 if square_readout else oldest
            next_token = np.array([factor * p, q + readout])
            window = np.vstack((window[1:], next_token))
            appended.append(next_token.tolist())
        paths[name] = appended
    native, cut = (np.array(paths[name]) for name in ("native", "cut"))
    np.testing.assert_array_equal(native[:, 0], -cut[:, 0])
    if square_readout:
        np.testing.assert_array_equal(native[:, 1], cut[:, 1])
    else:
        np.testing.assert_array_equal(native[:3, 1], cut[:3, 1])
        assert native[3, 1] != cut[3, 1]
    return paths


def observation_gain(measurement, editable, invisible_editable):
    visible_to_sensor = measurement @ invisible_editable
    explained = visible_to_sensor @ np.linalg.pinv(visible_to_sensor)
    return (
        np.linalg.pinv(visible_to_sensor) @ explained
        + np.linalg.pinv(measurement @ editable) @ (np.eye(len(measurement)) - explained)
    )


def refresh_cases():
    before = np.array([[0., 0.], [0., 1.]])
    measurement = np.array([[1., 1.]])
    allowed = np.eye(2)
    invisible = np.diag([0., 1.])
    residual = 1. - before @ measurement.T
    ordinary = before + residual @ np.linalg.pinv(measurement).T
    gain = observation_gain(measurement, allowed, invisible)
    protected = before + residual @ gain.T
    np.testing.assert_allclose(ordinary[:, 0], [0.5, 0.])
    np.testing.assert_allclose(protected, [[0., 1.], [0., 1.]])
    np.testing.assert_allclose(protected @ measurement.T, [[1.], [1.]])

    direct = np.array([[1., 0.]])
    direct_gain = observation_gain(direct, allowed, invisible)
    after_direct = protected + (1. - protected @ direct.T) @ direct_gain.T
    np.testing.assert_allclose(after_direct, [[1., 1.], [1., 1.]])

    old_and_new = np.array([[0., 0.], [1., 1.]])
    compare = np.array([1., -1.])
    newest = np.array([[0., 1.]])
    restricted_gain = observation_gain(newest, np.diag([0., 1.]), np.zeros((2, 2)))
    after_latest = old_and_new - (old_and_new @ newest.T) @ restricted_gain.T
    np.testing.assert_allclose(old_and_new @ compare, [0., 0.])
    np.testing.assert_allclose(after_latest @ newest.T, [[0.], [0.]])
    np.testing.assert_allclose(after_latest @ compare, [0., 1.])

    return {
        "same_observation_round_one": {
            "input": before.tolist(),
            "ordinary": ordinary.tolist(),
            "protected": protected.tolist(),
        },
        "direct_observation_round_two": after_direct.tolist(),
        "history_cannot_be_rewritten": {
            "input": old_and_new.tolist(),
            "after": after_latest.tolist(),
            "protected_readout_before": (old_and_new @ compare).tolist(),
            "protected_readout_after": (after_latest @ compare).tolist(),
        },
    }


def formation_across_real_arrivals():
    invisible = np.diag([0., 1., 1.])
    measurement = np.array([[0., 1., 0.]])
    factual_action = np.eye(3)[[0, 2, 1]]
    queries = np.array([[0., 1., 0.], [0., 0., 1.]])
    rounds = []
    for rows, observed in (
        (measurement, [2.]),
        (np.vstack((measurement, measurement @ factual_action)), [2., -1.]),
    ):
        sensor = rows @ invisible
        inverse = np.linalg.pinv(sensor)
        compatible = inverse @ observed
        unresolved = invisible - inverse @ sensor
        determined = np.linalg.norm(queries @ unresolved, axis=1) < 1e-10
        rounds.append({
            "compatible_reference": compatible.tolist(),
            "determined_queries": determined.tolist(),
            "identified_effects": [
                float(row @ compatible) if identifiable else None
                for row, identifiable in zip(queries, determined)
            ],
        })
    assert rounds[0]["identified_effects"] == [2., None]
    assert rounds[1]["identified_effects"] == [2., -1.]

    sensor = np.array([[1., 0.]])
    unresolved_queries = np.array([[0., 1.], [1., 1.]])
    constraints = np.vstack((sensor, unresolved_queries))
    right_side = np.array([1., 0., 0.])
    best_joint = np.linalg.pinv(constraints) @ right_side
    residual = constraints @ best_joint - right_side
    assert np.linalg.norm(residual) > 0.5
    return {
        "real_arrivals": rounds,
        "cannot_leave_both_unresolved_queries_native": float(np.linalg.norm(residual)),
    }


if __name__ == "__main__":
    print(json.dumps({
        "suffix_insensitive_to_sign": three_token_case(square_readout=True),
        "suffix_reveals_sign_at_h4": three_token_case(square_readout=False),
        "real_refresh": refresh_cases(),
        "evidence_for_candidate_effects": formation_across_real_arrivals(),
    }, indent=2))
