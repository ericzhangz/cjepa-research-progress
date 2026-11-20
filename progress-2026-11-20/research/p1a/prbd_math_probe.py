"""Fixed, no-fit PRBD response-transfer experiment. Run as a module from the project root."""

from .prbd_paths import ROOT as RELEASE_ROOT, output_path

import json
from pathlib import Path

import torch

from .prbd import assimilate, issue, quadrature


def tensor(values):
    return torch.tensor(values, dtype=torch.float64)


def outcomes(start, commands, gains):
    state = start.expand(len(gains), -1)
    for action in commands:
        state = state + gains * tensor(action)
    return state


def actual_endpoint(start, commands, gains):
    """Independent analytic endpoint for the constant-gain plant."""
    return start + tensor([gains[d] * sum(a[d] for a in commands) for d in range(2)])


def equal(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-10)


def transfer_checks():
    gains = tensor([[1, 1], [1, 3], [2, 1], [2, 3]])
    weights = tensor([1 / 2, 1 / 4, 1 / 6, 1 / 12])
    identity = torch.eye(len(gains), dtype=weights.dtype)
    noise = torch.eye(2, dtype=weights.dtype) * .5
    scales, masses = quadrature(nu=3., order=32)
    start = tensor([0, 0])
    probe = [(1, 0)]
    pending = issue(outcomes(start, probe, gains), weights, identity, noise)
    programs = {
        "diagonal": [(1, 1)],
        "reverse_horizontal": [(-1, 1)],
        "vertical": [(0, 1)],
        "diagonal_then_vertical_h4": [(1, 1), (0, 1), (0, 1), (0, 1)],
    }
    rows = []
    for true_gain in gains.tolist():
        # Only the factual observation is passed to assimilation.
        arrival = actual_endpoint(start, probe, true_gain)
        result = assimilate(pending, arrival, scales, masses)
        before, after = pending.prior, result.posterior
        class_odds = after[::2] / after[1::2]
        equal(class_odds, before[::2] / before[1::2])
        equal(result.log_likelihood[::2], result.log_likelihood[1::2])
        relative = (result.log_likelihood - result.log_likelihood.max()).exp()
        queries = {}
        for name, commands in programs.items():
            # Compare writes with the same refreshed root, action program and branches.
            ys = outcomes(arrival, commands, gains)
            left, right = before @ ys, after @ ys
            truth = actual_endpoint(arrival, commands, true_gain)
            shift = right - left
            covariance = (before * relative) @ (ys - left) / (before @ relative)
            equal(shift, covariance)
            equal(shift[1], tensor(0.))
            if name == "vertical":
                equal(right, left)
                equal(after[[0, 2]].sum(), before[[0, 2]].sum())
            else:
                assert abs(right[0] - truth[0]) < abs(left[0] - truth[0])
            queries[name] = dict(actions=commands, before=left.tolist(), after=right.tolist(),
                                 actual=truth.tolist(), shift=shift.tolist(),
                                 covariance_over_evidence=covariance.tolist())
        assert abs(queries["diagonal"]["shift"][0] + queries["reverse_horizontal"]["shift"][0]) < 1e-10
        equal(tensor(queries["diagonal"]["shift"]), tensor(queries["diagonal_then_vertical_h4"]["shift"]))

        # Break only the cross-action identity: same factual evidence and candidate values.
        diagonal = outcomes(arrival, programs["diagonal"], gains)
        swapped = diagonal[[2, 3, 0, 1]]
        swapped_before, swapped_after = before @ swapped, after @ swapped
        swapped_shift = swapped_after - swapped_before
        assert swapped_shift[0] * queries["diagonal"]["shift"][0] < 0
        true_x = queries["diagonal"]["actual"][0]
        assert abs(swapped_after[0] - true_x) > abs(swapped_before[0] - true_x)

        quiet_actions = [(0, 0)]
        quiet = issue(outcomes(arrival, quiet_actions, gains), after, identity, noise)
        quiet_arrival = actual_endpoint(arrival, quiet_actions, true_gain)
        retained = assimilate(quiet, quiet_arrival, scales, masses).posterior
        equal(retained, after)
        vertical = issue(outcomes(quiet_arrival, programs["vertical"], gains), retained, identity, noise)
        vertical_arrival = actual_endpoint(quiet_arrival, programs["vertical"], true_gain)
        revised = assimilate(vertical, vertical_arrival, scales, masses).posterior
        equal(revised[2:].sum(), after[2:].sum())
        assert abs(revised @ gains[:, 1] - true_gain[1]) < abs(retained @ gains[:, 1] - true_gain[1])
        rows.append(dict(actual_gain_for_scoring=true_gain, observation=arrival.tolist(),
                         prior=before.tolist(), posterior=after.tolist(), within_class_odds=class_odds.tolist(),
                         log_likelihood=result.log_likelihood.tolist(), log_predictive=result.log_predictive.item(),
                         queries=queries, swapped_diagonal_before=swapped_before.tolist(),
                         swapped_diagonal_after=swapped_after.tolist(), quiet_posterior=retained.tolist(),
                         second_observation=vertical_arrival.tolist(), after_vertical_probe=revised.tolist()))
    return rows


if __name__ == "__main__":
    from .prbd_checks import numerical_checks

    torch.set_num_threads(1)
    archive = RELEASE_ROOT / "runs" / "prbd_math_20261119"
    result = dict(torch_version=str(torch.__version__), dtype="float64", nu=3, quadrature_order=32,
                  transfer=transfer_checks(), numerics=numerical_checks())
    (output_path(archive / "math.json")).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(archive / "math.json")
