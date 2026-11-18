"""Group assignments by the strict inactive relations of one native solve."""

from dataclasses import dataclass
import importlib.util
from pathlib import Path

import torch


_SOURCE = Path(__file__).resolve().parents[2] / "research_sources_2026-10-22/lewm/module.py"
_SPEC = importlib.util.spec_from_file_location("lewm_response_class_module", _SOURCE)
_NATIVE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_NATIVE)
solve_activation_relations = _NATIVE.solve_activation_relations


@dataclass(frozen=True)
class ResponseClass:
    representative: int
    members: torch.Tensor
    activation: torch.Tensor
    pinned: torch.Tensor
    slack_upper: torch.Tensor


def partition_response_classes(activation, directions, thresholds, level_strengths, assignments):
    """Solve once per certified group at one common activation and action.

    ``assignments[n, e]`` selects a column of ``level_strengths[e]``. Group
    member indices refer to the supplied assignment rows, which can be any
    subset of the joint table. Level IDs tie the same parameters across rows.
    Use fixed zeros for off levels when differentiating, as in the binary model.

    Stationarity bounds the solution error in exact arithmetic because the
    objective is 1-strongly convex. The floating-point margin is a conservative
    roundoff guard for the staged prototype and is not a formal error bound.
    Boundary rows stay pinned so shared gradients are used only in strict
    inactive interiors. Full row rank is unnecessary for safe grouping.
    """
    if activation.ndim != 1 or not activation.numel() or directions.shape != (2, activation.numel()):
        raise ValueError("The native solver requires one activation vector and two direction rows")
    if thresholds.shape != (2,):
        raise ValueError("Expected one threshold per direction")
    if level_strengths.ndim != 2 or level_strengths.size(0) != 2 or not level_strengths.size(1):
        raise ValueError("Expected a nonempty table of strength levels for two directions")
    if assignments.ndim != 2 or assignments.size(1) != 2 or assignments.dtype != torch.long:
        raise ValueError("Expected assignment rows of two integer strength level IDs")
    if not activation.is_floating_point() or any(
        tensor.dtype != activation.dtype for tensor in (directions, thresholds, level_strengths)
    ):
        raise ValueError("Expected one floating-point dtype across native inputs and strength levels")
    if any(tensor.device != activation.device for tensor in (directions, thresholds, level_strengths, assignments)):
        raise ValueError("Expected native inputs and assignments on the same device")
    if not all(torch.isfinite(tensor).all() for tensor in (activation, directions, thresholds, level_strengths)):
        raise ValueError("Native inputs and strength levels must be finite")
    if (level_strengths < 0).any() or ((assignments < 0) | (assignments >= level_strengths.size(1))).any():
        raise ValueError("Strengths must be nonnegative and assignments must index the level table")

    remaining = torch.arange(len(assignments), device=assignments.device)
    relation_ids = torch.arange(directions.size(0), device=assignments.device)
    roundoff = 8 * (activation.numel() + directions.size(0) + 2) * torch.finfo(activation.dtype).eps
    groups = []
    while remaining.numel():
        representative = int(remaining[0])
        levels = assignments[representative]
        strengths = level_strengths[relation_ids, levels]
        delta, _, _ = solve_activation_relations(activation, directions, thresholds, strengths)
        corrected = activation + delta

        with torch.no_grad():
            slack = directions @ corrected - thresholds
            hinge_force = strengths * slack.clamp_min(0)
            residual = corrected - activation + directions.T @ hinge_force
            if not torch.isfinite(corrected).all() or not torch.isfinite(slack).all():
                raise RuntimeError("Native relation solve produced a nonfinite activation or slack")

            absolute_directions = directions.abs()
            slack_error = roundoff * (
                absolute_directions @ (activation.abs() + delta.abs()) + thresholds.abs()
            )
            residual_error = roundoff * (
                corrected.abs() + activation.abs() + absolute_directions.T @ hinge_force.abs()
            ) + absolute_directions.T @ (strengths * slack_error)
            distance = torch.linalg.vector_norm(residual.abs() + residual_error)
            slack_upper = slack + slack_error + torch.linalg.vector_norm(directions, dim=1) * distance
            pinned = ~(slack_upper < 0)
            matches = ((assignments[remaining] == levels) | ~pinned).all(dim=1)

        groups.append(ResponseClass(representative, remaining[matches], corrected, pinned, slack_upper))
        remaining = remaining[~matches]

    return tuple(groups)
