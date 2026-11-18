"""Enclose one-step physical outputs over all nonnegative relation strengths."""

from itertools import combinations

import torch
from torch import nn


def _dual_vertices(gram, drive):
    """Vertices of the axis segments and the closed two-active relaxation."""
    origin = torch.zeros_like(drive)
    axes = torch.eye(2, dtype=drive.dtype, device=drive.device)
    vertices = [origin, *(axes * (drive / gram.diag()).clamp_min(0)[:, None])]

    # For a positive dual coordinate i, the KKT equation requires
    # (Gm)_i < drive_i at finite strength. Drop strictness for the enclosure.
    normal = torch.cat((-axes, gram))
    limit = torch.cat((origin, drive))
    for i, j in combinations(range(4), 2):
        a, b = normal[i], normal[j]
        det = a[0] * b[1] - a[1] * b[0]
        if det == 0:
            continue
        point = torch.stack((limit[i] * b[1] - a[1] * limit[j],
                             a[0] * limit[j] - limit[i] * b[0])) / det
        # Small outward slack retains intersections rounded across a boundary.
        scale = 1 + normal.abs() @ point.abs() + limit.abs()
        if ((normal @ point - limit) <= 64 * torch.finfo(drive.dtype).eps * scale).all():
            vertices.append(point)
    return torch.stack(vertices)


def _normalized_change_interval(weight, centered_native, tangent, scale_min, scale_max, radial_min):
    # Fold the requested readout before taking componentwise extrema.
    numerator = tangent @ weight.T
    lower = numerator.amin(dim=0)
    upper = numerator.amax(dim=0)
    radial_edge = radial_min * (weight @ centered_native)
    return (torch.minimum(lower / scale_min, lower / scale_max) + radial_edge.clamp_max(0),
            torch.maximum(upper / scale_min, upper / scale_max) + radial_edge.clamp_min(0))


@torch.no_grad()
def response_bounds(model, h, gate, preactivation, coordinates):
    """Bound a single fixed native query in float64 up to arithmetic roundoff.

    The enclosing dual vertices may include limiting loads or relaxed points.
    Physical coordinates use the existing row-vector map ``latent @ W + b``.
    """
    operator = model.response_mixture
    transformer = model.predictor.transformer
    mlp = transformer.layers[-1].mlp.net
    norm, projection = transformer.norm, transformer.output_proj
    head = model.pred_proj
    first, batch_norm, activation, last = head.net
    if (operator.parameterization != "relational" or transformer.training
            or head.training or batch_norm.training):
        raise ValueError("Expected the joint relational operator and the evaluation-mode native head")
    if activation.approximate != "none":
        raise ValueError("The physical bound requires exact GELU in the native prediction head")
    if h.ndim != 1 or h.shape != gate.shape or preactivation.ndim != 1:
        raise ValueError("Expected one native residual, action gate and preactivation vector")

    h, gate, preactivation = h.double(), gate.double(), preactivation.double()
    physical_weight, physical_bias = (value.to(h) for value in coordinates)
    directions = operator.relation_directions.double()
    threshold = operator.relation_thresholds.double()
    if directions.shape != (2, preactivation.numel()) or threshold.shape != (2,):
        raise ValueError("Expected two fixed activation relations and one native query")
    gram = directions @ directions.T
    if torch.linalg.matrix_rank(gram) != 2:
        raise ValueError("The dual enclosure requires a full-rank two-relation Gram matrix")
    drive = mlp[2](preactivation) @ directions.T - threshold
    vertices = _dual_vertices(gram, drive)
    activation_delta = -vertices @ directions
    gated_delta = gate * (activation_delta @ mlp[4].weight.double().T)

    # For centered z=z0+delta and sigma(z)=sqrt(eps+mean(z^2)),
    # z/sigma-z0/sigma0 = tangent/sigma + radial*z0, where
    # radial=((sigma-sigma0)^2-mean(delta^2))/(2*sigma0^2*sigma).
    centered_native = h - h.mean()
    centered_delta = gated_delta - gated_delta.mean(-1, keepdim=True)
    width = h.numel()
    scale_native = (centered_native.square().mean() + norm.eps).sqrt()
    radial_tangent = (centered_delta @ centered_native) / (width * scale_native.square())
    tangent = centered_delta - radial_tangent[:, None] * centered_native
    # The norm of [sqrt(eps), z/sqrt(width)] is convex. Its tangent gives
    # a lower bound and the maximum at the vertices is an upper bound.
    scale_min = (scale_native * (1 + radial_tangent.amin())).clamp_min(norm.eps ** 0.5)
    scale_max = ((centered_native + centered_delta).square().mean(-1) + norm.eps).sqrt().amax()
    radial_min = -centered_delta.square().mean(-1).amax() / (2 * scale_native.square() * scale_min)

    # Fold the fixed affine layers up to the head GELU.
    native_ln = centered_native / scale_native
    if norm.elementwise_affine:
        ln_gain = norm.weight.double()
        native_ln = native_ln * ln_gain
        if norm.bias is not None:
            native_ln = native_ln + norm.bias.double()
    first_weight = first.weight.double()
    linear = first_weight
    if isinstance(projection, nn.Linear):
        weight = projection.weight.double()
        native_ln = weight @ native_ln
        if projection.bias is not None:
            native_ln = native_ln + projection.bias.double()
        linear = linear @ weight
    elif not isinstance(projection, nn.Identity):
        raise ValueError("The native output projection must be affine")
    if norm.elementwise_affine:
        linear = linear * ln_gain
    native_pre = first_weight @ native_ln
    if first.bias is not None:
        native_pre = native_pre + first.bias.double()
    bn_gain = (batch_norm.running_var.double() + batch_norm.eps).rsqrt()
    if batch_norm.affine:
        bn_gain = bn_gain * batch_norm.weight.double()
    native_pre = (native_pre - batch_norm.running_mean.double()) * bn_gain
    if batch_norm.affine:
        native_pre = native_pre + batch_norm.bias.double()
    linear = bn_gain[:, None] * linear

    # Compose the physical readout with the GELU tangent before bounding it.
    # The scalar GELU remainder is at most 0.5 sqrt(2/pi) times change^2.
    output = physical_weight.T @ last.weight.double()
    y_native = output @ activation(native_pre) + physical_bias
    if last.bias is not None:
        y_native = y_native + last.bias.double() @ physical_weight
    density = (-0.5 * native_pre.square()).exp() / (2 * torch.pi) ** 0.5
    gelu_slope = torch.special.ndtr(native_pre) + native_pre * density
    projected_tangent = (output * gelu_slope) @ linear
    interval_args = centered_native, tangent, scale_min, scale_max, radial_min
    hidden_lower, hidden_upper = _normalized_change_interval(linear, *interval_args)
    linear_lower, linear_upper = _normalized_change_interval(projected_tangent, *interval_args)
    hidden_change = torch.maximum(hidden_lower.abs(), hidden_upper.abs())
    remainder = (output.abs() @ hidden_change.square()) / (2 * torch.pi) ** 0.5
    guard = 128 * torch.finfo(h.dtype).eps * (1 + y_native.abs() + remainder
                                             + torch.maximum(linear_lower.abs(), linear_upper.abs()))

    return {
        "y_lower": y_native + linear_lower - remainder - guard,
        "y_upper": y_native + linear_upper + remainder + guard,
        "y_native": y_native,
        "max_dual_norm": vertices.norm(dim=-1).amax(),
        "max_activation_delta_norm": activation_delta.norm(dim=-1).amax(),
        "max_gated_residual_norm": gated_delta.norm(dim=-1).amax(),
        "drive": drive,
        "dual_vertices": vertices,
    }
