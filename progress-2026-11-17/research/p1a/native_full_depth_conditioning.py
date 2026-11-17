"""Use a persistent explanation as an input to every native AdaLN block."""

import torch
from torch import nn
from torch.nn import functional as F


class FullDepthConditioner(nn.Module):
    def __init__(self, model, code_width):
        super().__init__()
        transformer = model.predictor.transformer
        self.depth = len(transformer.layers)
        self.width = transformer.norm.normalized_shape[0]
        reduced_width = min(code_width, 16)
        self.shared = nn.Linear(code_width, reduced_width, bias=False)
        self.layers = nn.Linear(reduced_width, self.depth * self.width, bias=False)
        nn.init.zeros_(self.layers.weight)

    def forward(self, code):
        return self.layers(self.shared(code)).unflatten(-1, (self.depth, self.width))


class FirstEffectReadout(nn.Linear):
    def __init__(self, model):
        width = model.predictor.transformer.norm.normalized_shape[0]
        super().__init__(2 * width, 1)
        self.forced_exposure = None
        nn.init.zeros_(self.weight)
        nn.init.ones_(self.bias)

    def logits(self, state, condition):
        width = state.size(-1)
        return self(torch.cat((F.layer_norm(state[:, -1], (width,)),
                               F.layer_norm(condition[:, -1], (width,))), dim=-1)).squeeze(-1)


def _pair_representatives(shared):
    """Representative rows and inverse map for independent two-mode requests."""
    rows = torch.arange(2 * len(shared), device=shared.device)
    source = rows.clone()
    source[1::2] -= shared.long()
    retained = rows == source
    return rows[retained], (retained.cumsum(dim=0) - 1)[source]


def historical_support(log_support, transition):
    """Return the destination prior and P(parent | destination, actual history)."""
    joint = log_support.unsqueeze(-2) + transition.T.log()
    reachable = joint.isfinite().any(dim=-1, keepdim=True)
    safe_joint = torch.where(reachable, joint, 0)
    normalization = safe_joint.logsumexp(dim=-1, keepdim=True)
    log_prior = torch.where(reachable, normalization, -torch.inf).squeeze(-1)
    return log_prior, (safe_joint - normalization).exp() * reachable


def inherit_window(window, ancestry):
    return torch.einsum("bji,bitd->bjtd", ancestry, window)


def edge_posterior(log_weights, transition, log_likelihood):
    """Advance one actual block and condition its parent–destination weights."""
    parent = log_weights.logsumexp(dim=1)
    joint = parent[..., None] + transition.log()
    modes = joint.shape[1:]
    prior = joint.flatten(1).log_softmax(-1).unflatten(-1, modes)
    post = (prior + log_likelihood).flatten(1).log_softmax(-1).unflatten(-1, modes)
    return prior, post


def next_edge_ancestry(log_weights, transition):
    """Condition the retained edge bank on each subsequent query mode."""
    log_path = (log_weights[..., None] + transition.log()).flatten(1, 2).transpose(-1, -2)
    reachable = log_path.isfinite().any(dim=-1, keepdim=True)
    safe = torch.where(reachable, log_path, 0)
    return safe.softmax(-1) * reachable


def _first_effect_suffix(model, state, condition, offsets, common, factor):
    transformer = model.predictor.transformer
    for layer, block in enumerate(transformer.layers[1:], 1):
        common_part = common[:, layer]
        mode_part = offsets[:, layer] - common_part
        applied = torch.where(factor == 0, common_part, common_part + factor * mode_part)
        state = block(state, condition + applied[:, None])
    state = transformer.output_proj(transformer.norm(state))
    return model.pred_proj(state.flatten(0, 1)).unflatten(0, state.shape[:2])[:, -1]


def _relative_attention(block, native_input, reference_input, ancestry):
    """Normalize the last causal row across histories relative to the native hold request."""
    def heads(x):
        return x.unflatten(-1, (block.heads, -1)).transpose(-3, -2)

    q, k, v = map(heads, block.to_qkv(block.norm(native_input)).chunk(3, dim=-1))
    hold_q, hold_k, _ = map(heads, block.to_qkv(block.norm(reference_input)).chunk(3, dim=-1))
    hold_keys = torch.cat((k[..., :-1, :], hold_k), dim=-2)
    scores = (q[..., -1:, :] @ k.transpose(-2, -1)).squeeze(-2) * block.scale
    hold_scores = (hold_q @ hold_keys.transpose(-2, -1)).squeeze(-2) * block.scale
    relative = scores - scores[..., -1:]
    hold_relative = hold_scores - hold_scores[..., -1:]
    log_ratio = relative.logsumexp(-1) - hold_relative.logsumexp(-1)
    log_prior = ancestry.clamp_min(torch.finfo(ancestry.dtype).tiny).log()
    log_prior = log_prior.masked_fill(ancestry == 0, -torch.inf)
    reachable = ancestry.sum(dim=-1, keepdim=True) > 0
    log_mass = log_prior[..., None] + log_ratio[:, None]
    mass = torch.where(reachable[..., None], log_mass, 0).softmax(dim=2) * reachable[..., None]
    answers = (scores.softmax(-1).unsqueeze(-2) @ v).squeeze(-2)
    last = torch.einsum("bjih,bihd->bjhd", mass, answers).flatten(-2)
    prefix = F.scaled_dot_product_attention(q[..., :-1, :], k[..., :-1, :],
                                            v[..., :-1, :], is_causal=True)
    prefix = prefix.transpose(-3, -2).flatten(-2)
    # An unreachable destination had no residual or projection bias in the original inheritance.
    return block.to_out(prefix), block.to_out(last) * reachable, log_ratio, mass


def _inherit_first_effect(model, windows, controls, offsets, readout, ancestry, forced_exposure, trace,
                          reference_action=None):
    """Advance each destination from its weighted bank of native attention responses."""
    predictor = model.predictor
    transformer = predictor.transformer
    block = transformer.layers[0]
    batch, parents, length, _ = windows.shape
    destinations = offsets.size(1)
    state = transformer.input_proj(predictor.dropout(windows + predictor.pos_embedding[:, :length]))
    condition = transformer.cond_proj(controls)
    common = offsets.mean(dim=1)
    first = condition + common[:, 0, None]
    shift, scale, gate, mlp_shift, mlp_scale, mlp_gate = [
        x[:, None] for x in block.adaLN_modulation(first).chunk(6, -1)
    ]
    native_input = block.norm1(state) * (1 + scale) + shift
    if reference_action is None:
        attention = block.attn(native_input.flatten(0, 1)).unflatten(0, (batch, parents))
        residual = state + gate * attention
        inherited = inherit_window(residual, ancestry)
        if trace is not None:
            trace["parent_attention_residual"] = residual
    else:
        hold_condition = transformer.cond_proj(reference_action.expand(batch, -1)) + common[:, 0]
        hold_shift, hold_scale, *_ = block.adaLN_modulation(hold_condition).chunk(6, dim=-1)
        hold_input = block.norm1(state[:, :, -1:]) * (1 + hold_scale[:, None, None]) + hold_shift[:, None, None]
        prefix, last, ratio, mass = _relative_attention(block.attn, native_input, hold_input, ancestry)
        prefix = inherit_window(state[:, :, :-1] + gate[:, :, :-1] * prefix, ancestry)
        last = inherit_window(state[:, :, -1:], ancestry) + (gate[:, :, -1] * last).unsqueeze(-2)
        inherited = torch.cat((prefix, last), dim=-2)
        if trace is not None:
            trace.update(reference_log_mass_ratio=ratio, parent_attention_mass=mass)
    if trace is not None:
        trace["inherited_attention_residual"] = inherited
    mlp_input = block.norm2(inherited) * (1 + mlp_scale) + mlp_shift
    state = inherited + mlp_gate * block.mlp(mlp_input.flatten(0, 1)).unflatten(0, (batch, destinations))
    condition = condition.repeat_interleave(destinations, dim=0)
    logits = readout.logits(state.flatten(0, 1), condition)
    exposure = (2 * logits.sigmoid() - 1).relu()
    if forced_exposure is not None:
        exposure = torch.full_like(exposure, forced_exposure)
    forecast = _first_effect_suffix(model, state.flatten(0, 1), condition, offsets.flatten(0, 1),
                                    common.repeat_interleave(destinations, dim=0), exposure[:, None])
    return forecast, logits, exposure, (batch * parents, batch * destinations)


def _first_effect_step(model, window, controls, offsets, readout, compact, forced_exposure):
    predictor = model.predictor
    transformer = predictor.transformer
    common = offsets.unflatten(0, (-1, 2)).mean(dim=1).repeat_interleave(2, dim=0)
    bits = window.contiguous().view(torch.uint8).flatten(1)
    same_window = (bits[::2] == bits[1::2]).all(dim=-1)
    can_share = same_window if compact else torch.zeros_like(same_window)
    prefix_rows, prefix_inverse = _pair_representatives(can_share)
    state = predictor.dropout(window[prefix_rows] + predictor.pos_embedding[:, :window.size(1)])
    state = transformer.input_proj(state)
    condition = transformer.cond_proj(controls[prefix_rows])
    state = transformer.layers[0](state, condition + common[prefix_rows, 0, None])
    logits = readout.logits(state, condition)[prefix_inverse]
    exposure = (2 * logits.sigmoid() - 1).relu()
    if forced_exposure is not None:
        exposure = torch.full_like(exposure, forced_exposure)
    share_suffix = can_share & (exposure[::2] == 0) & (exposure[1::2] == 0)
    suffix_rows, inverse = _pair_representatives(share_suffix)
    prefix = prefix_inverse[suffix_rows]
    state, condition = state[prefix], condition[prefix]
    factor = exposure[suffix_rows, None]
    forecast = _first_effect_suffix(model, state, condition, offsets[suffix_rows], common[suffix_rows], factor)
    rows = (len(prefix_rows), len(suffix_rows))
    return forecast[inverse], logits, exposure, rows


def conditioned_rollout(model, history, past, actions, offsets, *, first_effect=None, compact=False,
                        trace=None, forced_exposure=None, ancestry=None, reference_action=None):
    """Expand separate native windows for explicit [batch, candidate, mode] requests."""
    batch, candidates, steps, action_width = actions.shape
    modes = offsets.size(1)
    if first_effect is not None and (modes != 2 or model.training):
        raise ValueError("First-effect routing uses two modes in native eval mode")
    if first_effect is not None and forced_exposure is None:
        forced_exposure = first_effect.forced_exposure
    if history.ndim == 3:
        history = history[:, None].expand(-1, modes, -1, -1)
    lineage = history.size(1)
    if ancestry is None and lineage != modes:
        raise ValueError("Independent rollout requires one history window per mode")
    if ancestry is not None and (first_effect is None or ancestry.shape != (batch, modes, lineage)):
        raise ValueError("History inheritance requires a distribution over histories for each mode")
    length, latent_width = history.shape[-2:]
    controls = torch.cat((past[:, None].expand(-1, candidates, -1, -1), actions), dim=2)
    controls = controls[:, :, None].expand(batch, candidates, modes, length + steps - 1, action_width)
    controls = controls.flatten(0, 2)
    window = history[:, None].expand(batch, candidates, lineage, length, latent_width).flatten(0, 2)
    offsets = offsets[:, None].expand(-1, candidates, -1, -1, -1).flatten(0, 2)
    forecasts, logits, exposures, rows = [], [], [], []
    for step in range(steps):
        command = controls[:, step:step + length]
        parents = None
        if step == 0 and ancestry is not None:
            parents = ancestry[:, None].expand(-1, candidates, -1, -1).flatten(0, 1)
        if parents is not None:
            bank = window.unflatten(0, (-1, lineage))
            arrival, score, exposure, step_rows = _inherit_first_effect(
                model, bank, command[::modes], offsets.unflatten(0, (-1, modes)),
                first_effect, parents, forced_exposure, trace, reference_action,
            )
            window = inherit_window(bank, parents).flatten(0, 1)
        elif first_effect is not None:
            arrival, score, exposure, step_rows = _first_effect_step(
                model, window, command, offsets, first_effect, compact, forced_exposure,
            )
        else:
            arrival = model.predict(window, command, condition_offsets=offsets)[:, -1]
        if first_effect is not None:
            logits.append(score.reshape(batch, candidates, modes))
            exposures.append(exposure.reshape(batch, candidates, modes))
            rows.append(step_rows)
        forecasts.append(arrival)
        window = torch.cat((window[:, 1:], arrival[:, None]), dim=1)
    if trace is not None and first_effect is not None:
        trace.update(logits=torch.stack(logits, dim=-1), exposure=torch.stack(exposures, dim=-1), rows=rows)
    return torch.stack(forecasts, dim=1).reshape(batch, candidates, modes, steps, latent_width)


def advance_log_support(log_support, switch_probability):
    """Binary physical regime transition at an actual control-block boundary."""
    if switch_probability == 0:
        return log_support
    stay = log_support.new_tensor(1.0 - switch_probability).log()
    switch = log_support.new_tensor(switch_probability).log()
    return torch.logaddexp(log_support + stay, log_support.flip(-1) + switch)


def write_observation(log_support, issued, arrived, scale, *, decode=None):
    """Score the issued action with either latent scales or a frozen XY covariance.

    For decoded evidence, scale is the lower Cholesky factor of the shared 2x2
    residual covariance and decode is the same frozen map for both latents.
    """
    if decode is None:
        standardized = (issued - arrived[..., None, :]) / scale
    else:
        residual = (decode(issued).to(scale) - decode(arrived).to(scale)[..., None, :])
        if scale.shape != (2, 2) or residual.shape[-1] != 2:
            raise ValueError("Decoded position evidence requires XY and a 2x2 Cholesky factor")
        standardized = torch.linalg.solve_triangular(scale, residual[..., None], upper=False).squeeze(-1)
    log_likelihood = -0.5 * standardized.square().sum(dim=-1)
    return (log_support + log_likelihood).log_softmax(dim=-1), log_likelihood
