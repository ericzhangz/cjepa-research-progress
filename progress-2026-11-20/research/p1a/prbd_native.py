"""Native LeWM action encoding, exact anchors and owned prototype rollout.

Core definitions copied from the executed P3b path; historical data drivers omitted.
"""

import numpy as np
import torch
from torch import nn


@torch.no_grad()
def encode_native_controls(host, raw, stats):
    """Embed independent five-primitive commands with one fixed native shape.

    The released action encoder is pointwise in time. Evaluating each block as
    [1,1,10] makes the raw-zero embedding bitwise identical in every batch and
    avoids a raw-action detector or special zero route in the predictor.
    """
    reference = next(host.parameters())
    raw = torch.as_tensor(raw, device=reference.device, dtype=reference.dtype)
    if raw.shape[-2:] != (5, 2) or not torch.isfinite(raw).all():
        raise ValueError("Expected finite raw controls [...,5,2]")
    if any(module.training for module in host.modules()):
        raise ValueError("Native action encoding requires an eval-mode frozen host")
    if host.action_encoder.patch_embed.kernel_size != (1,):
        raise ValueError("Independent block encoding requires the pointwise native embedder")
    mean = torch.as_tensor(stats["mean"], device=raw.device, dtype=raw.dtype)
    std = torch.as_tensor(stats["std_sample_correction_1"], device=raw.device, dtype=raw.dtype)
    if mean.shape != (1, 2) or std.shape != (1, 2) or not (std > 0).all():
        raise ValueError("Expected official primitive-action mean and positive sample std")
    blocks = ((raw - mean) / std).reshape(-1, 1, 10)
    if len(blocks) == 0:
        raise ValueError("At least one native action block is required")
    encoded = torch.cat([host.action_encoder(block[None])[:, 0] for block in blocks])
    return encoded.reshape(*raw.shape[:-2], encoded.size(-1))



def native_conditioned(host, window, controls, offsets):
    """Apply each LeWM block's offset to the latest token's action condition."""
    predictor = host.predictor
    transformer = predictor.transformer
    state = transformer.input_proj(predictor.dropout(window + predictor.pos_embedding[:, :window.size(1)]))
    condition = transformer.cond_proj(controls)
    for block, offset in zip(transformer.layers, offsets, strict=True):
        altered = torch.cat((condition[:, :-1], condition[:, -1:] + offset), dim=1)
        state = block(state, altered)
    output = transformer.output_proj(transformer.norm(state))
    return host.pred_proj(output.flatten(0, 1)).unflatten(0, window.shape[:2])



class NativePrototypes(nn.Module):
    """Fixed zero plus K-1 learnable native condition offsets.

    Offsets have native projected-condition units.  The fixed 0.01 standard
    deviation only breaks initialization symmetry; these checks measure its
    finite responses and gradients without fitting or choosing a useful scale.
    Supplying the frozen raw-zero ``null_control`` anchors each offset's effect
    to the latest-action contrast. Omitting it preserves the archived path.
    """

    def __init__(self, host, k=4, *, null_control=None):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k < 2:
            raise ValueError("At least two integer-valued prototypes are required")
        self.host = host.eval().requires_grad_(False)
        layers = host.predictor.transformer.layers
        dimension = layers[0].adaLN_modulation[-1].in_features
        reference = next(host.parameters())
        generator = torch.Generator().manual_seed(20261119)
        initial = .01 * torch.randn(k - 1, len(layers), dimension, generator=generator)
        self.offsets = nn.Parameter(initial.to(reference))
        if null_control is not None:
            null_control = torch.as_tensor(null_control, device=reference.device,
                                           dtype=reference.dtype).detach().clone()
            if null_control.shape != (host.action_encoder.embed[-1].out_features,) \
                    or not torch.isfinite(null_control).all():
                raise ValueError("null_control must be the finite native raw-zero embedding [A]")
        self.register_buffer("null_control", null_control)

    def train(self, mode=True):
        super().train(mode)
        self.host.eval()
        return self

    @property
    def conditions(self):
        """The nominal row is reconstructed as exact zeros, never learned."""
        return torch.cat((torch.zeros_like(self.offsets[:1]), self.offsets), dim=0)

    def _step(self, windows, controls):
        if any(module.training for module in self.host.modules()):
            raise ValueError("Native PRBD requires an eval-mode frozen host")
        batch, count, length, dimension = windows.shape
        if count != self.offsets.size(0) + 1 or controls.shape[:3] != (batch, count, length):
            raise ValueError("Each prototype requires its own aligned native window and controls")
        nominal = self.host.predict(windows[:, 0], controls[:, 0])[:, -1]
        other_windows = windows[:, 1:].reshape(-1, length, dimension)
        other_controls = controls[:, 1:].reshape(-1, length, controls.size(-1))
        offsets = self.offsets[None].expand(batch, -1, -1, -1).flatten(0, 1)
        offsets = offsets.transpose(0, 1)[:, :, None]
        conditioned = native_conditioned(self.host, other_windows, other_controls, offsets)[:, -1]
        conditioned = conditioned.reshape(batch, count - 1, dimension)
        if self.null_control is not None:
            # The reference changes only the latest command, retaining each
            # branch's own state history and every historical action.
            reference_controls = torch.cat((other_controls[:, :-1],
                self.null_control.expand(len(other_controls), 1, -1)), dim=1)
            zero_offsets = torch.zeros_like(offsets)
            action_zero = native_conditioned(self.host, other_windows, other_controls,
                                             zero_offsets)[:, -1]
            reference_response = native_conditioned(self.host, other_windows, reference_controls,
                                                    offsets)[:, -1]
            reference_zero = native_conditioned(self.host, other_windows, reference_controls,
                                                zero_offsets)[:, -1]
            delta_action = conditioned - action_zero.reshape(batch, count - 1, dimension)
            delta_reference = (reference_response - reference_zero).reshape(batch, count - 1, dimension)
            # Match the nominal call's [B,L,D] shape for every frozen branch.
            # Combining corrections before adding F0 also avoids FP32 drift
            # when the latest action is exactly its raw-zero reference.
            base = torch.stack([self.host.predict(windows[:, k], controls[:, k])[:, -1]
                                for k in range(1, count)], dim=1)
            conditioned = base + (delta_action - delta_reference)
        return torch.cat((nominal[:, None], conditioned), dim=1)

    def forward(self, window, controls):
        """Issue all K next-token responses from one common actual window."""
        count = self.offsets.size(0) + 1
        return self._step(window[:, None].expand(-1, count, -1, -1),
                          controls[:, None].expand(-1, count, -1, -1))

    def rollout(self, window, controls):
        """Free-roll each prototype's own window; return [batch, K, H, D]."""
        count, length = self.offsets.size(0) + 1, window.size(1)
        horizon = controls.size(1) - length + 1
        if horizon < 1:
            raise ValueError("Controls must include the past L-1 and at least one future action")
        windows = window[:, None].expand(-1, count, -1, -1)
        forecast = []
        for step in range(horizon):
            command = controls[:, None, step:step + length].expand(-1, count, -1, -1)
            predicted = self._step(windows, command)
            forecast.append(predicted)
            windows = torch.cat((windows[:, :, 1:], predicted[:, :, None]), dim=2)
        return torch.stack(forecast, dim=2)
