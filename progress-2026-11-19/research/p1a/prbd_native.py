"""Native LeWM condition path used by PRBD.

Extracted without the historical FourRoom-only execution/check driver.
The two retained definitions are unchanged from the research source.
"""

import torch
from torch import nn


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
    """

    def __init__(self, host, k=4):
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
