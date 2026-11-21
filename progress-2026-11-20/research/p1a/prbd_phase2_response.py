"""Capacity repair on the existing PRBD native condition path.

The nominal branch literally calls the frozen host. Other branches share one
trainable copy of the released final block, final normalization and pred_proj,
and retain the original per-prototype offsets in all native blocks. Module
evaluation mode is fixed: dropout and batch-normalization statistics do not
introduce an additional learned or stochastic prediction path.
"""

from copy import deepcopy

import torch

from .prbd_native import NativePrototypes
from .prbd_tworoom_speed_fit import load_host


RESPONSE_SOURCE = {
    "kind": "native_shared_final_block_norm_pred_proj_copy",
    "host": "official TwoRoom LeWM runs/prbd_official_protocols_20261119/tworoom",
    "initialization": "independent copies of frozen released weights; offsets seed20261119 std0.01",
    "nominal": "literal independent frozen host forward; bypasses all learned state",
    "null_control": None,
}


class CapacityNativePrototypes(NativePrototypes):
    """Existing null-free operator extended with a shared trainable suffix."""

    def __init__(self, host, k=4):
        super().__init__(host, k=k, null_control=None)
        transformer = self.host.predictor.transformer
        self.final_block = deepcopy(transformer.layers[-1]).requires_grad_(True)
        self.final_norm = deepcopy(transformer.norm).requires_grad_(True)
        self.pred_proj = deepcopy(self.host.pred_proj).requires_grad_(True)
        self.eval()

    @property
    def native_blocks(self):
        return (*self.host.predictor.transformer.layers[:-1], self.final_block)

    def response_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def train(self, mode=True):
        # Training means gradient optimization only; released inference semantics
        # (including BatchNorm running moments) remain fixed in every branch.
        super().train(False)
        return self

    def forward_tokens(self, window, controls, offsets):
        """Native full-token output for explicit [layer,batch,1,condition] offsets."""
        predictor = self.host.predictor
        transformer = predictor.transformer
        state = transformer.input_proj(predictor.dropout(
            window + predictor.pos_embedding[:, :window.size(1)]))
        condition = transformer.cond_proj(controls)
        for block, offset in zip(self.native_blocks, offsets, strict=True):
            altered = torch.cat((condition[:, :-1], condition[:, -1:] + offset), dim=1)
            state = block(state, altered)
        output = transformer.output_proj(self.final_norm(state))
        return self.pred_proj(output.flatten(0, 1)).unflatten(0, window.shape[:2])

    def selected_step(self, windows, controls, branch):
        """Oracle-only nonnominal branch selector; branch indices are 1..K-1."""
        if branch.shape != (len(windows),) or branch.dtype != torch.long:
            raise ValueError("Expected one integer branch index for every native window")
        if not bool(((branch >= 1) & (branch <= len(self.offsets))).all()):
            raise ValueError("Selected native suffix requires a nonnominal branch")
        offsets = self.offsets[branch - 1].transpose(0, 1)[:, :, None]
        return self.forward_tokens(windows, controls, offsets)[:, -1]

    def _step(self, windows, controls):
        if any(module.training for module in self.modules()):
            raise ValueError("Capacity PRBD requires fixed eval-mode native modules")
        batch, count, length, dimension = windows.shape
        if count != len(self.offsets) + 1 or controls.shape[:3] != (batch, count, length):
            raise ValueError("Each prototype requires its own aligned native inputs")
        nominal = self.host.predict(windows[:, 0], controls[:, 0])[:, -1]
        other_windows = windows[:, 1:].reshape(-1, length, dimension)
        other_controls = controls[:, 1:].reshape(-1, length, controls.size(-1))
        offsets = self.offsets[None].expand(batch, -1, -1, -1).flatten(0, 1)
        offsets = offsets.transpose(0, 1)[:, :, None]
        conditioned = self.forward_tokens(other_windows, other_controls, offsets)[:, -1]
        return torch.cat((nominal[:, None], conditioned.reshape(batch, count - 1, dimension)), 1)


def make_response_model(device="cpu", k=4):
    return CapacityNativePrototypes(load_host(device), k=k).eval()


def response_state(model):
    """Independent learned tensors only; never serializes/reloads the host."""
    names = ("offsets", "final_block.", "final_norm.", "pred_proj.")
    learned = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()
               if key == names[0] or key.startswith(names[1:])}
    return {"source": dict(RESPONSE_SOURCE), "k": len(model.offsets) + 1,
            "learned_state": learned}


def load_response_state(model, state):
    if state["source"] != RESPONSE_SOURCE or state["k"] != len(model.offsets) + 1:
        raise ValueError("Native response checkpoint source or prototype count differs")
    expected = response_state(model)["learned_state"]
    if set(state["learned_state"]) != set(expected):
        raise ValueError("Native response checkpoint learned tensor names differ")
    for key, value in state["learned_state"].items():
        if value.shape != expected[key].shape or not torch.isfinite(value).all():
            raise ValueError(f"Invalid native response tensor: {key}")
    incompatible = model.load_state_dict(state["learned_state"], strict=False)
    if incompatible.unexpected_keys or any(not key.startswith("host.") for key in incompatible.missing_keys):
        raise ValueError("Native response checkpoint has missing learned state")
    return model.eval()
