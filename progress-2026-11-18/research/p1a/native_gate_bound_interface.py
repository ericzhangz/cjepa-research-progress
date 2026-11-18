"""Check the final native MLP gate and the response bound at that interface."""

import argparse
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from history_match import load_host
from query_feedback_learning import write_csv, write_json


CANDIDATES = ("stop", "constant_0", "constant_2", "constant_4")
ROOT_COUNT = 8
ATOL = 2e-5
MIN_SLOPE = 1e-4


def verify_context(model, z, actions):
    transformer = model.predictor.predictor.transformer
    block = transformer.layers[-1]
    captured = {}

    def collect(name, value):
        if name in captured:
            raise RuntimeError(f"Repeated final block capture for {name}")
        captured[name] = value.clone()

    with ExitStack() as hooks:
        hooks.callback(block.adaLN_modulation.register_forward_hook(
            lambda _, args, out: collect("gate", out.chunk(6, dim=-1)[-1])).remove)
        hooks.callback(block.norm2.register_forward_pre_hook(
            lambda _, args: collect("post_attention", args[0])).remove)
        hooks.callback(block.mlp.register_forward_hook(
            lambda _, args, out: collect("mlp", out)).remove)
        hooks.callback(block.register_forward_hook(
            lambda _, args, out: collect("hidden", out)).remove)
        native = model.predict(z, actions)

    def tail(h):
        shape = h.shape
        projected = transformer.output_proj(transformer.norm(h))
        return model.predictor.projector(projected.flatten(0, 1)).reshape(*shape[:2], -1)

    hidden, gate, mlp = (captured[name] for name in ("hidden", "gate", "mlp"))
    reconstructed = captured["post_attention"] + gate * mlp
    reconstructed_latent = tail(reconstructed)
    torch.testing.assert_close(reconstructed, hidden, atol=ATOL, rtol=0)
    torch.testing.assert_close(reconstructed_latent, native, atol=ATOL, rtol=0)

    covector = torch.zeros_like(hidden[0, -1])
    covector[:2] = covector.new_tensor([1, -1]) / 2 ** 0.5
    direction = torch.zeros_like(covector)
    direction[:2] = 1
    s = hidden[:, -1] @ covector
    by_root = s.reshape(ROOT_COUNT, len(CANDIDATES))
    bound = (by_root.amin(dim=1) + by_root.amax(dim=1)) / 2
    bound = bound.repeat_interleave(len(CANDIDATES))
    triggered = s > bound
    native_direction = mlp[:, -1] * direction
    slope = native_direction @ covector
    valid = slope.abs() >= MIN_SLOPE
    usable = triggered & valid
    gate_delta = torch.zeros_like(s)
    gate_delta[usable] = -(s[usable] - bound[usable]) / slope[usable]

    def update_gate(_, args, out):
        replaced = out.clone()
        last_gate = replaced[:, -1].chunk(6, dim=-1)[-1]
        last_gate[usable] += gate_delta[usable, None] * direction
        return replaced

    with ExitStack() as hooks:
        hooks.callback(block.adaLN_modulation.register_forward_hook(update_gate).remove)
        hooks.callback(block.register_forward_hook(
            lambda _, args, out: collect("hook_hidden", out)).remove)
        hooked = model.predict(z, actions)

    expected = hidden.clone()
    expected[usable, -1] += gate_delta[usable, None] * native_direction[usable]
    expected_latent = tail(expected)
    actual_s = captured["hook_hidden"][:, -1] @ covector
    torch.testing.assert_close(captured["hook_hidden"], expected, atol=ATOL, rtol=0)
    torch.testing.assert_close(hooked, expected_latent, atol=ATOL, rtol=0)
    torch.testing.assert_close(actual_s[usable], bound[usable], atol=ATOL, rtol=0)
    torch.testing.assert_close(hooked[~triggered], native[~triggered], atol=0, rtol=0)
    torch.testing.assert_close(hooked[:, :-1], native[:, :-1], atol=0, rtol=0)
    errors = {
        "native_hidden_max_abs_error": (reconstructed - hidden).abs().flatten(1).amax(dim=-1),
        "native_latent_max_abs_error": (reconstructed_latent - native).abs().flatten(1).amax(dim=-1),
        "intervened_hidden_max_abs_error": (captured["hook_hidden"] - expected).abs().flatten(1).amax(dim=-1),
        "intervened_latent_max_abs_error": (hooked - expected_latent).abs().flatten(1).amax(dim=-1),
        "last_latent_change_norm": (hooked[:, -1] - native[:, -1]).norm(dim=-1),
    }
    return {name: value.cpu().numpy() for name, value in {
        **captured, **errors, "native": native, "hooked": hooked, "expected_hidden": expected,
        "expected_latent": expected_latent, "s": s, "bound": bound, "s_hook": actual_s,
        "slope": slope, "gate_delta": gate_delta, "triggered": triggered, "valid": valid,
    }.items()}


@torch.no_grad()
def run(args):
    host = load_host(args)
    model = host.model
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Frozen native eval mode is required")
    with np.load(args.branches, allow_pickle=False) as bank:
        names = bank["candidate_name"].tolist()
        selected = [names.index(name) for name in CANDIDATES]
        roots = bank["root_id"][:ROOT_COUNT]
        visible = bank["z_prefix"][:ROOT_COUNT].repeat(len(CANDIDATES), axis=0)
        raw_prefix = np.repeat(bank["prefix_raw_actions"][:ROOT_COUNT, None], len(CANDIDATES), axis=1)
        raw_next = bank["raw_actions"][:ROOT_COUNT, selected, :1]
    raw = np.concatenate((raw_prefix, raw_next), axis=2).reshape(len(visible), 3, 5, 2)
    z = torch.as_tensor(visible, device=host.device)
    normalized = torch.as_tensor(host.normalize(raw), device=host.device)
    actions = model.action_embed(model.action_encoder(normalized))
    rows, arrays, summary = [], {"root_id": roots, "candidate_name": np.array(CANDIDATES), "raw_actions": raw}, {}
    scalar_names = ("s", "bound", "s_hook", "slope", "gate_delta", "triggered", "valid",
                    "native_hidden_max_abs_error", "native_latent_max_abs_error",
                    "intervened_hidden_max_abs_error", "intervened_latent_max_abs_error", "last_latent_change_norm")
    for tokens in (1, 3):
        trace = verify_context(model, z[:, -tokens:], actions[:, -tokens:])
        arrays.update({f"T{tokens}_{name}": values for name, values in trace.items()})
        summary[str(tokens)] = {
            "requests": len(visible), "triggered": int(trace["triggered"].sum()),
            "triggered_invalid": int((trace["triggered"] & ~trace["valid"]).sum()),
            "min_abs_slope": float(abs(trace["slope"]).min()),
            **{name: float(trace[name].max()) for name in scalar_names if name.endswith("_error")},
        }
        for i, root in enumerate(roots):
            for j, candidate in enumerate(CANDIDATES):
                index = i * len(CANDIDATES) + j
                rows.append({"tokens": tokens, "root_id": int(root), "candidate": candidate,
                             **{name: trace[name][index].item() for name in scalar_names}})

    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "requests.csv", rows)
    write_json(args.output / "summary.json", summary)
    np.savez_compressed(args.output / "intermediates.npz", **arrays)
    print(f"Saved native final-gate interface results to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("research/phase00c/probe_heads.pt"))
    parser.add_argument("--branches", type=Path, default=Path("runs/p1a_v1/assessment/branches.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/native_gate_bound_20261112/interface"))
    args = parser.parse_args()
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        run(args)


if __name__ == "__main__":
    main()
