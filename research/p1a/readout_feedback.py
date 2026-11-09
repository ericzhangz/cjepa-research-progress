"""Small FourRoom study of recurrent feedback from the native readout kernel."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from p1a_reference import latent_rollout


class ReadoutFeedback(nn.Module):
    """One recurrent edge through the frozen projector's GELU."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        projector = model.predictor.projector
        self.input, self.norm, self.activation, self.readout = projector.net
        if not isinstance(projector.final_norm, nn.Identity) or not isinstance(self.readout, nn.Linear):
            raise ValueError("Readout feedback requires an affine final readout.")
        weight = self.readout.weight.detach().double()
        if int(torch.linalg.matrix_rank(weight)) != weight.shape[0]:
            raise ValueError("The observation overwrite requires full row rank.")
        basis = torch.linalg.qr(weight.T, mode="reduced").Q
        self.register_buffer("row_basis", basis.to(self.readout.weight))

    def kernel(self, activation):
        basis = self.row_basis
        return activation - (activation @ basis) @ basis.T

    def preactivation(self, z, action_embedding):
        core = self.model.predictor.predictor(z[:, None], action_embedding[:, None])[:, 0]
        return self.norm(self.input(core))

    def step(self, z, action_embedding, carry, gain, mode, basis=None):
        q = self.preactivation(z, action_embedding)
        if mode == "current":
            carry = self.kernel(self.activation(q))
        feedback = gain * carry
        if basis is not None:
            feedback = feedback @ basis.T
        h = self.activation(q + feedback)
        if basis is not None:
            next_carry = h @ basis
        else:
            next_carry = self.kernel(h) if mode == "kernel" else h
        return self.readout(h), next_carry

    def sequence(self, initial, observed, actions, gain, mode, last_observed, intervention="keep", basis=None):
        z = initial
        width = self.readout.in_features if basis is None else basis.shape[1]
        carry = z.new_zeros(len(z), width)
        outputs, states = [], []
        if basis is not None:
            strengths = gain.expand(*actions.shape[:2], -1)
        else:
            strengths = (gain(actions).tanh() if gain is not None else actions.new_zeros(*actions.shape[:2], 1))
        for i in range(actions.shape[1]):
            if i <= last_observed:
                z = observed[:, i]
            if i == last_observed and intervention == "erase":
                carry = torch.zeros_like(carry)
            if i == last_observed and intervention == "swap":
                carry = carry.roll(len(carry) // 2, dims=0)
            z, carry = self.step(z, actions[:, i], carry, strengths[:, i], mode, basis=basis)
            outputs.append(z)
            states.append(carry)
        return torch.stack(outputs, dim=1), torch.stack(states, dim=1), strengths


class LowRankFeedback(nn.Module):
    """The same orthogonal directions write and feed back a compact state."""

    def __init__(self, operator, rank, constrained=True):
        super().__init__()
        width = operator.readout.in_features
        if rank < 1 or rank > width - (operator.row_basis.shape[1] if constrained else 0):
            raise ValueError("Rank exceeds the available feedback subspace.")
        weight = operator.readout.weight
        self.directions = nn.Parameter(weight.new_empty(width, rank).normal_())
        self.gain = nn.Parameter(weight.new_zeros(rank))
        self.register_buffer("row_basis", operator.row_basis if constrained else None)

    def forward(self):
        directions = self.directions
        if self.row_basis is not None:
            directions = directions - self.row_basis @ (self.row_basis.T @ directions)
        basis = torch.linalg.qr(directions, mode="reduced").Q
        return basis, self.gain


def analytic_example():
    # Both histories read as z=0. The action chooses unequal preactivations.
    weight = torch.tensor([1.0, 1.0], dtype=torch.float64)
    histories = torch.tensor([[0.5, -0.5], [-0.5, 0.5]], dtype=torch.float64)
    preactivation = torch.tensor([1.0, -1.0], dtype=torch.float64)
    nonlinear = nn.functional.gelu(preactivation + histories) @ weight
    affine = (preactivation + histories) @ weight
    opposite_action = nn.functional.gelu(-preactivation + histories) @ weight
    return {"current": (histories @ weight).tolist(), "affine_next": affine.tolist(),
            "gelu_next": nonlinear.tolist(), "opposite_action_next": opposite_action.tolist()}


def delayed_actuator():
    """A command writes an unseen actuator mode. A later pulse moves the object."""
    command = torch.tensor([-1., -1., 1., 1.], dtype=torch.float64)
    pulse = torch.tensor([-1., 1., -1., 1.], dtype=torch.float64)
    # x_0=x_1=0, v_1=command, x_2=0.2*pulse*v_1. Both actions are executed.
    observed = command.new_zeros(4, 2, 1)
    controls = torch.stack([command, pulse], dim=1)
    embedded = torch.stack([controls, -controls], dim=-1)
    target = (0.2 * pulse * command)[:, None]
    activation = nn.GELU()
    readout = nn.Linear(2, 1, dtype=torch.float64)
    with torch.no_grad():
        readout.weight.fill_(1)
        readout.bias.copy_(-activation(torch.tensor([1., -1.], dtype=torch.float64)).sum())
    projector = SimpleNamespace(net=(nn.Identity(), nn.Identity(), activation, readout.requires_grad_(False)),
                                final_norm=nn.Identity())
    model = SimpleNamespace(predictor=SimpleNamespace(projector=projector, predictor=lambda z, a: a))
    operator = ReadoutFeedback(model)
    gain = nn.Linear(2, 1, dtype=torch.float64)
    nn.init.zeros_(gain.weight)
    nn.init.zeros_(gain.bias)
    optim = torch.optim.Adam(gain.parameters(), lr=0.05)
    for _ in range(100):
        pred, _, _ = operator.sequence(observed[:, 0], observed, embedded, gain, "kernel", 1)
        optim.zero_grad(set_to_none=True)
        (pred[:, -1] - target).square().mean().backward()
        optim.step()
    with torch.no_grad():
        pred, carry, strengths = operator.sequence(observed[:, 0], observed, embedded, gain, "kernel", 1)
        erased, _, _ = operator.sequence(observed[:, 0], observed, embedded, gain, "kernel", 1, "erase")
    return {"command": command.tolist(), "pulse": pulse.tolist(), "target": target.flatten().tolist(),
            "anchor_prediction": pred[:, 0, 0].tolist(), "carry_at_anchor": carry[:, 0].tolist(),
            "prediction": pred[:, 1, 0].tolist(), "prediction_after_erase": erased[:, 1, 0].tolist(),
            "trained_gain": strengths[:, 1, 0].tolist(), "mse": float((pred[:, -1] - target).square().mean()),
            "memoryless_mse_lower_bound": 0.04}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--low-rank", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.out}")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(41)
    sys.path[:0] = [str(args.model_dir / "upstream/h_jepa"), str(args.model_dir / "deps")]
    model = torch.load(args.model_dir / "fourroom_lewm_object.ckpt", map_location="cpu", weights_only=False)
    model = model.get_level(1).cuda().eval().requires_grad_(False)
    operator = ReadoutFeedback(model)

    with np.load(args.data, allow_pickle=False) as data:
        roots, candidate_ids = data["root_id"], data["candidate_id"]
        times = data["true_absolute_step"][:5]
        if times.tolist() != [15, 25, 30, 45, 50] or data["prefix_absolute_step"].tolist() != [0, 5, 10]:
            raise ValueError("Expected development trajectories with the recorded 0/5/10/15 anchors.")
        prefix = np.repeat(data["z_prefix"][:, None], len(candidate_ids), axis=1)
        past = np.repeat(data["prefix_normalized_actions"][:, None], len(candidate_ids), axis=1)
        actual = data["z_true"][:, :, :5]
        raw = np.concatenate([past, data["normalized_actions"][:, :, :8]], axis=2)
    n_candidates = len(candidate_ids)
    observed = torch.as_tensor(np.concatenate([prefix, actual[:, :, :1]], axis=2), device="cuda").flatten(0, 1)
    actual = torch.as_tensor(actual, device="cuda").flatten(0, 1)
    actions = torch.as_tensor(raw, device="cuda").flatten(0, 1)
    with torch.no_grad():
        embeddings = model.action_embed(model.action_encoder(actions))
    target_steps = (times // 5 - 1).tolist()
    train_end = 12 * n_candidates
    held = slice(train_end, None)
    batches = torch.randint(train_end, (96, 32), device="cuda")
    summary = {"analytic": analytic_example(), "delayed_actuator": delayed_actuator(), "train_roots": roots[:12].tolist(),
               "held_roots": roots[12:].tolist(), "target_env_steps": times.tolist(),
               "context_size": 1, "optimization_steps": len(batches), "seed": 41,
               "checkpoint": str(args.model_dir / "fourroom_lewm_object.ckpt"),
               "activation_width": operator.readout.in_features, "readout_width": operator.readout.out_features}
    arrays = {"root_ids": roots[12:], "candidate_ids": candidate_ids, "target_times": times,
              "targets": actual[held].cpu().numpy(), "observed": observed[held].cpu().numpy(),
              "actions": actions[held].cpu().numpy()}

    def run(indices, gain, mode, anchor, intervention="keep", basis=None):
        obs = observed[indices]
        return operator.sequence(obs[:, 0], obs, embeddings[indices], gain, mode, anchor, intervention, basis)

    def record(label, gain, mode, anchor, intervention="keep", basis=None):
        with torch.no_grad():
            pred, carry, strengths = run(held, gain, mode, anchor, intervention, basis)
            error = (pred[:, target_steps] - actual[held]).square().mean(dim=-1)
            arrays[label + "_predictions"] = pred.cpu().numpy()
            arrays[label + "_carry"] = carry.cpu().numpy()
            arrays[label + "_gains"] = strengths.cpu().numpy()
            arrays[label + "_mse"] = error.cpu().numpy()
            summary[label] = {"mse": error.mean(0).tolist(),
                              "mean_by_root": error.reshape(-1, n_candidates, len(times)).mean(1).tolist()}
            if args.low_rank and anchor == 2:
                residual = (pred[:, target_steps] - actual[held]).reshape(-1, n_candidates, len(times), pred.shape[-1])
                centered = residual - residual.mean(1, keepdim=True)
                summary[label]["action_contrast_mse_half_pairwise"] = centered.square().mean((0, 1, 3)).tolist()
            if mode == "kernel":
                summary[label]["max_kernel_readout"] = float((carry @ operator.readout.weight.T).abs().max())
            return pred

    native = record("native", None, "full", 3)
    with torch.no_grad():
        obs = observed[held][:, 3:4]
        ref = latent_rollout(model.predict, model.action_embed, obs, actions[held][:, :0], actions[held][:, 3:], 1)
        torch.testing.assert_close(native[:, 3:], ref, rtol=2e-5, atol=2e-5)
        summary["native_max_delta"] = float((native[:, 3:] - ref).abs().max())
        zero_kernel, _, _ = run(held, None, "kernel", 3)
        torch.testing.assert_close(zero_kernel, native, rtol=0, atol=0)
        native_context3 = latent_rollout(model.predict, model.action_embed, observed[held][:, 1:4],
                                         actions[held][:, 1:3], actions[held][:, 3:], 3)
        error3 = (native_context3[:, np.asarray(target_steps[1:]) - 3] - actual[held][:, 1:]).square().mean(-1)
        summary["native_context3"] = {"target_env_steps": times[1:].tolist(), "mse": error3.mean(0).tolist(),
                                      "mean_by_root": error3.reshape(-1, n_candidates, len(times) - 1).mean(1).tolist()}
        arrays["native_context3_predictions"] = native_context3.cpu().numpy()
        arrays["native_context3_mse"] = error3.cpu().numpy()
    if args.low_rank:
        rank = 16
        summary.update({"rank": rank, "state_float32_bytes": rank * 4,
                        "basis_float32_bytes": rank * operator.readout.in_features * 4,
                        "feedback_macs_per_sample_step": 2 * rank * operator.readout.in_features})
        for label, constrained in (("low_rank_kernel", True), ("low_rank_unconstrained", False)):
            torch.manual_seed(41)
            feedback = LowRankFeedback(operator, rank, constrained)
            optimizer = torch.optim.Adam(feedback.parameters(), lr=0.001)
            loss_trace = []
            with torch.no_grad():
                basis, gain = feedback()
                zero, _, _ = run(held, gain, "low_rank", 3, basis=basis)
                torch.testing.assert_close(zero, native, rtol=0, atol=0)
            for indices in batches:
                basis, gain = feedback()
                pred, _, _ = run(indices, gain, "low_rank", 3, basis=basis)
                loss = (pred[:, target_steps] - actual[indices]).square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_trace.append(loss.detach())
            with torch.no_grad():
                basis, gain = feedback()
                arrays[label + "_basis"] = basis.cpu().numpy()
                arrays[label + "_training_mse"] = torch.stack(loss_trace).cpu().numpy()
                record(label, gain, "low_rank", 3, basis=basis)
                record(label + "_erase", gain, "low_rank", 3, "erase", basis)
                record(label + "_anchor10", gain, "low_rank", 2, basis=basis)
                summary[label].update({"max_basis_readout": float((operator.readout.weight @ basis).abs().max()),
                                       "max_orthogonality_residual": float((basis.T @ basis - torch.eye(rank, device=basis.device)).abs().max()),
                                       "parameters": sum(p.numel() for p in feedback.parameters())})
            print(label, summary[label]["mse"], flush=True)
        record("native_anchor10", None, "full", 2)
    for mode in (() if args.low_rank else ("kernel", "full", "current")):
        gain = nn.Linear(embeddings.shape[-1], 1, device="cuda")
        nn.init.zeros_(gain.weight)
        nn.init.zeros_(gain.bias)
        optimizer = torch.optim.Adam(gain.parameters(), lr=0.001)
        loss_trace = []
        for indices in batches:
            pred, _, _ = run(indices, gain, mode, 3)
            loss = (pred[:, target_steps] - actual[indices]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.detach())
        arrays[mode + "_training_mse"] = torch.stack(loss_trace).cpu().numpy()
        arrays[mode + "_gain_weight"] = gain.weight.detach().cpu().numpy()
        arrays[mode + "_gain_bias"] = gain.bias.detach().cpu().numpy()
        record(mode, gain, mode, 3)
        if mode == "kernel":
            record("kernel_erase", gain, mode, 3, "erase")
            record("kernel_swap", gain, mode, 3, "swap")
            record("kernel_anchor10", gain, mode, 2)
            record("native_anchor10", None, "full", 2)
        print(mode, summary[mode]["mse"], flush=True)
    np.savez_compressed(args.out / "trace.npz", **arrays)
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
