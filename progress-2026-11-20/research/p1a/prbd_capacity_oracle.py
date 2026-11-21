"""Phase 1 supervised capacity attribution on the frozen G3 single trajectories.

This is an offline truth-conditioned diagnostic. It is not PRBD training and
does not change filtering, observation noise, or any deployment policy.
"""

import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from .prbd_consistency_data import SPLITS
from .prbd_native import NativePrototypes, native_conditioned
from .prbd_tworoom_speed_fit import load_host
from .prbd_paths import ROOT, output_path


OUT = ROOT / "runs/prbd_capacity_phase1_20261121"
TRAIN = ROOT / "runs/prbd_consistency_20261120/train.pt"
CONDITION_INDICES = {"slow_2.5": 1, "fast_7.5": 2}


class CapacityPrototypes(NativePrototypes):
    """A: original 3x6x192 offsets. B: extend that same native condition path.

    In B the latest native action condition enters a rank-four bilinear map
    with the finite condition index. The last two native blocks are jointly
    trainable and shared across conditions. The response is anchored by
    F0(W,a) + G(W,a;k) - G(W,a;0); the explicit nominal row is exactly F0.
    """

    def __init__(self, host, arm="A", rank=4):
        super().__init__(host, k=4, null_control=None)
        if arm not in ("A", "B"):
            raise ValueError("Capacity arm must be A or B")
        self.arm, self.rank = arm, rank
        if self.offsets.shape != (3, 6, 192):
            raise ValueError("Phase 1 A requires exactly 3456 original offsets")
        if arm == "B":
            self.last_blocks = copy.deepcopy(nn.ModuleList(list(
                host.predictor.transformer.layers[-2:])))
            self.last_blocks.requires_grad_(True).eval()
            reference = self.offsets
            generator = torch.Generator().manual_seed(20261121)
            down = torch.randn(192, rank, generator=generator) / np.sqrt(192)
            self.action_down = nn.Parameter(down.to(reference))
            self.action_up = nn.Parameter(reference.new_zeros(3, 6, rank, 192))

    def train(self, mode=True):
        super().train(mode)
        # Dropout is part of the released blocks; both G evaluations must be
        # deterministic to preserve the zero-condition subtraction exactly.
        if self.arm == "B":
            self.last_blocks.eval()
        return self

    def _g(self, window, controls, condition):
        predictor = self.host.predictor
        transformer = predictor.transformer
        state = transformer.input_proj(predictor.dropout(
            window + predictor.pos_embedding[:, :window.size(1)]))
        native = transformer.cond_proj(controls)
        if condition is None:
            delta = native.new_zeros(len(window), 6, 192)
        else:
            latest = native[:, -1] @ self.action_down
            delta = self.offsets[condition] + torch.einsum(
                "br,blrd->bld", latest, self.action_up[condition])
        for index, original in enumerate(transformer.layers):
            block = original if index < 4 else self.last_blocks[index - 4]
            altered = torch.cat((native[:, :-1],
                                 native[:, -1:] + delta[:, index, None]), dim=1)
            state = block(state, altered)
        output = transformer.output_proj(transformer.norm(state))
        return self.host.pred_proj(output.flatten(0, 1)).unflatten(0, window.shape[:2])[:, -1]

    def selected_step(self, window, controls, branch):
        """One response per row; branch indices are explicit offline labels."""
        if branch.shape != (len(window),) or not ((branch >= 1) & (branch <= 3)).all():
            raise ValueError("Selected training branches must be 1, 2, or pooled 3")
        index = branch - 1
        if self.arm == "A":
            offsets = self.offsets[index].transpose(0, 1)[:, :, None]
            return native_conditioned(self.host, window, controls, offsets)[:, -1]
        # F0 is frozen but retains input derivatives for free rollout.
        base = self.host.predict(window, controls)[:, -1]
        return base + (self._g(window, controls, index) - self._g(window, controls, None))

    def _step(self, windows, controls):
        if self.arm == "A":
            return super()._step(windows, controls)
        batch, count, length, dimension = windows.shape
        if count != 4 or controls.shape[:3] != (batch, count, length):
            raise ValueError("Each prototype requires its own native window")
        nominal = self.host.predict(windows[:, 0], controls[:, 0])[:, -1]
        other = self.selected_step(windows[:, 1:].reshape(-1, length, dimension),
                                  controls[:, 1:].reshape(-1, length, dimension),
                                  torch.arange(1, 4, device=windows.device).repeat(batch))
        return torch.cat((nominal[:, None], other.reshape(batch, 3, dimension)), dim=1)

    def selected_rollout(self, window, controls, branch):
        forecast = []
        for step in range(controls.size(1) - window.size(1) + 1):
            predicted = self.selected_step(window, controls[:, step:step + 3], branch)
            forecast.append(predicted)
            window = torch.cat((window[:, 1:], predicted[:, None]), dim=1)
        return torch.stack(forecast, dim=1)


def learned_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            if not name.startswith("host.")}


def load_oracle_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CapacityPrototypes(load_host(device), checkpoint["arm"], checkpoint["rank"]).eval()
    state = model.state_dict()
    for name, value in checkpoint["learned_state"].items():
        if name not in state or state[name].shape != value.shape:
            raise ValueError(f"Invalid capacity checkpoint entry {name}")
        state[name] = value.to(state[name])
    model.load_state_dict(state, strict=True)
    return model


def training_roots(path=TRAIN):
    """Use exactly the 62 usable G3 episodes; never read null_control."""
    batch = torch.load(path, map_location="cpu", weights_only=True)
    start, count = SPLITS["train"]
    speeds = np.random.default_rng(20261120).permutation(np.tile([2.5, 7.5], count // 2))
    # Truth is read only for this explicitly supervised capacity diagnostic.
    # Check the reconstructed collection assignment against its stored record.
    with np.load(TRAIN.parent / "diag_train.npz", allow_pickle=False) as physical:
        if (not np.array_equal(physical["seed"], batch["seed"].numpy())
                or not np.array_equal(physical["speed"], speeds)):
            raise ValueError("Stored physical speeds disagree with collection-source assignment")
    windows, controls, targets, branches, metadata = [], [], [], [], []
    usable, skipped = [], []
    for seed, z, a, length in zip(batch["seed"], batch["z"], batch["a"], batch["lengths"], strict=True):
        seed, length = int(seed), int(length)
        index = seed - start
        if index not in range(count):
            raise ValueError("A training seed does not belong to the frozen G3 split")
        if length < 3:
            skipped.append(seed)
            continue
        usable.append(seed)
        branch = 1 if speeds[index] == 2.5 else 2
        for root in range(2, length - 3):
            windows.append(z[root - 2:root + 1])
            controls.append(a[root - 2:root + 4])
            targets.append(z[root + 1:root + 5])
            branches.append(branch)
            metadata.append({"seed": seed, "root": root, "physical_speed": float(speeds[index]),
                             "oracle_branch": branch})
    if len(usable) != 62 or len(skipped) != 2:
        raise ValueError("Phase 1 requires the existing 62 usable trajectories")
    return (torch.stack(windows), torch.stack(controls), torch.stack(targets),
            torch.tensor(branches)), {"training_seeds": usable, "skipped_seeds": skipped,
            "roots": metadata, "condition_indices": CONDITION_INDICES,
            "speed_assignment_source": "prbd_consistency_data.collect: RNG20261120 permutation of tile([2.5,7.5],32)"}


def batch_loss(model, data, indices, details=False):
    window, controls, target, branch = [value[indices] for value in data]
    # The third nonnominal branch sees every trajectory's same true targets.
    # It gets the same total loss weight as the two labeled branches together.
    window, controls = window.repeat(2, 1, 1), controls.repeat(2, 1, 1)
    branch = torch.cat((branch, torch.full_like(branch, 3)))
    prediction = model.selected_rollout(window, controls, branch)
    target = target.repeat(2, 1, 1)
    h1 = (prediction[:, 0] - target[:, 0]).square().mean(dim=-1)
    h4 = (prediction[:, 3] - target[:, 3]).square().mean(dim=-1)
    loss = .5 * (h1.mean() + h4.mean())
    if details:
        count = len(indices)
        return loss, {"H1": float(h1.mean()), "H4": float(h4.mean()),
                      "oracle_H1": float(h1[:count].mean()), "oracle_H4": float(h4[:count].mean()),
                      "pooled_H1": float(h1[count:].mean()), "pooled_H4": float(h4[count:].mean())}
    return loss


@torch.no_grad()
def objective(model, data, batch_size=16):
    total, metrics = 0., {}
    for start in range(0, len(data[0]), batch_size):
        index = torch.arange(start, min(start + batch_size, len(data[0])), device=data[0].device)
        loss, pieces = batch_loss(model, data, index, True)
        total += float(loss) * len(index)
        for name, value in pieces.items():
            metrics[name] = metrics.get(name, 0.) + value * len(index)
    size = len(data[0])
    return {"objective": total / size, **{name: value / size for name, value in metrics.items()}}


def write_json(path, value):
    output_path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def save_checkpoint(model, directory, update, meta, name):
    torch.save({"arm": model.arm, "rank": model.rank, "learned_state": learned_state(model),
                "update": update, "training_seeds": meta["training_seeds"],
                "condition_indices": CONDITION_INDICES, "null_control": None,
                "third_branch": "Equal-weight pooled supervision from both physical speeds",
                "training_objective": "0.5*(H1 endpoint MSE + H4 endpoint MSE), oracle/pooled each weight0.5"},
               directory / name)


@torch.no_grad()
def export_training_trajectories(arm, output=OUT):
    """Retain the same chronological training roots at all fixed milestones."""
    directory = output / arm
    data, meta = training_roots()
    for name in ("initial", "update_000062", "final"):
        path = directory / f"{name}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        model = load_oracle_checkpoint(path, device="cpu")
        predicted = model.rollout(data[0][:16], data[1][:16])
        torch.save({"paths": predicted, "targets": data[2][:16],
                    "windows": data[0][:16], "controls": data[1][:16],
                    "oracle_branches": data[3][:16], "roots": meta["roots"][:16],
                    "condition_indices": CONDITION_INDICES},
                   output_path(directory / f"training_trajectories_{name}.pt"))


def structural_checks(model, data):
    window, controls, _, branch = [value[:2] for value in data]
    with torch.no_grad():
        actual = model.rollout(window, controls)
        native, current = [], window
        for step in range(4):
            value = model.host.predict(current, controls[:, step:step + 3])[:, -1]
            native.append(value)
            current = torch.cat((current[:, 1:], value[:, None]), dim=1)
        expected = torch.stack(native, dim=1)
        if not torch.equal(actual[:, 0], expected):
            raise AssertionError("Nominal H1-H4 must equal the frozen host bitwise")
        selected = model.selected_rollout(window, controls, branch)
        # Batch-dependent BLAS roundoff can affect different batch shapes.
        torch.testing.assert_close(selected, actual[torch.arange(2), branch], rtol=2e-5, atol=3e-6)
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    gradient = torch.autograd.grad(model.selected_rollout(window, controls, branch).square().mean(),
                                   [parameter for _, parameter in named], allow_unused=True)
    if not all(value is None or torch.isfinite(value).all() for value in gradient):
        raise AssertionError("Nonfinite native gradient")
    return {"nominal_H1_H4_bitwise_equal": True, "selected_branch_API_agrees": True,
            "finite_native_gradients": True, "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "native_offset_parameters": model.offsets.numel(), "null_control": None,
            "gradient_parameter_names": [name for name, _ in named],
            "gradient_norms": [None if value is None else float(value.norm()) for value in gradient]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B"), required=True)
    parser.add_argument("--output", type=Path, default=output_path(OUT / "protocol.json").parent)
    parser.add_argument("--export-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261121)
    if args.export_only:
        export_training_trajectories(args.arm, args.output)
        print(f"Exported {args.arm} fixed training trajectories on CPU", flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("Authorized capacity training requires the available CUDA device")
    directory = output_path(args.output / args.arm / "protocol.json").parent
    directory.mkdir(parents=True, exist_ok=True)
    cpu_data, meta = training_roots()
    data = tuple(value.to("cuda") for value in cpu_data)
    model = CapacityPrototypes(load_host("cuda"), args.arm).eval()
    protocol = {**meta, "arm": args.arm, "device": "cuda", "batch_size": 16,
        "optimizer": "Adam", "learning_rate": .0003, "schedule": {"2048": .00003, "3072": .000003},
        "max_updates": 4096, "minimum_updates": 3072, "objective_interval": 128,
        "convergence": "After >=3072 updates: relative full-training-objective decrease over previous 512 updates <=0.001, nonnegative; no development access",
        "selection": "Final training state only; no development selection",
        "normalization": "Equal roots; H1 and H4 endpoint latent MSE each weight0.5; selected true branch and pooled branch3 each weight0.5",
        "root_definition": "Every actual root with full H4 suffix, t=2..length-4",
        "bilinear_rank": 4 if args.arm == "B" else None,
        "bilinear_stress_check": "Finite gradient through action_up and free H4; zero branch remains bitwise F0",
        "data_source": str(TRAIN), "trajectory_count": 62, "root_count": len(cpu_data[0]),
        "null_control_used": False, "development_used": False}
    write_json(directory / "protocol.json", protocol)
    checks = structural_checks(model, data)
    write_json(directory / "initial_checks.json", checks)
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.0003)
    generator = torch.Generator().manual_seed(20261121)
    started = perf_counter()
    history = [{"update": 0, "seconds": 0., **objective(model, data)}]
    write_json(directory / "objective_history.json", history)
    print(json.dumps({"arm": args.arm, "initial": history[0], "checks": checks}), flush=True)
    save_checkpoint(model, directory, 0, meta, "initial.pt")
    converged = False
    with (directory / "updates.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, 4097):
            if update in (2048, 3072):
                for group in optimizer.param_groups:
                    group["lr"] *= .1
            indices = torch.randperm(len(data[0]), generator=generator)[:16].to("cuda")
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(model, data, indices)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite oracle training loss at update {update}")
            loss.backward()
            if not all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                       if parameter.grad is not None):
                raise FloatingPointError(f"Nonfinite oracle gradient at update {update}")
            optimizer.step()
            stream.write(json.dumps({"update": update, "objective_batch": float(loss),
                                     "seconds": perf_counter() - started}) + "\n")
            if update % 16 == 0:
                stream.flush()
            if update == 62:
                save_checkpoint(model, directory, update, meta, "update_000062.pt")
            if update % 128 == 0 or update == 62:
                row = {"update": update, "seconds": perf_counter() - started, **objective(model, data)}
                history.append(row)
                write_json(directory / "objective_history.json", history)
                save_checkpoint(model, directory, update, meta, "latest.pt")
                print(json.dumps({"arm": args.arm, **row}), flush=True)
                earlier = next((value for value in history if value["update"] == update - 512), None)
                if update >= 3072 and earlier is not None:
                    decrease = (earlier["objective"] - row["objective"]) / earlier["objective"]
                    if 0 <= decrease <= .001:
                        converged = True
                        break
    save_checkpoint(model, directory, update, meta, "final.pt")
    write_json(directory / "final_checks.json", structural_checks(model, data))
    write_json(directory / "result.json", {"arm": args.arm, "updates": update,
        "converged": converged, "termination": "objective plateau" if converged else "hard budget exhausted",
        "seconds": perf_counter() - started, "initial": history[0], "final": history[-1],
        "checkpoint": str(directory / "final.pt")})
    # Fixed train roots retain actual targets and all four free trajectories.
    with torch.no_grad():
        preview = model.rollout(data[0][:16], data[1][:16]).cpu()
    torch.save({"paths": preview, "targets": cpu_data[2][:16], "roots": meta["roots"][:16]},
               directory / "training_trajectories.pt")
    print(json.dumps({"arm": args.arm, "done": True, "converged": converged, "updates": update}), flush=True)


if __name__ == "__main__":
    main()
