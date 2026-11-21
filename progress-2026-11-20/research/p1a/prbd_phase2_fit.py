"""One frozen-budget unlabelled Phase2 fit; no development or physical labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from .prbd import assimilate_continuous, continuous_quadrature, issue
from .prbd_native import NativePrototypes
from .prbd_paths import SPEED_DATA, output_path
from .prbd_phase2 import (BUDGET, MIN_DEV_COEFFICIENT, OUT, assimilate_step,
                         batch_objective, fixed_transition, load_episodes, response_density)
from .prbd_tworoom_speed_fit import load_host


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def tensor_numbers(pieces):
    return {key: float(value.detach()) if torch.is_tensor(value) else value
            for key, value in pieces.items() if key != "trace"}


@torch.no_grad()
def calibrate_numerical_deadzone(host, episodes, noise):
    """Unlabelled native FP32 reproducibility, measured in fixed L coordinates.

    Measure identical F0 inputs at batch sizes 1,8,32. The deadzone is twice the
    largest observed whitened pair discrepancy plus a fixed eight-epsilon
    roundoff allowance transformed by ||L^-1||_2 and the actual F0 norm.
    Neither stationary labels nor physical displacement enter calibration.
    """
    inputs = [(z[t - 2:t + 1], a[t - 2:t + 1])
              for _, z, a in episodes for t in range(2, len(a))][:32]
    windows, actions = (torch.stack([row[j] for row in inputs]) for j in (0, 1))
    outputs = []
    for size in (1, 8, 32):
        outputs.append(torch.cat([host.predict(windows[i:i + size], actions[i:i + size])[:, -1]
                                  for i in range(0, len(windows), size)]).double())
    discrepancy = []
    for i in range(len(outputs)):
        for j in range(i):
            white = torch.linalg.solve_triangular(noise, (outputs[i] - outputs[j]).T, upper=False).T
            discrepancy.append(white.norm(dim=-1).amax())
    measured = torch.stack(discrepancy).amax()
    inverse_norm = torch.linalg.matrix_norm(torch.linalg.inv(noise), 2)
    maximum_native_norm = torch.stack(outputs).norm(dim=-1).amax()
    allowance = 8 * torch.finfo(torch.float32).eps * inverse_norm * maximum_native_norm
    threshold = 2 * measured + allowance
    return float(threshold), {"units": "maximum pairwise whitened response L2",
        "roots": len(inputs), "native_batch_sizes": [1, 8, 32],
        "maximum_measured_whitened_FP32_discrepancy": float(measured),
        "fixed_eight_epsilon_allowance": float(allowance),
        "inverse_noise_spectral_norm": float(inverse_norm),
        "maximum_nominal_response_L2": float(maximum_native_norm),
        "threshold": float(threshold), "stationary_or_physical_labels_used": False,
        "rule": "2*max measured discrepancy + 8*eps32*||L^-1||2*max||F0||2",
        "scope": "Numeric reproducibility allowance; not a fitted physical irrelevance gate"}


def probability_checks(device="cpu"):
    torch.manual_seed(20261121)
    errors, gradient_errors = [], []
    for dimension, degenerate in ((7, False), (7, True), (192, False), (192, True)):
        means = torch.randn(5, 4, dimension, dtype=torch.float64, device=device)
        if degenerate:
            means[1] = means[1, :1]
            means[2, 2] = means[2, 1]
            means[3] *= 1e-9
            means[4] *= 0
        means.requires_grad_()
        observations = torch.randn(5, dimension, dtype=torch.float64, device=device, requires_grad=True)
        lower = torch.eye(dimension, dtype=torch.float64, device=device) * .7
        lower[2, 0] = .1
        absolute, relative, _ = response_density(means, observations, lower)
        weight = torch.tensor([.1, .2, .3, .4], dtype=torch.float64, device=device)
        transition = torch.eye(4, dtype=torch.float64, device=device)
        reference = [assimilate_continuous(issue(m, weight, transition, lower, reference="uniform"), y)
                     for m, y in zip(means, observations, strict=True)]
        expected_absolute = torch.stack([r.log_likelihood for r in reference])
        expected_relative = torch.stack([r.relative_log_likelihood for r in reference])
        for actual, expected in ((absolute, expected_absolute), (relative, expected_relative)):
            errors.append(float((actual - expected).abs().max().detach()))
            torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-12)
        probe = torch.arange(1, 21, device=device, dtype=torch.float64).reshape(5, 4) / 20
        new_grads = torch.autograd.grad((absolute * probe + relative * probe.flip(1)).sum(),
                                        (means, observations), retain_graph=True)
        old_grads = torch.autograd.grad((expected_absolute * probe + expected_relative * probe.flip(1)).sum(),
                                        (means, observations), retain_graph=True)
        for actual, expected in zip(new_grads, old_grads, strict=True):
            gradient_errors.append(float((actual - expected).abs().max()))
            torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-10)
    equal = means.detach()[:1, :1].expand(1, 4, -1)
    nonuniform = torch.tensor([[.1, .2, .3, .4]], device=device, dtype=torch.float64)
    transition = fixed_transition(device)
    no_write = assimilate_step(equal, nonuniform, observations[:1], lower,
                               transition=transition, deadzone=0.)
    assert torch.equal(no_write.prior, no_write.posterior)
    assert not torch.equal(no_write.posterior, nonuniform)
    assert torch.isfinite(no_write.log_likelihood).all()
    # Smooth full-rank numerical derivative, independent of the scalar implementation.
    m = torch.randn(2, 4, 5, dtype=torch.float64, requires_grad=True, device=device)
    y = torch.randn(2, 5, dtype=torch.float64, requires_grad=True, device=device)
    l = torch.eye(5, dtype=torch.float64, device=device)
    assert torch.autograd.gradcheck(lambda a, b: response_density(a, b, l)[:2],
                                   (m, y), atol=2e-5, rtol=1e-4)
    return {"batched_vs_original_max_absolute_value_error": max(errors),
            "batched_vs_original_max_first_gradient_error": max(gradient_errors),
            "independent_finite_difference_gradcheck": True,
            "repeated_zero_weak_responses_finite": True,
            "zero_separation_returns_transported_prior_bitwise": True,
            "zero_separation_does_not_restore_previous_belief": True,
            "quadrature_nodes": len(continuous_quadrature()[0])}


def structural_checks(model, episodes, noise, deadzone):
    _, z, a = next(row for row in episodes if len(row[2]) >= 9)
    with torch.no_grad():
        path = model.rollout(z[None, :3], a[None, :6])
        native, window = [], z[None, :3]
        for t in range(4):
            predicted = model.host.predict(window, a[None, t:t + 3])[:, -1]
            native.append(predicted)
            window = torch.cat((window[:, 1:], predicted[:, None]), 1)
        assert torch.equal(path[:, 0], torch.stack(native, 1))
        assert model.null_control is None and model.offsets.numel() == 3456
        assert not any(p.requires_grad for p in model.host.parameters())
    _, pieces = batch_objective(model, [(0, z[:10], a[:9])], noise,
                                deadzone=0., return_trace=True)
    trace = pieces["trace"][0]
    late = trace[3]["horizon_nll"].mean()
    source_gradient, = torch.autograd.grad(late, trace[0]["arrival_relative"], retain_graph=True)
    if not torch.isfinite(source_gradient).all() or source_gradient.norm() == 0:
        raise AssertionError("Later-root suffix lost gradient through historical actual write")
    gradient, = torch.autograd.grad(late, model.offsets, retain_graph=True)
    nominal_gradient, = torch.autograd.grad(model.rollout(z[None, :3], a[None, :6])[:, 0].sum(), model.offsets)
    assert torch.isfinite(gradient).all() and (gradient.flatten(1).norm(dim=1) > 0).all()
    assert torch.equal(nominal_gradient, torch.zeros_like(nominal_gradient))
    return {"nominal_H1_H4_bitwise_equal_F0": True, "nominal_offset_gradient_exact_zero": True,
        "trainable_native_offset_parameters": model.offsets.numel(), "null_control_used": False,
        "frozen_host_parameters": sum(p.numel() for p in model.host.parameters()),
        "full_prefix_source_to_later_suffix_gradient_L2": float(source_gradient.norm()),
        "later_suffix_to_each_offset_gradient_L2": gradient.flatten(1).norm(dim=1).tolist(),
        "physical_labels_loaded_by_training": False, "phase1_weights_loaded": False,
        "deadzone": deadzone, "switch_inside_H1_H4": "constant-k approximation"}


def save_checkpoint(model, directory, name, update, noise, deadzone, meta):
    torch.save({"offsets": model.offsets.detach().cpu().clone(),
                "noise_cholesky": noise.detach().cpu(), "transition": fixed_transition(),
                "deadzone": deadzone, "update": update, "budget": BUDGET,
                "training_seeds": meta["training_seeds"], "null_control": None,
                "observation_law": "G1_uniform_bounded_185", "phase1_weights_used": False,
                "min_deviation_coefficient": MIN_DEV_COEFFICIENT,
                "prediction_window_switching": "constant-k approximation"}, directory / name)


@torch.no_grad()
def descriptive_training_measure(model, episodes, noise, deadzone):
    totals = {"objective": 0., "nll_per_dimension": 0., "min_deviation_per_dimension": 0.,
              "weighted_min_deviation": 0.}
    roots = 0
    for start in range(0, len(episodes), BUDGET["batch_episodes"]):
        selected = episodes[start:start + BUDGET["batch_episodes"]]
        loss, parts = batch_objective(model, selected, noise, deadzone=deadzone)
        totals["objective"] += float(loss) * len(selected)
        for key in totals.keys() - {"objective"}:
            totals[key] += float(parts[key]) * len(selected)
        roots += parts["read_roots"]
    return {**{key: value / len(episodes) for key, value in totals.items()}, "read_roots": roots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "train"))
    parser.add_argument("--train", type=Path, default=OUT / "train.pt")
    parser.add_argument("--output", type=Path, default=OUT / "A")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(BUDGET["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output = output_path(args.output / "protocol.json").parent
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "check":
        report = probability_checks("cpu")
        write_json(args.output / "probability_checks.json", report)
        print(json.dumps(report), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen Phase2 budget requires CUDA")
    episodes = load_episodes(args.train, "cuda")
    if len(episodes) < 180:
        raise ValueError("Phase2 requires the expanded 192-episode training archive")
    model = NativePrototypes(load_host("cuda"), k=4, null_control=None).eval()
    noise_source = SPEED_DATA / "fitted.pt"
    noise = torch.load(noise_source, map_location="cpu", weights_only=True)["noise_cholesky"].to("cuda")
    threshold, calibration = calibrate_numerical_deadzone(model.host, episodes, noise)
    meta = {"training_seeds": [row[0] for row in episodes]}
    protocol = {**BUDGET, **meta, "training_source": str(args.train), "noise_source": str(noise_source),
        "only_noise_cholesky_read_from_noise_source": True, "initialization": "unlabelled seed20261119 native offsets std0.01",
        "training_usable_episodes": len(episodes), "actual_targets": sum(len(row[2]) - 2 for row in episodes),
        "observation_model": "G1 uniform bounded 185 positive fixed nodes", "transition_hazard": 2 / 30,
        "transition_fitted_or_truth_conditioned": False, "calibration": calibration,
        "objective": "equal episodes, equal all legal roots, equal legal H=1..4 joint-prefix NLL/(H*D)",
        "min_deviation": "1e-3 * equal_episode mean_root sum_{k>0,d}(f_k-f_0)^2 / D; raw latent Euclidean",
        "min_deviation_equivalent_unnormalized_coefficient": .001,
        "future_switching": "constant-k within issued suffix; actual filter transports at every arrival",
        "selection": "final update2048 only; no early stop or development access",
        "sample_order": "CPU torch.Generator seed20261121, randperm(usable)[:4] per update",
        "root_selection": "all t=2..length-1, including terminal short suffixes", "native_dtype": "FP32 TF32 disabled",
        "probability_dtype": "FP64", "truth_or_phase1_weights_used": False,
        "full_training_monitor_updates": [0, 1024, BUDGET["updates"]],
        "numeric_kernel": "exact reduced QR then 4x4 SVD; analytic node contraction in response coordinates",
        "timing_restart": "Previous ambient implementation archived; same initial state, sampling RNG, objective and full budget",
        "deployment_order": "issue native means -> T transport -> actual arrival -> density -> optional numeric no-write"}
    write_json(args.output / "protocol.json", protocol)
    checks = structural_checks(model, episodes, noise, threshold)
    write_json(args.output / "initial_checks.json", checks)
    save_checkpoint(model, args.output, "initial.pt", 0, noise, threshold, meta)
    started = perf_counter()
    baseline = descriptive_training_measure(model, episodes, noise, threshold)
    write_json(args.output / "initial_objective.json", baseline)
    print(json.dumps({"initial": baseline, "checks": checks, "calibration": calibration}), flush=True)
    optimizer = torch.optim.Adam([model.offsets], lr=BUDGET["learning_rate"])
    generator = torch.Generator().manual_seed(BUDGET["seed"])
    history = [{"update": 0, "seconds": perf_counter() - started, **baseline}]
    with (args.output / "updates.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, BUDGET["updates"] + 1):
            if update in BUDGET["decay_updates"]:
                for group in optimizer.param_groups:
                    group["lr"] *= .1
            indices = torch.randperm(len(episodes), generator=generator)[:BUDGET["batch_episodes"]].tolist()
            selected = [episodes[i] for i in indices]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = batch_objective(model, selected, noise, deadzone=threshold)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at update{update}")
            loss.backward()
            if model.offsets.grad is None or not torch.isfinite(model.offsets.grad).all():
                raise FloatingPointError(f"Nonfinite/missing full native gradient at update{update}")
            row = {"update": update, "objective_batch": float(loss.detach()), **tensor_numbers(parts),
                   "native_gradient_L2": float(model.offsets.grad.norm()),
                   "per_prototype_gradient_L2": model.offsets.grad.flatten(1).norm(dim=1).tolist(),
                   "episode_seeds": [row[0] for row in selected], "seconds": perf_counter() - started}
            optimizer.step()
            stream.write(json.dumps(row) + "\n")
            if update % 8 == 0:
                stream.flush()
            if update % 128 == 0:
                save_checkpoint(model, args.output, "latest.pt", update, noise, threshold, meta)
                print(json.dumps({"update": update, "seconds": perf_counter() - started,
                                  "objective_batch": row["objective_batch"]}), flush=True)
            if update in (1024, BUDGET["updates"]):
                history.append({"update": update, "seconds": perf_counter() - started,
                                **descriptive_training_measure(model, episodes, noise, threshold)})
                write_json(args.output / "objective_history.json", history)
                print(json.dumps(history[-1]), flush=True)
    save_checkpoint(model, args.output, "final.pt", update, noise, threshold, meta)
    write_json(args.output / "final_checks.json", structural_checks(model, episodes, noise, threshold))
    write_json(args.output / "result.json", {"updates": update, "seconds": perf_counter() - started,
        "initial": history[0], "final": history[-1], "checkpoint": str(args.output / "final.pt"),
        "termination": "fixed single budget completed", "final_selection": True})
    print(json.dumps({"done": True, "updates": update, "seconds": perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
