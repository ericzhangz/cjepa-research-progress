"""One fixed-budget unlabeled Phase2 repair; no labels or oracle initialization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter

import torch

from .prbd import continuous_quadrature
from .prbd_paths import REPAIR, output_path
from .prbd_phase2 import (BUDGET, OUT, _BatchedBoundedDensity, filter_episode, fixed_transition,
                         load_episodes, load_phase2_checkpoint, repair_batch_objective,
                         response_density)
from .prbd_phase2_fit import probability_checks, tensor_numbers, write_json
from .prbd_phase2_response import make_response_model, response_state


REPAIR_OUT = REPAIR / "unlabeled"
OPERATOR = "shared_native_suffix_copy"


def dense_logs(basis, x, scales, masses, *, covariance=None, freeze_eigenvalues=False):
    """Independent small dense reference, used only by the numerical tests."""
    n, d, _ = basis.shape
    if covariance is None:
        raw = basis @ basis.transpose(1, 2)
        if freeze_eigenvalues:
            values, vectors = torch.linalg.eigh(raw)
            bounded = values.detach() / (1 + values.detach())
            covariance = (vectors * bounded[:, None]) @ vectors.transpose(1, 2)
        else:
            covariance = torch.linalg.solve(torch.eye(d, dtype=basis.dtype)[None] + raw, raw)
    matrices = torch.eye(d, dtype=basis.dtype)[None, None] + scales[None, :, None, None] * covariance[:, None]
    residual = x[:, :, None] - basis
    solved = torch.linalg.solve(matrices, residual[:, None])
    logdet = torch.linalg.slogdet(matrices).logabsdet
    logs = -.5 * (d * math.log(2 * math.pi) + logdet[:, :, None]
                   + (residual[:, None] * solved).sum(2))
    return torch.logsumexp(masses.log()[None, :, None] + logs, 1)


def derivative_checks():
    """Test the specified surrogate, including degenerate/zero covariance."""
    torch.manual_seed(20261121)
    scales, masses = continuous_quadrature()
    # Full positive distinct spectrum: differentiate only eigenvectors in an
    # independent dense reference. Production never differentiates QR or SVD.
    basis = (torch.randn(2, 4, 4, dtype=torch.float64) + 2 * torch.eye(4)).requires_grad_()
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    coefficients = torch.randn(2, 4, dtype=torch.float64)
    ordinary = _BatchedBoundedDensity.apply(basis, x, scales, masses, False)
    frozen = _BatchedBoundedDensity.apply(basis, x, scales, masses, True)
    assert all(torch.equal(a, b) for a, b in zip(ordinary, frozen, strict=True))
    dense = dense_logs(basis, x, scales, masses, freeze_eigenvalues=True)
    analytic = torch.autograd.grad((frozen[0] * coefficients).sum(), (basis, x), retain_graph=True)
    reference = torch.autograd.grad((dense * coefficients).sum(), (basis, x))
    dense_error = max(float((a - b).abs().max()) for a, b in zip(analytic, reference, strict=True))
    assert dense_error < 2e-10, dense_error
    radial_errors, rotation_errors, nonzero_rotation, finite_cases = [], [], [], []
    principal_radial_errors, degenerate_errors = [], []
    for singular in ([2., 1., .5, 0.], [1., 1., 0., 0.], [0., 0., 0., 0.]):
        basis = torch.zeros(1, 6, 4, dtype=torch.float64)
        basis[0, :4] = torch.diag(torch.tensor(singular, dtype=torch.float64))
        basis.requires_grad_()
        x = torch.tensor([[.4, -.7, .2, .6, 1.2, -.9]], dtype=torch.float64, requires_grad=True)
        coefficients = torch.tensor([[.7, -.2, .4, -.5]], dtype=torch.float64)
        frozen = _BatchedBoundedDensity.apply(basis, x, scales, masses, True)
        gradient, = torch.autograd.grad((frozen[0] * coefficients).sum(), basis)
        raw = basis.detach() @ basis.detach().transpose(1, 2)
        covariance = torch.linalg.solve(torch.eye(6)[None] + raw, raw)
        mean_logs = dense_logs(basis, x, scales, masses, covariance=covariance)
        mean_gradient, = torch.autograd.grad((mean_logs * coefficients).sum(), basis)
        covariance_gradient = gradient - mean_gradient
        principal_radial_errors.append(float(covariance_gradient[0, :4].diagonal().abs().max()))
        assert principal_radial_errors[-1] < 2e-11
        if singular == [1., 1., 0., 0.]:
            degenerate_errors.append(float(covariance_gradient[0, :2, :2].abs().max()))
            assert degenerate_errors[-1] < 2e-11
        radial = (covariance_gradient * basis).sum().abs()
        radial_errors.append(float(radial))
        assert radial < 2e-11, radial
        # Rotation mixes the nonzero response span with its ambient complement.
        generator = torch.zeros(6, 6, dtype=torch.float64)
        generator[0, 4], generator[4, 0] = .7, -.7
        generator[1, 2], generator[2, 1] = .4, -.4
        direction = generator[None] @ basis.detach()
        epsilon = 1e-5
        rotated = []
        for sign in (-1, 1):
            candidate = torch.matrix_exp(sign * epsilon * generator)[None] @ basis.detach()
            rotated.append((_BatchedBoundedDensity.apply(candidate, x.detach(), scales, masses, False)[0]
                            * coefficients).sum())
        finite_difference = (rotated[1] - rotated[0]) / (2 * epsilon)
        error = float(((gradient * direction).sum() - finite_difference).abs())
        rotation_errors.append(error)
        nonzero_rotation.append(float((covariance_gradient * direction).sum().abs()))
        assert error < 2e-8, error
        assert torch.isfinite(gradient).all()
        finite_cases.append(singular)
    assert max(nonzero_rotation) > .001, nonzero_rotation
    means = torch.randn(2, 4, 7, dtype=torch.float64, requires_grad=True)
    observations = torch.randn(2, 7, dtype=torch.float64, requires_grad=True)
    lower = torch.eye(7, dtype=torch.float64)
    weights = torch.tensor([[.3, -.1, .4, .7], [-.1, .2, -.3, .8]], dtype=torch.float64)
    normal = response_density(means, observations, lower)
    frozen = response_density(means, observations, lower, freeze_covariance_eigenvalues=True)
    normal_grad = torch.autograd.grad((normal[0] * weights).sum(), (means, observations))
    frozen_grad = torch.autograd.grad((frozen[0] * weights).sum(), (means, observations))
    mean_error = float((normal_grad[0].sum(1) - frozen_grad[0].sum(1)).abs().max())
    observation_error = float((normal_grad[1] - frozen_grad[1]).abs().max())
    assert mean_error < 1e-11 and observation_error < 1e-11
    return {"G1_forward_bitwise_unchanged": True,
            "distinct_positive_spectrum_dense_orientation_gradient_max_error": dense_error,
            "covariance_radial_gradient_max_error": max(radial_errors),
            "each_principal_covariance_radial_gradient_max_error": max(principal_radial_errors),
            "repeated_positive_eigenspace_covariance_gradient_max_error": max(degenerate_errors),
            "rotation_including_ambient_complement_finite_difference_max_error": max(rotation_errors),
            "retained_covariance_rotation_directional_derivative": nonzero_rotation,
            "common_mean_gradient_max_error": mean_error,
            "complete_observation_gradient_max_error": observation_error,
            "finite_singular_value_cases": finite_cases,
            "exact_degeneracy_convention": "zero derivative within each equal-eigenvalue block",
            "production_QR_or_SVD_differentiated": False}


def structural_checks(model, episodes, noise, deadzone, *, require_nonzero=True):
    _, z, a = next(row for row in episodes if len(row[2]) >= 10)
    parameters = tuple(model.response_parameters())
    with torch.no_grad():
        paths = model.rollout(z[None, :3], a[None, :6])
        native, window = [], z[None, :3]
        for t in range(4):
            next_state = model.host.predict(window, a[None, t:t + 3])[:, -1]
            native.append(next_state)
            window = torch.cat((window[:, 1:], next_state[:, None]), 1)
        assert torch.equal(paths[:, 0], torch.stack(native, 1))
        assert model.null_control is None and model.offsets.numel() == 3456
        assert not any(p.requires_grad for p in model.host.parameters())
    _, parts = repair_batch_objective(model, [(0, z[:11], a[:10])], noise, deadzone=0., return_trace=True)
    trace = parts["trace"][0]
    late = trace[-1]["horizon_nll"].mean()
    source_gradient, = torch.autograd.grad(late, trace[0]["arrival_relative"], retain_graph=True)
    assert torch.isfinite(source_gradient).all()
    gradients = torch.autograd.grad(late, parameters, retain_graph=True)
    assert all(torch.isfinite(g).all() for g in gradients)
    offset_gradient = gradients[0]
    shared_norm = torch.stack([g.square().sum() for g in gradients[1:]]).sum().sqrt()
    if require_nonzero:
        assert source_gradient.norm() > 0
        assert (offset_gradient.flatten(1).norm(dim=1) > 0).all() and shared_norm > 0
    nominal_gradients = torch.autograd.grad(model.rollout(z[None, :3], a[None, :6])[:, 0].sum(),
                                            parameters, allow_unused=True)
    assert all(g is None or not bool(g.any()) for g in nominal_gradients)
    # Explicit arithmetic test for the new suffix law, independent of its trace.
    first = trace[0]
    target = z[3:7].double()
    full = model.rollout(z[None, :3], a[None, :6])[0].double()
    residual = (target[None] - full).flatten(0, 1)
    white = torch.linalg.solve_triangular(noise, residual.T, upper=False).T.reshape_as(full)
    energies = .5 * white.square().sum(-1).cumsum(1).T
    expected = -torch.logsumexp(first["prior"].log()[None] - energies, -1) / (noise.size(0) * torch.arange(1, 5, device=noise.device))
    torch.testing.assert_close(first["horizon_nll"], expected, atol=2e-6, rtol=2e-6)
    return {"nominal_H1_H4_bitwise_equal_F0": True,
            "nominal_all_learned_gradients_zero_or_absent": True,
            "trainable_parameters": sum(p.numel() for p in parameters),
            "offset_parameters": model.offsets.numel(),
            "shared_native_suffix_parameters": sum(p.numel() for p in parameters[1:]),
            "full_prefix_earliest_actual_write_to_last_root_gradient_L2": float(source_gradient.norm()),
            "last_root_each_offset_gradient_L2": offset_gradient.flatten(1).norm(dim=1).tolist(),
            "last_root_shared_native_gradient_L2": float(shared_norm),
            "fixed_energy_beta_1_direct_arithmetic_verified": True,
            "all_native_gradients_finite": True, "optimizer_updates": 0,
            "truth_or_oracle_weights_loaded": False, "deadzone": deadzone}


def blindness_and_causality_checks(model, episodes, calibration, train_path):
    source = torch.load(train_path, map_location="cpu", weights_only=True)
    source.update({"speed": torch.full((len(source["z"]),), 999.),
                   "cut_points": torch.zeros(len(source["z"]), 2),
                   "oracle_state": {"offsets": torch.full_like(model.offsets.cpu(), 999.)}})
    with tempfile.TemporaryDirectory(prefix="prbd_repair_blindness_") as temporary:
        path = Path(temporary) / "contradictory_metadata.pt"
        torch.save(source, path)
        changed = load_episodes(path, str(model.offsets.device))
    assert len(changed) == len(episodes)
    for before, after in zip(episodes, changed, strict=True):
        assert before[0] == after[0] and torch.equal(before[1], after[1]) and torch.equal(before[2], after[2])
    _, z, a = next(row for row in episodes if len(row[2]) >= 9)
    altered = z.clone()
    altered[6:] += 10
    with torch.no_grad():
        original = filter_episode(model, z[:10], a[:9], **calibration)
        future_changed = filter_episode(model, altered[:10], a[:9], **calibration)
        for before, after in zip(original[:3], future_changed[:3], strict=True):
            assert torch.equal(before["responses"], after["responses"])
            assert torch.equal(before["step"].prior, after["step"].prior)
            assert torch.equal(before["step"].posterior, after["step"].posterior)
    return {"contradictory_metadata_and_fake_oracle_weights_cannot_cross_training_boundary": True,
            "training_boundary": ["z", "a", "seed", "lengths"],
            "future_actual_perturbation_from_index6_preserves_filter_roots_2_3_4_bitwise": True,
            "optimizer_updates": 0}


@torch.no_grad()
def descriptive_measure(model, episodes, noise, deadzone):
    totals = {"objective": 0., "nll_per_dimension": 0., "min_deviation_per_dimension": 0.,
              "weighted_min_deviation": 0.}
    early, late = [], []
    roots = targets = 0
    for start in range(0, len(episodes), BUDGET["batch_episodes"]):
        selected = episodes[start:start + BUDGET["batch_episodes"]]
        loss, parts = repair_batch_objective(model, selected, noise, deadzone=deadzone, return_trace=True)
        totals["objective"] += float(loss) * len(selected)
        for key in totals.keys() - {"objective"}:
            totals[key] += float(parts[key]) * len(selected)
        roots += parts["read_roots"]
        targets += parts["suffix_target_steps"]
        for trace in parts["trace"]:
            for indices, store in ((range(min(4, len(trace))), early),
                                   (range(max(0, len(trace) - 4), len(trace)), late)):
                rows = [trace[i] for i in indices]
                store.append({"fixed_energy_loss_per_dimension": float(torch.stack([r["horizon_nll"].mean() for r in rows]).mean()),
                              "mean_response_separation": float(torch.stack([r["separation"] for r in rows]).mean()),
                              "posterior_entropy": float(torch.stack([-(r["posterior"] * r["posterior"].log()).sum() for r in rows]).mean())})
    aggregate = lambda rows: {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
    return {**{key: value / len(episodes) for key, value in totals.items()},
            "read_roots": roots, "suffix_target_steps": targets,
            "early_first_four_actual_roots": aggregate(early),
            "late_last_four_actual_roots": aggregate(late)}


def save_checkpoint(model, directory, name, update, calibration, seeds):
    torch.save({"response_operator": OPERATOR, "response_state": response_state(model),
                "noise_cholesky": calibration["noise_cholesky"].detach().cpu(),
                "transition": calibration["transition"].detach().cpu(),
                "deadzone": calibration["deadzone"], "update": update, "budget": dict(BUDGET),
                "training_seeds": seeds, "null_control": None,
                "observation_law": "G1_uniform_bounded_185", "training_suffix_law": "fixed_white_Gaussian_energy_beta_1",
                "prefix_covariance_gradient": "frozen_eigenvalues_retained_mean_and_orientation",
                "min_deviation_coefficient": 1., "min_deviation_metric": "0.5_mean_nonnominal_white_squared_per_dimension",
                "oracle_or_phase1_weights_used": False,
                "prediction_window_switching": "constant-k approximation"}, directory / name)


def verify_budget(directory, episodes):
    rows = [json.loads(line) for line in (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == BUDGET["updates"]
    generator = torch.Generator().manual_seed(BUDGET["seed"])
    read_roots = suffix_steps = 0
    learning_rate = BUDGET["learning_rate"]
    for update, row in enumerate(rows, 1):
        if update in BUDGET["decay_updates"]:
            learning_rate *= .1
        assert row["learning_rate"] == learning_rate
        selected = [episodes[i] for i in torch.randperm(len(episodes), generator=generator)[:4].tolist()]
        assert row["update"] == update and row["episode_seeds"] == [e[0] for e in selected]
        assert row["read_roots"] == sum(len(a) - 2 for _, _, a in selected)
        assert row["suffix_target_steps"] == sum(sum(min(4, len(a) - t) for t in range(2, len(a))) for _, _, a in selected)
        assert math.isclose(row["objective_batch"], row["nll_per_dimension"] + row["weighted_min_deviation"], abs_tol=1e-10)
        assert row["weighted_min_deviation"] == row["min_deviation_per_dimension"]
        assert math.isfinite(row["native_gradient_L2"])
        read_roots += row["read_roots"]
        suffix_steps += row["suffix_target_steps"]
    old_rows = [json.loads(line) for line in (OUT / "A/updates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(a["episode_seeds"] == b["episode_seeds"] for a, b in zip(rows, old_rows, strict=True))
    return {"updates": len(rows), "sampled_episodes": 4 * len(rows), "sampled_read_roots": read_roots,
            "sampled_suffix_target_steps": suffix_steps, "original_episode_order_exactly_reused": True,
            "learning_rate_schedule_exactly_reused": True,
            "all_loss_recompositions_and_root_counts_verified": True,
            "fixed_beta_1_lambda_1": True, "all_native_gradients_finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "train"))
    parser.add_argument("--train", type=Path, default=OUT / "train.pt")
    parser.add_argument("--output", type=Path, default=REPAIR_OUT)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(BUDGET["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output = output_path(args.output / "protocol.json").parent
    if args.stage == "check":
        checks = {"original_default_regression": probability_checks(), "repair_derivative": derivative_checks()}
        write_json(args.output / "probability_checks.json", checks)
        print(json.dumps(checks), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen repair budget requires CUDA")
    if (args.output / "updates.jsonl").exists() or (args.output / "final.pt").exists():
        raise FileExistsError("This fixed-budget repair already started; preserve its existing updates and checkpoints")
    episodes = load_episodes(args.train, "cuda")
    if len(episodes) != 186:
        raise ValueError("Repair must use the same 186 usable trajectories as original Phase2")
    # Read only fixed calibration; the original trained offsets never initialize
    # this new fit. The shared suffix starts from released native host weights.
    source = torch.load(OUT / "A/initial.pt", map_location="cpu", weights_only=True)
    calibration = {"noise_cholesky": source["noise_cholesky"].to("cuda"),
                   "transition": source["transition"].to("cuda"), "deadzone": float(source["deadzone"])}
    torch.testing.assert_close(calibration["transition"].cpu(), fixed_transition(), atol=0, rtol=0)
    model = make_response_model("cuda")
    parameters = tuple(model.response_parameters())
    noise, deadzone = calibration["noise_cholesky"], calibration["deadzone"]
    seeds = [row[0] for row in episodes]
    protocol = {**BUDGET, "training_source": str(args.train), "training_usable_episodes": len(episodes),
                "training_seeds": seeds, "response_operator": OPERATOR,
                "trainable_parameters": sum(p.numel() for p in parameters),
                "initialization": response_state(model)["source"],
                "calibration_source": str(OUT / "A/initial.pt"),
                "only_noise_transition_deadzone_read_from_calibration_source": True,
                "observation_law_forward": "unchanged G1 uniform bounded 185 positive fixed nodes",
                "prefix_gradient": "full means and covariance orientation; covariance eigenvalues frozen",
                "exact_degeneracy_convention": "zero covariance differential inside equal-eigenvalue blocks",
                "suffix_energy": "E_k(H)=0.5*sum_{j<=H,d} [L^-1(z_j-f_kj)]_d^2, beta=1",
                "objective": "equal episodes; equal roots; equal legal H; -logsumexp(log(prior)-E)/(H*D)",
                "min_deviation": "lambda=1 * 0.5*mean_{k>0,d}[L^-1(f_k-f0)]_d^2 per factual root",
                "min_deviation_reason": "unit white per-branch Gaussian prior curvature; one fixed white observation per nonnominal branch on average",
                "sample_order": "same CPU Generator seed20261121 randperm(186)[:4] each update",
                "learning_rate_schedule": "original 3e-4, multiply0.1 before updates1024,1536",
                "selection": "final update2048; no sweeps, no early stop, no development access",
                "oracle_or_truth_or_phase1_weights_used": False,
                "native_dtype": "FP32; TF32 disabled", "probability_dtype": "FP64",
                "future_switching": "same constant-k suffix approximation; actual filter transport each arrival"}
    write_json(args.output / "protocol.json", protocol)
    checks = structural_checks(model, episodes, noise, deadzone)
    write_json(args.output / "initial_checks.json", checks)
    write_json(args.output / "blindness_and_causality_checks.json",
               blindness_and_causality_checks(model, episodes, calibration, args.train))
    save_checkpoint(model, args.output, "initial.pt", 0, calibration, seeds)
    # New/old checkpoint formats use the identical public forecast loader.
    reloaded, reload_calibration = load_phase2_checkpoint(args.output / "initial.pt", "cuda")
    with torch.no_grad():
        _, z, a = episodes[0]
        assert torch.equal(model(z[None, :3], a[None, :3]), reloaded(z[None, :3], a[None, :3]))
        assert torch.equal(reload_calibration["noise_cholesky"], noise)
    del reloaded
    started = perf_counter()
    initial = descriptive_measure(model, episodes, noise, deadzone)
    write_json(args.output / "initial_objective.json", initial)
    print(json.dumps({"initial": initial, "checks": checks}), flush=True)
    history = [{"update": 0, "seconds": perf_counter() - started, **initial}]
    write_json(args.output / "objective_history.json", history)
    optimizer = torch.optim.Adam(parameters, lr=BUDGET["learning_rate"])
    generator = torch.Generator().manual_seed(BUDGET["seed"])
    with (args.output / "updates.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, BUDGET["updates"] + 1):
            if update in BUDGET["decay_updates"]:
                for group in optimizer.param_groups:
                    group["lr"] *= .1
            indices = torch.randperm(len(episodes), generator=generator)[:BUDGET["batch_episodes"]].tolist()
            selected = [episodes[i] for i in indices]
            optimizer.zero_grad(set_to_none=True)
            loss, parts = repair_batch_objective(model, selected, noise, transition=calibration["transition"], deadzone=deadzone)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at update{update}")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f"Missing/nonfinite response gradient at update{update}")
            offset_norms = model.offsets.grad.flatten(1).norm(dim=1).tolist()
            gradient_norm = torch.stack([p.grad.square().sum() for p in parameters]).sum().sqrt()
            row = {"update": update, "objective_batch": float(loss.detach()), **tensor_numbers(parts),
                   "native_gradient_L2": float(gradient_norm), "per_prototype_gradient_L2": offset_norms,
                   "learning_rate": optimizer.param_groups[0]["lr"],
                   "episode_seeds": [e[0] for e in selected], "seconds": perf_counter() - started}
            optimizer.step()
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if update % 8 == 0:
                stream.flush()
            if update % 128 == 0:
                save_checkpoint(model, args.output, "latest.pt", update, calibration, seeds)
                print(json.dumps({"update": update, "seconds": perf_counter() - started,
                                  "objective_batch": row["objective_batch"]}), flush=True)
            if update in (1024, BUDGET["updates"]):
                history.append({"update": update, "seconds": perf_counter() - started,
                                **descriptive_measure(model, episodes, noise, deadzone)})
                write_json(args.output / "objective_history.json", history)
                print(json.dumps(history[-1]), flush=True)
    save_checkpoint(model, args.output, "final.pt", update, calibration, seeds)
    write_json(args.output / "final_checks.json", structural_checks(model, episodes, noise, deadzone, require_nonzero=False))
    budget = verify_budget(args.output, episodes)
    write_json(args.output / "budget_verification.json", budget)
    write_json(args.output / "result.json", {"updates": update, "seconds": perf_counter() - started,
                "initial": history[0], "final": history[-1], "checkpoint": str(args.output / "final.pt"),
                "termination": "one frozen budget completed; final selected without development", "budget": budget})
    print(json.dumps({"done": True, "updates": update, "seconds": perf_counter() - started}), flush=True)


if __name__ == "__main__":
    main()
