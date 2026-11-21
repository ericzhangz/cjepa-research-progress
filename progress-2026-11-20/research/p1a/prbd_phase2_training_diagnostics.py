"""Read-only initial/final loss and native-gradient decomposition."""

import argparse
import json
import math
from pathlib import Path
from .prbd_paths import output_path
import tempfile

import torch

from .prbd_phase2 import (BUDGET, OUT, assimilate_step, batch_objective, filter_episode,
                         load_episodes, load_phase2_checkpoint)


def verify_budget(directory):
    episodes = load_episodes(OUT / "train.pt")
    rows = [json.loads(line) for line in (directory / "updates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == BUDGET["updates"]
    generator = torch.Generator().manual_seed(BUDGET["seed"])
    total_roots, total_targets = 0, 0
    for update, row in enumerate(rows, start=1):
        assert row["update"] == update
        chosen = torch.randperm(len(episodes), generator=generator)[:BUDGET["batch_episodes"]].tolist()
        selected = [episodes[i] for i in chosen]
        assert row["episode_seeds"] == [item[0] for item in selected]
        roots = sum(len(a) - 2 for _, _, a in selected)
        targets = sum(sum(min(4, len(a) - t) for t in range(2, len(a))) for _, _, a in selected)
        assert row["read_roots"] == roots and row["suffix_target_steps"] == targets
        assert math.isclose(row["objective_batch"], row["nll_per_dimension"] + row["weighted_min_deviation"], abs_tol=1e-10)
        assert all(math.isfinite(value) for value in [row["native_gradient_L2"], *row["per_prototype_gradient_L2"]])
        total_roots += roots
        total_targets += targets
    report = {"updates": len(rows), "sampled_episodes": len(rows) * BUDGET["batch_episodes"],
        "sampled_read_roots": total_roots, "sampled_suffix_target_steps": total_targets,
        "exact_frozen_rng_episode_order_verified": True,
        "all_legal_root_and_suffix_counts_verified": True,
        "all_logged_loss_recompositions_verified": True,
        "all_native_gradients_finite": True, "independent_evaluation_used_for_selection": False}
    output_path(directory / "budget_verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def blindness_and_causality_checks(model, calibration, episodes):
    # Deliberately contradictory fake metadata cannot cross the four-tensor
    # boundary. These are synthetic labels and never an experiment target.
    source = torch.load(OUT / "train.pt", map_location="cpu", weights_only=True)
    source.update({"speed": torch.full((192,), 999.), "cut_points": torch.zeros(192, 2),
                   "null_control": torch.full((192,), -999.),
                   "phase1_offsets": torch.full((3, 6, 192), 999.)})
    with tempfile.NamedTemporaryFile(prefix="prbd_blindness_", suffix=".pt", delete=False) as temporary:
        path = Path(temporary.name)
    try:
        torch.save(source, path)
        changed = load_episodes(path, str(model.offsets.device))
        original = load_episodes(OUT / "train.pt", str(model.offsets.device))
        assert len(changed) == len(original)
        for a, b in zip(changed, original, strict=True):
            assert a[0] == b[0] and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2])
    finally:
        path.unlink()
    _, z, a = episodes[0]
    changed_z = z.clone()
    changed_z[6:] += 10
    with torch.no_grad():
        before = filter_episode(model, z[:10], a[:9], **calibration)
        after = filter_episode(model, changed_z[:10], a[:9], **calibration)
        for first, second in zip(before[:3], after[:3], strict=True):
            assert torch.equal(first["responses"], second["responses"])
            assert torch.equal(first["step"].prior, second["step"].prior)
            assert torch.equal(first["step"].posterior, second["step"].posterior)
        original_offsets = model.offsets.detach().clone()
        model.offsets.zero_()
        windows, actions, targets = [], [], []
        for _, states, controls in episodes:
            for t in range(2, len(controls)):
                windows.append(states[t - 2:t + 1])
                actions.append(controls[t - 2:t + 1])
                targets.append(states[t + 1])
                if len(windows) == 32:
                    break
            if len(windows) == 32:
                break
        means = model(torch.stack(windows), torch.stack(actions))
        prior = calibration["noise_cholesky"].new_tensor([.1, .2, .3, .4]).expand(len(means), -1)
        roundoff = assimilate_step(means, prior, torch.stack(targets), **calibration)
        assert roundoff.silent.all()
        assert torch.equal(roundoff.prior, roundoff.posterior)
        maximum_separation = float(roundoff.separation.max())
        model.offsets.copy_(original_offsets)
    return {"contradictory_fake_labels_cuts_null_and_phase1_offsets_ignored": True,
            "only_z_a_seed_lengths_reach_training": True,
            "future_actual_latent_perturbation_cannot_change_earlier_issuance_or_write": True,
            "perturbed_from_actual_index": 6, "unchanged_filter_roots": [2, 3, 4],
            "zero_condition_FP32_batch_roundoff_covered_by_fixed_deadzone": True,
            "zero_condition_maximum_whitened_pair_separation": maximum_separation,
            "zero_condition_no_write_returns_transported_prior_bitwise": True}


def diagnose(checkpoint, device="cpu"):
    model, calibration = load_phase2_checkpoint(checkpoint, device)
    episodes = load_episodes(OUT / "train.pt", device)[:4]
    loss, parts = batch_objective(model, episodes, calibration["noise_cholesky"],
                                  deadzone=calibration["deadzone"])
    gradients = {}
    for name, term in (("NLL", parts["nll_per_dimension"]),
                       ("weighted_min_dev", parts["weighted_min_deviation"]),
                       ("total", loss)):
        gradients[name], = torch.autograd.grad(term, model.offsets, retain_graph=True)
    torch.testing.assert_close(gradients["NLL"] + gradients["weighted_min_dev"], gradients["total"],
                               atol=2e-6, rtol=2e-5)
    report = {"checkpoint": str(checkpoint), "device": device,
        "sampling": "first four usable actual episodes, no updates",
        "seeds": [row[0] for row in episodes], "read_roots": parts["read_roots"],
        "losses": {key: float(value.detach()) for key, value in parts.items() if torch.is_tensor(value)},
        "gradient_L2": {name: float(value.norm()) for name, value in gradients.items()},
        "per_prototype_gradient_L2": {name: value.flatten(1).norm(dim=1).tolist() for name, value in gradients.items()},
        "gradient_sum_verified": True, "optimizer_updates": 0}
    report["blindness_and_causality"] = blindness_and_causality_checks(model, calibration, episodes)
    if checkpoint.stem == "final":
        report["budget"] = verify_budget(checkpoint.parent)
    name = checkpoint.stem
    output_path(checkpoint.with_name(name + "_gradient_decomposition.json")).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    torch.save({name: value.detach().cpu() for name, value in gradients.items()},
               output_path(checkpoint.with_name(name + "_gradient_decomposition.pt")))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    print(json.dumps(diagnose(args.checkpoint, args.device)), flush=True)


if __name__ == "__main__":
    main()
