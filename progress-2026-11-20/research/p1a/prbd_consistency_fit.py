"""G3 fixed-budget H1 versus episode-H1/H4 fitting on actual single episodes.

Both arms share the native action-increment operator, calibrated noise, initial
offsets, episode order, Adam(0.001), and one optimizer update per usable episode.
The H1 arm is the episode mixture exactly equivalent to sequential filtering
under static prototypes and fixed uniform-reference observation geometry.
"""

from .prbd_paths import ROOT as RELEASE_ROOT, CONSISTENCY, SPEED_DATA, output_path

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import torch

from .prbd import (assimilate_continuous, continuous_quadrature, issue)
from .prbd_native import NativePrototypes
from .prbd_public_fit import (consistency_episode_loss, fit, partition_paths,
                             windows)
from .prbd_tworoom_speed_fit import load_host


ROOT = RELEASE_ROOT
OUT = CONSISTENCY


def load_episodes(path):
    """Read actual encoded observations/actions only; metadata never trains."""
    batch = torch.load(path, map_location="cpu", weights_only=True)
    if not {"z", "a", "seed", "lengths", "null_control"} <= batch.keys():
        raise ValueError("Encoded actual episodes require z,a,seed,lengths,null_control")
    z, a, lengths, seeds = (batch[name] for name in ("z", "a", "lengths", "seed"))
    if (z.ndim != 3 or a.ndim != 3 or z.shape[0] != a.shape[0]
            or z.shape[1] != a.shape[1] + 1 or z.shape[2] != 192 or a.shape[2] != 192
            or lengths.shape != (len(z),) or seeds.shape != (len(z),)):
        raise ValueError("Expected z[N,T+1,192], a[N,T,192], seed[N], lengths[N]")
    if lengths.is_floating_point() or seeds.is_floating_point():
        raise ValueError("Actual lengths and reset seeds must be integer tensors")
    if (lengths < 0).any() or (lengths > a.size(1)).any() or len(set(seeds.tolist())) != len(seeds):
        raise ValueError("Invalid actual lengths or duplicate reset seeds")
    null_control = batch["null_control"].detach().clone()
    if null_control.shape != (192,) or not torch.isfinite(null_control).all():
        raise ValueError("Canonical native raw-zero reference must have shape [192]")
    episodes, skipped = [], []
    for seed, states, actions, length in zip(seeds, z, a, lengths, strict=True):
        identifier, size = int(seed), int(length)
        if size < 3:
            skipped.append({"seed": identifier, "actual_length": size,
                            "reason": "No actual successor after a three-state window"})
            continue
        states, actions = states[:size + 1].clone(), actions[:size].clone()
        if not torch.isfinite(states).all() or not torch.isfinite(actions).all():
            raise ValueError(f"Nonfinite actual input in episode {identifier}")
        episodes.append((identifier, states, actions, None))
    if not episodes:
        raise ValueError("No usable actual training episodes")
    return episodes, null_control, skipped


@torch.no_grad()
def development_stats(model, episodes, noise, transition, nodes, masses):
    """Descriptive per-episode proper scores and causal endpoint MSE.

    Forecast weights contain only observations preceding the forecast root.
    H4 endpoint statistics use the disjoint H4 roots and the actual terminal
    horizon (H<=4); counts identify terminal and full-four-step forecasts.
    """
    if any(module.training for module in model.modules()):
        raise ValueError("Development statistics require eval-mode predictors")
    count = len(transition)
    uniform = transition.new_full((count,), 1 / count)
    rows = []
    for seed, z, a, _ in episodes:
        loss, counts = consistency_episode_loss(model, z, a, noise, transition, nodes, masses)
        part_scores = {name: -float(torch.logsumexp(
            torch.tensor(details["prototype_episode_log_likelihood"], dtype=torch.float64)
            + uniform.log(), 0)) / details["scored_target_steps"]
                       for name, details in counts["partitions"].items()}
        responses = model(windows(z[:-1]), windows(a))
        belief, priors, entropy = uniform.clone(), {}, []
        one_step = {name: [] for name in ("nominal", "uniform", "causal")}
        for root, response in enumerate(responses, start=2):
            priors[root] = belief.clone()
            predictions = {"nominal": response[0].double(),
                           "uniform": uniform @ response.double(),
                           "causal": belief @ response.double()}
            for name, predicted in predictions.items():
                one_step[name].append(float((predicted - z[root + 1]).square().mean()))
            signed = issue(response, belief, transition, noise, reference="uniform")
            arrival = assimilate_continuous(signed, z[root + 1], nodes, masses)
            belief = arrival.posterior
            positive = belief > 0
            entropy.append(float(-(belief[positive] * belief[positive].log()).sum()))
        endpoints = []
        for root, path in partition_paths(model, z, a, 4):
            endpoint = path[:, -1].double()
            target = z[root + path.size(1)].double()
            predictions = {"nominal": endpoint[0], "uniform": uniform @ endpoint,
                           "causal": priors[root] @ endpoint}
            endpoints.append({"root": root, "horizon": path.size(1),
                              "mse": {name: float((predicted - target).square().mean())
                                      for name, predicted in predictions.items()}})
        rows.append({"seed": seed, "actual_length": len(a), "usable_targets": len(a) - 2,
                     "episode_h1_nll_per_target": part_scores["H1"],
                     "episode_h4_nll_per_target": part_scores["H4"],
                     "dual_composite_nll_per_target": float(loss),
                     "causal_h1_mse": {name: sum(values) / len(values)
                                        for name, values in one_step.items()},
                     "mean_posterior_entropy": sum(entropy) / len(entropy),
                     "terminal_posterior": belief.tolist(), "block_endpoints": endpoints})
    summary = {name: sum(row[name] for row in rows) / len(rows)
               for name in ("episode_h1_nll_per_target", "episode_h4_nll_per_target",
                            "dual_composite_nll_per_target", "mean_posterior_entropy")}
    summary["causal_h1_mse"] = {
        name: sum(row["causal_h1_mse"][name] for row in rows) / len(rows)
        for name in ("nominal", "uniform", "causal")}
    summary["block_endpoint_mse"] = {}
    for horizon in (1, 2, 3, 4):
        eligible = [[event for event in row["block_endpoints"] if event["horizon"] == horizon]
                    for row in rows]
        eligible = [events for events in eligible if events]
        if eligible:
            summary["block_endpoint_mse"][f"H{horizon}"] = {
                "eligible_episodes": len(eligible), "events": sum(map(len, eligible)),
                **{name: sum(sum(event["mse"][name] for event in events) / len(events)
                              for events in eligible) / len(eligible)
                   for name in ("nominal", "uniform", "causal")}}
    return {"aggregation": "Equal episode means; descriptive development statistics",
            "episodes": rows, "summary": summary}


def write_json(path, value):
    output_path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=OUT / "train.pt")
    parser.add_argument("--development", type=Path)
    parser.add_argument("--noise-checkpoint", type=Path, default=
                        SPEED_DATA / "fitted.pt")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output = output_path(args.output)
    started = perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(20261120)
    train, null_control, skipped = load_episodes(args.train)
    development = None
    if args.development is not None:
        development, dev_null, _ = load_episodes(args.development)
        if not torch.equal(null_control, dev_null):
            raise ValueError("Training and development require the same canonical raw-zero embedding")
        if set(row[0] for row in train) & set(row[0] for row in development):
            raise ValueError("Development reset seeds must be disjoint from training")
    host = load_host("cpu")
    model = NativePrototypes(host, null_control=null_control).eval()
    initial = model.offsets.detach().clone()
    source = torch.load(args.noise_checkpoint, map_location="cpu", weights_only=True)
    noise = source["noise_cholesky"].detach().clone().double()
    transition = torch.eye(len(model.conditions), dtype=torch.float64)
    nodes, masses = continuous_quadrature()
    args.output.mkdir(parents=True, exist_ok=True)
    baseline = None
    if development is not None:
        print("Measuring shared initialization on development episodes", flush=True)
        baseline = development_stats(model, development, noise, transition, nodes, masses)
        write_json(args.output / "g3_initialized_development.json", baseline)
    protocol = {
        "training_source": str(args.train.resolve()),
        "development_source": str(args.development.resolve()) if args.development else None,
        "noise_source": str(args.noise_checkpoint.resolve()),
        "training_seeds": [row[0] for row in train], "skipped_training_episodes": skipped,
        "usable_actual_targets": sum(len(row[2]) - 2 for row in train),
        "device": "cpu", "threads": 2, "torch": str(torch.__version__),
        "python": platform.python_version(), "optimizer": "Adam", "learning_rate": .001,
        "updates_per_usable_episode": 1, "passes": 1,
        "initial_belief": "uniform", "transition": "identity; one k per complete episode",
        "observation_model": "bounded continuous nu=3; fixed uniform response reference",
        "continuous_integration_nodes": len(nodes), "hyperparameter_search": False,
        "episode_h1": "One mixture after the product over every usable H1 target",
        "episode_dual": "Equal mean of H1 and disjoint H<=4 episode log densities, per target",
        "imagined_or_future_public_belief_writes": 0,
        "development_used_for_training_or_selection": False}
    arms = {}
    for objective in ("episode_h1", "episode_dual"):
        with torch.no_grad():
            model.offsets.copy_(initial)
        fit_started = perf_counter()
        print(f"Fitting {objective}: {len(train)} single-episode updates", flush=True)
        records = fit(model, train, noise, transition, nodes, masses, objective=objective,
                      geometry="continuous_bounded_uniform")
        fit_seconds = perf_counter() - fit_started
        _, check_z, check_a, _ = train[0]
        with torch.no_grad():
            state_windows, control_windows = windows(check_z[:-1]), windows(check_a)
            nominal = model(state_windows, control_windows)[:, 0]
            expected = host.predict(state_windows, control_windows)[:, -1]
            torch.testing.assert_close(nominal, expected, atol=0, rtol=0)
            noop_controls = torch.cat((check_a[:2], model.null_control.expand(4, -1)))[None]
            noop_paths = model.rollout(check_z[None, :3], noop_controls)[0]
            torch.testing.assert_close(noop_paths, noop_paths[:1].expand_as(noop_paths), atol=0, rtol=0)
            nonuniform = torch.arange(1, len(transition) + 1, dtype=torch.float64)
            nonuniform /= nonuniform.sum()
            noop_issue = issue(noop_paths[:, 0], nonuniform, transition, noise, reference="uniform")
            noop_arrival = assimilate_continuous(noop_issue, check_z[3], nodes, masses)
            assert torch.equal(noop_issue.basis, torch.zeros_like(noop_issue.basis))
            assert torch.equal(noop_arrival.log_likelihood,
                               noop_arrival.log_likelihood[:1].expand_as(noop_arrival.log_likelihood))
            assert torch.equal(noop_arrival.posterior, noop_issue.prior)
        checkpoint = args.output / f"g3_{objective}.pt"
        torch.save({"offsets": model.offsets.detach().clone(), "initialized_offsets": initial,
                    "null_control": null_control, "noise_cholesky": noise,
                    "transition": transition, "scales": nodes, "masses": masses,
                    "observation_law": "continuous_bounded_uniform_nu3",
                    "objective": objective, "training_seeds": protocol["training_seeds"]}, output_path(checkpoint))
        output = {"objective": objective, "optimization_steps": len(records),
                  "fit_seconds": fit_seconds, "checkpoint": str(checkpoint.resolve()),
                  "nominal_anchor_exact_after_fit": True,
                  "postfit_noop": {"native_h1_h4_all_prototypes_exact_equal": True,
                                   "observation_basis_exact_zero": True,
                                   "branch_density_exact_equal": True,
                                   "nonuniform_posterior_exactly_preserved": True},
                  "offset_change_l2": (model.offsets.detach() - initial).flatten(1).norm(dim=1).tolist(),
                  "training_records": records}
        if development is not None:
            eval_started = perf_counter()
            output["development"] = development_stats(model, development, noise, transition, nodes, masses)
            output["development_seconds"] = perf_counter() - eval_started
        arms[objective] = output
        write_json(args.output / f"g3_{objective}.json", output)
        print(f"{objective} saved after {fit_seconds:.3f} fit seconds", flush=True)
    write_json(args.output / "g3_fit.json", {"protocol": protocol, "arms": arms,
                                            "total_seconds": perf_counter() - started})
    print(args.output / "g3_fit.json", flush=True)


if __name__ == "__main__":
    main()
