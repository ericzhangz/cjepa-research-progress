"""Phase 2 factual-prefix baselines; the authors' LeWM online adapter is unchanged.

Shadow CEM uses the public reset goal, never a measured query endpoint. Its
population is captured before the executed action; adaptation follows arrival.
The shadow actions are not executed and this is not a task-success experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from .prbd_phase2 import BUDGET, load_episodes
from .prbd_tworoom_speed_fit import load_host
from .stage_a_baselines import GRUDynamics, OnlineAdapter
from .lewm_optta.adapter import OpTTAAdapter
from .lewm_optta.objective import Objective


from .prbd_paths import PHASE2 as DATA, output_path
OUT = DATA / "baselines"
GRU_WIDTH = 192
GRU_PARAMETERS = 3 * GRU_WIDTH ** 2 + 1350 * GRU_WIDTH + 192
SHADOW = {"horizon": 4, "num_samples": 16, "n_steps": 3, "topk": 4,
          "var_scale": 1., "objective_base": 2., "warm_start": False}


def write_json(path, value):
    output_path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def rollout(host, history, past, candidates):
    """Free native H-step rollout; past is the two already executed controls."""
    batch, count, horizon, dimension = candidates.shape
    window = history[:, None].expand(-1, count, -1, -1).reshape(-1, 3, dimension)
    before = past[:, None].expand(-1, count, -1, -1)
    controls = torch.cat((before, candidates), dim=2).flatten(0, 1)
    predictions = []
    for step in range(horizon):
        prediction = host.predict(window, controls[:, step:step + 3])[:, -1]
        predictions.append(prediction)
        window = torch.cat((window[:, 1:], prediction[:, None]), dim=1)
    return torch.stack(predictions, dim=1).unflatten(0, (batch, count))


def gru_training_loss(model, z, a, lengths):
    """Equal episodes/roots; equal legal H1..4, each H averages its full prefix."""
    batch, steps, dimension = a.shape
    states, hidden = [], None
    for root in range(steps):
        if root >= 2:
            states.append(hidden)
        hidden = model.observe(hidden, z[:, root], a[:, root])
    roots = torch.arange(2, steps, device=z.device)
    count = len(roots)
    state = torch.stack(states, dim=1).flatten(0, 1)
    current = z[:, 2:steps].flatten(0, 1)
    errors = []
    for offset in range(4):
        source = (roots + offset).clamp(max=steps - 1)
        target = (roots + offset + 1).clamp(max=steps)
        state = model.observe(state, current, a[:, source].flatten(0, 1))
        current = current + model.readout(state)
        errors.append((current.reshape(batch, count, dimension) - z[:, target]).square().mean(-1))
    errors = torch.stack(errors, dim=-1)
    horizon = torch.arange(1, 5, device=z.device)
    legal = roots[None, :, None] + horizon <= lengths[:, None, None]
    prefix = errors.cumsum(-1) / horizon
    root_loss = (prefix * legal).sum(-1) / legal.sum(-1).clamp_min(1)
    root_valid = roots[None] < lengths[:, None]
    return ((root_loss * root_valid).sum(-1) / root_valid.sum(-1)).mean()


def fit_gru(device):
    episodes = load_episodes(DATA / "train.pt", device)
    maximum = max(len(a) for _, _, a in episodes)
    z = torch.zeros(len(episodes), maximum + 1, 192, device=device)
    a = torch.zeros(len(episodes), maximum, 192, device=device)
    lengths = torch.tensor([len(row[2]) for row in episodes], device=device)
    for i, (_, zi, ai) in enumerate(episodes):
        z[i, :len(zi)], a[i, :len(ai)] = zi, ai
    torch.manual_seed(BUDGET["seed"])
    model = GRUDynamics(192, 192, GRU_PARAMETERS).to(device)
    if model.cell.hidden_size != GRU_WIDTH:
        raise AssertionError("The prespecified plain GRU width changed")
    optimizer = torch.optim.Adam(model.parameters(), lr=BUDGET["learning_rate"])
    generator = torch.Generator().manual_seed(BUDGET["seed"])
    losses = []
    synchronize(device)
    started = perf_counter()
    for update in range(1, BUDGET["updates"] + 1):
        if update in BUDGET["decay_updates"]:
            for group in optimizer.param_groups:
                group["lr"] *= .1
        rows = torch.randperm(len(episodes), generator=generator)[:BUDGET["batch_episodes"]].to(device)
        loss = gru_training_loss(model, z[rows], a[rows], lengths[rows])
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite plain-GRU loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        if update % 128 == 0:
            print("GRU", update, np.mean(losses[-128:]), flush=True)
    synchronize(device)
    seconds = perf_counter() - started
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    torch.save({"state": state, "parameters": GRU_PARAMETERS, "hidden": GRU_WIDTH,
                "budget": BUDGET, "training_seeds": [seed for seed, _, _ in episodes]}, output_path(OUT / "gru.pt"))
    write_json(OUT / "gru_training.json", {"budget": BUDGET, "hidden": GRU_WIDTH,
               "parameters": GRU_PARAMETERS, "training_seconds": seconds, "losses": losses,
               "training_seeds": [seed for seed, _, _ in episodes],
               "objective": "equal episodes/roots/legal H; each H mean prefix latent MSE",
               "pretrained_prior": False, "weight_updates_at_deployment": 0,
               "parameter_or_FLOP_matching_claimed": False})


class ShadowCost:
    """Native objective and a bridge from normalized CEM actions to 3-frame LeWM."""

    def __init__(self, host):
        self.host, self.last = host, None
        self.objective = Objective(SHADOW["objective_base"])
        self.step = 0

    def parameters(self):
        return self.host.parameters()

    def get_cost(self, info, normalized_candidates):
        batch, count = normalized_candidates.shape[:2]
        embedded = self.host.action_encoder(normalized_candidates.flatten(0, 1)).unflatten(0, (batch, count))
        history, past = info["history"][:, 0], info["past"][:, 0]
        goal = info["goal"][:, 0]
        prediction = rollout(self.host, history, past, embedded)
        path = torch.cat((history[:, None, -1:].expand(-1, count, -1, -1), prediction), dim=2)
        cost = self.objective(path, goal[:, None], self.step)
        self.last = OnlineAdapter.capture(history, past, embedded, goal[:, None], cost, replan=self.step)
        return cost

    def score_population(self, context, pool, elites):
        if pool != "all":
            raise ValueError("This frozen comparison uses the authors' full final CEM population")
        prediction = rollout(self.host, context.emb, context.past, context.candidates)
        current = context.emb[:, None, -1:].expand(-1, prediction.size(1), -1, -1)
        return self.objective(torch.cat((current, prediction), dim=2), context.goal_emb, context.step).mean()


class LatentAdapter(OpTTAAdapter):
    """The shared frozen encoder permits the author's adapter to consume cached latents."""

    def _encode(self, frames, actions):
        return frames, actions


def build_online(host, method, steps):
    cost = ShadowCost(host)
    config = SimpleNamespace(lr=5e-5, steps=1, buffer_size=5, finetune_encoder=False, encoder_lr=None,
        optimism=SimpleNamespace(rho=float(method == "optta"), schedule="increasing",
            schedule_steps=steps, pool="all", elites=0, ratio_eps=1e-12))
    return LatentAdapter(host, cost, config), cost


def build_solver(cost, seed, device):
    if os.environ.get("PRBD_ENV_SOURCE"):
        sys.path.insert(0, os.environ["PRBD_ENV_SOURCE"])
    from gymnasium.spaces import Box
    from stable_worldmodel.planning.solver.cem import CEMSolver
    solver = CEMSolver(cost, batch_size=1, device=device, seed=seed,
        **{key: SHADOW[key] for key in ("num_samples", "var_scale", "n_steps", "topk")})
    solver.configure(action_space=Box(-1., 1., shape=(1, 2), dtype=np.float32), n_envs=1,
                     config=SimpleNamespace(horizon=SHADOW["horizon"], action_block=5))
    return solver


def shadow_plan(solver, cost, z, a, goal, root, device):
    cost.step = root - 2
    synchronize(device)
    started = perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        solver.solve({"history": z[root - 2:root + 1][None], "past": a[root - 2:root][None],
                      "goal": goal[None]})
    synchronize(device)
    return cost.last, perf_counter() - started


def query(model, method, z, a, root, candidates, hidden=None):
    with torch.no_grad():
        if method == "gru":
            return model.rollout(hidden, z[root][None], candidates[None])[0]
        return rollout(model, z[root - 2:root + 1][None], a[root - 2:root][None], candidates[None])[0]


def evaluate(method, device, shadow_frozen=False):
    model = load_host(device) if method != "gru" else GRUDynamics(192, 192, GRU_PARAMETERS).to(device)
    if method == "gru":
        model.load_state_dict(torch.load(OUT / "gru.pt", weights_only=True, map_location=device)["state"])
    model.eval()
    online = method in ("adajepa", "optta")
    adapter, cost = build_online(model, method, 30) if online else (None, None)
    needs_goals = online or shadow_frozen
    if shadow_frozen:
        cost = ShadowCost(model)
    metadata = {"method": method, "device": str(device), "shadow_budget": SHADOW if needs_goals else None,
                "episodes": [], "adapter_records": [], "source_adapter":
                "https://github.com/akshaymete/optta/blob/main/lewm/code/lewm_optta/adapter.py" if online else None}
    for split in ("static", "switch"):
        batch = torch.load(DATA / f"{split}.pt", map_location=device, weights_only=True)
        goals = torch.load(DATA / f"goals_{split}.pt", weights_only=True, map_location=device) if needs_goals else None
        predictions = torch.full((len(batch["seed"]), 2, 2, 4, 192) if split == "static" else
                                 (len(batch["seed"]), 2, batch["a"].size(-2), 192), float("nan"))
        for index, seed in enumerate(batch["seed"].tolist()):
            goal = None
            if goals is not None:
                goal_seeds = goals["seed"].tolist()
                goal = goals["z_goal"][goal_seeds.index(seed)]
            for stream in range(2):
                z = batch["z"][index, stream, 0] if split == "static" else batch["z"][index, stream]
                a = batch["a"][index, stream, 0] if split == "static" else batch["a"][index, stream]
                length = int(batch["lengths"][index, stream].min()) if split == "static" else int(batch["lengths"][index, stream])
                last = min(6, length) if split == "static" else length
                # Paired speed streams share stochastic computation, while
                # independent reset seeds remain independent replicates.
                torch.manual_seed(seed)
                row = {"split": split, "seed": seed, "stream": stream, "updates": 0,
                       "shadow_seconds": 0., "adaptation_seconds": 0., "prediction_seconds": 0.,
                       "state_seconds": 0., "state_updates": 0,
                       "shadow_solves": 0, "positive_optimism_steps": 0}
                if adapter is not None:
                    adapter.reset()
                    # Two initial transitions provide the same factual prefix;
                    # adaptation starts after the first three-frame issuance.
                    for initial in range(min(2, length)):
                        adapter.add_segment(z[None, initial:initial + 2], a[None, initial:initial + 1])
                solver = build_solver(cost, seed, device) if goals is not None and length >= 3 else None
                hidden = None
                if method == "gru":
                    synchronize(device)
                    started = perf_counter()
                    with torch.no_grad():
                        for t in range(min(2, length)):
                            hidden = model.observe(hidden, z[t][None], a[t][None])
                    synchronize(device)
                    row["state_seconds"] += perf_counter() - started
                    row["state_updates"] += min(2, length)
                for root in range(2, last):
                    if split == "switch":
                        synchronize(device)
                        started = perf_counter()
                        predicted = query(model, method, z, a, root, a[None, root:root + 1], hidden)
                        synchronize(device)
                        row["prediction_seconds"] += perf_counter() - started
                        predictions[index, stream, root] = predicted[0, 0].cpu()
                    context = None
                    if solver is not None:
                        context, elapsed = shadow_plan(solver, cost, z, a, goal, root, device)
                        row["shadow_seconds"] += elapsed
                        row["shadow_solves"] += 1
                    if adapter is not None:
                        # context was formed solely from z[:root+1] before z[root+1] is admitted.
                        adapter.add_segment(z[None, root:root + 2], a[None, root:root + 1])
                        synchronize(device)
                        started = perf_counter()
                        records = adapter.step(context, episode=seed * 2 + stream, replan=root - 1)
                        if any(record["n_frames"] > adapter.buffer_size + 1 for record in records):
                            raise AssertionError("The author's five-segment buffer exceeded five actual transitions")
                        synchronize(device)
                        row["adaptation_seconds"] += perf_counter() - started
                        row["updates"] += len(records)
                        row["positive_optimism_steps"] += sum(record["alpha"] is not None and record["alpha"] > 0 for record in records)
                        metadata["adapter_records"].append({"split": split, "seed": seed, "stream": stream,
                                                            "root": root, "records": records})
                    if method == "gru":
                        synchronize(device)
                        started = perf_counter()
                        with torch.no_grad():
                            hidden = model.observe(hidden, z[root][None], a[root][None])
                        synchronize(device)
                        row["state_seconds"] += perf_counter() - started
                        row["state_updates"] += 1
                if split == "static" and length >= 6:
                    root = 6
                    synchronize(device)
                    started = perf_counter()
                    for program in range(2):
                        horizon = min(4, int(batch["lengths"][index, stream, program]) - root)
                        if horizon > 0:
                            actions = batch["a"][index, stream, program, root:root + horizon][None]
                            predicted = query(model, method, z, a, root, actions, hidden)
                            predictions[index, stream, program, :horizon] = predicted[0].cpu()
                    synchronize(device)
                    row["prediction_seconds"] += perf_counter() - started
                metadata["episodes"].append(row)
            if (index + 1) % 8 == 0:
                print(method, split, index + 1, flush=True)
        torch.save({"seed": batch["seed"].cpu(), "predictions": predictions, "method": method}, output_path(OUT / f"{split}_{method}.pt"))
        write_json(OUT / f"timing_{method}.json", metadata)
    if adapter is not None:
        metadata["adapted_parameters"] = sum(p.numel() for p in adapter.pred_params)
        metadata["online_settings"] = {"lr": adapter.lr, "steps": adapter.steps, "buffer_size": adapter.buffer_size,
            "rho": adapter.rho, "schedule": adapter.schedule, "schedule_steps": adapter.K,
            "encoder_finetuned": adapter.finetune_encoder, "resets_every_seed_stream": True,
            "initial_factual_prefix": 2, "initial_prefix_buffer_segments": 2,
            "first_adapted_transition_source": 2}
        adapter.reset()
    metadata["GPU_shared_with_PRBD_training"] = True
    metadata["shadow_rng_seed_recipe"] = "seed; paired initial-speed streams share stochastic computations"
    metadata["timing_summary"] = {
        split: {field: sum(row[field] for row in metadata["episodes"] if row["split"] == split)
                for field in ("shadow_seconds", "prediction_seconds", "adaptation_seconds",
                              "state_seconds", "state_updates", "shadow_solves", "updates",
                              "positive_optimism_steps")}
        for split in ("static", "switch")}
    write_json(OUT / f"timing_{method}.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit_gru", "frozen", "gru", "adajepa", "optta"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shadow-frozen", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "fit_gru":
        fit_gru(args.device)
    else:
        evaluate(args.stage, args.device, args.shadow_frozen)


if __name__ == "__main__":
    main()
