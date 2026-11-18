"""Matched FourRoom planning pilot on new layouts after the native Oracle stage."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from .native_full_depth_conditioning import advance_log_support, write_observation
from .native_topology_condition import ROOT, RUN, forecast, load_readout, load_trained
from .physical_topology_gate import (
    ACTION_BLOCK, CENTER, DEADLINE, ELITES, GOAL_RADIUS, ITERATIONS, SAMPLES,
    ReversibleFourRoom, VisibleMapCost, block, cem_population, make_scene, probe_and_dormancy,
)

sys.path.insert(0, str(ROOT / "runs/native_full_depth_mpc_optta_20261115/upstream"))
from lewm_optta.adapter import OpTTAAdapter
from lewm_optta.objective import PlanContext


SEEDS = tuple(range(2026111900, 2026111908))
HORIZON = 5
SWITCH = .01
ARMS = ("stock", "native", "uniform", "keep_old", "persistent", "oracle", "adajepa", "optta")
ADAPTIVE = ("adajepa", "optta")
MODELED = ("uniform", "keep_old", "persistent", "oracle")


class TorchVisibleMapCost:
    """Use the exact visible paths from the physical scorer and differentiate their lengths."""

    def __init__(self, score, device):
        points, residual = zip(*[list(zip(*choices)) for choices in score.paths.values()])
        self.first = torch.as_tensor(np.asarray(points), dtype=torch.float32, device=device)
        self.residual = torch.as_tensor(np.asarray(residual), dtype=torch.float32, device=device)

    def __call__(self, xy):
        right, bottom = (xy >= CENTER).unbind(-1)
        room_index = torch.where(bottom, torch.where(right, 2, 3), right.long())
        candidate_cost = torch.linalg.vector_norm(xy[..., None, :] - self.first[room_index], dim=-1)
        return (candidate_cost + self.residual[room_index]).min(dim=-1).values


@dataclass
class TopologyPlanContext(PlanContext):
    past: torch.Tensor


class ForecastCost:
    def __init__(self, host, conditioner, gate, decode, geometry):
        self.host = host
        self.conditioner = conditioner
        self.gate = gate
        self.decode = decode
        self.geometry = geometry

    def terminal_xy(self, history, past, raw_plans):
        roll = forecast(self.host, self.conditioner, self.gate, history, past, raw_plans,
                        compact=self.conditioner is not None)
        return self.decode(roll[..., -1, :])

    def score(self, history, past, raw_plans):
        return self.geometry(self.terminal_xy(history, past, raw_plans))

    def score_population(self, ctx, pool, _elites):
        if pool != "all":
            raise ValueError("Expected OpTTA optimism on the whole final population")
        return self.score(ctx.emb, ctx.past, ctx.candidates).mean()


def build_adapter(host, cost, arm):
    if arm not in ADAPTIVE:
        return None
    config = SimpleNamespace(
        lr=5e-5, steps=1, buffer_size=5, finetune_encoder=False, encoder_lr=None,
        optimism=SimpleNamespace(
            rho=1. if arm == "optta" else 0., schedule="increasing", schedule_steps=DEADLINE,
            pool="all", elites=ELITES, ratio_eps=1e-12,
        ),
    )
    return OpTTAAdapter(host, cost, config)


class RealHistory:
    def __init__(self, cost, arm, scene, goal, scale):
        self.cost = cost
        self.host = cost.host
        self.decode = cost.decode
        self.arm = arm
        self.scene = scene
        self.goal = goal
        self.device = self.host.action_mean.device
        self.means = torch.tensor((.485, .456, .406), device=self.device)[:, None, None]
        self.scales = torch.tensor((.229, .224, .225), device=self.device)[:, None, None]
        self.scale = scale
        self.adapter = build_adapter(self.host, self.cost, arm)
        self.log_support = torch.full((1, 2), -np.log(2), device=self.device, dtype=torch.float32)
        self.previous_mode = None
        self.frames, self.latents, self.actions = [], [], []
        self.pending = None
        self.context = None
        self.events, self.adaptation = [], []

    @torch.no_grad()
    def encode(self, rgb):
        frame = (torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float()
                 / 255 - self.means) / self.scales
        latent = self.host.encode({"pixels": frame[None, None]})["emb"][0, 0]
        return frame, latent

    def begin(self, initial_rgb):
        frame, latent = self.encode(initial_rgb)
        self.frames.append(frame)
        self.latents.append(latent)

    def window(self):
        if len(self.latents) < 3:
            raise ValueError("The native forecaster requires three arrived frames")
        return torch.stack(self.latents[-3:])[None], torch.stack(self.actions[-2:])[None]

    @torch.no_grad()
    def issue(self, raw_command):
        if self.pending is not None:
            raise ValueError("A previous actual control has no received observation")
        action = torch.as_tensor(raw_command, device=self.device, dtype=torch.float32)
        prior, issued = None, None
        if self.arm == "persistent" or (self.arm == "keep_old" and self.previous_mode is None):
            prior = advance_log_support(self.log_support, SWITCH)
            if len(self.latents) >= 3:
                history, past = self.window()
                roll = forecast(self.host, self.cost.conditioner, self.cost.gate, history, past,
                                action[None, None, None], compact=True)
                issued = roll[:, 0, :, 0]
        self.pending = action, prior, issued

    def arrive(self, rgb):
        if self.pending is None:
            raise ValueError("Unexpected arrival without a dispatched actual control")
        frame, latent = self.encode(rgb)
        action, prior, issued = self.pending
        likelihood = None
        if prior is not None:
            self.log_support = prior
            if issued is not None:
                self.log_support, likelihood = write_observation(prior, issued, latent[None], self.scale)
        if self.adapter is not None:
            factual = torch.stack((self.frames[-1], frame))[None]
            normalized = ((action - self.host.action_mean) / self.host.action_std).float().flatten()
            self.adapter.add_segment(factual, normalized[None, None])
            started = time.perf_counter()
            rows = self.adapter.step(self.context, episode=self.scene.seed, replan=len(self.actions) + 1)
            self.adaptation.append({"elapsed_seconds": time.perf_counter() - started, "steps": rows,
                                    "had_previous_plan": self.context is not None})
        self.events.append({
            "posterior": self.log_support.exp()[0].tolist() if prior is not None else None,
            "log_likelihood": likelihood[0].tolist() if likelihood is not None else None,
        })
        self.frames.append(frame)
        self.latents.append(latent)
        self.actions.append(action)
        self.pending = None
        self.context = None

    def seal_old_mode(self):
        if self.arm == "keep_old":
            self.previous_mode = int(self.log_support.argmax(dim=-1).item())

    @torch.no_grad()
    def choose(self, rng, index, *, oracle_mode=None):
        if (oracle_mode is not None) != (self.arm == "oracle"):
            raise ValueError("Only the learned Oracle receives the actual current condition")
        history, past = self.window()
        current = self.decode(history[0, -1]).cpu().numpy()
        goal_xy = torch.as_tensor(self.goal, device=self.device)
        terminal_xy = []

        def cost(plans):
            raw = torch.from_numpy(np.repeat(plans[:, :, None], ACTION_BLOCK, axis=2)).to(self.device)[None]
            endpoints = self.cost.terminal_xy(history, past, raw)
            scores = self.cost.geometry(endpoints)
            if self.adapter is not None:
                self.context = TopologyPlanContext(history, goal_xy, raw, scores[..., 0], index, past)
            by_mode = scores[0].transpose(0, 1)
            terminal_xy.append(endpoints[0].cpu().numpy())
            return by_mode.expand(2, -1).cpu().numpy()

        proposals, scores = cem_population(self.scene, self.goal, current, HORIZON, rng, cost)
        if self.arm == "oracle":
            weights = np.eye(2)[oracle_mode]
        elif self.arm == "keep_old":
            if self.previous_mode is None:
                raise ValueError("Seal the old evidence before the physical switch")
            weights = np.eye(2)[self.previous_mode]
        elif self.arm == "persistent":
            weights = self.log_support.exp()[0].cpu().numpy()
        else:
            weights = np.array([.5, .5])
        ranked = weights @ scores
        best = int(ranked.argmin())
        return block(proposals[best, 0]), {
            "decoded_current_xy": current.tolist(), "selected_candidate": best,
            "selected_h5_controls": proposals[best].tolist(),
            "predicted_terminal_xy": np.concatenate(terminal_xy)[best].tolist(),
            "weights": weights.tolist(), "chosen_mode_cost": scores[:, best].tolist(),
            "selected_cost": float(ranked[best]), "candidate_proposals": len(proposals),
        }


class ObservedFourRoom(ReversibleFourRoom):
    """Deliver actual action and observation callbacks around the canonical execution path."""

    def __init__(self, scene, history):
        super().__init__(scene)
        self.history = history

    def execute(self, commands):
        self.history.issue(commands)
        actual = super().execute(commands)
        self.history.arrive(self.render())
        return actual


def switched_episode(scene, initial_mode, arm, host, conditioner, gate, decode, scale):
    started = time.perf_counter()
    start, target = scene.northwest, scene.southeast
    cost = ForecastCost(host, conditioner, gate, decode, TorchVisibleMapCost(
        VisibleMapCost(scene, target), host.action_mean.device,
    ))
    history = RealHistory(cost, arm, scene, target, scale)
    env = ObservedFourRoom(scene, history)
    try:
        env.begin(start, target, initial_mode)
        history.begin(env.render())
        prefix = []
        old, _ = probe_and_dormancy(env, start, prefix, "old")
        history.seal_old_mode()
        switch = len(prefix)
        env.set_mode(1 - initial_mode)
        revised, _ = probe_and_dormancy(env, start, prefix, "reversed")

        ticks = []
        for index in range(DEADLINE):
            rng = np.random.default_rng(np.random.SeedSequence([scene.seed, index]))
            mode = env.mode if arm == "oracle" else None
            commands, choices = history.choose(rng, index, oracle_mode=mode)
            real_path = env.execute(commands)
            distance = float(np.linalg.norm(real_path - target, axis=-1).min())
            ticks.append({**choices, "actual_raw_xy": real_path.tolist(), "raw_action": commands.tolist(),
                          "best_physical_distance": distance})
            if distance < GOAL_RADIUS:
                break
        best_distance = min(tick["best_physical_distance"] for tick in ticks)
        return {
            "success": best_distance < GOAL_RADIUS, "best_physical_distance": best_distance,
            "blocks": len(ticks), "old_evidence": old, "reversed_evidence": revised,
            "switch_index": switch, "old_inferred_mode": history.previous_mode,
            "scripted_prefix": prefix, "actual_observation_events": history.events,
            "adaptation": history.adaptation, "ticks": ticks, "seconds": time.perf_counter() - started,
        }
    finally:
        if history.adapter is not None:
            history.adapter.reset()
        env.close()


def verify_geometry():
    """Policy-independent check of the replacement scorer and CEM budget."""
    scene = make_scene(2026111401)
    reference = VisibleMapCost(scene, scene.southeast)
    score = TorchVisibleMapCost(reference, "cpu")
    rng = np.random.default_rng(112)
    positions = rng.uniform(17, 207, size=(2, 96, 2)).astype(np.float32)
    positions[0, :4] = [(112, 112), (112, 75), (151, 112), (112, 150)]
    xy = torch.tensor(positions, requires_grad=True)
    result = score(xy)
    np.testing.assert_allclose(result.detach().numpy(), reference(positions), rtol=1e-6, atol=1e-5)
    result.mean().backward()
    if xy.grad is None or not torch.isfinite(xy.grad).all():
        raise AssertionError("Visible map cost must admit finite optimism gradients")

    per_round = []

    def synthetic_cost(plans):
        per_round.append(plans.shape)
        score = np.square(plans).sum(axis=(1, 2))
        return np.stack((score, score))

    plans, costs = cem_population(scene, scene.southeast, scene.northwest, HORIZON, rng, synthetic_cost)
    assert per_round == [(SAMPLES, HORIZON, 2)] * ITERATIONS
    assert plans.shape == (SAMPLES * ITERATIONS, HORIZON, 2)
    np.testing.assert_array_equal(costs[0], costs[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=ARMS)
    parser.add_argument("--part", choices=("smoke", "remaining", "all"), default="all")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    verify_geometry()
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    decode = load_readout(device, run=args.run)
    scale = None
    if {"persistent", "keep_old"}.intersection(args.arms):
        scale = torch.load(args.run / "likelihood_scale.pt", map_location=device, weights_only=True)
        if scale.shape != (192,) or not torch.isfinite(scale).all() or not (scale > 0).all():
            raise ValueError("Expected the fixed held-apart per-coordinate likelihood scale")

    seeds = {"smoke": SEEDS[:1], "remaining": SEEDS[1:], "all": SEEDS}[args.part]
    settings = {"scope": "paired switched-root leg", "seeds": seeds, "horizon": HORIZON,
                "deadline": DEADLINE, "CEM_rounds": ITERATIONS, "CEM_candidates": SAMPLES,
                "elites_per_component": ELITES, "factual_switch_prior": SWITCH,
                "AdaJEPA_OpTTA": {"rho": [0., 1.], "lr": 5e-5, "buffer": 5, "steps": 1}}
    rows = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as out:
        out.write(json.dumps({"settings": settings}) + "\n")
        for arm in args.arms:
            checkpoint = "conditioned" if arm in MODELED else "native" if arm in ADAPTIVE else arm
            host, conditioner, gate = load_trained(args.run, checkpoint, device)
            for seed in seeds:
                scene = make_scene(seed)
                for initial_mode in (0, 1):
                    torch.manual_seed(seed)
                    trace = switched_episode(scene, initial_mode, arm, host, conditioner, gate, decode, scale)
                    row = {"seed": seed, "initial_mode": initial_mode, "arm": arm, **trace}
                    out.write(json.dumps(row, allow_nan=False) + "\n")
                    out.flush()
                    rows.append(row)
                    print(arm, seed, initial_mode, trace["success"], flush=True)
            del host, conditioner, gate

    summary = {}
    for arm in args.arms:
        records = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            "successes": sum(row["success"] for row in records), "episodes": len(records),
            "mean_best_physical_distance": float(np.mean([row["best_physical_distance"] for row in records])),
            "mean_planning_blocks": float(np.mean([row["blocks"] for row in records])),
        }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps({"settings": settings, "results": summary}, indent=2), encoding="utf-8",
    )


if __name__ == "__main__":
    main()
