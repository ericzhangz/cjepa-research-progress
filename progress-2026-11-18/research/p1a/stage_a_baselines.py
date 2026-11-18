"""Predictor-only test-time adaptation and a plain recurrent predictor for Stage A."""

import copy
from dataclasses import dataclass
import math
import time
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from runs.native_full_depth_mpc_optta_20261115.upstream.lewm_optta.adapter import OpTTAAdapter
from runs.native_full_depth_mpc_optta_20261115.upstream.lewm_optta.objective import PlanContext


def predictor_parameter_count(host):
    """Count the native predictor and prediction projection, excluding the shared encoders."""
    return sum(parameter.numel() for module in (host.predictor, host.pred_proj)
               for parameter in module.parameters())


@dataclass
class CapturedPlan(PlanContext):
    past: torch.Tensor


class OnlineAdapter(OpTTAAdapter):
    """Apply the authors' LeWM predictor-only AdaJEPA or OpTTA objective in latent space."""

    def __init__(self, host, rollout, plan_cost, *, method, replans):
        if method not in ("adajepa", "optta") or replans < 1:
            raise ValueError("Expected AdaJEPA or OpTTA and a positive replanning budget")
        if any(parameter.requires_grad for parameter in host.parameters()):
            raise ValueError("Expected a frozen host before selecting the online adaptation parameters")

        def score_population(ctx, pool, _elites):
            if pool != "all":
                raise ValueError("OpTTA uses every candidate from the final CEM iteration")

            def plan_loss():
                trajectory = rollout(host, ctx.emb, ctx.past, ctx.candidates)
                return plan_cost(trajectory, ctx.goal_emb).mean()

            return self._time_optimism("planning_forward_seconds", plan_loss)

        config = SimpleNamespace(
            lr=5e-5, steps=1, buffer_size=5, finetune_encoder=False, encoder_lr=None,
            optimism=SimpleNamespace(
                rho=1.0 if method == "optta" else 0.0, schedule="increasing", schedule_steps=replans,
                pool="all", elites=0, ratio_eps=1e-12,
            ),
        )
        super().__init__(host, SimpleNamespace(score_population=score_population), config)
        self.method = method
        self._rollout = rollout
        self._plan_cost = plan_cost
        self._episode = None
        self._optimism_time = {}

    def _encode(self, latents, actions):
        return latents, actions

    def forecast(self, history, past, candidates):
        return self._rollout(self.model, history, past, candidates)

    def _time_optimism(self, field, operation, *args):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = operation(*args)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._optimism_time[field] = time.perf_counter() - started
        return result

    def optimism_gradient(self, loss_pred, loss_opt):
        """Time the upstream two-gradient calculation, including prediction and planning."""
        return self._time_optimism("balanced_gradients_seconds", super().optimism_gradient, loss_pred, loss_opt)

    @torch.no_grad()
    @torch.inference_mode(False)
    def fork(self):
        """Copy the live predictor and replay state for an independent diagnostic branch."""
        host = copy.deepcopy(self.model).eval().requires_grad_(False)
        branch = OnlineAdapter(host, self._rollout, self._plan_cost,
                               method=self.method, replans=self.K)
        for (_, saved), (_, branch_saved) in zip(self._snapshot, branch._snapshot, strict=True):
            branch_saved.copy_(saved)
        branch.buffer = [(latents.clone(), actions.clone()) for latents, actions in self.buffer]
        branch.n_updates = self.n_updates
        branch._episode = self._episode
        return branch

    def reset(self):
        super().reset()
        self._episode = None

    @staticmethod
    @torch.inference_mode(False)
    def capture(history, past, candidates, goal, scores, *, replan):
        """Keep the evaluated final population and its factual root for the next real arrival."""
        return CapturedPlan(
            emb=history.detach().clone(), goal_emb=goal.detach().clone(), candidates=candidates.detach().clone(),
            costs=scores.detach().clone(), step=replan, past=past.detach().clone(),
        )

    @torch.inference_mode(False)
    def arrive(self, before, action, after, *, episode, replan, plan=None):
        """Consume one actual transition with latents [1,D] and issued action embeddings [1,A]."""
        if (before.ndim != 2 or before.shape != after.shape or before.size(0) != 1
                or action.ndim != 2 or action.size(0) != 1):
            raise ValueError("An online adapter follows one factual episode at a time")
        if self._episode is not None and episode != self._episode:
            raise ValueError("Reset the adapter before following a new factual episode")
        self.add_segment(torch.stack((before.detach(), after.detach()), dim=1), action.detach()[:, None].clone())
        self._episode = episode
        self._optimism_time.clear()
        records = self.step(plan, episode=episode, replan=replan)
        for record in records:
            record.update(self._optimism_time)
        return records

    @property
    def stats(self):
        return {
            "method": self.method, "lr": self.lr, "steps_per_update": self.steps, "buffer_size": self.buffer_size,
            "rho": self.rho, "schedule": self.schedule, "schedule_steps": self.K, "pool": self.pool,
            "ratio_eps": self.eps, "adapted_parameters": sum(parameter.numel() for parameter in self.pred_params),
        }


class GRUDynamics(nn.Module):
    """A plain GRU predicting latent increments from state-action tokens."""

    def __init__(self, latent_dim, action_dim, target_parameters):
        super().__init__()
        if min(latent_dim, action_dim, target_parameters) < 1:
            raise ValueError("GRU dimensions and target parameter count must be positive")
        slope = 3 * (latent_dim + action_dim) + 6 + latent_dim

        def parameter_count(width):
            return 3 * width ** 2 + slope * width + latent_dim

        lower = max(1, int((math.sqrt(slope ** 2 + 12 * max(0, target_parameters - latent_dim)) - slope) / 6))
        width = min((lower, lower + 1), key=lambda candidate: abs(parameter_count(candidate) - target_parameters))
        self.cell = nn.GRUCell(latent_dim + action_dim, width)
        self.readout = nn.Linear(width, latent_dim)
        self.target_parameters = target_parameters

    def observe(self, hidden, latent, action):
        """Advance factual state once for the source latent and action of an arrived transition."""
        return self.cell(torch.cat((latent, action), dim=-1), hidden)

    def rollout(self, hidden, latent, candidates):
        """Independently roll out [batch,candidate,horizon,action] from one live state per batch row."""
        batch, count, horizon, _ = candidates.shape
        if not count or not horizon or latent.size(0) != batch:
            raise ValueError("Expected nonempty candidate programs at one root per batch row")
        state = None if hidden is None else hidden.repeat_interleave(count, dim=0)
        current = latent.repeat_interleave(count, dim=0)
        actions = candidates.flatten(0, 1)
        trajectory = []
        for step in range(horizon):
            state = self.observe(state, current, actions[:, step])
            current = current + self.readout(state)
            trajectory.append(current)
        return torch.stack(trajectory, dim=1).unflatten(0, (batch, count))

    def training_loss(self, latents, actions, *, root, horizon):
        """Backpropagate future latent loss through the true prefix and the free rollout."""
        hidden = None
        for step in range(root):
            hidden = self.observe(hidden, latents[:, step], actions[:, step])
        endpoint = root + horizon
        predicted = self.rollout(hidden, latents[:, root], actions[:, None, root:endpoint])[:, 0]
        return F.mse_loss(predicted, latents[:, root + 1:endpoint + 1])

    @property
    def stats(self):
        return {"hidden_size": self.cell.hidden_size, "target_parameters": self.target_parameters,
                "parameters": sum(parameter.numel() for parameter in self.parameters())}


def _check_archive(latents, actions, batch_size, horizon):
    if (latents.ndim != 3 or actions.ndim != 3 or latents.size(0) != actions.size(0)
            or latents.size(1) != actions.size(1) + 1 or not latents.size(0)
            or min(batch_size, horizon) < 1 or horizon > actions.size(1)):
        raise ValueError("Expected a nonempty archive with one more latent than executed actions")


@torch.no_grad()
def evaluate_gru(model, latents, actions, *, batch_size, horizon):
    """Measure free forecast losses from every eligible root in held-out factual trajectories."""
    _check_archive(latents, actions, batch_size, horizon)
    device = next(model.parameters()).device
    steps = actions.size(1) - horizon + 1
    total = torch.zeros(horizon, device=device, dtype=torch.float64)
    for begin in range(0, len(latents), batch_size):
        observed = latents[begin:begin + batch_size].to(device)
        executed = actions[begin:begin + batch_size].to(device)
        hidden = None
        for root in range(steps):
            endpoint = root + horizon
            predicted = model.rollout(hidden, observed[:, root], executed[:, None, root:endpoint])[:, 0]
            error = predicted - observed[:, root + 1:endpoint + 1]
            total += error.square().sum(dim=(0, 2), dtype=torch.float64)
            hidden = model.observe(hidden, observed[:, root], executed[:, root])
    roots = steps * len(latents)
    mse = total / (roots * latents.size(-1))
    if not torch.isfinite(mse).all():
        raise FloatingPointError("Nonfinite held-out Stage A GRU loss")
    return {"trajectories": len(latents), "roots": roots, "mse": mse.mean().item(),
            "mse_by_horizon": mse.tolist()}


def fit_gru(model, latents, actions, *, updates, batch_size, horizon, seed, lr=1e-4, heldout=None):
    """Fit against complete factual latent/action trajectories from an offline fitting split."""
    _check_archive(latents, actions, batch_size, horizon)
    if updates < 1:
        raise ValueError("Expected a positive number of offline updates")
    if heldout is not None:
        _check_archive(*heldout, batch_size, horizon)
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    model.train()
    for _ in range(updates):
        rows = torch.randint(len(latents), (batch_size,), generator=generator)
        root = int(torch.randint(actions.size(1) - horizon + 1, (), generator=generator))
        endpoint = root + horizon
        loss = model.training_loss(latents[rows, :endpoint + 1].detach().to(device),
                                   actions[rows, :endpoint].detach().to(device), root=root, horizon=horizon)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite Stage A GRU training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    result = {**model.stats, "updates": updates, "batch_size": batch_size, "horizon": horizon,
              "fitting_trajectories": len(latents), "seed": seed, "lr": lr, "losses": losses}
    if heldout is not None:
        result["heldout"] = evaluate_gru(model, *heldout, batch_size=batch_size, horizon=horizon)
    return result
