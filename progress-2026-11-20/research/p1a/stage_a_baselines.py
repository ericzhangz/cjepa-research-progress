"""Executed Phase 2 GRU and plan-capture dependencies extracted from Stage A."""

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .lewm_optta.objective import PlanContext


@dataclass
class CapturedPlan(PlanContext):
    past: torch.Tensor


class OnlineAdapter:
    @staticmethod
    @torch.inference_mode(False)
    def capture(history, past, candidates, goal, scores, *, replan):
        """Keep the evaluated final population and its factual root for the next real arrival."""
        return CapturedPlan(
            emb=history.detach().clone(), goal_emb=goal.detach().clone(), candidates=candidates.detach().clone(),
            costs=scores.detach().clone(), step=replan, past=past.detach().clone(),
        )


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
