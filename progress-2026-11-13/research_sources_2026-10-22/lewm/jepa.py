"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from module import NativeResponseMixture, validate_first_block_memory

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb, first_block_memory=None, memory_site=None):
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        if first_block_memory is None and memory_site is None:
            preds = self.predictor(emb, act_emb)
        else:
            preds = self.predictor(emb, act_emb, first_block_memory=first_block_memory,
                                   memory_site=memory_site)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    def enable_response_mixture(self, latent_scale=None, sigma_bounds=None):
        """Attach the trainable §16 operation after loading the native checkpoint."""
        if hasattr(self, "response_mixture"):
            raise ValueError("The response mixture is already installed")
        self.response_mixture = NativeResponseMixture(
            self.predictor.transformer, self.pred_proj, latent_scale, sigma_bounds).train(self.training)
        return self.response_mixture

    def predict_response(self, emb, act_emb, log_weights, *, factual=False):
        """One last-token prediction, an unobserved advance, and optional factual evidence."""
        if not hasattr(self, "response_mixture"):
            raise ValueError("Install the native response mixture before using its state")
        if self.training:
            raise ValueError("Use eval mode for response prediction, including during offline fitting")
        log_weights = self.response_mixture.prepare_state(log_weights, len(emb))
        return self._predict_response(emb, act_emb, log_weights, factual=factual)

    def _predict_response(self, emb, act_emb, log_weights, *, factual=False):
        inputs = self.predictor(emb, act_emb, response_inputs=True)
        return self.response_mixture(inputs, self.predictor.transformer, self.pred_proj,
                                     log_weights, factual=factual)

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3,
                first_block_memory=None, memory_site=None, *, response_log_weights=None):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        if response_log_weights is not None:
            if not hasattr(self, "response_mixture"):
                raise ValueError("Install the native response mixture before using its state")
            if self.training:
                raise ValueError("Use eval mode for response rollout")
            if first_block_memory is not None or memory_site is not None:
                raise ValueError("Use one native regulation path per rollout")
            response_log_weights = self.response_mixture.prepare_state(response_log_weights, B)
            response_log_weights = response_log_weights.repeat_interleave(S, dim=0)
        if first_block_memory is not None and (B < 1 or S < 1 or H < 1):
            raise ValueError('Native memory rollout requires nonempty batch, candidates, and history')
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        validate_first_block_memory(first_block_memory, memory_site, _init["emb"])
        memory_kwargs = {}
        if first_block_memory is not None:
            memory_kwargs = {
                'first_block_memory': first_block_memory[:, None].expand(-1, S, -1).reshape(B * S, 192),
                'memory_site': memory_site,
            }
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # The last slot of act_0 is the first command to predict.
        HS = history_size
        for t in range(T - H + 1):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            if response_log_weights is None:
                pred_emb = self.predict(emb_trunc, act_trunc, **memory_kwargs)[:, -1:]
            else:
                next_z, response_log_weights, _ = self._predict_response(
                    emb_trunc, act_trunc, response_log_weights)
                pred_emb = next_z[:, None]
            emb = torch.cat([emb, pred_emb], dim=1)
            if t < T - H:
                act = torch.cat([act, act_future[:, t:t + 1]], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"
        # One state per real root. Keep it outside candidate/goal tensor slicing.
        response_log_weights = info_dict.get("response_log_weights")
        if response_log_weights is not None:
            if self.training or not hasattr(self, "response_mixture"):
                raise ValueError("Install the response mixture and use eval mode before scoring")
            info_dict = {k: v for k, v in info_dict.items() if k != "response_log_weights"}

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"][:, None]
        if response_log_weights is None:
            info_dict = self.rollout(info_dict, action_candidates)
        else:
            info_dict = self.rollout(info_dict, action_candidates,
                                     response_log_weights=response_log_weights.to(device))

        cost = self.criterion(info_dict)

        return cost
