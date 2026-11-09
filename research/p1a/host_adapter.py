"""Frozen adapters for the exact P0b/P0c FourRoom LeWM and physical probes."""

from contextlib import contextmanager
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torchvision.transforms import v2

from models.probers import build_prober
from p1a_reference import latent_rollout


METHODS = ("stale_1", "refresh_1", "refresh_3")
PROBE_COLUMNS = ("xy_x", "xy_y", "distractor0_x", "distractor0_y", "min_distractor_dist")
HEAD_DIMS = {"ego_xy": 2, "distractor_xy": 2, "min_distractor_dist": 1}
ATOL = 1e-5
RTOL = 1e-5


class Timings:
    """CUDA event timings for actual module calls and synchronized wall times for stages."""

    def __init__(self):
        self.records = {}

    @contextmanager
    def stage(self, name):
        self.name = name
        self.events = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        ops = {}
        for key, first, last, units in self.events:
            stat = ops.setdefault(key, {"calls": 0, "total_ms": 0.0, "units": 0})
            stat["calls"] += 1
            stat["total_ms"] += first.elapsed_time(last)
            stat["units"] += units
        self.records[name] = {"wall_ms": elapsed * 1000, "operations": ops}

    def measure(self, operation, units, fn, *args):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        value = fn(*args)
        end.record()
        self.events.append((operation, start, end, units))
        return value


class FrozenHost:
    def __init__(self, checkpoint: Path, normalizer: Path, probe: Path):
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        object_ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = object_ckpt.get_level(1).to(self.device).eval().requires_grad_(False)
        self.model.interpolate_pos_encoding = True
        if not isinstance(self.model.action_encoder, nn.Identity):
            raise ValueError("P0b native rollout contract requires the flat identity action encoder.")

        saved_stats = torch.load(normalizer, map_location="cpu", weights_only=False)["stats"]["action"]
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.asarray(saved_stats["mean"], dtype=np.float64).reshape(-1)
        self.scaler.scale_ = np.asarray(saved_stats["std"], dtype=np.float64).reshape(-1)
        self.scaler.var_ = self.scaler.scale_ ** 2
        self.scaler.n_features_in_ = self.scaler.mean_.size
        self.scaler.n_samples_seen_ = saved_stats["count"]

        saved_probe = torch.load(probe, map_location="cpu", weights_only=False)
        if tuple(saved_probe["columns"]) != PROBE_COLUMNS:
            raise ValueError("P0c probe physical-column order differs from the saved contract.")
        self.probes = nn.ModuleDict({
            key: build_prober(input_dim=192, output_dim=dim) for key, dim in HEAD_DIMS.items()
        })
        self.probes.load_state_dict(saved_probe["heads"], strict=True)
        self.probes.to(self.device).eval().requires_grad_(False)
        self.mean = np.asarray(saved_probe["label_mean"], dtype=np.float64)
        self.std = np.asarray(saved_probe["label_std"], dtype=np.float64)
        self.preprocess = v2.Compose([
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            v2.Resize(size=224),
        ])

    def normalize(self, raw):
        if raw.shape[-2:] != (5, 2):
            raise ValueError("Expected five raw [dx,dy] actions per model block.")
        shape = raw.shape[:-2]
        return self.scaler.transform(raw.reshape(-1, 2)).reshape(*shape, 10)

    def _preprocess(self, rgb):
        # Source images are archived as uint8 NHWC by the environment collector.
        if rgb.shape[-3:] != (224, 224, 3) or rgb.dtype != np.uint8:
            raise ValueError("Expected uint8 RGB with image dimensions [224,224,3].")
        return self.preprocess(torch.from_numpy(np.ascontiguousarray(rgb)).permute(0, 3, 1, 2).to(self.device))

    def encode(self, rgb, timing, batch_size=32):
        shape = rgb.shape[:-3]
        frames = rgb.reshape(-1, *rgb.shape[-3:])
        latents = []
        for start in range(0, len(frames), batch_size):
            images = self._preprocess(frames[start:start + batch_size])
            emb = timing.measure("encoder", len(images), self.model.encode, {"pixels": images[:, None]})
            latents.append(emb["embed_0"][:, 0])
        return torch.cat(latents).reshape(*shape, 192)

    def decode(self, latents, timing, batch_size=1024):
        shape = latents.shape[:-1]
        values = []
        for batch in latents.reshape(-1, 192).split(batch_size):
            def forward(x):
                return torch.cat([self.probes[key](x) for key in HEAD_DIMS], dim=-1)
            values.append(timing.measure("three_frozen_probes", len(batch), forward, batch))
        decoded = torch.cat(values).cpu().numpy().reshape(*shape, len(PROBE_COLUMNS))
        return (decoded * self.std + self.mean).astype(np.float32)

    def rollout(self, z_history, past, future, history_size, timing, capture=None):
        # Preserve the upstream CUDA action-prefix tensor extent. The embedder
        # is tokenwise mathematically, but shorter extents change FP32 rounding.
        all_actions = torch.cat([past, future], dim=1) if past.shape[1] else future
        prefix_length = past.shape[1]

        def embed(blocks):
            nonlocal action_input, prefix_length
            action_input = blocks
            prefix_length += 1
            prefix = all_actions[:, :prefix_length]
            if prefix_length > past.shape[1] + 1:
                prefix = prefix.contiguous()
            full_embedding = timing.measure(
                "action_embed", prefix.shape[0] * prefix.shape[1],
                self.model.action_embed, self.model.action_encoder(prefix),
            )
            return full_embedding[:, -blocks.shape[1]:]

        def predict(z, a):
            result = timing.measure("predictor", z.shape[0] * z.shape[1], self.model.predict, z, a)
            if capture is not None and len(capture) < 2:
                capture.append((z.clone(), action_input.clone(), result[:, -1:].clone()))
            return result

        action_input = None
        return latent_rollout(predict, embed, z_history, past, future, history_size)

    def native(self, current_pixels, z, future, timing):
        """Unmodified official single-frame rollout with pre-encoded start, as in P0b."""
        candidates = len(future)
        pixels = current_pixels[None, None].expand(1, candidates, -1, -1, -1, -1)
        latent = z.reshape(1, 1, 1, -1).expand(1, candidates, 1, -1)
        action_forward = self.model.action_embed.forward
        predict = self.model.predict

        def timed_embed(actions):
            return timing.measure("action_embed", actions.shape[0] * actions.shape[1], action_forward, actions)

        def timed_predict(emb, act_emb):
            return timing.measure("predictor", emb.shape[0] * emb.shape[1], predict, emb, act_emb)

        with patch.object(self.model.action_embed, "forward", timed_embed), \
             patch.object(self.model, "predict", timed_predict):
            info = {"pixels": pixels, "embed_0": latent}
            return self.model.rollout(info, future[None])["predicted_embed_0"][0, :, 1:]


def check_equal(actual, reference, name):
    error = (actual - reference).abs()
    tolerance = ATOL + RTOL * reference.abs()
    if not bool(torch.isfinite(error).all()) or bool((error > tolerance).any()):
        raise ValueError(f"{name} exceeded native coordinate tolerance; max abs error {float(error.max())}.")
    return {
        "coordinates": error.numel(),
        "max_abs_error": float(error.max()),
        "max_fraction_of_coordinate_tolerance": float((error / tolerance).max()),
    }


def verify_w_context(captured, history, past, future):
    if len(captured) != 2:
        raise ValueError("W did not capture two actual native predictor input calls.")
    states0, actions0, pred0 = captured[0]
    states1, actions1, _ = captured[1]
    expected_states1 = torch.cat([history[:, 1:], pred0], dim=1)
    expected_actions0 = torch.cat([past, future[:, :1]], dim=1)
    expected_actions1 = torch.cat([past[:, 1:], future[:, :2]], dim=1)
    for actual, expected in ((states0, history), (states1, expected_states1),
                             (actions0, expected_actions0), (actions1, expected_actions1)):
        if not torch.equal(actual, expected):
            raise ValueError("W actual first-two input histories or action blocks are misaligned.")
    return {
        "validated_candidates": len(history),
        "call_0": {
            "state_step": [0, 5, 10], "state_kind": ["real", "real", "real"],
            "action_intervals": [[0, 5], [5, 10], [10, 15]],
            "state_shape": list(states0.shape), "raw_normalized_action_shape": list(actions0.shape),
        },
        "call_1": {
            "state_step": [5, 10, 15], "state_kind": ["real", "real", "predicted"],
            "action_intervals": [[5, 10], [10, 15], [15, 20]],
            "state_shape": list(states1.shape), "raw_normalized_action_shape": list(actions1.shape),
        },
    }
