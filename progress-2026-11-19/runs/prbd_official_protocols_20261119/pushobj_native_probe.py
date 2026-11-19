"""PushObj release inventory and native zero-anchor compatibility probe only.

Requires the official AdaJEPA source/assets and a CUDA PyTorch environment.
No environment is created or stepped. The bundled non-pickle panel preserves
the eight already captured inputs; see ASSETS.md for the upstream requirements.
"""

import argparse
import json
from pathlib import Path
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch import nn


OUT = Path(__file__).resolve().parent


class NativeRankOne(nn.Module):
    """One proposal-compatible native map W(u)=W0+u AB^T; no trained effect."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        generator = torch.Generator().manual_seed(20261119)
        self.A = nn.Parameter((.01 * torch.randn(original.out_features, 1, generator=generator)).to(original.weight))
        self.B = nn.Parameter((.01 * torch.randn(original.in_features, 1, generator=generator)).to(original.weight))
        self.u = nn.Parameter(original.weight.new_zeros(()))

    def forward(self, value):
        # Execute the conditioned route even at u=0; do not dispatch to F0.
        return self.original(value) + self.u * ((value @ self.B) @ self.A.T)


def synchronize():
    torch.cuda.synchronize()


def timed(function):
    synchronize()
    started = time.perf_counter()
    result = function()
    synchronize()
    return result, time.perf_counter() - started


def inventory(assets):
    goals = []
    for shape in ("T", "L", "Z", "+", "I", "small_tee", "square"):
        path = assets / f"data/pushobj_eval/val_{shape}/plan_targets.pkl"
        with path.open("rb") as stream:
            data = pickle.load(stream)
        segments = data["segments"]
        if len(segments) != data["n_samples"] or any(segment["shape"] != shape for segment in segments):
            raise ValueError(f"Goal-bank metadata disagree for {shape}")
        goals.append({"shape": shape, "path": path.relative_to(assets).as_posix(), "bytes": path.stat().st_size,
                      "segments": len(segments), "trajectory_environment_actions": data["traj_len"],
                      "first_state_shape": list(segments[0]["states"].shape),
                      "first_action_shape": list(segments[0]["actions"].shape),
                      "stored_RGB": False})
    return {"goals": goals, "checkpoint_bytes":
            (assets / "checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth").stat().st_size,
            "staged_training_trajectory_directory_exists": (assets / "data/pushobj_multishape").exists()}


def main(args):
    source = args.source.resolve()
    assets = args.assets.resolve()
    sys.path.insert(0, str(source))
    from datasets.img_transforms import default_transform
    from datasets.pusht_dset import ACTION_MEAN, ACTION_STD, PROPRIO_MEAN, PROPRIO_STD
    from models.visual_world_model import VWorldModel
    from omegaconf import OmegaConf
    from preprocessor import Preprocessor

    if not torch.cuda.is_available():
        raise RuntimeError("The published ViTPredictor mask requires CUDA; no CPU replacement is used")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    checkpoint_path = assets / "checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"
    config_path = checkpoint_path.parents[1] / "hydra.yaml"
    config = OmegaConf.load(config_path)
    checkpoint, loading_seconds = timed(lambda: torch.load(checkpoint_path, map_location=device, weights_only=False))
    # Reuse every released component. Avoid plan.py's unused DINO Hub startup.
    model = VWorldModel(image_size=config.img_size, num_hist=config.num_hist, num_pred=config.num_pred,
                        encoder=checkpoint["encoder"], proprio_encoder=checkpoint["proprio_encoder"],
                        action_encoder=checkpoint["action_encoder"], decoder=checkpoint["decoder"],
                        predictor=checkpoint["predictor"], proprio_dim=config.proprio_emb_dim,
                        action_dim=config.action_emb_dim, concat_dim=config.concat_dim,
                        num_action_repeat=config.num_action_repeat, num_proprio_repeat=config.num_proprio_repeat,
                        train_encoder=False, train_predictor=False, train_decoder=False,
                        straighten=config.training.straighten, stop_grad=config.training.stop_grad).to(device)
    model.eval()
    model.requires_grad_(False)
    with np.load(args.panel, allow_pickle=False) as panel:
        archived = {"obs_0": {"visual": panel["visual"], "proprio": panel["proprio"]},
                    "state_0": panel["state_0"],
                    "gt_actions": torch.from_numpy(panel["gt_actions"])}
        captured_indices = panel["official_val_I_target_indices"].tolist()
    with (assets / "data/pushobj_eval/val_I/plan_targets.pkl").open("rb") as stream:
        official_segments = pickle.load(stream)["segments"]
    target_indices = random.Random(100).sample(range(len(official_segments)), 8)
    if captured_indices != target_indices:
        raise ValueError("Captured panel order differs from the original seed-100 selection")
    selected = [official_segments[index] for index in target_indices]
    raw_actions = torch.stack([torch.tensor(segment["actions"], dtype=torch.float32) for segment in selected])
    expected_actions = ((raw_actions / 100 - ACTION_MEAN) / ACTION_STD).reshape(8, 5, 10)
    torch.testing.assert_close(archived["gt_actions"], expected_actions, atol=0, rtol=0)
    expected_starts = torch.stack([torch.tensor(segment["states"][0], dtype=torch.float32) for segment in selected])
    torch.testing.assert_close(torch.from_numpy(archived["state_0"][:, :5]), expected_starts, atol=0, rtol=0)
    processor = Preprocessor(ACTION_MEAN, ACTION_STD, None, None, PROPRIO_MEAN, PROPRIO_STD,
                             default_transform(config.img_size))
    obs = {name: tensor.to(device) for name, tensor in processor.transform_obs(archived["obs_0"]).items()}
    actions = archived["gt_actions"].to(device)
    if actions.shape != (8, 5, 10) or obs["visual"].shape != (8, 1, 3, 224, 224):
        raise ValueError("Expected archived official eight-task starts and five normalized action blocks")
    with torch.no_grad():
        tokens = model.encode(obs, actions[:, :1])
        base_one, one_seconds = timed(lambda: model.predict(tokens))
        (_, base_free), free_seconds = timed(lambda: model.rollout(obs, actions))
    native_map = model.predictor.transformer.layers[-1][1].net[4]
    if native_map.weight.shape != (404, 2048):
        raise ValueError("Unexpected published native FFN map layout")
    adapter = NativeRankOne(native_map).eval()
    model.predictor.transformer.layers[-1][1].net[4] = adapter
    try:
        with torch.no_grad():
            zero_one, zero_one_seconds = timed(lambda: model.predict(tokens))
            (_, zero_free), zero_free_seconds = timed(lambda: model.rollout(obs, actions))
        probe = torch.linspace(-1, 1, base_one.shape[-1], device=device)
        zero_gate_gradient, = torch.autograd.grad((model.predict(tokens) * probe).sum(), adapter.u)
        with torch.no_grad():
            adapter.u.fill_(1.)
        nonzero = model.predict(tokens)
        a_gradient, b_gradient = torch.autograd.grad((nonzero * probe).sum(), (adapter.A, adapter.B))
        nonzero_delta = (nonzero.detach() - base_one).flatten(1).norm(dim=-1)
        checks = {"zero_one_step_exact": torch.equal(base_one, zero_one),
                  "zero_free_five_steps_exact": torch.equal(base_free, zero_free),
                  "zero_one_step_max_abs_error": float((base_one-zero_one).abs().max()),
                  "zero_free_max_abs_error_per_prediction_horizon":
                      (base_free[:, 1:]-zero_free[:, 1:]).abs().amax((0, 2, 3)).tolist(),
                  "zero_gate_fixed_probe_gradient": float(zero_gate_gradient),
                  "nonzero_gate_output_change_l2_by_archived_input": nonzero_delta.tolist(),
                  "nonzero_gate_A_gradient_l2": float(a_gradient.norm()),
                  "nonzero_gate_B_gradient_l2": float(b_gradient.norm())}
        if not (checks["zero_one_step_exact"] and checks["zero_free_five_steps_exact"] and
                zero_gate_gradient.isfinite() and zero_gate_gradient != 0 and
                (nonzero_delta > 0).all() and a_gradient.isfinite().all() and b_gradient.isfinite().all() and
                (a_gradient != 0).any() and (b_gradient != 0).any()):
            raise ValueError("Native zero-equivalence or gradient path check failed")
    finally:
        model.predictor.transformer.layers[-1][1].net[4] = native_map
    report = {"scope": "Stage-1 compatibility only; no learned PRBD conditions, training, environment steps, planning, or task-value claim",
              "passed": True, "checkpoint": str(checkpoint_path), "checkpoint_epoch": checkpoint["epoch"],
              "training_config": str(config_path), "native_source": str(source), "archived_input": str(args.panel),
              "inventory": inventory(assets), "published_host": {
                  "encoder": type(model.encoder).__name__, "visual_width": model.encoder.emb_dim,
                  "predictor": type(model.predictor).__name__, "predictor_positions_shape": list(model.predictor.pos_embedding.shape),
                  "predictor_layers": len(model.predictor.transformer.layers),
                  "attention_heads": model.predictor.transformer.layers[0][0].heads,
                  "proprio_embedding_width": model.proprio_dim, "action_embedding_width": model.action_dim,
                  "input_tokens_shape": list(tokens.shape), "free_output_shape": list(base_free.shape),
                  "native_AdaLN": False, "attention_mask": "causal by frame; unregistered CUDA tensor in published source"},
              "input_convention": {"RGB": "archived uint8 -> /255 -> official resize/crop -> normalize mean/std .5",
                  "proprio": "archived x,y,vx,vy with official training mean/std",
                  "actions": "archived already normalized commands; 5 model blocks each concatenate 5 environment 2D actions",
                  "start_observations": 1, "num_hist_cap": 3, "frameskip": 5, "future_environment_actions": 25,
                  "official_val_I_target_indices": target_indices,
                  "official_start_states_and_normalized_actions_exact_match": True,
                  "selection": "all eight existing frozen-run starts in archived order; none selected by this probe"},
              "conditional_port": {"module": "predictor.transformer.layers[-1][1].net[4]",
                  "native_weight_shape": list(native_map.weight.shape), "formula": "W(u)=W0+u A B^T",
                  "rank": 1, "A_B_parameters": adapter.A.numel()+adapter.B.numel(),
                  "scope": "same native final FFN map for current issuance and every future prediction; all token positions",
                  "initialization": "independent fixed seed 20261119, A/B std .01, tested u=0 and u=1; no selection",
                  "nominal_dispatch": "full conditioned path is evaluated at u=0; no bypass",
                  "learned_conditions_available": False},
              "checks": checks, "timing_seconds": {"checkpoint_loading": loading_seconds,
                  "native_one_step_cold": one_seconds, "native_free_H5": free_seconds,
                  "conditioned_zero_one_step": zero_one_seconds, "conditioned_zero_free_H5": zero_free_seconds},
              "runtime": {"python": sys.version, "torch": torch.__version__, "device": str(device),
                  "TF32": False, "inference_mode": "eval; all released weights frozen", "dtype": "float32"}}
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "passed": report["passed"], "checks": checks}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Official AdaJEPA checkout containing models/, datasets/, and preprocessor.py")
    parser.add_argument("--assets", type=Path, required=True,
                        help="Official assets root containing checkpoints/ and data/")
    parser.add_argument("--panel", type=Path, default=OUT / "pushobj_frozen_starts.npz",
                        help="Captured eight-start panel (loaded with allow_pickle=False)")
    parser.add_argument("--output", type=Path, default=OUT / "pushobj_native_probe.reproduced.json")
    main(parser.parse_args())
