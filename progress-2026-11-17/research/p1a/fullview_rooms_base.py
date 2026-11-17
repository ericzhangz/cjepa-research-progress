"""Measure native room forecasts on same-root action pairs with full RGB."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

BLOCK = 5
HORIZON = 4
FIT_SEEDS = np.arange(920000, 920064)
TEST_SEEDS = np.arange(930000, 930048)


class FourRoom:
    permits_goal_hit = False

    def __init__(self, args):
        from omegaconf import OmegaConf
        from stable_worldmodel.envs.four_room.env import FourRoomDistractorsEnv

        world = OmegaConf.to_container(OmegaConf.load(args.world_config).world, resolve=False)
        for key in ("env_name", "num_envs", "max_episode_steps", "history_size", "frame_skip"):
            world.pop(key)
        self.env = FourRoomDistractorsEnv(
            **{
                **world,
                "min_distractors": 1,
                "max_distractors": 1,
                "terminate_on_goal": False,
                "retarget_on_goal": False,
                "render_mode": "rgb_array",
            }
        )
        self.model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get_level(1)
        self.model.interpolate_pos_encoding = True
        assert isinstance(self.model.action_encoder, nn.Identity)
        stats = torch.load(args.normalizer, map_location="cpu", weights_only=False)["stats"]["action"]
        self.action_mean = np.asarray(stats["mean"], dtype=np.float64).reshape(2)
        self.action_std = np.asarray(stats["std"], dtype=np.float64).reshape(2)
        self.coordinates = "native FourRoom XY, one unit per rendered pixel"

    def physical(self, info):
        return info["xy"], np.r_[info["closed_door_idx"], info["distractor0_xy"]]

    def encode(self, pixels):
        return self.model.encode({"pixels": pixels[:, None]})["embed_0"][:, 0]

    def action_embedding(self, actions):
        return self.model.action_embed(self.model.action_encoder(actions))


class TwoRoomPhysics:
    permits_goal_hit = True
    coordinates = "native TwoRoom XY, one unit per rendered pixel"

    def __init__(self):
        from stable_worldmodel.envs.two_room.env import TwoRoomEnv

        self.env = TwoRoomEnv(render_mode="rgb_array")

    def physical(self, info):
        return info["proprio"], np.r_[info["goal_state"], self.env.wall_axis,
                                      self.env.wall_thickness, self.env.door_positions]


def paths_for_seed(seed):
    rng = np.random.default_rng(seed + 610000)
    headings = rng.uniform(-np.pi, np.pi, size=4)
    suffix = headings[3] + rng.normal(0, np.pi / 4, size=3).cumsum()
    left = np.r_[headings[:3], suffix]
    right = left.copy()
    right[2] += np.pi
    angles = np.stack([left, right])
    actions = np.stack([np.cos(angles), np.sin(angles)], -1).astype(np.float32)
    return np.repeat(actions[:, :, None], BLOCK, axis=2)


def replay(host, seed, actions):
    _, info = host.env.reset(seed=int(seed))
    states, nuisance = [], []
    images = [host.env.render()]
    xy, other = host.physical(info)
    states.append(np.asarray(xy).copy())
    nuisance.append(np.asarray(other).copy())
    for block in actions:
        for action in block:
            _, _, terminated, truncated, info = host.env.step(action)
            if truncated or (terminated and not host.permits_goal_hit):
                raise RuntimeError(f"Stock simulator stopped during seed {seed}.")
            xy, other = host.physical(info)
            states.append(np.asarray(xy).copy())
            nuisance.append(np.asarray(other).copy())
        images.append(host.env.render())
    return np.stack(images), np.stack(states), np.stack(nuisance)


def collect(host, seeds):
    rgbs, physical, nuisance, all_actions = [], [], [], []
    for seed in seeds:
        actions = paths_for_seed(int(seed))
        paths = [replay(host, int(seed), branch) for branch in actions]
        np.testing.assert_array_equal(actions[0, :2], actions[1, :2])
        np.testing.assert_array_equal(actions[0, 3:], actions[1, 3:])
        np.testing.assert_allclose(actions[0, 2], -actions[1, 2], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(paths[0][0][:3], paths[1][0][:3])
        np.testing.assert_array_equal(paths[0][1][:11], paths[1][1][:11])
        np.testing.assert_array_equal(paths[0][2], paths[1][2])
        rgbs.append(np.stack([path[0] for path in paths]))
        physical.append(np.stack([path[1] for path in paths]))
        nuisance.append(np.stack([path[2] for path in paths]))
        all_actions.append(actions)
    return tuple(np.stack(items) for items in (rgbs, physical, nuisance, all_actions))


@torch.inference_mode()
def encode(host, rgb, device):
    preprocess = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.Resize(size=(224, 224)),
    ])
    # The guaranteed identical three-frame prefix only needs to be encoded once.
    scenes, branches, times = rgb.shape[:3]
    unique = np.concatenate([rgb[:, 0, :3], rgb[:, 0, 3:], rgb[:, 1, 3:]], axis=1)
    frames = torch.from_numpy(unique.reshape(-1, *rgb.shape[-3:])).permute(0, 3, 1, 2)
    encoded = torch.cat([
        host.encode(preprocess(batch.to(device))).cpu() for batch in frames.split(32)
    ]).reshape(scenes, 3 + 2 * HORIZON, 192)
    prefix = encoded[:, None, :3].expand(-1, branches, -1, -1)
    future = encoded[:, 3:].reshape(scenes, branches, HORIZON, 192)
    return torch.cat([prefix, future], dim=2).reshape(scenes, branches, times, 192).clone()


@torch.inference_mode()
def native_forecasts(host, truth, actions, device):
    shape = truth.shape
    factual = truth.reshape(-1, 3 + HORIZON, 192).to(device)
    normalized = ((actions - host.action_mean) / host.action_std).astype(np.float32)
    flat_actions = torch.from_numpy(normalized.reshape(-1, 2 + HORIZON, 2 * BLOCK)).to(device)
    embeddings = host.action_embedding(flat_actions)
    free_window = factual[:, :3]
    teacher, free = [], []
    for step in range(HORIZON):
        condition = embeddings[:, step:step + 3]
        true_window = factual[:, step:step + 3]
        teacher.append(host.model.predict(true_window, condition)[:, -1])
        next_free = host.model.predict(free_window, condition)[:, -1]
        free.append(next_free)
        free_window = torch.cat([free_window[:, 1:], next_free[:, None]], dim=1)
    torch.testing.assert_close(teacher[0], free[0], rtol=0, atol=0)
    assert torch.isfinite(embeddings).all()
    return [torch.stack(path, dim=1).reshape(*shape[:2], HORIZON, 192).cpu() for path in (teacher, free)]


def fit_mlp(x, y):
    y = y.to(torch.float32)
    mean, sigma = y.mean(0), y.std(0)
    target = (y - mean) / sigma
    torch.manual_seed(10611)
    mlp = nn.Sequential(nn.Linear(192, 256), nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 2))
    optimizer = torch.optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(100):
        for idx in torch.randperm(len(x)).split(128):
            loss = (mlp(x[idx]) - target[idx]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return mlp.eval().requires_grad_(False), mean, sigma


def fit_probes(latents, states):
    # Use only physical labels paired with real encoded training RGB.
    x = torch.cat([latents[:, 0], latents[:, 1, 3:]], axis=1).reshape(-1, 192)
    y = torch.from_numpy(np.concatenate([states[:, 0], states[:, 1, 3:]], axis=1))
    y = y.reshape(-1, 2).to(torch.float32)
    mean, sigma = y.mean(0), y.std(0)
    target = (y - mean) / sigma
    x_design = torch.cat([x.double(), torch.ones(len(x), 1, dtype=torch.float64)], dim=1)
    penalty = 1e-3 * torch.eye(x_design.size(1), dtype=torch.float64)
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ target.double())
    linear = nn.Linear(192, 2)
    with torch.no_grad():
        linear.weight.copy_(weights[:-1].T)
        linear.bias.copy_(weights[-1])
    return {"linear": (linear.eval().requires_grad_(False), mean, sigma), "mlp": fit_mlp(x, y)}


def original_probe(path):
    from models.probers import build_prober

    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert tuple(saved["columns"][:2]) == ("xy_x", "xy_y")
    probe = build_prober(input_dim=192, output_dim=2)
    probe.load_state_dict({
        name.removeprefix("ego_xy."): value for name, value in saved["heads"].items()
        if name.startswith("ego_xy.")
    }, strict=True)
    return probe.eval().requires_grad_(False), *[torch.as_tensor(saved[k][:2]) for k in (
        "label_mean", "label_std"
    )]


def rmse(vector):
    return np.sqrt(np.square(vector).sum(axis=-1).mean(axis=0))


def measure(y, c, l, f):
    y, c, l, f = (array.astype(np.float64) for array in (y, c, l, f))
    endpoint_terms = c - y, l - c, f - l
    closure = np.max(np.abs(sum(endpoint_terms) - (f - y)))
    assert closure <= 1e-10
    delta_y, delta_c, delta_l, delta_f = [array[:, 0] - array[:, 1] for array in (y, c, l, f)]
    denom = np.square(delta_y).sum(-1)
    assert np.all(denom.sum(axis=0) > 0)
    selection = np.random.default_rng(11015).integers(0, len(y), (2000, len(y)))

    def effect(delta):
        dots = (delta * delta_y).sum(-1)
        boot = dots[selection].sum(1) / denom[selection].sum(1)
        return {
            "rms": rmse(delta).tolist(),
            "bias_rms": rmse(delta - delta_y).tolist(),
            "gain": (dots.sum(0) / denom.sum(0)).tolist(),
            "gain_95ci": np.quantile(boot, [0.025, 0.975], axis=0).T.tolist(),
        }

    return {
        "closure_max_absolute": float(closure),
        "endpoint_rmse": {k: rmse((v - y).reshape(-1, HORIZON, 2)).tolist() for k, v in (
            ("C", c), ("L", l), ("F", f)
        )},
        "effect": {k: effect(v) for k, v in (("Y", delta_y), ("C", delta_c), ("L", delta_l), ("F", delta_f))},
        "effect_gain_terms": {
            k: ((v * delta_y).sum((0, 2)) / denom.sum(0)).tolist() for k, v in (
                ("C-Y", delta_c - delta_y), ("L-C", delta_l - delta_c), ("F-L", delta_f - delta_l)
            )
        },
        "endpoint_term_rms": {
            key: rmse(value.reshape(-1, HORIZON, 2)).tolist() for key, value in zip(
                ("C-Y", "L-C", "F-L"), endpoint_terms
            )
        },
        "median_physical_separation": np.median(np.linalg.norm(delta_y, axis=-1), axis=0).tolist(),
        "zero_separation_count": (denom == 0).sum(0).tolist(),
    }


def run(args):
    torch.set_num_threads(args.threads)
    if args.host == "tworoom":
        host = TwoRoomPhysics()
        rgb, xy, nuisance, actions = collect(host, TEST_SEEDS)
        host.env.close()
        endpoints = xy[:, :, 3 * BLOCK::BLOCK]
        physical_delta = endpoints[:, 0] - endpoints[:, 1]
        args.out.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(args.out / "paired_physics.npz", test_seeds=TEST_SEEDS, rgb=rgb,
                            xy_per_primitive=xy, nuisance_per_primitive=nuisance,
                            raw_actions=actions, y=endpoints)
        report = {"host": "tworoom", "coordinates": host.coordinates, "scenes": len(TEST_SEEDS),
                  "primitive_actions_per_model_block": BLOCK, "exact_shared_complete_rgb_prefix": True,
                  "physical_effect_rms": rmse(physical_delta).tolist(),
                  "model_measured": False,
                  "missing_assets": ["complete published weights.pt", "dataset action normalization"]}
        (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"tworoom: wrote official physical controls to {args.out}", flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    host = FourRoom(args)
    host.model.to(device).eval().requires_grad_(False)
    args.out.mkdir(parents=True, exist_ok=False)
    train_rgb, train_xy, _, _ = collect(host, FIT_SEEDS)
    print(f"{args.host}: collected {len(FIT_SEEDS)} readout fit scenes", flush=True)
    train_latent = encode(host, train_rgb, device)
    del train_rgb
    probes = fit_probes(train_latent, train_xy[:, :, ::BLOCK])
    if args.probe:
        probes["p0c_mlp"] = original_probe(args.probe)
    test_rgb, test_xy, nuisance, actions = collect(host, TEST_SEEDS)
    print(f"{args.host}: collected {len(TEST_SEEDS)} independent paired scenes", flush=True)
    actual = encode(host, test_rgb, device)
    teacher, free = native_forecasts(host, actual, actions, device)
    y = test_xy[:, :, 3 * BLOCK::BLOCK].copy()
    arrays = {"fit_seeds": FIT_SEEDS, "test_seeds": TEST_SEEDS, "rgb": test_rgb,
              "xy_per_primitive": test_xy, "nuisance_per_primitive": nuisance, "raw_actions": actions,
              "actual_latent": actual.numpy(), "teacher_latent": teacher.numpy(), "free_latent": free.numpy(),
              "y": y, "fit_latent": train_latent.numpy(), "fit_xy": train_xy[:, :, ::BLOCK]}
    report = {
        "host": args.host, "coordinates": host.coordinates, "primitive_actions_per_model_block": BLOCK,
        "checkpoint": str(args.checkpoint), "normalizer": str(args.normalizer),
        "action_mean": host.action_mean.tolist(), "action_std": host.action_std.tolist(),
        "readout_fit_seeds": FIT_SEEDS.tolist(), "evaluation_seeds": TEST_SEEDS.tolist(),
        "prefix_actions": 2 * BLOCK, "prediction_horizons": list(range(1, HORIZON + 1)),
        "first_block_opposite_and_suffix_identical": True, "exact_shared_complete_rgb_prefix": True,
        "exact_common_nuisance_replay": True, "exact_h1_teacher_free_latent": True, "readouts": {},
    }
    truth_delta = (actual[:, 0, 3:] - actual[:, 1, 3:]).double().numpy()
    latent_energy = np.square(truth_delta).sum((0, 2))
    report["native_latent_effect"] = {
        "factual_rms_l2": rmse(truth_delta).tolist(),
        **{
            name: (((path[:, 0] - path[:, 1]).double().numpy() * truth_delta).sum((0, 2))
                   / latent_energy).tolist()
            for name, path in (("L_gain", teacher), ("F_gain", free))
        },
    }
    for name, (probe, mean, std) in probes.items():
        with torch.no_grad():
            c, l, f = [(probe(latents) * std + mean).numpy() for latents in (actual[:, :, 3:], teacher, free)]
        report["readouts"][name] = measure(y, c, l, f)
        arrays.update({f"{name}_{key}": value for key, value in (("c", c), ("l", l), ("f", f))})
    host.env.close()
    np.savez_compressed(args.out / "paired_arrays.npz", **arrays)
    torch.save({name: {"state_dict": probe.state_dict(), "mean": mean, "std": std}
                for name, (probe, mean, std) in probes.items()}, args.out / "readouts.pt")
    (args.out / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{args.host}: wrote {args.out}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=["fourroom", "tworoom"], required=True,
                        help="TwoRoom currently collects simulator controls only.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--world-config", type=Path)
    parser.add_argument("--probe", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.host == "fourroom" and not all((args.checkpoint, args.normalizer, args.world_config)):
        parser.error("FourRoom requires --checkpoint, --normalizer, and --world-config.")
    run(args)


if __name__ == "__main__":
    main()
