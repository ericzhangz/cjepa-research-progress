"""Attribute the frozen §38 rollout error on its archived physical branches."""

import json

import numpy as np
import torch

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import conditioned_rollout
from .native_full_depth_first_effect import branch_offsets
from .native_full_depth_reversible import action_embeddings, encode_rgb
from .native_full_depth_task_geometry import chart_at_latent, chart_pose
from .native_history_matched_fit import (
    FITTED, HORIZONS, OUTPUT as MATCHED, PHYSICS, SEED, hold_and_relative, image_x, pairing,
)
from .native_reversible_pusht import ACTION_BLOCK, CANDIDATES, ReversiblePushT, make_scene, replay


OUTPUT = PHYSICS.parent / "native_history_rollout_attribution_20261115"
PULSE = (0, 5)


def replay_rgb(archive):
    commands, common, plans = (archive[key] for key in ("commands", "common_modes", "plans"))
    scenes = archive["scene_ids"]
    initial, sensors, roots, future = (archive[key] for key in
                                      ("initial_rgb", "observed_x", "root_state", "future_state"))
    root_frames = np.empty((*roots.shape[:-1], 3, *initial.shape[1:]), dtype=initial.dtype)
    future_frames = np.empty((*future.shape[:-1], *initial.shape[1:]), dtype=initial.dtype)
    env = ReversiblePushT()
    try:
        for scene_index, seed in enumerate(scenes):
            scene = make_scene(int(seed))
            for history, mode in np.ndindex((2, 2)):
                index = scene_index, history, mode
                schedule = list(zip([*map(int, common), history, mode], commands))
                np.testing.assert_array_equal(env.start(scene, schedule[0][0]), initial[scene_index])
                images = []
                for previous_mode, command in schedule:
                    env.set_regime(previous_mode)
                    images.append(env.execute(command))
                np.testing.assert_array_equal(env.get_obs(), roots[index])
                np.testing.assert_allclose(image_x(np.stack(images)), sensors[index], rtol=0, atol=1e-8)
                root_frames[index] = images[-3:]
                for candidate, plan in enumerate(plans):
                    replay(env, scene, schedule, mode)
                    np.testing.assert_array_equal(env.observe(), root_frames[index][-1])
                    np.testing.assert_array_equal(env.get_obs(), roots[index])
                    for step, command in enumerate(plan):
                        endpoint = (*index, candidate, step)
                        future_frames[endpoint] = env.execute(command)
                        np.testing.assert_array_equal(env.get_obs(), future[endpoint])
    finally:
        env.close()
    return root_frames, future_frames


def error_terms(values):
    return {
        "transport": values["free"] - values["real_window"],
        "one_step": values["real_window"] - values["encoded"],
        "readout": values["encoded"] - values["physical"],
    }


def projection(vector, target):
    axes = (0,) + tuple(range(2, vector.ndim - 2))
    target_power = target.square().sum(-1).mean(dim=axes)
    power = vector.square().sum(-1).mean(dim=axes)
    cross = (vector * target).sum(-1).mean(dim=axes)
    amplitude = cross / target_power.sqrt().clamp_min(1e-8)
    return {
        "rms": power.sqrt().tolist(),
        "signed_gain": (cross / target_power.clamp_min(1e-8)).tolist(),
        "signed_projected_amplitude": amplitude.tolist(),
        "orthogonal_rms": (power - amplitude.square()).clamp_min(0).sqrt().tolist(),
    }


def summarize(values):
    truth = values["physical"]
    truth_gap = truth.diff(dim=1).squeeze(1)
    endpoint_axes = (0, 1) + tuple(range(3, truth.ndim - 2))
    arms = {}
    for name, forecast in values.items():
        metrics = {key: value for key, value in pairing(forecast, truth).items() if not key.startswith("scene_")}
        metrics.update(projection(forecast.diff(dim=1).squeeze(1), truth_gap))
        metrics["trajectory_rmse"] = (forecast - truth).square().sum(-1).mean(dim=endpoint_axes).sqrt().tolist()
        arms[name] = metrics
    return {
        "arms": arms,
        "history_gap_errors": {
            key: projection(term.diff(dim=1).squeeze(1), truth_gap) for key, term in error_terms(values).items()
        },
    }


@torch.inference_mode()
def run(device):
    torch.manual_seed(SEED)
    model, adapter, gate, hold, *_ = substep.load_core(device)
    fitted = torch.load(FITTED / "forecast_only.pt", map_location=device, weights_only=True)
    adapter.load_state_dict(fitted["adapter"])
    weight, bias = (fitted["chart"][key] for key in ("weight", "bias"))
    with np.load(PHYSICS / "physical.npz", allow_pickle=False) as archive:
        scenes, raw_plans = archive["scene_ids"], archive["plans"]
        np.testing.assert_array_equal(scenes, np.arange(56000, 56016))
        np.testing.assert_array_equal(raw_plans[PULSE[0], 1:], raw_plans[PULSE[1], 1:])
        if np.array_equal(raw_plans[PULSE[0], 0], raw_plans[PULSE[1], 0]):
            raise ValueError("The pulse candidates must differ at the first block")
        root_rgb, future_rgb = replay_rgb(archive)
        print(f"Verified {np.prod(future_rgb.shape[:4])} physical branches", flush=True)
        physical = torch.as_tensor(archive["future_state"][..., 2:4], device=device)
        past = action_embeddings(model, hold.new_tensor(archive["commands"][-2:]))
        plans = action_embeddings(model, hold.new_tensor(raw_plans))

    root = encode_rgb(model, root_rgb)
    arrived = encode_rgb(model, future_rgb)
    branch_shape = arrived.shape[:-2]
    mode = torch.arange(2, device=device)[None, None, :, None].expand(branch_shape).flatten()
    roots = root.unsqueeze(-3).expand(*branch_shape, *root.shape[-2:]).flatten(0, 3)
    future = arrived.flatten(0, 3)
    windows = torch.cat((roots, future), dim=1)
    controls = torch.cat((past.expand(*branch_shape, *past.shape), plans.expand(*branch_shape, *plans.shape[-2:])),
                         dim=-2).flatten(0, 3)
    offsets = branch_offsets(adapter, len(roots))
    index = torch.arange(len(roots), device=device)

    def forecast(window, previous, actions):
        prediction = conditioned_rollout(model, window, previous, actions[:, None], offsets, first_effect=gate)
        return prediction[index, 0, mode]

    horizon = plans.size(1)
    free = forecast(roots, controls[:, :2], controls[:, 2:])
    real_window = torch.cat([
        forecast(windows[:, step:step + 3], controls[:, step:step + 2], controls[:, step + 2:step + 3])
        for step in range(horizon)
    ], dim=1)
    torch.testing.assert_close(free[:, 0], real_window[:, 0], rtol=2e-4, atol=3e-5)
    values = {name: chart_pose(chart_at_latent(latent, weight, bias))[..., :2].double().reshape(physical.shape)
              for name, latent in (("free", free), ("real_window", real_window), ("encoded", future))}
    values["physical"] = physical
    parts = error_terms(values)
    torch.testing.assert_close(sum(parts.values()), values["free"] - physical, rtol=0, atol=1e-10)

    published = json.loads((MATCHED / "full_view.json").read_text(encoding="utf-8"))["frozen"]
    report = {"endpoint": summarize(values)}
    for family, subset in zip(("hold", "relative"), zip(*(hold_and_relative(value) for value in values.values()))):
        selected = dict(zip(values, subset))
        old = selected["free"][..., HORIZONS, :].new_tensor(published[family]["scene_predictions"])
        torch.testing.assert_close(selected["free"][..., HORIZONS, :], old, rtol=1e-5, atol=1e-3)
        report[family] = summarize(selected)
    pulse_first, pulse_second = PULSE
    pulse = {name: value[..., pulse_first, :, :] - value[..., pulse_second, :, :] for name, value in values.items()}
    report["first_block_pulse"] = summarize(pulse)

    summary = {
        "scene_ids": scenes.tolist(), "candidates": list(CANDIDATES), "pulse_indices": list(PULSE),
        "horizon_blocks": list(range(1, horizon + 1)), "horizon_raw_controls": [ACTION_BLOCK * (i + 1) for i in range(horizon)],
        "summary_axes": ["current mode (0 open, 1 braked)", "horizon"],
        "branch_axes": ["scene", "physical history", "current mode", "candidate", "horizon", "physical XY"],
        "first_pulse_horizon_with_identical_native_command_slice": 4,
        "verified": {
            "physical_branches": int(np.prod(branch_shape)), "physical_endpoints": int(np.prod(branch_shape) * horizon),
            "free_matches_section_38": True, "h1_max_latent_difference": (free[:, 0] - real_window[:, 0]).abs().max().item(),
            "max_error_decomposition_residual": (sum(parts.values()) - (values["free"] - physical)).abs().max().item(),
        },
        "metrics": report,
    }
    arrays = {"scene_ids": scenes, "candidates": np.array(CANDIDATES), "raw_plans": raw_plans}
    arrays.update({name: value.cpu().numpy() for name, value in (values | parts).items()})
    return summary, arrays


def main():
    paths = OUTPUT / "summary.json", OUTPUT / "branches.npz"
    for path in paths:
        if path.exists():
            raise FileExistsError(path)
    summary, arrays = run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    OUTPUT.mkdir(exist_ok=True)
    with paths[0].open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    with paths[1].open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
