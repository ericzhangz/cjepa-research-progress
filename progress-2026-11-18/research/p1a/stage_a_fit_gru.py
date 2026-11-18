"""Fit the capacity-matched Stage A GRU after the fixed FourRoom host qualification."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .native_topology_condition import action_embeddings
from .stage_a_baselines import GRUDynamics, evaluate_gru, fit_gru, predictor_parameter_count
from .stage_a_fourroom import (
    HORIZON, LOCKS, Native, OfficialFourRoom, OpenPhysics, Task,
    candidate_bank, door_pairs, lifecycle_episode, protocol,
)

ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION = ROOT / "runs/stage_a_fourroom_20261117"
OUT = ROOT / "runs/stage_a_20261117"
FIT_SEED = 2026112200
VALIDATION_SEED = 2026112400
EPISODES = (96, 16)
UPDATES = 1440
BATCH = 16


def require_qualified(directory=QUALIFICATION):
    physical, native = (json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
                        for name in ("physical", "native"))
    settings = protocol(fixed_distractor="fixed_distractor" in physical["protocol"])
    if physical["protocol"] != settings or native["protocol"] != settings:
        raise RuntimeError("FourRoom training requires the pinned common protocol")
    outcomes = [entry["passed"] for run in (physical, native) for entry in run["qualification"]["summary"].values()]
    outcomes.extend(entry["passed"] for entry in native["qualification"]["geometry"].values() if "passed" in entry)
    outcomes.extend(row["distinct_routes"] and row["complete_two_door_routes"] and row["positive_uniform_regret"]
                    and row["chronology"]["matched_factual_three_frame_roots"] for row in physical["forks"])
    if not outcomes or not all(outcomes):
        raise RuntimeError("FourRoom host qualification did not pass")
    return physical


@torch.inference_mode()
def encode_frames(native, rgb):
    out = []
    for start in range(0, len(rgb), 32):
        batch = torch.from_numpy(np.ascontiguousarray(rgb[start:start+32])).to(native.device)
        batch = batch.permute(0, 3, 1, 2).float() / 255.
        batch = (batch - native.mean) / native.std
        out.append(native.host.encode({"pixels": batch[:, None]})["emb"][:, 0].float().cpu())
    # Downstream GRU autograd saves its inputs even if its shared encoders are frozen.
    return torch.cat(out)


def data(native, templates, first, count, *, fixed_distractor=False):
    """Collect actual AAB/ABB trajectories followed by one drawn, executed H12 program."""
    env = OfficialFourRoom(fixed_distractor=fixed_distractor)
    physics = OpenPhysics(env)
    latents, actions, layout_rows = [], [], []
    try:
        for i in range(count):
            template = templates[i % len(templates)]
            seed = first + i
            direction = int(template["chronology"]["episode_a"]["direction"])
            door = template["chronology"]["episode_a"]["door"]
            _, axis, _, endpoints = next(item for item in door_pairs() if item[0] == door and item[2] == direction)
            start = np.asarray(template["start"], dtype=np.float32)
            goal = np.asarray(template["goal"], dtype=np.float32)
            schedule = (0, 0, 1) if (i // 4) % 2 == 0 else (0, 1, 1)
            initial = (i // 8) % 2
            _, trace = lifecycle_episode(env, physics, door, axis, direction, endpoints[0], goal, seed, initial,
                                         mode_schedule=schedule, anchor=start)
            task = Task(f"training_{seed}", "two_doors", tuple(map(float, start)), tuple(map(float, goal)), seed, ())
            bank = candidate_bank(task, start, 0, HORIZON)
            family = (i // 2) % 2
            slot = family * (len(bank) // 2)
            if i % 4:
                # Preserve both nominal routes and independently varied lawful controls.
                slot += int(np.random.default_rng(seed).integers(1, len(bank)//2 - 9))
            program = bank[slot]
            np.testing.assert_array_equal(env.xy, start)
            extra = []
            for commands in program:
                env.execute(commands)
                extra.append(env.render())
            rgb = np.concatenate((trace["rgb"], np.stack(extra)), axis=0)
            raw = np.concatenate((trace["raw_actions"], program), axis=0)
            latents.append(encode_frames(native, rgb))
            with torch.no_grad():
                actions.append(action_embeddings(native.host, raw).float().cpu())
            layout_rows.append({"seed": seed, "relative_switch_order": list(schedule), "program_slot": slot,
                                "observed_blocks": len(raw), "template": template["task"]})
            if (i+1) % 16 == 0:
                print(f"Collected {i+1}/{count} factual episodes (starting seed {first})", flush=True)
    finally:
        env.close()
    return torch.stack(latents), torch.stack(actions), layout_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default=QUALIFICATION, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (OUT if args.directory.resolve() == QUALIFICATION else args.directory)
    qualified = require_qualified(args.directory)
    fixed_distractor = "fixed_distractor" in qualified["protocol"]
    torch.set_num_threads(2)
    torch.manual_seed(FIT_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    native = Native(device)
    started = time.perf_counter()
    train_z, train_a, train_cases = data(native, qualified["forks"], FIT_SEED, EPISODES[0],
                                        fixed_distractor=fixed_distractor)
    valid_z, valid_a, valid_cases = data(native, qualified["forks"], VALIDATION_SEED, EPISODES[1],
                                        fixed_distractor=fixed_distractor)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"fitting_latents": train_z, "fitting_actions": train_a, "validation_latents": valid_z,
                "validation_actions": valid_a}, output / "gru_factual_training_data.pt")
    print(f"Fitting GRU with {len(train_z)} factual sequences, {train_a.shape[1]} blocks each", flush=True)
    target = predictor_parameter_count(native.host)
    recurrent = GRUDynamics(train_z.shape[-1], train_a.shape[-1], target).to(device)
    initial = evaluate_gru(recurrent, valid_z, valid_a, batch_size=BATCH, horizon=HORIZON)
    print(f"Initial held-out H{HORIZON} error: {initial}", flush=True)
    report = fit_gru(recurrent, train_z, train_a, updates=UPDATES, batch_size=BATCH, horizon=HORIZON,
                     seed=FIT_SEED, heldout=(valid_z, valid_a))
    torch.save({"state_dict": recurrent.cpu().state_dict(), "latent_dim": train_z.shape[-1],
                "action_dim": train_a.shape[-1], "target_parameters": target}, output / "gru.pt")
    report.update({"initial_heldout": initial, "train_cases": train_cases, "validation_cases": valid_cases,
                   "total_wall_seconds": time.perf_counter() - started, "qualifying_protocol": qualified["protocol"],
                   "information": "frozen arrived RGB latents and corresponding block action embeddings only"})
    training_path = output / "gru_training.json"
    training_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Finished {UPDATES} updates; {training_path}", flush=True)


if __name__ == "__main__":
    main()
