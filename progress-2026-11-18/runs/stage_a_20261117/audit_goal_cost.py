"""Score the pinned H12 FourRoom bank on exact RGB outcomes with the official goal criterion."""

import json
from pathlib import Path
import time

import numpy as np
import torch

from research.p1a.native_topology_writein_attribution import first_block_groups
from research.p1a.stage_a_fit_gru import encode_frames
from research.p1a.stage_a_fourroom import (
    ACTION_BLOCK, LOCKS, ROOT, Native, OfficialFourRoom, OpenPhysics, VisibleMapCost,
    fixed_tasks, latent_goal_cost, write_json,
)

HERE = Path(__file__).resolve().parent
QUAL = ROOT / "runs/stage_a_fourroom_20261117"
WORLDS = (None, *LOCKS)
PHASES = ("initial_probe_retained_fork", "reversal_1_retained_fork", "reversal_2_retained_fork")
LAST_RAW = (55, 60)


def frozen_records():
    selected = {}
    with (QUAL / "baseline_replay/rounds.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["arm"] == "frozen":
                for entry in row["rows"]:
                    for label, mode in entry["phase_labels"]:
                        if label in PHASES:
                            key = (row["episode"].rsplit("_", 1)[0], label, mode)
                            assert key not in selected
                            selected[key] = entry["selection"]
    assert len(selected) == 24
    return selected


def reset(env, task, lock, goal_rgb):
    env.ego_lock = lock
    env.reset(seed=task.seed, options={"state": task.start, "target_state": task.goal})
    assert env.closed_door is None and not env.retarget_on_goal
    np.testing.assert_array_equal(env._target_img.permute(1, 2, 0).numpy(), goal_rgb)


def render_xy(env, xy):
    return env._render_frame(torch.as_tensor(xy, dtype=torch.float32)).permute(1, 2, 0).numpy()


def exact_latents(native, env, positions):
    unique, inverse = np.unique(positions.reshape(-1, 2), axis=0, return_inverse=True)
    chunks = []
    for start in range(0, len(unique), 32):
        frames = np.stack([render_xy(env, point) for point in unique[start:start+32]])
        chunks.append(encode_frames(native, frames))
    return torch.cat(chunks)[inverse].reshape(*positions.shape[:-1], -1).to(native.device)


def outcome(cost, commands, index):
    groups, minima = first_block_groups(commands, cost)
    return {"candidate": int(index), "first_block_group": int(groups[index]),
            "first_block_regret": float(minima[index] - cost.min()),
            "full_plan_regret": float(cost[index] - cost.min())}


def terminal_readouts(task, t, paths, commands, native, env, check_env, goal_rgb, goal_latent,
                      history_z, background):
    clips = []
    checks = []
    pixels = [probe * ACTION_BLOCK for probe in range(1, commands.shape[1]+1)]
    assert tuple(pixels[-2:]) == LAST_RAW
    for raw_step in LAST_RAW:
        env.distractor_positions = background[t*ACTION_BLOCK+raw_step].clone()
        clips.append(exact_latents(native, env, paths[..., raw_step-1, :]))
    z = torch.stack(clips, dim=-2)
    # criterion removes slot 0, so supply the actual common root latent before H11/H12.
    padded = torch.cat((history_z.expand(*z.shape[:2], 1, -1), z), dim=-2)
    scores = latent_goal_cost(native.host, padded, goal_latent.expand(len(WORLDS), -1))
    direct = (z - goal_latent[:, None, None]).square().sum(dim=(-2, -1))
    torch.testing.assert_close(scores, direct)
    for w, lock in enumerate(WORLDS):
        # Independent scalar author-env check of representative candidates from both route families.
        for index in (0, commands.shape[0] // 2):
            reset(check_env, task, lock, goal_rgb)
            for _ in range(t):
                check_env.execute(np.zeros((ACTION_BLOCK, 2), dtype=np.float32))
            for block, raw in enumerate(commands[index], 1):
                xy = check_env.execute(raw)
                np.testing.assert_array_equal(xy, paths[w, index, (block-1)*ACTION_BLOCK:block*ACTION_BLOCK])
                if block*ACTION_BLOCK in LAST_RAW:
                    env.distractor_positions = background[(t+block)*ACTION_BLOCK].clone()
                    np.testing.assert_array_equal(check_env.render(), render_xy(env, xy[-1]))
                    checks.append((w, int(index), block*ACTION_BLOCK))
    return scores.cpu().numpy(), checks


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    native = Native(torch.device("cuda"))
    env, check_env = OfficialFourRoom(), OfficialFourRoom()
    physics = OpenPhysics(env)
    physical = json.loads((QUAL / "physical.json").read_text(encoding="utf-8"))
    frozen = frozen_records()
    tasks = {task.name: task for task in fixed_tasks()}
    with np.load(QUAL / "fork_candidates.npz", allow_pickle=False) as f:
        commands, archived_paths, archived_costs, goals = (f[name] for name in
            ("raw_candidates", "raw_xy_by_mode", "candidate_cost_by_mode", "goal_rgb"))
    with np.load(QUAL / "fork_lifecycle_rgb_controls.npz", allow_pickle=False) as f:
        history_rgb, history_xy, distractor_xy, history_goals = (f[name] for name in
            ("rgb", "raw_xy", "distractor_xy", "goal_rgb"))
    reports, exact_scores, physical_scores = [], [], []
    try:
        for i, fork in enumerate(physical["forks"]):
            task = tasks[fork["task"]]
            goal_rgb = goals[i]
            np.testing.assert_array_equal(history_goals[i], np.stack([goal_rgb]*2))
            goal_latent = native.encode(goal_rgb)[None]
            root = np.asarray(task.start, dtype=np.float32)
            paths = np.stack([physics.rollout(root, commands[i], lock) for lock in WORLDS])
            np.testing.assert_array_equal(paths[1:], archived_paths[i])
            cost = VisibleMapCost(task, np.asarray(task.goal, dtype=np.float32))(paths[..., -1, :])
            np.testing.assert_array_equal(cost[1:], archived_costs[i])
            physical_scores.append(cost)
            roots = [x["fork_block"] for x in fork["chronology"]["episode_a"]["probes"]]
            reset(env, task, None, goal_rgb)
            quiet = np.zeros(2, dtype=np.float32)
            background = [env.distractor_positions.clone()]
            for t in range(max(roots)*ACTION_BLOCK+LAST_RAW[-1]):
                env.step(quiet)
                background.append(env.distractor_positions.clone())
            actual_background = torch.stack(background[::ACTION_BLOCK])[: distractor_xy.shape[2], 0].numpy()
            np.testing.assert_array_equal(distractor_xy[i], np.stack([actual_background]*2))
            fork_scores = []
            for phase, t in zip(PHASES, roots, strict=True):
                for mode in (0, 1):
                    for j in range(t-2, t+1):
                        env.distractor_positions = background[j*ACTION_BLOCK].clone()
                        np.testing.assert_array_equal(render_xy(env, history_xy[i, mode, j*ACTION_BLOCK]),
                                                      history_rgb[i, mode, j])
                score, checks = terminal_readouts(task, t, paths, commands[i], native, env, check_env,
                                                  goal_rgb, goal_latent, native.encode(history_rgb[i, 0, t]), background)
                fork_scores.append(score)
                true_route_indices = cost[1:].argmin(axis=-1)
                uniform_physical = int(cost[1:].mean(axis=0).argmin())
                assert uniform_physical == fork["uniform_index"]
                uniform_latent = int(score[1:].mean(axis=0).argmin())
                open_latent = int(score[0].argmin())
                report = {"task": task.name, "phase": phase, "root_block": t,
                          "render_checks": checks, "distractor_xy_at_raw_55_and_60":
                              [background[t*ACTION_BLOCK+c][0].tolist() for c in LAST_RAW],
                          "open_model": outcome(cost[0], commands[i], open_latent), "true_worlds": []}
                for mode in (0, 1):
                    effect = cost[mode+1]
                    archived = frozen[(task.name, phase, mode)]
                    previous = outcome(effect, commands[i], archived["selected"])
                    np.testing.assert_allclose(previous["first_block_regret"], archived["first_block_regret_by_mode"][mode], atol=1e-6, rtol=0)
                    np.testing.assert_allclose(previous["full_plan_regret"], archived["full_plan_regret_by_mode"][mode], atol=1e-6, rtol=0)
                    true_latent = int(score[mode+1].argmin())
                    report["true_worlds"].append({
                        "mode": mode, "lock": LOCKS[mode], "frozen_free_latent": previous,
                        "exact_true_latent": outcome(effect, commands[i], true_latent),
                        "exact_open_latent": outcome(effect, commands[i], open_latent),
                        "exact_uniform_latent": outcome(effect, commands[i], uniform_latent),
                        "exact_uniform_physical": outcome(effect, commands[i], uniform_physical),
                        "exact_true_physical": outcome(effect, commands[i], true_route_indices[mode]),
                        "exact_true_latent_wrong_route_minus_right": float(score[mode+1, true_route_indices[1-mode]]
                                                                         - score[mode+1, true_route_indices[mode]]),
                    })
                reports.append(report)
                print(task.name, t, "open",
                      round(report["open_model"]["first_block_regret"], 3), "true",
                      [round(row["exact_true_latent"]["first_block_regret"], 3) for row in report["true_worlds"]],
                      flush=True)
            exact_scores.append(np.stack(fork_scores))
    finally:
        env.close()
        check_env.close()
    rows = [row for report in reports for row in report["true_worlds"]]
    names = ("frozen_free_latent", "exact_true_latent", "exact_open_latent", "exact_uniform_latent",
             "exact_uniform_physical", "exact_true_physical")
    avg = {name: {key: float(np.mean([row[name][key] for row in rows]))
                  for key in ("first_block_regret", "full_plan_regret")} for name in names}
    avg["exact_open_latent_evaluated_open"] = {key: float(np.mean([r["open_model"][key] for r in reports]))
                                               for key in ("first_block_regret", "full_plan_regret")}
    assert abs(avg["frozen_free_latent"]["first_block_regret"] - 75.0963393897) < 1e-5
    assert abs(avg["exact_uniform_physical"]["first_block_regret"] - 32.7303771973) < 1e-5
    np.savez_compressed(HERE / "goal_cost_audit_scores.npz", score_exact_rgb=np.stack(exact_scores),
                        score_physical=np.stack(physical_scores))
    write_json(HERE / "goal_cost_audit.json", {
        "score_axes": ["fork", "phase", "world (open, bottom lock, top lock)", "candidate"],
        "physical_score_axes": ["fork", "world (open, bottom lock, top lock)", "candidate"],
        "actual_rgb_forecast_raw_steps": LAST_RAW, "candidates_per_bank": int(commands.shape[1]),
        "frozen_weight_derivatives": 0, "oracle_score": "official last-2 summed squared latent distance",
        "average": avg, "rows": reports, "wall_seconds": time.perf_counter()-started})
    print(avg, flush=True)


if __name__ == "__main__":
    main()
