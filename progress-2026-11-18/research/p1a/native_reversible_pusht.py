"""Full-view PushT with an invisible, reversible one-sided brake on the T."""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from gym_pusht.envs.pusht import PushTEnv
from pymunk import Body, Vec2d


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "runs/native_reversible_pusht_20261115"
IMAGE_SIZE = 96
ACTION_BLOCK = 5
ACTION_SCALE = 100.0
ACTION_BOUND = 0.30
BRAKE_Y = (160.0, 340.0)
BRAKE_KEEP = 0.10
SPLITS = {"fit": (55110, 24), "development": (55210, 8), "heldout": (55310, 12)}
CANDIDATES = ("push", "rising_push", "falling_push", "hold", "retreat", "return_push")
PLANNER_SEEDS = (55430, 55431, 55432, 55433)
ORACLE_CLOSED_LOOP_OUTPUT = ROOT / "runs/native_full_depth_task_geometry_20261115"
ORACLE_CLOSED_LOOP_SEEDS = (55730, 55731, 55732, 55733)
ORACLE_CLOSED_LOOP_SCHEDULE = (1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1)
ORACLE_CLOSED_LOOP_CEM = dict(horizon=2, samples=16, iterations=3, elites=4)


def constant_action(x, y):
    return np.tile(np.array([x, y], dtype=np.float32), (ACTION_BLOCK, 1))


PUSH = constant_action(0.24, 0.0)
HOLD = constant_action(0.0, 0.0)
RETREAT = constant_action(-0.21, 0.0)


@dataclass(frozen=True)
class Scene:
    reset: np.ndarray
    goal: np.ndarray


def make_scene(seed):
    rng = np.random.default_rng(seed)
    x, y, angle = rng.uniform([165.0, 177.0, -0.18], [245.0, 262.0, 0.18])
    approach = rng.uniform(33.0, 45.0)
    local_y = rng.uniform(44.0, 54.0)
    return Scene(np.array([x - approach, y + local_y, x, y, angle]),
                 np.array([x + 35.0, y, angle]))


class ReversiblePushT(PushTEnv):
    """Only RGB enters the planner. State and intervention hooks serve the experiment runner."""

    def __init__(self):
        super().__init__(obs_type="state", render_mode="rgb_array",
                         observation_width=IMAGE_SIZE, observation_height=IMAGE_SIZE)
        self._mode = 0

    def _integrate_t(self, body, dt):
        vx, vy = body.velocity
        if self._mode == 1 and vx > 0.0 and BRAKE_Y[0] <= body.position.y <= BRAKE_Y[1]:
            body.velocity = vx * BRAKE_KEEP, vy
            self.brake_substeps += 1
        Body.update_position(body, dt)

    def set_regime(self, mode):
        """Apply a hidden intervention without modifying any position, velocity or visual."""
        if mode not in (0, 1):
            raise ValueError("Unknown PushT regime")
        self._mode = mode

    def start(self, scene, mode):
        self.set_regime(mode)
        super().reset(seed=0, options={"reset_to_state": scene.reset})
        self.goal_pose = scene.goal.copy()
        self.block.position_func = self._integrate_t
        self.brake_substeps = 0
        return self.observe()

    def observe(self):
        return self._render(visualize=False)

    def goal_image(self, scene):
        """Render the supplied task's goal pose at a canonical left-side contact position."""
        goal_env = PushTEnv(obs_type="state", render_mode="rgb_array",
                            observation_width=IMAGE_SIZE, observation_height=IMAGE_SIZE)
        try:
            x, y, angle = scene.goal
            goal_env.reset(seed=0, options={"reset_to_state": [x - 33.0, y + 48.0, x, y, angle]})
            goal_env.block.position = x, y
            goal_env.space.reindex_shapes_for_body(goal_env.block)
            goal_env.goal_pose = scene.goal.copy()
            return goal_env._render(visualize=False)
        finally:
            goal_env.close()

    def execute(self, block, *, observe_step=None):
        """Execute a full LeWM tick, returning its full-view RGB observation only."""
        block = np.asarray(block)
        if block.shape != (ACTION_BLOCK, 2) or not np.isfinite(block).all() or np.max(np.abs(block)) > ACTION_BOUND + 1e-7:
            raise ValueError("Expected five finite bounded relative PushT commands")
        frame = None
        for command in block:
            self._step_relative(command)
            if observe_step is not None:
                frame = self.observe()
                observe_step(frame)
        return self.observe() if frame is None else frame

    def _step_relative(self, command):
        target = np.clip(np.asarray(self.agent.position) + ACTION_SCALE * command, 0.0, 512.0)
        # Continue through a goal hit so fixed-time offline branches remain paired.
        super().step(Vec2d(*target))


def candidate_actions():
    controls = np.asarray([PUSH, constant_action(0.20, -0.10), constant_action(0.20, 0.10), HOLD, RETREAT, PUSH])
    plans = np.repeat(controls[:, None], 4, axis=1)
    plans[5, 0] = RETREAT
    return plans


def collect_split(output, label):
    first, count = SPLITS[label]
    scene_ids = np.arange(first, first + count)
    plans = candidate_actions()
    n_candidates, horizon = plans.shape[:2]
    shape = count, 2, n_candidates, horizon + 1
    rgb = np.empty((*shape, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    state = np.empty((*shape, 5), dtype=np.float32)
    coverage = np.empty(shape, dtype=np.float32)
    brake = np.zeros(shape, dtype=np.int16)
    goal = np.empty((count, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    env = ReversiblePushT()
    try:
        for row, seed in enumerate(scene_ids):
            scene = make_scene(seed)
            goal[row] = env.goal_image(scene)
            for mode, candidate in np.ndindex((2, n_candidates)):
                rgb[row, mode, candidate, 0] = env.start(scene, mode)
                state[row, mode, candidate, 0] = env.get_obs()
                coverage[row, mode, candidate, 0] = env._get_coverage()
                for tick, block in enumerate(plans[candidate], 1):
                    previous = env.brake_substeps
                    rgb[row, mode, candidate, tick] = env.execute(block)
                    state[row, mode, candidate, tick] = env.get_obs()
                    coverage[row, mode, candidate, tick] = env._get_coverage()
                    brake[row, mode, candidate, tick] = env.brake_substeps - previous
    finally:
        env.close()
    np.testing.assert_array_equal(rgb[:, :, :, 0], np.broadcast_to(rgb[:, :1, :1, 0], rgb[:, :, :, 0].shape))
    np.testing.assert_array_equal(state[:, :, :, 0], np.broadcast_to(state[:, :1, :1, 0], state[:, :, :, 0].shape))
    np.testing.assert_array_equal(rgb[:, 0, [3, 4]], rgb[:, 1, [3, 4]])
    np.testing.assert_array_equal(state[:, 0, [3, 4]], state[:, 1, [3, 4]])
    np.testing.assert_array_equal(brake[:, 0], 0)
    hold = rgb[:, 0, 3]
    hold_render_settling = (hold != hold[:, :1]).any(axis=(1, 2, 3, 4))
    hold_state = state[:, 0, 3, :, 2:4]
    hold_shift = np.linalg.norm(hold_state - hold_state[:, :1], axis=-1)
    push_gap = state[:, 0, 0, :, 2] - state[:, 1, 0, :, 2]
    if np.any(push_gap[:, 1] <= 0.0):
        raise AssertionError(f"A one-tick contact does not distinguish all {label} roots: {push_gap[:, 1]}")
    path = output / f"{label}.npz"
    np.savez_compressed(path, scene_id=scene_ids, candidate=np.asarray(CANDIDATES),
                        regime=np.arange(2), actions=np.broadcast_to(plans, (count, *plans.shape)),
                        rgb=rgb, state=state, brake_substeps=brake, coverage=coverage, goal_rgb=goal)
    return {"scene_range": [int(scene_ids[0]), int(scene_ids[-1])], "scenes": count,
            "root_rgb_and_visible_state_exact": True, "hold_and_retreat_exact": True,
            "hold_scenes_with_native_reset_settling": int(hold_render_settling.sum()),
            "hold_max_object_xy_reset_settling": float(hold_shift.max()),
            "push_gap_x_min_by_tick": push_gap.min(0).tolist(),
            "push_gap_x_median_by_tick": np.median(push_gap, axis=0).tolist()}


def replay(env, scene, history, mode):
    initial_mode = history[0][0] if history else mode
    env.start(scene, initial_mode)
    for past_mode, command in history:
        env.set_regime(past_mode)
        env.execute(command)
    env.set_regime(mode)


def lifecycle(output):
    # The centerline runs through the T's center of gravity. It keeps consecutive
    # probes in the brake band while the separate paired splits vary pose.
    scene = Scene(np.array([162.0, 295.0, 200.0, 250.0, 0.0]),
                  np.array([235.0, 250.0, 0.0]))
    env, alternative = ReversiblePushT(), ReversiblePushT()
    phases = ["probe", "retreat", "dormant_1", "dormant_2", "dormant_3",
              "return", "recontact", "reverse_open", "reverse_brake"]
    history, records, rgb, state, contrasts, opposite_rgb, raw_gap = [], [], [], [], [], [], []
    try:
        rgb.append(env.start(scene, 1))
        state.append(env.get_obs())
        for phase in phases:
            if phase == "retreat":
                control = RETREAT
            elif phase.startswith("dormant"):
                control = HOLD
            else:
                control = PUSH
            mode = 0 if phase == "reverse_open" else 1
            previous_frame, previous_state = env.observe(), env.get_obs()
            env.set_regime(mode)
            np.testing.assert_array_equal(previous_frame, env.observe())
            np.testing.assert_array_equal(previous_state, env.get_obs())
            replay(alternative, scene, history, 1 - mode)
            np.testing.assert_array_equal(previous_frame, alternative.observe())
            np.testing.assert_array_equal(previous_state, alternative.get_obs())
            before_brake = env.brake_substeps
            tick_gap = []
            for command in control:
                env._step_relative(command)
                alternative._step_relative(command)
                tick_gap.append(env.block.position.x - alternative.block.position.x)
            real_frame = env.observe()
            alternative_frame = alternative.observe()
            after, other = env.get_obs(), alternative.get_obs()
            shift = after[2:4] - previous_state[2:4]
            gap = after[2:4] - other[2:4]
            identical = bool(np.array_equal(after, other) and np.array_equal(real_frame, alternative_frame))
            records.append({"phase": phase, "regime": mode, "block_shift_xy": shift.tolist(),
                            "actual_minus_opposite_xy": gap.tolist(), "actual_equals_opposite": identical,
                            "raw_push_gap_x": tick_gap,
                            "brake_substeps": env.brake_substeps - before_brake})
            history.append((mode, control.copy()))
            rgb.append(real_frame)
            state.append(after)
            contrasts.append(other)
            opposite_rgb.append(alternative_frame)
            raw_gap.append(tick_gap)
    finally:
        env.close()
        alternative.close()
    by_phase = {record["phase"]: record for record in records}
    for phase in ("dormant_1", "dormant_2", "dormant_3"):
        if not by_phase[phase]["actual_equals_opposite"]:
            raise AssertionError(f"Unexpected brake effect on {phase}")
    for phase in ("probe", "return", "recontact", "reverse_open", "reverse_brake"):
        record = by_phase[phase]
        expected_sign = 1 if record["regime"] == 0 else -1
        if record["actual_equals_opposite"] or record["actual_minus_opposite_xy"][0] * expected_sign <= 0:
            raise AssertionError(f"Missing signed brake evidence on {phase}: {records}")
    np.savez_compressed(output / "lifecycle.npz", phase=np.asarray(phases),
                        mode=np.asarray([m for m, _ in history]), action=np.stack([a for _, a in history]),
                        rgb=np.stack(rgb), state=np.stack(state), opposite_state=np.stack(contrasts),
                        opposite_rgb=np.stack(opposite_rgb), raw_gap_x=np.stack(raw_gap), goal_rgb=env.goal_image(scene))
    return records


def physical_contract():
    """Exercise identical stock dynamics in open mode and real contact outside the brake band."""
    scene = make_scene(55420)
    env, reference = ReversiblePushT(), PushTEnv(obs_type="state", render_mode="rgb_array")
    try:
        initial = env.start(scene, 0)
        reference.reset(seed=0, options={"reset_to_state": scene.reset})
        reference.goal_pose = scene.goal.copy()
        np.testing.assert_array_equal(initial, reference._render(visualize=False))
        for block in (PUSH, HOLD, constant_action(0.10, -0.15), RETREAT):
            actual = env.execute(block)
            for command in block:
                target = np.clip(np.asarray(reference.agent.position) + ACTION_SCALE * command, 0.0, 512.0)
                reference.step(Vec2d(*target))
            np.testing.assert_array_equal(actual, reference._render(visualize=False))
            np.testing.assert_array_equal(env.get_obs(), reference.get_obs())
    finally:
        env.close()
        reference.close()

    env = ReversiblePushT()
    outside = Scene(np.array([162.0, 145.0, 200.0, 100.0, 0.0]),
                    np.array([235.0, 100.0, 0.0]))
    try:
        env.start(outside, 0)
        initial = env.get_obs()
        open_rgb = env.execute(PUSH)
        open_state = env.get_obs()
        env.start(outside, 1)
        braked_rgb = env.execute(PUSH)
        np.testing.assert_array_equal(open_rgb, braked_rgb)
        np.testing.assert_array_equal(open_state, env.get_obs())
        displacement = float(open_state[2] - initial[2])
        if displacement <= 0 or env.brake_substeps:
            raise AssertionError("A push outside the band must move the block freely")
    finally:
        env.close()
    return {"open_equals_stock_dynamics": True, "outside_band_equals_open": True,
            "outside_band_actual_x_shift": displacement}


def cem(cost, rng, *, horizon=2, samples=16, iterations=3, elites=4):
    """Choose a sequence of constant per-tick controls via common seeded CEM."""
    mean = np.zeros((horizon, 2), dtype=np.float32)
    spread = np.full_like(mean, 0.18)
    best_cost = np.inf
    best_plan = None
    for _ in range(iterations):
        sample = rng.normal(mean, spread, (samples, horizon, 2)).astype(np.float32)
        sample[0] = mean if best_plan is None else best_plan[:, 0]
        actions = np.repeat(np.clip(sample, -ACTION_BOUND, ACTION_BOUND)[:, :, None], ACTION_BLOCK, axis=2)
        scores = np.asarray(cost(actions))
        if scores.shape != (samples,) or not np.isfinite(scores).all():
            raise ValueError("CEM expects one finite scalar cost per candidate")
        order = np.argsort(scores, kind="stable")[:elites]
        if scores[order[0]] < best_cost:
            best_cost = scores[order[0]]
            best_plan = actions[order[0]].copy()
        chosen = actions[order, :, 0]
        mean = chosen.mean(axis=0)
        spread = np.maximum(chosen.std(axis=0), 0.04)
    return best_plan, float(best_cost)


def native_cost(model, pixels, goal, past_actions, *, normalize):
    """Bind legal current pixels/goal/history to the original LeWM get_cost interface."""
    import torch

    frames = np.asarray(pixels)
    past = np.asarray(past_actions, dtype=np.float32).reshape(len(frames) - 1, ACTION_BLOCK, 2)
    if not 1 <= len(frames) <= 3 or frames.shape[1:] != (IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("LeWM requires one to three full-view RGB frames")
    from torchvision.transforms.functional import resize

    device = next(model.parameters()).device
    rgb = torch.as_tensor(np.concatenate((frames, np.asarray(goal)[None])), device=device)
    rgb = resize(rgb.permute(0, 3, 1, 2).float() / 255, [224, 224], antialias=True)
    mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
    std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
    rgb = (rgb - mean) / std
    frame_tensors, goal_tensor = rgb[:-1], rgb[-1:]

    @torch.inference_mode()
    def cost(plans):
        n = len(plans)
        prefix = np.broadcast_to(past, (n, *past.shape))
        entire = torch.as_tensor(np.concatenate((prefix, plans), axis=1), device=device)
        entire = normalize(entire).flatten(-2)[None]
        info = {"pixels": frame_tensors[None, None].expand(1, n, -1, -1, -1, -1),
                "goal": goal_tensor[None, None].expand(1, n, -1, -1, -1, -1),
                "action": entire}
        scores = model.get_cost(info, entire)
        return scores[0].detach().cpu().numpy()

    return cost


def run_mpc(seed, score=None, *, schedule=(1, 1, 1, 0, 0, 1, 1), horizon=2,
            samples=16, iterations=3, elites=4, fixed_oracle_regime=None,
            uniform_oracle=False, begin_execution=None):
    """Run one full episode with fixed CEM. A supplied score sees only legal inputs.

    score(frames[T+1,96,96,3], executed[T,5,2], goal[96,96,3], plans[N,H,5,2])
    returns float costs[N]. With no supplied score, use a privileged physical oracle
    that branches live dynamics under the true *current* regime for this horizon.
    The uniform oracle averages coverage from separate open and braked rollouts.
    An optional execution callback receives the chosen block and returns an RGB observer.
    """
    if score is not None and (fixed_oracle_regime is not None or uniform_oracle):
        raise ValueError("A legal score cannot also select a physical oracle regime")
    if uniform_oracle and fixed_oracle_regime is not None:
        raise ValueError("The uniform physical oracle cannot also fix one regime")
    scene = make_scene(seed)
    env = ReversiblePushT()
    fork = ReversiblePushT() if score is None else None
    history, records, states = [], [], []
    try:
        frames = [env.start(scene, schedule[0])]
        goal = env.goal_image(scene)
        states.append(env.get_obs())
        coverage = [env._get_coverage()]
        for index, mode in enumerate(schedule):
            env.set_regime(mode)
            assumed = mode if fixed_oracle_regime is None else fixed_oracle_regime
            regimes = (0, 1) if uniform_oracle else (assumed,)

            def cost(actions):
                if score is not None:
                    actual_commands = np.asarray([command for _, command in history], dtype=np.float32)
                    actual_commands = actual_commands.reshape(index, ACTION_BLOCK, 2)
                    return score(np.stack(frames), actual_commands, goal.copy(), actions)
                values = []
                for plan in actions:
                    covered = 0.0
                    for regime in regimes:
                        replay(fork, scene, history, regime)
                        for block in plan:
                            fork.execute(block)
                        covered += fork._get_coverage()
                    values.append(1.0 - covered / len(regimes))
                return values

            plan, terminal_cost = cem(cost, np.random.default_rng(seed + index), horizon=horizon,
                                      samples=samples, iterations=iterations, elites=elites)
            before_brake = env.brake_substeps
            observe_step = None if begin_execution is None else begin_execution(plan[0].copy())
            frames.append(env.execute(plan[0], observe_step=observe_step))
            history.append((mode, plan[0].copy()))
            states.append(env.get_obs())
            coverage.append(env._get_coverage())
            records.append({"tick": index + 1, "regime": mode, "coverage_before": coverage[-2],
                            "coverage_after": coverage[-1], "planned_terminal_cost": terminal_cost,
                            "brake_substeps": env.brake_substeps - before_brake,
                            "executed_relative_action": plan[0, 0].tolist()})
    finally:
        env.close()
        if fork is not None:
            fork.close()
    trace = {"rgb": np.stack(frames), "state": np.stack(states), "goal_rgb": goal,
             "actions": np.stack([command for _, command in history]), "regime": np.asarray(schedule),
             "coverage": np.asarray(coverage), "success": np.asarray(max(coverage) > env.success_threshold)}
    return records, trace


def oracle_smoke(output):
    policies = (("true_current_mode", None), ("always_open", 0))
    result = {"horizon": 2, "samples": 16, "iterations": 3, "elites": 4,
              "schedule": [1, 1, 1, 0, 0, 1, 1], "scenes": {}}
    traces = {}
    for seed in PLANNER_SEEDS:
        result["scenes"][str(seed)] = {}
        for policy, fixed in policies:
            records, trace = run_mpc(seed, fixed_oracle_regime=fixed)
            result["scenes"][str(seed)][policy] = {
                "success": bool(trace["success"]), "best_coverage": float(trace["coverage"].max()),
                "final_coverage": float(trace["coverage"][-1]), "ticks": records}
            traces.update({f"{seed}_{policy}_{k}": v for k, v in trace.items()})
    np.savez_compressed(output / "oracle-smoke.npz", **traces)
    return result


def oracle_closed_loop_scene(seed):
    results = {}
    for uniform, policy in ((False, "true_current_mode"), (True, "uniform_current_modes")):
        start = time.perf_counter()
        records, trace = run_mpc(seed, schedule=ORACLE_CLOSED_LOOP_SCHEDULE, uniform_oracle=uniform,
                                 **ORACLE_CLOSED_LOOP_CEM)
        seconds = time.perf_counter() - start
        results[policy] = {"success": bool(trace["success"]), "best_coverage": float(trace["coverage"].max()),
                           "final_coverage": float(trace["coverage"][-1]), "seconds": seconds,
                           "coverage": trace["coverage"].tolist(), "state": trace["state"].tolist(),
                           "actions": trace["actions"].tolist(), "ticks": records}
        print(seed, policy, f"success={bool(trace['success'])}",
              f"final={trace['coverage'][-1]:.6f}", f"seconds={seconds:.1f}", flush=True)
    return seed, results


def oracle_closed_loop():
    from concurrent.futures import ProcessPoolExecutor

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=len(ORACLE_CLOSED_LOOP_SEEDS)) as executor:
        scenes = dict(executor.map(oracle_closed_loop_scene, ORACLE_CLOSED_LOOP_SEEDS))
    first = scenes[ORACLE_CLOSED_LOOP_SEEDS[0]]
    policy_summary = {}
    for policy_index, policy in enumerate(first, 1):
        outcomes = [scene[policy] for scene in scenes.values()]
        candidates = len(ORACLE_CLOSED_LOOP_SCHEDULE) * ORACLE_CLOSED_LOOP_CEM["samples"] * ORACLE_CLOSED_LOOP_CEM["iterations"]
        policy_summary[policy] = {"successes": sum(scene["success"] for scene in outcomes),
                                  "mean_best_coverage": float(np.mean([scene["best_coverage"] for scene in outcomes])),
                                  "mean_final_coverage": float(np.mean([scene["final_coverage"] for scene in outcomes])),
                                  "median_episode_seconds": float(np.median([scene["seconds"] for scene in outcomes])),
                                  "cem_candidates_per_episode": candidates,
                                  "physical_candidate_rollouts_per_episode": policy_index * candidates}
    return {**ORACLE_CLOSED_LOOP_CEM, "seeds": ORACLE_CLOSED_LOOP_SEEDS,
            "schedule": ORACLE_CLOSED_LOOP_SCHEDULE, "seconds": time.perf_counter() - start,
            "summary": policy_summary, "scenes": scenes}


def frozen_smoke(output):
    """Evaluate the released checkpoint with its native terminal latent cost."""
    import sys
    import torch

    sys.path.insert(0, str(ROOT / "research_sources_2026-10-22/lewm_assets"))
    from prepare_assets import load_model

    torch.set_num_threads(3)
    model = load_model("cpu").eval().requires_grad_(False)
    norm = ROOT / "oic_step1/runs/pusht_continuation_2026-10-31/action_normalization.json"
    config = json.loads(norm.read_text(encoding="utf-8"))

    def normalize(actions):
        return (actions - actions.new_tensor(config["mean"])) / actions.new_tensor(config["std"])

    result, traces = {}, {}
    for seed in PLANNER_SEEDS:
        previous_length = 0
        scorer = None

        def score(frames, executed, goal, plans):
            nonlocal previous_length, scorer
            if len(frames) != previous_length:
                previous_length = len(frames)
                missing = max(0, 3 - previous_length)
                latest = np.concatenate((np.repeat(frames[:1], missing, axis=0), frames[-3:]))
                past = np.concatenate((np.repeat(HOLD[None], missing, axis=0), executed[-2:]))
                scorer = native_cost(model, latest, goal, past, normalize=normalize)
            return scorer(plans)

        records, trace = run_mpc(seed, score)
        result[str(seed)] = {"success": bool(trace["success"]), "best_coverage": float(trace["coverage"].max()),
                             "final_coverage": float(trace["coverage"][-1]), "ticks": records}
        traces.update({f"{seed}_{k}": v for k, v in trace.items()})
    np.savez_compressed(output / "frozen-smoke.npz", **traces)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", nargs="+", choices=[*SPLITS, "physical-contract", "lifecycle",
                                                     "oracle-smoke", "oracle-closed-loop", "frozen-smoke"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for phase in args.phase:
        output = args.output or (ORACLE_CLOSED_LOOP_OUTPUT if phase == "oracle-closed-loop" else OUTPUT)
        output.mkdir(parents=True, exist_ok=True)
        filename = "oracle_closed_loop.json" if phase == "oracle-closed-loop" else f"{phase}.json"
        if phase == "physical-contract":
            result = physical_contract()
        elif phase == "lifecycle":
            result = lifecycle(output)
        elif phase == "oracle-smoke":
            result = oracle_smoke(output)
        elif phase == "oracle-closed-loop":
            result = oracle_closed_loop()
        elif phase == "frozen-smoke":
            result = frozen_smoke(output)
        else:
            result = collect_split(output, phase)
        (output / filename).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if phase.endswith("smoke") or phase == "oracle-closed-loop":
            print(phase, output / filename, flush=True)
        else:
            print(phase, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
