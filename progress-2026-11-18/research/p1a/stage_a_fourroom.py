"""Qualify the frozen FourRoom host on the author's original door geometry."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from stable_worldmodel.envs.four_room.env import DOOR_ORDER, FourRoomDistractorsEnv

from .native_topology_condition import forecast, load_host, load_readout
from .physical_topology_gate import (
    ACTION_BLOCK, CENTER, DOOR_CLEARANCE, GOAL_RADIUS, RADIUS, SPEED,
    VisibleMapCost, door_xy, doors_between, room,
)


ROOT = Path(__file__).resolve().parents[2]
WORLD_CONFIG = ROOT / 'runs/hjepa_official_fourroom_flat_20261115/fourroom_flat.author.yaml'
DOORS = dict(zip(DOOR_ORDER, (56, 168, 56, 168), strict=True))
HORIZON = 12
SHORT_FORK_HORIZON = 5
REPLANS = 12
CANDIDATES = 384
MIN_SUCCESS = .70
MAX_GEOMETRY_RATE = .05
SEED = 2026111700
LOCKS = ('bottom', 'top')
FIXED_DISTRACTOR_XY = (190., 190.)


@dataclass(frozen=True)
class Task:
    name: str
    family: str
    start: tuple[float, float]
    goal: tuple[float, float]
    seed: int
    required_doors: tuple[str, ...]
    door_positions = DOORS


def door_pairs():
    for door, center in DOORS.items():
        axis = int(door in ('left', 'right'))
        for sign in (1, -1):
            pair = np.empty((2, 2), dtype=np.float32)
            pair[:, 1 - axis] = center
            pair[:, axis] = CENTER + sign * np.array((-12.5, 27.5), dtype=np.float32)
            yield door, axis, sign, pair


def fixed_tasks():
    """Sixteen directed single-door goals and eight opposite-room goals."""
    tasks = []
    for door, axis, sign, pair in door_pairs():
        for offset in (-5, 5):
            points = pair.copy()
            points[:, 1 - axis] += offset
            tasks.append(Task(f'{door}_{sign:+d}_{offset:+d}', 'one_door', *map(tuple, points.tolist()),
                              SEED + len(tasks), (door,)))
    corners = ((80., 80.), (144., 80.), (144., 144.), (80., 144.))
    for source, start in enumerate(corners):
        goal = corners[(source + 2) % 4]
        for repeat in range(2):
            tasks.append(Task(f'opposite_{source}_{repeat}', 'two_doors', start, goal,
                              SEED + len(tasks), ()))
    return tuple(tasks)


class OfficialFourRoom(FourRoomDistractorsEnv):
    """The author's distractor environment, with a collision-only ego intervention."""

    def __init__(self, *, fixed_distractor=False):
        self.ego_lock = None
        self.fixed_distractor = fixed_distractor
        world = OmegaConf.to_container(OmegaConf.load(WORLD_CONFIG).world, resolve=False)
        for key in ('env_name', 'num_envs', 'max_episode_steps', 'history_size', 'frame_skip'):
            world.pop(key)
        if world['door_positions'] != DOORS:
            raise ValueError('Expected the original four door positions')
        if fixed_distractor:
            world['distractor_policy'] = 'static'
            world['distractor_explore']['teleport_tau'] = None
        super().__init__(**(world | {'close_door_prob': 0., 'min_distractors': 1,
                                     'max_distractors': 1, 'terminate_on_goal': False}))

    def reset(self, seed=None, options=None):
        if self.fixed_distractor:
            options = dict(options or {}, distractor_xy=FIXED_DISTRACTOR_XY)
        return super().reset(seed=seed, options=options)

    def _apply_collisions(self, before, proposed):
        if self.ego_lock is None or before is not self.agent_position:
            return super()._apply_collisions(before, proposed)
        self.closed_door = self.ego_lock
        try:
            return super()._apply_collisions(before, proposed)
        finally:
            self.closed_door = None

    @property
    def xy(self):
        return self.agent_position.numpy().copy()

    def execute(self, commands):
        commands = np.asarray(commands, dtype=np.float32)
        if commands.shape != (ACTION_BLOCK, 2) or not np.isfinite(commands).all() or np.abs(commands).max() > 1:
            raise ValueError('Expected five legal independently parameterized raw actions')
        path = []
        for action in commands:
            _, _, terminated, truncated, info = self.step(action)
            if terminated or truncated:
                raise RuntimeError('Unexpected simulator termination in fixed-action episode')
            path.append(info['xy'])
        return np.stack(path)


class OpenPhysics:
    """Batched all-open author collision equations for the physics comparator."""

    def __init__(self, env):
        self.near, self.far = CENTER - env.wall_thickness // 2 - RADIUS, CENTER + env.wall_thickness // 2 + RADIUS
        self.bounds = (env.BORDER_SIZE + RADIUS, env.IMG_SIZE - env.BORDER_SIZE - RADIUS)
        self.aperture = env.door_sizes.numpy() + DOOR_CLEARANCE

    def advance(self, xy, action, lock=None):
        proposal = np.clip(xy + action * SPEED, *self.bounds)
        x, y = proposal[..., 0], proposal[..., 1]
        for axis, coordinate, tangent, names, width in (
            (0, x, y, ('top', 'bottom'), self.aperture[:2]),
            (1, y, x, ('left', 'right'), self.aperture[2:]),
        ):
            aperture = np.abs(tangent[..., None] - np.array([DOORS[name] for name in names])) <= width
            clear = (aperture & np.array([name != lock for name in names])).any(-1)
            from_near = xy[..., axis] < CENTER
            coordinate[from_near & (coordinate > self.near) & ~clear] = self.near - .5
            coordinate[~from_near & (coordinate < self.far) & ~clear] = self.far + .5
        return proposal

    def rollout(self, xy, plans, lock=None):
        position = np.broadcast_to(xy, (len(plans), 2))
        path = []
        for block in plans.transpose(1, 0, 2, 3):
            for raw in block.transpose(1, 0, 2):
                position = self.advance(position, raw, lock)
                path.append(position)
        return np.stack(path, axis=1)

    def wall_penetration(self, xy):
        return np.stack([
            (xy[..., axis] > self.near) & (xy[..., axis] < self.far) &
            (np.abs(xy[..., 1 - axis, None] - np.array([56, 168])) > width).all(-1)
            for axis, width in enumerate((self.aperture[:2], self.aperture[2:]))
        ], axis=-1)


def route_controls(task, xy, direction, horizon):
    """Generate one proposal per direction around the original open map."""
    goal = np.asarray(task.goal, dtype=np.float32)
    source = int(room(xy))
    points = []
    for step, name in enumerate(doors_between(source, int(room(goal)), direction)):
        center = door_xy(task, name)
        axis = int(name in ('left', 'right'))
        entry = center.copy()
        entry[axis] += -11.5 if source in ((0, 3) if axis == 0 else (0, 1)) else 11.5
        # After entering a doorway, the next receding plan continues to its exit.
        toward_door = np.sign(center[axis] - entry[axis])
        in_aperture = abs(xy[1 - axis] - center[1 - axis]) <= 16 + DOOR_CLEARANCE
        past_entry = (xy[axis] - entry[axis]) * toward_door >= 0
        if step or not (in_aperture and past_entry):
            points.append(entry)
        points.append(2 * center - entry)
        source = (source + direction) % 4
    points.append(goal)
    position = xy.copy()
    controls = []
    raw_horizon = horizon * ACTION_BLOCK
    for target in points:
        while np.max(np.abs(target - position)) > 1e-4 and len(controls) < raw_horizon:
            action = np.clip((target - position) / SPEED, -1., 1.)
            controls.append(action)
            position += SPEED * action
    controls += [np.zeros(2, dtype=np.float32)] * (raw_horizon - len(controls))
    return np.asarray(controls, dtype=np.float32).reshape(horizon, ACTION_BLOCK, 2)


def candidate_bank(task, nominal, replan, horizon=HORIZON):
    """Shared proposals use issued controls on an open map and no actual state or lock."""
    rng = np.random.default_rng(np.random.SeedSequence((task.seed, replan)))
    templates = np.stack([route_controls(task, nominal, direction, horizon) for direction in (1, -1)])
    candidates = rng.normal(0., .30, (2, CANDIDATES // 2, horizon, ACTION_BLOCK, 2)).astype(np.float32)
    candidates += templates[:, None]
    candidates[:, 0] = templates
    candidates = candidates.reshape(CANDIDATES, horizon, ACTION_BLOCK, 2)
    candidates[-9] = 0.
    headings = np.array([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)])
    candidates[-8:] = headings[:, None, None, :]
    return np.clip(candidates, -1., 1.)


def actual_crossings(path, physics):
    """Classify a traversal only on reaching the far edge of the wall strip."""
    current = np.sign(path[0] - CENTER)
    crossings = []
    for step, point in enumerate(path[1:], 1):
        for axis in (0, 1):
            coordinate, tangent = point[axis], point[1 - axis]
            side = -1 if coordinate <= physics.near else 1 if coordinate >= physics.far else 0
            if not side or side == current[axis]:
                continue
            names = ('top', 'bottom') if axis == 0 else ('left', 'right')
            name = names[int(tangent >= CENTER)]
            crossings.append({'raw_step': step, 'door': name, 'direction': side})
            current[axis] = side
    return crossings


def geometry_diagnostics(physics, start, decoded, selected):
    """Flag invalid free endpoints and impossible crossing under the exact issued actions."""
    previous = np.vstack((start, decoded[:-1]))
    invalid = physics.wall_penetration(decoded)
    invalid_previous = physics.wall_penetration(previous)
    exact = physics.rollout(previous, selected[:, None])[:, -1]
    enters = (decoded >= CENTER) != (previous >= CENTER)
    feasible = (exact >= CENTER) == (decoded >= CENTER)
    false_cross = enters & ~feasible & ~invalid_previous.any(-1, keepdims=True)
    return {'solid_wall_endpoint_axes': invalid.tolist(), 'invalid_start_axes': invalid_previous.tolist(),
            'false_cross_axes': false_cross.tolist(), 'predicted_xy': decoded.tolist(),
            'physical_replay_from_predicted_xy': exact.tolist()}


def latent_goal_cost(host, trajectory, goal):
    """The shipped latent goal criterion, shared by all learned forecasters."""
    return host.level.criterion({'predicted_embed_0': trajectory, 'goal_embed_0': goal[:, None, None],
                                 'cost_last_n': 2})


class Native:
    def __init__(self, device):
        self.host = load_host(device)
        self.decode = load_readout(device)
        self.device = device
        self.mean = torch.tensor((.485, .456, .406), device=device)[:, None, None]
        self.std = torch.tensor((.229, .224, .225), device=device)[:, None, None]

    def encode(self, rgb):
        image = torch.from_numpy(np.ascontiguousarray(rgb)).to(self.device).permute(2, 0, 1).float() / 255.
        return self.host.encode({'pixels': ((image - self.mean) / self.std)[None, None]})['emb'][0, 0]

    def score(self, latents, commands, candidates, goal):
        past = torch.as_tensor(np.asarray(commands[-2:]), device=self.device)[None]
        roll = torch.cat([
            forecast(self.host, None, None, torch.stack(latents[-3:])[None], past,
                     torch.as_tensor(candidate, device=self.device)[None])[:, :, 0]
            for candidate in np.array_split(candidates, 4)
        ], dim=1)
        return latent_goal_cost(self.host, roll, goal[None])[0].cpu().numpy(), roll[0]


def qualification_episode(task, env, physics, native=None):
    goal = np.asarray(task.goal, dtype=np.float32)
    env.reset(seed=task.seed, options={'state': task.start, 'target_state': task.goal})
    nominal = np.asarray(task.start, dtype=np.float32)
    observed = [env.xy]
    commands = [np.zeros((ACTION_BLOCK, 2), dtype=np.float32) for _ in range(2)]
    latents = [native.encode(env.render())] if native else []
    goal_latent = native.encode(env._target_img.permute(1, 2, 0).numpy()) if native else None
    for action in commands:
        observed.extend(env.execute(action))
        if native:
            latents.append(native.encode(env.render()))
    geometry = VisibleMapCost(task, goal)
    decisions = []
    for replan in range(REPLANS):
        candidates = candidate_bank(task, nominal, replan)
        start = env.xy
        physical_paths = physics.rollout(start, candidates)
        physical_costs = geometry(physical_paths[:, -1])
        if native:
            scores, predicted = native.score(latents, commands, candidates, goal_latent)
        else:
            scores = physical_costs
        if not np.isfinite(scores).all():
            raise ValueError(f'Non-finite candidate scores for {task.name}')
        best = int(scores.argmin())
        raw = candidates[best, 0]
        action_path = env.execute(raw)
        np.testing.assert_array_equal(action_path, physical_paths[best, :ACTION_BLOCK])
        nominal = physics.rollout(nominal, candidates[best:best + 1, :1])[0, -1]
        observed.extend(action_path)
        commands.append(raw)
        row = {'replan': replan, 'selected_index': best, 'selected_score': float(scores[best]),
               'physical_lookahead_regret': float(physical_costs[best] - physical_costs.min()),
               'raw_action_block': raw.tolist(), 'actual_raw_xy': action_path.tolist(),
               'best_actual_distance': float(np.linalg.norm(action_path - goal, axis=-1).min())}
        if native:
            actual_latent = native.encode(env.render())
            previous_decode = native.decode(latents[-1]).cpu().numpy()
            predicted_decode = native.decode(predicted[best]).cpu().numpy()
            row['decoded_current_xy'] = previous_decode.tolist()
            actual_decode = native.decode(actual_latent).cpu().numpy()
            row['decoded_actual_arrival_xy'] = actual_decode.tolist()
            row['decoded_actual_arrival_wall_axes'] = physics.wall_penetration(actual_decode).tolist()
            row['geometry'] = geometry_diagnostics(physics, previous_decode, predicted_decode, candidates[best])
            latents.append(actual_latent)
        decisions.append(row)
        if row['best_actual_distance'] < GOAL_RADIUS:
            break
    path = np.asarray(observed)
    crossings = actual_crossings(path, physics)
    required = (tuple(row['door'] for row in crossings) == task.required_doors if task.required_doors
                else len(crossings) >= 2)
    return {'task': task.name, 'family': task.family, 'seed': task.seed, 'start': task.start, 'goal': task.goal,
            'success': bool(required and decisions[-1]['best_actual_distance'] < GOAL_RADIUS),
            'crossings': crossings, 'decisions': decisions}


def phase_summary(rows):
    summary = {}
    for family in ('one_door', 'two_doors', 'all'):
        group = [row for row in rows if family == 'all' or row['family'] == family]
        success = sum(row['success'] for row in group)
        summary[family] = {'successes': success, 'episodes': len(group), 'passed': success >= MIN_SUCCESS * len(group)}
    return summary


def lifecycle_episode(env, physics, door, axis, sign, start, goal, seed, initial_mode,
                      *, mode_schedule=(0, 1, 0), neutral_repetitions=1, anchor=None):
    """Run three matched probes with two mode flips and continuous factual windows."""
    modes = tuple((initial_mode + mode) % 2 for mode in mode_schedule)
    env.ego_lock = LOCKS[modes[0]]
    anchor = start if anchor is None else anchor
    _, info = env.reset(seed=seed, options={'state': anchor, 'target_state': goal})
    frames, positions, distractors = [env.render()], [env.xy], [info['distractor0_xy'].copy()]
    commands, rows, probes = [], [], []

    def execute(raw, event):
        before = env.xy
        path = env.execute(raw)
        commands.append(raw)
        positions.extend(path)
        frames.append(env.render())
        distractors.append(env.distractor_positions.numpy().copy()[0])
        rows.append({'event': event, 'raw_actions': raw.tolist(), 'before': before.tolist(), 'after': path.tolist()})
        return path

    direction = np.zeros((ACTION_BLOCK, 2), dtype=np.float32)
    direction[:, axis] = sign

    def neutral():
        for _ in range(neutral_repetitions):
            first = execute(-direction, 'neutral_away')
            second = execute(direction, 'neutral_return')
            if not np.all(np.linalg.norm(np.diff(np.vstack((anchor, first, second)), axis=0), axis=1) == SPEED):
                raise AssertionError('Neutral movement must advance on all ten primitive commands')
            np.testing.assert_array_equal(env.xy, anchor)

    def move_fixed(target, event):
        for _ in range(2):
            raw = []
            nominal = env.xy
            for _ in range(ACTION_BLOCK):
                action = np.clip((target - nominal) / SPEED, -1., 1.)
                nominal += action * SPEED
                raw.append(action)
            execute(np.stack(raw), event)
        np.testing.assert_array_equal(env.xy, target)

    neutral()
    initial_fork = len(commands)
    for probe_no, mode in enumerate(modes):
        switch_block = None
        if probe_no and mode != modes[probe_no - 1]:
            switch_block = len(commands)
            before_switch = env.render()
            env.ego_lock = LOCKS[mode]
            np.testing.assert_array_equal(before_switch, env.render())
        if not np.array_equal(anchor, start):
            move_fixed(start, 'approach_mouth')
        pre_block = len(commands)
        probe_path = np.vstack((env.xy, execute(direction, 'probe'), execute(direction, 'probe')))
        crossed = actual_crossings(probe_path, physics)
        should_cross = env.ego_lock != door
        expected = [(door, sign)] if should_cross else []
        if [(value['door'], value['direction']) for value in crossed] != expected:
            raise AssertionError(f'Reversible probe did not follow stock {door} collision')
        post_block = len(commands)
        move_fixed(start, 'recover')
        if not np.array_equal(anchor, start):
            move_fixed(anchor, 'return_to_fork')
        neutral()
        probes.append({'probe': probe_no, 'mode': mode, 'lock': env.ego_lock, 'switch_block': switch_block,
                       'pre_block': pre_block, 'post_block': post_block, 'fork_block': len(commands),
                       'crossings': crossed, 'open': should_cross})
    return {'door': door, 'direction': sign, 'seed': seed, 'initial_mode': initial_mode, 'probes': probes,
            'start': start.tolist(), 'goal': goal.tolist(), 'anchor': anchor.tolist(), 'initial_fork_block': initial_fork,
            'neutral_nonzero_raw_steps': 2 * ACTION_BLOCK * neutral_repetitions,
            'neutral_blocks': 2 * neutral_repetitions, 'blocks': rows}, {
                'rgb': np.stack(frames), 'raw_xy': np.stack(positions), 'raw_actions': np.stack(commands),
                'distractor_xy': np.stack(distractors), 'goal_rgb': env._target_img.permute(1, 2, 0).numpy(),
            }


def validate_lifecycle_pair(paired):
    (left, a), (right, b) = paired
    np.testing.assert_array_equal(a['distractor_xy'], b['distractor_xy'])
    np.testing.assert_array_equal(a['goal_rgb'], b['goal_rgb'])
    for probe in left['probes']:
        pre, post, fork = (probe[key] for key in ('pre_block', 'post_block', 'fork_block'))
        for block in (pre, fork):
            np.testing.assert_array_equal(a['rgb'][block - 2:block + 1], b['rgb'][block - 2:block + 1])
            np.testing.assert_array_equal(a['raw_actions'][block - 2:block], b['raw_actions'][block - 2:block])
        np.testing.assert_array_equal(a['raw_actions'][pre:post], b['raw_actions'][pre:post])
        if np.array_equal(a['rgb'][post], b['rgb'][post]):
            raise AssertionError('Actual doorway outcomes must be visible in legally arrived RGB')
    return {'episode_a': left, 'episode_b': right, 'matched_factual_three_frame_roots': True,
            'matched_distractors': True}, {key: np.stack((a[key], b[key])) for key in a}


def lifecycle_suite(env, physics):
    pairs, traces = [], []
    for door, axis, sign, pair in door_pairs():
        if door in LOCKS:
            paired = [lifecycle_episode(env, physics, door, axis, sign, *pair, SEED + 100 + len(pairs), mode)
                      for mode in range(2)]
            row, trace = validate_lifecycle_pair(paired)
            pairs.append(row)
            traces.append(trace)
    env.ego_lock = None
    return pairs, {key: np.stack([trace[key] for trace in traces]) for key in traces[0]}


def physical_contract(env, physics):
    rng = np.random.default_rng(SEED)
    anchors = np.array([(x, y) for x in (103., 103.5, 112., 120.5, 121.)
                        for y in (38.25, 56., 73.75, 150.25, 168., 185.75)], dtype=np.float32)
    xy = np.vstack((anchors, anchors[:, ::-1], rng.uniform(17.5, 206.5, (64, 2)))).astype(np.float32)
    controls = rng.uniform(-1, 1, xy.shape).astype(np.float32)
    for lock in (None, *LOCKS):
        env.ego_lock = lock
        env.reset(seed=SEED)
        expected = physics.advance(xy, controls, lock)
        actual = []
        for position, control in zip(xy, controls, strict=True):
            env._set_state(position)
            actual.append(env.step(control)[-1]['xy'])
        np.testing.assert_array_equal(actual, expected)
    env.ego_lock = None
    return 3 * len(xy)


def decision_bank(task, nominal, replan, physics, actual, horizon=HORIZON):
    """Return lock-blind candidates and separate exact counterfactual labels."""
    candidates = candidate_bank(task, nominal, replan, horizon)
    paths = np.stack([physics.rollout(actual, candidates, lock) for lock in LOCKS])
    cost = VisibleMapCost(task, np.asarray(task.goal, dtype=np.float32))
    return candidates, paths, cost(paths[..., -1, :])


def fork_suite(env, physics):
    """Measure both physical choices at the same two-door, mode-blind fork."""
    rows, traces, histories = [], [], []
    for task in fixed_tasks():
        if task.name not in ('opposite_0_0', 'opposite_0_1', 'opposite_2_0', 'opposite_2_1'):
            continue
        start = np.asarray(task.start, dtype=np.float32)
        goal = np.asarray(task.goal, dtype=np.float32)
        candidates, paths, costs = decision_bank(task, start, 0, physics, start)
        _, _, short_cost = decision_bank(task, start, 0, physics, start, SHORT_FORK_HORIZON)
        short_uniform = short_cost.mean(0).argmin()
        true = costs.argmin(-1)
        uniform = int(costs.mean(0).argmin())
        uniform_regret = costs[:, uniform] - costs.min(-1)
        stale_regret = costs[np.arange(2), true[::-1]] - costs.min(-1)
        paired = []
        actual_crossed = []
        door = 'top' if start[0] < CENTER else 'bottom'
        sign = 1 if door == 'top' else -1
        mouth = np.array((CENTER - sign * 12.5, DOORS[door]), dtype=np.float32)
        for mode in range(2):
            paired.append(lifecycle_episode(env, physics, door, 0, sign, mouth, goal,
                                            task.seed, mode, anchor=start))
            np.testing.assert_array_equal(env.xy, start)
            actual = np.concatenate([env.execute(commands) for commands in candidates[true[mode]]])
            np.testing.assert_array_equal(actual, paths[mode, true[mode]])
            actual_crossed.append(actual_crossings(np.vstack((start, actual)), physics))
        pair, history = validate_lifecycle_pair(paired)
        histories.append(history)
        initial_fork = pair['episode_a']['initial_fork_block']
        first_doors = [crossing[0]['door'] if crossing else None for crossing in actual_crossed]
        different_routes = all(first_doors) and first_doors[0] != first_doors[1]
        different_commands = np.any(candidates[true[0], 0] != candidates[true[1], 0])
        rows.append({'task': task.name, 'start': task.start, 'goal': task.goal, 'seed': task.seed,
                     'true_indices': true.tolist(), 'uniform_index': uniform, 'stale_indices': true[::-1].tolist(),
                     'uniform_regret': uniform_regret.tolist(), 'stale_regret': stale_regret.tolist(),
                     'h5_uniform_regret': (short_cost[:, short_uniform] - short_cost.min(-1)).tolist(),
                     'actual_crossings_by_true_mode': actual_crossed, 'same_factual_window': True,
                     'chronology': pair,
                     'distinct_routes': bool(different_routes and different_commands),
                     'complete_two_door_routes': all(len(crossing) == 2 for crossing in actual_crossed),
                     'positive_uniform_regret': bool(uniform_regret.sum() > 0)})
        traces.append({'raw_candidates': candidates, 'raw_xy_by_mode': paths, 'candidate_cost_by_mode': costs,
                       'rgb': history['rgb'][0, initial_fork - 2:initial_fork + 1],
                       'raw_factual_controls': history['raw_actions'][0, initial_fork - 2:initial_fork],
                       'goal_rgb': history['goal_rgb'][0]})
    env.ego_lock = None
    return (rows, {key: np.stack([trace[key] for trace in traces]) for key in traces[0]},
            {key: np.stack([history[key] for history in histories]) for key in histories[0]})


def protocol(*, fixed_distractor=False):
    settings = {'original_door_positions': DOORS, 'host': 'official FourRoomDistractors + seed42 LeWM',
            'known_open_cases': {'one_door': 16, 'two_doors': 8}, 'minimum_success_per_family': MIN_SUCCESS,
            'horizon_blocks': HORIZON, 'raw_actions_per_block': ACTION_BLOCK, 'max_replans': REPLANS,
            'candidate_count_per_replan': CANDIDATES, 'goal_radius_strict': GOAL_RADIUS,
            'task_seeds': [task.seed for task in fixed_tasks()], 'lifecycle_matched_pairs': 4,
            'two_door_forks': ('opposite_0_0 opposite_0_1 opposite_2_0 opposite_2_1').split(),
            'fork_horizon_blocks': HORIZON,
            'short_fork_reference_horizon_blocks': SHORT_FORK_HORIZON,
            'goal_cost': {'learned': 'level.criterion, final two steps against frozen RGB goal latent',
                          'physics': 'visible-map path length at exact physical candidate endpoint'},
            'geometry_diagnostics': {
                'maximum_flagged_fraction_each': MAX_GEOMETRY_RATE,
                'solid_wall_endpoint': 'Decoded endpoint inside x/y collision strip and outside open apertures',
                'false_cross': 'Decoded crossing from a geometrically valid prior decoded endpoint, where '
                               'replaying the exact five selected raw actions from that endpoint does not cross',
            }}
    if fixed_distractor:
        settings['fixed_distractor'] = {'xy': list(FIXED_DISTRACTOR_XY), 'policy': 'static', 'teleport_tau': None}
    return settings


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def qualification_suite(env, physics, native=None):
    rows = []
    for task in fixed_tasks():
        started = time.perf_counter()
        row = qualification_episode(task, env, physics, native)
        row['elapsed_seconds'] = time.perf_counter() - started
        rows.append(row)
        print(f'{task.name} success={row["success"]} replans={len(row["decisions"])} '
              f'seconds={row["elapsed_seconds"]:.2f}', flush=True)
    report = {'summary': phase_summary(rows), 'tasks': rows}
    if native:
        decisions = [decision for row in rows for decision in row['decisions']]
        geometry = {}
        for indicator in ('solid_wall_endpoint_axes', 'false_cross_axes', 'invalid_start_axes'):
            flags = np.asarray([decision['geometry'][indicator] for decision in decisions]).any(-1)
            count = int(flags.sum())
            geometry[indicator] = {'flagged': count, 'selected_forecast_blocks': flags.size,
                                   'fraction': count / flags.size}
            if indicator != 'invalid_start_axes':
                geometry[indicator]['passed'] = count <= MAX_GEOMETRY_RATE * flags.size
        actual = np.asarray([decision['decoded_actual_arrival_wall_axes'] for decision in decisions]).any(-1)
        geometry['decoded_actual_arrival_wall_axes'] = {'flagged': int(actual.sum()), 'actual_arrivals': len(actual),
                                                       'fraction': float(actual.mean())}
        report['geometry'] = geometry
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('physical', 'native'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fixed-distractor', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    settings = protocol(fixed_distractor=args.fixed_distractor)
    env = OfficialFourRoom(fixed_distractor=args.fixed_distractor)
    try:
        physics = OpenPhysics(env)
        physical_path = args.output / 'physical.json'
        if args.phase == 'physical':
            checks = physical_contract(env, physics)
            pairs, arrays = lifecycle_suite(env, physics)
            forks, fork_arrays, fork_histories = fork_suite(env, physics)
            if args.fixed_distractor:
                for trace in (arrays, fork_histories):
                    positions = trace['distractor_xy']
                    np.testing.assert_array_equal(positions, np.broadcast_to(FIXED_DISTRACTOR_XY, positions.shape))
            qualification = qualification_suite(env, physics)
            args.output.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output / 'lifecycle_rgb_controls.npz', **arrays)
            np.savez_compressed(args.output / 'fork_candidates.npz', **fork_arrays)
            np.savez_compressed(args.output / 'fork_lifecycle_rgb_controls.npz', **fork_histories)
            write_json(physical_path, {'protocol': settings, 'scalar_collision_checks': checks,
                                       'lifecycle': pairs, 'forks': forks, 'qualification': qualification})
        else:
            pinned = json.loads(physical_path.read_text(encoding='utf-8'))
            if (pinned['protocol'] != settings
                    or len(pinned['lifecycle']) != settings['lifecycle_matched_pairs']
                    or [fork['task'] for fork in pinned['forks']] != settings['two_door_forks']
                    or not all(fork['distinct_routes'] and fork['positive_uniform_regret']
                               and fork['complete_two_door_routes'] for fork in pinned['forks'])
                    or not all(entry['passed'] for entry in pinned['qualification']['summary'].values())):
                raise RuntimeError('Stage A physical prerequisites are not established')
            qualification = qualification_suite(env, physics, Native(torch.device(args.device)))
            write_json(args.output / 'native.json', {'protocol': settings, 'qualification': qualification})
        print(json.dumps(qualification['summary']), flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    main()
