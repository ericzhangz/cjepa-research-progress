"""Executed native TwoRoom rollout and record stacking, without archive/download helpers.

The optional PRBD_ENV_SOURCE directory may contain an installed author package.
No environment is imported until an actual collection is explicitly requested.
"""

import os
import sys

import numpy as np


def environment_class():
    source = os.environ.get("PRBD_ENV_SOURCE")
    if source and source != sys.path[0]:
        sys.path.insert(0, source)
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv
    return TwoRoomEnv


def rollout(seed, speed, commands):
    """Execute every stored transition through actual env.step, stopping at done."""
    blocks = len(commands)
    speeds = np.broadcast_to(np.asarray(speed, dtype=np.float32), (blocks,))
    if not np.isfinite(speeds).all() or np.any((speeds < 1.75) | (speeds > 10.5)):
        raise ValueError("Speed schedule must lie in the native public variation range")
    env = environment_class()(render_mode="rgb_array", render_target=False)
    obs, info = env.reset(seed=int(seed), options={"variation_values": {
        "agent.speed": np.array([speeds[0]], dtype=np.float32)}})
    if float(env.variation_space["agent"]["speed"].value.item()) != speeds[0]:
        raise ValueError("Native public speed variation was not applied")
    rgb = np.zeros((blocks+1, 224, 224, 3), dtype=np.uint8)
    rgb[0] = env.render()
    frame_valid = np.zeros(blocks+1, dtype=bool)
    frame_valid[0] = True
    block_valid = np.zeros(blocks, dtype=bool)
    primitive_valid = np.zeros((blocks, 5), dtype=bool)
    positions = np.full((blocks*5+1, 2), np.nan, dtype=np.float32)
    positions[0] = info["state"].copy()
    collision = np.zeros((blocks, 5), dtype=bool)
    terminated = np.zeros((blocks, 5), dtype=bool)
    truncated = np.zeros((blocks, 5), dtype=bool)
    goal = info["goal_state"].copy()
    count = 0
    done = False
    for block, actions in enumerate(commands):
        env.variation_space["agent"]["speed"].set_value(np.array([speeds[block]], dtype=np.float32))
        if float(env.variation_space["agent"]["speed"].value.item()) != speeds[block]:
            raise ValueError("Native public speed schedule was not applied")
        for primitive, action in enumerate(actions):
            before = positions[count].copy()
            obs, reward, term, trunc, info = env.step(action)
            count += 1
            actual = np.asarray(info["state"], dtype=np.float32).copy()
            positions[count] = actual
            expected = before + np.clip(action, -1, 1) * speeds[block]
            collision[block, primitive] = np.max(np.abs(actual - expected)) > 1e-5
            primitive_valid[block, primitive] = True
            terminated[block, primitive] = term
            truncated[block, primitive] = trunc
            if term or trunc:
                done = True
                break
        if primitive_valid[block].all():
            rgb[block+1] = env.render()
            frame_valid[block+1] = True
            block_valid[block] = True
        if done:
            break
    final_rgb = env.render()
    env.close()
    return {"rgb": rgb, "frame_valid": frame_valid, "block_valid": block_valid,
            "primitive_valid": primitive_valid, "actions": commands,
            "final_rgb": final_rgb}, {
        "positions": positions, "goal": goal, "collision": collision,
        "terminated": terminated, "truncated": truncated,
        "primitive_count": np.array(count, dtype=np.int64),
        "seed": np.array(seed, dtype=np.int64), "speed": np.array(speed, dtype=np.float32)}


def stack_records(records):
    return {name: np.stack([record[name] for record in records]) for name in records[0]}
