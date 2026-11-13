"""Build action forks in the official PushT physics environment. See the frozen protocol."""
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/pusht_continuation_2026-10-31'
CONFIG = ROOT/'configs/pusht_continuation.json'


def environment():
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    from gym_pusht.envs.pusht import PushTEnv
    if version('gym-pusht') != '0.1.8':
        raise RuntimeError('This task fixes gym-pusht 0.1.8')
    return PushTEnv(obs_type='state', visualization_width=96, visualization_height=96)


def scenes(rng, n):
    reset = np.empty((n, 2, 2, 5))
    plans = np.zeros((n, 4, 15, 2))
    for idx in range(n):
        center = rng.uniform(205, 305, 2)
        psi = rng.uniform(-np.pi,np.pi)
        angles = psi + np.array([0, np.pi/2, np.pi/4])
        pushes = np.stack((np.cos(angles),np.sin(angles)),axis=1)
        plans[idx,:3] = .22*pushes[:,None]
        agents = center - rng.uniform(94,110,(2,1))*pushes[:2]
        theta = rng.uniform(-np.pi,np.pi)
        poses = theta + np.array([0,rng.uniform(.65,1.7)])
        for i in (0,1):
            for j in (0,1):
                reset[idx,i,j] = [*agents[j],*center,poses[i]]
    return reset, plans


def simulate(env, reset, plan):
    state, _ = env.reset(seed=0,options={'reset_to_state':reset})
    for steps, rel in enumerate(plan,1):
        state, _, success, truncated, info = env.step(state[:2]+100.*rel)
        if truncated:
            raise RuntimeError('Raw unwrapped environment unexpectedly truncated')
        if success:
            break
    return float(info['coverage']), steps


def main():
    config = json.loads(CONFIG.read_text())
    RUN.mkdir(parents=True, exist_ok=True)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    with (RUN/'PHYSICS_FREEZE.json').open('x') as f:
        json.dump({'protocol':config, 'config_sha256':sha(CONFIG),
                   'physics_code_sha256':sha(Path(__file__)),
                   'runtime':{n:version(n) for n in ('gym-pusht','gymnasium','pymunk','pygame','shapely')}},f,indent=2)
    env = environment()
    reset, plans = scenes(np.random.default_rng(config['scene_seed']),
                          config['train_scenes']+config['evaluation_scenes'])
    states = np.empty_like(reset)
    scores = np.empty((*reset.shape[:3],4))
    steps = np.zeros_like(scores, dtype=int)
    rgb = np.empty((*reset.shape[:3],96,96,3),dtype=np.uint8)
    try:
        env.reset(seed=0)
        env.block.angle = env.goal_pose[2]
        env.block.position = env.goal_pose[:2].tolist()
        env.agent.position = (256,400)
        # Native debug_draw uses cached shape transforms after direct body moves.
        env.space.reindex_shapes_for_body(env.block)
        env.space.reindex_shapes_for_body(env.agent)
        if not np.isclose(env._get_coverage(),1.):
            raise RuntimeError('Rendered goal pose does not coincide with the official target')
        goal = env.render()
        for k,i,j in np.ndindex(reset.shape[:3]):
            states[k,i,j], _ = env.reset(seed=0,options={'reset_to_state':reset[k,i,j]})
            rgb[k,i,j] = env.render()
            for a, plan in enumerate(plans[k]):
                scores[k,i,j,a],steps[k,i,j,a] = simulate(env,reset[k,i,j],plan)
    finally:
        env.close()
    with (RUN/'physics_panel.npz').open('xb') as f:
        np.savez_compressed(f,reset=reset,states=states,plans=plans,rgb=rgb,goal_rgb=goal,
                            coverage=scores,steps=steps,scene_ids=np.arange(len(reset)))
    print('physics panel',scores.shape,sha(RUN/'physics_panel.npz'),flush=True)


if __name__ == '__main__':
    main()
