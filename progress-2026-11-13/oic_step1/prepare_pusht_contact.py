"""Execute a shared PushT action, then score all remaining candidates from that history."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from prepare_pusht_continuation import ROOT, environment, scenes


CONFIG = ROOT/'configs/pusht_contact.json'
RUN = ROOT/'runs/pusht_contact_2026-11-01'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sensor_position(env):
    return np.array(env.block.local_to_world((0., 120.)))


def execute(env, reset, target, suffix, prefix_steps):
    """Replay exactly the same prefix for each branch, retaining the live world for its suffix."""
    state, _ = env.reset(seed=0, options={'reset_to_state':reset})
    initial = state.copy()
    initial_tip = sensor_position(env)
    contacts = np.zeros(prefix_steps, dtype=np.int32)
    step_index = 0
    def count_contact(arbiter, space, data):
        a, b = (shape.body for shape in arbiter.shapes)
        if {a, b} == {env.agent, env.block}:
            contacts[step_index] += 1
        env._handle_collision(arbiter, space, data)
    env.collision_handeler.post_solve = count_contact
    for step_index in range(prefix_steps):
        state, _, done, truncated, info = env.step(target)
        if truncated:
            raise RuntimeError('Unexpected truncated raw environment')
        if done:
            break
    actual_prefix_steps = step_index + 1
    after = state.copy()
    tip = sensor_position(env)
    frame = env._render(visualize=False)
    env.collision_handeler.post_solve = env._handle_collision
    suffix_steps = 0
    if not done:
        for suffix_steps, rel in enumerate(suffix, 1):
            state, _, done, truncated, info = env.step(state[:2] + 100.*rel)
            if truncated:
                raise RuntimeError('Unexpected truncated raw environment')
            if done:
                break
    return ((initial, after, initial_tip, tip, contacts, actual_prefix_steps, frame),
            info['coverage'], suffix_steps)


def make_panel(env, reset, plans, first_scene_id, prefix_steps):
    n = len(reset)
    targets = reset[:,0,0,2:4] + [0., 45.]
    np.testing.assert_array_equal(reset[...,2:4], np.broadcast_to(reset[:,:1,:1,2:4],reset[...,2:4].shape))
    records = {
        'reset': reset, 'plans': plans, 'prefix_target': targets, 'scene_ids': np.arange(first_scene_id,first_scene_id+n),
        'initial_state': np.empty_like(reset), 'post_state': np.empty_like(reset),
        'initial_tip': np.empty((n,2,2,2)), 'post_tip': np.empty((n,2,2,2)),
        'contacts': np.empty((n,2,2,prefix_steps),dtype=np.int32), 'prefix_steps': np.empty((n,2,2),dtype=np.int32),
        'post_rgb': np.empty((n,2,2,96,96,3),dtype=np.uint8),
        'coverage': np.empty((n,2,2,4)), 'suffix_steps': np.empty((n,2,2,4),dtype=np.int32)}
    prefix_keys = ['initial_state','post_state','initial_tip','post_tip','contacts','prefix_steps','post_rgb']
    for s,i,j in np.ndindex((n,2,2)):
        for a, suffix in enumerate(plans[s]):
            prefix, score, steps = execute(env, reset[s,i,j], targets[s], suffix, prefix_steps)
            for key,value in zip(prefix_keys,prefix):
                if a == 0:
                    records[key][s,i,j] = value
                elif not np.array_equal(records[key][s,i,j], value):
                    raise RuntimeError(f'Candidate fork changed the physical prefix: {s,i,j,a,key}')
            records['coverage'][s,i,j,a] = score
            records['suffix_steps'][s,i,j,a] = steps
    records['sensor'] = (records['post_tip'][...,0] > targets[:,None,None,0]).astype(np.int8)
    return records


def boundary_counts(panel):
    target_x = panel['prefix_target'][:,None,None,0]
    displacement = np.linalg.norm(panel['post_state'][...,2:4]-panel['initial_state'][...,2:4],axis=-1)
    clean = panel['sensor']
    return {'scenes':len(clean), 'worlds':clean.size, 'worlds_with_pusher_T_contact':int((panel['contacts'].sum(-1)>0).sum()),
            'worlds_absorbed_during_prefix':int((panel['suffix_steps']==0).all(-1).sum()),
            'worlds_with_changed_sensor_bit_vs_reset':int(np.count_nonzero(clean != (panel['initial_tip'][...,0]>target_x))),
            'scenes_with_both_sensor_bits':int((clean.min((1,2)) != clean.max((1,2))).sum()),
            'median_block_displacement':float(np.median(displacement)), 'max_block_displacement':float(displacement.max())}


def physics_lock():
    return {'config_sha256':digest(CONFIG),'generator_sha256':digest(Path(__file__)),
            'base_physics_sha256':digest(ROOT/'prepare_pusht_continuation.py'),
            'protocol':json.loads(CONFIG.read_text()),'runtime':{n:version(n) for n in ('gym-pusht','gymnasium','pymunk','pygame','shapely')}}


def prepare(which):
    RUN.mkdir(parents=True,exist_ok=True)
    lock_path = RUN/'PHYSICS_FREEZE.json'
    lock = physics_lock()
    if which == 'freeze':
        with lock_path.open('x') as f:
            json.dump(lock, f, indent=2)
        return
    if json.loads(lock_path.read_text()) != lock:
        raise RuntimeError('Physics protocol or environment changed after freeze')
    if which == 'confirmation':
        rec = RUN/'recurrent'
        proof = json.loads((rec/'PRECONFIRMATION_LOCK.json').read_text())
        expected = {f'fits/gru{s}.pt' for s in (11,22,33)}
        hashes = proof['checkpoint_sha256']
        if (set(hashes) != expected or any(digest(rec/k) != v for k,v in hashes.items())
                or proof['freeze_sha256'] != digest(rec/'FREEZE.json')):
            raise RuntimeError('Three final models and their protocol must be locked before generating confirmation')
    config = lock['protocol']
    spec = config[which]
    dest = RUN/f'{which}.npz'
    if dest.exists():
        raise FileExistsError(dest)
    reset, all_plans = scenes(np.random.default_rng(spec['seed']),spec['count'])
    env = environment()
    try:
        panel = make_panel(env, reset, all_plans[:,:,:config['suffix_steps']],spec['first_scene_id'],config['common_prefix_steps'])
    finally:
        env.close()
    with dest.open('xb') as f:
        np.savez_compressed(f,**panel)
    counts = boundary_counts(panel)
    with (RUN/f'{which}_physics.json').open('x') as f:
        json.dump({'panel_sha256':digest(dest),'physics_freeze_sha256':digest(lock_path),'counts':counts},f,indent=2)
    print(which,counts,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['freeze','train','confirmation'])
    prepare(parser.parse_args().phase)
