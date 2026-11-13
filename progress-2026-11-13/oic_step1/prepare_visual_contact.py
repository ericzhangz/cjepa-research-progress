"""A fixed local RGB stream with real PushT input controls and exact pixel-history classes."""
import argparse
import json
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from prepare_pusht_contact import digest
from prepare_pusht_continuation import ROOT, environment


CONFIG = ROOT/'configs/visual_contact.json'
RUN = ROOT/'runs/visual_contact_2026-11-01'


def scene(psi):
    v = np.array([np.cos(psi), np.sin(psi)])
    n = np.array([-v[1], v[0]])
    # Native T has local CoG (0,45). The goal origin and angle are public.
    g = np.array([256,256]) + 45/np.sqrt(2)*np.array([-1,1])
    center = g - 82*v + 10*n
    reset = []
    for i, j in np.ndindex(2,2):
        agent = center - 85*v + (2*j-1)*24*n
        reset.append([*agent, *(center-[0,45]), np.pi/4+(2*i-1)*.85])
    prefix = np.repeat(np.stack([np.zeros(2), .40*v])[:,None], 5, axis=1)
    velocities = .22*np.stack([v, .5*v+np.sqrt(.75)*n, .5*v-np.sqrt(.75)*n, np.zeros(2)])
    suffix = np.broadcast_to(velocities[:,None,None], (4,2,5,2)).copy()
    return np.array(reset), prefix, suffix, np.stack([center+32*v, center+8*v, center-8*v+24*n])


def local_rgb(frame, xy, width):
    x,y = np.rint(xy*96/512).astype(int)
    k = width//2
    crop = frame[y-k:y+k+1,x-k:x+k+1]
    if crop.shape != (width,width,3):
        raise ValueError('Public camera lies outside the arena')
    return cv2.resize(crop, (96,96), interpolation=cv2.INTER_NEAREST)


def branch(env, reset, early_actions, suffix, cameras):
    state, info = env.reset(seed=0, options={'reset_to_state':reset})
    image = env._render(visualize=False)
    observed = [local_rgb(image,cameras[0],19)]
    future = []
    done = False
    contacts, steps = [], []
    for t, actions in enumerate([*early_actions,*suffix]):
        had_contact = False
        count = 0
        for rel in actions:
            if done:
                break
            state, _, done, truncated, info = env.step(state[:2]+100.*rel)
            if truncated:
                raise RuntimeError('Unexpected PushT truncation')
            had_contact |= info['n_contacts'] > 0
            count += 1
        image = env._render(visualize=False)
        if t < 2:
            observed.append(local_rgb(image,cameras[t+1],(15,3)[t]))
        else:
            future.append(image)
        contacts.append(had_contact)
        steps.append(count)
    return np.stack(observed), np.stack(future), info['coverage'], contacts, steps


def pixel_groups(frames):
    labels = []
    known = {}
    for row in frames:
        key = row.tobytes()
        labels.append(known.setdefault(key,len(known)))
    return np.array(labels,dtype=np.uint8)


def conditional(coverage, groups):
    return np.stack([coverage[groups==g].mean(axis=0) for g in groups])


def make_panel(env, angles, ids):
    n = len(angles)
    rgb = (96,96,3)
    p = dict(scene_ids=ids, psi=angles, observed_actions=np.empty((n,2,5,2)), actions=np.empty((n,4,2,5,2)),
             obs_rgb=np.empty((n,4,3,*rgb),np.uint8), future_rgb=np.empty((n,4,4,2,*rgb),np.uint8),
             coverage=np.empty((n,4,4)), steps=np.empty((n,4,4,4),np.int8), contacts=np.zeros((n,4,4,4),bool))
    for s,angle in enumerate(angles):
        reset, prefix, actions, cameras = scene(angle)
        p['observed_actions'][s],p['actions'][s] = prefix,actions
        for w,a in np.ndindex(4,4):
            obs,fut,score,hit,steps = branch(env, reset[w], prefix, actions[a], cameras)
            if a==0:
                p['obs_rgb'][s,w] = obs
            elif not np.array_equal(obs,p['obs_rgb'][s,w]):
                raise RuntimeError('Candidate altered an earlier observed frame')
            p['future_rgb'][s,w,a],p['coverage'][s,w,a] = fut,score
            p['steps'][s,w,a],p['contacts'][s,w,a] = steps,hit
    if np.unique(p['observed_actions'][:,1,0],axis=0).shape[0]!=n:
        raise ValueError('Duplicate public scene controls')
    for tag,subset in [('pre',slice(0,2)),('post',slice(0,3)),('latest',slice(2,3))]:
        groups = np.stack([pixel_groups(x[:,subset]) for x in p['obs_rgb']])
        p[tag+'_group'] = groups
        p['values_'+tag] = np.stack([conditional(c,g) for c,g in zip(p['coverage'],groups)])
    opt = [y>=y.max(-1,keepdims=True)-1e-12 for y in (p['values_pre'],p['values_post'])]
    p['changed'] = ~(opt[0] & opt[1]).any(-1)
    return p


def identify(p, gates):
    pre, post, latest = [p['values_'+s] for s in ('pre','post','latest')]
    def regret(stale):
        return float((post.max(-1)-np.take_along_axis(post,stale.argmax(-1)[...,None],-1)[...,0]).mean())
    measures = dict(changed_mass=float(p['changed'].mean()),
                    earlier_visual_history_oracle_regret=regret(latest),
                    retain_early_oracle_action_regret=regret(pre),
                    terminal_range_mean=float(np.ptp(p['coverage'],axis=-1).mean()))
    checks = {k:measures[k[:-4]]>=v for k,v in gates.items()}
    return dict(measures=measures, gates=checks, all_pass=all(checks.values()), scenes=len(post),
                changed_worlds=int(p['changed'].sum()), worlds=post.size//4,
                scenes_with_changed=int(p['changed'].any(-1).sum()),
                contacts_on_common_action=int(p['contacts'][:,:,0,1].sum()),
                early_absorption=int((p['steps'][:,:,0,:2].sum(-1)<10).sum()),
                full_suffix_world_actions=int((p['steps'][...,2:].sum(-1)==10).sum()),
                maximum_world_actions=int(np.prod(p['coverage'].shape)),
                modal_post_action_frequencies=np.bincount(post.argmax(-1).ravel(),minlength=4).__truediv__(post.shape[0]*4).tolist(),
                worlds_with_latest_only_ambiguity=int(sum(sum(np.sum(g==x)>1 for x in g) for g in p['latest_group'])))


def prepare(phase):
    RUN.mkdir(exist_ok=True,parents=True)
    cfg = json.loads(CONFIG.read_text())
    freeze = RUN/'PHYSICS_FREEZE.json'
    record = {'sources':{str(f.relative_to(ROOT)):digest(f) for f in (CONFIG,Path(__file__),ROOT/'prepare_pusht_contact.py',ROOT/'prepare_pusht_continuation.py')},
              'runtime':{n:version(n) for n in ('gym-pusht','pymunk','pygame','shapely')},'config':cfg}
    if phase=='freeze':
        with freeze.open('x') as f: json.dump(record,f,indent=2)
        return
    if record != json.loads(freeze.read_text()):
        raise RuntimeError('Frozen physics inputs have changed')
    if phase=='confirmation':
        hashes=json.loads((RUN/'OPEN_CONFIRMATION.json').read_text())['sha256']
        required={'model/PRECONFIRMATION_LOCK.json','score/SCORER_LOCK.json','PHYSICS_FREEZE.json'}
        if not required<=hashes.keys() or any(digest(RUN/key)!=value for key,value in hashes.items()):
            raise RuntimeError('Learner, scoring and physics locks must match before confirmation')
    spec=cfg[phase]
    out=RUN/f'{phase}.npz'
    if out.exists(): raise FileExistsError(out)
    angles=np.random.default_rng(spec['seed']).uniform(-np.pi,np.pi,spec['count'])
    env=environment()
    try: p=make_panel(env,angles,np.arange(spec['first_scene_id'],spec['first_scene_id']+len(angles)))
    finally: env.close()
    with out.open('xb') as f: np.savez_compressed(f,**p)
    result=identify(p,cfg['train_identifiability_gates'])
    result.update(panel_sha256=digest(out),freeze_sha256=digest(freeze))
    with (RUN/f'{phase}_identifiability.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['freeze','train','confirmation'])
    prepare(parser.parse_args().phase)
