"""Frozen author LeWM action rankings on its original archive and the PushT continuation panel."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/pusht_continuation_2026-10-31'
ASSETS = ROOT.parent/'research_sources_2026-10-22/lewm_assets'
sys.path.insert(0,str(ASSETS))
from prepare_assets import load_model, SOURCE, SOURCE_REV, MODEL_REV
sys.path.insert(0, str(SOURCE))
from module import validate_first_block_memory


def pixels(image, device):
    x = torch.from_numpy(image).to(device).permute(0,3,1,2).float()/255
    mean = x.new_tensor([.485,.456,.406])[None,:,None,None]
    std = x.new_tensor([.229,.224,.225])[None,:,None,None]
    return resize((x-mean)/std,[224,224],interpolation=InterpolationMode.BILINEAR,antialias=True)


def encode(model, frames):
    shape = frames.shape
    imgs = frames.reshape(-1,*shape[-3:])
    chunks = []
    for start in range(0,len(imgs),32):
        x = pixels(imgs[start:start+32], next(model.parameters()).device)
        chunks.append(model.encode({'pixels': x[:,None]})['emb'][:,0])
    return torch.cat(chunks).reshape(*shape[:-3],192)


def rollout(model, history, actions, mean, std, first_block_memory=None, memory_site=None):
    """history[B,H,D], relative commands[B,C,H−1+K,5,2], return predicted endpoint per plan."""
    b,c,t = actions.shape[:3]
    h = history.shape[1]
    if actions.shape[3:]!=(5,2) or not 1<=h<=3 or t<h or history.shape!=(b,h,192):
        raise ValueError('Incompatible history or five-step action blocks')
    validate_first_block_memory(first_block_memory, memory_site, history)
    memory_kwargs = {}
    if first_block_memory is not None:
        if c < 1:
            raise ValueError('Native memory rollout requires at least one candidate')
        memory_kwargs = {
            'first_block_memory': first_block_memory[:, None].expand(-1, c, -1).reshape(b*c, 192),
            'memory_site': memory_site,
        }
    z = history[:,None].expand(-1,c,-1,-1).reshape(b*c,h,192)
    command = ((actions-mean)/std).reshape(b*c,t,10)
    embedding = model.action_encoder(command)
    for i in range(h-1,t):
        start = max(0,i-2)
        pred = model.predict(z,embedding[:,start:i+1], **memory_kwargs)[:,-1:]
        z = torch.cat((z,pred),1)[:,-3:]
    return pred.reshape(b,c,192)


def observed_archives(model, panel, mean, std):
    z = encode(model,panel['rgb'])
    actions = torch.as_tensor(panel['action'],device=z.device).reshape(-1,3,5,2)
    pair = torch.from_numpy(np.array([0,1])).to(z.device)
    # Check full author pixel->rollout->goal cost procedure on a 1- and a 3-frame history.
    diffs = {}
    for h in (1,3):
        raw = actions[pair][None].clone()
        raw[:,:,:h-1] = actions[0,:h-1]
        candidate = ((raw-mean)/std).flatten(-2)
        obs = pixels(panel['rgb'][0,:h],z.device)[None,None].expand(1,2,-1,-1,-1,-1)
        goal = pixels(panel['rgb'][0,3:4],z.device)[None,None].expand(1,2,-1,-1,-1,-1)
        author = model.get_cost({'pixels':obs,'goal':goal,'action':candidate[:,:,:h]},candidate)
        endpoint = rollout(model,z[0:1,:h],raw,mean,std)
        cached = (endpoint-z[0,3]).square().sum(-1)
        torch.testing.assert_close(cached,author,rtol=2e-5,atol=1e-3)
        diffs[str(h)] = (cached-author).abs().max().item()
    true = z[:,3]
    groups = {}
    for i,(frame,state) in enumerate(zip(panel['rgb'][:,0],panel['state'][:,0])):
        groups.setdefault((frame.tobytes(),state.tobytes()),[]).append(i)
    rankings=[]
    for ids in groups.values():
        if len(ids)<2:
            continue
        pred = rollout(model,z[ids[:1],:1],actions[ids][None],mean,std)[0]
        # Each recorded endpoint is a declared fixed goal; every other suffix has
        # an observed counterfactual outcome from the exact same start and visual context.
        estimated = (pred[None]-true[ids,None]).square().mean(-1)
        actual = (true[ids][None]-true[ids,None]).square().mean(-1)
        rankings.append({'ids':ids,'predicted_cost':estimated.tolist(),'actual_cost':actual.tolist(),
                         'top1_hit':(estimated.argmin(-1)==actual.argmin(-1)).tolist(),
                         'uniform_chance':1/len(ids)})
    factual_pred = rollout(model,z[:,:3],actions[:,None],mean,std)[:,0]
    return {'native_full_cost_max_abs_check':diffs, 'recorded_fork_rankings':rankings,
            'eligible_goals':sum(len(x['ids']) for x in rankings),
            'recovered_goal_plan':sum(sum(x['top1_hit']) for x in rankings),
            'one_block_factual_mse':(factual_pred-true).square().mean().item(),
            'one_block_copy_mse':(z[:,2]-true).square().mean().item(),
            'archive_scope':'first 16 author replay trajectories, one initial physical configuration, H=3 for factual forecasting, H=1 for truly shared starts'}


def score_posterior(cost,coverage):
    result = {}
    for stage in ('early','revision'):
        regret = np.zeros(len(cost)); match = np.zeros(len(cost)); uniform = np.zeros(len(cost))
        for b,c in np.ndindex(2,2):
            for r in (range(2) if stage=='revision' else (None,)):
                agent = np.where(np.arange(2)==b,.85,.15)
                block = np.where(np.arange(2)==c,.65,.35)
                if r is not None:
                    block *= np.where(np.arange(2)==r,.95,.05)
                joint = block[:,None]*agent[None,:]/4
                chance = joint.sum()
                post = joint/chance
                score = np.einsum('ij,sija->sa',post,coverage)
                chosen = np.einsum('ij,sija->sa',post,cost).argmin(1)
                best = score.max(1)
                selected = score[np.arange(len(score)),chosen]
                regret += chance*(best-selected)
                uniform += chance*(best-score.mean(1))
                match += chance*np.isclose(best,selected,rtol=0,atol=1e-12)
        result[stage] = {'scene_mean_regret':regret.mean(),'scene_mean_best_action_rate':match.mean(),
                         'uniform_mean_regret':uniform.mean(), 'scene_regret':regret.tolist()}
    return result


@torch.inference_mode()
def main():
    dest = RUN/'NATIVE_HOST.json'
    if dest.exists():
        raise FileExistsError(dest)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model = load_model('cuda').eval().requires_grad_(False)
    archive = RUN/'native_archive_panel.npz'
    norm = json.loads((RUN/'action_normalization.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=norm['panel_sha256']:
        raise ValueError('Author panel hash mismatch')
    with np.load(archive,allow_pickle=False) as panel:
        mean,std=[torch.tensor(panel[k],dtype=torch.float32,device='cuda') for k in ('mean','std')]
        direct=observed_archives(model,panel,mean,std)
    physics = RUN/'physics_panel.npz'
    with np.load(physics,allow_pickle=False) as p:
        ids = np.arange(96,144)
        np.testing.assert_array_equal(p['scene_ids'][ids],ids)
        z = encode(model,p['rgb'][ids])
        goal = encode(model,p['goal_rgb'][None])[0]
        plans=torch.as_tensor(p['plans'][ids],dtype=torch.float32,device=z.device).reshape(len(ids),4,3,5,2)
        costs = []
        for idx in range(len(ids)):
            predicted = rollout(model,z[idx].reshape(4,1,192),plans[idx:idx+1].expand(4,-1,-1,-1,-1),mean,std)
            costs.append((predicted-goal).square().sum(-1).reshape(2,2,4).cpu().numpy())
        costs=np.stack(costs)
        # Physical outcomes enter only after all predictions have been committed.
        coverage = p['coverage'][ids]
        np.savez_compressed(RUN/'native_costs.npz',scene_ids=ids,cost=costs)
        full_choice = costs.argmin(-1)
        observed_coverage = np.take_along_axis(coverage, full_choice[...,None],axis=-1)[...,0]
        regret=coverage.max(-1)-observed_coverage
        result={'source_revision':SOURCE_REV,'checkpoint_revision':MODEL_REV,'archive':direct,
                'physics':score_posterior(costs,coverage),
                'fully_observed_scene_mean_regret':float(regret.mean()),
                'worlds_per_scene':4,'actions_per_world':4,'blocks_per_candidate':3,
                'scope':'native unconditional world forecasts from each fully specified hypothetical reset image, plus exact external posterior averaging. Goal is the fixed RGB goal, success scored with physical terminal T coverage.'}
    dest.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print('archive',direct['eligible_goals'],direct['recovered_goal_plan'],
          direct['one_block_factual_mse'],direct['one_block_copy_mse'],flush=True)
    print('sim', {k:v['scene_mean_regret'] for k,v in result['physics'].items()}, flush=True)


if __name__=='__main__':
    main()
