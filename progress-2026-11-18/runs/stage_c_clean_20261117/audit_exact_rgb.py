"""Score exact clean FourRoom RGB in the official matched H12 candidate bank."""

import json
import runpy
from pathlib import Path

import numpy as np
import torch

from research.p1a.stage_a_fourroom import Native, OfficialFourRoom, fixed_tasks, latent_goal_cost, write_json


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    here = Path(__file__).resolve().parent
    audit = runpy.run_path(str(here.parent / 'stage_a_20261117/audit_goal_cost.py'))
    qualified = here.parent / 'stage_a_fourroom_clean_20261117'
    physical = json.loads((qualified / 'physical.json').read_text(encoding='utf-8'))
    tasks = {task.name: task for task in fixed_tasks()}
    native = Native(torch.device('cuda'))
    env = OfficialFourRoom(fixed_distractor=True)
    rows = []
    try:
        with np.load(qualified / 'fork_candidates.npz', allow_pickle=False) as archive:
            for i, fork in enumerate(physical['forks']):
                task = tasks[fork['task']]
                goal = archive['goal_rgb'][i]
                audit['reset'](env, task, None, goal)
                encoded_goal = native.encode(goal)
                positions = archive['raw_xy_by_mode'][i][..., [54, 59], :]
                terminal = audit['exact_latents'](native, env, positions)
                root = native.encode(env.render()).expand(*terminal.shape[:2], 1, -1)
                full = torch.cat((root, terminal), dim=-2)
                scores = latent_goal_cost(native.host, full, encoded_goal.expand(2, -1)).cpu().numpy()
                physical_cost = archive['candidate_cost_by_mode'][i]
                true_choices = physical_cost.argmin(-1)
                selected = scores.argmin(-1)
                for mode in range(2):
                    regret = audit['outcome'](physical_cost[mode], archive['raw_candidates'][i], selected[mode])
                    gap = scores[mode, true_choices[1-mode]] - scores[mode, true_choices[mode]]
                    rows.append(dict(task=fork['task'], physical_condition=mode, regret=regret,
                                     wrong_route_minus_right=float(gap)))
    finally:
        env.close()
    # Fixed background and identical roots give the same exact terminal cost at all three retentions.
    result = {'unique_root_condition_pairs': len(rows), 'retained_fork_queries': 3*len(rows),
              'mean_first_regret': float(np.mean([r['regret']['first_block_regret'] for r in rows])),
              'correct_oracle_route_directions': int(sum(r['wrong_route_minus_right'] > 0 for r in rows)),
              'rows': rows}
    write_json(here / 'exact_rgb_goal_audit.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
