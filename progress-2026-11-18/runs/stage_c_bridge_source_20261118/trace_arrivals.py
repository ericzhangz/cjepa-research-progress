"""Record every factual arrival in the fixed ABA/BAB replay, without planning."""

import json
from pathlib import Path

import numpy as np
import torch

from research.p1a.stage_a_fourroom import ROOT, Native, OfficialFourRoom, OpenPhysics, write_json
from research.p1a.stage_a_replay import _arrival_truth, _episodes, _labels, _model_inputs
from research.p1a.stage_c_paired import two_offsets
from research.p1a.stage_c_paired_replay import sequential_arrivals


@torch.inference_mode()
def main():
    run = Path(__file__).resolve().parent
    archive = ROOT / 'runs/stage_a_fourroom_clean_20261117'
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    native = Native(device)
    state = torch.load(run / 'paired_unlabeled.pt', map_location=device, weights_only=True)
    native.host.predictor.transformer.layers[-2:].load_state_dict(state['final_two_blocks'], strict=True)
    scale = json.loads((run / 'paired_unlabeled_fit.json').read_text())['s2_obs']
    mapping = json.loads((run / 'fit_alignment.json').read_text())['physical_to_explanation']
    physical = json.loads((archive / 'physical.json').read_text())
    offsets = two_offsets(state['b'], state['v'])
    rows = []
    env = OfficialFourRoom(fixed_distractor=True)
    try:
        physics = OpenPhysics(env)
        with np.load(archive / 'fork_lifecycle_rgb_controls.npz', allow_pickle=False) as factual:
            for _, fork, order, meta, arrays in _episodes(physical, dict(factual)):
                z, actions, _ = _model_inputs(native, arrays)
                p, e, issued, _, _ = sequential_arrivals(device, native.host, z, actions, offsets, scale)
                labels, modes = _labels(meta)
                truth = _arrival_truth(physics, meta, arrays['raw_xy'], arrays['raw_actions'])
                for t in range(3, len(p)):
                    # A switch declared at t affects the next issued block.
                    mode = int(modes[t-1])
                    k = mapping[mode]
                    rows.append({'episode': f"{fork['task']}_{order}", 'arrival': t,
                                 'action_event': meta['blocks'][t-1]['event'], 'phase_labels': labels[t],
                                 'physical_mode_eval_only': mode, **truth[t-1],
                                 'oriented_evidence': float((2*k-1) * e[t-3]),
                                 'support_before': float(p[t-1, k]), 'support_after': float(p[t, k]),
                                 'issued_difference_l2': float((issued[t-3, 1] - issued[t-3, 0]).norm())})
    finally:
        env.close()
    write_json(run / 'arrival_trace.json', {'rows': rows})


if __name__ == '__main__':
    main()
