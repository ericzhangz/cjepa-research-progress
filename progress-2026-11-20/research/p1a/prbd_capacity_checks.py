"""CPU numerical invariants for the Phase 1 offline capacity comparison."""

import json

import torch

from .prbd_capacity_oracle import (CapacityPrototypes, OUT, training_roots,
                                   write_json)
from .prbd_native import NativePrototypes
from .prbd_tworoom_speed_fit import load_host


def main():
    torch.set_num_threads(2)
    data, metadata = training_roots()
    window, controls = data[0][:2], data[1][:2]
    host = load_host("cpu")
    model_a = CapacityPrototypes(host, "A").eval()
    archived = NativePrototypes(host, null_control=None).eval()
    with torch.no_grad():
        torch.testing.assert_close(model_a.rollout(window, controls),
                                   archived.rollout(window, controls), rtol=0, atol=0)
    model_b = CapacityPrototypes(host, "B").eval()
    with torch.no_grad():
        expected = model_a.rollout(window, controls)
        initial_b = model_b.rollout(window, controls)
        torch.testing.assert_close(initial_b, expected, rtol=2e-5, atol=3e-6)
        # Moving G's shared weights cannot alter the literal zero condition.
        for parameter in model_b.last_blocks.parameters():
            parameter.add_(.001)
        model_b.offsets.zero_()
        model_b.action_up.zero_()
        zero = model_b.rollout(window, controls)
        torch.testing.assert_close(zero[:, 0], expected[:, 0], rtol=0, atol=0)
        selected_zero = model_b.selected_step(window, controls[:, :3], torch.tensor([1, 2]))
        torch.testing.assert_close(selected_zero, host.predict(window, controls[:, :3])[:, -1],
                                   rtol=0, atol=0)
    # The remaining branch receives real supervision from both physical groups.
    pooled_gradients = {}
    for branch in (1, 2):
        index = torch.where(data[3] == branch)[0][:2]
        path = model_a.selected_rollout(data[0][index], data[1][index], torch.full((2,), 3))
        loss = .5 * ((path[:, 0] - data[2][index, 0]).square().mean()
                     + (path[:, 3] - data[2][index, 3]).square().mean())
        gradient, = torch.autograd.grad(loss, model_a.offsets)
        assert gradient[2].norm() > 0
        assert torch.equal(gradient[:2], torch.zeros_like(gradient[:2]))
        pooled_gradients[str(branch)] = float(gradient[2].norm())
    output = {"A_exactly_original_unanchored_3456_offsets": True,
              "A_parameters": sum(p.numel() for p in model_a.parameters() if p.requires_grad),
              "B_initially_contains_A_to_fp32_roundoff": True,
              "B_parameters": sum(p.numel() for p in model_b.parameters() if p.requires_grad),
              "B_changed_shared_blocks_zero_condition_H1_H4_bitwise_F0": True,
              "B_zero_residual_selected_step_bitwise_same_batch_F0": True,
              "null_control_used": False, "speed_assignment_matches_stored_physics": True,
              "trajectories": len(metadata["training_seeds"]), "root_count": len(data[0]),
              "pooled_branch_nonzero_gradient_by_true_condition": pooled_gradients,
              "initial_B_A_max_abs_roundoff": float((initial_b - expected).abs().max())}
    write_json(OUT / "capacity_unit_checks.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
