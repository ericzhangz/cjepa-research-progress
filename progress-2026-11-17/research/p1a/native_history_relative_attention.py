"""Run the first frozen test of reference-relative history attention."""

import json

import torch

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import conditioned_rollout
from .native_full_depth_first_effect import branch_offsets
from .native_full_depth_reversible import SWITCH
from .native_full_depth_task_geometry import chart_at_latent, chart_pose
from .native_history_matched_fit import hold_and_relative, load_panel, pairing, trace
from .native_history_partial_arrivals import FITTED, HOLD, HORIZONS, OUTPUT as SCALAR, query_bank


OUTPUT = SCALAR.parent / "native_history_relative_attention_20261115"


def rms(x):
    return x.square().mean().sqrt().item()


@torch.no_grad()
def run(device):
    model, adapter, gate, hold, *_ = substep.load_core(device)
    fitted = torch.load(FITTED / "forecast_only.pt", map_location=device, weights_only=True)
    adapter.load_state_dict(fitted["adapter"])
    weight, bias = (fitted["chart"][key] for key in ("weight", "bias"))
    calibration = json.loads((SCALAR / "stage.json").read_text(encoding="utf-8"))
    transition = hold.new_tensor([[1 - SWITCH, SWITCH], [SWITCH, 1 - SWITCH]])
    config = (transition, weight, bias, hold.new_tensor(calibration["sensor_drift"]),
              hold.new_tensor(calibration["sensor_std"]))
    panel = load_panel(model, hold)
    batch = len(panel["ids"]) * 4
    plans = panel["plans"][None].expand(batch, -1, -1, -1)
    past = panel["past"][-2:][None].expand(batch, -1, -1)

    def call(windows, past, actions, ancestry=None, *, relative=False, info=None):
        return conditioned_rollout(
            model, windows, past, actions, branch_offsets(adapter, len(windows)),
            first_effect=gate, ancestry=ancestry, trace=info,
            reference_action=hold if relative else None,
        )

    def relative(windows, past, actions, ancestry=None):
        return call(windows, past, actions, ancestry, relative=True)

    _, _, (source, windows, prior, post, known, _) = trace(call, panel, slice(None), config)
    _, _, (relative_source, relative_windows, _, relative_post, relative_known, _) = trace(
        relative, panel, slice(None), config,
    )
    bank, evidence = query_bank(windows, known, transition)
    omit = known.logsumexp(1, keepdim=True) + prior.log_softmax(1)
    _, no_last_evidence = query_bank(windows, omit, transition)
    old = call(bank, past, plans, evidence)
    info = {}
    new = call(bank, past, plans, evidence, relative=True, info=info)
    old_omit = call(bank, past, plans, no_last_evidence)
    new_omit = relative(bank, past, plans, no_last_evidence)
    bank_after, evidence_after = query_bank(relative_windows, relative_known, transition)
    entire = relative(bank_after, past, plans, evidence_after)

    fixed_hold = plans[:, HOLD:HOLD + 1]
    single = torch.zeros_like(evidence)
    single[..., -1] = 1
    equal = bank[:, -1:].expand_as(bank)
    match = {
        "same_hold_embedding": (plans[:, HOLD] - hold).abs().max().item(),
        "hold_latest_max_abs": (old[:, HOLD, :, 0] - new[:, HOLD, :, 0]).abs().max().item(),
        "hold_rollout_max_abs": (old[:, HOLD] - new[:, HOLD]).abs().max().item(),
        "one_parent_max_abs": (call(bank, past, plans, single) -
                               relative(bank, past, plans, single)).abs().max().item(),
        "equal_parents_max_abs": (call(equal, past, plans, evidence) -
                                  relative(equal, past, plans, evidence)).abs().max().item(),
    }
    for name, error in match.items():
        if error > 4e-5:
            raise AssertionError(f"{name}: {error}")
    hold_info = {}
    call(bank, past, fixed_hold, evidence, relative=True, info=hold_info)
    match["hold_log_ratio_max_abs"] = hold_info["reference_log_mass_ratio"].abs().max().item()
    masses = info["parent_attention_mass"].unflatten(0, (batch, plans.size(1)))
    default = evidence[:, None, :, :, None].expand_as(masses)
    match["hold_mass_max_abs"] = (masses[:, HOLD] - default[:, HOLD]).abs().max().item()
    if max(match["hold_log_ratio_max_abs"], match["hold_mass_max_abs"]) > 4e-5:
        raise AssertionError(match)

    row = torch.arange(batch, device=device)
    mode = row % 2
    truth = hold_and_relative(panel["xy"])

    def to_xy(prediction):
        selected = prediction[row, :, mode][:, :, HORIZONS]
        return chart_pose(chart_at_latent(selected, weight, bias))[..., :2].unflatten(0, (-1, 2, 2))

    def fields(values):
        return dict(zip(("hold", "relative"), (
            pairing(prediction, target) for prediction, target in zip(hold_and_relative(values), truth)
        )))

    old_xy, new_xy, old_omit_xy, new_omit_xy, entire_xy = map(to_xy, (old, new, old_omit, new_omit, entire))
    # Difference-of-differences isolates the changed role of the last real ancestor write.
    changed_write = (new_xy - new_omit_xy) - (old_xy - old_omit_xy)
    write = {}
    for name, written_xy, omitted_xy in (("fixed", old_xy, old_omit_xy), ("relative", new_xy, new_omit_xy)):
        written = hold_and_relative(written_xy)[1]
        omitted = hold_and_relative(omitted_xy)[1]
        improvement = (omitted - truth[1]).square().sum(-1) - (written - truth[1]).square().sum(-1)
        write[name] = {
            "scene_action_mse_reduction": improvement.tolist(),
            "mean_mse_reduction": improvement.mean(dim=(0, 1, 3)).tolist(),
            "positive_scene_fraction": (improvement.mean(dim=(1, 3)) > 0).float().mean(0).tolist(),
        }
    return {
        "scenario_ids": panel["ids"], "invariants": match,
        "fixed_root_attention_mass_rms": rms(masses - default),
        "fixed_root_physical_delta_rms": (new_xy - old_xy).square().mean(dim=(0, 1, 2, 3, 5)).sqrt().tolist(),
        "fixed_root_write_interaction_rms": changed_write.square().mean(dim=(0, 1, 2, 3, 5)).sqrt().tolist(),
        "full_replay_last_support_rms": rms(relative_post.exp() - post.exp()),
        "source_baseline": [tensor.tolist() for tensor in source],
        "source_relative": [tensor.tolist() for tensor in relative_source],
        "attention_mass": masses.tolist(), "attention_baseline_mass": evidence.tolist(),
        "reference_log_mass_ratio": info["reference_log_mass_ratio"].tolist(),
        "records": {name: fields(values) for name, values in (
            ("fixed_root_baseline", old_xy), ("fixed_root_relative", new_xy),
            ("fixed_root_omit_baseline", old_omit_xy), ("fixed_root_omit_relative", new_omit_xy),
            ("entire_relative_replay", entire_xy),
        )},
        "last_real_write": write,
    }


if __name__ == "__main__":
    OUTPUT.mkdir(exist_ok=True)
    target = OUTPUT / "stage.json"
    if target.exists():
        raise FileExistsError(target)
    result = run(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False, separators=(",", ":"))
    print(target, flush=True)
