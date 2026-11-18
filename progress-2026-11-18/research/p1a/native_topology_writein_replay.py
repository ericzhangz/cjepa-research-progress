"""Replay actual FourRoom writes and score fixed H5 plans from the arrived windows."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .native_full_depth_conditioning import advance_log_support, write_observation
from .native_topology_condition import (
    HOLDOUT, ROOT, RUN as REFERENCE, SWITCH, forecast, load_readout, load_trained, predict_sources,
)
from .physical_topology_gate import ACTION_BLOCK, PhysicalRollout, VisibleMapCost, make_scene, shared_cem


RUN = ROOT / "runs/native_topology_paired_20261115"
CHECKPOINTS = {
    "start": ROOT / "runs/native_topology_prefix_20261115/prefix",
    "control": RUN / "control",
    "paired": RUN / "paired",
}
STAGES = ("first_probe", "reentry_probe", "reversed_probe", "after_reversal_quiet")
POLICIES = ("actual", "omit", "uniform", "true_mode")
HORIZON = 5
BATCH = 128


def prepare_banks(archive):
    """Check exact reuse of factual windows and fix the same two-mode pool for all arms."""
    mode, event, valid = (archive[key] for key in ("life_mode", "life_event", "life_valid"))
    xy, z, action = (archive[key] for key in ("life_xy", "life_z", "life_action"))
    np.testing.assert_array_equal(xy[:, :, 1:][valid], archive["life_raw_xy"][valid, -1])
    count = valid.sum(-1)
    np.testing.assert_array_equal(valid, np.arange(valid.shape[-1]) < count[..., None])
    np.testing.assert_array_equal(np.sort(event[event > 0]), np.repeat([1, 2, 3], count.size))
    assert np.all(event[..., :2] == 0)
    assert np.all(archive["life_informative"][event > 0])
    assert np.all(mode[event == 3] == 1 - np.where(event == 3)[1])
    assert np.all(mode[event == 3] == mode[(*np.indices(count.shape), count - 1)].ravel())

    scene_ids, positions, histories, past, controls, terminal, choices = [], [], [], [], [], [], []
    for i, seed in enumerate(archive["seed"]):
        scene = make_scene(int(seed))
        physical, score = PhysicalRollout(scene), VisibleMapCost(scene, scene.southeast)
        for root in (0, 1, 2):
            if root == 2:
                initial = np.arange(2)
                last = count[i] - 1
                for key, factual in (("root_xy", xy), ("root_z", z), ("root_action", action)):
                    span = [-1, 0] if key == "root_action" else [-1, 0, 1]
                    matched = factual[i, initial[:, None], last[:, None] + span]
                    np.testing.assert_array_equal(matched, np.broadcast_to(archive[key][i, 2], matched.shape))
                start = archive["root_xy"][i, 2, -1]
                history, previous = archive["root_z"][i, 2], archive["root_action"][i, 2]
                stage = np.full(2, 3)
            else:
                initial, last = np.where((event[i] > 0) & (mode[i] == root))
                assert len(last) == 3
                root_xy = xy[i, initial, last + 1]
                root_z = z[i, initial[:, None], last[:, None] + [-1, 0, 1]]
                root_action = action[i, initial[:, None], last[:, None] + [-1, 0]]
                for measured in (root_xy, root_z, root_action):
                    np.testing.assert_array_equal(measured, np.broadcast_to(measured[0], measured.shape))
                start, history, previous = root_xy[0], root_z[0], root_action[0]
                stage = event[i, initial, last] - 1
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0]))
            plans, costs = shared_cem(scene, physical, score, start, HORIZON, rng)
            plans = np.repeat(plans[:, :, None], ACTION_BLOCK, axis=2)
            choices.extend((i, b, t, s, len(positions)) for b, t, s in zip(initial, last, stage))
            scene_ids.append(seed)
            positions.append(start)
            histories.append(history)
            past.append(previous)
            controls.append(plans)
            terminal.append(costs)
    return dict(seed=np.asarray(scene_ids), root_xy=np.stack(positions), root_z=np.stack(histories),
                root_action=np.stack(past), action=np.stack(controls), physical_cost=np.stack(terminal),
                choices=np.asarray(choices))


def replay_evidence(modules, archive, scale, device):
    """Use only the actual action's issued H1 latents and its subsequently arrived latent."""
    panel = {k: torch.as_tensor(v, device=device) for k, v in archive.items() if k != "future_z"}
    issued, exposure, index = predict_sources(*modules, panel)
    rows, initial, time = index
    arrived = panel["life_z"][rows, initial, time + 1]
    alternate = panel["life_cf_z"][index]
    modes = panel["life_mode"][index].long()
    np.testing.assert_array_equal(arrived.cpu(), alternate[torch.arange(len(modes), device=device), modes].cpu())
    shape = panel["life_valid"].shape
    actual = issued.new_full((*shape[:2], 2), -float(np.log(2)))
    omit_reversed = actual.clone()
    prior_log = issued.new_full((*shape, 2), torch.nan)
    actual_log, omit_log, likelihoods = (prior_log.clone() for _ in range(3))
    for step in range(2, shape[-1]):
        select = time == step
        a, b = rows[select], initial[select]
        prior = advance_log_support(actual[a, b], SWITCH)
        without = advance_log_support(omit_reversed[a, b], SWITCH)
        actual[a, b], ll = write_observation(prior, issued[select], arrived[select], scale)
        include = (panel["life_event"][a, b, step] != 3)[:, None]
        omit_reversed[a, b] = (without + ll * include).log_softmax(-1)
        prior_log[a, b, step], actual_log[a, b, step] = prior, actual[a, b]
        omit_log[a, b, step], likelihoods[a, b, step] = omit_reversed[a, b], ll

    # On a pair, the actual departure from the target midpoint contributes half the true gap.
    # The remaining evidence is the projection of the common midpoint error onto the issued mode gap.
    def signed_gap(values):
        return (values[:, 0] - values[:, 1]) * (1 - 2 * modes[:, None])

    pred_gap = signed_gap(issued.double()) / scale
    true_gap = signed_gap(alternate.double()) / scale
    midpoint_error = (alternate.double().mean(1) - issued.double().mean(1)) / scale
    residual = (arrived.double() - issued.double().mean(1)) / scale
    gap = (pred_gap * residual).sum(-1)
    pair_effect = 0.5 * (pred_gap * true_gap).sum(-1)
    midpoint_effect = (pred_gap * midpoint_error).sum(-1)
    torch.testing.assert_close(gap, pair_effect + midpoint_effect, rtol=1e-10, atol=1e-10)
    actual_ll = likelihoods[index].double()
    ll_gap = (actual_ll[:, 0] - actual_ll[:, 1]) * (1 - 2 * modes)
    torch.testing.assert_close(gap, ll_gap, rtol=5e-4, atol=1e-2)
    tracks = dict(prior=prior_log, actual=actual_log, omit_reversed=omit_log, log_likelihood=likelihoods)
    samples = dict(mode=modes, exposure=exposure, signed_gap=gap, pair_effect=pair_effect,
                   midpoint_effect=midpoint_effect, pred_gap_norm=pred_gap.norm(dim=-1),
                   true_gap_norm=true_gap.norm(dim=-1), midpoint_error_norm=midpoint_error.norm(dim=-1),
                   actual_residual_norm=residual.norm(dim=-1))
    return {key: value.cpu().numpy() for key, value in (tracks | samples).items()} | {
        "source_index": torch.stack(index).cpu().numpy(),
    }


def score_banks(modules, banks, decode, device):
    learned = np.empty_like(banks["physical_cost"])
    for i, seed in enumerate(banks["seed"]):
        scene = make_scene(int(seed))
        geometry = VisibleMapCost(scene, scene.southeast)
        h, p, plans = (torch.as_tensor(banks[k][i:i + 1], device=device)
                       for k in ("root_z", "root_action", "action"))
        for start in range(0, plans.size(1), BATCH):
            sl = slice(start, start + BATCH)
            z = forecast(*modules, h, p, plans[:, sl], compact=True)[0, ..., -1, :]
            learned[i, :, sl] = geometry(decode(z).cpu().numpy()).T
    if not np.isfinite(learned).all():
        raise ValueError("Expected finite scores in all fixed H5 plans")
    return learned


def decisions(archive, banks, evidence, learned):
    rows, initial, time, stage, pool = banks["choices"].T
    index = rows, initial, time
    mode = archive["life_mode"][index]
    before, after, omission = (np.exp(evidence[k][index]) for k in ("prior", "actual", "omit_reversed"))
    omit = np.where((stage == 3)[:, None], omission, before)
    weights = np.stack((after, omit, np.full_like(after, .5), np.eye(2)[mode]), axis=1)
    best = (weights @ learned[pool]).argmin(axis=-1)
    physical = banks["physical_cost"][pool, mode]
    physical_best = physical.argmin(axis=-1)
    physical_uniform = banks["physical_cost"][pool].mean(axis=1).argmin(axis=-1)
    expected_cost = np.take_along_axis(physical, best, axis=1)
    exact_min = physical.min(axis=1)
    command = banks["action"][pool[:, None], best, 0, 0]
    physical_command = banks["action"][pool, physical_best, 0, 0]
    physical_uniform_command = banks["action"][pool, physical_uniform, 0, 0]
    physical_uniform_regret = np.take_along_axis(physical, physical_uniform[:, None], axis=1)[:, 0] - exact_min
    physical_space = (physical_uniform_regret > 0) & (physical_command != physical_uniform_command).any(-1)
    return dict(stage=stage, mode=mode, weights=weights, best=best, command=command,
                true_cost=expected_cost, regret=expected_cost - exact_min[:, None], physical_best=physical_best,
                physical_uniform=physical_uniform, physical_uniform_regret=physical_uniform_regret,
                physical_decision_space=physical_space)


def summarize(archive, evidence, selected):
    index = tuple(evidence["source_index"])
    events = archive["life_event"][index]
    informative = archive["life_informative"][index]
    groups = {name: events == i + 1 for i, name in enumerate(STAGES[:3])}
    groups["noninformative"] = ~informative
    groups["other_informative"] = informative & (events == 0)
    samples = {}
    sign = 2 * evidence["mode"] - 1
    before = np.diff(evidence["prior"][index], axis=-1)[:, 0] * sign > 0
    after = np.diff(evidence["actual"][index], axis=-1)[:, 0] * sign > 0
    for name, take in groups.items():
        gap = evidence["signed_gap"][take]
        samples[name] = {"arrivals": int(take.sum()), "supports_true": int((gap > 0).sum()),
                         "supports_wrong": int((gap < 0).sum()), "ties": int((gap == 0).sum()),
                         "gained_true_majority": int((take & ~before & after).sum()),
                         "lost_true_majority": int((take & before & ~after).sum()),
                         "gap_quantiles": np.quantile(gap, [0, .01, .5, .99, 1]).tolist(),
                         **{f"{k}_mean": float(evidence[k][take].mean()) for k in (
                             "pair_effect", "midpoint_effect", "pred_gap_norm", "true_gap_norm",
                             "midpoint_error_norm", "actual_residual_norm",
                         )}}

    stages = {}
    for i, name in enumerate(STAGES):
        stage = selected["stage"] == i
        space = selected["physical_decision_space"][stage]
        weights, regret, command, best = (selected[k][stage] for k in ("weights", "regret", "command", "best"))
        modes = selected["mode"][stage]
        probabilities = np.take_along_axis(weights, modes[:, None, None], axis=-1)[..., 0]
        policy_metrics = {}
        for j, policy in enumerate(POLICIES):
            policy_metrics[policy] = {
                "true_support_mean": float(probabilities[:, j].mean()),
                "true_support_over_half": int((probabilities[:, j] > .5).sum()),
                "mean_h5_regret": float(regret[:, j].mean()),
                "mean_h5_regret_where_physical_first_action_matters": float(regret[space, j].mean()) if space.any() else None,
            }
        comparisons = {}
        for j, policy in enumerate(POLICIES[1:3], start=1):
            changed = (command[:, 0] != command[:, j]).any(-1)
            improvement = regret[:, j] - regret[:, 0]
            comparisons[f"actual_vs_{policy}"] = {
                "changed_candidates": int((best[:, 0] != best[:, j]).sum()),
                "changed_first_actions": int(changed.sum()),
                "changed_first_actions_where_physical_first_action_matters": int((changed & space).sum()),
                "more_favorable_open_loop_h5": int((improvement > 0).sum()),
                "less_favorable_open_loop_h5": int((improvement < 0).sum()),
                "mean_open_loop_h5_improvement": float(improvement.mean()),
                "mean_open_loop_h5_improvement_with_changed_first_action": float(improvement[changed].mean()) if changed.any() else None,
            }
        stages[name] = {"arrivals": int(stage.sum()), "physical_first_action_and_regret_change": int(space.sum()),
                        "physical_uniform_mean_h5_regret": float(selected["physical_uniform_regret"][stage].mean()),
                        "policies": policy_metrics, **comparisons}
    return dict(source=samples, stage=stages)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare", action="store_true", help="Verify factual roots and save the physics-scored banks")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    RUN.mkdir(parents=True, exist_ok=True)
    archive_path = REFERENCE / "data_holdout.npz"
    bank_path = RUN / "writein_fixed_h5_banks.npz"
    with np.load(archive_path, allow_pickle=False) as source:
        archive = dict(source)
    np.testing.assert_array_equal(archive["seed"], list(HOLDOUT))
    if args.prepare:
        banks = prepare_banks(archive)
        np.savez_compressed(bank_path, **banks)
        print(f"Verified {len(banks['choices'])} factual read requests over {len(banks['seed'])} fixed H5 banks")
        return

    with np.load(bank_path, allow_pickle=False) as source:
        banks = dict(source)
    device = torch.device(args.device)
    scale = torch.load(REFERENCE / "likelihood_scale.pt", map_location=device, weights_only=True)
    decode = load_readout(device, REFERENCE)
    results = {}
    for name, run in CHECKPOINTS.items():
        modules = load_trained(run, "conditioned", device)
        evidence = replay_evidence(modules, archive, scale, device)
        learned = score_banks(modules, banks, decode, device)
        selected = decisions(archive, banks, evidence, learned)
        results[name] = summarize(archive, evidence, selected)
        np.savez_compressed(RUN / f"writein_{name}.npz", learned_cost=learned,
                            **evidence, **{f"choice_{k}": v for k, v in selected.items()})
        del modules
        print(f"Finished {name}", flush=True)
    report = {"archive": str(archive_path.relative_to(ROOT)), "bank": str(bank_path.relative_to(ROOT)),
              "training": {name: str(run.relative_to(ROOT) / "conditioned.pt") for name, run in CHECKPOINTS.items()},
              "policies": list(POLICIES), "stage_order": list(STAGES),
              "first_command_decision_space": "Physical true-mode and exact uniform change first command and true H5 cost",
              "omit": "Drop only the current likelihood at each probe, or the reversed probe for the final quiet root",
              "scope": "Fixed factual histories and physics-generated fixed candidate plans. No adaptive execution.",
              "switch_probability": SWITCH, "results": results}
    (RUN / "writein_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
