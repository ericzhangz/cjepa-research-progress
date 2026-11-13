"""Compare merged and expanded §16 forecasts on archived PushT action branches."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from native_response_mixture_pilot import (
    DEVELOPMENT, NORMALIZATION, QUERY_STEPS, ROOT, forecasts, future_inputs, issue_event,
    load_panel, matched_worlds, subsequent_roots,
)


def h2_response_tables(model, windows, previous, candidates):
    roots, forks, horizon, _ = candidates.shape
    if horizon != 2:
        raise ValueError("Expanded reference requires exactly two candidate actions")
    history = windows.repeat_interleave(forks, dim=0)
    control = torch.cat((previous[:, None].expand(-1, forks, -1, -1), candidates), dim=2)
    emb = model.action_encoder(control.flatten(0, 1))
    initial = model.response_mixture.initial_state(len(history))
    _, _, issued = model.predict_response(history, emb[:, :3], initial, factual=True)

    branches = torch.cat((
        history[:, None, 1:].expand(-1, 3, -1, -1), issued.means[:, :, None],
    ), dim=2).flatten(0, 1)
    branch_emb = emb[:, 1:4].repeat_interleave(3, dim=0)
    branch_prior = model.response_mixture.initial_state(len(branches))
    _, _, expanded = model.predict_response(branches, branch_emb, branch_prior, factual=True)
    return issued.means.double(), issued.reset_logit.double().sigmoid(), expanded.means.unflatten(0, (-1, 3)).double()


def expand_h2(model, windows, previous, candidates, beliefs):
    count, forks = candidates.shape[:2]
    h1, reset, table = h2_response_tables(model, windows, previous, candidates)
    prior = model.response_mixture.log_prior.double().softmax(dim=-1)
    transition = (1.0 - reset)[:, None, None] * torch.eye(3, dtype=h1.dtype, device=h1.device)
    transition = transition + reset[:, None, None] * prior[None, None, :]
    results = {}
    for label, belief in beliefs.items():
        p = belief.double().softmax(dim=-1).repeat_interleave(forks, dim=0)
        weights = p[..., None] * transition
        q = weights.sum(dim=1)
        z1 = (p[..., None] * h1).sum(dim=1)
        independent = (p[..., None, None] * q[:, None, :, None] * table).sum(dim=(1, 2))
        expanded = (weights[..., None] * table).sum(dim=(1, 2))
        merged = forecasts(model, windows, previous, candidates, belief).double()
        torch.testing.assert_close(merged[:, :, 0], z1.reshape(count, forks, -1), rtol=1e-4, atol=4e-6)
        results.update({
            f"{label}_merged": merged,
            f"{label}_product": torch.stack((merged[:, :, 0], independent.reshape(count, forks, -1)), dim=2),
            f"{label}_joint": torch.stack((merged[:, :, 0], expanded.reshape(count, forks, -1)), dim=2),
            f"{label}_weights_h2": weights.reshape(count, forks, 3, 3),
        })
    results["first_modes"] = h1.reshape(count, forks, 3, -1)
    results["second_modes_by_parent"] = table.reshape(count, forks, 3, 3, -1)
    results["transition_h1"] = transition.reshape(count, forks, 3, 3)
    return results


def collect(root, device):
    sys.path.insert(0, str(ROOT / "research_sources_2026-10-22/lewm_assets"))
    from prepare_assets import load_model

    actions = json.loads(NORMALIZATION.read_text(encoding="utf-8"))
    normalizer = [torch.tensor(actions[key], device=device) for key in ("mean", "std")]
    panel = load_panel(DEVELOPMENT, device, normalizer)
    model = load_model(device).eval().requires_grad_(False)
    model.enable_response_mixture().requires_grad_(False)
    checkpoint = torch.load(root / "pilot.pt", map_location=device, weights_only=True)
    model.predictor.transformer.layers[-1].load_state_dict(checkpoint["last_block"])
    model.pred_proj.load_state_dict(checkpoint["pred_proj"])
    model.response_mixture.load_state_dict(checkpoint["response_mixture"])
    model.event_coordinates = checkpoint["event_coordinates"]

    recorded = {}
    def keep(name, value):
        recorded.setdefault(name, []).append(value.unflatten(0, (-1, 4)).cpu().numpy())

    with torch.no_grad():
        for start in range(0, len(panel.scene_id), 4):
            rows = slice(start, start + 4)
            visible, commands, _, _, skip, write, _, _ = issue_event(model, panel, rows)
            states = subsequent_roots(model, visible, commands, write)
            beliefs = {"skip": skip, "write": states[30], "carry": write}
            args = future_inputs(panel, rows, visible, commands, 30)
            for label, value in expand_h2(model, *args, beliefs).items():
                keep(label, value)
            for label, state in beliefs.items():
                keep(f"p30_{label}", state.exp())
            # At step 20, exactly one no-evidence factual transition has elapsed since the write.
            w20, past20, choices = future_inputs(panel, rows, visible, commands, 20)
            after_20 = model.predict_response(visible[:, 1:4], model.action_encoder(commands[:, 1:4]), write)[1]
            trace = {}
            for label, state in (("skip", skip), ("write", after_20), ("carry", write)):
                pred = forecasts(model, w20, past20, choices, state, trace=trace if label == "write" else None)
                keep(f"t20_{label}_merged", pred)
                keep(f"p20_{label}", state.exp())
            keep("t20_first_modes", trace["h1_modes"])

    a = {key: np.concatenate(parts) for key, parts in recorded.items()}
    pairs = matched_worlds(panel, slice(None))
    a.update(scene_id=panel.scene_id, pair_scene=pairs[0].cpu().numpy(),
             positive_world=pairs[1].cpu().numpy(), negative_world=pairs[2].cpu().numpy(),
             axis=panel.forward_axis.cpu().double().numpy())
    for step in (20, 30):
        a[f"true_xy{step}"] = panel.object_xy[:, :, QUERY_STEPS.index(step)].cpu().double().numpy()
        a[f"true_z{step}"] = panel.future_full[:, :, QUERY_STEPS.index(step)].cpu().numpy()
    return a, [value.cpu().double().numpy() for value in checkpoint["object_coordinates"]]


def summarize(archive, coordinates):
    w, bias = coordinates
    pair_scene, plus, minus = [archive[k] for k in ("pair_scene", "positive_world", "negative_world")]
    first, second = np.triu_indices(4, 1)

    def scores(xy):
        return np.einsum("swabc,sc->swab", xy, archive["axis"])

    def order(score):
        gap = score[:, :, first] - score[:, :, second]
        return np.where(np.abs(gap) > 1e-6, np.sign(gap), 0)

    results = {}
    for step, arms in ((20, [f"t20_{a}_merged" for a in ("skip", "write", "carry")]),
                       (30, [f"{a}_{f}" for a in ("skip", "write", "carry")
                             for f in ("merged", "product", "joint")])):
        target = archive[f"true_xy{step}"]
        true_scores = scores(target)
        signs = order(true_scores)
        valid = signs != 0
        reversal = signs[pair_scene, plus] * signs[pair_scene, minus] < 0
        metrics = {}
        for arm in arms:
            prediction = archive[arm]
            xy = prediction @ w + bias
            error = xy - target
            relative = error - error.mean(axis=2, keepdims=True)
            sim = scores(xy)
            pred_order = order(sim)
            correct = signs == pred_order
            credit = (correct + 0.5 * (pred_order == 0)) * valid
            both = correct[pair_scene, plus] & correct[pair_scene, minus] & reversal
            realized = np.take_along_axis(true_scores, sim.argmax(axis=2)[:, :, None], axis=2).squeeze(2)
            regret = true_scores.max(axis=2) - realized
            sq_by_scene = (relative**2).sum(-1).mean((1, 2))
            metrics[arm] = {
                "latent_mse": ((prediction - archive[f"true_z{step}"])**2).mean((0, 1, 2, 4)).tolist(),
                "centered_xy_rmse": np.sqrt(sq_by_scene.mean(0)).tolist(),
                "absolute_xy_rmse": np.sqrt((error**2).sum(-1).mean((0, 1, 2))).tolist(),
                "concordance": (credit.sum((0, 1, 2)) / valid.sum((0, 1, 2))).tolist(),
                "both_correct_reversals": both.sum((0, 1)).tolist(),
                "regret": regret.mean((0, 1)).tolist(),
            }
            archive[f"{arm}_centered_mse_by_scene"] = sq_by_scene
            archive[f"{arm}_candidate_scores"] = sim
            archive[f"{arm}_rank"] = pred_order
            archive[f"{arm}_strict_reversal"] = both
        results[str(step)] = {"reversal_counts": reversal.sum((0, 1)).tolist(), "arms": metrics}

    terms = {}
    for initial in ("skip", "write", "carry"):
        merge, product, joint = [archive[f"{initial}_{label}"][:, :, :, 1] for label in ("merged", "product", "joint")]
        curvature, association = product - merge, joint - product
        archive[f"{initial}_curvature_h2"] = curvature
        archive[f"{initial}_association_h2"] = association

        def centered_physical(values):
            xy = values @ w
            return xy - xy.mean(axis=2, keepdims=True)

        q, a = [centered_physical(value) for value in (curvature, association)]
        error = centered_physical(merge)
        truth = archive["true_xy30"][:, :, :, 1]
        error -= truth - truth.mean(axis=2, keepdims=True)
        by_scene = ((2 * error + q + a) * (q + a)).sum(-1).mean((1, 2))
        archive[f"{initial}_expand_minus_merge_centered_mse_by_scene"] = by_scene
        rng = np.random.default_rng(1611)
        bootstrap = rng.integers(0, len(by_scene), (5000, len(by_scene)))
        mrank = archive[f"{initial}_merged_rank"][..., 1]
        jrank = archive[f"{initial}_joint_rank"][..., 1]
        terms[initial] = {
            "curvature_centered_xy_rms": float(np.sqrt((q * q).sum(-1).mean())),
            "association_centered_xy_rms": float(np.sqrt((a * a).sum(-1).mean())),
            "total_centered_xy_rms": float(np.sqrt(((q + a)**2).sum(-1).mean())),
            "expanded_minus_merged_centered_mse": float(by_scene.mean()),
            "centered_mse_change_95pct": np.quantile(by_scene[bootstrap].mean(1), (.025, .975)).tolist(),
            "improved_scenes": int((by_scene < 0).sum()),
            "changed_h2_candidate_orders": int((mrank != jrank).sum()),
        }
    results["h2_terms"] = terms

    for step, mode_key in ((20, "t20_first_modes"), (30, "first_modes")):
        mode = archive[mode_key] @ w
        mode -= mode.mean(axis=2, keepdims=True)
        spread = mode[..., :, None, :] - mode[..., None, :, :]
        diameter = np.sqrt((spread**2).sum(-1)).max(axis=(-2, -1))
        pos = archive[f"true_xy{step}"][..., 0, :]
        diff = pos[pair_scene, plus] - pos[pair_scene, minus]
        diff -= diff.mean(axis=1, keepdims=True)
        results[str(step)]["h1_mode_centered_diameter_mean"] = float(diameter.mean())
        results[str(step)]["h1_true_paired_centered_xy_rms"] = float(np.sqrt((diff**2).sum(-1).mean()))
        archive[f"t{step}_h1_centered_mode_diameter"] = diameter
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    archive, coordinates = collect(args.checkpoint, args.device)
    results = summarize(archive, coordinates)
    args.output.mkdir(parents=True)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "intermediates.npz", **archive)
    print(json.dumps(results["h2_terms"], indent=2))


if __name__ == "__main__":
    main()
