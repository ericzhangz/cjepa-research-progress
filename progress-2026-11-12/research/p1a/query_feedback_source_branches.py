"""Evaluate fixed source-future checkpoints on recorded FourRoom branches."""

import argparse
from pathlib import Path

import numpy as np
import torch

from history_match import load_host
from native_response import FeedbackMemory, NativeResponseRegulator
from query_feedback_learning import HORIZONS, write_csv, write_json


ROOT_IDS = np.arange(717000, 717032)
CANDIDATE_IDS = np.arange(32)
CARRIER = 1
REFRESH_CANDIDATES = np.array([1, 9, 10])
TARGETS = np.arange(10, 126, 5)
MATCHED_TARGETS = np.array([25, 30, 40])
CANDIDATES = {10: CANDIDATE_IDS, 20: REFRESH_CANDIDATES}
MEMORY_MODES = {10: ("retained", "cleared"), 20: ("retained", "stale_memory", "cleared")}
CHECKPOINT_ARMS = ("query_feedback", "native_finetune")
HORIZON_SLOTS = np.array(HORIZONS) - 1


def float_array(archive, name, shape):
    array = archive[name]
    if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite FP32 {name} with shape {shape}")
    return array


def assert_shared(array, candidates, label):
    group = array[:, candidates]
    np.testing.assert_array_equal(group, np.broadcast_to(group[:, :1], group.shape), err_msg=label)


def load_recorded(args):
    roots, candidates = len(ROOT_IDS), len(CANDIDATE_IDS)
    sparse_steps = [15, 25, 30, 45, 50, 85, 90, 120, 125]
    with np.load(args.branches, allow_pickle=False) as bank, np.load(args.forecasts, allow_pickle=False) as replay:
        for archive in (bank, replay):
            np.testing.assert_array_equal(archive["root_id"], ROOT_IDS)
            np.testing.assert_array_equal(archive["candidate_id"], CANDIDATE_IDS)
        names = bank["candidate_name"]
        np.testing.assert_array_equal(names, replay["candidate_name"])
        np.testing.assert_array_equal(names[REFRESH_CANDIDATES], ["constant_0", "turn_0_-1", "turn_0_+1"])
        np.testing.assert_array_equal(bank["prefix_absolute_step"], [0, 5, 10])
        np.testing.assert_array_equal(bank["true_rgb_absolute_step"], sparse_steps)
        np.testing.assert_array_equal(bank["absolute_step"], np.arange(126))
        np.testing.assert_array_equal(replay["target_step"], TARGETS)
        prefix = float_array(bank, "z_prefix", (roots, 3, 192))
        observed = float_array(replay, "observed_z", (roots, candidates, 24, 192))
        sparse_truth = float_array(bank, "z_true", (roots, candidates, len(sparse_steps), 192))
        prefix_raw = float_array(bank, "prefix_raw_actions", (roots, 2, 5, 2))
        prefix_norm = float_array(bank, "prefix_normalized_actions", (roots, 2, 10))
        raw = float_array(bank, "raw_actions", (roots, candidates, 23, 5, 2))
        norm = float_array(bank, "normalized_actions", (roots, candidates, 23, 10))
        xy = float_array(bank, "agent_xy", (roots, candidates, 126, 2))
    np.testing.assert_array_equal(observed[:, :, 0], np.broadcast_to(prefix[:, None, 2], observed[:, :, 0].shape))
    np.testing.assert_array_equal(observed[:, :, np.searchsorted(TARGETS, sparse_steps)], sparse_truth)
    assert_shared(xy[:, :, :11], CANDIDATE_IDS, "All t10 candidates must share the physical prefix")
    assert_shared(raw[:, :, :2], REFRESH_CANDIDATES, "Raw controls through t20")
    assert_shared(norm[:, :, :2], REFRESH_CANDIDATES, "Normalized controls through t20")
    assert_shared(observed[:, :, :3], REFRESH_CANDIDATES, "Observed latents through t20")
    assert_shared(xy[:, :, :21], REFRESH_CANDIDATES, "Physical prefix through t20")
    next_controls = raw[:, REFRESH_CANDIDATES, 2]
    if any(len(np.unique(controls, axis=0)) != len(REFRESH_CANDIDATES) for controls in next_controls):
        raise ValueError("The three suffixes must issue distinct controls from the shared t20 state")
    return {
        "candidate_name": names,
        "factual_z": np.concatenate((prefix, observed[:, CARRIER, 1:3]), axis=1),
        "prefix_raw": prefix_raw, "prefix_norm": prefix_norm, "raw": raw, "norm": norm,
        "observed": observed, "xy": xy,
    }


def load_checkpoint(model, operator, path, arm):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected = {"epoch": 60, "optimizer_steps": 240, "arm": arm, "feedback": "source"}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Expected the fixed final source-future checkpoint for {arm}: {path}")
    weight = operator.attention.to_qkv.weight
    names = [name for name, parameter in model.named_parameters() if parameter is weight]
    selected = checkpoint["selected_parameters"]
    if len(names) != 1 or set(selected) != set(names) or weight.numel() != 294912:
        raise ValueError(f"Expected only the first native attention QKV weight: {path}")
    value = selected[names[0]]
    if value.shape != weight.shape or value.dtype != weight.dtype or not torch.isfinite(value).all():
        raise ValueError(f"Incompatible QKV weight in {path}")
    weight.copy_(value)


def form_memory(operator, visible, actions, memory, end, timing):
    predictions = []
    for transition in range(memory.size, end):
        start = max(0, transition - 2)
        issued = timing.measure("predictor", len(visible), operator.predict,
                                visible[:, start:transition + 1], actions[:, start:transition + 1], memory)
        if issued.source is None or issued.source.slot != transition - start:
            raise RuntimeError("Factual prediction must retain the last-token source")
        updated = timing.measure("observe", len(visible), operator.observe,
                                 memory, issued, visible[:, transition + 1])
        if updated.size != transition + 1:
            raise RuntimeError("Each real arrival must write one record")
        for old, new in ((memory.queries, updated.queries), (memory.values, updated.values)):
            torch.testing.assert_close(new[:, :, :-1], old, atol=0, rtol=0)
        torch.testing.assert_close(updated.queries[:, :, -1], issued.query, atol=0, rtol=0)
        memory = updated
        predictions.append(issued.latent)
    return memory, torch.stack(predictions, dim=1)


def rollout(predict_next, history, actions, timing):
    predictions = []
    for offset in range(max(HORIZONS)):
        successor = timing.measure("predictor", len(history), predict_next, history, actions[:, offset:offset + 3])
        predictions.append(successor)
        history = torch.cat((history[:, 1:], successor[:, None]), dim=1)
    return torch.stack(predictions, dim=1)


def saved(array):
    result = array.cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite value produced by the fixed replay")
    return result


def decode_xy(host, latent, timing):
    xy = host.decode(latent, timing)[..., :2].copy()
    if not np.isfinite(xy).all():
        raise ValueError("The frozen agent probe produced nonfinite XY")
    return xy


@torch.no_grad()
def evaluate_arm(args, data, arm, checkpoint):
    host = load_host(args)
    from host_adapter import Timings

    model = host.model
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("The entire host must be frozen in evaluation mode")
    np.testing.assert_array_equal(host.normalize(data["prefix_raw"]), data["prefix_norm"])
    np.testing.assert_array_equal(host.normalize(data["raw"]), data["norm"])
    timing = Timings()
    visible = torch.as_tensor(data["factual_z"], device=host.device)
    factual_actions = np.concatenate((data["prefix_norm"], data["norm"][:, CARRIER, :2]), axis=1)

    def embed(normalized):
        return model.action_embed(model.action_encoder(torch.as_tensor(normalized, device=host.device)))

    with timing.stage("neutral_action_embedding"):
        zero = host.normalize(np.zeros((1, 1, 5, 2), dtype=np.float32))
        neutral = timing.measure("action_embed", 1, embed, zero)[0, 0]
    operator = NativeResponseRegulator(model, neutral, feedback="source")
    load_checkpoint(model, operator, checkpoint, arm)
    source_arm = arm == "query_feedback"
    root_count = len(ROOT_IDS)
    forecasts, memory_trace, calls = {}, {}, {}
    if source_arm:
        memory = operator.empty_memory(root_count)
        with timing.stage("factual_action_embedding"):
            factual_embedded = timing.measure("action_embed", root_count * 4, embed, factual_actions)
        memory_trace["factual_embedded_actions"] = saved(factual_embedded)

    for step, candidates in CANDIDATES.items():
        block = (step - 10) // 5
        end = step // 5
        candidate_count = len(candidates)
        batch = root_count * candidate_count
        prefix = np.broadcast_to(factual_actions[:, None, end - 2:end],
                                 (root_count, candidate_count, 2, 10))
        suffix = data["norm"][:, candidates, block:block + max(HORIZONS)]
        conditions = np.concatenate((prefix, suffix), axis=2).reshape(batch, 2 + max(HORIZONS), 10)
        with timing.stage(f"step{step}_action_embedding"):
            actions = timing.measure("action_embed", batch * conditions.shape[1], embed, conditions)
        history = visible[:, end - 2:end + 1].repeat_interleave(candidate_count, dim=0)
        modes = MEMORY_MODES[step] if source_arm else ("native",)
        if source_arm:
            with timing.stage(f"step{step}_factual_formation"):
                memory, prediction = form_memory(operator, visible, factual_embedded, memory, end, timing)
            if step == 10:
                m2 = memory
            memory_trace.update({f"M{end}_queries": saved(memory.queries), f"M{end}_values": saved(memory.values),
                                 f"step{step}_factual_predictions": saved(prediction)})
            if operator.source_capture_calls != end or operator.feedback_feature_calls != 2 * end:
                raise RuntimeError("Source captures and consequence responses must occur only on factual transitions")

        forecasts[step] = {}
        for mode in modes:
            label = f"step{step}_{mode}"
            if source_arm:
                if mode == "cleared":
                    state = operator.empty_memory(batch)
                else:
                    selected = m2 if mode == "stale_memory" else memory
                    state = FeedbackMemory(selected.queries.repeat_interleave(candidate_count, dim=0),
                                           selected.values.repeat_interleave(candidate_count, dim=0))
                before = operator.calls

                def predict_next(z, a):
                    issued = operator.predict(z, a, state, retain_source=False)
                    if issued.source is not None:
                        raise RuntimeError("An imagined branch retained a factual source")
                    return issued.latent
            else:
                def predict_next(z, a):
                    return model.predict(z, a)[:, -1]
            with timing.stage(label):
                prediction = rollout(predict_next, history, actions, timing)
            calls[label] = timing.records[label]["operations"]["predictor"]["calls"]
            if source_arm and (operator.calls - before != max(HORIZONS) or
                               operator.source_capture_calls != end or operator.feedback_feature_calls != 2 * end):
                raise RuntimeError("An imagined branch captured a source or fulfilled a transition")
            forecast = prediction.reshape(root_count, candidate_count, max(HORIZONS), 192)
            with timing.stage(f"{label}_probe"):
                xy = decode_xy(host, forecast, timing)
            forecasts[step][mode] = (saved(forecast), xy)
        if source_arm:
            np.testing.assert_array_equal(saved(memory.queries), memory_trace[f"M{end}_queries"])
            np.testing.assert_array_equal(saved(memory.values), memory_trace[f"M{end}_values"])

    if not source_arm:
        # Decode the common frozen encoder's truth with the same frozen probe.
        for step, candidates in CANDIDATES.items():
            block = (step - 10) // 5
            truth = data["observed"][:, candidates, block + 1:block + 1 + max(HORIZONS)]
            with timing.stage(f"step{step}_true_latent_probe"):
                latent = torch.as_tensor(truth, device=host.device)
                memory_trace[f"step{step}_true_probe_xy"] = decode_xy(host, latent, timing)
    if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise RuntimeError("Temporary native predictor hooks remained installed")
    trace = {"stages": timing.records, "forecast_batch_calls": calls,
             "source_capture_calls": operator.source_capture_calls,
             "consequence_answer_calls": operator.feedback_feature_calls, "encoder_calls": 0}
    return forecasts, memory_trace, trace


def panel(arrays, name, step, candidates, targets, truth_probe, data, forecasts):
    block = (step - 10) // 5
    slots = (targets - step) // 5 - 1
    indices = np.searchsorted(CANDIDATES[step], candidates)
    truth = data["observed"][:, candidates][:, :, block + 1 + slots]
    true_xy = data["xy"][:, candidates][:, :, targets]
    arrays.update({f"{name}_candidate_id": candidates, f"{name}_target_step": targets,
                   f"{name}_true_z": truth, f"{name}_true_xy": true_xy})
    scores = {}
    for mode, (latent, xy) in forecasts[step].items():
        latent = latent[:, indices][:, :, slots]
        xy = xy[:, indices][:, :, slots]
        arrays[f"{name}_{mode}_z"] = latent
        arrays[f"{name}_{mode}_probe_xy"] = xy
        residual = latent.astype(np.float64) - truth
        scores[mode] = {
            "latent_mse": np.square(residual).mean(axis=-1),
            "centered_latent_mse": np.square(residual - residual.mean(axis=1, keepdims=True)).mean(axis=-1),
        }
    probe_xy = truth_probe[:, indices][:, :, slots]
    arrays[f"{name}_true_probe_xy"] = probe_xy
    scores["true_probe"] = {}
    for mode, metrics in scores.items():
        xy = arrays[f"{name}_{mode}_probe_xy"]
        residual = xy.astype(np.float64) - true_xy
        metrics.update(
            position_distance=np.linalg.norm(residual, axis=-1),
            centered_relative_displacement_error=np.linalg.norm(residual - residual.mean(axis=1, keepdims=True), axis=-1),
        )
    return scores


def summarize(output, arrays, panels):
    comparison_sets = {
        "step10": (("retained", "cleared"), ("retained", "native")),
        "step20": (("retained", "stale_memory"), ("retained", "cleared"), ("retained", "native")),
        "matched_targets": (("step20_retained", "step10_retained"), ("step20_stale_memory", "step10_retained"),
                            ("step20_retained", "step20_stale_memory"), ("step20_cleared", "step10_cleared"),
                            ("step20_native", "step10_native")),
    }
    rows, pairs, summary = [], [], {}
    for name, (targets, scores) in panels.items():
        root_scores = {}
        for mode, metrics in scores.items():
            root_scores[mode] = {}
            for metric, values in metrics.items():
                arrays[f"{name}_{mode}_{metric}"] = values
                means = values.mean(axis=1)
                root_scores[mode][metric] = means
                for root, root_means in zip(ROOT_IDS, means, strict=True):
                    for target, value in zip(targets, root_means, strict=True):
                        rows.append({"panel": name, "root_id": int(root), "target_step": int(target),
                                     "mode": mode, "metric": metric, "error": float(value)})
        summary[name] = {}
        for method, reference in comparison_sets[name]:
            comparison = summary[name][f"{method}_vs_{reference}"] = {}
            for metric, error in root_scores[method].items():
                gain = root_scores[reference][metric] - error
                for root, root_gain in zip(ROOT_IDS, gain, strict=True):
                    for target, value in zip(targets, root_gain, strict=True):
                        pairs.append({"panel": name, "root_id": int(root), "target_step": int(target),
                                      "method": method, "reference": reference, "metric": metric,
                                      "error_reduction": float(value)})
                report = []
                for target, values in zip([*targets.tolist(), "equal_target_mean"],
                                          [*gain.T, gain.mean(axis=1)], strict=True):
                    report.append({"target_step": target, "mean_error_reduction": float(values.mean()),
                                   "median_error_reduction": float(np.median(values)),
                                   "improved_roots": int(np.count_nonzero(values > 0))})
                comparison[metric] = report
    write_csv(output / "root_errors.csv", rows)
    write_csv(output / "root_pairs.csv", pairs)
    return summary


def run(args):
    data = load_recorded(args)
    checkpoints = {arm: args.checkpoints / arm / "final_selected_parameters.pt" for arm in CHECKPOINT_ARMS}
    if missing := [str(path) for path in checkpoints.values() if not path.is_file()]:
        raise FileNotFoundError(f"Missing fixed final checkpoints: {missing}")
    args.output.mkdir(parents=True, exist_ok=False)
    source, memory, source_timing = evaluate_arm(args, data, "query_feedback", checkpoints["query_feedback"])
    native, probes, native_timing = evaluate_arm(args, data, "native_finetune", checkpoints["native_finetune"])
    arrays = {"root_id": ROOT_IDS, "candidate_id": CANDIDATE_IDS, "candidate_name": data["candidate_name"],
              "carrier_candidate_id": np.array(CARRIER), "factual_step": np.arange(0, 21, 5),
              "factual_z": data["factual_z"], "factual_raw_actions": np.concatenate(
                  (data["prefix_raw"], data["raw"][:, CARRIER, :2]), axis=1),
              "factual_normalized_actions": np.concatenate((data["prefix_norm"], data["norm"][:, CARRIER, :2]), axis=1),
              "horizon": np.array(HORIZONS), **memory}
    panels, matched_scores = {}, {}
    for step, candidates in CANDIDATES.items():
        source[step].update(native[step])
        truth_probe = probes[f"step{step}_true_probe_xy"]
        targets = step + 5 * np.array(HORIZONS)
        name = f"step{step}"
        panels[name] = targets, panel(arrays, name, step, candidates, targets, truth_probe, data, source)
        matched = panel(arrays, f"matched_{name}", step, REFRESH_CANDIDATES, MATCHED_TARGETS,
                        truth_probe, data, source)
        matched_scores.update({f"{name}_{mode}": metrics for mode, metrics in matched.items()})
    for field in ("candidate_id", "target_step", "true_z", "true_xy"):
        np.testing.assert_array_equal(arrays[f"matched_step10_{field}"], arrays[f"matched_step20_{field}"])
    panels["matched_targets"] = MATCHED_TARGETS, matched_scores
    comparisons = summarize(args.output, arrays, panels)
    np.savez_compressed(args.output / "branches.npz", **arrays)
    write_json(args.output / "summary.json", {
        "branches": str(args.branches.resolve()), "forecasts": str(args.forecasts.resolve()),
        "checkpoints": {arm: str(path.resolve()) for arm, path in checkpoints.items()},
        "statistical_unit": "32 previously studied assessment roots",
        "physical_probe": str(args.probe.resolve()), "timing": dict(query_feedback=source_timing, native_finetune=native_timing),
        "paired_error_reductions": comparisons, "completed": True,
    })
    print(f"Saved fixed branch and refresh replay to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("research/phase00c/probe_heads.pt"))
    parser.add_argument("--branches", type=Path, default=Path("runs/p1a_v1/assessment/branches.npz"))
    parser.add_argument("--forecasts", type=Path, default=Path("runs/p1a_feedback/forecasts.npz"))
    parser.add_argument("--checkpoints", type=Path, default=Path("runs/query_feedback_20261112/learning_source"))
    parser.add_argument("--output", type=Path, default=Path("runs/query_feedback_20261112/intermediates/source_branches"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        run(args)


if __name__ == "__main__":
    main()
