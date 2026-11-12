"""Fixed offline teaching of the already specified native query-feedback path.

This is a development study on previously used P0b episodes, not a final task
evaluation. The protocol fixes two arms, one schedule and the final epoch.
"""

import argparse
import csv
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from history_match import load_host
from native_response import FeedbackMemory, NativeResponseRegulator


SEED = 1112
EPOCHS = 60
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
PREFIX = 6
HORIZONS = (1, 2, 4, 8)
SUFFIX = 8
ARMS = ("query_feedback", "native_finetune")
SPLITS = {"train": (0, 30), "development": (30, 50)}
MODES = ("full", "cleared_at_query", "shuffled_at_query")
SELECTED_PARAMETER_COUNTS = {"canonical": 393408, "source": 294912}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def assert_eval(model):
    if any(module.training for module in model.modules()):
        raise RuntimeError("Every host module must remain in evaluation mode")


def counters(operator):
    return {"predictor_calls": operator.calls,
            "feedback_feature_calls": operator.feedback_feature_calls,
            "source_capture_calls": operator.source_capture_calls}


def counter_difference(operator, before):
    return {key: value - before[key] for key, value in counters(operator).items()}


def select_parameters(model, operator):
    model.requires_grad_(False)
    if operator.feedback == "source":
        selected_ids = {id(operator.attention.to_qkv.weight)}
    else:
        modules = (operator.attention.to_qkv, operator.attention.to_out)
        selected_ids = {id(parameter) for module in modules for parameter in module.parameters()}
    selected = {name: parameter for name, parameter in model.named_parameters()
                if id(parameter) in selected_ids}
    if sum(parameter.numel() for parameter in selected.values()) != SELECTED_PARAMETER_COUNTS[operator.feedback]:
        raise RuntimeError(f"Unexpected first-attention parameter count for {operator.feedback} feedback")
    for parameter in selected.values():
        parameter.requires_grad_(True)
    return selected


def selected_state(selected):
    return {name: value.detach().cpu().clone() for name, value in selected.items()}


def build_prefix(operator, z, actions, feedback):
    """Six factual transitions; this is the only function that writes memory."""
    memory = operator.empty_memory(len(z))
    retain_source = feedback and operator.feedback == "source"
    predictions = []
    for t in range(PREFIX):
        begin = max(0, t - 2)
        issued = operator.predict(z[:, begin:t + 1], actions[:, begin:t + 1], memory,
                                  retain_source=retain_source)
        if (issued.source is not None) != retain_source:
            raise RuntimeError("Source attention must be retained exactly when issuing factual source feedback")
        predictions.append(issued.latent)
        if feedback:
            # z_(t+1) arrives only after issuing the prediction for action a_t.
            memory = operator.observe(memory, issued, z[:, t + 1])
        expected_size = t + 1 if feedback else 0
        if memory.size != expected_size:
            raise RuntimeError("Factual memory length differs from the completed transitions")
    return memory, torch.stack(predictions, dim=1)


def free_suffix(operator, initial_history, actions, memory):
    """Eight imagined transitions; no future observation or write is available."""
    source_captures = operator.source_capture_calls
    feedback_features = operator.feedback_feature_calls
    records = memory.size
    history = initial_history
    predictions = []
    for offset in range(SUFFIX):
        t = PREFIX + offset
        # The token at physical time 5*j is conditioned on outgoing block a_j.
        issued = operator.predict(history, actions[:, t - 2:t + 1], memory, retain_source=False)
        if issued.source is not None:
            raise RuntimeError("An imagined suffix prediction retained source attention")
        predictions.append(issued.latent)
        history = torch.cat((history[:, 1:], issued.latent[:, None]), dim=1)
    if (operator.source_capture_calls != source_captures
            or operator.feedback_feature_calls != feedback_features or memory.size != records):
        raise RuntimeError("The imagined suffix must only read the existing memory")
    return torch.stack(predictions, dim=1)


def objective(prefix_prediction, suffix_prediction, z):
    prefix_error = (prefix_prediction - z[:, 1:PREFIX + 1].detach()).square().mean(-1)
    slots = torch.tensor(HORIZONS, device=z.device) - 1
    future_error = (suffix_prediction[:, slots]
                    - z[:, PREFIX + 1 + slots].detach()).square().mean(-1)
    # Six prefix terms and four suffix terms each receive weight 1/10.
    loss = torch.cat((prefix_error, future_error), dim=1).mean()
    return loss, prefix_error, future_error


@torch.no_grad()
def evaluate(operator, z, actions, arm, snapshot, output):
    assert_eval(operator.model)
    before = counters(operator)
    torch.cuda.synchronize()
    start = time.perf_counter()
    uses_feedback = arm == "query_feedback"
    archive_memory = uses_feedback and operator.feedback == "source"
    modes = MODES if uses_feedback else ("full",)
    arrays = {"episode_id": np.arange(50), "horizons": np.asarray(HORIZONS),
              "target_environment_step": 5 * (PREFIX + np.asarray(HORIZONS)),
              "prefix_targets": z[:, 1:7].cpu().numpy(),
              "suffix_targets": z[:, 7:15].cpu().numpy(),
              "prefix_predictions": np.empty((50, 6, 192), dtype=np.float32),
              "prefix_mse": np.empty((50, 6), dtype=np.float32)}
    for mode in modes:
        arrays[mode + "_suffix_predictions"] = np.empty((50, 8, 192), dtype=np.float32)
        arrays[mode + "_horizon_mse"] = np.empty((50, 4), dtype=np.float32)
    if uses_feedback:
        arrays["shuffled_memory_source_episode"] = np.empty(50, dtype=np.int64)
    if archive_memory:
        memory_shape = (len(z), operator.heads, PREFIX, operator.head_width)
        arrays["prefix_memory_queries"] = np.empty(memory_shape, dtype=np.float32)
        arrays["prefix_memory_values"] = np.empty(memory_shape, dtype=np.float32)
    horizon_rows, prefix_rows, summaries, scores = [], [], {}, {}
    for split, (begin, end) in SPLITS.items():
        actual, conditions = z[begin:end], actions[begin:end]
        memory, prefix_prediction = build_prefix(operator, actual, conditions, uses_feedback)
        if archive_memory:
            arrays["prefix_memory_queries"][begin:end] = memory.queries.cpu().numpy()
            arrays["prefix_memory_values"][begin:end] = memory.values.cpu().numpy()
        prefix_error = (prefix_prediction - actual[:, 1:7]).square().mean(-1).cpu().numpy()
        arrays["prefix_predictions"][begin:end] = prefix_prediction.cpu().numpy()
        arrays["prefix_mse"][begin:end] = prefix_error
        ids = np.arange(begin, end)
        summaries[split], scores[split] = {}, {}
        for row, episode in enumerate(ids):
            for t in range(PREFIX):
                prefix_rows.append({"arm": arm, "snapshot": snapshot, "split": split,
                                    "episode_id": int(episode), "transition": t,
                                    "actual_successor_environment_step": 5 * (t + 1),
                                    "mse": float(prefix_error[row, t])})
        for mode in modes:
            queried = memory
            if mode == "cleared_at_query":
                queried = operator.empty_memory(len(actual))
            elif mode == "shuffled_at_query":
                # All episodes of this entire split move together, not minibatches.
                queried = FeedbackMemory(memory.queries.roll(1, 0), memory.values.roll(1, 0))
                arrays["shuffled_memory_source_episode"][begin:end] = np.roll(ids, 1)
            suffix_prediction = free_suffix(operator, actual[:, 4:7], conditions, queried)
            _, _, future_error = objective(prefix_prediction, suffix_prediction, actual)
            future = future_error.cpu().numpy()
            arrays[mode + "_suffix_predictions"][begin:end] = suffix_prediction.cpu().numpy()
            arrays[mode + "_horizon_mse"][begin:end] = future
            scores[split][mode] = future
            combined = np.concatenate((prefix_error, future), axis=1).mean(axis=1)
            summaries[split][mode] = {
                "episodes": len(ids), "prefix_six_term_mean": float(prefix_error.mean()),
                "suffix_four_term_mean": float(future.mean()),
                "ten_term_objective": float(combined.mean()),
                "horizon_mean_mse": future.mean(axis=0).tolist(),
                "horizon_median_episode_mse": np.median(future, axis=0).tolist(),
            }
            for row, episode in enumerate(ids):
                for h, horizon in enumerate(HORIZONS):
                    horizon_rows.append({"arm": arm, "snapshot": snapshot, "split": split,
                                         "mode": mode, "episode_id": int(episode),
                                         "horizon": horizon, "target_environment_step": 5 * (PREFIX + horizon),
                                         "mse": float(future[row, h])})
    torch.cuda.synchronize()
    np.savez_compressed(output / f"{snapshot}_outputs.npz", **arrays)
    write_csv(output / f"{snapshot}_episode_horizons.csv", horizon_rows)
    write_csv(output / f"{snapshot}_episode_prefix.csv", prefix_rows)
    summary = {"feedback": operator.feedback, "scores": summaries,
               "compute": counter_difference(operator, before),
               "elapsed_seconds_including_output_write": time.perf_counter() - start}
    write_json(output / f"{snapshot}_summary.json", summary)
    return scores, summary


def train(operator, z, actions, selected, batch_plan, arm, output):
    # A new Adam is constructed for each independent arm.
    optimizer = torch.optim.Adam(list(selected.values()), lr=LEARNING_RATE, weight_decay=0.)
    before = counters(operator)
    torch.cuda.synchronize()
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    allocation_before = torch.cuda.memory_allocated()
    epochs, step = [], 0
    with (output / "training_steps.jsonl").open("w", encoding="utf-8") as log:
        for epoch, order in enumerate(batch_plan, 1):
            epoch_sums = np.zeros(3, dtype=np.float64)
            for begin in range(0, len(order), BATCH_SIZE):
                assert_eval(operator.model)
                indices = torch.as_tensor(order[begin:begin + BATCH_SIZE], device=z.device)
                actual, conditions = z[indices], actions[indices]
                optimizer.zero_grad(set_to_none=True)
                # Rebuild from empty under the current parameters at every step.
                memory, prefix_prediction = build_prefix(operator, actual, conditions, arm == "query_feedback")
                suffix_prediction = free_suffix(operator, actual[:, 4:7], conditions, memory)
                loss, prefix_error, future_error = objective(prefix_prediction, suffix_prediction, actual)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"Nonfinite loss at {arm}, epoch {epoch}, update {step + 1}")
                loss.backward()
                norm_square = 0.
                for name, parameter in selected.items():
                    if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                        raise FloatingPointError(f"Invalid selected gradient: {arm}, {name}, update {step + 1}")
                    norm_square += float(parameter.grad.square().sum())
                if any(p.grad is not None for p in operator.model.parameters() if not p.requires_grad):
                    raise RuntimeError("A frozen host parameter received a gradient")
                values = (float(loss.detach()), float(prefix_error.detach().mean()),
                          float(future_error.detach().mean()))
                optimizer.step()
                step += 1
                record = {"epoch": epoch, "optimizer_step": step,
                          "episode_ids": order[begin:begin + BATCH_SIZE].tolist(),
                          "ten_term_loss": values[0], "prefix_six_term_mean": values[1],
                          "suffix_four_term_mean": values[2], "selected_gradient_l2": norm_square ** .5}
                log.write(json.dumps(record) + "\n")
                log.flush()
                epoch_sums += len(indices) * np.asarray(values)
                # Neither records nor their computation graph cross an update.
                del memory, prefix_prediction, suffix_prediction, loss, prefix_error, future_error
            mean = epoch_sums / 30
            epochs.append({"epoch": epoch, "optimizer_steps": step,
                           "ten_term_loss": float(mean[0]), "prefix_six_term_mean": float(mean[1]),
                           "suffix_four_term_mean": float(mean[2])})
            if epoch % 10 == 0:
                print(f"{arm}: epoch {epoch}/{EPOCHS}, updates {step}, train loss {mean[0]:.7f}", flush=True)
    torch.cuda.synchronize()
    if step != 240:
        raise RuntimeError("Each arm must complete the predefined 240 updates")
    operator.model.zero_grad(set_to_none=True)
    summary = {"feedback": operator.feedback, "optimizer_steps": step, "epochs": EPOCHS,
               "elapsed_seconds": time.perf_counter() - start,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "allocation_before_training_bytes": allocation_before,
               "compute": counter_difference(operator, before),
               "sample_prediction_calls": EPOCHS * 30 * (PREFIX + SUFFIX),
               "scope": "Training only, excluding initial/final evaluation; includes Adam updates and step logging."}
    write_csv(output / "training_epochs.csv", epochs)
    write_json(output / "training_summary.json", summary)
    return summary


def parameter_changes(initial, final):
    output = {}
    for name in initial:
        delta = final[name] - initial[name]
        output[name] = {"coordinates": delta.numel(), "difference_l2": float(delta.norm()),
                        "max_absolute_difference": float(delta.abs().max()),
                        "changed_coordinates": int(torch.count_nonzero(delta)),
                        "initial_l2": float(initial[name].norm()), "final_l2": float(final[name].norm())}
    return output


def compare_completed(scores, output):
    pairs = {
        "feedback_full_vs_native_finetune": (("query_feedback", "final", "full"), ("native_finetune", "final", "full")),
        "feedback_full_vs_same_parameters_cleared": (("query_feedback", "final", "full"), ("query_feedback", "final", "cleared_at_query")),
        "feedback_full_vs_same_parameters_shuffled": (("query_feedback", "final", "full"), ("query_feedback", "final", "shuffled_at_query")),
        "feedback_final_vs_initial": (("query_feedback", "final", "full"), ("query_feedback", "initial", "full")),
        "native_finetune_final_vs_initial": (("native_finetune", "final", "full"), ("native_finetune", "initial", "full")),
    }
    summaries, rows = {}, []
    for name, (method, reference) in pairs.items():
        summaries[name] = {}
        for split, (begin, end) in SPLITS.items():
            actual = scores[method[0]][method[1]][split][method[2]]
            base = scores[reference[0]][reference[1]][split][reference[2]]
            differences = base.astype(np.float64) - actual.astype(np.float64)
            summaries[name][split] = []
            for slot, horizon in enumerate(HORIZONS):
                delta = differences[:, slot]
                summaries[name][split].append({
                    "horizon": horizon, "episodes": len(delta),
                    "method_mean_mse": float(actual[:, slot].mean()),
                    "reference_mean_mse": float(base[:, slot].mean()),
                    "mean_improvement": float(delta.mean()), "median_improvement": float(np.median(delta)),
                    "minimum_improvement": float(delta.min()), "maximum_improvement": float(delta.max()),
                    "improved_episodes": int((delta > 0).sum()), "worsened_episodes": int((delta < 0).sum()),
                })
                for row, episode in enumerate(range(begin, end)):
                    rows.append({"comparison": name, "split": split, "episode_id": episode,
                                 "horizon": horizon, "method_mse": float(actual[row, slot]),
                                 "reference_mse": float(base[row, slot]), "improvement": float(delta[row])})
    write_csv(output / "episode_comparisons.csv", rows)
    write_json(output / "comparisons.json", summaries)
    return summaries


def execute(args, *, feedback="canonical"):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    with np.load(args.inputs, allow_pickle=False) as source:
        ids = source["episode_id"].copy()
        times = source["observation_env_step"].copy()
        observed = source["z_observed"][:, :15].copy()  # z_15 is deliberately unused.
        raw_actions = source["action_sequence"][:, :14].copy()
    if ids.tolist() != list(range(50)) or times.tolist() != list(range(0, 76, 5)):
        raise ValueError("Expected the original 50 ordered P0b episodes and five-step grid")
    rng = np.random.default_rng(SEED)
    batch_plan = np.stack([rng.permutation(30) for _ in range(EPOCHS)])
    parameter_count = SELECTED_PARAMETER_COUNTS[feedback]
    parameter_scope = "first attention QKV weight" if feedback == "source" else "first attention QKV/O"
    write_json(args.output / "protocol.json", {
        "source": str(args.inputs.resolve()), "model_dir": str(args.model_dir.resolve()),
        "feedback": feedback, "seed": SEED, "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "optimizer": "Adam",
        "weight_decay": 0., "optimizer_steps_per_arm": 240, "capacity": 64,
        "prefix_transitions": PREFIX, "suffix_steps": SUFFIX, "horizons": list(HORIZONS),
        "training_episode_ids": list(range(30)), "development_episode_ids": list(range(30, 50)),
        "shared_epoch_episode_order": batch_plan.tolist(), "encoder_calls": 0,
        "host_mode": "eval", "selected_parameters": parameter_scope,
        "selected_parameter_count": parameter_count,
        "checkpoint_selection": "Initial and fixed final epoch 60 only; no development-based selection.",
        "statistical_unit": "Episode; all episodes were previously used in project development.",
        "comparison_scope": "Same parameters, labels, batches and update count; feedback has additional compute.",
    })
    shared_initial, scores, arm_reports = None, {}, {}
    for arm in ARMS:
        output = args.output / arm
        output.mkdir()
        # No trained host, memory or optimizer is reused across arms.
        host = load_host(args)
        model = host.model
        assert_eval(model)
        z = torch.as_tensor(observed, device=host.device)
        raw = torch.as_tensor(host.normalize(raw_actions), device=host.device, dtype=z.dtype)
        with torch.no_grad():
            actions = model.action_embed(model.action_encoder(raw))
            neutral_actions = torch.as_tensor(host.normalize(np.zeros((1, 1, 5, 2), dtype=np.float32)),
                                              device=host.device, dtype=z.dtype)
            neutral = model.action_embed(model.action_encoder(neutral_actions))[0, 0]
        operator = NativeResponseRegulator(model, neutral, feedback=feedback)
        selected = select_parameters(model, operator)
        initial = selected_state(selected)
        if shared_initial is None:
            shared_initial = initial
            torch.save({"selected_parameters": shared_initial, "epoch": 0, "feedback": feedback},
                       args.output / "initial_selected_parameters.pt")
        elif set(initial) != set(shared_initial) or any(not torch.equal(initial[k], shared_initial[k]) for k in initial):
            raise RuntimeError("The two arms did not start from the same selected host parameters")
        scores[arm] = {}
        scores[arm]["initial"], initial_report = evaluate(operator, z, actions, arm, "initial", output)
        training = train(operator, z, actions, selected, batch_plan, arm, output)
        scores[arm]["final"], final_report = evaluate(operator, z, actions, arm, "final", output)
        final = selected_state(selected)
        torch.save({"selected_parameters": final, "epoch": EPOCHS, "arm": arm,
                    "optimizer_steps": 240, "feedback": feedback}, output / "final_selected_parameters.pt")
        changes = parameter_changes(initial, final)
        write_json(output / "parameter_changes.json", changes)
        if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
            raise RuntimeError("Temporary hooks remained after the completed arm")
        arm_reports[arm] = {"initial": initial_report, "final": final_report,
                            "training": training, "parameter_changes": changes,
                            "final_hooks_empty": True}
        del selected, operator, model, host, z, raw, actions, neutral, neutral_actions
        gc.collect()
        torch.cuda.empty_cache()
    comparisons = compare_completed(scores, args.output)
    write_json(args.output / "summary.json", {"completed": True, "feedback": feedback, "arms": arm_reports,
                                               "comparisons": comparisons})
    parameter_description = ("首 attention QKV 的原始权重，W_O 冻结" if feedback == "source"
                             else "首 attention QKV/O")
    lines = ["# Query feedback：固定离线学习阶段", "",
             f"`{feedback}` 模式下，两臂均完成固定 60 epoch / 240 次 Adam 更新。训练参数为{parameter_description}，共 {parameter_count:,} 个原生参数，模型始终 eval。前缀 6 项和后缀 4 项等权。开发数据仅用于初始和固定最终快照的评价。", "",
             "数据为此前使用过的 50 条 P0b 专家轨迹：0–29 拟合，30–49 为本次拟合未使用的开发留出；它们不是新测试集。统计单位为 episode。", "",
             "| 开发对照（正值为 full 改善） | H | 平均 MSE 改善 | 中位数 | 改善 episode / 20 |",
             "|---|---:|---:|---:|---:|"]
    for name in ("feedback_full_vs_native_finetune", "feedback_full_vs_same_parameters_cleared", "feedback_full_vs_same_parameters_shuffled"):
        for row in comparisons[name]["development"]:
            lines.append(f"| {name} | {row['horizon']} | {row['mean_improvement']:+.8f} | {row['median_improvement']:+.8f} | {row['improved_episodes']} / 20 |")
    lines += ["", "| 训练臂 | 预测调用 | 反馈特征调用 | 源状态保留 | 训练耗时（秒） |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        training = arm_reports[arm]["training"]
        calls = training["compute"]
        lines.append(f"| {arm} | {calls['predictor_calls']} | {calls['feedback_feature_calls']} | "
                     f"{calls['source_capture_calls']} | {training['elapsed_seconds']:.3f} |")
    lines += ["", "full 与同参数 cleared 的差异说明已训练模型对当前记忆的依赖；full 与 native_finetune 的差异比较整条训练和状态路径。shuffled 只说明经历对应关系的敏感性，错误状态造成损害本身不证明正常状态有益。这些对照均不能证明 query 地址或 r+delta 写入优于一般记忆方法。", "",
              "本阶段没有同起点的未见动作分支、额外真实刷新后的续算或最终任务评价，不能据此声称跨动作迁移、候选排序、长期稳定性或任务成功。", "",
              "逐 episode 原始预测、六项前缀误差、各 horizon 误差及训练曲线在两个 arm 子目录；初末选定参数和参数差独立保存。完整分布见 comparisons.json 与 episode_comparisons.csv。上表记录实际调用次数与训练耗时。反馈臂的预测调用还包含记忆读取。", ""]
    if feedback == "source":
        lines += ["反馈臂的初末 outputs.npz 同时保存每个 episode 在前缀结束后的六条 query/value 记录。", ""]
    (args.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"completed": True, "output": str(args.output), "updates_per_arm": 240}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("research/phase00c/probe_heads.pt"))
    parser.add_argument("--inputs", type=Path, default=Path("research/phase00b/predictions.npz"))
    parser.add_argument("--feedback", choices=SELECTED_PARAMETER_COUNTS, default="canonical")
    parser.add_argument("--output", type=Path, help="New output directory (defaults to learning or learning_source)")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("runs/query_feedback_20261112") / (
            "learning_source" if args.feedback == "source" else "learning")
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            execute(args, feedback=args.feedback)
    except Exception as error:
        write_json(args.output / "failure.json", {"completed": False, "feedback": args.feedback,
                                                   "exception": type(error).__name__,
                                                   "message": str(error), "configuration_changed": False})
        raise


if __name__ == "__main__":
    main()
