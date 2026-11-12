"""Measure native predictor and source feedback costs on the fixed factual request."""

import argparse
import csv
from functools import partial
from pathlib import Path
import time

import numpy as np
import torch

from history_match import load_host
from native_response import FeedbackMemory, NativeResponseRegulator
from query_feedback_intermediates import write_json
from query_feedback_source_persistence import exact, load_prefix, memory_signature


SOURCE = 5
WINDOW = slice(3, 6)
BATCHES = (1, 50)
WARMUP = 10
REPEATS = 30
OPERATIONS = (
    "native_predict", "source_empty_read", "source_M5_read",
    "source_M5_issue", "source_observe", "canonical_observe",
)
CONTRASTS = {
    "empty_wrapper_minus_native": (("source_empty_read",), "native_predict"),
    "M5_read_minus_empty": (("source_M5_read",), "source_empty_read"),
    "source_retention": (("source_M5_issue",), "source_M5_read"),
    "source_observe_minus_canonical": (("source_observe",), "canonical_observe"),
    "source_factual_issue_plus_observe_minus_native": (
        ("source_M5_issue", "source_observe"), "native_predict",
    ),
}


def cycle(operations, round_index):
    start = round_index % len(operations)
    return operations[start:] + operations[:start]


def measure(operations, batch_size, device):
    for index in range(WARMUP):
        for _, fn in cycle(operations, index):
            fn()
        torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    end.record()
    torch.cuda.synchronize(device)
    rows = []
    for index in range(REPEATS):
        for position, (name, fn) in enumerate(cycle(operations, index)):
            t0 = time.perf_counter()
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize(device)
            wall_ms = (time.perf_counter() - t0) * 1000
            rows.append({"batch_size": batch_size, "repeat": index + 1, "position": position + 1,
                         "operation": name, "cuda_ms": start.elapsed_time(end), "wall_ms": wall_ms})
            del result
    return rows


def stats(values):
    p10, median, p90 = np.quantile(values, (.1, .5, .9))
    return {"median": float(median), "p10": float(p10), "p90": float(p90)}


def summarize(rows):
    metrics = ("cuda_ms", "wall_ms")
    by_op = {name: [row for row in rows if row["operation"] == name] for name in OPERATIONS}
    columns = {name: {metric: np.array([row[metric] for row in records]) for metric in metrics}
               for name, records in by_op.items()}
    for values in columns.values():
        if any(len(series) != REPEATS or not np.isfinite(series).all() or (series <= 0).any()
               for series in values.values()):
            raise RuntimeError("Expected 30 positive finite timings for each operation and clock")
    return {
        "operations": {name: {metric: stats(values) for metric, values in times.items()}
                       for name, times in columns.items()},
        "paired_round_differences": {
            name: {metric: stats(sum(columns[op][metric] for op in left) - columns[right][metric])
                   for metric in metrics}
            for name, (left, right) in CONTRASTS.items()
        },
    }


def payload_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def timing_range(stat):
    return f"{stat['median']:.4f} [{stat['p10']:.4f}, {stat['p90']:.4f}]"


def verify_request(native, empty_read, read, issued, canonical_issue, source, memory, actual, archive):
    exact(native, empty_read.latent)
    exact(empty_read.readout, torch.zeros_like(empty_read.readout))
    for name in ("query", "readout", "latent"):
        exact(getattr(read, name), getattr(issued, name))
        exact(getattr(canonical_issue, name), getattr(issued, name))
    if (read.source is not None or empty_read.source is not None or canonical_issue.source is not None
            or issued.source is None or issued.source.slot != 2):
        raise RuntimeError("Only factual source issuance must retain the last slot")
    result = source.observe(memory, issued, actual)
    replay = source.observe(memory, issued, actual)
    if result is replay or result is memory or result.size != 6 or memory.size != 5:
        raise RuntimeError("Factual observation must return a fresh memory with one new record")
    exact(result.queries, replay.queries)
    exact(result.values, replay.values)
    exact(result.queries[:, :, :SOURCE], memory.queries)
    exact(result.values[:, :, :SOURCE], memory.values)
    checks = {}
    for name, value in (("query", issued.query), ("readout", issued.readout), ("latent", issued.latent),
                        ("M6_queries", result.queries), ("M6_values", result.values)):
        torch.testing.assert_close(value, archive[name], rtol=1e-5, atol=1e-5)
        checks[name] = (value - archive[name]).abs().max().item()
    return checks


def benchmark_batch(host, source, canonical, z, actions, batch_size, archive):
    history = z[:batch_size, WINDOW]
    conditions = actions[:batch_size, WINDOW]
    actual = z[:batch_size, SOURCE + 1]
    archive = {name: tensor[:batch_size] for name, tensor in archive.items()}
    memory = FeedbackMemory(archive["M6_queries"][:, :, :SOURCE].contiguous(),
                            archive["M6_values"][:, :, :SOURCE].contiguous())
    empty = source.empty_memory(batch_size)
    saved_queries, saved_values = memory.queries.clone(), memory.values.clone()
    signature = memory_signature(memory)

    def native_predict():
        return host.model.predict(history, conditions)[:, -1]

    empty_read = partial(source.predict, history, conditions, empty, retain_source=False)
    memory_read = partial(source.predict, history, conditions, memory, retain_source=False)
    source_issue = partial(source.predict, history, conditions, memory, retain_source=True)
    issued = source_issue()
    canonical_issue = canonical.predict(history, conditions, memory, retain_source=False)
    checks = verify_request(native_predict(), empty_read(), memory_read(), issued, canonical_issue,
                            source, memory, actual, archive)

    operations = tuple(zip(OPERATIONS, (
        native_predict, empty_read, memory_read, source_issue,
        partial(source.observe, memory, issued, actual),
        partial(canonical.observe, memory, canonical_issue, actual),
    ), strict=True))
    before = (source.calls, source.source_capture_calls, source.feedback_feature_calls,
              canonical.calls, canonical.feedback_feature_calls)
    rows = measure(operations, batch_size, z.device)
    after = (source.calls, source.source_capture_calls, source.feedback_feature_calls,
             canonical.calls, canonical.feedback_feature_calls)
    count = WARMUP + REPEATS
    if tuple(a - b for a, b in zip(after, before)) != (3 * count, count, 2 * count, 0, 2 * count):
        raise RuntimeError("Unexpected predictor, capture, or consequence count in benchmark")
    if memory_signature(memory) != signature or empty.size != 0:
        raise RuntimeError("Benchmark altered the input state")
    exact(memory.queries, saved_queries)
    exact(memory.values, saved_values)

    source_bytes = {name: payload_bytes(getattr(issued.source, name))
                    for name in ("output", "log_partition", "shift", "scale")}
    common_bytes = {name: payload_bytes(getattr(issued, name)) for name in ("latent", "query", "readout")}
    summary = summarize(rows)
    summary.update(
        archive_replay_max_abs=checks,
        retain_on_off_query_readout_latent_exact=True,
        native_empty_latent_exact=True,
        input_state_preserved=True,
        M5_payload_bytes=payload_bytes(memory.queries) + payload_bytes(memory.values),
        pending_prediction_payload_bytes=common_bytes,
        extra_pending_source_payload_bytes=source_bytes,
        extra_pending_source_total_bytes=sum(source_bytes.values()),
    )
    return rows, summary


def write_report(path, summary):
    lines = [
        "# 源反馈的原生执行成本",
        "",
        f"设备为 {summary['device']}。原始冻结宿主使用 FP32 与 MATH SDPA。固定环境步 25 的真实请求，"
        "使用 z3 至 z5 与第 3 至第 5 个动作块，兑现使用环境步 30 的实际 z6。M5 从持久性实验的源反馈 M6 "
        "取前五条记录。B=1 使用 episode 0，B=50 使用全部经历。",
        "",
        "每项先预热 10 次，再按固定循环顺序独立测量 30 次。每次调用分别记录 CUDA event 间隔和同步墙钟。"
        "兑现重复使用计时前发出的请求，并在每次调用后丢弃新返回的 M6。源请求保留开关前后的 query、"
        "readout 和预测 latent 逐元素一致，空态预测与原生预测逐元素一致。观察编码、动作嵌入和数据传输在计时之外。",
        "",
        "| B | 操作 | CUDA ms 中位数 [p10, p90] | 同步墙钟 ms 中位数 [p10, p90] |",
        "|---:|---|---:|---:|",
    ]
    for batch_size, batch in summary["batches"].items():
        for name, result in batch["operations"].items():
            ranges = [timing_range(stat) for stat in result.values()]
            lines.append(f"| {batch_size} | `{name}` | {' | '.join(ranges)} |")
    lines.extend((
        "",
        "保留源时额外计算最后一个 query 对历史 key 的 log-partition，并保留原生头输出、log-partition "
        "与请求的 shift、scale。兑现计算实际与预测后继的两组 K/V，随后追加一条记录。"
        "Canonical 对照使用同一 M5 上预发请求的输出，两次计算中性条件下的后继 value。",
        "",
        "| B | 同轮配对差值 | CUDA ms 中位数 [p10, p90] | 同步墙钟 ms 中位数 [p10, p90] |",
        "|---:|---|---:|---:|",
    ))
    for batch_size, batch in summary["batches"].items():
        for name, label in (("source_retention", "M5 发出并保留源减去 M5 只读"),
                            ("source_factual_issue_plus_observe_minus_native", "源发出与兑现之和减去原生预测")):
            paired = batch["paired_round_differences"][name]
            lines.append(f"| {batch_size} | {label} | {timing_range(paired['cuda_ms'])} | "
                         f"{timing_range(paired['wall_ms'])} |")
    payloads = summary["batches"]
    lines.extend((
        "",
        f"从发出到兑现，源专用待处理张量为每经历 {payloads[1]['extra_pending_source_total_bytes']:,} 字节，"
        f"B=50 合计 {payloads[50]['extra_pending_source_total_bytes']:,} 字节。"
        "字节数按有效元素计算。源请求还紧凑复制了共同需要的 query。实测间隔反映当前实现的 hooks、"
        "张量复制和调度，CUDA event 间隔可含主机发射间隙，墙钟包含计时调用与逐次同步开销。"
        "这些数值对应这台设备上的固定三 token 请求和五记录状态。",
        "",
        "[原始样本](samples.csv) · [配对差值和张量字节数](summary.json) · "
        "[复现脚本](../../../../research/p1a/query_feedback_source_cost.py)",
    ))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def run(args):
    host = load_host(args)
    model = host.model
    if any(module.training for module in model.modules()) or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("Expected the original frozen eval host")
    ids, steps, observed, raw = load_prefix(args)
    z = torch.as_tensor(observed, device=host.device)

    def embed(raw_actions):
        normalized = torch.as_tensor(host.normalize(raw_actions), device=z.device, dtype=z.dtype)
        return model.action_embed(model.action_encoder(normalized))

    actions = embed(raw)
    neutral = embed(np.zeros((1, 1, 5, 2), np.float32))[0, 0]
    with np.load(args.persistence, allow_pickle=False) as saved:
        for name, original in (("episode_id", ids), ("observation_environment_steps", steps),
                               ("z_observed", observed), ("factual_raw_actions", raw)):
            if not np.array_equal(saved[name], original):
                raise ValueError(f"Source persistence and P0b disagree on {name}")
        archive = {name: torch.as_tensor(saved["formation_" + name][:, SOURCE], device=z.device)
                   for name in ("query", "readout", "latent")}
        archive.update({name: torch.as_tensor(saved[name], device=z.device)
                        for name in ("M6_queries", "M6_values")})

    source = NativeResponseRegulator(model, neutral, feedback="source")
    canonical = NativeResponseRegulator(model, neutral, feedback="canonical")
    rows, batches = [], {}
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for batch_size in BATCHES:
            samples, batches[batch_size] = benchmark_batch(host, source, canonical, z, actions, batch_size, archive)
            rows.extend(samples)
    if any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise RuntimeError("A temporary model hook remained attached")

    summary = {
        "checkpoint": str((args.model_dir / "fourroom_lewm_object.ckpt").resolve()),
        "normalizer": str((args.model_dir / "normalizer.pt").resolve()),
        "P0b_inputs": str(args.inputs.resolve()), "source_persistence": str(args.persistence.resolve()),
        "device": torch.cuda.get_device_name(z.device), "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "dtype": "float32", "attention_backend": "MATH", "TF32": False,
        "source_window_environment_steps": steps[WINDOW].tolist(),
        "source_factual_action_blocks": list(range(WINDOW.start, WINDOW.stop)),
        "actual_observation_environment_step": int(steps[SOURCE + 1]),
        "episode_ids_by_batch": {size: ids[:size].tolist() for size in BATCHES},
        "warmup_per_operation": WARMUP, "measured_repeats_per_operation": REPEATS,
        "cycle_base_order": OPERATIONS, "cycle_rule": "Left rotate by zero-based round index modulo six",
        "timings": {
            "cuda_ms": "Elapsed time between CUDA events on the current stream",
            "wall_ms": "perf_counter around both event records, the call and device synchronization",
            "paired_round_differences": "Subtractions or sum-minus-baseline within each fixed cyclic round",
        },
        "batches": batches,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(args.output / "summary.json", summary)
    write_report(args.output / "REPORT.md", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--probe", type=Path, default=Path("research/phase00c/probe_heads.pt"))
    parser.add_argument("--inputs", type=Path, default=Path("research/phase00b/predictions.npz"))
    base = Path("runs/query_feedback_20261112/intermediates")
    parser.add_argument("--persistence", type=Path, default=base / "source_persistence" / "variables.npz")
    parser.add_argument("--output", type=Path, default=base / "source_cost")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    run(args)


if __name__ == "__main__":
    main()
