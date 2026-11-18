"""Launch the shipped FourRoom flat evaluator on a verified release input."""

import argparse
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--ranges-ready", type=Path)
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--probe-index", type=int)
    parser.add_argument("--gpu-memory-fraction", type=float)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    assert 1 <= args.num_eval <= 50
    assert args.tasks.is_file()
    assert args.checkpoint.is_file() and args.checkpoint.with_name("normalizer.pt").is_file()
    if args.ranges_ready is not None:
        required = range(args.num_eval + 1)
        if args.probe_index is not None:
            required = (args.probe_index, args.probe_index + 1)
        for index in required:
            assert (args.ranges_ready / f"{index:03}.txt").is_file(), f"Release range {index} is missing"

    os.environ["HJEPA_HOME"] = str(args.checkpoint.parent)
    sys.path[:0] = [str(args.source / "h_jepa"), str(args.source)]

    import torch
    from planning_eval import load_eval_config, run_planning_eval

    torch.set_num_threads(2)
    if args.gpu_memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    cfg = load_eval_config(config_name="fourroom_flat")
    cfg.policy = str(args.checkpoint.resolve())
    cfg.load_eval_trajs_path = str(args.tasks.resolve())
    cfg.eval.num_eval = args.num_eval
    results = args.output.resolve()
    if args.probe_index is not None:
        assert 0 <= args.probe_index < 50
        cfg.eval.num_eval = 1
        cfg.seed += args.probe_index
        results = results / f"probe_{args.probe_index:03}"
    # The official evaluator's default is one resumable rollout per chunk. mmap
    # leaves unneeded trajectory frames unread when using the sparse release file.
    original_load = torch.load
    task_path = args.tasks.resolve()

    def load_eval_assets(path, *positional, **kwargs):
        if isinstance(path, (Path, str, os.PathLike)) and Path(path).resolve() == task_path:
            if args.probe_index is not None:
                from fetch_endpoint_ranges import load_ready_probe
                return load_ready_probe(task_path, args.probe_index)
            kwargs["mmap"] = True
        return original_load(path, *positional, **kwargs)

    torch.load = load_eval_assets
    print(f"Torch {torch.__version__}; GPU {torch.cuda.get_device_name()}; tasks {cfg.eval.num_eval}", flush=True)
    run_planning_eval(cfg, results_dir=results)
    if args.probe_index is not None:
        destination = args.output.resolve() / "chunks" / f"tasks_{args.probe_index:03}-{args.probe_index + 1:03}.json"
        destination.parent.mkdir(exist_ok=True, parents=True)
        incoming = destination.with_suffix(".tmp")
        shutil.copyfile(results / "chunks" / "tasks_000-001.json", incoming)
        os.replace(incoming, destination)
    print(f"CUDA peak allocated {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB; "
          f"peak reserved {torch.cuda.max_memory_reserved() / 2**20:.1f} MiB", flush=True)


if __name__ == "__main__":
    main()
