"""Small mechanism experiment for the proposed native finite-difference workspace.

This is an analysis module on the existing P1A host path, not a replacement
operator or a trained JEPA method. No goals, probes, candidate scores or final
task outcomes are read. Outputs retain each coordinate and intermediate state.
"""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np


def words(q, order):
    return [w for k in range(order + 1) for w in product(range(q), repeat=k)]


def compile_coordinates(step, readout, initial, q, order):
    """Evaluate a complete native prefix tree, then form ordered differences."""
    tree = {(): initial}
    values = {(): np.asarray(readout(initial), dtype=np.float64)}
    for w in words(q, order)[1:]:
        tree[w] = step(tree[w[:-1]], w[-1])
        values[w] = np.asarray(readout(tree[w]), dtype=np.float64)
    coordinates = {}
    for w in values:
        total = np.zeros_like(values[()])
        for mask in range(1 << len(w)):
            subword = tuple(a for i, a in enumerate(w) if mask & (1 << i))
            total += (-1) ** (len(w) - len(subword)) * values[subword]
        coordinates[w] = total
    return coordinates, values, len(tree) - 1


def advance(coordinates, action):
    # Read all terms from the old state: this is a simultaneous update.
    return {
        w: v + coordinates[(action,) + w]
        if (action,) + w in coordinates else v.copy()
        for w, v in coordinates.items()
    }


def rollout_coordinates(coordinates, sequence, trace=None):
    state = {w: v.copy() for w, v in coordinates.items()}
    result = []
    for action in sequence:
        state = advance(state, action)
        result.append(state[()].copy())
        if trace is not None:
            trace.append(np.stack(list(state.values())))
    return np.stack(result), state


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def analytic_stage(out):
    # Constant acceleration has a genuine second-order finite closure for all
    # +/- actions. Alternating shears supply an intentionally non-closing case.
    systems = {
        "constant_acceleration": (
            lambda x, a: np.array([x[0] + x[1], x[1] + (1 if a == 0 else -1)]),
            np.array([0., 0.]),
        ),
        "alternating_shears": (
            lambda x, a: np.array([x[0] + x[1], x[1]]) if a == 0
            else np.array([x[0], x[1] + x[0]]),
            np.array([1., 2.]),
        ),
    }
    sequences = {"all_a": (0,) * 8, "all_b": (1,) * 8,
                 "alternating": (0, 1) * 4, "turn_after_two": (0, 0) + (1,) * 6}
    rows, arrays = [], {}
    for name, (step, initial) in systems.items():
        full, tree, calls = compile_coordinates(step, lambda x: x, initial, 2, 3)
        labels = list(full)
        arrays[name + "_coordinate_words"] = np.array([str(w) for w in labels])
        arrays[name + "_coordinates"] = np.stack([full[w] for w in labels])
        for label, sequence in sequences.items():
            x, native = initial.copy(), []
            for a in sequence:
                x = step(x, a)
                native.append(x.copy())
            native = np.stack(native)
            arrays[name + "_" + label + "_native"] = native
            for order in (1, 2, 3):
                c = {w: v for w, v in full.items() if len(w) <= order}
                trace = []
                predicted, _ = rollout_coordinates(c, sequence, trace)
                arrays[f"{name}_{label}_order{order}"] = predicted
                arrays[f"{name}_{label}_working_coordinates_order{order}"] = np.stack(trace)
                for h in range(1, 9):
                    rows.append({"system": name, "sequence": label, "order": order,
                                 "horizon": h, "rmse": rms(predicted[h - 1] - native[h - 1])})
        # Independent hand-computed endpoints exercise noncommuting action order.
        if name == "constant_acceleration":
            np.testing.assert_array_equal(step(step(initial, 0), 1), [1., 0.])
            np.testing.assert_array_equal(step(step(initial, 1), 0), [-1., 0.])
        else:
            np.testing.assert_array_equal(step(step(initial, 0), 1), [3., 5.])
            np.testing.assert_array_equal(step(step(initial, 1), 0), [4., 3.])
    closure_error = max(r["rmse"] for r in rows
                        if r["system"] == "constant_acceleration" and r["order"] >= 2)
    np.testing.assert_allclose(closure_error, 0., atol=1e-12, rtol=0)

    square = lambda x, a: x ** 2
    old, _, _ = compile_coordinates(square, lambda x: x, np.array([1.]), 1, 2)
    refreshed = {w: v.copy() for w, v in old.items()}
    refreshed[()] = np.array([2.])
    rebuilt, _, calls = compile_coordinates(square, lambda x: x, np.array([2.]), 1, 2)
    stale_next, _ = rollout_coordinates(refreshed, (0,))
    rebuilt_next, _ = rollout_coordinates(rebuilt, (0,))
    observation = {"old_latent": 1., "new_observation": 2., "native_next": 4.,
                   "replace_zero_only_next": float(stale_next[0, 0]),
                   "rebuild_next": float(rebuilt_next[0, 0]), "rebuild_native_calls": calls}
    np.testing.assert_array_equal(stale_next, [[2.]])
    np.testing.assert_array_equal(rebuilt_next, [[4.]])
    arrays["observation_old_coordinates"] = np.stack(list(old.values()))
    arrays["observation_rebuilt_coordinates"] = np.stack(list(rebuilt.values()))
    np.savez_compressed(out / "analytic_intermediates.npz", **arrays)
    with (out / "analytic_steps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    save_json(out / "analytic_results.json", {
        "closure_max_rmse_order2_or_3": closure_error, "observation_example": observation,
        "order_counterexamples": {"acceleration_ab": [1, 0], "acceleration_ba": [-1, 0],
                                  "shears_ab": [3, 5], "shears_ba": [4, 3]},
        "h8": [r for r in rows if r["horizon"] == 8],
        "meaning": "Exact analytic closure and deliberate nonclosure; not physical JEPA accuracy."})
    return rows


class NativeStep:
    """One step through the existing P1A rollout and original released modules."""
    def __init__(self, model, alphabet, context, arrays, trace_prefix):
        import torch
        from p1a_reference import latent_rollout
        self.torch, self.kernel = torch, latent_rollout
        self.model, self.alphabet, self.context = model, alphabet, context
        self.calls, self.arrays, self.prefix = 0, arrays, trace_prefix
        self.handles = []
        for name, module in model.named_modules():
            if module.__class__.__name__ == "ConditionalBlock":
                self.handles.append(module.register_forward_hook(self.hook(name)))

    def hook(self, name):
        def capture(module, inputs, output):
            if self.calls <= 2:
                self.arrays[f"{self.prefix}_call{self.calls}_{name}_out"] = output.detach().cpu().numpy()
        return capture

    def __call__(self, state, action):
        torch = self.torch
        z, past = state
        self.calls += 1
        zt = torch.as_tensor(z[None], dtype=torch.float32, device="cuda")
        pt = torch.as_tensor(past[None], dtype=torch.float32, device="cuda")
        at = torch.as_tensor(self.alphabet[action][None, None], dtype=torch.float32, device="cuda")

        def embed(a):
            result = self.model.action_embed(self.model.action_encoder(a))
            if self.calls <= 2:
                self.arrays[f"{self.prefix}_call{self.calls}_actions"] = a.detach().cpu().numpy()
                self.arrays[f"{self.prefix}_call{self.calls}_action_embedding"] = result.detach().cpu().numpy()
            return result

        if self.calls <= 2:
            self.arrays[f"{self.prefix}_call{self.calls}_latent_history"] = z.copy()
        pred = self.kernel(self.model.predict, embed, zt, pt, at, self.context)[0].cpu().numpy()
        znew = np.concatenate([z, pred])[-self.context:]
        anew = np.concatenate([past, self.alphabet[action][None]])
        anew = anew[-(self.context - 1):] if self.context > 1 else anew[:0]
        return znew, anew

    def close(self):
        for h in self.handles:
            h.remove()


def native_stage(args, out):
    # Import the installed numeric runtime before adding historical model dependencies.
    import torch
    sys.path[:0] = [str(args.model_dir / "upstream/h_jepa"), str(args.model_dir / "deps")]
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = torch.load(args.model_dir / "fourroom_lewm_object.ckpt", map_location="cpu", weights_only=False)
    model = model.get_level(1).cuda().eval().requires_grad_(False)
    with np.load(args.development, allow_pickle=False) as source:
        # Intentionally read only existing development inputs and one arrived observation.
        roots = source["root_id"][:3]
        histories = source["z_prefix"][:3]
        pasts = source["prefix_normalized_actions"][:3]
        actions = source["normalized_actions"][:3]
        observed15 = source["z_true"][:3, 1, 0]
        source_times = source["true_absolute_step"]
        if int(source_times[0]) != 15:
            raise ValueError("This experiment requires the recorded east-branch observation at step 15.")
    arrays, rows, refreshes, call_counts = {}, [], [], []
    sequences = {"constant_0": (0,) * 8, "constant_2": (1,) * 8,
                 "turn_0_+1": (0, 0) + (1,) * 6, "turn_2_-1": (1, 1) + (0,) * 6}
    candidate_ids = {"constant_0": 1, "constant_2": 3, "turn_0_+1": 10, "turn_2_-1": 13}
    with torch.inference_mode(), torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        for root_index, root_id in enumerate(roots):
            alphabet = actions[root_index, [1, 3], 0]
            for label, sequence in sequences.items():
                np.testing.assert_allclose(alphabet[list(sequence)], actions[root_index, candidate_ids[label], :8], atol=1e-6)
            for context in (1, 3):
                tag = f"root{int(root_id)}_context{context}"
                initial = (histories[root_index, -context:].copy(),
                           pasts[root_index, -(context - 1):].copy() if context > 1 else pasts[root_index, :0].copy())
                step = NativeStep(model, alphabet, context, arrays, tag)
                full, tree, compile_calls = compile_coordinates(step, lambda s: s[0][-1], initial, 2, 3)
                order_words = list(full)
                arrays[tag + "_words"] = np.array([str(w) for w in order_words])
                arrays[tag + "_coordinates"] = np.stack([full[w] for w in order_words])
                arrays[tag + "_native_tree_outputs"] = np.stack([tree[w] for w in order_words])
                arrays[tag + "_action_alphabet"] = alphabet
                max_prefix_error = 0.
                for label, sequence in sequences.items():
                    state, native = initial, []
                    for a in sequence:
                        state = step(state, a)
                        native.append(state[0][-1].copy())
                    native = np.stack(native)
                    arrays[tag + "_" + label + "_native"] = native
                    for order in (1, 2, 3):
                        c = {w: v for w, v in full.items() if len(w) <= order}
                        trace = []
                        predicted, _ = rollout_coordinates(c, sequence, trace)
                        arrays[f"{tag}_{label}_order{order}"] = predicted
                        arrays[f"{tag}_{label}_working_coordinates_order{order}"] = np.stack(trace)
                        for h in range(1, 9):
                            error = rms(predicted[h - 1] - native[h - 1])
                            displacement = rms(native[h - 1] - initial[0][-1])
                            rows.append({"root_id": int(root_id), "context": context, "sequence": label,
                                         "order": order, "horizon": h, "environment_steps": 5 * h,
                                         "native_rmse": error, "native_displacement_rms": displacement,
                                         "error_over_displacement": error / displacement if displacement else None})
                            if h <= order:
                                max_prefix_error = max(max_prefix_error, error)
                # A native predicted successor, then the actual east-branch observation arrives.
                predicted_state = step(initial, 0)
                rebuilt_predicted, _, predicted_rebuild_calls = compile_coordinates(
                    step, lambda s: s[0][-1], predicted_state, 2, 3)
                observed_state = (predicted_state[0].copy(), predicted_state[1].copy())
                observed_state[0][-1] = observed15[root_index]
                rebuilt, _, rebuild_calls = compile_coordinates(step, lambda s: s[0][-1], observed_state, 2, 3)
                target_predicted = step(predicted_state, 1)[0][-1]
                target = step(observed_state, 1)[0][-1]
                arrays[tag + "_arrived_observation15"] = observed15[root_index]
                arrays[tag + "_predicted_successor_history"] = predicted_state[0]
                arrays[tag + "_observed_successor_history"] = observed_state[0]
                arrays[tag + "_native_next_after_predicted"] = target_predicted
                arrays[tag + "_native_next_after_observed"] = target
                arrays[tag + "_predicted_rebuilt_coordinates"] = np.stack([rebuilt_predicted[w] for w in order_words])
                arrays[tag + "_observation_rebuilt_coordinates"] = np.stack([rebuilt[w] for w in order_words])
                for order in (1, 2, 3):
                    old = {w: v for w, v in full.items() if len(w) <= order}
                    carried = advance(old, 0)
                    before_observation, _ = rollout_coordinates(carried, (1,))
                    exact_prior = {w: v.copy() for w, v in rebuilt_predicted.items() if len(w) <= order}
                    exact_prior[()] = observed15[root_index].astype(np.float64)
                    isolated_observation, _ = rollout_coordinates(exact_prior, (1,))
                    carried[()] = observed15[root_index].astype(np.float64)
                    predicted, _ = rollout_coordinates(carried, (1,))
                    rebuilt_next, _ = rollout_coordinates({w: v for w, v in rebuilt.items() if len(w) <= order}, (1,))
                    refreshes.append({"root_id": int(root_id), "context": context, "order": order,
                                      "replace_latent_only_native_rmse": rms(predicted[0] - target),
                                      "no_observation_transport_rmse": rms(before_observation[0] - target_predicted),
                                      "observation_only_rmse_from_exact_prior": rms(isolated_observation[0] - target),
                                      "rebuild_native_rmse": rms(rebuilt_next[0] - target),
                                      "first_order_coordinate_change_rms": rms(carried[(1,)] - rebuilt[(1,)]),
                                      "observation_only_first_order_change_rms": rms(rebuilt_predicted[(1,)] - rebuilt[(1,)])})
                    arrays[f"{tag}_observation_carried_order{order}"] = np.stack(list(carried.values()))
                    arrays[f"{tag}_observation_exact_prior_zero_only_order{order}"] = np.stack(list(exact_prior.values()))
                call_counts.append({"root_id": int(root_id), "context": context,
                                    "compile_calls_order3": compile_calls, "rebuild_calls_order3": rebuild_calls,
                                    "predicted_anchor_rebuild_calls_order3": predicted_rebuild_calls,
                                    "total_native_step_calls": step.calls, "exact_prefix_max_rmse": max_prefix_error})
                step.close()
                print(tag, "prefix_rmse", max_prefix_error, "native_calls", step.calls, flush=True)
    np.savez_compressed(out / "native_intermediates.npz", **arrays)
    for name, records in (("native_steps.csv", rows), ("observation_steps.csv", refreshes)):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    save_json(out / "native_results.json", {
        "checkpoint": str(args.model_dir / "fourroom_lewm_object.ckpt"), "development_inputs": str(args.development),
        "python": sys.executable, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "dtype": "native float32, difference arithmetic float64", "root_ids": roots.tolist(),
        "alphabet": ["constant_0", "constant_2"], "call_counts": call_counts,
        "h8": [r for r in rows if r["horizon"] == 8], "observation_update": refreshes,
        "meaning": "Deviation from the SAME native predictor; not physical error, task success, or candidate ranking.",
        "limitation": "Two existing action controls, three development roots, no continuous-action generalization; no speed benchmark."})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analytic", "native", "all"), default="all")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--development", type=Path,
                        default=Path("D:/cjepa/runs/p1a_v1/development/branches.npz"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("analytic", "all"):
        analytic_stage(args.out)
        print("Analytic mechanism stage completed", flush=True)
    if args.stage in ("native", "all"):
        if args.model_dir is None:
            parser.error("--model-dir is required for the native stage")
        native_stage(args, args.out)
        print("Native mechanism stage completed", flush=True)


if __name__ == "__main__":
    main()
