"""Offline heldout TwoRoom speed measurements on real, independently run records.

The source is issued at t2, its actual z3 is written, and the resulting belief
is read at the refreshed actual t6 window.  No target after t6 enters a forecast.
The Gaussian comparator changes only this source observation density.
"""

from __future__ import annotations

import math

import torch

from .prbd import assimilate, issue


HORIZONS = (1, 2, 3, 4)
PROGRAMS = ("movement", "stop")
STRATEGIES = ("uniform", "nominal", "prbd_source_only", "gaussian_source_only",
              "all_arrivals", "omit_source")
COMPARISONS = {
    "prbd_source_only_vs_uniform": ("prbd_source_only", "uniform"),
    "gaussian_source_only_vs_uniform": ("gaussian_source_only", "uniform"),
    "all_arrivals_vs_omit_source": ("all_arrivals", "omit_source"),
    "prbd_source_only_vs_gaussian_source_only":
        ("prbd_source_only", "gaussian_source_only"),
    "prbd_source_only_vs_nominal": ("prbd_source_only", "nominal"),
}


def _list(value):
    return value.detach().cpu().tolist()


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _contrast(prediction, truth, families):
    """Retain additive sufficient statistics; never average per-case ratios.

    families contain each speed's K possible route contrasts.  Independent
    simplex weights therefore have the exact signed projection interval below.
    For shared roots this is the usual max-minus-min convex support cap.
    """
    prediction, truth = prediction.double(), truth.double()
    left, right = (value.double() for value in families)
    squared = float(truth.square().sum())
    predicted_squared = float(prediction.square().sum())
    dot = float(prediction @ truth)
    left_scores, right_scores = left @ truth, right @ truth
    lower = float(left_scores.min() - right_scores.max())
    upper = float(left_scores.max() - right_scores.min())
    cap_squared = float((left[:, None] - right[None]).square().sum(-1).max())
    return {
        "truth_l2": math.sqrt(squared), "prediction_l2": math.sqrt(predicted_squared),
        "error_l2": float((prediction - truth).norm()),
        "truth_squared_norm": squared, "prediction_squared_norm": predicted_squared,
        "prediction_truth_dot": dot, "absolute_prediction_truth_dot": abs(dot),
        "error_squared_norm": float((prediction - truth).square().sum()),
        "signed_recovery": _ratio(dot, squared),
        "absolute_projected_recovery": _ratio(abs(dot), squared),
        "norm_recovery": math.sqrt(predicted_squared / squared) if squared > 0 else None,
        "relative_squared_error_gain_vs_zero": _ratio(2 * dot - predicted_squared, squared),
        "convex_signed_projection_lower_numerator": lower,
        "convex_signed_projection_cap_numerator": upper,
        "convex_absolute_projection_cap_numerator": max(abs(lower), abs(upper)),
        "convex_norm_cap_squared": cap_squared,
        "convex_signed_recovery_cap": _ratio(upper, squared),
        "convex_absolute_projected_recovery_cap": _ratio(max(abs(lower), abs(upper)), squared),
        "convex_norm_recovery_cap": math.sqrt(cap_squared / squared) if squared > 0 else None,
    }


def _pool_contrasts(rows):
    keys = ("truth_squared_norm", "prediction_squared_norm", "prediction_truth_dot",
            "absolute_prediction_truth_dot", "error_squared_norm",
            "convex_signed_projection_lower_numerator",
            "convex_signed_projection_cap_numerator",
            "convex_absolute_projection_cap_numerator", "convex_norm_cap_squared")
    result = {"count": len(rows), "zero_truth_count": sum(
        row["truth_squared_norm"] == 0 for row in rows)}
    result.update({key + "_sum": math.fsum(row[key] for row in rows) for key in keys})
    denominator = result["truth_squared_norm_sum"]
    dot = result["prediction_truth_dot_sum"]
    predicted_squared = result["prediction_squared_norm_sum"]
    result.update({
        "pooled_signed_recovery": _ratio(dot, denominator),
        "pooled_absolute_projected_recovery": _ratio(
            result["absolute_prediction_truth_dot_sum"], denominator),
        "pooled_norm_recovery": math.sqrt(predicted_squared / denominator)
        if denominator > 0 else None,
        "pooled_relative_squared_error_gain_vs_zero": _ratio(
            2 * dot - predicted_squared, denominator),
        "pooled_convex_signed_recovery_cap": _ratio(
            result["convex_signed_projection_cap_numerator_sum"], denominator),
        "pooled_convex_absolute_projected_recovery_cap": _ratio(
            result["convex_absolute_projection_cap_numerator_sum"], denominator),
        "pooled_convex_norm_recovery_cap": math.sqrt(
            result["convex_norm_cap_squared_sum"] / denominator)
        if denominator > 0 else None,
    })
    if rows and "interaction_squared_error_gain" in rows[0]:
        result["interaction_squared_error_gain_sum"] = math.fsum(
            row["interaction_squared_error_gain"] for row in rows)
        result["improved_interaction_seed_count"] = sum(
            row["interaction_squared_error_gain"] > 0 for row in rows)
    return result


def _odds_identity(issued, arrival):
    """Check every unordered prototype pair in stable log coordinates."""
    prior_log = issued.prior.log()
    likelihood = arrival.relative_log_likelihood
    posterior_log = torch.log_softmax(prior_log + likelihood, dim=0)
    pairs = []
    for i in range(len(prior_log)):
        for j in range(i + 1, len(prior_log)):
            increment = (posterior_log[i] - posterior_log[j]) - (prior_log[i] - prior_log[j])
            expected = likelihood[i] - likelihood[j]
            absolute_density_difference = arrival.log_likelihood[i] - arrival.log_likelihood[j]
            stored_positive = bool((arrival.posterior[[i, j]] > 0).all())
            stored_error = None
            if stored_positive:
                stored = arrival.posterior[i].log() - arrival.posterior[j].log()
                stored_error = float((stored - (prior_log[i] - prior_log[j]) - expected).abs())
            pairs.append({"i": i, "j": j, "posterior_log_odds_increment": float(increment),
                          "log_likelihood_ratio": float(expected),
                          "identity_absolute_error": float((increment - expected).abs()),
                          "absolute_log_density_ratio_roundoff": float(
                              (absolute_density_difference - expected).abs()),
                          "stored_posterior_identity_absolute_error": stored_error})
    stored_errors = [row["stored_posterior_identity_absolute_error"] for row in pairs
                     if row["stored_posterior_identity_absolute_error"] is not None]
    return {"pair_count": len(pairs), "pairs": pairs,
            "stored_positive_pair_count": len(stored_errors),
            "max_absolute_error": max(stored_errors, default=None),
            "max_logspace_absolute_error": max(row["identity_absolute_error"] for row in pairs),
            "underflowed_posterior_entries": int((arrival.posterior == 0).sum())}


def _endpoint(branches, root, target, weights):
    branches, root, target = (value.double() for value in (branches, root, target))
    predictions = {name: weight @ branches for name, weight in weights.items()}
    strategies = {}
    for name, prediction in predictions.items():
        error = float((prediction - target).square().sum())
        displacement = float((prediction - root).square().sum())
        strategies[name] = {"squared_error": error, "mse": error / target.numel(),
                            "prediction_displacement_squared_norm": displacement,
                            "prediction_displacement_l2": math.sqrt(displacement)}
    comparisons = {}
    for name, (after, before) in COMPARISONS.items():
        move = predictions[after] - predictions[before]
        inner = float(move @ (target - predictions[before]))
        move_squared = float(move.square().sum())
        gain = strategies[before]["squared_error"] - strategies[after]["squared_error"]
        comparisons[name] = {"squared_error_gain": gain,
                             "prediction_change_squared_norm": move_squared,
                             "prediction_change_l2": math.sqrt(move_squared),
                             "toward_actual_residual_dot": inner,
                             "gain_identity_absolute_error": abs(gain - 2 * inner + move_squared)}
    return {"actual_displacement_squared_norm": float((target - root).square().sum()),
            "strategies": strategies, "comparisons": comparisons}


def _summarize(rows):
    result = {"seeds": len(rows), "horizons": {}}
    for horizon in HORIZONS:
        key = str(horizon)
        eligible = [row for row in rows if key in row["interactions"]]
        interaction = {
            "eligible_seed_count": len(eligible),
            "full_prediction": {strategy: _pool_contrasts([
                row["interactions"][key]["full_prediction"][strategy] for row in eligible])
                for strategy in STRATEGIES},
            "belief_only_revision": {comparison: _pool_contrasts([
                row["interactions"][key]["belief_only_revision"][comparison] for row in eligible])
                for comparison in COMPARISONS},
        }
        # Every strategy uses this same set and denominator, including Gaussian.
        interaction["shared_truth_squared_norm_sum"] = math.fsum(
            row["interactions"][key]["truth_squared_norm"] for row in eligible)
        endpoint = {}
        for program in PROGRAMS:
            by_speed = {}
            for speed in (0, 1):
                records = [record for row in rows for record in row["endpoints"]
                           if record["program"] == program and record["speed_index"] == speed
                           and record["horizon"] == horizon]
                count = len(records)
                by_speed[str(speed)] = {
                    "record_count": count,
                    "actual_displacement_squared_norm_mean": math.fsum(
                        record["actual_displacement_squared_norm"] for record in records) / count
                    if count else None,
                    "strategies": {strategy: {
                        metric + "_mean": math.fsum(record["strategies"][strategy][metric]
                                                     for record in records) / count if count else None
                        for metric in ("squared_error", "mse", "prediction_displacement_squared_norm")}
                        for strategy in STRATEGIES},
                    "comparisons": {comparison: {
                        "squared_error_gain_sum": math.fsum(
                            record["comparisons"][comparison]["squared_error_gain"] for record in records),
                        "squared_error_gain_mean": math.fsum(
                            record["comparisons"][comparison]["squared_error_gain"] for record in records
                        ) / count if count else None,
                        "improved_record_count": sum(
                            record["comparisons"][comparison]["squared_error_gain"] > 0 for record in records),
                        "prediction_change_squared_norm_mean": math.fsum(
                            record["comparisons"][comparison]["prediction_change_squared_norm"]
                            for record in records) / count if count else None,
                    } for comparison in COMPARISONS}}
            pairs = [[record for record in row["endpoints"]
                      if record["program"] == program and record["horizon"] == horizon]
                     for row in rows]
            pairs = [pair for pair in pairs if len(pair) == 2]
            endpoint[program] = {"by_speed": by_speed, "paired_seed_count": len(pairs),
                "paired_comparisons": {comparison: {
                    "improved_seed_pair_count": sum(math.fsum(
                        record["comparisons"][comparison]["squared_error_gain"] for record in pair) > 0
                        for pair in pairs),
                    "paired_squared_error_gain_sum": math.fsum(
                        record["comparisons"][comparison]["squared_error_gain"]
                        for pair in pairs for record in pair),
                } for comparison in COMPARISONS}}
        result["horizons"][key] = {"state_by_action_interaction": interaction,
                                     "actual_endpoint_errors": endpoint}
    evidence = [source for row in rows for source in row["sources"]]
    result["source_evidence"] = {"actual_record_count": len(evidence),
        "max_log_odds_identity_absolute_error": max(
            (source["odds_identity"]["max_absolute_error"] for source in evidence
             if source["odds_identity"]["max_absolute_error"] is not None), default=None),
        "max_logspace_odds_identity_absolute_error": max(
            (source["odds_identity"]["max_logspace_absolute_error"] for source in evidence), default=None),
        "source_odds_total_pair_count": sum(source["odds_identity"]["pair_count"] for source in evidence),
        "source_odds_stored_positive_pair_count": sum(
            source["odds_identity"]["stored_positive_pair_count"] for source in evidence),
        "total_variation_from_uniform_mean": math.fsum(
            source["total_variation_from_uniform"] for source in evidence) / len(evidence)
        if evidence else None,
        "gaussian_total_variation_from_uniform_mean": math.fsum(
            source["gaussian_total_variation_from_uniform"] for source in evidence) / len(evidence)
        if evidence else None,
        "source_prediction_improved_record_count": sum(
            source["source_endpoint"]["comparisons"]["prbd_source_only_vs_uniform"]["squared_error_gain"] > 0
            for source in evidence)}
    result["source_speed_contrasts"] = {}
    for program in PROGRAMS:
        sources = [row["source_speed_contrasts"][program] for row in rows
                   if program in row["source_speed_contrasts"]]
        result["source_speed_contrasts"][program] = {
            "seed_count": len(sources),
            "posterior_total_variation_between_speeds_mean": math.fsum(
                source["posterior_total_variation_between_speeds"] for source in sources) / len(sources)
            if sources else None,
            "strategies": {name: _pool_contrasts([source["strategies"][name] for source in sources])
                           for name in ("uniform", "prbd_source_only", "gaussian_source_only")}}
    return result


@torch.no_grad()
def measure(model, fitted, cases):
    """Return JSON-compatible rows and pooled summaries, without changing weights.

    Each case supplies seed, z[2,2,T+1,D], a[2,2,T,A], valid actual block
    lengths[2,2], and independently checked matched_rgb.  Program 0 is the
    orthogonal movement suffix, program 1 is zero.  Missing actual endpoints
    never become targets; their coverage remains visible in the result.
    """
    if not cases:
        raise ValueError("At least one heldout reset seed is required")
    if any(module.training for module in model.modules()):
        raise ValueError("Heldout measurement requires an eval-mode model")
    count = len(model.conditions)
    noise, transition, scales, masses = (fitted[name] for name in (
        "noise_cholesky", "transition", "scales", "masses"))
    if not torch.equal(transition.double(), torch.eye(
            count, dtype=torch.float64, device=transition.device)):
        raise ValueError("This fixed-condition speed measurement requires T=I")
    uniform = transition.new_full((count,), 1 / count, dtype=torch.float64)
    nominal = torch.zeros_like(uniform)
    nominal[0] = 1
    rows, seen = [], set()

    def write(responses, prior, actual):
        signed = issue(responses, prior, transition, noise)
        return signed, assimilate(signed, actual, scales, masses)

    for case in cases:
        seed, z, actions = case["seed"], case["z"], case["a"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed in seen:
            raise ValueError("Heldout seeds must be unique integers")
        seen.add(seed)
        if any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
               or not torch.isfinite(value).all() for value in (z, actions)):
            raise ValueError("Heldout states and actions must be finite floating tensors")
        if (z.ndim != 4 or actions.ndim != 4 or z.shape[:2] != (2, 2)
                or actions.shape[:2] != (2, 2) or z.shape[2] != actions.shape[2] + 1
                or z.shape[-1] != 192 or actions.shape[-1] != 192):
            raise ValueError("Expected real z[2,2,T+1,192] and embedded a[2,2,T,192]")
        if z.device != uniform.device or actions.device != uniform.device:
            raise ValueError("Heldout data, fitted tensors and model must share a device")
        lengths = torch.as_tensor(case["lengths"])
        if (lengths.shape != (2, 2) or lengths.dtype == torch.bool
                or lengths.is_floating_point() or (lengths < 0).any()
                or (lengths > actions.shape[2]).any()):
            raise ValueError("lengths must be integral [2,2] counts of actual valid blocks")
        raw_match = case.get("matched_rgb", False)
        if not isinstance(raw_match, bool):
            raise ValueError("matched_rgb must be an independently checked boolean")
        blocks = [[int(lengths[s, p]) for p in (0, 1)] for s in (0, 1)]
        row = {"seed": seed, "lengths": blocks, "matched_rgb": raw_match,
               "records": [], "sources": [], "source_speed_contrasts": {},
               "endpoints": [], "interactions": {}}
        prefix_equal = []
        for speed in (0, 1):
            common = min(6, *blocks[speed])
            prefix_equal.append(bool(torch.equal(z[speed, 0, :common + 1],
                                                 z[speed, 1, :common + 1])
                                     and torch.equal(actions[speed, 0, :common],
                                                     actions[speed, 1, :common])))
        if not all(prefix_equal):
            raise ValueError(f"Seed {seed}: independently run programs disagree before their fork")
        root_available = all(length >= 6 for record in blocks for length in record)
        encoded_root_error = float((z[:, :, 4:7] - z[0, 0, 4:7]).abs().max()) \
            if root_available else None
        encoded_root_equal = root_available and bool(torch.equal(
            z[:, :, 4:7], z[0, 0, 4:7].expand(2, 2, -1, -1)))
        encoded_root_close = root_available and bool(torch.allclose(
            z[:, :, 4:7], z[0, 0, 4:7].expand(2, 2, -1, -1), atol=1e-6, rtol=1e-6))
        shared_history_commands = root_available and bool(torch.equal(
            actions[:, :, 4:6], actions[0, 0, 4:6].expand(2, 2, -1, -1)))
        commands_match = all(torch.equal(actions[0, p, :min(blocks[0][p], blocks[1][p])],
                                         actions[1, p, :min(blocks[0][p], blocks[1][p])])
                             for p in (0, 1))
        row["root_checks"] = {
            "all_four_actual_roots_available": root_available,
            "program_prefix_identical_by_speed": prefix_equal,
            "encoded_root_exactly_equal": encoded_root_equal,
            "encoded_root_allclose": encoded_root_close,
            "encoded_root_max_absolute_difference": encoded_root_error,
            "history_commands_equal": shared_history_commands,
            "program_commands_equal_across_speeds": commands_match,
            "matched_subset_eligible": bool(raw_match and encoded_root_close
                                              and shared_history_commands and commands_match),
        }
        weights, source_responses, paths = {}, {}, {}
        for speed in (0, 1):
            for program in (0, 1):
                length = blocks[speed][program]
                record = {"speed_index": speed, "program": PROGRAMS[program],
                          "valid_actual_blocks": length, "source_arrival_available": length >= 3,
                          "refreshed_root_available": length >= 6,
                          "actual_endpoint_horizons": [h for h in HORIZONS if length >= 6 + h]}
                row["records"].append(record)
                if length < 3:
                    continue
                responses = model(z[speed, program, None, :3],
                                  actions[speed, program, None, :3])[0].double()
                if responses.shape != (count, z.shape[-1]) or not torch.isfinite(responses).all():
                    raise ValueError("Native source responses must have finite [K,D] shape")
                signed, arrival = write(responses, uniform, z[speed, program, 3])
                x = torch.linalg.solve_triangular(
                    signed.noise_cholesky, (z[speed, program, 3].double() - signed.center)[:, None],
                    upper=False)[:, 0]
                gaussian_scores = signed.basis.T @ x - .5 * signed.gram.diagonal()
                gaussian = torch.softmax(uniform.log() + gaussian_scores, dim=0)
                state = {"uniform": uniform, "nominal": nominal,
                         "prbd_source_only": arrival.posterior,
                         "gaussian_source_only": gaussian, "all_arrivals": arrival.posterior,
                         "omit_source": uniform}
                source_responses[speed, program] = responses
                source_endpoint = _endpoint(responses, z[speed, program, 2], z[speed, program, 3], state)
                row["sources"].append({
                    "speed_index": speed, "program": PROGRAMS[program],
                    "source_state_indices": [0, 1, 2], "source_action_index": 2,
                    "arrival_state_index": 3,
                    "prior": _list(uniform), "posterior": _list(arrival.posterior),
                    "gaussian_posterior": _list(gaussian),
                    "log_likelihood": _list(arrival.log_likelihood),
                    "relative_log_likelihood": _list(arrival.relative_log_likelihood),
                    "gaussian_relative_log_likelihood": _list(gaussian_scores),
                    "total_variation_from_uniform": float((arrival.posterior - uniform).abs().sum() / 2),
                    "gaussian_total_variation_from_uniform": float((gaussian - uniform).abs().sum() / 2),
                    "odds_identity": _odds_identity(signed, arrival),
                    "source_endpoint": source_endpoint})
                timeline = [{"arrival_state_index": 3, "all_arrivals": _list(arrival.posterior),
                             "omit_source": _list(uniform)}]
                for t in range(3, min(6, length)):
                    response = model(z[speed, program, None, t - 2:t + 1],
                                     actions[speed, program, None, t - 2:t + 1])[0]
                    for strategy in ("all_arrivals", "omit_source"):
                        _, update = write(response, state[strategy], z[speed, program, t + 1])
                        state[strategy] = update.posterior
                    timeline.append({"arrival_state_index": t + 1,
                                     **{name: _list(state[name]) for name in ("all_arrivals", "omit_source")}})
                row["sources"][-1]["chronological_timeline"] = timeline
                row["sources"][-1]["retained_weights_at_last_actual_arrival"] = {
                    name: _list(value) for name, value in state.items()}
                weights[speed, program] = state
                horizon = min(4, actions.shape[2] - 6)
                if length < 6 or horizon < 1:
                    continue
                # Each prototype owns its predicted window, via NativePrototypes.rollout.
                forecast = model.rollout(z[speed, program, None, 4:7],
                                         actions[speed, program, None, 4:6 + horizon])[0].double()
                if forecast.shape != (count, horizon, z.shape[-1]) or not torch.isfinite(forecast).all():
                    raise ValueError("Native free paths must have finite [K,H,D] shape")
                paths[speed, program] = forecast
                for h in HORIZONS:
                    if h <= horizon and length >= 6 + h:
                        row["endpoints"].append({"speed_index": speed, "program": PROGRAMS[program],
                            "horizon": h, "target_state_index": 6 + h,
                            **_endpoint(forecast[:, h - 1], z[speed, program, 6],
                                        z[speed, program, 6 + h], state)})
        for program in (0, 1):
            if any((speed, program) not in source_responses for speed in (0, 1)):
                continue
            families = [source_responses[speed, program] for speed in (0, 1)]
            truth = z[0, program, 3].double() - z[1, program, 3].double()
            row["source_speed_contrasts"][PROGRAMS[program]] = {
                "posterior_total_variation_between_speeds": float((
                    weights[0, program]["prbd_source_only"] - weights[1, program]["prbd_source_only"]
                ).abs().sum() / 2),
                "source_windows_equal": bool(torch.equal(z[0, program, :3], z[1, program, :3])),
                "strategies": {name: _contrast(
                    weights[0, program][name] @ families[0] - weights[1, program][name] @ families[1],
                    truth, families) for name in ("uniform", "prbd_source_only", "gaussian_source_only")}}
        row["independent_replay_checks"] = []
        for speed in (0, 1):
            both_source = all((speed, p) in source_responses for p in (0, 1))
            row["independent_replay_checks"].append({
                "speed_index": speed, "both_source_arrivals_available": both_source,
                "independently_issued_source_max_absolute_difference": float((
                    source_responses[speed, 0] - source_responses[speed, 1]).abs().max())
                if both_source else None,
                "retained_belief_max_absolute_difference": {name: float((
                    weights[speed, 0][name] - weights[speed, 1][name]).abs().max())
                    for name in STRATEGIES} if root_available else None})
        for h in HORIZONS:
            if not all((s, p) in paths and paths[s, p].shape[1] >= h
                       and blocks[s][p] >= 6 + h for s in (0, 1) for p in (0, 1)):
                continue
            families = [paths[s, 0][:, h - 1] - paths[s, 1][:, h - 1] for s in (0, 1)]
            truth = (z[0, 0, 6 + h].double() - z[1, 0, 6 + h].double()
                     - z[0, 1, 6 + h].double() + z[1, 1, 6 + h].double())
            route_predictions = {name: [
                weights[s, 0][name] @ paths[s, 0][:, h - 1]
                - weights[s, 1][name] @ paths[s, 1][:, h - 1] for s in (0, 1)]
                for name in STRATEGIES}
            predictions = {name: value[0] - value[1] for name, value in route_predictions.items()}
            revisions = {}
            for comparison, (after, before) in COMPARISONS.items():
                centered = [families[s] - route_predictions[before][s] for s in (0, 1)]
                revisions[comparison] = _contrast(predictions[after] - predictions[before], truth, centered)
                revisions[comparison]["interaction_squared_error_gain"] = float(
                    (predictions[before] - truth).square().sum()
                    - (predictions[after] - truth).square().sum())
            row["interactions"][str(h)] = {"truth_squared_norm": float(truth.square().sum()),
                "true_speed_by_program_interaction": _list(truth),
                "full_prediction": {name: _contrast(value, truth, families)
                                    for name, value in predictions.items()},
                "belief_only_revision": revisions}
        row["status"] = ("matched_root" if row["root_checks"]["matched_subset_eligible"] else
                         "unmatched_root" if root_available else "truncated_before_common_root")
        row["complete_H1"] = "1" in row["interactions"]
        row["complete_H4"] = "4" in row["interactions"]
        rows.append(row)
    records = [record for row in rows for record in row["records"]]
    matched = [row for row in rows if row["root_checks"]["matched_subset_eligible"]]
    return {
        "protocol": {"source_issued_at": 2, "source_actual_arrival": 3,
                     "refreshed_actual_root": 6, "root_state_indices": [4, 5, 6],
                     "rollout_control_indices": [4, 5, 6, 7, 8, 9], "horizons": list(HORIZONS),
                     "programs": list(PROGRAMS), "initial_belief": "uniform", "transport": "identity",
                     "source_only": "Only z3 updates uniform; no later arrival writes its retained belief",
                     "continuous": "All actual arrivals z3..z6 versus the same chronology omitting only z3",
                     "gaussian_comparator": "Same responses and R; N(f_k,R), one source update, no refit",
                     "interaction_orientation": "(speed0 movement - speed1 movement) - (speed0 stop - speed1 stop)",
                     "aggregation": "Shared endpoint eligibility; sum numerator / sum truth squared norm; no mean of ratios",
                     "quiet_false_displacement": "Stop prediction displacement from its own actual z6; actual drift also reported",
                     "future_truth_use": "Scoring only; no z7..z10 passed to forecasts or belief writes",
                     "independent_record_unit": "One actual simulator rollout per speed/program/reset seed"},
        "coverage": {"reset_seeds": len(rows), "independent_actual_records": len(records),
                     "source_arrival_records": sum(record["source_arrival_available"] for record in records),
                     "refreshed_root_records": sum(record["refreshed_root_available"] for record in records),
                     "matched_rgb_seed_count": sum(row["matched_rgb"] for row in rows),
                     "matched_encoded_and_command_seed_count": len(matched),
                     "independent_source_replay_pair_count": sum(
                         check["both_source_arrivals_available"]
                         for row in rows for check in row["independent_replay_checks"]),
                     "unmatched_common_root_seed_count": sum(
                         row["status"] == "unmatched_root" for row in rows),
                     "truncated_before_common_root_seed_count": sum(
                         row["status"] == "truncated_before_common_root" for row in rows),
                     "complete_H1_seed_count": sum(row["complete_H1"] for row in rows),
                     "complete_H4_seed_count": sum(row["complete_H4"] for row in rows),
                     "actual_endpoint_record_count_by_horizon": {str(h): sum(
                         h in record["actual_endpoint_horizons"] for record in records) for h in HORIZONS}},
        "all_outcomes": _summarize(rows), "matched_roots": _summarize(matched), "seeds": rows,
    }
