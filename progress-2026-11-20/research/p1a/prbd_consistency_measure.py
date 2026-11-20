"""Fixed G3/G4-compatible source-transfer evaluation on actual TwoRoom records.

The forecast boundary receives actual z0..z6 only. Each future program and
prototype owns its native rollout window. Physical diagnostics enter matching,
stratification and scoring, never a model input or belief update.
"""

from __future__ import annotations

from .prbd_paths import CONSISTENCY, output_path

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .prbd import assimilate_continuous, issue
from .prbd_evidence_attribution import vector_stats
from .prbd_native import NativePrototypes
from .prbd_tworoom_speed_fit import load_host


OUT = CONSISTENCY
HORIZONS = (1, 2, 3, 4)
STRATEGIES = ("nominal", "uniform", "single_source", "all_arrivals", "omit_source")
COMPARISONS = {"single_vs_uniform": ("single_source", "uniform"),
               "all_vs_uniform": ("all_arrivals", "uniform"),
               "all_vs_omit": ("all_arrivals", "omit_source")}
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 4000, 20261120


def _values(value):
    return value.detach().cpu().tolist()


def _ratio(numerator, denominator):
    return numerator / denominator if denominator > 0 else None


def _difference(after, before, target):
    after, before, target = (value.double() for value in (after, before, target))
    move, residual = after - before, target - before
    before_sse, after_sse = float(residual.square().sum()), float((after-target).square().sum())
    gain, dot, squared = before_sse-after_sse, float(move@residual), float(move.square().sum())
    return {"before_squared_error":before_sse,"after_squared_error":after_sse,
            "squared_error_gain":gain,"relative_squared_error_gain":_ratio(gain,before_sse),
            "revision_squared_norm":squared,"toward_residual_dot":dot,
            "gain_identity_absolute_error":abs(gain-2*dot+squared),
            "revision_truth_dot":float(move@target)}


@torch.no_grad()
def forecast_prefix(model, fitted, z_prefix, actions):
    """Issue from z0..z6 only; no future observation argument is accepted."""
    if z_prefix.shape != (2,2,7,192) or actions.shape != (2,2,10,192):
        raise ValueError("Expected exactly actual z[2,2,7,192] and controls[2,2,10,192]")
    k = len(model.conditions)
    uniform = torch.full((k,),1/k,dtype=torch.float64)
    nominal = torch.zeros_like(uniform)
    nominal[0] = 1
    supports = {name:torch.empty((2,k),dtype=torch.float64) for name in STRATEGIES}
    forecasts, traces, noops = [], [], []
    noise, transition = fitted["noise_cholesky"], fitted["transition"]
    nodes, masses = fitted["scales"], fitted["masses"]
    for speed in (0,1):
        z, a = z_prefix[speed,0], actions[speed,0]
        state = {"all_arrivals":uniform.clone(),"omit_source":uniform.clone()}
        speed_trace = []
        for t in range(2,6):
            means = model(z[None,t-2:t+1],a[None,t-2:t+1])[0]
            event = {"action_t":t,"arrival_index":t+1,"weights":{}}
            for chronology in ("all_arrivals","omit_source"):
                before = state[chronology]
                signed = issue(means,before,transition,noise,reference="uniform")
                arrival = assimilate_continuous(signed,z[t+1],nodes,masses)
                skipped = chronology=="omit_source" and t==2
                after = before if skipped else arrival.posterior
                state[chronology] = after
                event["weights"][chronology] = {"before":_values(before),"issued_prior":_values(signed.prior),
                    "after":_values(after),"relative_log_likelihood":_values(arrival.relative_log_likelihood),
                    "omitted":skipped,"total_variation_from_before":float((after-before).abs().sum()/2)}
                if t in (4,5):
                    equal_means = torch.equal(means,means[:1].expand_as(means))
                    equal_density = torch.equal(arrival.log_likelihood,arrival.log_likelihood[:1].expand_as(arrival.log_likelihood))
                    observation_exact = torch.equal(arrival.posterior,signed.prior)
                    public_exact=torch.equal(after,before)
                    if not (equal_means and equal_density and observation_exact and public_exact):
                        raise AssertionError("A pinned no-op introduced a prototype likelihood or observation write")
                    noops.append({"speed_index":speed,"chronology":chronology,"action_t":t,
                        "means_exactly_equal":equal_means,"likelihoods_exactly_equal":equal_density,
                        "observation_no_write_exact":observation_exact,
                        "public_weights_no_write_exact":public_exact,
                        "public_max_abs_change":float((after-before).abs().max()),
                        "public_log_odds_max_abs_change":_finite_log_odds_error(after,before)})
            if t==2:
                supports["single_source"][speed] = state["all_arrivals"]
            speed_trace.append(event)
        for name in ("all_arrivals","omit_source"):
            supports[name][speed] = state[name]
        supports["nominal"][speed],supports["uniform"][speed] = nominal,uniform
        paths = [model.rollout(z_prefix[speed,p,4:7][None],actions[speed,p,4:10][None])[0]
                 for p in (0,1)]
        forecasts.append(torch.stack(paths))
        traces.append(speed_trace)
    paths = torch.stack(forecasts).double()
    common = paths[:,:,0]
    predictions = {name:common+torch.einsum("sk,spkhd->sphd",weight,paths-common[:,:,None])
                   for name,weight in supports.items()}
    return {"supports":supports,"future_responses":paths,"predictions":predictions,
            "actual_write_traces":traces,"no_op_checks":noops}


def _finite_log_odds_error(after,before):
    positive = (before>0)&(after>0)
    if int(positive.sum())<2:
        return 0.
    a,b = after[positive].log(),before[positive].log()
    return float(((a-a[0])-(b-b[0])).abs().max())


def score_case(seed,z,physical,predicted,direction):
    """Consume future targets only after all supports and paths were issued."""
    targets = z[:,:,7:11].double()
    physical_targets = physical[:,:,35:51:5].double()
    truth = targets[0,0]-targets[0,1]-targets[1,0]+targets[1,1]
    physical_truth = physical_targets[0,0]-physical_targets[0,1]-physical_targets[1,0]+physical_targets[1,1]
    predictions = predicted["predictions"]
    row = {"seed":seed,"probe_direction":direction,"horizons":{},
        "supports":{name:_values(value) for name,value in predicted["supports"].items()},
        "between_speed_support_TV":{name:float((weight[0]-weight[1]).abs().sum()/2)
                                    for name,weight in predicted["supports"].items()},
        "actual_write_traces":predicted["actual_write_traces"],"no_op_checks":predicted["no_op_checks"]}
    for h in HORIZONS:
        key,hi = str(h),h-1
        contrast = {name:value[0,0,hi]-value[0,1,hi]-value[1,0,hi]+value[1,1,hi]
                    for name,value in predictions.items()}
        endpoint = {name:{"mse_by_speed_program":_values((value[:,:,hi]-targets[:,:,hi]).square().mean(-1)),
                         "squared_error_by_speed_program":_values((value[:,:,hi]-targets[:,:,hi]).square().sum(-1)),
                         "prediction_displacement_l2_by_speed_program":_values((value[:,:,hi]-z[:,:,6].double()).norm(dim=-1))}
                    for name,value in predictions.items()}
        comparisons = {}
        for label,(after,before) in COMPARISONS.items():
            comparison = _difference(contrast[after],contrast[before],truth[hi])
            before_error = (predictions[before][:,:,hi]-targets[:,:,hi]).square().mean(-1)
            after_error = (predictions[after][:,:,hi]-targets[:,:,hi]).square().mean(-1)
            delta = before_error-after_error
            comparison.update({"endpoint_mse_gain_by_speed_program":_values(delta),
                "paired_endpoint_mse_gain":float(delta.mean()),
                "paired_endpoint_mse_before":float(before_error.mean()),
                "paired_endpoint_mse_after":float(after_error.mean()),
                "paired_movement_mse_gain":float(delta[:,0].mean()),
                "paired_stop_mse_gain":float(delta[:,1].mean())})
            comparisons[label] = comparison
        row["horizons"][key] = {"physical_interaction_xy":_values(physical_truth[hi]),
            "physical_nonzero":bool((physical_truth[hi]!=0).any()),
            "actual_latent_interaction_l2":float(truth[hi].norm()),
            "actual_endpoint_displacement_l2_by_speed_program":_values((targets[:,:,hi]-z[:,:,6].double()).norm(dim=-1)),
            "interaction":{name:vector_stats(value,truth[hi]) for name,value in contrast.items()},
            "endpoints":endpoint,"comparisons":comparisons}
    return row


def _ci(samples):
    if len(samples)==0:
        return None
    return np.quantile(samples,[.025,.975]).tolist()


def _safe_bootstrap_ratio(numerator,denominator,indices):
    top,bottom = numerator[indices].sum(-1),denominator[indices].sum(-1)
    valid = bottom>0
    return _ci(top[valid]/bottom[valid]) if valid.any() else None


def summarize(rows,*,with_directions=True):
    result = {"independent_reset_count":len(rows),"seeds":[row["seed"] for row in rows],"horizons":{}}
    if not rows:
        return result
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0,len(rows),(BOOTSTRAP_DRAWS,len(rows)))
    for h in map(str,HORIZONS):
        scored = [row["horizons"][h] for row in rows]
        target = np.array([r["interaction"]["uniform"]["truth_squared_norm"] for r in scored])
        group = {"physical_nonzero_reset_count":sum(r["physical_nonzero"] for r in scored),
                 "interaction_truth_squared_norm_sum":float(target.sum()),
                 "bootstrap_positive_truth_denominator_draws":int((target[draws].sum(-1)>0).sum()),
                 "strategies":{},"comparisons":{}}
        for name in STRATEGIES:
            dot = np.array([r["interaction"][name]["prediction_truth_dot"] for r in scored])
            sse = np.array([r["interaction"][name]["error_squared_norm"] for r in scored])
            endpoint = np.array([r["endpoints"][name]["mse_by_speed_program"] for r in scored])
            group["strategies"][name] = {"signed_recovery":_ratio(float(dot.sum()),float(target.sum())),
                "signed_recovery_ci95":_safe_bootstrap_ratio(dot,target,draws),
                "interaction_squared_error_sum":float(sse.sum()),
                "sign_counts":{label:int((dot>0).sum()) if label=="positive" else int((dot<0).sum()) if label=="negative" else int((dot==0).sum()) for label in ("positive","negative","zero")},
                "absolute_endpoint_mse_mean":float(endpoint.mean()),
                "absolute_endpoint_mse_mean_ci95":_ci(endpoint.mean((1,2))[draws].mean(-1)),
                "absolute_endpoint_mse_by_speed_program":endpoint.mean(0).tolist(),
                "absolute_endpoint_mse_by_speed_program_ci95":np.quantile(endpoint[draws].mean(1),[.025,.975],axis=0).tolist(),
                "absolute_endpoint_mse_by_program":endpoint.mean((0,1)).tolist()}
        for name in COMPARISONS:
            comparisons = [r["comparisons"][name] for r in scored]
            gain = np.array([r["squared_error_gain"] for r in comparisons])
            before = np.array([r["before_squared_error"] for r in comparisons])
            endpoint = np.array([r["paired_endpoint_mse_gain"] for r in comparisons])
            endpoint_before = np.array([r["paired_endpoint_mse_before"] for r in comparisons])
            revision_dot = np.array([r["revision_truth_dot"] for r in comparisons])
            endpoint_delta = np.array([r["endpoint_mse_gain_by_speed_program"] for r in comparisons])
            total_before,total_target = float(before.sum()),float(target.sum())
            loo_recovery = [_ratio(float(revision_dot.sum()-revision_dot[i]),float(target.sum()-target[i])) for i in range(len(rows))]
            loo_gain = [_ratio(float(gain.sum()-gain[i]),float(before.sum()-before[i])) for i in range(len(rows))]
            group["comparisons"][name] = {"interaction_squared_error_gain_sum":float(gain.sum()),
                "interaction_relative_squared_error_gain":_ratio(float(gain.sum()),total_before),
                "interaction_relative_squared_error_gain_ci95":_safe_bootstrap_ratio(gain,before,draws),
                "bootstrap_positive_reference_error_draws":int((before[draws].sum(-1)>0).sum()),
                "interaction_improved_reset_count":int((gain>0).sum()),
                "belief_revision_signed_recovery":_ratio(float(revision_dot.sum()),total_target),
                "belief_revision_signed_recovery_ci95":_safe_bootstrap_ratio(revision_dot,target,draws),
                "paired_endpoint_mse_gain_mean":float(endpoint.mean()),
                "paired_endpoint_mse_gain_mean_ci95":_ci(endpoint[draws].mean(-1)),
                "paired_endpoint_relative_mse_gain":_ratio(float(endpoint.sum()),float(endpoint_before.sum())),
                "paired_endpoint_relative_mse_gain_ci95":_safe_bootstrap_ratio(endpoint,endpoint_before,draws),
                "endpoint_mse_gain_by_speed_program":endpoint_delta.mean(0).tolist(),
                "endpoint_mse_gain_by_speed_program_ci95":np.quantile(endpoint_delta[draws].mean(1),[.025,.975],axis=0).tolist(),
                "paired_endpoint_improved_reset_count":int((endpoint>0).sum()),
                "paired_movement_improved_reset_count":sum(r["paired_movement_mse_gain"]>0 for r in comparisons),
                "paired_stop_improved_reset_count":sum(r["paired_stop_mse_gain"]>0 for r in comparisons),
                "leave_one_out_belief_revision_signed_recovery":loo_recovery,
                "leave_one_out_interaction_relative_gain":loo_gain,
                "leave_one_out_endpoint_mse_gain_mean":[float(np.delete(endpoint,i).mean()) if len(endpoint)>1 else None for i in range(len(endpoint))],
                "max_gain_identity_absolute_error":max(r["gain_identity_absolute_error"] for r in comparisons)}
        result["horizons"][h] = group
    if with_directions:
        result["by_probe_direction"] = {name:summarize([r for r in rows if r["probe_direction"]==name],with_directions=False)
            for name in sorted({r["probe_direction"] for r in rows})}
    return result


def gate(summary):
    """Separate every predeclared condition; never select the stronger arm."""
    count = summary["independent_reset_count"]
    checks = {}
    for h in ("1","4"):
        for name in ("single_vs_uniform","all_vs_uniform"):
            if not count:
                checks[f"H{h}_{name}"] = {"passed":False,"reason":"No matched physical-nonzero resets"}
                continue
            row = summary["horizons"][h]["comparisons"][name]
            criteria = {"positive_signed_revision":row["belief_revision_signed_recovery"] is not None and row["belief_revision_signed_recovery"]>0,
                "positive_interaction_error_gain":row["interaction_squared_error_gain_sum"]>0,
                "majority_interaction_improvements":row["interaction_improved_reset_count"]>count/2}
            checks[f"H{h}_{name}"] = {**criteria,"passed":all(criteria.values()),
                "separate_endpoint_diagnostics":{"positive_endpoint_mse_gain":row["paired_endpoint_mse_gain_mean"]>0,
                    "majority_endpoint_improvements":row["paired_endpoint_improved_reset_count"]>count/2}}
    return {"passed":bool(count) and all(r["passed"] for r in checks.values()),"checks":checks,
            "primary_arm_prespecified":"episode_dual","primary_paired_metric":"per-reset cross-action interaction squared-error gain",
            "endpoint_improvements_reported_separately_not_gate":True,"CIs_are_reported_not_an_added_gate":True}


def _diagnostic_positions(path):
    with np.load(path,allow_pickle=False) as archive:
        for key in ("positions","dev_positions","validation_positions"):
            if key in archive:
                return torch.from_numpy(archive[key].copy()).double()
    raise ValueError("Physical diagnostics require positions[S,2,2,51,2]")


@torch.no_grad()
def evaluate(checkpoint,data,diagnostics):
    started = perf_counter()
    batch = torch.load(data,map_location="cpu",weights_only=True)
    fitted = torch.load(checkpoint,map_location="cpu",weights_only=True)
    z,a = batch["z"],batch["a"]
    physical = _diagnostic_positions(diagnostics)
    if z.shape!=(len(z),2,2,11,192) or a.shape!=(len(z),2,2,10,192) or physical.shape!=(len(z),2,2,51,2):
        raise ValueError("Development requires z[S,2,2,11,192],a[S,2,2,10,192],physical[S,2,2,51,2]")
    if not torch.equal(batch["null_control"],fitted["null_control"]):
        raise ValueError("Model and evaluation must use the same canonical raw-zero native embedding")
    if not torch.equal(fitted["transition"],torch.eye(4,dtype=torch.float64)):
        raise ValueError("This fixed-speed stage requires T=I")
    if fitted.get("observation_law")!="continuous_bounded_uniform_nu3":
        raise ValueError("Evaluation requires the G1 continuous bounded uniform-reference law")
    seeds = batch["seed"].tolist()
    if len(set(seeds))!=len(seeds) or set(seeds)&set(fitted.get("training_seeds",[])):
        raise ValueError("Evaluation reset seeds must be unique and disjoint from training")
    model = NativePrototypes(load_host("cpu"),null_control=batch["null_control"]).eval().requires_grad_(False)
    model.offsets.copy_(fitted["offsets"])
    rows,excluded,tensors = [],[],[]
    for index,seed in enumerate(seeds):
        states,controls,positions = z[index],a[index],physical[index]
        root,commands = states[:,:,4:7],controls[:,:,4:6]
        checks = {"all_endpoints_complete":bool((batch["lengths"][index]>=10).all()),
            "actual_rgb_matched":bool(batch["matched_rgb"][index]),
            "native_root_exactly_matched":torch.equal(root,root[:1,:1].expand_as(root)),
            "recent_controls_exactly_matched":torch.equal(commands,commands[:1,:1].expand_as(commands)),
            "same_program_controls_equal_across_speeds":torch.equal(controls[0],controls[1]),
            "physical_root_exactly_matched":torch.equal(positions[:,:,[20,25,30]],positions[:1,:1,[20,25,30]].expand(2,2,-1,-1)),
            "program_actual_prefix_exactly_replayed":torch.equal(states[:,0,:7],states[:,1,:7]) and torch.equal(controls[:,0,:6],controls[:,1,:6]),
            "raw_zero_references_exact":torch.equal(controls[:,:,4:6],batch["null_control"].expand(2,2,2,-1))}
        if not all(checks.values()):
            excluded.append({"seed":seed,"checks":checks,"reasons":[k for k,v in checks.items() if not v],
                             "native_root_max_abs_difference":float((root-root[:1,:1]).abs().max())})
            continue
        if not torch.isfinite(states).all() or not torch.isfinite(controls).all() or not torch.isfinite(positions).all():
            raise ValueError(f"Eligible seed {seed} has nonfinite actual input or physical diagnostic")
        for t in (4,5):
            if not torch.equal(positions[:,:,5*t],positions[:,:,5*(t+1)]) or not torch.equal(states[:,:,t],states[:,:,t+1]):
                raise ValueError(f"Pinned no-op in seed {seed} changed the physical position or actual latent")
        pulse = positions[0,0,15]-positions[0,0,10]
        if not (pulse!=0).any():
            direction="zero"
        else:
            axis=int(pulse.abs().argmax())
            direction=("left","right")[(pulse[0]>0).item()] if axis==0 else ("up","down")[(pulse[1]>0).item()]
        predicted = forecast_prefix(model,fitted,states[:,:,:7],controls)
        if not torch.equal(predicted["future_responses"][0],predicted["future_responses"][1]):
            raise AssertionError("Matched windows and commands did not produce the same native future family")
        row = score_case(seed,states,positions,predicted,direction)
        row["eligibility_checks"]=checks
        row["future_family_exactly_shared_across_speeds"]=True
        rows.append(row)
        tensors.append({"seed":seed,"supports":predicted["supports"],"future_responses":predicted["future_responses"],
                        "predictions":predicted["predictions"],"true_future":states[:,:,7:11]})
    panels={"all_matched":summarize(rows),
            "physical_nonzero":summarize([r for r in rows if r["horizons"]["1"]["physical_nonzero"]]),
            "physical_zero":summarize([r for r in rows if not r["horizons"]["1"]["physical_nonzero"]])}
    report={"checkpoint":str(Path(checkpoint).resolve()),"objective":fitted.get("objective","unspecified"),
        "data":str(Path(data).resolve()),"physical_diagnostics":str(Path(diagnostics).resolve()),
        "forecast_boundary":"actual z0..z6 only; future z7..z10 used only in score_case",
        "observation_law":"G1 continuous bounded nu3, fixed uniform response reference",
        "fixed_timing":"t2 source -> actual z3; t3 reverse -> actual z4; t4,t5 zero -> actual z5,z6; t6 orthogonal/zero -> H1..H4",
        "bootstrap":{"unit":"paired independent reset seed","draws":BOOTSTRAP_DRAWS,"seed":BOOTSTRAP_SEED,
                     "interval":"percentile 95%; ratios use pooled numerator/denominator in each resample"},
        "coverage":{"offered_reset_count":len(seeds),"matched_reset_count":len(rows),"exclusions":excluded},
        "panels":panels,"seeds":rows,"gate":gate(panels["physical_nonzero"]),
        "precision":{"native":"FP32 CPU","belief_and_scoring":"FP64","no_op_observation_updates_exact":True,
            "mixture":"F_nominal + sum_k w_k(F_k-F_nominal), preserving exact common response",
            "unique_actual_noop_arrival_count":4*len(rows),"chronology_noop_check_count":8*len(rows),
            "max_public_noop_normalization_change":max((n["public_max_abs_change"] for r in rows for n in r["no_op_checks"]),default=0.)},
        "elapsed_seconds":perf_counter()-started,"no_G4_or_P4_execution":True}
    return report,{"cases":tensors}


def render_report(report):
    summary=report["panels"]["physical_nonzero"]
    lines=[f"# {report['objective']} fixed-history evaluation","",
        f"Matched resets: {report['coverage']['matched_reset_count']}/{report['coverage']['offered_reset_count']}; physical nonzero: {summary['independent_reset_count']}. Exclusions are retained in JSON.","",
        "| H | Comparison | Signed revision | Interaction error gain | Improved interaction resets | Paired endpoint MSE gain | Improved endpoint resets |",
        "|---:|---|---:|---:|---:|---:|---:|"]
    for h in ("1","4"):
        if not summary["independent_reset_count"]:
            continue
        for name in COMPARISONS:
            row=summary["horizons"][h]["comparisons"][name]
            recovery=row["belief_revision_signed_recovery"]
            relative=row["interaction_relative_squared_error_gain"]
            recovery_text=_display_interval(recovery,row["belief_revision_signed_recovery_ci95"],percent=True)
            gain_text=_display_interval(relative,row["interaction_relative_squared_error_gain_ci95"],percent=True)
            endpoint_text=_display_interval(row["paired_endpoint_mse_gain_mean"],row["paired_endpoint_mse_gain_mean_ci95"])
            lines.append(f"| {h} | {name} | {recovery_text} | {gain_text} | {row['interaction_improved_reset_count']}/{summary['independent_reset_count']} | {endpoint_text} | {row['paired_endpoint_improved_reset_count']}/{summary['independent_reset_count']} |")
    lines += ["",f"G3 gate: {'PASS' if report['gate']['passed'] else 'FAIL'}. The primary arm is prespecified as episode_dual. Every gate component is recorded; confidence intervals do not introduce another gate.","",
        "Brackets show 95% percentile intervals from 4,000 paired-reset bootstrap draws, seed 20261120. Leave-one-reset-out values, every root's direction/sign, absolute move/stop and speed errors, stationary traces and all-minus-omit increments are retained in JSON. Physical-zero and all-matched panels are reported separately. Only actual arrivals update belief; all future paths own their predicted windows.",""]
    if summary["independent_reset_count"]:
        lines += ["| H | Strategy | Movement endpoint MSE | Stop endpoint MSE |",
                  "|---:|---|---:|---:|"]
        for h in ("1","4"):
            for name in STRATEGIES:
                move,stop=summary["horizons"][h]["strategies"][name]["absolute_endpoint_mse_by_program"]
                lines.append(f"| {h} | {name} | {move:.8f} | {stop:.8f} |")
        lines += ["", "| Panel | Resets | H1 full interaction SSE gain | H4 full interaction SSE gain |",
                  "|---|---:|---:|---:|"]
        for name,panel in report["panels"].items():
            count=panel["independent_reset_count"]
            if count:
                h1,h4=(panel["horizons"][h]["comparisons"]["all_vs_uniform"]["interaction_squared_error_gain_sum"] for h in ("1","4"))
                lines.append(f"| {name} | {count} | {h1:+.8f} | {h4:+.8f} |")
        lines += ["",f"Every zero-action observation is an exact public no-write: {report['precision']['unique_actual_noop_arrival_count']} unique actual arrivals, {report['precision']['chronology_noop_check_count']} chronology checks, maximum public weight change {report['precision']['max_public_noop_normalization_change']:.3g}.",""]
    return "\n".join(lines)


def _display_interval(value,interval,*,percent=False):
    if value is None:
        return "undefined"
    scale,digits,suffix=(100,4,"%") if percent else (1,8,"")
    point=f"{scale*value:+.{digits}f}{suffix}"
    if interval is None:
        return point
    return point+f" [{scale*interval[0]:+.{digits}f}, {scale*interval[1]:+.{digits}f}]"


def codechecks():
    before=torch.tensor([.1,-.3],dtype=torch.float64)
    target=torch.tensor([.8,.2],dtype=torch.float64)
    after=torch.tensor([.4,-.1],dtype=torch.float64)
    delta=_difference(after,before,target)
    assert delta["squared_error_gain"]>0 and delta["gain_identity_absolute_error"]<1e-15
    assert _difference(before,before,target)["squared_error_gain"]==0
    numerator=np.array([1.,3.,5.]); denominator=np.array([2.,4.,6.])
    draws=np.array([[0,1,2],[2,1,0]])
    np.testing.assert_allclose(_safe_bootstrap_ratio(numerator,denominator,draws),[.75,.75],atol=0,rtol=0)
    common=torch.tensor([1e6,-2e6],dtype=torch.float64)
    weights=torch.tensor([.07,.13,.29,.51],dtype=torch.float64)
    identical=common.expand(4,-1)
    torch.testing.assert_close(common+weights@(identical-common),common,atol=0,rtol=0)
    z=torch.zeros(2,2,11,192,dtype=torch.float64)
    physical=torch.zeros(2,2,51,2,dtype=torch.float64)
    z[0,0,7:]=1;z[1,0,7:]=2
    physical[0,0,35:,0]=1;physical[1,0,35:,0]=2
    predictions={name:z[:,:,7:].clone() for name in STRATEGIES}
    predictions["uniform"].zero_(); predictions["nominal"].zero_();predictions["omit_source"].zero_()
    row=score_case(1,z,physical,{"predictions":predictions,"supports":{name:weights.expand(2,-1) for name in STRATEGIES},
        "actual_write_traces":[],"no_op_checks":[]},"right")
    checked=summarize([row],with_directions=False)
    assert checked["horizons"]["1"]["comparisons"]["single_vs_uniform"]["interaction_relative_squared_error_gain"]==1.
    assert checked["horizons"]["4"]["comparisons"]["all_vs_uniform"]["paired_endpoint_mse_gain_mean"]>0
    assert gate(checked)["passed"]
    return {"passed":True,"gain_identity":delta["gain_identity_absolute_error"],
            "paired_bootstrap_uses_pooled_ratio":True,"common_response_mixture_exact":True,
            "perfect_transfer_scoring_and_gate_fixture":True,"forecast_accepts_no_future_target":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path)
    parser.add_argument("--data",type=Path,default=OUT/"dev.pt")
    parser.add_argument("--diagnostics",type=Path,default=OUT/"diag_dev.npz")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--check",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(2)
    if args.check:
        checked=codechecks()
        OUT.mkdir(parents=True,exist_ok=True)
        (output_path(OUT/"g3_measure_checks.json")).write_text(json.dumps(checked,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(checked),flush=True)
        return
    if args.checkpoint is None:
        parser.error("--checkpoint is required for an evaluation")
    report,tensors=evaluate(args.checkpoint,args.data,args.diagnostics)
    output=output_path(args.output or args.checkpoint.with_name(args.checkpoint.stem+"_measure.json"))
    output_path(output).write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    output_path(output.with_suffix(".md")).write_text(render_report(report),encoding="utf-8")
    torch.save(tensors,output_path(output.with_name(output.stem+"_tensors.pt")))
    print(json.dumps({"output":str(output),"gate":report["gate"],"seconds":report["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
