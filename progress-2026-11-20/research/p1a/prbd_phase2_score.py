"""Offline scoring only. All methods issue and save predictions before this step."""
import json

import numpy as np
import torch

from .prbd_phase2 import OUT, fixed_transition
from .prbd_paths import ROOT, output_path
from .prbd_phase2_measure import transport, tv


BASELINES = ("frozen", "gru", "adajepa", "optta")


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def save(name, value, output_dir=OUT):
    output_path(output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def paired_mean(x):
    x = np.asarray(x, dtype=np.float64)
    if not len(x):
        return None
    draws = np.random.default_rng(20261121).integers(len(x), size=(4000, len(x)))
    return {"mean":float(x.mean()), "ci95":np.percentile(x[draws].mean(1), [2.5, 97.5]).tolist(),
            "positive_roots":int((x>0).sum()), "roots":len(x)}


def predictions(split, seed, predictions_dir=OUT, data_dir=OUT):
    models = {}
    package = load(predictions_dir / f"pred_{split}_prbd.pt")
    if not torch.equal(package["seed"], seed):
        raise ValueError("PRBD evaluation root order differs")
    if split == "static":
        models.update(package["predictions"])
    else:
        models["full"] = package["predictions"]
        models["uniform"] = package["uniform"]
    for method in BASELINES:
        pkg = load(data_dir / "baselines" / f"{split}_{method}.pt")
        if not torch.equal(pkg["seed"], seed):
            raise ValueError(f"Different evaluation roots for {method}")
        models[method] = pkg["predictions"].double()
    return models


def contrast(x):
    return (x[:, 0, 0]-x[:, 0, 1]) - (x[:, 1, 0]-x[:, 1, 1])


def static_masks(batch, pos, raw):
    matched, included, exclusions = [], [], []
    for i, seed in enumerate(batch["seed"].tolist()):
        z, a, p = batch["z"][i], batch["a"][i], pos[i]
        checks = dict(complete=bool((batch["lengths"][i]>=10).all()), rgb=bool(batch["matched_rgb"][i]),
            window=bool((z[:,:,4:7]==z[:1,:1,4:7]).all()), controls=bool((a[:,:,4:6]==a[:1,:1,4:6]).all()),
            physical_root=bool((p[:,:,[20,25,30]]==p[:1,:1,[20,25,30]]).all()),
            replay=bool((z[:,0,:7]==z[:,1,:7]).all() and (a[:,0,:6]==a[:,1,:6]).all()),
            shared_menu=bool((raw[i,0]==raw[i,1]).all() and (a[0]==a[1]).all()))
        if all(checks.values()):
            matched.append(i)
            if torch.any(contrast(pos[i:i+1,:,:,35]-pos[i:i+1,:,:,30])):
                included.append(i)
            else:
                exclusions.append([seed, ["physical_interaction_zero"]])
        else:
            exclusions.append([seed, [k for k,v in checks.items() if not v]])
    if not included:
        raise ValueError("No complete, nonzero matched static roots")
    return matched, included, exclusions


def static_panel(truth, models, seeds):
    truth = truth.double()
    if not all(p.shape == truth.shape and torch.isfinite(p).all() for p in models.values()):
        raise ValueError("Missing or nonfinite static forecast")
    delta = contrast(truth)
    family = {k:contrast(v) for k,v in models.items()}
    errors = {k:(v-delta).square().sum(-1) for k,v in family.items()}
    by_h, per_root = {}, {}
    for h in range(4):
        dh = delta[:,h]
        den = dh.square().sum()
        rows = {}
        for name, v in family.items():
            reference = errors["uniform"][:,h]
            err = errors[name][:,h]
            gain = reference - err
            mse = (models[name][:,:,:,h]-truth[:,:,:,h]).square().mean((0,1,3))
            rows[name] = {
                "directed_percent":float(100*((v[:,h]-family["uniform"][:,h])*dh).sum()/den) if den else None,
                "interaction_sse_mean":float(err.mean()),
                "sse_gain_percent":float(100*gain.sum()/reference.sum()) if reference.sum() else None,
                "improved":int((gain>0).sum()), "mse_move_stop":mse.tolist(),
                "gain_over_uniform":paired_mean(gain), "prbd_gain_over_method":paired_mean(err-errors["full"][:,h])}
        rows["full_vs_omit"]={"directed_percent":float(100*((family["full"][:,h]-family["omit_first"][:,h])*dh).sum()/den) if den else None,
                              "sse_gain":paired_mean(errors["omit_first"][:,h]-errors["full"][:,h])}
        by_h[str(h+1)] = rows
        per_root[str(h+1)] = dict(seed=seeds.tolist(), truth_energy=dh.square().sum(-1).tolist(),
            methods={name:dict(sse=errors[name][:,h].tolist(), dot_delta_from_uniform=
                ((v[:,h]-family["uniform"][:,h])*dh).sum(-1).tolist()) for name,v in family.items()})
    return {"n":len(seeds), "horizon":by_h}, per_root


def writes(records, indices, lengths, positions, seeds, *, inspect_mass=False):
    vals = {k:[] for k in ("transport", "write", "total", "no_deadzone", "separation", "at_uniform")}
    exact, silent = 0, 0
    examples = []
    log_leaders, nominal_changes = [], []
    for t in (4,5):
        rec = records[t-2]
        # Two candidate futures replay the identical source prefix. Count it once.
        stable = (positions[indices,:,0,5*t:5*(t+1)+1]==positions[indices,:,0,5*t:5*t+1]).flatten(2).all(-1)
        valid = (lengths[indices,:,0]>t) & stable
        def field(k):
            return rec[k][indices,:,0][valid]
        w, prior, post = (field(k) for k in ("w", "prior", "posterior"))
        log_leaders.extend(field("relative_logs").argmax(-1).tolist())
        nominal_changes.extend((post[:,0]-prior[:,0]).tolist())
        at_uniform = field("relative_logs").softmax(-1)
        uniform = torch.full_like(at_uniform, .25)
        for k,x in (("transport",tv(prior,w)), ("write",tv(prior,post)), ("total",tv(w,post)),
                    ("no_deadzone",tv(prior,field("raw_posterior"))), ("separation",field("separation")),
                    ("at_uniform",torch.where(field("silent"), 0., tv(uniform, at_uniform)))):
            vals[k].extend(x.tolist())
        for j,(root,speed) in enumerate(torch.nonzero(valid).tolist()):
            examples.append(dict(seed=int(seeds[indices[root]]), speed=[2.5,7.5][speed], transition=t,
                prior=prior[j].tolist(), posterior=post[j].tolist(), tv=float(tv(prior[j],post[j])),
                separation=float(field("separation")[j])))
        exact += int((post==prior).all(-1).sum())
        silent += int(field("silent").sum())
    v = np.array(vals["write"])
    start, final = records[2]["w"][indices,:,0], records[3]["posterior"][indices,:,0]
    no_write=transport(transport(start, fixed_transition()), fixed_transition())
    both=(lengths[indices,:,0]>=6)&(positions[indices,:,0,20:31]==positions[indices,:,0,20:21]).flatten(2).all(-1)
    cumulative=tv(final[both], no_write[both]).numpy()
    result = {"n":len(v), "exact":exact, "near":int((v<=.01).sum()), "over_half":int((v>.5).sum()),
            "highest_likelihood_by_branch":np.bincount(log_leaders,minlength=4).tolist(),
            "nominal_mass_increased":int((np.array(nominal_changes)>0).sum()),
            "mean_nominal_mass_change":float(np.mean(nominal_changes)),
            "strongest_write":max(examples,key=lambda e:e["tv"]),
            "two_quiet_actual_vs_transport_only":{"n":len(cumulative), "mean":float(cumulative.mean()),
                                                  "p95":float(np.quantile(cumulative,.95)), "max":float(cumulative.max())},
            "deadzone":silent, "statistics":{k:dict(mean=float(np.mean(x)), p95=float(np.quantile(x,.95)),
                                                maximum=float(max(x))) for k,x in vals.items()}}
    if inspect_mass:
        # Total moved probability mass comprises net nominal transfer plus
        # any simultaneous reallocation between the three learned branches.
        nominal_transfer = np.abs(nominal_changes)
        if (v + 1e-12 < nominal_transfer).any():
            raise ValueError("Nominal mass transfer exceeds whole posterior TV")
        result["mean_abs_nominal_mass_change"] = float(nominal_transfer.mean())
        result["mean_remaining_redistribution_tv"] = float((v - nominal_transfer).mean())
    return result


def static(data_dir=OUT, predictions_dir=OUT, output_dir=OUT, *, inspect_mass=False):
    batch = load(data_dir/"static.pt")
    with np.load(data_dir/"diag_static.npz") as data:
        pos = torch.from_numpy(data["positions"].copy()).double()
    with np.load(data_dir/"data_static.npz") as data:
        matched, indices, exclusions = static_masks(batch, pos, data["actions"])
    models = predictions("static", batch["seed"], predictions_dir, data_dir)
    all_results = {"offered":len(batch["seed"]), "matched":len(matched), "physical_nonzero":len(indices),
                   "exclusions":exclusions, "probes":{}}
    for name, rows in (("matched",matched), ("physical_nonzero",indices)):
        truth = batch["z"][rows,:,:,7:11,:]
        panel, audit = static_panel(truth, {k:p[rows] for k,p in models.items()}, batch["seed"][rows])
        all_results[name+"_panel"] = panel
        save("static_"+name+"_roots.json",audit,output_dir)
    # Historical position readout was fitted on author HDF5 only. It is an evaluator.
    coef=load(ROOT/"runs/prbd_phase1_20261121"/"official_xy_readout.pt")["coef"].double()
    physical_truth=pos[indices,:,:,35:51:5]
    physical={k:v[indices].double()@coef[:-1]+coef[-1] for k,v in models.items()}
    all_results["physical_xy_panel"], _ = static_panel(physical_truth, physical, batch["seed"][indices])
    for stage, file in (("initial","initial_static_writes.pt"), ("trained","pred_static_prbd.pt")):
        records = load(predictions_dir/file)["records"]
        all_results["probes"][stage] = {
            group:writes(records, rows, batch["lengths"], pos, batch["seed"], inspect_mass=inspect_mass)
            for group,rows in (("all",list(range(len(batch["seed"])))), ("matched",matched), ("physical_nonzero",indices))}
    save("static_results.json",all_results,output_dir)
    return all_results


def recovered(times, correct, cut):
    for i in range(1,len(times)):
        if correct[i-1] and correct[i]:
            return int(times[i-1]-cut), int(times[i]-cut)
    return None, None


def switches(data_dir=OUT, predictions_dir=OUT, output_dir=OUT):
    batch = load(data_dir/"switch.pt")
    cf = load(data_dir/"switch_counterfactual.pt")
    if not torch.equal(batch["seed"], cf["seed"]):
        raise ValueError("Switch judge root mismatch")
    pairs = cf["z_cf"].double()
    with np.load(data_dir/"diag_switch_counterfactual.npz") as data:
        informative = torch.tensor(data["differing_rgb"])
    with np.load(data_dir/"diag_switch.npz") as data:
        speed=torch.tensor(data["speed"])
        cuts=data["switch_blocks"].copy()
    # Only factual full blocks shared by every method are scored; informative requires two complete alternatives.
    truth = batch["z"][...,1:,:].double()
    lens = batch["lengths"]
    clock=torch.arange(truth.size(-2))
    valid=(clock>=2)&(clock<lens[...,None])
    informative &= valid & ((pairs[:,:,:,0]!=pairs[:,:,:,1]).any(-1))
    models=predictions("switch",batch["seed"],predictions_dir,data_dir)
    results, errors = {}, {}
    for name, pred in models.items():
        if pred.shape!=truth.shape or not torch.isfinite(pred[valid]).all():
            raise ValueError(f"Missing legal switch forecasts for {name}")
        mse=(pred-truth).square().mean(-1)
        distances=(pred[...,None,:]-pairs).square().mean(-1)
        correct=torch.where(speed==2.5,distances[...,0]<distances[...,1],distances[...,1]<distances[...,0])
        events=[]
        for i,seed in enumerate(batch["seed"].tolist()):
            for arm in range(2):
                c1,c2=map(int,cuts[i,arm])
                for left,t,end in ((2,c1,c2),(c1,c2,len(clock))):
                    direction="to_fast" if ((arm==0)==(t==c1)) else "to_slow"
                    times=clock[informative[i,arm] & (clock>=t) & (clock<end)]
                    past=clock[informative[i,arm] & (clock>=left) & (clock<t)]
                    r,confirm=recovered(times.tolist(),correct[i,arm,times].tolist(),t)
                    evaluated=clock[valid[i,arm] & (clock>=t) & (clock<end)]
                    events.append(dict(seed=seed, arm=arm, direction=direction, cut=t,
                        executed_blocks=len(evaluated), informative=len(times), correct=int(correct[i,arm,times].sum()),
                        pre_correct=bool(len(past)>=2 and correct[i,arm,past[-2:]].all()), lag=r, confirmed_lag=confirm,
                        sse_per_dimension=float(mse[i,arm,evaluated].sum())))
        errors[name]={}
        tables={}
        for direction in ("to_slow","to_fast"):
            group=[ev for ev in events if ev["direction"]==direction]
            for prefix, subgroup in (("",group), ("pre_correct_",[e for e in group if e["pre_correct"]])):
                identifiable=[e for e in subgroup if e["informative"]>=2]
                ok=[e for e in identifiable if e["lag"] is not None]
                blocks=sum(e["executed_blocks"] for e in subgroup)
                tables[prefix+direction]=dict(offered=len(subgroup), opportunities=len(identifiable),
                    one_opportunity=sum(e["informative"]==1 for e in subgroup), no_opportunities=sum(e["informative"]==0 for e in subgroup),
                    recovered=len(ok), censored=len(identifiable)-len(ok), blocks=blocks,
                    median_lag=float(np.median([e["lag"] for e in ok])) if ok else None,
                    median_confirmation=float(np.median([e["confirmed_lag"] for e in ok])) if ok else None,
                    lag_zero=sum(e["lag"]==0 for e in ok),
                    correct=sum(e["correct"] for e in subgroup), informative=sum(e["informative"] for e in subgroup),
                    mse=sum(e["sse_per_dimension"] for e in subgroup)/blocks if blocks else None)
            errors[name][direction]={}
            for seed in batch["seed"].tolist():
                own=[e for e in group if e["seed"]==seed]
                n=sum(e["executed_blocks"] for e in own)
                if n:
                    errors[name][direction][seed]=sum(e["sse_per_dimension"] for e in own)/n
        results[name]={"summary":tables,"overall_mse":float(mse[valid].mean())}
        save("switch_"+name+"_events.json",events,output_dir)
    for name in models:
        for direction in ("to_slow","to_fast"):
            ref=errors[name][direction]
            results[name]["summary"][direction]["prbd_gain_over_method"]=paired_mean(
                [ref[s]-errors["full"][direction][s] for s in ref])
    save("switch_results.json",results,output_dir)
    return results


if __name__=="__main__":
    torch.set_num_threads(3)
    static()
    switches()
