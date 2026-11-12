"""Native AdaLN interaction and evidence from four executed context/action cells."""

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from native_gate_bound_paired import CALIBRATION, array, load_host
from native_gate_bound_pusht import decode, physical_errors
from native_joint_cycle_data import cycle_bank
from query_feedback_learning import write_csv, write_json


MODES = (0., -1., 1.)  # The first minimizer preserves native behavior on ties.
WRITERS = (3, 1)       # Stop, +60 degrees, executed at each context.
QUERIES = (0, 2)       # Straight, -60 degrees, never used for the write.
TRAIN_STEPS = 64


def rectangle(values):
    if values.shape[1:3] != (2, 2):
        raise ValueError("The writer needs two completed actions at each of two contexts")
    return (values[:, 1, 1]-values[:, 1, 0])-(values[:, 0, 1]-values[:, 0, 0])


@dataclass(frozen=True)
class Evidence:
    cost: torch.Tensor
    completed: int = 0

    def observe(self, actual_cells, nominal_cells, corner_effect):
        target = rectangle(actual_cells.double()-nominal_cells.double())
        error = (target[:, None]-corner_effect.double()).square().sum(-1)
        return Evidence(self.cost+error, self.completed+1)

    def mode(self):
        return self.cost.argmin(-1)


def analytic_example():
    nominal = torch.tensor([[[[0.,0.],[1.,1.]],[[2.,0.],[5.,1.]]]], dtype=torch.float64)
    effect = torch.tensor([[[0.,0.],[-2.,0.],[2.,0.]]], dtype=torch.float64)
    context = torch.tensor([[[[7.,-9.]],[[24.,1.]]]], dtype=torch.float64)
    command = torch.tensor([[[[5.,4.],[-6.,2.]]]], dtype=torch.float64)
    target = nominal + context + command
    target[:, 1, 1] += effect[:, 1]
    empty = Evidence(torch.zeros(1,3,dtype=torch.float64))
    old = empty.observe(target, nominal, effect)
    assert int(old.mode()) == 1 and int(empty.mode()) == 0
    torch.testing.assert_close(rectangle(target-nominal), effect[:, 1], atol=0, rtol=0)
    pair = ((target-nominal)[:, 1, 1]-(target-nominal)[:, 0, 1])
    assert int((pair[:, None]-effect).square().sum(-1).argmin(-1)) != int(old.mode())
    strong_new = nominal + context + command
    strong_new[:, 1, 1] += 3*effect[:, 2]
    revised = old.observe(strong_new, nominal, 3*effect)
    assert int(revised.mode()) == 2 and int(old.mode()) == 1
    try:
        empty.observe(target[:, :1], nominal[:, :1], effect)
    except ValueError:
        pass
    else:
        raise AssertionError("An incomplete set of executed cells was accepted")
    # A nuisance that appears only in cell 11 can reproduce exactly the joint change.
    return dict(additive_nuisance_cancelled=True, incomplete_cycle_rejected=True, later_cycle_revises=True,
                same_command_pair_confounded=True, joint_nuisance_is_unidentifiable=True)


class NativeJointPort:
    def __init__(self, host, moments):
        self.host = host
        self.moments = moments
        self.predictor = host.predictor
        self.transformer = self.predictor.transformer
        self.block = self.transformer.layers[0]
        self.channel_mask = None
        zero = moments[0].new_zeros((1, 1, 5, 2))
        c0 = self.transformer.cond_proj(self.embed(zero))
        self.shift0, self.scale0 = self.block.adaLN_modulation(c0).chunk(6, dim=-1)[:2]

    def embed(self, raw):
        return self.host.action_encoder(((raw-self.moments[0])/self.moments[1]).flatten(-2))

    def predict(self, z, c, anchor, strength, zero_action, *, trace=False):
        if not trace and bool((strength == 0).all()):
            return self.host.predict(z, c), {}
        block = self.block
        reference = block.norm1(self.transformer.input_proj(self.predictor.dropout(
            anchor + self.predictor.pos_embedding[0, z.shape[1]-1])))
        match = (z[:, -1] == anchor).all(-1) | zero_action | (strength == 0.)
        store = {}

        def keep_v(_, args, result):
            store["v"] = result[:, -1]

        def keep_scale(_, args, result):
            store["shift"], store["scale"] = (value[:, -1] for value in result.chunk(6, dim=-1)[:2])

        def modify(_, args):
            native = args[0]
            j = (store["v"]-reference)*(store["scale"]-self.scale0[0, 0])
            effect = j if self.channel_mask is None else j*self.channel_mask
            current = torch.where(match[:, None], native[:, -1], native[:, -1]+strength[:, None]*effect)
            edited = torch.cat((native[:, :-1], current[:, None]), dim=1)
            if trace:
                store.update(reference=reference, j=j, input=native, edited=edited, pass_mask=match)
            return (edited, *args[1:])

        with ExitStack() as hooks:
            hooks.callback(block.norm1.register_forward_hook(keep_v).remove)
            hooks.callback(block.adaLN_modulation.register_forward_hook(keep_scale).remove)
            hooks.callback(block.attn.register_forward_pre_hook(modify).remove)
            if trace:
                hooks.callback(block.attn.norm.register_forward_hook(
                    lambda _, args, y: store.update(norm=y)).remove)
                hooks.callback(block.attn.to_qkv.register_forward_hook(
                    lambda _, args, y: store.update(qkv=y)).remove)
            out = self.host.predict(z, c)
        return out, store if trace else {}


def requests(port, bank, context, actions):
    device = port.moments[0].device
    n, c, a = len(bank["scene_id"]), len(context), len(actions)
    z = torch.as_tensor(bank["history"][:, context], dtype=torch.float32, device=device)
    z = z[:, :, None].expand(n,c,a,3,192).reshape(-1,3,192)
    raw = torch.as_tensor(bank["commands"][:, context][:, :, actions], dtype=torch.float32, device=device)
    raw = raw.reshape(-1,3,5,2)
    anchor = torch.as_tensor(bank["history"][:, 0, -1], dtype=torch.float32, device=device)
    anchor = anchor.repeat_interleave(c*a, dim=0)
    return z, port.embed(raw), anchor, (raw[:, -1].flatten(1) == 0.).all(1)


def predict_requests(port, request, strength, *, trace=False):
    z,c,anchor,zero = request
    return port.predict(z, c, anchor, strength, zero, trace=trace)


def interface(port, bank):
    small = {key: value[:8] for key,value in bank.items()}
    data = requests(port, small, [0,1], list(WRITERS))
    empty = data[0].new_zeros(len(data[0]))
    base, origin = predict_requests(port, data, empty, trace=True)
    down, minus = predict_requests(port, data, empty-1., trace=True)
    up, plus = predict_requests(port, data, empty+1., trace=True)
    native = port.host.predict(data[0],data[1])
    torch.testing.assert_close(base,native,atol=0,rtol=0)
    anchored = minus["pass_mask"]
    for output, trace in ((down,minus),(up,plus)):
        torch.testing.assert_close(output[anchored],base[anchored],atol=0,rtol=0)
        torch.testing.assert_close(trace["qkv"][:,:-1],origin["qkv"][:,:-1],atol=0,rtol=0)
        torch.testing.assert_close(output[:,:-1],base[:,:-1],atol=2e-5,rtol=0)
    v, r, s, m = (minus[key] for key in ("v","reference","scale","shift"))
    zero_s,zero_m = port.scale0[0,0],port.shift0[0,0]
    def a(x, scale, shift):
        return x*(1+scale)+shift
    explicit = a(v,s,m)-a(v,zero_s,zero_m)-a(r,s,m)+a(r,zero_s,zero_m)
    torch.testing.assert_close(minus["j"],explicit,atol=2e-6,rtol=0)
    torch.testing.assert_close(minus["edited"][~anchored,-1], (a(v,zero_s,zero_m)+a(r,s,m)-a(r,zero_s,zero_m))[~anchored],
                               atol=2e-6,rtol=0)
    return dict(requests=len(base), anchored=int(anchored.sum()),
                rectangle_max_abs=float((minus["j"]-explicit).abs().max()),
                post_attention_norm_delta=float((minus["norm"]-origin["norm"]).abs().max()),
                last_latent_delta=float((down-base).abs().max()),
                midpoint_defect=float((base-(down+up)/2).abs().max()),
                prefix_output_max_abs=float((down[:,:-1]-base[:,:-1]).abs().max()))


def facts_and_queries(port, bank):
    n = len(bank["scene_id"])
    fact_request = requests(port,bank,[0,1],list(WRITERS))
    zeros = fact_request[0].new_zeros(4*n)
    facts = predict_requests(port,fact_request,zeros)[0][:,-1].reshape(n,2,2,192)
    corner = tuple(value[3::4] for value in fact_request)
    ends = [facts[:,1,1]]
    for mode in MODES[1:]:
        ends.append(predict_requests(port,corner,zeros[:n]+mode)[0][:,-1])
    delta = torch.stack(ends,dim=1)-facts[:,1,1,None]
    return facts,delta,requests(port,bank,[1],list(QUERIES))


def query(port, data, chosen):
    s = data[0].new_tensor(MODES)[chosen].repeat_interleave(2)
    return predict_requests(port,data,s)[0][:,-1].reshape(len(chosen),2,192)


def errors(bank, calibration, output):
    actual = bank["true_physical"][:,1,QUERIES]
    decoded = decode(array(output),calibration)
    normal = bank["normal"][:,None]
    diff = decoded-actual
    relative = ((diff[...,:2]-diff[...,:2].mean(1,keepdims=True))*normal).sum(-1)
    return dict(**physical_errors(decoded,actual,normal), centered_normal=np.abs(relative),
                latent_mse=np.square(array(output)-bank["truth_latent"][:,1,QUERIES]).mean(-1))


def table(port, data, n):
    index = torch.zeros(n,dtype=torch.long,device=data[0].device)
    return torch.stack([query(port,data,index+i) for i in range(len(MODES))],dim=1)


def fit_mask(port, bank, calibration):
    fact_request = requests(port,bank,[0,1],list(WRITERS))
    query_request = requests(port,bank,[1],list(QUERIES))
    n, device = len(bank["scene_id"]), fact_request[0].device
    zeros = fact_request[0].new_zeros(n)
    nominal = predict_requests(port,fact_request,zeros.repeat_interleave(4))[0][:,-1].reshape(n,2,2,192)
    observed = torch.as_tensor(bank["truth_latent"][:,:,WRITERS], dtype=torch.float32, device=device)
    target = rectangle(observed-nominal)
    temperature = 2*target.square().mean()
    if not bool(torch.isfinite(temperature) & (temperature>0.)):
        raise RuntimeError("The fit-only contrast scale is not finite and positive")
    corner = tuple(x[3::4] for x in fact_request)
    original_query = query(port,query_request,zeros.long())
    original_corner = nominal[:,1,1]
    w = torch.zeros(192, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([w],lr=.01)
    readout = torch.as_tensor(calibration["readout"],dtype=torch.float32,device=device)
    actual = torch.as_tensor((bank["true_physical"][:,1,QUERIES]-calibration["mean"])/calibration["scale"],
                             dtype=torch.float32,device=device)
    logs = []
    with torch.enable_grad():
        for step in range(1,TRAIN_STEPS+1):
            port.channel_mask = w.sigmoid()
            predictions = [original_query]
            candidates = [original_corner]
            for index,s in enumerate(MODES[1:],start=1):
                predictions.append(query(port,query_request,zeros.long()+index))
                candidates.append(predict_requests(port,corner,zeros+s)[0][:,-1])
            d = torch.stack(candidates,dim=1)-original_corner[:,None]
            fact_error = (target[:,None]-d).square().sum(-1)
            probabilities = torch.softmax(-fact_error/temperature,dim=-1)
            error = torch.stack(predictions,dim=1) @ readout - actual[:,None]
            future = (error-error.mean(2,keepdim=True)).square().mean((2,3))
            loss = (probabilities*future).sum(1).mean()
            if step == 1:
                writer = torch.autograd.grad((probabilities*future.detach()).sum(1).mean(),w,retain_graph=True)[0]
                reader = torch.autograd.grad((probabilities.detach()*future).sum(1).mean(),w,retain_graph=True)[0]
                first_gradients = dict(writer_norm=float(writer.norm()),reader_norm=float(reader.norm()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not bool(torch.isfinite(w.grad).all()):
                raise RuntimeError("Nonfinite gradient on the original joint interaction")
            optimizer.step()
            hard = future.gather(1,fact_error.argmin(-1,keepdim=True)).mean()
            logs.append(dict(step=step, expected_loss=float(loss.detach()), hard_loss=float(hard.detach()),
                             grad_norm=float(w.grad.norm())))
    port.channel_mask = w.detach().sigmoid()
    return logs,dict(temperature=float(temperature),**first_gradients,
                    mask_min=float(port.channel_mask.min()),mask_max=float(port.channel_mask.max()))


@torch.no_grad()
def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    toy = analytic_example()
    host,moments,device = load_host()
    port = NativeJointPort(host,moments)
    with np.load(args.calibration,allow_pickle=False) as prior:
        calibration = dict(prior)
    fit = cycle_bank(host,"train")
    measured = interface(port,fit)
    if args.train_mask:
        logs,training = fit_mask(port,fit,calibration)
        trained = port.channel_mask
        print(training,flush=True)
    fit_request = requests(port,fit,[1],list(QUERIES))
    fit_predictions = table(port,fit_request,len(fit["scene_id"]))
    fit_scores = [float(errors(fit,calibration,fit_predictions[:,j])["centered_normal"].mean()) for j in range(3)]
    fixed = int(np.argmin(fit_scores))

    held = cycle_bank(host,"test")
    nominal,finite,queries = facts_and_queries(port,held)
    actual = torch.as_tensor(held["truth_latent"][:,:,WRITERS],dtype=torch.float32,device=device)
    state = Evidence(torch.zeros(len(actual),3,dtype=torch.float64,device=device)).observe(actual,nominal,finite)
    modes = state.mode()
    pair = (actual[:,1,1]-nominal[:,1,1])-(actual[:,0,1]-nominal[:,0,1])
    pair_mode = (pair[:,None]-finite).double().square().sum(-1).argmin(-1)
    live = query(port,queries,modes)
    if not torch.equal(modes,state.mode()):
        raise RuntimeError("A prediction mutated its historical evidence")
    forecasts = table(port,queries,len(actual))
    root_index = torch.arange(len(actual),device=device)
    torch.testing.assert_close(live,forecasts[root_index,modes],atol=2e-5,rtol=0)
    oracle_loss = np.stack([errors(held,calibration,forecasts[:,i])["centered_normal"].mean(1)
                            for i in range(3)],axis=1)
    oracle = torch.as_tensor(oracle_loss.argmin(-1),device=device)
    predictions = {"native":forecasts[:,0],"paired":forecasts[root_index,pair_mode],"cycle":live,
                   "fixed":forecasts[:,fixed],"oracle":forecasts[root_index,oracle]}
    if args.train_mask:
        port.channel_mask = torch.full_like(trained,.5)
        initial_nominal,initial_effect,initial_query = facts_and_queries(port,held)
        initial = Evidence(torch.zeros_like(state.cost)).observe(actual,initial_nominal,initial_effect).mode()
        predictions["initial_mask"] = query(port,initial_query,initial)
        port.channel_mask = trained
    rows, metrics = [], {}
    base = errors(held,calibration,predictions["native"])["centered_normal"]
    rng = np.random.default_rng(20261112)
    draws = rng.integers(0,48,(4000,48))
    for name, prediction in predictions.items():
        result = errors(held,calibration,prediction)
        gain = (base-result["centered_normal"]).reshape(48,8).mean(1)
        metrics[name] = dict(**{key:float(value.mean()) for key,value in result.items()},
                             improved_scenes=int((gain>0.).sum()),worsened_scenes=int((gain<0.).sum()),
                             gain_ci95=np.quantile(gain[draws].mean(1),[.025,.975]).tolist())
        for i,scene in enumerate(held["scene_id"]):
            for j,action in enumerate(QUERIES):
                rows.append(dict(mode=name,scene=int(scene),world=int(held["world_id"][i]),action=action,
                                 **{key:float(value[i,j]) for key,value in result.items()}))
    args.output.mkdir(parents=True,exist_ok=False)
    write_csv(args.output/"requests.csv",rows)
    write_json(args.output/"interface.json",dict(synthetic=toy,**measured))
    summary = dict(fixed_mode=MODES[fixed],fit_losses=fit_scores,metrics=metrics,
                   cycle_counts=torch.bincount(modes,minlength=3).cpu().tolist(),
                   paired_counts=torch.bincount(pair_mode,minlength=3).cpu().tolist(),
                   oracle_counts=torch.bincount(oracle,minlength=3).cpu().tolist(),
                   cycle_oracle_matches=int((modes==oracle).sum()),
                   cycle_changed_queries=int((live!=forecasts[:,0]).any(-1).sum()))
    write_json(args.output/"summary.json",summary)
    if args.train_mask:
        write_csv(args.output/"training_steps.csv",logs)
        write_json(args.output/"training.json",training)
        np.save(args.output/"learned_channel_mask.npy",array(trained))
    np.savez_compressed(args.output/"cycles_and_outputs.npz",scene_id=held["scene_id"],world_id=held["world_id"],
                        actual_cells=array(actual),nominal_cells=array(nominal),finite_corner_effect=array(finite),
                        scores=array(state.cost),cycle=array(modes),paired=array(pair_mode),oracle=array(oracle),
                        true_query=held["truth_latent"][:,1,QUERIES], true_query_physical=held["true_physical"][:,1,QUERIES],
                        **{f"{name}_prediction":array(value) for name,value in predictions.items()})
    print(summary,flush=True)


@torch.inference_mode()
def cost(args):
    for name in ("cost.csv","cost.json"):
        if (args.output/name).exists():
            raise FileExistsError(args.output/name)
    host,moments,device = load_host()
    port = NativeJointPort(host,moments)
    port.channel_mask = torch.as_tensor(np.load(args.output/"learned_channel_mask.npy",allow_pickle=False),device=device)
    bank = cycle_bank(host,"test")
    with np.load(args.output/"cycles_and_outputs.npz",allow_pickle=False) as record:
        np.testing.assert_array_equal(record["scene_id"],bank["scene_id"])
        np.testing.assert_array_equal(record["world_id"],bank["world_id"])
        root = int(np.flatnonzero(record["cycle"] != 0)[0])
        selected = slice(root,root+1)
        selected_bank = {key:value[selected] for key,value in bank.items()}
        read_mode = torch.as_tensor(record["cycle"][selected],device=device)
        actual = torch.as_tensor(record["actual_cells"][selected],device=device)
        nominal = torch.as_tensor(record["nominal_cells"][selected],device=device)
        effect = torch.as_tensor(record["finite_corner_effect"][selected],device=device)
        expected = torch.as_tensor(record["cycle_prediction"][selected],device=device)
    data = requests(port,selected_bank,[1],list(QUERIES))
    facts = requests(port,selected_bank,[1],[WRITERS[1]])
    base_mode = torch.zeros_like(read_mode)
    strengths = [moments[0].new_tensor([s]) for s in MODES[1:]]
    empty = Evidence(torch.zeros(1,3,dtype=torch.float64,device=device))
    torch.testing.assert_close(query(port,data,read_mode),expected,atol=2e-5,rtol=0)

    def two_modes():
        return tuple(predict_requests(port,facts,s)[0] for s in strengths)

    operations = dict(native_query=lambda:query(port,data,base_mode),joint_query=lambda:query(port,data,read_mode),
                      extra_two_fact_modes=two_modes, write_four=lambda:empty.observe(actual,nominal,effect))
    for _ in range(10):
        for op in operations.values():
            op()
    torch.cuda.synchronize()
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    rows,labels = [],list(operations)
    for j in range(30):
        for name in labels[j%4:]+labels[:j%4]:
            before = perf_counter()
            start.record()
            operations[name]()
            end.record()
            end.synchronize()
            rows.append(dict(round=j,operation=name,cuda_ms=start.elapsed_time(end),wall_ms=(perf_counter()-before)*1000))
    measurements = {}
    for field in ("cuda_ms","wall_ms"):
        times = {name:np.array([row[field] for row in rows if row["operation"]==name]) for name in labels}
        times["joint_minus_native"] = times["joint_query"]-times["native_query"]
        for name,value in times.items():
            measurements.setdefault(name,{})[field] = np.quantile(value,[.1,.5,.9]).tolist()
    summary = dict(device=torch.cuda.get_device_name(),scene=int(bank["scene_id"][root]),
                   world=int(bank["world_id"][root]),read_strength=MODES[int(read_mode)],
                   per_root_scores_bytes=24,per_root_anchor_bytes=768,partial_fact_cache_included=False,
                   timing_p10_p50_p90=measurements)
    write_csv(args.output/"cost.csv",rows)
    write_json(args.output/"cost.json",summary)
    print(summary,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration",type=Path,default=CALIBRATION)
    parser.add_argument("--output",type=Path,default=Path("runs/native_joint_cycle_20261112/pusht"))
    parser.add_argument("--train-mask",action="store_true")
    parser.add_argument("--cost",action="store_true")
    args=parser.parse_args()
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
        (cost if args.cost else run)(args)
