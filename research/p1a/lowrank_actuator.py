"""Fixed-budget delayed-actuator study on the shared readout-feedback path."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from readout_feedback import LowRankFeedback, ReadoutFeedback


class ActuatorCore(nn.Module):
    def forward(self, z, action):
        u, w, x = action[..., 0], action[..., 1], z[..., 0]
        return torch.stack((u, -u, w, -w, x, -x), dim=-1)


class ConstantGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, action):
        return self.raw.expand(*action.shape[:-1], 1)


def make_operator():
    readout = nn.Linear(6, 1, dtype=torch.float64)
    with torch.no_grad():
        readout.weight.copy_(torch.tensor([[1., 1., -1., -1., 1., -1.]]))
        readout.bias.zero_()
    projector = SimpleNamespace(
        net=(nn.Identity(), nn.Identity(), nn.GELU(), readout.requires_grad_(False)),
        final_norm=nn.Identity())
    model = SimpleNamespace(predictor=SimpleNamespace(
        projector=projector, predictor=ActuatorCore()))
    return ReadoutFeedback(model)


def make_data(seed, count, paired):
    rng = np.random.default_rng(seed)
    x, command, nuisance, pulse = rng.uniform(-1., 1., size=(4, count))
    pulse[:count // 4] = 0.
    if paired:
        x = np.tile(x, 2)
        command = np.concatenate((command, -command))
        nuisance, pulse = np.tile(nuisance, 2), np.tile(pulse, 2)
    # Actual write action stores command; distractor is independent and x is unchanged.
    # The subsequent pulse applies the stored command to the visible position.
    actions = np.stack((np.stack((command, nuisance), axis=-1),
                        np.stack((pulse, pulse), axis=-1)), axis=1)
    observed = np.repeat(x[:, None, None], 2, axis=1)
    target = (x + 0.2 * pulse * command)[:, None]
    raw = dict(x=x, command=command, nuisance=nuisance, pulse=pulse,
               actions=actions, observed=observed, target=target)
    return {key: torch.from_numpy(value) for key, value in raw.items()}


def predict(operator, model, data, mode, intervention="keep"):
    basis = None
    if mode == "low_rank":
        basis, gain = model()
    else:
        gain = model
    return operator.sequence(data["observed"][:, 0], data["observed"],
                             data["actions"], gain, mode, 1,
                             intervention=intervention, basis=basis)


def summarize(prediction, data):
    error = (prediction[:, -1] - data["target"]).square().flatten()
    zero = data["pulse"] == 0
    half = len(error) // 2
    true_difference = data["target"][:half] - data["target"][half:]
    predicted_difference = prediction[:half, -1] - prediction[half:, -1]
    return {"mse": float(error.mean()), "nonzero_pulse_mse": float(error[~zero].mean()),
            "zero_pulse_mse": float(error[zero].mean()),
            "paired_difference_mse": float((predicted_difference - true_difference).square().mean()),
            "sample_squared_error_quantiles": torch.quantile(
                error, error.new_tensor([0., .25, .5, .75, .9, 1.])).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if any(args.out.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {args.out}")
    torch.set_num_threads(2)
    operator = make_operator()
    train, held = make_data(401, 1024, False), make_data(402, 256, True)
    config = {"train_seed": 401, "held_seed": 402, "initialization_seed": 41,
              "train_samples": 1024, "held_pairs": 256, "held_samples": 512,
              "steps": 800, "learning_rate": .03, "rank": 1,
              "optimizer": "Adam", "dtype": "float64", "device": "cpu",
              "zero_pulse_fraction": .25,
              "training_target": "Only x_after_pulse; native host is frozen.",
              "data_equations": "write(c,n): x1=x0, hidden_command=c; pulse(p,p): x2=x1+0.2*p*c",
              "native_core": "q=(u,-u,w,-w,x,-x); V=(1,1,-1,-1,1,-1); GELU; bias=0",
              "swap": "Exchange the two test halves, paired on x,n,p and opposite c.",
              "fixed_budget": "All four models use the same full training batch, optimizer and steps; no selection."}
    results = {"config": config, "models": {}}
    arrays = {key: value.numpy() for key, value in held.items()}
    no_history, _, _ = predict(operator, None, held, "kernel")
    results["models"]["no_history"] = {"keep": summarize(no_history, held)}
    results["memoryless_empirical_optimum_mse"] = float(
        (held["target"] - held["x"][:, None]).square().mean())
    arrays["no_history_prediction"] = no_history.numpy()
    relevant = torch.tensor([1., -1., 0., 0., 0., 0.], dtype=torch.float64) / 2 ** .5
    nuisance = torch.tensor([0., 0., 1., -1., 0., 0.], dtype=torch.float64) / 2 ** .5
    for name, mode, constrained in (
            ("lowrank_kernel", "low_rank", True),
            ("lowrank_unconstrained", "low_rank", False),
            ("full_kernel", "kernel", None),
            ("full_activation", "full", None)):
        torch.manual_seed(config["initialization_seed"])
        model = (LowRankFeedback(operator, rank=1, constrained=constrained)
                 if mode == "low_rank" else ConstantGain())
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        loss_trace = []
        for step in range(config["steps"]):
            prediction, _, _ = predict(operator, model, train, mode)
            loss = (prediction[:, -1] - train["target"]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if step % 100 == 0 or step == config["steps"] - 1:
                loss_trace.append([step, float(loss.detach())])
        with torch.no_grad():
            record = {"parameter_count": sum(p.numel() for p in model.parameters()),
                      "training_loss_trace": loss_trace}
            for intervention in ("keep", "erase", "swap"):
                prediction, states, strengths = predict(operator, model, held, mode, intervention)
                record[intervention] = summarize(prediction, held)
                arrays[f"{name}_{intervention}_prediction"] = prediction.numpy()
                arrays[f"{name}_{intervention}_states"] = states.numpy()
                arrays[f"{name}_{intervention}_gain"] = strengths.numpy()
            if mode == "low_rank":
                basis, gain = model()
                record.update(basis=basis.tolist(), gain=gain.tolist(),
                              readout_basis=(operator.readout.weight @ basis).tolist(),
                              relevant_direction_overlap=float((basis[:, 0] @ relevant).square()),
                              nuisance_direction_overlap=float((basis[:, 0] @ nuisance).square()))
            else:
                record["gain"] = model.raw.tanh().tolist()
            results["models"][name] = record
            print(json.dumps({name: {key: record[key]["mse"] for key in ("keep", "erase", "swap")}},
                             ensure_ascii=False), flush=True)
    np.savez_compressed(args.out / "held_samples.npz", **arrays)
    np.savez_compressed(args.out / "train_samples.npz", **{k: v.numpy() for k, v in train.items()})
    (args.out / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "no_history_mse": results["models"]["no_history"]["keep"]["mse"]}), flush=True)


if __name__ == "__main__":
    main()
