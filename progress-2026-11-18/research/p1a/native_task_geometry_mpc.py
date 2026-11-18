"""Run the existing PushT CEM with live actual arrivals and a common object readout."""

import json
import math
import time

import numpy as np
import torch

from . import native_full_depth_substep as substep
from .native_full_depth_conditioning import advance_log_support
from .native_full_depth_first_effect import rollout
from .native_full_depth_reversible import SWITCH, action_embeddings, block_windows, encode_rgb
from .native_full_depth_task_geometry import OUTPUT, chart_at_latent, shape_cost
from .native_response_relations_fit import native_rollout
from .native_reversible_pusht import run_mpc


SCENES = (55730, 55731, 55732, 55733)
SCHEDULE = (1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1)
ARMS = ("native_frozen", "shape_uniform", "shape_persistent")


class GeometryScore:
    def __init__(self, model, adapter, gate, hold, prefix, measurement, arm):
        self.model, self.adapter, self.gate, self.hold, self.prefix = model, adapter, gate, hold, prefix
        self.measurement = measurement
        self.arm = arm
        self.latents = []
        self.controls = []
        self.goal = None
        self.log_support = hold.new_full((1, 2), -math.log(2)).double()
        self.updates = []
        self.score_seconds = []
        self.feedback_seconds = []
        self.forward_rows = 0

    def controls_for(self, commands):
        raw = torch.as_tensor(commands, dtype=self.hold.dtype, device=self.hold.device)
        return action_embeddings(self.model, raw)

    def readout(self, z):
        return chart_at_latent(z, self.measurement["weight"], self.measurement["bias"])

    def window(self):
        actual = torch.stack(self.latents, dim=1)
        actions = torch.stack(self.controls, dim=1) if self.controls else actual.new_empty((1, 0, self.hold.numel()))
        return block_windows(actual, actions, len(self.latents) - 1, self.hold)

    @torch.inference_mode()
    def __call__(self, frames, executed, goal, plans):
        start = time.perf_counter()
        if self.goal is None:
            self.goal = encode_rgb(self.model, goal[None])[0]
            if self.arm != "native_frozen":
                self.goal = self.readout(self.goal)
        if len(self.latents) < len(frames):
            if len(self.latents) != len(frames) - 1:
                raise ValueError("CEM observation must advance by one completed native block")
            self.latents.append(encode_rgb(self.model, frames[-1:]))
            if len(executed):
                self.controls.append(self.controls_for(executed[-1:]))
        history, past = self.window()
        commands = self.controls_for(plans[None])
        if self.arm == "native_frozen":
            future = native_rollout(self.model, history, past, commands)[0, :, -1]
            value = (future - self.goal).square().sum(-1)
            self.forward_rows += commands.size(1) * commands.size(2) * len(self.model.predictor.transformer.layers)
        else:
            trace = {}
            future = rollout(self.model, self.adapter, self.gate, history, past, commands,
                             compact=True, trace=trace)[0, :, :, -1]
            self.forward_rows += sum(before + (len(self.model.predictor.transformer.layers) - 1) * after
                                     for before, after in trace["rows"])
            if self.arm == "shape_uniform":
                support = self.log_support.exp()[0]
            else:
                support = advance_log_support(self.log_support, SWITCH).exp()[0]
            value = shape_cost(self.readout(future), self.goal) @ support
        values = value.cpu().numpy()
        self.score_seconds.append(time.perf_counter() - start)
        return values

    @torch.inference_mode()
    def begin_execution(self, command):
        if self.arm != "shape_persistent":
            return None
        start = time.perf_counter()
        history, past = self.window()
        actual_action = torch.as_tensor(command[None], device=self.hold.device, dtype=self.hold.dtype)
        issued = self.readout(torch.stack([
            substep.predict_step(self.model, self.adapter, self.gate, self.prefix, history, past, actual_action, step)
            for step in range(1, 6)
        ], dim=-2))
        self.log_support = advance_log_support(self.log_support, SWITCH)
        observations = []
        record = {"prior": self.log_support.exp()[0].tolist(), "phase": [], "issued": issued[0].tolist()}
        self.updates.append(record)
        self.feedback_seconds.append(time.perf_counter() - start)

        @torch.inference_mode()
        def arrive(rgb):
            compute_start = time.perf_counter()
            observations.append(self.readout(encode_rgb(self.model, rgb[None])))
            j = len(observations)
            width = issued.size(-1)
            stop = j * width
            new = substep.chronological_evidence(
                issued[:, :, :j], torch.stack(observations, dim=-2), self.measurement["drift"][:stop],
                self.measurement["covariance"][:stop, :stop],
            )[..., -1]
            self.log_support = (self.log_support + new).log_softmax(-1)
            record["phase"].append({"support": self.log_support.exp()[0].tolist(), "log_likelihood": new[0].tolist(),
                                    "observed": observations[-1][0].tolist()})
            self.feedback_seconds.append(time.perf_counter() - compute_start)

        return arrive


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, adapter, gate, hold, *_ = substep.load_core(device)
    prefix = substep.NativeActionPrefixes(hold.numel()).to(device).eval()
    parameters = torch.load(substep.OUTPUT / "prefix.pt", map_location=device, weights_only=True)
    prefix.load_state_dict(parameters["prefix"])
    readout = torch.load(OUTPUT / "coordinates.pt", map_location=device, weights_only=True)
    result = {"seeds": SCENES, "schedule": SCHEDULE, "horizon": 2, "samples": 16, "iterations": 3,
              "elite_count": 4, "arms": {}}
    for arm in ARMS:
        results = {}
        for seed in SCENES:
            score = GeometryScore(model, adapter, gate, hold, prefix, readout, arm)
            start = time.perf_counter()
            ticks, trace = run_mpc(seed, score, schedule=SCHEDULE, begin_execution=score.begin_execution)
            wall_time = time.perf_counter() - start
            if arm == "shape_persistent" and (len(score.updates) != len(SCHEDULE)
                                              or any(len(block["phase"]) != 5 for block in score.updates)):
                raise RuntimeError("A real block arrived without exactly five chronological writes")
            results[str(seed)] = {"success": bool(trace["success"]), "best_coverage": float(trace["coverage"].max()),
                                  "final_coverage": float(trace["coverage"][-1]), "ticks": ticks,
                                  "actual_coverage": trace["coverage"].tolist(),
                                  "actual_state": trace["state"].tolist(), "feedback": score.updates,
                                  "score_seconds": score.score_seconds, "feedback_seconds": score.feedback_seconds,
                                  "planner_block_rows": score.forward_rows, "total_seconds": wall_time}
            print(arm, seed, results[str(seed)]["success"], results[str(seed)]["final_coverage"], flush=True)
        result["arms"][arm] = results
    with (OUTPUT / "closed_loop.json").open("x", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
