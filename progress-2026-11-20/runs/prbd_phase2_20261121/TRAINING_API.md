# Phase2 training and probability API

The scientific run is `A/final.pt` at update 2048. The interrupted
`timing_ambient_archive` is solely a numerical implementation timing record.
It must not be selected as a scientific checkpoint.

## Model and legal input boundary

`research.p1a.prbd_phase2.load_phase2_checkpoint(path, device)` returns the
existing `NativePrototypes(k=4, null_control=None)` and a calibration dictionary
with `noise_cholesky`, `transition`, and `deadzone`. Only the 3456 original
native offsets learn; the 18,034,478 host parameters are frozen. The nominal
branch is literal F0. No true condition, physical coordinate, switch point,
null action reference, or Phase1 weight enters training or deployment.

`load_episodes(path, device)` admits only `z`, `a`, `seed`, and `lengths` from
the actual single-trajectory archive. It returns `(seed, z, a)` tuples for the
186 episodes with at least one legal three-state-to-successor transition.
All legal short prefixes are retained. The 192 collected reset seeds include
five length-zero and one length-two episode that cannot supply a legal root.

## Probability and temporal order

```python
model, calibration = load_phase2_checkpoint(checkpoint, device)
means = model(actual_window[None], aligned_controls[None])[0]
arrival = assimilate_step(means, belief, actual_successor, **calibration)
belief = arrival.posterior
```

The native means issue before the successor is read. The row-stochastic
transition has diagonal `1-2/30` and off-diagonal `(2/30)/3`. It is fixed from
the collection design rate, with no access to an individual switch point.
`assimilate_step` first computes normalized `w^- = w @ T`, then scores the
actual arrival. When maximum pairwise whitened response separation is within
the calibrated numerical tolerance, it returns `w^-` and skips only the write.
It does not restore the old public belief, change the forecasts, or modify
the reported normalized observation density.

`response_density(means[...,K,D], observations[...,D], L)` returns absolute
branch log densities, branch-zero-relative log densities, and maximum pairwise
whitened Euclidean separation. The G1 law uses uniform response centering,
bounded covariance `I + xi*B(B.T B+I)^-1 B.T`, and all 185 fixed positive nodes.
The QR plus 4-by-4 SVD implementation makes no rank decision. Its values and
analytic first derivatives are checked against the original G1 kernel,
including repeated, zero, and weak responses at D=192.

`filter_episode` returns the response family and `StepResult` at every legal
actual root. `StepResult` contains `prior`, `posterior`, `log_likelihood`,
`relative_log_likelihood`, `log_predictive`, `separation`, and `silent`.
Set `deadzone=0` on the same fixed means for read-only attribution without the
positive numerical tolerance. The density itself is unchanged by that flag.

## Frozen learning contract

Training is exactly 2048 Adam updates at batch size four whole actual episodes,
with CPU generator seed 20261121 and `randperm(186)[:4]` per update. The initial
native offsets use their original seed 20261119 and standard deviation 0.01.
The learning rate starts at 0.0003 and falls tenfold at updates 1024 and 1536.
The final state is used regardless of development or independent evaluation.

At every root, the entire earlier actual filtering prefix remains in the
gradient graph. The current belief transports once for the first issued action.
Each prototype owns its free H1..H4 rollout window. Its branch log densities
are summed along the suffix and mixed over k once per horizon. Each H-prefix
score is divided by H*D, then legal horizons, roots within an episode, and
episodes are averaged. Overlapping suffixes define a composite conditional
objective, not an independence claim about repeated actual observations.

Prediction-window switching is a **constant-condition approximation**: one k
owns the whole issued suffix. It does not strictly marginalize future switches.
Actual time still transports and absorbs at every real arrival. Future actual
targets score the suffix; they never replace imagined states or write public
belief during that suffix.

The only added training term is
`0.001 * mean_episode mean_root sum_{k>0,d}(f_k-f_0)^2 / D`, in raw native
latent Euclidean units. Dividing the NLL and this term by D keeps its
coefficient equivalent to 0.001 in the unnormalized formula. It uses the
first issued response at each actual root, and has no condition label.

The numerical write tolerance is fixed before training, using only the first
32 unlabelled actual roots and fixed L. The same F0 inputs are evaluated at
FP32 batch sizes 1, 8 and 32. The tolerance is
`2*max_measured_whitened_discrepancy + 8*eps32*||L^-1||2*max||F0||2`,
which measured 0.00040074605 in whitened pairwise L2 units. No stationary
labels or physical target differences enter this calibration.

Every update saves all loss terms and native gradient norms in `updates.jsonl`.
Full training-library monitoring occurs only at updates 0, 1024 and 2048.
`initial_checks.json`, `final_checks.json`, `probability_checks.json`, and
initial/final gradient decompositions capture the concrete numerical contracts.
