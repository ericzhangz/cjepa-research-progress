# G3: one prototype over one actual episode

The usable observations are `z[3],...,z[T]`; the first three actual states
provide the initial native window. Every usable episode receives one optimizer
update, in source order, using Adam at 0.001. Both arms start from identical
offsets, the canonical raw-zero action reference, and the same frozen host and
calibrated observation covariance. There is no parameter or checkpoint search.

Let `S1` partition the usable targets into singleton blocks. Let `S4` partition
them into consecutive blocks of four, followed by the actual shorter terminal
block. Each block starts from its own actual three-state root; all predictions
inside that block then use the prototype's own free predicted window. No actual
target replaces a predicted token within a free block.

For either partition, each one-step density is the same bounded continuous
nu=3 observation law used for arrival assimilation. Its center and response
geometry use the fixed uniform mean of all prototype responses, independent of
the filtering posterior. With `l[s,k]` denoting that normalized factor, the
episode density is

`p_S(z[3:T+1] | z[0:3], a) = sum_k (1/K) product_s l[s,k]`.

The product spans every target, including all blocks, before the single mixture
over `k`. One prototype therefore explains the entire episode. A new observed
block root refreshes the native window; it does not reset the prototype.
Normalization follows by integrating blocks in reverse chronological order:
within a block every factor is normalized conditional on its already-observed
root, and those factors do not depend on that block's target values. The
uniform mixture of these normalized fixed-k episode densities is normalized.

The H1 arm minimizes `-log p_S1 / (T-2)`. Since the observation geometry does not
depend on the posterior and the condition is static, this is exactly the sum
of sequential Bayesian predictive log densities, divided by the same target
count. This equality applies to gradients as well as values.

The dual arm minimizes `-(log p_S1 + log p_S4) / (2*(T-2))`. This is an explicitly
declared composite of two proper episode log scores. Each target occurs once
in each density. The dual criterion is not presented as the log of a single
normalized product density; its fixed one-half weights add no tunable parameter.

Neither objective writes public belief. In causal development statistics, only
actual one-step arrivals update belief, after each forecast has been issued.
Future targets and imagined states never enter a public update.

`g3_checks.json` records episode/sequential value and gradient equality,
independent scalar-density expansion, shared-k versus per-block remix,
finite-difference gradients, terminal coverage, actual-root refresh, free versus
teacher-forced windows, and future-target isolation. Legacy original/bounded
trajectory fixtures are recorded separately in `g3_legacy_trajectory_checks.json`.

## Completed fixed-budget fit

Both arms completed one update for each of 62 usable episodes, all with 16
complete blocks, scoring 868 actual targets per partition. Of the 64 prescribed
reset seeds, 2026112009 and 2026112030 produced zero complete blocks and therefore
provided no usable three-state native window plus actual successor. They were
recorded and skipped in both arms using the same rule.

| Arm | Updates | Density factors | Fit seconds | Checkpoint |
| --- | ---: | ---: | ---: | --- |
| Episode H1 | 62 | 868 | 19.03 | `g3_episode_h1.pt` |
| Equal H1/H4 episode composite | 62 | 1736 | 71.82 | `g3_episode_dual.pt` |

The complete trainer took 93.41 seconds on CPU with two PyTorch threads. Initial
offsets, canonical null control, calibrated noise, transition, integration nodes,
and integration masses are bitwise identical between the two saved checkpoints.
The saved learned offsets have shape `[3,6,192]` and are finite in both arms.
Only the actual `z,a,lengths,seed,null_control` fields occur in the training file;
no physical speed/state or paired counterfactual outcomes entered fitting.

Both trained arms passed exact frozen-nominal equality on all H1 windows of the
first training episode. A four-step raw-zero continuation, retaining that
episode's two actual historical actions, also returned exactly identical native
paths for every prototype. Its observation basis was exactly zero, branch
densities were exactly equal, and a nonuniform prior was exactly preserved.
The structural fixture was rerun successfully with the final G1 kernel.

`g3_output_validation.json` independently checks the serialized parameter/noise
tensors, source episode order, update counts, terminal coverage, absence of
public writes in either objective, finite losses/gradients, and shared initial
conditions. `g3_fit.json` contains the full per-episode records. These are
implementation and fit-completion results. The separately scoped development
evaluation and postfit continuous-integration audit determine predictive and
numerical adequacy; no G4/P4 work was performed by this trainer.
