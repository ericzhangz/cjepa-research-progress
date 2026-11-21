# Phase2 unlabelled Arm A: completed fixed budget

`final.pt` is the final update-2048 state. Training completed normally in
1252.30 seconds (20.9 minutes). It did not read independent evaluation, physical
condition labels, switch points, or Phase1 truth-conditioned weights. The
original null-free native operator has 3456 learned offsets and 18,034,478
frozen host parameters.

The collected archive contains all 192 seeds; 186 have a legal three-state
window and successor. There are 5522 distinct legal training roots. Exact
sample-order replay verified 2048 updates, 8192 episode exposures, 243039 root
exposures and 923004 suffix-target exposures. Every logged objective equals
its logged NLL plus weighted minimum-deviation term; every native gradient
is finite. See `budget_verification.json`.

| Full training-library quantity | Initial | Final 2048 |
|---|---:|---:|
| Composite suffix NLL / dimension | 0.0742733774 | -0.2698387699 |
| Raw latent minimum deviation / dimension | 0.0000106331 | 0.0019987101 |
| Weighted minimum-deviation term | 0.0000000106 | 0.0000019987 |

These are descriptive fitting quantities, not evidence of independent
prediction or task benefit. Each legal actual root retains the entire actual
filtering prefix gradient. Every imagined suffix keeps one prototype over
H1..H4; switches inside this window are an explicitly constant-condition
approximation, not strict future-switch marginalization. Actual time always
transports by fixed T and then absorbs the actual arrival.

Initial and final native H1..H4 nominal paths equal frozen F0 bitwise; their
gradient to learned offsets is exactly zero. The final gradient from a later
root suffix through an earlier actual write is 0.0002388999; gradients to all
three learned prototypes remain nonzero. Full initial/final NLL, penalty and
total 3456-dimensional gradient tensors and their verified sum are saved in
`*_gradient_decomposition.pt` and accompanying JSON files.

The CPU and CUDA QR-SVD implementations agree with the original 185-node
G1 density, including zero, repeated and weak responses at D=192. CUDA maximum
absolute log-density and first-gradient differences are 1.14e-13 and 2.72e-13.
Independent finite differences and the legacy probability regressions pass.
QR reduces the tall response matrix to an exact 4-by-4 SVD without a rank
threshold; quadrature-dependent gradient contractions stay in response
coordinates until the final multiplication into latent space.

The numerical no-write threshold is frozen at 0.00040074605 whitened pairwise
L2 units, calibrated from unlabelled native FP32 reproducibility and fixed L.
No stationary label enters calibration. Setting all nonnominal offsets to
zero on 32 real roots gives maximum numerical separation 0.00003791350, within
that threshold. All such writes return the transported prior bitwise, including
a nonuniform prior that changes under T. The old belief is never restored.

Deliberately contradictory fake speed, cut-point, null-control and Phase1-offset
metadata cannot change any admitted training input. Perturbing future actual
latents from index six cannot change root-2/3/4 issued means, priors or actual
posteriors. This verifies the concrete input and temporal boundary.

The interrupted ambient-SVD timing run is preserved separately in
`../timing_ambient_archive`; it is not a scientific checkpoint. Restarting
used the original initial state and RNG, with the same objective and all 2048
updates, solely after numerical equivalence and speed were established.

The full contract and calling examples are in `../TRAINING_API.md` and
`protocol.json`. The independent final evaluation remains a separate step.
