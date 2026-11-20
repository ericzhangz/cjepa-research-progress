# G1: continuous bounded observation law

G1 passes the error contract locked in `g1_lock.json`. No training, response-scale choice, data collection, new task rollout, or node search occurred in this check. The final reproducible verifier is `research/p1a/prbd_continuous_checks.py`; `prbd_consistency_checks.py` contains the separate G3 training-structure checks.

## Model and integration

The new primary-path API is in `research/p1a/prbd.py`:

```python
nodes, masses = continuous_quadrature()
signed = issue(responses, weights, transition, noise_cholesky,
               reference="uniform")
arrival = assimilate_continuous(signed, observation, nodes, masses)
```

The center is the fixed uniform mean of the issued branch responses. Therefore the branch observation distribution is independent of the public belief and its transport. `issued.prior` still equals the separately transported prior and is used for Bayes normalization. The continuous helper rejects an issuance with the legacy prior-dependent reference. Legacy `issue()` and `assimilate()` calls retain their old defaults and old Q32 law.

On the uniform-reference path with identity transport, `issued.prior` now preserves the supplied valid public weights bitwise, including probability-sum roundoff. The value correction adds an exact zero with the normalized transport derivative, so ordinary weight, transition, and response gradients remain intact. Tests with sums 1, 0.9999999999999998, and 1.0000000000000002 preserve both the issued prior and equal-response posterior bitwise; the largest derivative difference from explicit normalized transport is 3.34e-16. Nonidentity transport and the legacy reference retain their existing calculation. The G2 44-arrival tests were rerun with direct incoming-public equality assertions and returned zero public-state and log-odds drift for both archived offset settings. Details are in `g1_identity_prior_roundoff.json` and the regenerated `g2.json`.

The retained probability model is the continuous inverse-Gamma mixture with nu=3 and bounded response covariance `P=B(I+B.T B)^-1 B.T`. Both bounding constants are fixed at one in the calibrated whitened units. The approximation uses 185 positive log-xi trapezoid nodes with spacing 0.25 on [-6,40]. Weights include the inverse-Gamma density and change-of-variables Jacobian, then are normalized. Each constituent ambient Gaussian and the implemented finite mixture are normalized densities.

The omitted mixing mass is 4.30e-262 below log-xi=-6 and 1.21e-26 above log-xi=40. These mass bounds do not alone guarantee relative density accuracy at arbitrary observations. Actual log-ratio, posterior, absolute-log-density, and gradient errors are checked against a wider adaptive continuous reference. The finite positive rule is an approximation to the continuous law; it does not acquire an infinite-tail limit merely from weight normalization.

At the largest positive nodes, the legacy Woodbury Gram system is ill-conditioned because a centered response family has a structural null direction. The continuous kernel evaluates the bounded covariance in a thin SVD with every singular direction retained. It uses the analytic Gaussian score for backward, avoiding singular-vector derivatives at repeated or zero singular values. No rank threshold or covariance jitter is introduced. With `U=(I+BB.T)^-1`, `C=I+xi*P`, and `v=C^-1(x-b_j)`, the component derivatives are

```
d log N_j / dx = -v
d log N_j / dB = xi*U*(v*v.T-C^-1)*U*B + v*e_j.T.
```

The integrated derivative averages these scores using the conditional scale responsibilities. The usual PyTorch chain then propagates through uniform centering, whitening, and the actual native predictor to its learnable offsets. Only first derivatives are supported; requesting a second derivative fails explicitly. A separate algebraic form removes the ambient common residual before posterior normalization, preserving response-orthogonal evidence cancellation.

## Locked checks and results

The predeclared limits are max pair log-Bayes-factor error 1e-4 nat, posterior TV 1e-5, absolute branch log-density error 1e-4 nat, and gradient L2 error at most `1e-4*(1+reference norm)` for every tested absolute branch and every tested pair. Signed H1/H4 conclusions must match the adaptive reference in aggregate and per seed.

The two archived checkpoints each contribute the same 22 unique initial real sources (11 matched roots, two actual speeds, exact program duplicates removed). For every source the check includes all six branch pairs and all four absolute branch densities, their response gradients, and their full gradients in all 3 x 6 x 192 learnable native offset coordinates. The native model is reevaluated on the actual encoded input; its full-gradient comparisons use the ordinary float32 host. Three complete-parameter direction contractions are included for every objective.

| Maximum observed error | P12 checkpoint | P3 bounded-D checkpoint | Locked limit |
| --- | ---: | ---: | ---: |
| Pair logBF (nat) | 5.19e-13 | 8.14e-13 | 1e-4 |
| Posterior TV | 8.64e-14 | 1.63e-13 | 1e-5 |
| Absolute branch log density (nat) | 1.99e-12 | 9.67e-13 | 1e-4 |
| Full native gradient error / (1+reference norm) | 4.20e-8 | 3.54e-8 | 1e-4 |
| Direct native directional finite-difference error | 4.37e-8 | 9.44e-8 | 1e-4*(1+absolute reference) |

The direct directional differences additionally compare the adaptive SciPy density at positive and negative full-parameter perturbations, with step 1e-4, on both speeds of one real source per checkpoint and three deterministic directions. These supplemental host forwards use float64 so that float32 rounding is not mistaken for a derivative. They cover all absolute branches and all pairs.

SciPy independently integrates density and Gaussian scores using adaptive Gauss-Kronrod intervals over log-xi in [-12,100], with branchwise density rescaling. The largest reported integration relative error bound is 1.14e-9. The reference does not use the 185 PyTorch nodes or PyTorch's custom backward.

The signed predictions have the same per-seed H1/H4 signs under the fixed rule and continuous reference. Aggregate source-only signed recovery is P12 H1 -0.2146%, H4 -0.7197%; P3 bounded-D H1 +0.03952%, H4 +0.18677%. Thus the old +3.131% Q32 H4 result is not restored by an accurate integration of the retained continuous law. This check establishes numerical consistency, not useful sustained prediction.

### New G3 data and final weights

The subsequent G1 audit on the actual new G3 archives and both final checkpoints also passes. It scans 14,232 factual H1 and disjoint free-H4 factors across the shared initialization and final H1/dual weights, selects 107 chronology/range cases (54 with nonuniform causal prefixes), and checks complete native-offset gradients in 46 cases. Maximum errors are 5.76e-13 nat for pair logBF, 1.04e-13 posterior TV, 3.64e-12 nat for absolute log density, 2.60e-9 normalized response-gradient error, and 5.69e-7 normalized native-gradient error. Every one of the 248 training and 1,034 development factual native-null arrivals per checkpoint preserves the supplied public weights bitwise. Full ranges, selection criteria, and results are in [G1_G3_AUDIT.md](G1_G3_AUDIT.md) and `g1_g3_audit.json`.

An additional same-issuance replay checks both final models on all 42 matched development resets. Reconstructed t2 source and t3 reversal logBFs reproduce the fixed evaluator traces exactly. Substituting adaptive continuous likelihoods through single-source, complete-write and omit-source prefixes retains every per-seed and aggregate H1/H4 direction, as well as aggregate interaction-error-gain signs. In the primary dual arm, complete-write versus uniform remains negative: H1 -0.0650024%, H4 -0.0284324%. The positive increment from retaining the initial source also remains positive. Thus the reversal-arrival failure is not an integration artifact. Details are in [G1_G3_SIGNED_REFERENCE.md](G1_G3_SIGNED_REFERENCE.md) and `g1_g3_signed_reference.json`. No G4 or P4 was executed.

Additional tests cover dimensions 1, 3, 7, and 192; response amplitudes 0, 1e-8, 0.1, 1, 100, and 1e4; repeated responses; physically equal responses; nonuniform support; zero prior support; a response-orthogonal perturbation of magnitude 1e6; finite-difference response gradchecks; and bitwise observation-law independence from both prior weights and transition. The maximum synthetic logBF error is 9.58e-7 nat at the large-amplitude stress case, below the locked 1e-4 limit. Identical responses return the issued transported prior exactly while retaining ordinary Bayesian mean derivatives. All old synthetic checks and Q32 archived posterior regressions pass.

## Cost and reproducibility

The complete check takes 43.6 s with the loco Python interpreter, CPU, and one PyTorch thread in this run. For actual D=192, K=4 data, 100 repetitions measure 1.10 ms per issuance and arrival without gradients and 13.0 ms including the observation-kernel backward. Native prediction cost is excluded from that microbenchmark.

```
python -m research.p1a.prbd_continuous_checks
```

The numerical output is `g1.json`; the prelocked contract is `g1_lock.json`. The tested envelope does not prove accuracy for all future trained response families or arbitrarily remote observations. Reuse `check_arrays()` on newly measured real observations when their response/residual range changes. No later research stage was executed by this verifier.
