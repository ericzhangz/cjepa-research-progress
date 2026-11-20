# G1 numerical coverage on new G3 data and weights

The fixed 185-node continuous-law approximation is checked against adaptive SciPy integration on the actual new training and development archives, at the shared initialization and both final G3 checkpoints. Selection uses chronology and numerical range endpoints. No training, task-score selection, G4, or P4 occurs in this verifier.

All factual H1 factors and all factors in each disjoint free H<=4 partition are scanned. Each selected observation uses its actual causal prefix belief; the geometry remains independent of that belief. Paired development histories are used only as actual replay inputs for numerical validation. They do not enter fitting.

The existing limits remain 1e-4 nat for every pair logBF and absolute branch log density, 1e-5 posterior TV, and gradient L2 error <=1e-4*(1+reference norm). Every selected case checks all four absolute branches and all six pairs in response space. Range-stress cases and a full H4 endpoint also check all native-offset coordinates.

| Weights | Split | Factors | Scanned | Selected | Max logBF error | Max TV | Max absolute-log error | Max response gradient error/(1+norm) | Max native gradient error/(1+norm) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| initialized | train | factual_H1 | 868 | 7 | 3.77e-13 | 5.67e-14 | 9.09e-13 | 2.6e-09 | 1.45e-07 |
| initialized | train | free_H4_partition | 868 | 11 | 3.41e-13 | 6.57e-14 | 3.64e-12 | 6.79e-10 | 3.42e-07 |
| initialized | development | factual_H1 | 1504 | 7 | 1.99e-13 | 3.27e-14 | 5.12e-13 | 8.71e-11 | 1.19e-07 |
| initialized | development | free_H4_partition | 1504 | 10 | 3.45e-13 | 5.84e-14 | 9.09e-13 | 3.88e-10 | 5.68e-07 |
| episode_h1 | train | factual_H1 | 868 | 9 | 5.76e-13 | 1.04e-13 | 5.12e-13 | 2.03e-10 | 1.13e-07 |
| episode_h1 | train | free_H4_partition | 868 | 11 | 5.33e-13 | 8.77e-14 | 9.09e-13 | 6.08e-11 | 2.6e-07 |
| episode_h1 | development | factual_H1 | 1504 | 8 | 3.41e-13 | 6.37e-14 | 1.36e-12 | 1.43e-11 | 6.48e-09 |
| episode_h1 | development | free_H4_partition | 1504 | 9 | 3.23e-13 | 5.1e-14 | 9.09e-13 | 3.56e-11 | 3.71e-07 |
| episode_dual | train | factual_H1 | 868 | 7 | 3.91e-13 | 5.19e-14 | 3.41e-13 | 3.5e-11 | 1.71e-07 |
| episode_dual | train | free_H4_partition | 868 | 11 | 5.12e-13 | 9.8e-14 | 1.71e-12 | 7.45e-11 | 1.22e-07 |
| episode_dual | development | factual_H1 | 1504 | 7 | 1.28e-13 | 1.39e-14 | 4.55e-13 | 1.98e-11 | 0 |
| episode_dual | development | free_H4_partition | 1504 | 10 | 5.44e-13 | 7.36e-14 | 4.55e-13 | 6.48e-11 | 4.39e-07 |

| Weights | Split | Factors | Whitened residual max | Branch deviation norm range | Prefix entropy range | Native-null factors scanned |
|---|---|---|---:|---:|---:|---:|
| initialized | train | factual_H1 | 124.995 | [0, 1.25363] | [1.33217, 1.38629] | 248 |
| initialized | train | free_H4_partition | 144.016 | [0, 1.49271] | [1.35652, 1.38629] | 248 |
| initialized | development | factual_H1 | 113.338 | [0, 1.56909] | [1.33401, 1.38629] | 1034 |
| initialized | development | free_H4_partition | 113.338 | [0, 1.56909] | [1.3351, 1.38629] | 1034 |
| episode_h1 | train | factual_H1 | 125.607 | [0, 4.16188] | [0.821871, 1.38629] | 248 |
| episode_h1 | train | free_H4_partition | 144.938 | [0, 5.95623] | [0.951675, 1.38629] | 248 |
| episode_h1 | development | factual_H1 | 117.302 | [0, 4.87419] | [1.18559, 1.38629] | 1034 |
| episode_h1 | development | free_H4_partition | 117.302 | [0, 4.87418] | [1.19613, 1.38629] | 1034 |
| episode_dual | train | factual_H1 | 132.031 | [0, 10.2561] | [0.973951, 1.38629] | 248 |
| episode_dual | train | free_H4_partition | 147.401 | [0, 12.755] | [1.0388, 1.38629] | 248 |
| episode_dual | development | factual_H1 | 116.505 | [0, 11.9446] | [0.583628, 1.38629] | 1034 |
| episode_dual | development | free_H4_partition | 116.505 | [0, 11.9446] | [0.583628, 1.38629] | 1034 |

Factual native-null arrivals are checked across the complete scanned cohorts: all prototype means are exactly equal and the returned posterior equals the supplied public weights bitwise. The identity-transport value correction retains normalized weight and transition derivatives; its independent regression is in `g1_identity_prior_roundoff.json`. Legacy paths retain their previous normalization.

The JSON retains every selected identity, actual prefix weights, selection criterion, response/range metrics, integration error bound, all per-objective gradient errors, and full numerical ranges. Numerical agreement does not establish predictive usefulness; that requires the separate G3 development results.

The independent replay of final G3 directed H1/H4 conclusions through adaptive continuous likelihoods is reported separately in `G1_G3_SIGNED_REFERENCE.md` and `g1_g3_signed_reference.json`.

Runtime: 460.926 s, CPU, one PyTorch thread.

Reproduce: `python -m research.p1a.prbd_consistency_g1_audit`. The selection/tolerance contract is `g1_g3_audit_lock.json`; full output is `g1_g3_audit.json`.
