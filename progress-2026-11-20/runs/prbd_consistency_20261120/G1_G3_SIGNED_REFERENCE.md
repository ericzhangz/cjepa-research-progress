# Continuous-reference replay preserves the new G3 signs

Both final models pass the locked numerical sign check on all 42 matched development resets. The native t2 source and t3 reversal responses were reconstructed with the evaluator's identical FP32 two-thread single-case forward. Their fixed-rule log-Bayes factors reproduce every saved trace exactly (maximum discrepancy zero). Adaptive SciPy continuous likelihoods then replace the two actual informative arrivals in the single-source, complete-write, and omit-source prefixes. The two following native-null updates are exact no-writes, already checked on the complete new cohorts. The future branch families and actual targets are the saved evaluator tensors.

Every per-seed H1/H4 directed numerator retains its sign for all three comparisons under both final models. Aggregate directed numerators and aggregate interaction-error gains also retain their signs. No new histories, fitting, G4, or P4 were run.

| Final weights | Maximum logBF error (nat) | Maximum prefix posterior TV |
|---|---:|---:|
| episode_h1 | 5.68e-13 | 1.18e-13 |
| episode_dual | 4.41e-13 | 9.17e-14 |

| Final weights | H | Comparison | Fixed rule signed recovery | Continuous reference signed recovery |
|---|---:|---|---:|---:|
| episode_h1 | 1 | single_vs_uniform | +0.03883016% | +0.03883016% |
| episode_h1 | 1 | all_vs_uniform | +0.07125706% | +0.07125706% |
| episode_h1 | 1 | all_vs_omit | +0.03772029% | +0.03772029% |
| episode_h1 | 4 | single_vs_uniform | +0.04188910% | +0.04188910% |
| episode_h1 | 4 | all_vs_uniform | +0.08826166% | +0.08826166% |
| episode_h1 | 4 | all_vs_omit | +0.04020013% | +0.04020013% |
| episode_dual | 1 | single_vs_uniform | +0.05740154% | +0.05740154% |
| episode_dual | 1 | all_vs_uniform | -0.06500240% | -0.06500240% |
| episode_dual | 1 | all_vs_omit | +0.05817481% | +0.05817481% |
| episode_dual | 4 | single_vs_uniform | +0.05179264% | +0.05179264% |
| episode_dual | 4 | all_vs_uniform | -0.02843241% | -0.02843241% |
| episode_dual | 4 | all_vs_omit | +0.05262865% | +0.05262865% |

In the primary episode-dual arm, the complete-write versus uniform recovery remains negative at both H1 and H4 under the continuous reference. The positive complete-write versus omit-source increment also remains positive. Thus the actual reversal observations drive the complete state in the wrong direction relative to uniform; this conclusion is not caused by the fixed integration rule.

Runtime: 97.607 s. Full per-seed numerical values and signs are in `g1_g3_signed_reference.json`.

Reproduce: `python -m research.p1a.prbd_consistency_sign_check`.
