# Phase2 native response capacity diagnostic

One fixed, privileged-label offline capacity test completed. These results establish capacity for the selected native response family; they do not establish that unlabeled PRBD learns or identifies the branches. Phase3 was not run.

The existing NativePrototypes path is extended with one shared independently copied released final block, final norm and pred_proj for all nonnominal branches. The nominal branch literally calls frozen F0. The copied native modules contain 2,591,552 trainable parameters, exactly the author AdaJEPA count; original condition offsets add 3,456, for 2,595,008 total. Dropout and BatchNorm running moments retain eval semantics. No null-action subtraction, online optimization or fallback was introduced.

The fixed budget was declared before fitting: 2,048 Adam updates, four ordinary actual training episodes per update, learning rate 0.0003 with the original update1024/1536 decays. It completed in 347.764 seconds. The original192 offered episodes yield the same186 usable episodes with at least three action blocks. All legal roots and H1–H4 were used. The objective is raw unwhitened latent predictive squared error, equal episodes/roots/legal H, with each H averaging its prefix. The unlabeled repair arm uses a fixed whitened prediction energy and a deviation penalty. Its observed prefix uses the G1 likelihood. Full-training prefix MSE was 0.11281881 at initialization and 0.01647294 at the final checkpoint.

Actual training speed selects branch1=slow or branch2=fast separately at each executed action. Branch3 retains its initial offsets and shares the trained suffix. During independent evaluation all four native families are issued first. The oracle then selects the true speed of the first issued action and fixes that branch over the entire candidate. Future speed and switch cut points never select subsequent imagined branches. Labels are privileged diagnostic inputs, never deployed method inputs.

## Independent switch endpoints

Exact unscaled raw latent endpoint MSE, averaged over192 coordinates. Both original post-cut windows and the same48 paired reset IDs are retained.

| Direction | Oracle | Same-family uniform | AdaJEPA | OpTTA |
|---|---:|---:|---:|---:|
| to_slow | 0.02688036 | 0.04367059 | 0.03527770 | 0.03579986 |
| to_fast | 0.02493828 | 0.04097548 | 0.04986144 | 0.04924274 |

Paired bootstrap resamples48 complete reset IDs, retaining both arms and both cuts;4,000 draws, seed20261121. Positive gain means the oracle has lower MSE.

| Direction | Comparator | MSE gain |95% paired CI | Improved roots |
|---|---|---:|---|---:|
| to_slow | uniform |0.01679022 |[0.01238531, 0.02164701] |47/48 |
| to_slow | adajepa |0.00839733 |[0.00244658, 0.01546273] |31/48 |
| to_slow | optta |0.00891950 |[0.00303835, 0.01578584] |31/48 |
| to_fast | uniform |0.01603721 |[0.01169164, 0.02101696] |48/48 |
| to_fast | adajepa |0.02492317 |[0.01715231, 0.03359527] |42/48 |
| to_fast | optta |0.02430446 |[0.01664775, 0.03285446] |42/48 |

The original two-consecutive identifiable-action judge recovered94/96 events in each direction, with median lag1. This uses privileged current-speed branch selection and is a capacity diagnostic, not a support-update result.

## Independent static interaction

The frozen48 offered static roots yield45 complete,39 matched-window and34 matched nonzero-physical-interaction roots. No incomplete baseline endpoints enter a scalar result. Primary numbers below use the same34 nonzero roots as original Phase2. Interaction SSE is unscaled squared norm of `(slow move−slow stop)−(fast move−fast stop)`.

| H | Oracle interaction SSE | Uniform | AdaJEPA | OpTTA | Gain vs uniform | Directed recovery |
|---|---:|---:|---:|---:|---:|---:|
|1 |24.534816 | 71.596780 | 69.008479 | 69.127416 |65.7320% |+43.3710% |
|4 |29.170808 | 71.596780 | 62.386664 | 62.553573 |59.2568% |+42.3519% |

H4 oracle interaction SSE improves32/34 roots. Mean paired SSE gain vs uniform is 42.425972 [27.549422, 59.082022]; vs AdaJEPA it is 33.215856 [20.426504, 47.896506]. The H4 raw endpoint MSE is 0.06314063, with move/stop 0.08290338 / 0.04337787.

## Numerical integrity

- Initial and final nominal H1–H4 are bitwise equal to independently evaluated F0; every max absolute error is zero.
- Nominal output has exactly zero gradient to every learned parameter. The host remains frozen and suffix parameter storage is independent.
- Learned-only final checkpoint serialization round-trip is bitwise exact. The25 learned-state tensors contain no host weights.
- Slow/fast offsets differ by L2=4.94903. On39 matched roots their native output RMS difference at H1–H4 is 0.190985, 0.231252, 0.206138, 0.222250.
- No parameter sweep, early stop, independent-score checkpoint selection, test optimization or Phase3 execution occurred.

[protocol.json](repair_oracle__protocol.json) was persisted before fitting. [updates.jsonl](repair_oracle__updates.jsonl), [initial.pt](repair_oracle__initial.pt), [final.pt](repair_oracle__final.pt), [initial_checks.json](repair_oracle__initial_checks.json), [final_checks.json](repair_oracle__final_checks.json) and [serialization_checks.json](repair_oracle__serialization_checks.json) retain execution evidence. [static_predictions.pt](repair_oracle__static_predictions.pt) and [switch_predictions.pt](repair_oracle__switch_predictions.pt.zip) retain the frozen families and comparisons. Per-root scores and original event judges are saved separately.

An initial reporting attempt included incomplete static executions and correctly failed JSON finite-value validation because archived baselines have NaN at unavailable endpoints. Reporting was corrected to the original complete/matched validity contract and reused the same frozen predictions. No training or model selection was repeated.
