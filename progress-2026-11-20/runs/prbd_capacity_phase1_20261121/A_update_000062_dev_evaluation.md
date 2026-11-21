# Phase 1 dev oracle capacity

Offered 48; matched 42; physical nonzero 39. The main official-readout JSON files include all root exclusions. Native replay emits the corresponding narrow-readout JSON.

Oracle labels select the preassigned slow/fast branches. This diagnoses supervised native capacity, without evaluating a deployed belief estimator. Physical decoding is one OLS readout fitted only to old training frames and frozen before held-out scoring.

| H | Method | Latent recovery | Physical recovery | Move latent MSE | Stop latent MSE | Move displacement RMSE | Stop displacement RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | oracle_speed_branch | 0.8134946 | 0.7062261 | 0.1466534 | 0.0571047 | 21.1263494 | 14.5551450 |
| 1 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1475179 | 0.0570867 | 21.0539561 | 14.5630678 |
| 1 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1474598 | 0.0569340 | 21.1495483 | 14.5546740 |
| 1 | oracle_velocity_frozen | 98.3879177 | 108.7901431 | 0.0592035 | 0.0570867 | 22.6456098 | 14.5630678 |

H1 actual encoded endpoint absolute XY RMSE: 20.548442 map units; actual encoded displacement RMSE (move, stop): [17.80839943565023, 0.0].

| 4 | oracle_speed_branch | 0.7751998 | -3.2939926 | 0.1813765 | 0.0945523 | 20.8494159 | 16.0680802 |
| 4 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1802178 | 0.0951407 | 20.5805431 | 16.0663581 |
| 4 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1784163 | 0.0914126 | 20.8925057 | 16.0313429 |
| 4 | oracle_velocity_frozen | 100.3006060 | 106.4596252 | 0.0937792 | 0.0951407 | 22.5852140 | 16.0663581 |

H4 actual encoded endpoint absolute XY RMSE: 20.548442 map units; actual encoded displacement RMSE (move, stop): [17.80839943565023, 0.0].

Signed recoveries are percentages. RMSE uses the mean over both XY coordinates, speeds and roots, separately for movement/stop. Physical displacement is D(predicted latent) minus D(actual root latent), compared with actual endpoint-minus-root position. Physical recovery projects the predicted speed-by-program interaction onto the corresponding real interaction. The raw-velocity frozen control uses true speed only for action scaling before official normalization; its collision-free, unsaturated subset is reported separately.

The native replay JSON retains 4,000 paired-reset bootstrap intervals, direction strata, leave-one-root-out recovery, physical-zero/all-matched panels, endpoint vectors and actual-encoding position/displacement errors. The native replay command in the Phase 1 report emits the corresponding free latent trajectories as a PT file.
