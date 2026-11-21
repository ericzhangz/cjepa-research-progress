# Phase 1 dev oracle capacity

Offered 48; matched 42; physical nonzero 39. The main official-readout JSON files include all root exclusions. Native replay emits the corresponding narrow-readout JSON.

Oracle labels select the preassigned slow/fast branches. This diagnoses supervised native capacity, without evaluating a deployed belief estimator. Physical decoding is one OLS readout fitted only to the old simulator-reset G3 training split (1056 actual encoded frames) and frozen before this held-out scoring. Its source record namespace is simulator_reset_seeds.

| H | Method | Latent recovery | Physical recovery | Move latent MSE | Stop latent MSE | Move displacement RMSE | Stop displacement RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | oracle_speed_branch | 37.9771287 | 68.4950796 | 0.0653580 | 0.0157715 | 22.4844851 | 11.0257920 |
| 1 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1475179 | 0.0570867 | 21.0539561 | 14.5630678 |
| 1 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1293071 | 0.0169397 | 22.8839637 | 11.4043289 |
| 1 | oracle_velocity_frozen | 98.3879177 | 108.7901431 | 0.0592035 | 0.0570867 | 22.6456098 | 14.5630678 |

H1 actual encoded endpoint absolute XY RMSE: 20.548442 map units; actual encoded displacement RMSE (move, stop): [17.80839943565023, 0.0].

| 4 | oracle_speed_branch | 32.3750961 | 26.6393179 | 0.1178767 | 0.0555126 | 23.8110785 | 11.7144783 |
| 4 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1802178 | 0.0951407 | 20.5805431 | 16.0663581 |
| 4 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1787329 | 0.0670980 | 23.0152230 | 12.2806158 |
| 4 | oracle_velocity_frozen | 100.3006060 | 106.4596252 | 0.0937792 | 0.0951407 | 22.5852140 | 16.0663581 |

H4 actual encoded endpoint absolute XY RMSE: 20.548442 map units; actual encoded displacement RMSE (move, stop): [17.80839943565023, 0.0].

Signed recoveries are percentages. RMSE uses the mean over both XY coordinates, speeds and roots, separately for movement/stop. Physical displacement is D(predicted latent) minus D(actual root latent), compared with actual endpoint-minus-root position. Physical recovery projects the predicted speed-by-program interaction onto the corresponding real interaction. The raw-velocity frozen control uses true speed only for action scaling before official normalization; its collision-free, unsaturated subset is reported separately.

The native replay JSON retains 4,000 paired-reset bootstrap intervals, direction strata, leave-one-root-out recovery, physical-zero/all-matched panels, endpoint vectors and actual-encoding position/displacement errors. The native replay command in the Phase 1 report emits the corresponding free latent trajectories as a PT file.
