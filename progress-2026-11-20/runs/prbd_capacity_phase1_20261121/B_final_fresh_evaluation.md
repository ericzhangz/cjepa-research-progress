# Phase 1 fresh oracle capacity

Offered 48; matched 45; physical nonzero 44. The main official-readout JSON files include all root exclusions. Native replay emits the corresponding narrow-readout JSON.

Oracle labels select the preassigned slow/fast branches. This diagnoses supervised native capacity, without evaluating a deployed belief estimator. Physical decoding is one OLS readout fitted only to the old simulator-reset G3 training split (1056 actual encoded frames) and frozen before this held-out scoring. Its source record namespace is simulator_reset_seeds.

| H | Method | Latent recovery | Physical recovery | Move latent MSE | Stop latent MSE | Move displacement RMSE | Stop displacement RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | oracle_speed_branch | 45.1383508 | 43.9922844 | 0.0704764 | 0.0225102 | 18.3414520 | 10.5094426 |
| 1 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1683143 | 0.0542269 | 21.9875750 | 13.7130272 |
| 1 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1637414 | 0.0240839 | 18.4533190 | 10.5371054 |
| 1 | oracle_velocity_frozen | 100.3604040 | 70.9855015 | 0.0568350 | 0.0542269 | 21.8208141 | 13.7130272 |

H1 actual encoded endpoint absolute XY RMSE: 19.047418 map units; actual encoded displacement RMSE (move, stop): [16.194467403950043, 0.0].

| 4 | oracle_speed_branch | 45.1690622 | 31.6442850 | 0.1244973 | 0.0927786 | 19.0964531 | 12.8322175 |
| 4 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1920109 | 0.0863093 | 22.1562316 | 14.9701754 |
| 4 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.2149441 | 0.0942798 | 19.0268817 | 12.4317404 |
| 4 | oracle_velocity_frozen | 98.5778603 | 68.8135655 | 0.0828997 | 0.0863093 | 21.9984209 | 14.9701754 |

H4 actual encoded endpoint absolute XY RMSE: 19.047418 map units; actual encoded displacement RMSE (move, stop): [16.194467403950043, 0.0].

Signed recoveries are percentages. RMSE uses the mean over both XY coordinates, speeds and roots, separately for movement/stop. Physical displacement is D(predicted latent) minus D(actual root latent), compared with actual endpoint-minus-root position. Physical recovery projects the predicted speed-by-program interaction onto the corresponding real interaction. The raw-velocity frozen control uses true speed only for action scaling before official normalization; its collision-free, unsaturated subset is reported separately.

The native replay JSON retains 4,000 paired-reset bootstrap intervals, direction strata, leave-one-root-out recovery, physical-zero/all-matched panels, endpoint vectors and actual-encoding position/displacement errors. The native replay command in the Phase 1 report emits the corresponding free latent trajectories as a PT file.
