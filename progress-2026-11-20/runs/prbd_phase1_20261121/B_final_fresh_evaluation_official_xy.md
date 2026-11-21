# Phase 1 fresh oracle capacity

Offered 48; matched 45; physical nonzero 44. All exclusions and per-root scores remain in JSON.

Oracle labels select the preassigned slow/fast branches. This diagnoses supervised native capacity, without evaluating a deployed belief estimator. Physical decoding is one OLS readout fitted only to the author's released TwoRoom training HDF5 (8192 actual encoded frames) and frozen before this held-out scoring. Its source record namespace is author_tworoom_training_hdf5.

| H | Method | Latent recovery | Physical recovery | Move latent MSE | Stop latent MSE | Move displacement RMSE | Stop displacement RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | oracle_speed_branch | 45.1383508 | 26.1087310 | 0.0704764 | 0.0225102 | 9.1380819 | 7.5050934 |
| 1 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1683143 | 0.0542269 | 12.1784522 | 10.8040788 |
| 1 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1637414 | 0.0240839 | 9.1079841 | 7.5487663 |
| 1 | oracle_velocity_frozen | 100.3604040 | 83.4814040 | 0.0568350 | 0.0542269 | 12.0282042 | 10.8040788 |

H1 actual encoded endpoint absolute XY RMSE: 5.200238 map units; actual encoded displacement RMSE (move, stop): [4.816581004116084, 0.0].

| 4 | oracle_speed_branch | 45.1690622 | 24.3889401 | 0.1244973 | 0.0927786 | 9.1683130 | 8.1136577 |
| 4 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1920109 | 0.0863093 | 13.9757409 | 12.7052646 |
| 4 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.2149441 | 0.0942798 | 9.4091834 | 8.3895742 |
| 4 | oracle_velocity_frozen | 98.5778603 | 81.6378818 | 0.0828997 | 0.0863093 | 13.7284044 | 12.7052646 |

H4 actual encoded endpoint absolute XY RMSE: 5.200238 map units; actual encoded displacement RMSE (move, stop): [4.816581004116084, 0.0].

Signed recoveries are percentages. RMSE uses the mean over both XY coordinates, speeds and roots, separately for movement/stop. Physical displacement is D(predicted latent) minus D(actual root latent), compared with actual endpoint-minus-root position. Physical recovery projects the predicted speed-by-program interaction onto the corresponding real interaction. The raw-velocity frozen control uses true speed only for action scaling before official normalization; its collision-free, unsaturated subset is reported separately.

The JSON also retains 4,000 paired-reset bootstrap intervals, direction strata, leave-one-root-out recovery, physical-zero/all-matched panels, endpoint vectors and actual-encoding position/displacement errors. The native replay command in the Phase 1 report emits the corresponding free latent trajectories as a PT file.
