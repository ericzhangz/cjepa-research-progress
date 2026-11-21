# Phase 1 fresh oracle capacity

Offered 48; matched 45; physical nonzero 44. All exclusions and per-root scores remain in JSON.

Oracle labels select the preassigned slow/fast branches. This diagnoses supervised native capacity, without evaluating a deployed belief estimator. Physical decoding is one OLS readout fitted only to the author's released TwoRoom training HDF5 (8192 actual encoded frames) and frozen before this held-out scoring. Its source record namespace is author_tworoom_training_hdf5.

| H | Method | Latent recovery | Physical recovery | Move latent MSE | Stop latent MSE | Move displacement RMSE | Stop displacement RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | oracle_speed_branch | 1.1105472 | 1.0349662 | 0.1666083 | 0.0543395 | 12.1708510 | 10.7987308 |
| 1 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1683143 | 0.0542269 | 12.1784522 | 10.8040788 |
| 1 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1684295 | 0.0541813 | 12.1710573 | 10.7986670 |
| 1 | oracle_velocity_frozen | 100.3604040 | 83.4814040 | 0.0568350 | 0.0542269 | 12.0282042 | 10.8040788 |

H1 actual encoded endpoint absolute XY RMSE: 5.200238 map units; actual encoded displacement RMSE (move, stop): [4.816581004116084, 0.0].

| 4 | oracle_speed_branch | 2.0180483 | 2.5056043 | 0.1904531 | 0.0864397 | 13.9433185 | 12.6863777 |
| 4 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1920109 | 0.0863093 | 13.9757409 | 12.7052646 |
| 4 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1905092 | 0.0830683 | 13.9498275 | 12.6931418 |
| 4 | oracle_velocity_frozen | 98.5778603 | 81.6378818 | 0.0828997 | 0.0863093 | 13.7284044 | 12.7052646 |

H4 actual encoded endpoint absolute XY RMSE: 5.200238 map units; actual encoded displacement RMSE (move, stop): [4.816581004116084, 0.0].

Signed recoveries are percentages. RMSE uses the mean over both XY coordinates, speeds and roots, separately for movement/stop. Physical displacement is D(predicted latent) minus D(actual root latent), compared with actual endpoint-minus-root position. Physical recovery projects the predicted speed-by-program interaction onto the corresponding real interaction. The raw-velocity frozen control uses true speed only for action scaling before official normalization; its collision-free, unsaturated subset is reported separately.

The JSON also retains 4,000 paired-reset bootstrap intervals, direction strata, leave-one-root-out recovery, physical-zero/all-matched panels, endpoint vectors and actual-encoding position/displacement errors. The native replay command in the Phase 1 report emits the corresponding free latent trajectories as a PT file.
