# Phase 1 dev oracle capacity

Offered 48; matched 42; physical nonzero 39. All exclusions and per-root scores remain in JSON.

Oracle labels select the preassigned slow/fast branches. This diagnoses supervised native capacity, without evaluating a deployed belief estimator. Physical decoding is one OLS readout fitted only to the author's released TwoRoom training HDF5 (8192 actual encoded frames) and frozen before this held-out scoring. Its source record namespace is author_tworoom_training_hdf5.

| H | Method | Latent recovery | Physical recovery | Move latent MSE | Stop latent MSE | Move displacement RMSE | Stop displacement RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | oracle_speed_branch | 37.9771287 | 45.3144969 | 0.0653580 | 0.0157715 | 9.4104986 | 8.3882538 |
| 1 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1475179 | 0.0570867 | 11.9657879 | 10.9129939 |
| 1 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1293071 | 0.0169397 | 9.6458901 | 8.3177263 |
| 1 | oracle_velocity_frozen | 98.3879177 | 95.5617137 | 0.0592035 | 0.0570867 | 11.8949432 | 10.9129939 |

H1 actual encoded endpoint absolute XY RMSE: 4.663223 map units; actual encoded displacement RMSE (move, stop): [3.9931774683477523, 0.0].

| 4 | oracle_speed_branch | 32.3750961 | 34.3860105 | 0.1178767 | 0.0555126 | 9.7663808 | 9.0561524 |
| 4 | nominal_frozen | 0.0000000 | 0.0000000 | 0.1802178 | 0.0951407 | 13.1344122 | 12.0441657 |
| 4 | pooled_single_branch | 0.0000000 | 0.0000000 | 0.1787329 | 0.0670980 | 9.8567625 | 8.9036891 |
| 4 | oracle_velocity_frozen | 100.3006060 | 97.4427184 | 0.0937792 | 0.0951407 | 13.0992880 | 12.0441657 |

H4 actual encoded endpoint absolute XY RMSE: 4.663223 map units; actual encoded displacement RMSE (move, stop): [3.9931774683477523, 0.0].

Signed recoveries are percentages. RMSE uses the mean over both XY coordinates, speeds and roots, separately for movement/stop. Physical displacement is D(predicted latent) minus D(actual root latent), compared with actual endpoint-minus-root position. Physical recovery projects the predicted speed-by-program interaction onto the corresponding real interaction. The raw-velocity frozen control uses true speed only for action scaling before official normalization; its collision-free, unsaturated subset is reported separately.

The JSON also retains 4,000 paired-reset bootstrap intervals, direction strata, leave-one-root-out recovery, physical-zero/all-matched panels, endpoint vectors and actual-encoding position/displacement errors. The native replay command in the Phase 1 report emits the corresponding free latent trajectories as a PT file.
