# Frozen TwoRoom P1/P2 evidence attribution

Checkpoint: `p3_original_h1.pt`. Fixed 16 seeds; 15 complete, 11 raw matched, 9 with nonzero physical interaction.

All percentages are signed projection recovery; endpoints are raw 192-dimensional latent MSE. The physical-zero seeds remain in the raw panel.

| Panel | H | Original source | Bounded source | Gaussian source | Directional oracle cap |
|---|---:|---:|---:|---:|---:|
| matched_raw | 1 | -0.08239% | -0.34519% | -0.72120% | 6.2178% |
| matched_raw | 2 | -0.16282% | -0.57415% | -1.32461% | 11.6793% |
| matched_raw | 3 | -0.24551% | -0.63673% | -1.78575% | 16.5020% |
| matched_raw | 4 | -0.32814% | -0.54027% | -2.10023% | 20.6111% |
| matched_nonzero | 1 | -0.08239% | -0.34519% | -0.72120% | 6.2178% |
| matched_nonzero | 2 | -0.16282% | -0.57415% | -1.32461% | 11.6793% |
| matched_nonzero | 3 | -0.24551% | -0.63673% | -1.78575% | 16.5020% |
| matched_nonzero | 4 | -0.32814% | -0.54027% | -2.10023% | 20.6111% |

| Source (11 roots) | Signed recovery | Slow/fast support TV | TV from uniform |
|---|---:|---:|---:|
| original | +0.4609% | 0.107774 | 0.122644 |
| bounded | +7.0151% | 0.514395 | 0.329517 |
| gaussian | +9.3046% | 0.999962 | 0.748236 |

| H | Method | Movement endpoint MSE | Stop endpoint MSE | Stop predicted displacement RMS |
|---:|---|---:|---:|---:|
| 1 | uniform | 0.20880828 | 0.05615831 | 3.283656 |
| 1 | original_source_only | 0.20896587 | 0.05607644 | 3.281261 |
| 1 | bounded_source_only | 0.20986817 | 0.05652852 | 3.294462 |
| 1 | gaussian_source_only | 0.21134232 | 0.05670944 | 3.299729 |
| 2 | uniform | 0.22070394 | 0.07172584 | 3.710978 |
| 2 | original_source_only | 0.22086122 | 0.07137948 | 3.702008 |
| 2 | bounded_source_only | 0.22446137 | 0.07380112 | 3.764282 |
| 2 | gaussian_source_only | 0.22827590 | 0.07469807 | 3.787087 |
| 3 | uniform | 0.23025472 | 0.08453854 | 4.028821 |
| 3 | original_source_only | 0.23025802 | 0.08376773 | 4.010412 |
| 3 | bounded_source_only | 0.23813385 | 0.08953808 | 4.146241 |
| 3 | gaussian_source_only | 0.24520211 | 0.09168194 | 4.195585 |
| 4 | uniform | 0.23996975 | 0.09879659 | 4.355335 |
| 4 | original_source_only | 0.23966368 | 0.09746339 | 4.325849 |
| 4 | bounded_source_only | 0.25333496 | 0.10787460 | 4.551035 |
| 4 | gaussian_source_only | 0.26456128 | 0.11180977 | 4.633301 |

| H | Original retained source increment | Bounded retained source increment |
|---:|---:|---:|
| 1 | -0.07021% | -0.25217% |
| 2 | -0.13159% | -0.45643% |
| 3 | -0.17482% | -0.64363% |
| 4 | -0.19743% | -0.80890% |

The JSON retains every root's source likelihood, support, prototype-wise signed contribution, exact oracle support pair and ties, midpoint orientation efficiency, absolute endpoint errors, stop effects, and all-writes-minus-skip-source increment. The tensor artifact contains all native source and future responses and chronological supports.

Forecast boundary: exactly actual z0..z6 enter `predict_prefix`; z7..z10 enter only `score_case`. Independent movement/stop source responses and supports are checked exactly. Original/Gaussian source, endpoint, and interaction numbers are checked against the archived fit when evaluating that checkpoint.

Seed 2026111974 remains included; see its per-root signed values and oracle pairs below.

| H | Original recovery | Bounded recovery | Gaussian recovery | Oracle slow/fast |
|---:|---:|---:|---:|---|
| 1 | -0.36610% | +0.04934% | -1.34397% | 3 / 1 |
| 2 | -0.72287% | +0.07875% | -2.66483% | 3 / 1 |
| 3 | -1.12625% | +0.11285% | -4.15340% | 3 / 1 |
| 4 | -1.57166% | +0.15234% | -5.79184% | 3 / 1 |

Validation: `{"applicable": false, "reason": "Checkpoint differs from archived original fit", "nominal_anchor_exact_by_construction": true, "independent_program_source_responses_exact": true, "independent_program_supports_exact": true, "forecast_input_boundary": "predict_prefix accepts exactly seven actual states z0..z6; no target argument", "future_states_used_for_scoring_only": true, "max_midpoint_identity_linf_error": 1.2924231997113456e-15, "max_projection_contribution_sum_error": 5.417888360170764e-14}`
