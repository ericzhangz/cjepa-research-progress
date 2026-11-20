# Frozen TwoRoom P1/P2 evidence attribution

Checkpoint: `p3_bounded_d.pt`. Fixed 16 seeds; 15 complete, 11 raw matched, 9 with nonzero physical interaction.

All percentages are signed projection recovery; endpoints are raw 192-dimensional latent MSE. The physical-zero seeds remain in the raw panel.

| Panel | H | Original source | Bounded source | Gaussian source | Directional oracle cap |
|---|---:|---:|---:|---:|---:|
| matched_raw | 1 | -0.03732% | +0.63365% | +0.73947% | 7.4826% |
| matched_raw | 2 | -0.06715% | +1.33093% | +1.46112% | 14.0745% |
| matched_raw | 3 | -0.09131% | +2.17093% | +2.19374% | 19.9319% |
| matched_raw | 4 | -0.10822% | +3.13147% | +2.92875% | 24.9580% |
| matched_nonzero | 1 | -0.03732% | +0.63365% | +0.73947% | 7.4826% |
| matched_nonzero | 2 | -0.06715% | +1.33093% | +1.46112% | 14.0745% |
| matched_nonzero | 3 | -0.09131% | +2.17093% | +2.19374% | 19.9319% |
| matched_nonzero | 4 | -0.10822% | +3.13147% | +2.92875% | 24.9580% |

| Source (11 roots) | Signed recovery | Slow/fast support TV | TV from uniform |
|---|---:|---:|---:|
| original | +0.4575% | 0.113571 | 0.125427 |
| bounded | +7.5539% | 0.538948 | 0.342737 |
| gaussian | +9.6852% | 0.999963 | 0.744490 |

| H | Method | Movement endpoint MSE | Stop endpoint MSE | Stop predicted displacement RMS |
|---:|---|---:|---:|---:|
| 1 | uniform | 0.20861291 | 0.05592334 | 3.276779 |
| 1 | original_source_only | 0.20866424 | 0.05585882 | 3.274888 |
| 1 | bounded_source_only | 0.20795486 | 0.05622006 | 3.285461 |
| 1 | gaussian_source_only | 0.20896710 | 0.05611285 | 3.282326 |
| 2 | uniform | 0.21975793 | 0.07072515 | 3.685000 |
| 2 | original_source_only | 0.21969151 | 0.07042094 | 3.677067 |
| 2 | bounded_source_only | 0.22057436 | 0.07289979 | 3.741224 |
| 2 | gaussian_source_only | 0.22363926 | 0.07315092 | 3.747663 |
| 3 | uniform | 0.22804947 | 0.08225364 | 3.974003 |
| 3 | original_source_only | 0.22770433 | 0.08157259 | 3.957517 |
| 3 | bounded_source_only | 0.23224281 | 0.08781370 | 4.106121 |
| 3 | gaussian_source_only | 0.23851635 | 0.08906528 | 4.135279 |
| 4 | uniform | 0.23603937 | 0.09470743 | 4.264250 |
| 4 | original_source_only | 0.23527184 | 0.09353337 | 4.237736 |
| 4 | bounded_source_only | 0.24547299 | 0.10509556 | 4.492032 |
| 4 | gaussian_source_only | 0.25612471 | 0.10800967 | 4.553884 |

| H | Original retained source increment | Bounded retained source increment |
|---:|---:|---:|
| 1 | +0.00800% | -0.47490% |
| 2 | +0.02206% | -0.87130% |
| 3 | +0.04040% | -1.24422% |
| 4 | +0.06329% | -1.58386% |

The JSON retains every root's source likelihood, support, prototype-wise signed contribution, exact oracle support pair and ties, midpoint orientation efficiency, absolute endpoint errors, stop effects, and all-writes-minus-skip-source increment. The tensor artifact contains all native source and future responses and chronological supports.

Forecast boundary: exactly actual z0..z6 enter `predict_prefix`; z7..z10 enter only `score_case`. Independent movement/stop source responses and supports are checked exactly. Original/Gaussian source, endpoint, and interaction numbers are checked against the archived fit when evaluating that checkpoint.

Seed 2026111974 remains included; see its per-root signed values and oracle pairs below.

| H | Original recovery | Bounded recovery | Gaussian recovery | Oracle slow/fast |
|---:|---:|---:|---:|---|
| 1 | -0.25075% | +1.42967% | +0.80428% | 3 / 2 |
| 2 | -0.51070% | +2.78852% | +1.49914% | 3 / 2 |
| 3 | -0.81672% | +4.24860% | +2.15978% | 3 / 2 |
| 4 | -1.16588% | +5.79529% | +2.77894% | 3 / 2 |

Validation: `{"applicable": false, "reason": "Checkpoint differs from archived original fit", "nominal_anchor_exact_by_construction": true, "independent_program_source_responses_exact": true, "independent_program_supports_exact": true, "forecast_input_boundary": "predict_prefix accepts exactly seven actual states z0..z6; no target argument", "future_states_used_for_scoring_only": true, "max_midpoint_identity_linf_error": 1.6577993318389606e-15, "max_projection_contribution_sum_error": 7.105427357601002e-14}`
