# Frozen TwoRoom P1/P2 evidence attribution

Checkpoint: `p3_original_d.pt`. Fixed 16 seeds; 15 complete, 11 raw matched, 9 with nonzero physical interaction.

All percentages are signed projection recovery; endpoints are raw 192-dimensional latent MSE. The physical-zero seeds remain in the raw panel.

| Panel | H | Original source | Bounded source | Gaussian source | Directional oracle cap |
|---|---:|---:|---:|---:|---:|
| matched_raw | 1 | -0.02989% | +0.74874% | +0.99285% | 7.7038% |
| matched_raw | 2 | -0.05220% | +1.56385% | +1.94864% | 14.4959% |
| matched_raw | 3 | -0.06698% | +2.52393% | +2.90487% | 20.5300% |
| matched_raw | 4 | -0.07271% | +3.60476% | +3.84955% | 25.7060% |
| matched_nonzero | 1 | -0.02989% | +0.74874% | +0.99285% | 7.7038% |
| matched_nonzero | 2 | -0.05220% | +1.56385% | +1.94864% | 14.4959% |
| matched_nonzero | 3 | -0.06698% | +2.52393% | +2.90487% | 20.5300% |
| matched_nonzero | 4 | -0.07271% | +3.60476% | +3.84955% | 25.7060% |

| Source (11 roots) | Signed recovery | Slow/fast support TV | TV from uniform |
|---|---:|---:|---:|
| original | +0.4574% | 0.113725 | 0.125115 |
| bounded | +7.6458% | 0.541008 | 0.343657 |
| gaussian | +9.8684% | 0.999968 | 0.749689 |

| H | Method | Movement endpoint MSE | Stop endpoint MSE | Stop predicted displacement RMS |
|---:|---|---:|---:|---:|
| 1 | uniform | 0.20859452 | 0.05590638 | 3.276282 |
| 1 | original_source_only | 0.20862787 | 0.05584509 | 3.274486 |
| 1 | bounded_source_only | 0.20772180 | 0.05620671 | 3.285071 |
| 1 | gaussian_source_only | 0.20870086 | 0.05626997 | 3.286919 |
| 2 | uniform | 0.21967895 | 0.07065633 | 3.683207 |
| 2 | original_source_only | 0.21957815 | 0.07036200 | 3.675528 |
| 2 | bounded_source_only | 0.22011669 | 0.07288495 | 3.740843 |
| 2 | gaussian_source_only | 0.22308322 | 0.07345741 | 3.755506 |
| 3 | uniform | 0.22786941 | 0.08210016 | 3.970293 |
| 3 | original_source_only | 0.22747079 | 0.08143960 | 3.954289 |
| 3 | bounded_source_only | 0.23158280 | 0.08781585 | 4.106171 |
| 3 | gaussian_source_only | 0.23766698 | 0.08953027 | 4.146060 |
| 4 | uniform | 0.23572434 | 0.09443669 | 4.258150 |
| 4 | original_source_only | 0.23488368 | 0.09329738 | 4.232387 |
| 4 | bounded_source_only | 0.24463888 | 0.10513077 | 4.492784 |
| 4 | gaussian_source_only | 0.25501242 | 0.10864519 | 4.567261 |

| H | Original retained source increment | Bounded retained source increment |
|---:|---:|---:|
| 1 | +0.02181% | -0.43502% |
| 2 | +0.04744% | -0.79441% |
| 3 | +0.07551% | -1.13388% |
| 4 | +0.10597% | -1.44415% |

The JSON retains every root's source likelihood, support, prototype-wise signed contribution, exact oracle support pair and ties, midpoint orientation efficiency, absolute endpoint errors, stop effects, and all-writes-minus-skip-source increment. The tensor artifact contains all native source and future responses and chronological supports.

Forecast boundary: exactly actual z0..z6 enter `predict_prefix`; z7..z10 enter only `score_case`. Independent movement/stop source responses and supports are checked exactly. Original/Gaussian source, endpoint, and interaction numbers are checked against the archived fit when evaluating that checkpoint.

Seed 2026111974 remains included; see its per-root signed values and oracle pairs below.

| H | Original recovery | Bounded recovery | Gaussian recovery | Oracle slow/fast |
|---:|---:|---:|---:|---|
| 1 | -0.23660% | +1.55687% | +1.05494% | 3 / 2 |
| 2 | -0.47951% | +3.05542% | +2.02287% | 3 / 2 |
| 3 | -0.76861% | +4.65646% | +2.96211% | 3 / 2 |
| 4 | -1.10096% | +6.34449% | +3.86387% | 3 / 2 |

Validation: `{"applicable": false, "reason": "Checkpoint differs from archived original fit", "nominal_anchor_exact_by_construction": true, "independent_program_source_responses_exact": true, "independent_program_supports_exact": true, "forecast_input_boundary": "predict_prefix accepts exactly seven actual states z0..z6; no target argument", "future_states_used_for_scoring_only": true, "max_midpoint_identity_linf_error": 1.4068607390171906e-15, "max_projection_contribution_sum_error": 4.973799150320701e-14}`
