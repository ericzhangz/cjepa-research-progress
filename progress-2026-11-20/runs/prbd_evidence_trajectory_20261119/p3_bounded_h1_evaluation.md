# Frozen TwoRoom P1/P2 evidence attribution

Checkpoint: `p3_bounded_h1.pt`. Fixed 16 seeds; 15 complete, 11 raw matched, 9 with nonzero physical interaction.

All percentages are signed projection recovery; endpoints are raw 192-dimensional latent MSE. The physical-zero seeds remain in the raw panel.

| Panel | H | Original source | Bounded source | Gaussian source | Directional oracle cap |
|---|---:|---:|---:|---:|---:|
| matched_raw | 1 | -0.08005% | -0.25019% | -0.72679% | 6.5380% |
| matched_raw | 2 | -0.15829% | -0.39042% | -1.38812% | 12.2785% |
| matched_raw | 3 | -0.23917% | -0.36435% | -1.92924% | 17.3466% |
| matched_raw | 4 | -0.32036% | -0.18190% | -2.34862% | 21.6625% |
| matched_nonzero | 1 | -0.08005% | -0.25019% | -0.72679% | 6.5380% |
| matched_nonzero | 2 | -0.15829% | -0.39042% | -1.38812% | 12.2785% |
| matched_nonzero | 3 | -0.23917% | -0.36435% | -1.92924% | 17.3466% |
| matched_nonzero | 4 | -0.32036% | -0.18190% | -2.34862% | 21.6625% |

| Source (11 roots) | Signed recovery | Slow/fast support TV | TV from uniform |
|---|---:|---:|---:|
| original | +0.4711% | 0.109055 | 0.123341 |
| bounded | +7.2580% | 0.524310 | 0.334092 |
| gaussian | +9.5459% | 0.999967 | 0.746011 |

| H | Method | Movement endpoint MSE | Stop endpoint MSE | Stop predicted displacement RMS |
|---:|---|---:|---:|---:|
| 1 | uniform | 0.20876294 | 0.05611554 | 3.282405 |
| 1 | original_source_only | 0.20891804 | 0.05603716 | 3.280112 |
| 1 | bounded_source_only | 0.20967692 | 0.05649001 | 3.293339 |
| 1 | gaussian_source_only | 0.21174859 | 0.05669030 | 3.299172 |
| 2 | uniform | 0.22050782 | 0.07153828 | 3.706123 |
| 2 | original_source_only | 0.22065761 | 0.07119698 | 3.697272 |
| 2 | bounded_source_only | 0.22412937 | 0.07372141 | 3.762248 |
| 2 | gaussian_source_only | 0.22921212 | 0.07465009 | 3.785871 |
| 3 | uniform | 0.22980907 | 0.08411005 | 4.018598 |
| 3 | original_source_only | 0.22979840 | 0.08334561 | 4.000295 |
| 3 | bounded_source_only | 0.23770769 | 0.08942115 | 4.143532 |
| 3 | gaussian_source_only | 0.24677845 | 0.09164252 | 4.194683 |
| 4 | uniform | 0.23917813 | 0.09802555 | 4.338307 |
| 4 | original_source_only | 0.23885275 | 0.09670000 | 4.308875 |
| 4 | bounded_source_only | 0.25286133 | 0.10771711 | 4.547712 |
| 4 | gaussian_source_only | 0.26687693 | 0.11180820 | 4.633268 |

| H | Original retained source increment | Bounded retained source increment |
|---:|---:|---:|
| 1 | -0.06209% | -0.26458% |
| 2 | -0.11649% | -0.47636% |
| 3 | -0.15403% | -0.66785% |
| 4 | -0.17244% | -0.83374% |

The JSON retains every root's source likelihood, support, prototype-wise signed contribution, exact oracle support pair and ties, midpoint orientation efficiency, absolute endpoint errors, stop effects, and all-writes-minus-skip-source increment. The tensor artifact contains all native source and future responses and chronological supports.

Forecast boundary: exactly actual z0..z6 enter `predict_prefix`; z7..z10 enter only `score_case`. Independent movement/stop source responses and supports are checked exactly. Original/Gaussian source, endpoint, and interaction numbers are checked against the archived fit when evaluating that checkpoint.

Seed 2026111974 remains included; see its per-root signed values and oracle pairs below.

| H | Original recovery | Bounded recovery | Gaussian recovery | Oracle slow/fast |
|---:|---:|---:|---:|---|
| 1 | -0.38237% | +0.21196% | -1.10019% | 3 / 1 |
| 2 | -0.75278% | +0.40336% | -2.17359% | 3 / 1 |
| 3 | -1.17255% | +0.61487% | -3.39098% | 3 / 1 |
| 4 | -1.63664% | +0.84596% | -4.73579% | 3 / 1 |

Validation: `{"applicable": false, "reason": "Checkpoint differs from archived original fit", "nominal_anchor_exact_by_construction": true, "independent_program_source_responses_exact": true, "independent_program_supports_exact": true, "forecast_input_boundary": "predict_prefix accepts exactly seven actual states z0..z6; no target argument", "future_states_used_for_scoring_only": true, "max_midpoint_identity_linf_error": 1.3548190347378863e-15, "max_projection_contribution_sum_error": 4.263256414560601e-14}`
