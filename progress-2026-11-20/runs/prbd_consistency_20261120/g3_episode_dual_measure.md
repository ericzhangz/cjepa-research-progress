# episode_dual fixed-history evaluation

Matched resets: 42/48; physical nonzero: 39. Exclusions are retained in JSON.

| H | Comparison | Signed revision | Interaction error gain | Improved interaction resets | Paired endpoint MSE gain | Improved endpoint resets |
|---:|---|---:|---:|---:|---:|---:|
| 1 | single_vs_uniform | +0.0574% [+0.0334, +0.0839] | +0.1146% [+0.0668, +0.1674] | 36/39 | +0.00004542 [+0.00001899, +0.00007869] | 35/39 |
| 1 | all_vs_uniform | -0.0650% [-0.4128, +0.1145] | -0.1364% [-0.8461, +0.2288] | 35/39 | -0.00002603 [-0.00023257, +0.00009911] | 33/39 |
| 1 | all_vs_omit | +0.0582% [+0.0355, +0.0818] | +0.1165% [+0.0710, +0.1636] | 36/39 | +0.00004612 [+0.00001992, +0.00007860] | 34/39 |
| 4 | single_vs_uniform | +0.0518% [+0.0310, +0.0784] | +0.1035% [+0.0620, +0.1567] | 36/39 | +0.00003937 [+0.00001751, +0.00006889] | 37/39 |
| 4 | all_vs_uniform | -0.0284% [-0.3038, +0.1198] | -0.0589% [-0.6135, +0.2394] | 34/39 | -0.00001727 [-0.00020937, +0.00010134] | 33/39 |
| 4 | all_vs_omit | +0.0526% [+0.0329, +0.0776] | +0.1052% [+0.0658, +0.1549] | 36/39 | +0.00003994 [+0.00001813, +0.00006888] | 37/39 |

G3 gate: FAIL. The primary arm is prespecified as episode_dual. Every gate component is recorded; confidence intervals do not introduce another gate.

Brackets show 95% percentile intervals from 4,000 paired-reset bootstrap draws, seed 20261120. Leave-one-reset-out values, every root's direction/sign, absolute move/stop and speed errors, stationary traces and all-minus-omit increments are retained in JSON. Physical-zero and all-matched panels are reported separately. Only actual arrivals update belief; all future paths own their predicted windows.

| H | Strategy | Movement endpoint MSE | Stop endpoint MSE |
|---:|---|---:|---:|
| 1 | nominal | 0.14751794 | 0.05708674 |
| 1 | uniform | 0.14740465 | 0.05708674 |
| 1 | single_source | 0.14731380 | 0.05708674 |
| 1 | all_arrivals | 0.14745671 | 0.05708674 |
| 1 | omit_source | 0.14754895 | 0.05708674 |
| 4 | nominal | 0.18021781 | 0.09514066 |
| 4 | uniform | 0.18024318 | 0.09514066 |
| 4 | single_source | 0.18016445 | 0.09514066 |
| 4 | all_arrivals | 0.18027771 | 0.09514066 |
| 4 | omit_source | 0.18035759 | 0.09514066 |

| Panel | Resets | H1 full interaction SSE gain | H4 full interaction SSE gain |
|---|---:|---:|---:|
| all_matched | 42 | -3.08743592 | -1.33216469 |
| physical_nonzero | 39 | -3.08743332 | -1.33216389 |
| physical_zero | 3 | -0.00000260 | -0.00000080 |

Every zero-action observation is an exact public no-write: 168 unique actual arrivals, 336 chronology checks, maximum public weight change 0.
