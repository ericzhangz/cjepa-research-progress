# episode_h1 fixed-history evaluation

Matched resets: 42/48; physical nonzero: 39. Exclusions are retained in JSON.

| H | Comparison | Signed revision | Interaction error gain | Improved interaction resets | Paired endpoint MSE gain | Improved endpoint resets |
|---:|---|---:|---:|---:|---:|---:|
| 1 | single_vs_uniform | +0.0388% [+0.0246, +0.0549] | +0.0776% [+0.0492, +0.1097] | 34/39 | +0.00002945 [+0.00001423, +0.00004873] | 35/39 |
| 1 | all_vs_uniform | +0.0713% [+0.0537, +0.0909] | +0.1423% [+0.1070, +0.1815] | 34/39 | +0.00006880 [+0.00003462, +0.00011029] | 33/39 |
| 1 | all_vs_omit | +0.0377% [+0.0234, +0.0539] | +0.0754% [+0.0468, +0.1078] | 34/39 | +0.00002816 [+0.00001322, +0.00004689] | 35/39 |
| 4 | single_vs_uniform | +0.0419% [+0.0282, +0.0576] | +0.0837% [+0.0563, +0.1151] | 35/39 | +0.00003215 [+0.00001581, +0.00005281] | 35/39 |
| 4 | all_vs_uniform | +0.0883% [+0.0688, +0.1104] | +0.1764% [+0.1376, +0.2206] | 35/39 | +0.00007132 [+0.00003710, +0.00011285] | 34/39 |
| 4 | all_vs_omit | +0.0402% [+0.0268, +0.0556] | +0.0804% [+0.0536, +0.1112] | 35/39 | +0.00003066 [+0.00001510, +0.00005037] | 35/39 |

G3 gate: PASS. The primary arm is prespecified as episode_dual. Every gate component is recorded; confidence intervals do not introduce another gate.

Brackets show 95% percentile intervals from 4,000 paired-reset bootstrap draws, seed 20261120. Leave-one-reset-out values, every root's direction/sign, absolute move/stop and speed errors, stationary traces and all-minus-omit increments are retained in JSON. Physical-zero and all-matched panels are reported separately. Only actual arrivals update belief; all future paths own their predicted windows.

| H | Strategy | Movement endpoint MSE | Stop endpoint MSE |
|---:|---|---:|---:|
| 1 | nominal | 0.14751794 | 0.05708674 |
| 1 | uniform | 0.14745906 | 0.05708674 |
| 1 | single_source | 0.14740016 | 0.05708674 |
| 1 | all_arrivals | 0.14732146 | 0.05708674 |
| 1 | omit_source | 0.14737778 | 0.05708674 |
| 4 | nominal | 0.18021781 | 0.09514066 |
| 4 | uniform | 0.18025618 | 0.09514066 |
| 4 | single_source | 0.18019189 | 0.09514066 |
| 4 | all_arrivals | 0.18011354 | 0.09514066 |
| 4 | omit_source | 0.18017487 | 0.09514066 |

| Panel | Resets | H1 full interaction SSE gain | H4 full interaction SSE gain |
|---|---:|---:|---:|
| all_matched | 42 | +3.22139279 | +3.99295654 |
| physical_nonzero | 39 | +3.22139529 | +3.99295718 |
| physical_zero | 3 | -0.00000250 | -0.00000064 |

Every zero-action observation is an exact public no-write: 168 unique actual arrivals, 336 chronology checks, maximum public weight change 0.
