# G3 fixed-development influence accounting

The dual failure is concentrated in reset **2026112128**: its collision-free inverse-action arrival changes the support in a direction that is harmful for that arm's frozen future response family. The two zero-action arrivals contribute exactly zero. This is a post hoc diagnostic of the original 39-root physical-nonzero panel; the full-panel failure remains the result.

Let `d` be the actual slow-minus-fast movement-minus-stop interaction, `Y_k` the frozen native movement-minus-stop response, and `q=w_slow-w_fast`. The signed numerator is `N=(qY)·d` in **latent squared units**; signed recovery is `N/||d||²`. Interaction SSE gain is `2N-||qY||²`. Endpoint MSE divides by 192 coordinates and averages the four speed/program cells, so it is a different quantity. Uniform gives exactly zero paired interaction because each arm's entire future family is exactly shared across speeds.

## Closed signed-numerator accounting

All 39 primary roots; the common `sum ||d||²` is 2263.384856 at H1 and H4. The other three matched roots have exactly zero truth interaction, so all signed-numerator totals also apply to all 42 matched roots. The original six exclusions remain unchanged. Single means t2 only; full means t2+t3+t4+t5; omit skips t2.

| Arm | H | Single N | Inverse t3 increment | Zero t4+t5 | Full N | Omit N | Full−omit N |
|---|---:|---:|---:|---:|---:|---:|---:|
| h1 | 1 | +0.878876 | +0.733946 | 0 | +1.612821 | +0.759066 | +0.853755 |
| h1 | 4 | +0.948112 | +1.049590 | 0 | +1.997701 | +1.087817 | +0.909884 |
| dual | 1 | +1.299218 | -2.770472 | 0 | -1.471254 | -2.787974 | +1.316720 |
| dual | 4 | +1.172267 | -1.815801 | 0 | -0.643535 | -1.834724 | +1.191189 |

The first source still helps after the inverse: full−omit is positive for both arms/horizons. Full−single equals the t3 increment exactly; the zero arrivals preserve all public weights bitwise. For dual, full−single SSE gain closes as **−5.540944−0.141115=−5.682060 (H1)** and **−3.631603−0.043897=−3.675500 (H4)**: twice the numerator increment minus the change in squared prediction norm. Across all roots/arms, maximum write-sum closure error is 4.00e-15 and maximum disagreement with original saved scores is 3.45e-14.

## Influential reset and leave-one-out sensitivity

Reset 2026112128 is the largest absolute dual full-vs-uniform SSE contributor at both horizons. It carries only 202.301775/2263.384856 = 8.94% of the truth-energy denominator.

| Arm | H | Root single N | Root inverse ΔN | Root full N | Root omit N | Root full SSE gain |
|---|---:|---:|---:|---:|---:|---:|
| h1 | 1 | +0.054230 | +0.001513 | +0.055743 | +0.012526 | +0.109329 |
| h1 | 4 | +0.095409 | +0.277381 | +0.372791 | +0.297071 | +0.744735 |
| dual | 1 | +0.163490 | -3.566025 | -3.402535 | -3.581355 | -6.946667 |
| dual | 4 | +0.100394 | -2.754164 | -2.653770 | -2.766159 | -5.349750 |

| Arm | H | Full N, all 39 → other 38 | SSE gain, all → other 38 | Paired endpoint MSE gain, all → other 38 |
|---|---:|---:|---:|---:|
| h1 | 1 | +1.612821 → +1.557078 | +3.221395 → +3.112066 | +0.00006880 → +0.00005384 |
| h1 | 4 | +1.997701 → +1.624910 | +3.992957 → +3.248222 | +0.00007132 → +0.00005587 |
| dual | 1 | -1.471254 → +1.931281 | -3.087433 → +3.859234 | -0.00002603 → +0.00006634 |
| dual | 4 | -0.643535 → +2.010235 | -1.332164 → +4.017586 | -0.00001727 → +0.00006767 |

Only **1/39** dual leave-one-out panels has a positive full numerator and SSE gain at either horizon: the one removing this reset. Dual full signed recovery changes −0.0650% → +0.0937% at H1 and −0.0284% → +0.0975% at H4. Every H1-arm leave-one-out panel stays positive. Removing this reset is sensitivity analysis, not an exclusion rule or a revised gate.

## Actual source and inverse; fixed future projection

The stored speeds are 2.5 and 7.5. The source moves left by `[-4.21875,0]` / `[-12.65625,0]`; the inverse moves right by exactly the opposite amounts. All 40 stored t2/t3 collision-mask entries are false, termination/truncation are false, and the maximum actual-minus-commanded primitive displacement error is 0. The inverse position sequence exactly time-reverses the source and returns to `[99.2294693,86.9803238]`. The mask means a displacement discrepancy greater than 1e-5; this trace shows no collision-induced rejection/clipping. Future movement is orthogonal downward by `[0,4.21875]` / `[0,12.65625]`. These are environment XY units, not latent units.

Dual supports in prototype order `[0,1,2,3]`:

| Speed | After source t2 | After inverse t3 (= after both zeros) |
|---|---|---|
| 2.5 | [0.25364, 0.24919, 0.24963, 0.24754] | [0.10025, 0.04122, 0.83967, 0.01886] |
| 7.5 | [0.22429, 0.26099, 0.26830, 0.24643] | [0.24375, 0.26286, 0.28238, 0.21100] |

At t3, relative log likelihoods versus prototype 0 are `[0,-0.870923,+2.141343,-1.646090]` for speed 2.5 and `[0,-0.076097,-0.032097,-0.238433]` for speed 7.5 (nats). Prototype 2 receives 0.839671 slow-speed support, but its future truth-direction coefficient is negative.

Define `c_k=(Y_k-mean_j Y_j)·d/||d||`. These coefficients are in **latent units**; `N=||d|| q·c`, with `||d||=14.223283` at this reset. Centering removes only a common response, since `sum q=0`.

| Frozen family | H | c[0], c[1], c[2], c[3] |
|---|---:|---|
| h1 | 1 | -0.018594, -0.109842, +0.100759, +0.027676 |
| h1 | 4 | -0.013243, -0.151058, +0.074600, +0.089700 |
| dual | 1 | +0.124742, -0.109212, -0.331599, +0.316070 |
| dual | 4 | +0.054680, -0.037826, -0.253999, +0.237144 |

Dual slow/fast TV separation grows **0.030472 → 0.557288** (overlap 0.969528 → 0.442712). The cosine between the support contrast and `c` changes **+0.642 → −0.760** at H1 and **+0.543 → −0.816** at H4. Thus the failure is a wrong projection direction, rather than indistinguishable supports. The H1 arm's full contrast remains positively aligned (cosines +0.082/+0.440).

As a coordinate-level diagnostic, swapping the saved full supports and saved families gives:

| Support source | Family source | H1 N | H4 N |
|---|---|---:|---:|
| h1 | h1 | +0.055743 | +0.372791 |
| h1 | dual | +2.010674 | +1.351746 |
| dual | h1 | +1.107245 | +0.849402 |
| dual | dual | -3.402535 | -2.653770 |

Both crossed combinations are positive; the negative result occurs for the learned dual-support/dual-family combination. This localizes an alignment failure in their combination. Across arms, prototype indices need not identify the same physical explanation, so the swaps do not isolate a physical causal mechanism or prove either component correct.

## Absolute endpoints and limits

At this root, dual H1 moving endpoint MSE (slow/fast) changes **0.479761/0.519981 → 0.496114/0.517772** from uniform to full; H4 changes **0.717614/0.587554 → 0.732012/0.586135**. The slow-speed deterioration dominates the fast-speed improvement. Stop MSE remains exactly **0.110611 (H1)** / **0.344715 (H4)** for all supports; that shared frozen prediction error cancels out of the interaction. The root's paired full-vs-uniform MSE gains are **−0.00353611/−0.00324490** at H1/H4.

The separate [G1 adaptive continuous-reference replay](G1_G3_SIGNED_REFERENCE.md) preserves every matched root's H1/H4 sign and the aggregate signs. For the dual model, maximum informative-arrival log-Bayes-factor error is 4.41e-13 nat and maximum prefix-posterior TV is 9.17e-14. The observed sign failure survives replacing the fixed integration rule with the independent reference.

This establishes exact attribution within the saved computation: an actual, physically reversible inverse observation changed support, and its projection on the fixed future family produced the negative term. It does not establish which unlabeled prototype is physically true, why optimization learned this mismatch, or a general failure rate. The influential reset was identified after seeing development results; no training, parameter, data exclusion, G4, or P4 was changed. Detailed vectors, every root, and closure checks are in `g3_influence.json`; regenerate with `python -m research.p1a.prbd_consistency_influence`.
