# Phase 1: shared full-latent Student-t observation candidate

The candidate uses exactly the historical native means, fixed whitening L, and initial public support. Its scale is R=LL^T; nu=3 gives covariance 3R. There is no temperature or covariance matching. Historical training and probability defaults are unchanged.

For r_k=L^-1(z-f_k), q_k=||r_k||^2 and D=192, log l_k=lgamma((D+3)/2)-lgamma(3/2)-(D/2)log(3*pi)-log|L|-((D+3)/2)log(1+q_k/3). This is the normalized integral of N(r_k;0,xi I), xi~InvGamma(3/2,3/2). No SVD or nodes are needed; gradients use ordinary differentiable arithmetic.

This is a different probability law. Historical conditional covariance is I+xi P(B), with independent orthogonal Gaussian noise; the continuous law's marginal whitened covariance is I+3P(B). The candidate's whitened covariance is 3I. The candidate shares one radial scale across all latent coordinates. Thus an orthogonal residual enters the denominator of every Bayes factor and generally attenuates evidence. Equal means still give equal likelihoods and preserve their odds; zero response gives an exact no-write.

For fixed means, all candidate pair logBFs tend to zero in every distant direction. Its whitened residual score has norm <=(D+3)/(2sqrt(3)); pair logBF magnitude is therefore bounded by this constant times whitened mean separation. The 185-node law is normalized but has a finite Gaussian-mixture far tail, whose pair logBF need not tend to zero. The controlled file contains measured tail values and the analytic largest-node asymptote, without claiming the finite rule equals the continuous law at infinity.

Historical G3 contains a source action, an opposite return action, and two null arrivals. Physical speed remains static within each episode. It contains no real hidden-condition switch. Controlled contradictory observations below test mathematical updating only; they do not establish real-environment reversal capability.

The controlled two-branch witness fixes means -e1,+e1 in D=192 and prior (0.1,0.9). One arrival at -e1 gives branch-zero support 0.241165 under the historical law (logBF 1.050921 < log 9), and 1-1.19e-35 under shared t3 (logBF 82.611541 > log 9). Thus only t3 crosses one half on this single counterevidence. On the identical predeclared four-arrival sequence (+e1,-e1,-e1,+e1), t3 first flips to branch zero at arrival 3; the old law never crosses one half in those four arrivals. Neither result is a claim of universal reversal speed.

| Weights | Law | H | Source vs uniform | All vs uniform | All vs omit |
|---|---|---:|---:|---:|---:|
| episode_h1 | historical_bounded_185 | 1 | +0.0388302% | +0.0712571% | +0.0377203% |
| episode_h1 | candidate_shared_student_t_nu3 | 1 | +0.9136908% | +1.2659692% | +0.3889231% |
| episode_h1 | historical_bounded_185 | 4 | +0.0418891% | +0.0882617% | +0.0402001% |
| episode_h1 | candidate_shared_student_t_nu3 | 4 | +0.9782688% | +1.2698579% | +0.4473478% |
| episode_dual | historical_bounded_185 | 1 | +0.0574015% | -0.0650024% | +0.0581748% |
| episode_dual | candidate_shared_student_t_nu3 | 1 | +0.7130824% | +0.7236899% | +0.5295103% |
| episode_dual | historical_bounded_185 | 4 | +0.0517926% | -0.0284324% | +0.0526287% |
| episode_dual | candidate_shared_student_t_nu3 | 4 | +0.6740901% | +0.8017908% | +0.5204884% |

These are fixed-weight replay diagnostics, not candidate training results or independent confirmation. The JSON retains every support timeline, identical-prior likelihood comparison, H1/H4 readout, and exact null check. The accompanying controlled counterexample shows that the candidate sacrifices historical response-orthogonal evidence invariance.

The previous G1 audit's response_deviation_norm_max is a maximum B-column norm, not the Frobenius or spectral norm. Learned dual train H4 reaches 12.754951 and dual development reaches 11.944585. They are not uniformly near zero. Exact audit ranges are copied to the JSON with their source and definition.

Reproduce: python -m research.p1a.prbd_shared_student_t_phase1. Runtime 86.535 s. No training, new data collection, G4 or P4.
