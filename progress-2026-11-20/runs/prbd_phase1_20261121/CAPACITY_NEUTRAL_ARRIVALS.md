# Phase 1: actual neutral arrivals without an algebraic null anchor

Replay the same 39 matched G3 resets at both static physical speeds. Each arm receives 156 actual zero-command arrivals (t4,t5). Native responses are issued before the successor, and causal support uses actual t2..t5 arrivals only. A/B are the supervised capacity checkpoints, not PRBD probability-trained candidates. The archived whitening scale R remains fixed; the candidate uses shared full-latent t3 with covariance 3R.

| Arm | Speed-branch whitened pair spread min/max | Speed-pair abs logBF min/max | Binary equal-prior TV mean | Four-branch equal-prior TV mean | Causal write TV mean/max |
|---|---:|---:|---:|---:|---:|
| A | 2.572144/12.439075 | 0.010092/55.928954 | 0.364690 | 0.466533 | 0.038387/0.999039837 |
| B | 1.990565/11.638133 | 0.074996/49.333034 | 0.461926 | 0.678041 | 0.069366/0.999999993 |

The speed-pair columns use the two offline labeled speed branches (1 and 2), so this result is not merely suppression of the nominal branch. Every arm has 0/156 exactly equal response families and 156/156 nonzero causal neutral writes. Strong prior concentration can make a nonzero write numerically tiny; the JSON includes likelihood spans and uniform-prior checks separately.

These measurements show that historical G3's exact no-write property does not transfer to A/B after removing the algebraic null anchor. They do not establish calibrated probabilities for the supervised models: R was retained from the historical fit, and no density training or recalibration was performed. No new trajectories, Phase 2, G4 or planning were run.

Reproduce: python -m research.p1a.prbd_capacity_null_checks. Runtime 9.911 s.
