# Executed PRBD core and saved-record replay

Run the commands below from `progress-2026-11-20/`. The code packages the executed P0–P3 and P3b G1–G3 computations. Measurement commands use saved actual records and fixed checkpoints. They do not fit parameters, collect histories, or execute a planner.

## Assets and paths

Install the checked package versions in `research/requirements-prbd.txt`, using a PyTorch build appropriate for your platform. The fit/measurement checks used Python 3.10.15, Torch 2.1.1+cu121 and two CPU threads. The P0 figure also needs Matplotlib. Simulator dependencies are only needed for an explicitly requested fresh collection.

Obtain the unmodified `jepa.py` and `module.py` from the [LeWM author repository](https://github.com/lucas-maes/le-wm), and `config.json` plus `weights.pt` from the [official TwoRoom model release](https://huggingface.co/quentinll/lewm-tworooms/tree/main). Default locations are `external/le-wm/` and `external/lewm-tworooms/` inside this snapshot. Alternatively set `PRBD_LEWM_SOURCE` and `PRBD_MODEL_DIR` to their directories before starting Python. The strict loader reconstructs the recorded ViT tiny architecture and loads the complete state dictionary with `strict=True, weights_only=True`. The small prototype checkpoints do not contain these host weights.

The two primary run directories are included in this snapshot. Historical speed inputs and full-dataset action statistics default to the preceding `../progress-2026-11-19/runs/` snapshot in the same repository. `PRBD_SPEED_DATA_DIR` can explicitly replace its `prbd_tworoom_speed_20261119` directory. These historical inputs include `fitted.pt`, `encoded_fit.pt`, `encoded_validation.pt`, the raw NPZ splits and physical diagnostics. `action_statistics.json` is read from the preceding snapshot's `prbd_official_protocols_20261119/tworoom/` directory for optional RGB/action encoding and G2 preparation.

Every writer preserves the packaged run files. Default outputs mirror the run directory names under `reproduce-output/`. Set `PRBD_OUTPUT_DIR` to another output root; explicit output paths outside the packaged `runs/` directories are honored. Input and output paths are separate. For example, a regenerated G2 preparation is written separately while the saved G2 replay continues to read the distributed `g2_prepared.pt`.

PowerShell example:

```powershell
$env:PRBD_LEWM_SOURCE = "C:/assets/le-wm"
$env:PRBD_MODEL_DIR = "C:/assets/lewm-tworooms"
```

Bash example:

```bash
export PRBD_LEWM_SOURCE=/assets/le-wm
export PRBD_MODEL_DIR=/assets/lewm-tworooms
```

## Numerical and structural checks without host assets

```bash
python -m research.p1a.prbd_math_probe
python -m research.p1a.prbd_self_masking
python -m research.p1a.prbd_trajectory_checks
python -m research.p1a.prbd_consistency_checks
python -m research.p1a.prbd_consistency_measure --check
```

These run the fixed analytic transfer, finite/continuous scalar references, owned-window trajectory checks, fixed-prototype episode partition checks and the evaluator fixture. The complete host-free G1 tests are callable as `synthetic_checks()` and `identity_prior_roundoff_checks()` in `prbd_continuous_checks.py`.

## P0–P3 saved-checkpoint replay

The observed-source continuous reference uses the saved branch tensors and needs no host:

```bash
python -m research.p1a.prbd_observed_sources
python -m research.p1a.prbd_source_reference_readout
python -m research.p1a.prbd_response_oracle runs/prbd_evidence_trajectory_20261119/p12_tensors.pt reproduce-output/p2_oracle.json
```

Reconstruct native branch responses with the external host:

```bash
python -m research.p1a.prbd_evidence_attribution
python -m research.p1a.prbd_evidence_attribution --checkpoint runs/prbd_evidence_trajectory_20261119/p3_bounded_d.pt --output reproduce-output/p3_bounded_d_evaluation.json
python -m research.p1a.prbd_trajectory_fit --summarize
```

The other three archived P3 controls can be measured by changing the checkpoint and output names. The summary reads the four distributed fit/evaluation records and writes a new summary. Truth-informed oracle calculations remain offline capacity diagnostics.

## P3b G1–G3 saved-checkpoint replay

```bash
python -m research.p1a.prbd_continuous_checks
python -m research.p1a.prbd_consistency_native check
python -m research.p1a.prbd_consistency_native measure --g1-passed
python -m research.p1a.prbd_consistency_measure --checkpoint runs/prbd_consistency_20261120/g3_episode_h1.pt
python -m research.p1a.prbd_consistency_measure --checkpoint runs/prbd_consistency_20261120/g3_episode_dual.pt
python -m research.p1a.prbd_consistency_g1_audit
python -m research.p1a.prbd_consistency_sign_check
python -m research.p1a.prbd_consistency_influence
```

G1 independently checks continuous likelihoods, actual native gradients and historical sources. The G3 audit scans the distributed training/development encodings and three fixed offset settings. The sign replay replaces the two informative actual arrivals with adaptive continuous integrals. Influence accounting uses saved predictions and physical traces without any model forward. The distributed report's G3 failure and unused G4/P4 status remain the interpretation of these records.

For comparisons with archived JSON, exclude input/output/executable path metadata and elapsed time. Native kernels and package versions can affect roundoff. Use the supplied `train.pt`, `dev.pt` and historical encodings to preserve the original downstream inputs.

## Optional offline regeneration interfaces

No fit or new collection was run while preparing this code release. The existing commands retain those separate, explicit interfaces:

```bash
python -m research.p1a.prbd_consistency_data encode train
python -m research.p1a.prbd_consistency_data encode dev
python -m research.p1a.prbd_consistency_fit --output reproduce-output/refit-p3b
python -m research.p1a.prbd_trajectory_fit bounded_d
```

Encoding reads the saved RGB/commands and uses the fixed pointwise action shape. The fit defaults consume the supplied encodings and frozen noise, performing the original fixed single-pass updates. Training outputs are separate from distributed checkpoints.

`prbd_consistency_data collect train` and `collect dev` explicitly perform fresh simulator interaction. They require the unmodified [stable-worldmodel TwoRoom environment](https://github.com/galilai-group/stable-worldmodel) and its runtime assets; the original collection used stable-worldmodel 0.1.1, Gymnasium 1.4.0 and a separate Python 3.11 environment. Install it normally, or point `PRBD_ENV_SOURCE` at the author package root. P3b's fixed command programs are generated directly, so this collection path does not require the earlier HDF5 trajectory archive. It does not call a predictor or planner. Fresh collection is unnecessary for the saved-record replays above.

## Extraction and verification

`prbd.py` is copied unchanged. The retained `encode_native_controls`, `native_conditioned`, `NativePrototypes` and every retained shared-fit definition match the executed source directly. Release changes concern module extraction, explicit external paths and separate output destinations. The strict loader contains the original model-construction/load body; unrelated acquisition helpers and historical environment drivers are omitted. The P3b rollout and stack operations retain the executed native environment operations with an import performed only when collection is requested.

The original 20 November release contained 27 Python files, including two package markers, a path module and a strict loader. Release preparation checked source-definition equality, all module imports, the structural/numerical invariants and saved-checkpoint predictions against the archived numerical records. Verification outputs were written outside this snapshot; no archived result or checkpoint was changed.

## Phase 2 · 21 November update

Run these commands from this snapshot. The first checks the G1 probability law. The second recomputes the independent scores from saved predictions of all five methods and the corresponding actual outcomes. The third renders the report from the saved numerical results.

```bash
python -m research.p1a.prbd_phase2_fit check
python -m research.p1a.prbd_phase2_score
python -m research.p1a.prbd_phase2_report
```

These commands need PyTorch and NumPy. The CPU check uses the original 185 nodes, analytic gradients and a finite-difference check. Scoring also uses the supplied Phase 1 fixed XY readout for a secondary coordinate panel. The archived reference for each command is in `runs/prbd_phase2_20261121/`. Default output goes under `reproduce-output/prbd_phase2_20261121/`; each command reads the published inputs. The report command reads the saved scores when run with defaults.

To re-issue all PRBD forecasts directly from the trained offsets, configure the external LeWM source and TwoRoom host above, then run:

```bash
python -m research.p1a.prbd_phase2_measure runs/prbd_phase2_20261121/A/final.pt --device cuda
```

With the original CUDA FP32 host, release verification reproduced the 67 tensor fields in `pred_static_prbd.pt`, `pred_switch_prbd.pt` and `initial_static_writes.pt` with exact values and validity masks. The independent scorer reproduced the archived static and switching panels, per-root / per-event records and admission decisions. Timing metadata describes the hardware and load of each execution.

### Chaining fresh outputs in a separate directory

For a chained replay, copy the published Phase 2 run to a new location **outside the packaged runs directory**, then point `PRBD_PHASE2_RUN_DIR` to that copy. Every Phase 2 stage will read its inputs and write its outputs there, including the newly computed scores consumed by the report renderer. The Phase 1 readout and historical Ledoit–Wolf scale still come from their distributed snapshot paths. For example, from a clean snapshot on PowerShell:

```powershell
$phase2Replay = Join-Path $PWD 'reproduce-output/full-phase2'
New-Item -ItemType Directory -Force (Split-Path $phase2Replay) | Out-Null
Copy-Item -LiteralPath 'runs/prbd_phase2_20261121' -Destination $phase2Replay -Recurse
$env:PRBD_PHASE2_RUN_DIR = $phase2Replay
python -m research.p1a.prbd_phase2_measure "$phase2Replay/A/final.pt" --device cuda
python -m research.p1a.prbd_phase2_score
python -m research.p1a.prbd_phase2_report
```

The same working directory supports optional regeneration. With the external host in place, `python -m research.p1a.prbd_phase2_fit train` uses the supplied `train.pt` and fixed scale from the preceding speed snapshot, then performs the complete 2,048-update fit. `python -m research.p1a.prbd_phase2_training_diagnostics "$phase2Replay/A/final.pt"` checks its gradients, budget and causal input boundary. The fit regenerates the final and initial checkpoints, which the forecast command consumes. Rerunning `prbd_phase2_baselines` with `fit_gru`, then `gru`, reproduces the recurrent control. The `frozen --shadow-frozen`, `adajepa`, and `optta` stages require the stable-worldmodel planner and environment dependencies. The released runner pairs the random seeds across initial speeds. The historical Frozen shadow timer used separate stream seeds; its shadow plans do not enter frozen forecasts.

Both online controls use the [upstream author implementation](https://github.com/akshaymete/optta/tree/main/lewm/code/lewm_optta). The two executed upstream modules and their MIT license are included in `research/p1a/lewm_optta/`. The Stage A dependency contains the executed recurrent model, captured-plan structure and capture function. Fresh simulation and CEM use the author's stable-worldmodel package. Install it with its environment dependencies or set `PRBD_ENV_SOURCE` to its source directory. Optional `python -m research.p1a.prbd_phase2_data collect train` and `encode train` regenerate a split in the working copy. Use `static` or `switch` for the two independent split names. Actual RGB, commands, encoded trajectories, counterfactual evaluation records and the public reset goals already accompany the archived Phase 2 run.
