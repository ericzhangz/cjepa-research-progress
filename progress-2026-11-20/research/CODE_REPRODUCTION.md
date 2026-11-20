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

The release contains 27 Python files, including two package markers, a path module and a strict loader. Release preparation checked source-definition equality, all module imports, the structural/numerical invariants and saved-checkpoint predictions against the archived numerical records. Verification outputs were written outside this snapshot; no archived result or checkpoint was changed.
