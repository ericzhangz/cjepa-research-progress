# Reproduce the released PRBD intermediate checks

Run these commands from this `progress-2026-11-19` directory. The snapshot packages the executed PRBD numerical path, native LeWM prototypes, shared single-pass fitter and latest TwoRoom measurement. It does not report a new benchmark run, planner evaluation or task success rate. Preparation of this release did not retrain the model or collect new trajectories.

## Dependencies and external author assets

The checked TwoRoom fit/measurement environment and this release's replay used PyTorch 2.1.1+cu121, NumPy 1.25.2, SciPy 1.13.0, scikit-learn 1.6.1, Transformers 4.46.1 and einops 0.8.0. The earlier math archive records Torch 2.1.0+cu121. `research/requirements-prbd.txt` records the checked replay package versions; install a PyTorch build appropriate for your platform. Fit and measurement run on CPU with two Torch threads; RGB encoding uses CUDA when available, with TF32 disabled.

Obtain original `jepa.py` and `module.py` from the [LeWM author repository](https://github.com/lucas-maes/le-wm). Put their source directory at `external/le-wm` in the command examples, or replace `--source` with your own directory. The loader imports those original classes, reconstructs the released ViT tiny architecture and requires an exact state-dict load with `strict=True, weights_only=True`.

Obtain `config.json` and `weights.pt` from the [official TwoRoom model release](https://huggingface.co/quentinll/lewm-tworooms/tree/main). Put both at `external/lewm-tworooms`, or replace `--model-dir`. The small configuration and full-dataset action statistics are also recorded under `runs/prbd_official_protocols_20261119/tworoom/`; upstream weights remain external. Do not substitute PushT weights for TwoRoom weights.

The included raw RGB/command splits, encoded splits, fitted condition offsets and JSON measurements are under `runs/prbd_tworoom_speed_20261119/`. `fitted.pt` contains only prototype offsets, their initialization and the fixed observation-model tensors; it does not contain the upstream host weights. Physical state/speed diagnostics are separate and never passed to the model or fitter. The original sources and URLs above are the ones recorded in the execution's protocol and source manifests; no content-hash workflow is required.

## Math and saved-checkpoint measurement

The fixed analytic/synthetic checks need no host weights or simulator:

```bash
python -m research.p1a.prbd_math_probe --output reproduce-output/math.json
```

This reruns the same pre-existing cases: independent ambient-covariance reference, density normalization, finite quadrature versus the continuous inverse-gamma reference, exact no-write/equal-response behavior, large orthogonal residuals, weak responses, causal gradients, multimodal support, full-trajectory costs and the analytic transfer example. The continuous heavy-tail asymptotic is not claimed for the finite Gaussian mixture. Compare numerical fields with `runs/prbd_math_20261119/math.json`; different library builds can introduce roundoff.

Replay both initialized and fitted measurements on the included encoded heldout records:

```bash
python -m research.p1a.prbd_tworoom_speed_fit measure --source external/le-wm --model-dir external/lewm-tworooms --report-dir reproduce-output/tworoom
```

This reads the included `fitted.pt` and writes `initialized_measure.json` and `fitted_measure.json` in the requested report directory. It performs no optimizer updates or environment interaction. Compare with the archived same-named files, excluding `elapsed_seconds`. Every predeclared record, including unmatched roots and early termination, remains in the coverage ledger. The same four prototypes, identity transport, 32-node quadrature and Gaussian source-write comparator are used. This is a diagnostic of cross-action prediction after a real arrival, not the author's standard closed-loop protocol.

The primary source is issued at model time 2 and only actual `z3` writes its source belief. The common refreshed root is `[z4,z5,z6]`; future actual `z7..z10` are scoring targets only. Each prototype free-rolls its own predicted window. H1–H4 interactions and paired endpoint errors use shared eligibility and pooled sufficient statistics, not averages of per-case ratios.

## Re-encode and repeat the fixed offline fit

To preserve the distributed checkpoint and reports, first create `reproduce-output/refit` and copy `data_calibration.npz`, `data_fit.npz` and `data_validation.npz` from `runs/prbd_tworoom_speed_20261119/` into it. Then run:

```bash
python -m research.p1a.prbd_tworoom_speed_fit encode --source external/le-wm --model-dir external/lewm-tworooms --output reproduce-output/refit
python -m research.p1a.prbd_tworoom_speed_fit fit --source external/le-wm --model-dir external/lewm-tworooms --output reproduce-output/refit
python -m research.p1a.prbd_tworoom_speed_fit measure --source external/le-wm --model-dir external/lewm-tworooms --output reproduce-output/refit
```

Alternatively, copy the three included `encoded_*.pt` files into the new directory and skip `encode`. Encoding uses the supplied complete-dataset float32 action mean and sample standard deviation, then flattens five primitive controls; RGB uses the recorded ImageNet normalization. Equal-frame checks use fixed padded encoder batches. Changes in the encoder's device/library kernels can affect exact encodings, so use the included encodings to replay the original downstream fit/measurement input.

Calibration estimates the shared noise once from default-speed nominal residuals on its separate 32 reset seeds. The fit makes one chronological pass through the 32 disjoint training records, Adam learning rate 0.001, one update per usable episode; only the three nonzero prototypes' native condition offsets are learned. The nominal route remains exact F0. Speed labels, physical coordinates, validation outcomes and paired counterfactuals do not enter training. See `PROTOCOL.md`, `data_manifest.json` and `fit.json` in the run directory for the fixed split, masks, actually scored arrivals and training records.

## Optional regeneration of the raw physical records

Raw records are already included. To collect them again independently, obtain the [original TwoRoom trajectory archive](https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/main/tworoom.tar.zst) and extract `tworoom.h5`. Install the original [stable-worldmodel environment](https://github.com/galilai-group/stable-worldmodel), including its runtime dependencies and assets; the collection used stable-worldmodel 0.1.1, Python 3.11.15, Torch 2.8.0+cu128 and Gymnasium 1.4.0, in a separate environment. `h5py` is needed to read the released HDF5. Preserve the corresponding unmodified [TwoRoom environment source](https://github.com/galilai-group/stable-worldmodel/blob/main/stable_worldmodel/envs/two_room/env.py) as your reference file.

```bash
python -m research.p1a.prbd_tworoom_speed_data --official-hdf5 external/tworoom.h5 --reference-env-source external/stable-worldmodel/stable_worldmodel/envs/two_room/env.py --output reproduce-output/raw
```

The command checks literal source-text equality against the imported environment and uses its actual reset/step/render operations. It selects the first 64 chronological published episodes having at least 80 finite controls, keeps the first 80 controls of each and uses the same fixed reset seeds, calibration speed 5.0, balanced training speeds 2.5/7.5 and all 16 heldout seeds. Validation independently replays each speed/program; collisions and termination are retained through masks. No predictor or planner is called during collection. Rendering or upstream-version differences must be resolved at the source rather than by silently accepting incompatible roots.

## Extraction and release edits

- `research/p1a/prbd.py`, `prbd_checks.py` and `prbd_tworoom_speed_measure.py` are unchanged copies of the executed numerical files.
- `prbd_native.py` retains the unchanged `native_conditioned` and `NativePrototypes` definitions. Its historical FourRoom-only data selection and check driver are excluded.
- `prbd_public_fit.py` retains the unchanged `windows`, `calibrate` and `fit` definitions used by TwoRoom. Its historical FourRoom acquisition, measurement import and execution driver are excluded.
- `prbd_math_probe.py` adds a configurable output destination and creates its parent directory; its analytic case calculations are unchanged.
- `prbd_tworoom_speed_fit.py` adds explicit external source/model paths and configurable data, statistics and measurement destinations. Encoding, calibration, fitting and measurement calls retain their executed computations and defaults.
- `research_sources_2026-10-22/lewm_assets/prepare_assets.py` retains the original strict `load_model` body, replacing machine-specific/default paths with explicit arguments. Download and content-audit helpers are excluded.
- `prbd_tworoom_speed_data.py` reads the full original HDF5 directly in place of the original machine's metadata-prefix/action-column extraction and discovers the installed environment source in place of its private vendor path. Episode selection, seeds, actions, native rollout, validity masks, speed assignments and diagnostic computations are retained. Reports identify this storage-interface change. The earlier partial archives, full upstream trajectory archive and upstream weights are external.

For this release, the retained core definitions were checked directly against the executed source, all packaged Python compiled, and all three module help/import paths passed. Saved-checkpoint TwoRoom replay matched every archived initialized/fitted JSON field exactly after excluding `elapsed_seconds`. The fixed mathematical assertions passed; the largest numerical difference from the earlier Torch 2.1.0 archive was 4.44e-11 in a finite-difference derivative. No fitting or new simulator collection was performed during release preparation. The portable full-HDF5 collection interface was checked for compilation/import/help only; it was not used to collect new records.
