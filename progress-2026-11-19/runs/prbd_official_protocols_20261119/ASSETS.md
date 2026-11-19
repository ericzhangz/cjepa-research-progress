# Archived host probes: inputs and execution

These scripts package the host checks recorded on 2026-11-19. Their numerical
probe logic is retained. Release changes expose external paths, use captured
non-pickle panels, and write new results to `*.reproduced.json` by default.
The historical JSON files remain the recorded evidence. Re-running either
script performs host inference; it does not train weights or step an environment.

## PushT / LeWM

Run from the release root, with the dependencies needed by the official LeWM
source and the supplied strict model loader:

```sh
python -m research.p1a.prbd_official_pusht measure --lewm-source PATH_TO_LEWM --model-dir PATH_TO_LEWM_PUSHT_MODEL
```

External inputs:

- The official [le-wm source](https://github.com/lucas-maes/le-wm), containing
  `jepa.py` and `module.py`. The recorded host used the tiny ViT LeWM architecture.
- `config.json` and `weights.pt` from the published
  [lewm-pusht model](https://huggingface.co/quentinll/lewm-pusht).
- Python packages required by that source and loader, including NumPy, PyTorch,
  Transformers, and the upstream imports. The probe runs on CPU and uses the
  included loader's strict, `weights_only=True` state-dictionary loading.

The included `pusht_published_prefixes.npz` is the actual captured probe panel.
It contains RGB `[16, 9, 224, 224, 3]` (`uint8`), raw actions `[16, 40, 2]`
(`float32`), frame indices `[0,5,10,15,20,25,30,35,40]`, episode indices `0..15`,
and the action normalization arrays. Both panel loaders use `allow_pickle=False`.
The first 16 published trajectories are a fixed demonstration prefix, not an
independent held-out task sample.

Panel RGB and commands came from the published
[PushT expert training archive](https://huggingface.co/datasets/quentinll/lewm-pusht/blob/655cd446b9929369d7d406001da85c15d1457850/pusht_expert_train.h5.zst).
Normalization follows the original training convention: exclude non-finite
action rows, compute float32 per-axis means and sample standard deviations,
normalize, then concatenate five consecutive 2D actions. The captured statistics
cover 2,336,736 finite rows, using the public
[full action-column mirror](https://huggingface.co/datasets/JU9999/lewm_pusht_64px/tree/038d29b9b493319e094b16a65bda0340f8032201).
The original archive and action mirror are optional for this packaged probe;
the historical partial-archive extraction tool is not part of this release.

The script retains the original 16-trajectory posterior/forecast diagnostic and
the exact zero-condition comparison through five future prediction steps.
Identity noise/transport and untrained native conditions make this a host
interface diagnostic, not evidence of learned task value or paper reproduction.

## PushObj / AdaJEPA

Run from the release root in a CUDA PyTorch environment with the official
AdaJEPA dependencies:

```sh
python runs/prbd_official_protocols_20261119/pushobj_native_probe.py --source PATH_TO_ADAJEPA --assets PATH_TO_OFFICIAL_ASSETS
```

The required source is the public
[AdaJEPA repository](https://github.com/agentic-learning-ai-lab/adajepa), including
`models/`, `datasets/`, and `preprocessor.py`. Follow its `environment.yaml` for
dependencies. The captured runtime used Python 3.9.19 and PyTorch 2.3.0+cu121.
The published predictor has an unregistered CUDA attention-mask tensor; this
probe preserves that path and requires CUDA. No CPU replacement is supplied.

Download the released assets linked in the
[official README](https://github.com/agentic-learning-ai-lab/adajepa#released-checkpoints-and-eval-data)
and arrange the supplied asset directory as follows:

```text
PATH_TO_OFFICIAL_ASSETS/
  checkpoints/pushobj_shape_shift/
    hydra.yaml
    checkpoints/model_latest.pth
  data/pushobj_eval/
    val_T/plan_targets.pkl
    val_L/plan_targets.pkl
    val_Z/plan_targets.pkl
    val_+/plan_targets.pkl
    val_I/plan_targets.pkl
    val_small_tee/plan_targets.pkl
    val_square/plan_targets.pkl
```

All seven goal banks are required because the original probe includes their
inventory. Each captured bank had 1,000 segments of 25 actions; the banks store
states/actions, not RGB. Training trajectories under `data/pushobj_multishape`
are not required. The checkpoint serializes whole official model components,
so this script explicitly uses `torch.load(..., weights_only=False)` and needs
the corresponding official source classes. The goal banks are also the original
upstream pickle files. These external serialized assets should come from the
official release.

The included `pushobj_frozen_starts.npz` is a lossless subset of the existing
eight-task frozen-run capture, not a newly rendered or simulated dataset:

| Array | Shape | Dtype |
| --- | --- | --- |
| `visual` | `[8,1,224,224,3]` | `uint8` |
| `proprio` | `[8,1,4]` | `float32` |
| `state_0` | `[8,7]` | `float32` |
| `gt_actions` | `[8,5,10]` | `float32` |
| `official_val_I_target_indices` | `[8]` | `int64` |

The official `val_I` indices, in captured order, are
`[149,470,465,974,789,178,722,402]`, selected by the original seed-100 run.
The script checks that order, the five physical start-state coordinates, and
every normalized command against the official target file. Raw official actions
are divided by 100, normalized with the official `ACTION_MEAN`/`ACTION_STD`, and
packed into five blocks of ten values. RGB and proprioception are preserved from
the captured starts; target files alone cannot reconstruct those RGB arrays
without environment rendering. The larger historical frozen-run pickle is not
needed by this packaged probe.

The preserved check inserts the seeded rank-one map in the final native FFN,
tests exact equality at gate zero for one-step and five-step predictions, and
checks output/gradient activity at a nonzero gate. It does not learn PRBD
conditions, run a planner, report closed-loop success, or establish task value.

Both commands accept `--panel` and `--output` to select files explicitly. Timing
values and floating-point details can depend on the runtime; no new host
measurements were run while preparing this public package.
