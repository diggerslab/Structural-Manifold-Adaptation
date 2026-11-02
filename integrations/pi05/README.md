# Training and Evaluation: π0.5 Reference Implementation

This integration exposes the actual **JAX/OpenPI** training path used by the SMA
research implementation. It is not a new Pi0.5 model. The model implementation is
the pinned upstream [OpenPI](https://github.com/Physical-Intelligence/openpi/tree/15a9616a00943ada6c20a0f158e3adb39df2ccac),
loaded from the official `pi05_libero` checkpoint. We publish the data adapters,
native optimizer loop, frozen-feature/probe preparation, and both evaluation
sides. Other model families are deliberately excluded.

**Release status:** CPU/source contract tests pass. These trainers were extracted
from executed research code, but the portable package has not been retrained or
evaluated end-to-end on GPUs. Model weights, demonstration HDF5 files and our
custom paired scene assets are separate inputs, not included in this Git repo.

## Same model and loss, two data-selection methods

| Setting | Baseline | SMA |
|---|---|---|
| Initial model | Official Pi0.5 LIBERO | Same |
| Train data / normalization | Kitchen TRAIN only | Same |
| Update scope | Full native parameters; no LoRA, no EMA | Same |
| Native objective | All native action channels, valid-time mask | Same |
| GPUs / global batch | 4 / 64, JAX FSDP | Same |
| Peak learning rate | **2.5e-5** (selected U_B recipe) | Same |
| Learning rate clock | 30 s warmup, cosine to 0.2 peak at 1200 s | Same |
| Data selection | Frozen task/episode/time-balanced schedule | 32 uniform + 32 structure-priority |
| Extra forward work | None | Frozen initial features/probes; one probe batch per minute |
| Save nodes | 5, 10, 15, 20 min | Same |
| Inference | Original Pi0.5, 10 actions / 10 denoising steps | Same |

The optimizer sees the original 32-channel native loss. The sampler observes a
detached seven-real-action-coordinate residual. This is **not** a new supervised
loss or consistency regularizer. Probe work is inside the fit clock; initial
feature construction is reported separately. Loading/AOT compilation perform no
retained learning updates and are reported as preparation. Synchronous model
saves are included in `T_fit`; a missed time node is explicitly recorded.

The reference SMA runtime refreshes priorities **every 60 fit seconds**, not every
optimizer step. It keeps eight prefetched immutable batch tickets (four workers ×
prefetch factor two). The core NumPy example's per-step refresh is only an API
example. `fit_GPUh` excludes model loading and initial graph preparation; account
for the entire allocation separately when reporting total cost.

## 1. Install model code in a dedicated directory

Use Linux, Python 3.11 and the CUDA/JAX environment specified by the pinned
OpenPI `pyproject.toml` / `uv.lock`. The reference trainer requires four visible
CUDA devices. It does not allocate resources, use SSH, or launch a scheduler.
Choose a large-data disk for dependencies, weights and outputs.

```bash
# From the SMA repository root. Choose your own storage paths.
export SMA_ROOT="$PWD"
export ASSETS=/your/large-disk/sma
python integrations/pi05/setup_openpi.py --destination "$ASSETS/openpi"
cd "$ASSETS/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install --no-deps -e "$SMA_ROOT"
uv pip install h5py scipy
export PI_PYTHON="$ASSETS/openpi/.venv/bin/python"
cd "$SMA_ROOT"
```

The setup script checks the fixed upstream commit and applies
[`openpi-real7-score.patch`](patches/openpi-real7-score.patch). Only an optional
loss return branch is added; the default native model path is untouched. The
trainer checks native loss equality before its first retained update.

Obtain `gs://openpi-assets/checkpoints/pi05_libero/` using the upstream download
instructions, or reuse your existing complete checkpoint. Do not initialize
from a previously fine-tuned checkpoint when reproducing the baseline comparison.
Weights and underlying components retain their upstream license/access terms.

## 2. Prepare shared TRAIN data

`episodes.json` is a list; one entry per independent TRAIN episode:

```json
[
  {
    "task_id": 1, "cell_id": 0, "room": "kitchen", "split": "train",
    "hdf5": "hdf5/task_01_demo.hdf5", "episode": "demo_1", "length": 74,
    "episode_group_id": "source-demo-1",
    "instruction": "pick up the black bowl and hold it above the table"
  }
]
```

The runnable reference requires all ten task IDs 1..10. The example above shows
one entry, not a complete dataset. HDF5 group `data/<episode>` contains:

- `obs/agentview_rgb`, `obs/eye_in_hand_rgb`: aligned RGB frames;
- `obs/ee_states`: `[T,6]` xyz + axis-angle radians;
- `obs/gripper_states`: `[T,2]` gripper joint positions;
- `actions`: `[T,7]` normalized OSC delta controls, -1 open / +1 close.

Images are flipped in both image axes by the reference Pi adapter before native
OpenPI transforms. Do not apply a second flip in the data export. Padded temporal
targets are masked; no episode boundary is crossed. Paired renderings of the same
source demonstration must share `episode_group_id`.

```bash
"$PI_PYTHON" integrations/pi05/prepare_data.py \
  --episodes "$ASSETS/data/episodes.json" --output "$ASSETS/prepared"
```

For **exact historical data ordering**, additionally supply `--schedule` with the
original frozen `[640000,3]` control schedule, and `--stats` with the original
TRAIN statistics. Without these, the utility creates a new documented balanced
schedule and TRAIN-only empirical quantiles. That is a reproducible new run,
not a claim to reproduce the historical sampled indices or reported SR exactly.
The historical kitchen source had 395 episodes / 42,712 frames. This source
dataset is not downloaded or redistributed by this release.

## 3. Run ordinary fine-tuning

```bash
"$PI_PYTHON" integrations/pi05/run.py init --method baseline \
  --run "$ASSETS/runs/baseline" --openpi "$ASSETS/openpi" \
  --checkpoint "$ASSETS/pi05_libero" --data "$ASSETS/prepared"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PI_PYTHON" integrations/pi05/run.py train \
  --run "$ASSETS/runs/baseline"
```

## 4. Run SMA fine-tuning from the same initial weights

```bash
"$PI_PYTHON" integrations/pi05/run.py init --method sma \
  --run "$ASSETS/runs/sma" --openpi "$ASSETS/openpi" \
  --checkpoint "$ASSETS/pi05_libero" --data "$ASSETS/prepared"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PI_PYTHON" integrations/pi05/run.py prepare \
  --run "$ASSETS/runs/sma"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PI_PYTHON" integrations/pi05/run.py train \
  --run "$ASSETS/runs/sma"
```

Preparation builds frozen visual/language features, the structural graph and
initial real7 residuals. It does not update model weights. Both training commands
write `output/model_5min` ... `output/model_20min`, native optimizer states,
timing logs and consumption logs. Use separate run directories. The short-chain
trainer refuses an interrupted optimizer resume instead of silently restarting
the fit clock; automatic full training recovery is **not implemented**. Sampler
state includes pending tickets, but that alone is not full trainer recovery.

## 5. Evaluate initial, baseline and SMA through the same service

Install [vla-evaluation-harness](https://github.com/allenai/vla-evaluation-harness/tree/4aeb4369640e8019d46af9534ce9b957e486ad38)
at the revision in `dependencies.json` in the policy and simulator environments.
The model environment needs the harness core package; the simulator environment
also needs the compatible LIBERO/robosuite/MuJoCo stack and custom scene assets.
The historical environment includes `regenerate_obs_from_state`; a stock LIBERO
install without that extension cannot run this paired scene benchmark unchanged.

```bash
git clone https://github.com/allenai/vla-evaluation-harness.git "$ASSETS/harness"
git -C "$ASSETS/harness" checkout 4aeb4369640e8019d46af9534ce9b957e486ad38
# Run in each corresponding environment; no simulator is installed by SMA.
pip install -e "$ASSETS/harness"
```

Provide an evaluation assets directory containing `TASKS_30.json`,
`EVAL_DEV_STATES.npz`, and a JSONL paired plan. See
[`evaluation-assets.md`](evaluation-assets.md) for exact schemas. The benchmark
wrapper and success rules are published; scene textures/XML, BDDL objects,
HDF5 files and frozen simulator states are not bundled. This is the remaining
external dependency for reproducing our exact 3×10 environment.

```bash
# Terminal 1: one policy GPU. Use corresponding run statistics for every arm.
export SMA_RUN_DIR="$ASSETS/runs/sma"
CUDA_VISIBLE_DEVICES=0 "$PI_PYTHON" integrations/pi05/scripts/serve_pi05.py \
  --checkpoint "$SMA_RUN_DIR/output/model_10min" --port 8000

# Terminal 2: a compatible simulator environment, concurrent independent sessions.
# checkpoint-id should be the immutable digest from your checkpoint receipt.
python integrations/pi05/evaluate.py --assets "$ASSETS/eval-assets" \
  --plan "$ASSETS/eval-assets/EVAL_PLAN.jsonl" --url ws://127.0.0.1:8000 \
  --checkpoint-id YOUR_CHECKPOINT_DIGEST --workers 6 \
  --output "$ASSETS/eval/sma-10min"
```

Repeat with the baseline checkpoint and the official initial checkpoint, using
new output directories and the **same plan and TRAIN statistics**. Official-native
normalization is a different protocol and must be labeled separately. Serve only
one model per process. Set `MUJOCO_EGL_DEVICE_ID` for your visible simulator GPU;
do not assume CUDA visibility and EGL indices are interchangeable.

Policy computation remains native **B1**; the six simulation processes are
concurrent windows, not a B6 model tensor. The reference execute/replan chunk is
10, denoising steps 10, environment wait 0, horizon 520, control frequency 20 Hz,
with independent per-session policy RNG. This release does not substitute an
untested batched inference path. Uniform and SMA share the entire eval path.

Completed cases are retained. Technical errors stop the run and remain pending;
they are not recorded as task failures. Fix the error, inspect/remove that run's
`STOP` marker and rerun the exact command. At most three attempts per case are
allowed. `SUMMARY.json` reports every room/task cell; partial success rates are
explicitly distinguished from the final rate.

## Source map and validation

- `scripts/train_pi05.py`: ordinary U_B control;
- `scripts/train_pi05_adaptive.py`: same optimizer/loss plus SMA feedback;
- `scripts/prepare_structure.py`: frozen initial features and regional probes;
- `adapters/adaptive_pi_runtime.py`: minute-based refresh and prefetched tickets;
- `scripts/serve_pi05.py`: native model load and per-session inference;
- `adapters/r5_benchmark.py`, `scripts/eval_worker.py`: paired evaluation;
- `evaluation_contract.py`: identity-bound result aggregation.

Tests cover the unchanged trainer update function, baseline/SMA common config,
probe tickets with the public sampler, prefetch-state restoration, fit-clock
boundaries, input rejection, and complete/partial evaluation aggregation. CPU
tests cannot establish CUDA compatibility, native model loading or robot SR.
See `source-provenance.json` and `THIRD_PARTY_NOTICES.md`.
