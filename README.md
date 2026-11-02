<p align="center">
  <img src="assets/sma-logo.png" alt="SMA: a structural manifold with graph nodes and an adaptive path" width="230">
</p>

<h1 align="center">SMA: Pi05 Training and Evaluation</h1>

<p align="center">
  <b>Structural Manifold Adaptation · 结构流形驱动的模型快速适配</b><br>
  <a href="integrations/pi05/README.md">Pi05 Guide</a> ·
  <a href="docs/algorithm.md">Method</a> ·
  <a href="README.zh-CN.md">中文</a>
</p>

SMA accelerates short-budget adaptation by selecting training windows using a
structural graph, coverage and learning residuals. This repository provides
**Pi05 model loading, baseline fine-tuning, SMA fine-tuning and paired evaluation**.
Both methods share the native Pi05 architecture, loss and inference pipeline.

## Method

**Visual-language features + robot state + expert actions → structural graph → adaptive training batches.**

Each SMA batch contains 50% uniform samples and 50% structure-prioritized samples.
Priorities combine regional coverage with residual feedback from training.

## Setup

```bash
git clone https://github.com/diggerslab/Structural-Manifold-Adaptation.git
cd Structural-Manifold-Adaptation

# Use a dedicated directory for the pinned OpenPI model code.
export ASSETS=/your/large-disk/sma
python integrations/pi05/setup_openpi.py --destination "$ASSETS/openpi"
```

Follow the [Pi05 environment setup](integrations/pi05/README.md#1-install-model-code-in-a-dedicated-directory),
then run the commands below in that environment. Training uses four GPUs and a
global batch of 64. Supply the official `pi05_libero` weights and your TRAIN data;
the paired 3×10 evaluation also requires its scene assets and fixed case plan.

## Pi05 Training

Prepare the shared training data:

```bash
python integrations/pi05/prepare_data.py \
  --episodes "$ASSETS/data/episodes.json" --output "$ASSETS/prepared"
```

**Baseline fine-tuning**

```bash
python integrations/pi05/run.py init --method baseline \
  --run "$ASSETS/runs/baseline" --openpi "$ASSETS/openpi" \
  --checkpoint "$ASSETS/pi05_libero" --data "$ASSETS/prepared"
CUDA_VISIBLE_DEVICES=0,1,2,3 python integrations/pi05/run.py train \
  --run "$ASSETS/runs/baseline"
```

**SMA fine-tuning**

```bash
python integrations/pi05/run.py init --method sma \
  --run "$ASSETS/runs/sma" --openpi "$ASSETS/openpi" \
  --checkpoint "$ASSETS/pi05_libero" --data "$ASSETS/prepared"
CUDA_VISIBLE_DEVICES=0,1,2,3 python integrations/pi05/run.py prepare \
  --run "$ASSETS/runs/sma"
CUDA_VISIBLE_DEVICES=0,1,2,3 python integrations/pi05/run.py train \
  --run "$ASSETS/runs/sma"
```

Both runs start from the same official weights, use a peak learning rate of
**2.5e-5**, and save **5/10/15/20-minute** checkpoints. SMA preparation extracts
frozen features and builds the graph before training.

## Pi05 Evaluation

Start the policy service:

```bash
export SMA_RUN_DIR="$ASSETS/runs/sma"
CUDA_VISIBLE_DEVICES=0 python integrations/pi05/scripts/serve_pi05.py \
  --checkpoint "$SMA_RUN_DIR/output/model_10min" --port 8000
```

In a second terminal, run concurrent simulator windows:

```bash
export SIM_PYTHON=/path/to/simulator-env/bin/python
python integrations/pi05/evaluate.py \
  --assets "$ASSETS/eval-assets" --plan "$ASSETS/eval-assets/EVAL_PLAN.jsonl" \
  --url ws://127.0.0.1:8000 --checkpoint-id YOUR_CHECKPOINT_DIGEST \
  --sim-python "$SIM_PYTHON" --workers 6 --output "$ASSETS/eval/sma-10min"
```

Evaluate the initial, baseline and SMA checkpoints with the same case plan and
normalization, using a separate output directory for each. `SUMMARY.json`
contains overall and per-scene/task success rates. Policy inference is native
B1; simulation sessions run concurrently.

[Data format and full commands](integrations/pi05/README.md) ·
[Evaluation asset format](integrations/pi05/evaluation-assets.md) ·
[Validation](docs/validation.md) · [MIT License](LICENSE)
