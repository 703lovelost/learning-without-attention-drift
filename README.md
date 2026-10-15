# OLMo-2 1B EAGLE-3 training with SpecForge

This repository configures SpecForge's native offline EAGLE-3 pipeline for
`allenai/OLMo-2-0425-1B-Instruct` and UltraChat 200k. SpecForge is pinned as a
Git submodule. Project code downloads and normalizes data, registers the OLMo-2
assistant-mask delimiters for feature capture, composes typed YAML modules, and
invokes `specforge train`; it does not implement a trainer.

## Status

The full pipeline smoke is **verified on one NVIDIA RTX 3060** with driver
580.178.04, Python 3.11.17, and PyTorch 2.13.0+cu130. It prepared 8 train and 8
evaluation examples, captured features, trained for two optimizer steps,
evaluated, wrote checkpoints, exported the drafter, and generated a response
through SGLang EAGLE-3. Peak sampled GPU memory was 9,021 MiB of 12,288 MiB.
This validates the wiring at smoke scale; it does not establish full-dataset
training quality or validate a resumed GPU training run. See
[the plan](docs/IMPLEMENTATION_PLAN.md), [progress log](docs/PROGRESS.md), and
[decisions](docs/DECISIONS.md).

## Installation

Use Python 3.11 or newer. For a host with an NVIDIA driver that supports CUDA
13.0, create an isolated environment and install the CUDA 13.0 PyTorch wheel
first. SpecForge pins PyTorch 2.13.0, so this also matches its declared
dependency:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cu130
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
git submodule update --init --recursive
python -m pip install -e ./specforge
```

Install MLflow to run the integration coverage for tracking both enabled and
disabled, or to use remote tracking. Set credentials/URI in the environment;
`.env.example` documents the supported variables. The project does not load
`.env` automatically.

```bash
python -m pip install mlflow
```

## Configuration

The experiment YAML composes model, dataset, training, and evaluation modules:

- `configs/models/olmo2_1b.yaml`: Hugging Face model/tokenizer revision, target
  dtype, OLMo-2 target feature layers, and draft architecture size.
- `configs/datasets/ultrachat_200k.yaml`: UltraChat train/evaluation splits,
  deterministic seed, sequence length, templates, and feature paths.
- `configs/training/eagle3_single_gpu.yaml`: one-layer pre-norm EAGLE-3,
  microbatch, accumulation, mixed precision, AdamW optimizer, optimizer
  offload, TTT length, loss controls, and save/evaluation cadence. The composer
  checks the project-level optimizer selector against SpecForge's built-in
  AdamW implementation and removes it before native schema validation.
- `configs/evaluation/default.yaml`: validation cadence and evaluation limits.
- `configs/experiments/olmo2_1b_baseline.yaml`: module references, output path,
  MLflow settings, and optional resume checkpoint.

The OLMo-2 instruct tokenizer renders prompts itself. The feature adapter
registers its `<|user|>`, `<|assistant|>`, and `<|endoftext|>` boundaries with
SpecForge for assistant-only loss masking, following the [model card](https://huggingface.co/allenai/OLMo-2-0425-1B-Instruct).
If `tokenizer_path` is changed, feature preparation checks that its token IDs
and chat template match the target tokenizer.

The config composer resolves the configured model revision to a local Hugging
Face snapshot. Its immutable snapshot path is written into the generated
SpecForge YAML. Change the model and dataset `revision` fields from `main` to
immutable commit IDs for a fully pinned reproduction.

## Run the pipeline

Set up the source import path and run a short data preparation first:

```bash
export PYTHONPATH="$PWD/src"
python -m eagle3_olmo2.cli prepare-data --limit 16
python -m eagle3_olmo2.cli prepare-features --split all
python -m eagle3_olmo2.cli compose
```

The feature step captures train and validation features separately using
SpecForge's `prepare_hidden_states.py`, with capture batch size 1 and a
conservative SGLang static-memory fraction capped from detected free VRAM and
an estimate of target-weight memory. Training uses microbatch 1 and CPU
optimizer-state offload; available and peak VRAM are recorded per stage. The
generated run YAML can be inspected at
`outputs/olmo2_1b_eagle3/run.yaml`.

GPU-backed commands check CUDA availability before downloading data or models.
`prepare-data` and the config composer remain usable on a CPU-only host.

Run a full short smoke training and evaluation:

```bash
python -m eagle3_olmo2.cli full --limit 16
```

Run training later from prepared feature files:

```bash
python -m eagle3_olmo2.cli train
```

`full` and `train` invoke the native `specforge train --config ...` CLI. Set
`training.max_steps` low in the training YAML for the first optimizer-step
smoke. Before training, the runner counts prepared feature files and adjusts
the evaluation interval to divide the expected final optimizer step, so the
last step is evaluated even when a short dataset ends before the configured
period. To resume all optimizer, scheduler, step, and RNG state, set
`infrastructure.resume_from_checkpoint` in the experiment YAML or edit the
generated SpecForge config's `training.resume_from`. If resuming from an older
step than the recorded best checkpoint, the runner clears that future best
alias before training so the resumed timeline can select its own best by
acceptance length.

Run the native SGLang EAGLE-3 export and one target-plus-drafter request after a
successful training run with:

```bash
python -m eagle3_olmo2.cli inference
```

`full` runs this smoke automatically after training. Its response and SGLang
`spec_accept_length`, when returned by the pinned server, are saved to
`outputs/olmo2_1b_eagle3/inference-smoke.json`.

## Tracking and checkpoints

With `MLFLOW_TRACKING_URI` set, the composer enables SpecForge's MLflow tracker
and logs the resolved run config, training/evaluation metrics, optimizer step,
performance values, and capture/training/evaluation VRAM metrics. The runner
also saves scalar history and SVG plots locally and attaches them to MLflow
after training. Without it, SpecForge logs locally to stdout while the runner
saves local metric history/plots. Native
checkpointing uses `save_interval`, `max_checkpoints`, and a final checkpoint;
SpecForge's checkpoint manager selects best checkpoints by
`eval/simulated_acc_len` and writes resume state. Checkpoints are under the
configured `output_dir`.

The native trainer exposes acceptance-rate metrics at speculation depths and
simulated acceptance length during offline evaluation. The verified RTX 3060
smoke selected `eval/simulated_acc_len` 0.0604 at step 2. The best and latest
aliases point to that step's checkpoint. Its SGLang inference returned
“2 plus 2 equals 4.” with a speculative acceptance length of 1.1429.

## Testing and limitations

CPU-only unit tests are in `tests/`. GPU integration tests are marked separately
and require the model downloads, CUDA, SpecForge dependencies, and an NVIDIA
GPU. Run the CPU tests with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

The end-to-end GPU test in `tests/gpu/` is opt-in because it downloads the model
and dataset, trains, exports, and starts SGLang. The full pipeline has now been
run manually at smoke scale on the RTX 3060. Run the acceptance test on suitable
CUDA hardware with:

```bash
EAGLE3_GPU_TEST=1 PYTHONPATH=src python -m unittest discover -s tests/gpu -v
```

The manual smoke verified model/tokenizer loading, OLMo chat-template masking,
target feature capture, two optimizer steps, evaluation, checkpoint writing,
draft export, and inference. The separate GPU test's resume branch has not yet
been verified. See `docs/PROGRESS.md` for the run directory, measured VRAM, and
metrics.
