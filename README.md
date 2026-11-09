# OLMo-2 1B EAGLE-3 training with SpecForge

This repository configures SpecForge's native offline EAGLE-3 pipeline for
`allenai/OLMo-2-0425-1B-Instruct` and UltraChat 200k. SpecForge is pinned as a
Git submodule. Project code downloads and normalizes data, registers the OLMo-2
assistant-mask delimiters for feature capture, composes typed YAML modules, and
invokes `specforge train`; it does not implement a trainer.

## Status

The previous project README recorded a full pipeline smoke on one NVIDIA RTX
3060 with driver 580.178.04, Python 3.11.17, and PyTorch 2.13.0+cu130. It
reported 8 train and 8 evaluation examples, two optimizer steps, checkpoint
export, SGLang inference, and peak sampled GPU memory of 9,021 MiB. This audit
also passed the opt-in GPU smoke: 8 train and 8 evaluation examples, two
optimizer steps, checkpoint resume, export, SGLang inference, local MLflow, and
metric plots. The historical memory and metric values above were not asserted
to match. Smoke-scale results do not establish full-dataset training quality.
See [implementation
status](docs/implementation-status.md) and [project goals](docs/project-goals.md).

## Installation

Use Python 3.11 or newer and `uv`. Clone with the SpecForge submodule:

```bash
# Clone the project repository with its SpecForge submodule, then enter it.
test -f .env || cp .env.example .env
# Edit the EAGLE3_*, HF_HOME, DVC_*, and cache paths in .env to external storage.
uv sync --locked --extra pipeline --group dev
uv run --env-file .env pre-commit install
```

Pre-commit checks whitespace, YAML/TOML syntax, merge conflicts, large files,
filename collisions, broken symlinks, and private keys. Pinned Black, isort,
flake8, and Prettier hooks format and lint project files. Black runs with
`--workers 1` directly, without a shell wrapper or a process pool. After setup,
run `uv run --env-file .env pre-commit run --all-files` to check tracked files.

The pipeline reads its paths from `.env`. `EAGLE3_DATA_DIR` holds downloaded
datasets and captured features; `EAGLE3_OUTPUT_DIR` holds checkpoints, plots,
configs, and inference results; `EAGLE3_CACHE_DIR` holds processed dataset and
feature caches. `HF_HOME` holds Hugging Face downloads, `TMPDIR` holds temporary
files, and the CUDA/Torch cache variables move compiler caches off the
workspace. `PYTHONDONTWRITEBYTECODE=1` prevents Python from writing `.pyc`
bytecode files. The CLI creates configured directories and rejects relative
paths. Do not replace an existing `.env` with the example; merge the required
path variables into it.

The pipeline extra installs the pinned SpecForge submodule and its runtime
dependencies, including PyTorch and SGLang. CUDA wheel availability depends on
the host driver and the package index used by uv; verify the selected PyTorch
build before starting GPU stages:

```bash
uv run --env-file .env python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CUDA unavailable")'
```

Tracking settings can be exported in the shell or loaded explicitly with
`uv run --env-file .env eagle3-olmo2 ...`; the application does not load `.env`
automatically. MLflow credentials remain process environment values and are
excluded from the resolved run configuration and tracker parameters.
`.env.example` documents the MLflow variables. GPU stages need a CUDA-capable
PyTorch install, the CUDA Toolkit compiler (`/usr/local/cuda/bin/nvcc` for the
pinned SGLang JIT path), a C++ compiler, model/dataset access, and enough GPU
memory. The CUDA version shown by `nvidia-smi` describes driver support; it
does not confirm that `nvcc` is installed. The reported smoke used an RTX 3060
with 12 GiB.

## Data and DVC

The default source is the public `HuggingFaceH4/ultrachat_200k` dataset. The
pipeline writes its working files to the external paths configured in `.env`.
DVC cache and remote storage are separate external directories; configure them
in `.env` before initializing DVC:

```bash
test -f .env || cp .env.example .env
uv run --env-file .env dvc init
uv run --env-file .env eagle3-olmo2-dvc-setup
uv run dvc cache dir
uv run dvc remote list
```

`eagle3-olmo2-dvc-setup` writes machine-specific cache/remote settings to
`.dvc/config.local`, which must not be committed. This repository currently has
no DVC artifact pointers. DVC cache/remote configuration does not relocate the
pipeline's working files; their locations come from `EAGLE3_DATA_DIR` and
`EAGLE3_OUTPUT_DIR`. Publish shared DVC artifacts only after choosing DVC
pointer paths that match the external-storage layout.

Use a shared remote for collaboration. The local remote in `.env.example` is a
machine-local example and does not transfer artifacts to another machine by
itself. `prepare-features`, `train`, `inference`, and `full` run `dvc pull`
when `.dvc` pointers or a `dvc.lock` exist; a `dvc.yaml` without a lockfile is
not treated as cached data. Pull failures stop the command. Without DVC refs,
the existing Hugging Face data preparation workflow continues normally.

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

Hydra composes the existing model, dataset, training, and evaluation modules.
Use the `eagle3-olmo2-hydra` entry point for dotted overrides, for example:

```bash
uv run --env-file .env eagle3-olmo2-hydra full training.max_steps=2 cli.limit=16
```

The command writes the resolved module configs under the configured output
directory and forwards to the existing SpecForge pipeline. Full training and
inference still require CUDA, model access, and DVC artifacts when refs exist.
See [implementation status](docs/implementation-status.md) for validation
limits and current DVC remote limitations.

## Run the pipeline

Run a short data preparation first:

```bash
uv run --env-file .env eagle3-olmo2 prepare-data --limit 16
uv run --env-file .env eagle3-olmo2 prepare-features --split all
uv run --env-file .env eagle3-olmo2 compose
```

The feature step captures train and validation features separately using
SpecForge's `prepare_hidden_states.py`, with capture batch size 1 and a
conservative SGLang static-memory fraction capped from detected free VRAM and
an estimate of target-weight memory. Training uses microbatch 1 and CPU
optimizer-state offload; available and peak VRAM are recorded per stage. The
generated run YAML is written under the directory configured by
`EAGLE3_OUTPUT_DIR`.

GPU-backed commands check CUDA availability before downloading data or models.
`prepare-data` and the config composer remain usable on a CPU-only host.

Run a full short smoke training and evaluation:

```bash
uv run --env-file .env eagle3-olmo2 full --limit 16
```

Run training later from prepared feature files:

```bash
uv run --env-file .env eagle3-olmo2 train
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
uv run --env-file .env eagle3-olmo2 inference
```

`full` runs this smoke automatically after training. Its response and SGLang
`spec_accept_length`, when returned by the pinned server, are saved to
the `EAGLE3_OUTPUT_DIR` directory configured in `.env`.

## Tracking and checkpoints

With `MLFLOW_TRACKING_URI` set in the process environment or
`infrastructure.mlflow_tracking_uri` set in the experiment YAML, the composer
enables SpecForge's MLflow tracker
and logs the resolved run config, training/evaluation metrics, optimizer step,
performance values, and capture/training/evaluation VRAM metrics. The runner
also saves scalar history and SVG plots locally and attaches them to MLflow
after training, including the current Git commit SHA as `project.git_commit`.
Without it, SpecForge logs locally to stdout while the runner saves local
metric history/plots for metrics emitted by the trainer. When the native logs
contain those metrics, the plot set includes `train_loss.svg`,
`validation_loss.svg`, `acceptance_length.svg`, and per-depth acceptance-rate
plots. Native
checkpointing uses `save_interval`, `max_checkpoints`, and a final checkpoint;
SpecForge's checkpoint manager selects best checkpoints by
`eval/simulated_acc_len` and writes resume state. Checkpoints are under the
configured `output_dir`.

Run the CPU checks and configured quality hooks with:

```bash
uv run --env-file .env --locked --extra pipeline --group dev pytest
uv run --env-file .env pre-commit run --all-files
uv lock --check
```

The native trainer exposes acceptance-rate metrics at speculation depths and
simulated acceptance length during offline evaluation. The previous README
recorded `eval/simulated_acc_len` 0.0604 at step 2 and an SGLang response of
“2 plus 2 equals 4.” with speculative acceptance length 1.1429. These values
are historical reports and were not reproduced during this audit.

## Testing and limitations

CPU-only unit tests are in `tests/`. GPU integration tests are marked separately
and require the model downloads, CUDA, SpecForge dependencies, and an NVIDIA
GPU. Run the CPU tests with:

```bash
uv run --env-file .env --locked --extra pipeline --group dev pytest
```

The end-to-end GPU test in `tests/gpu/` is opt-in because it downloads the model
and dataset, trains, exports, and starts SGLang. The project previously reported
a full pipeline smoke at small scale on an RTX 3060. Run the acceptance test on
suitable CUDA hardware with:

```bash
EAGLE3_GPU_TEST=1 uv run --env-file .env --locked --extra pipeline --group dev pytest tests/gpu/test_pipeline_smoke.py -v
```

The opt-in GPU smoke passed, including model/tokenizer loading, OLMo chat-template
masking, train/eval feature capture, two optimizer steps, evaluation, checkpoint
resume, draft export, SGLang inference, MLflow metrics, and plot artifacts.
CPU unit tests and the GPU integration test cover separate parts of the project;
see the implementation status for remaining shared-storage and external-service
work.
