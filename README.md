# OLMo-2 1B EAGLE-3 with SpecForge

Train a one-layer EAGLE-3 drafter for `allenai/OLMo-2-0425-1B-Instruct` on UltraChat
200k. During training, the frozen OLMo-2 backbone computes activations from
blocks 1, 7, and 12 (zero-based) and its final layer for each batch.
SpecForge trains the drafter on assistant response tokens and saves resumable
checkpoints. During generation, OLMo-2 verifies the drafter's proposed tokens.

UltraChat SFT → normalized conversations → token IDs and loss masks →
live OLMo-2 batches → SpecForge checkpoint → SGLang draft export.

## 1. Set up the environment

Run the commands in Bash from the repository root on Linux with Git and
[uv](https://docs.astral.sh/uv/). Keep the SpecForge submodule at the revision pinned by
the repository. On Talos, point `UV_PROJECT_ENVIRONMENT` to the project
environment in `/opt/venvs` before running uv.

```bash
git submodule update --init --recursive
uv python install 3.11
uv sync --locked --extra pipeline --group dev
test -f .env || cp .env.example .env
```

Use writable, absolute paths in `.env`. The [environment template](.env.example) lists
the storage and compiler cache settings.

| Variable                             | Contents                                                                   |
| ------------------------------------ | -------------------------------------------------------------------------- |
| `EAGLE3_DATA_DIR`                    | Prepared JSONL, validation/test features, and their DVC refs               |
| `EAGLE3_OUTPUT_DIR`                  | Checkpoints, metrics, configs, and evaluation reports                      |
| `EAGLE3_CACHE_DIR` and `HF_HOME`     | Tokenized training inputs, target index, and Hugging Face downloads        |
| `DVC_CACHE_DIR` and `DVC_REMOTE_DIR` | Separate, non-nested directories for the DVC object cache and local remote |

The pipeline resolves pinned dataset, model, and tokenizer revisions. The first run
needs access to Hugging Face to populate the caches. See the
[dataset pin](configs/datasets/ultrachat_200k.yaml) and
[model pin](configs/models/olmo2_1b.yaml).

Feature capture, training, validation, and inference use one NVIDIA GPU. The locked
Linux environment uses PyTorch 2.13 with CUDA 13 and SGLang 0.5.18. Install a compatible
NVIDIA driver, the CUDA Toolkit with `nvcc`, and a C++ compiler for SGLang kernel
compilation.

```bash
uv run --env-file .env python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

The last value must be `True` for GPU stages. Training keeps the frozen target
weights in CPU RAM by default and transfers them to the GPU on demand. This reduces
peak VRAM at the cost of transfers on each batch. With sufficient VRAM, pass
`model.train_target_cpu_offload=false` to keep the target on the GPU. Use the pilot
at the intended context length to estimate VRAM and runtime for a full run.

To record experiments in MLflow, set `MLFLOW_TRACKING_URI` in `.env` to an
absolute SQLite URI such as `sqlite:////mnt/ml-data/eagle3/outputs/mlflow.db`
and set `MLFLOW_EXPERIMENT_NAME` to group runs. Training writes directly to
SQLite.

## 2. Run smoke and pilot training

Run the smoke first. After it completes, run the pilot to check resource use
at a larger scale. Each command prepares the train split and captures validation
features before training. To run individual stages instead, see
[Run training stages separately](#run-training-stages-separately).

```bash
uv run --env-file .env eagle3-olmo2 --profile smoke full &&
uv run --env-file .env eagle3-olmo2 --profile pilot full
```

After pilot training, run validation to inspect offline metrics and compare
target-only and speculative generation on the development prompts:

```bash
uv run --env-file .env eagle3-olmo2 --profile pilot validate
```

The `full` **command** prepares train and validation data, captures validation
features, trains with live OLMo-2 activations, registers the best and latest
checkpoints in DVC, and runs one generation request. The **profile** sets the scale
and output directory.

| Profile |        Train rows | Validation rows | Effective batch | Optimizer steps                   |
| ------- | ----------------: | --------------: | --------------: | --------------------------------- |
| `smoke` |          Up to 16 |         Up to 8 |               4 | 2                                 |
| `pilot` |         Up to 512 |       Up to 128 |               1 | One epoch, based on eligible rows |
| `full`  | All eligible rows |     Up to 1,024 |               1 | One epoch, based on eligible rows |

Each profile writes to `$EAGLE3_OUTPUT_DIR/<profile>/`. A completed end-to-end run has
`training-status.json` with status `complete`, a `checkpoint-manifest.json`, and
`inference-smoke.json`. Use a separate output directory for each training attempt.

## 3. Run full training

Review the pilot's `training-plan.json`, `training-status.json`, `gpu-memory.json`, and
`train-input-manifest.json` before approving the full run. The approval directory
must contain a completed live-capture pilot and its generation smoke result. The
runner projects validation capture storage for the working files, DVC cache, and
remote. Training activations are never written to disk.

```bash
uv run --env-file .env eagle3-olmo2 --profile full full \
  'cli.approved_pilot=${oc.env:EAGLE3_OUTPUT_DIR}/pilot'
```

The full profile keeps the train split and optimizer steps uncapped. It processes up to
1,024 tokens per sequence, uses AdamW with CPU optimizer offload and a cosine schedule,
and evaluates and saves every 1,000 steps. SpecForge also saves the final step.

Pilot and full share the optimizer settings, including a learning rate of
`1e-4`, 1% warmup, and TTT length 8. The cosine schedule covers the actual
number of optimizer steps in each profile.
Treat one full epoch as the first quality baseline. Compare checkpoint curves
for validation loss and acceptance, then confirm the selected checkpoint with
generation acceptance and target-only parity. Consider a longer run if
validation still improves at the end. Set the intended epoch count before
starting a comparison run so its scheduler uses that horizon.

## 4. Validate the trained checkpoint

```bash
uv run --env-file .env eagle3-olmo2 --profile full validate
```

This runs offline evaluation on the validation features, then compares target-only and
speculative generation on [24 development prompts](configs/prompts/quick.jsonl). It
selects the best checkpoint by simulated acceptance length, with latest as the fallback.
Give `cli.checkpoint` the absolute path to a registered step directory to evaluate a
different checkpoint.

After checkpoint selection, capture and evaluate the held-out test partition:

```bash
uv run --env-file .env eagle3-olmo2 --profile full prepare-data cli.split=test
uv run --env-file .env eagle3-olmo2 --profile full prepare-features cli.split=test
uv run --env-file .env eagle3-olmo2 --profile full validate cli.split=test
```

The final command uses the [separate final prompt suite](configs/prompts/final.jsonl).
Look in `validate-*/offline-eval.json` for loss and simulated acceptance, and in
`validate-*/generation.html` for responses and the generation comparison.

## 5. Generate text or export a draft

```bash
uv run --env-file .env eagle3-olmo2 --profile full inference \
  'cli.prompt="Explain why the sky is blue."'
```

The default `speculative` mode pulls the selected checkpoint from DVC and exports it to
`inference-*/draft_sglang/`. Set `cli.mode=target-only` to run with just OLMo-2, or
`cli.mode=compare` to run both modes sequentially and compare greedy output token IDs
and latency. Run comparison on a prompt suite with
`cli.suite=configs/prompts/quick.jsonl` in place of the prompt argument. Each JSONL row
needs a unique `id`, `category`, `language`, `context_bucket`, `messages` ending in a
user turn, a positive `max_new_tokens`, and `check_type` (`manual`, `exact`, or `json`).
The last two checks also need `expected`.

Open `inference-*/generation.html` for the human-readable report. The accompanying JSON
and JSONL contain token IDs, per-request metrics, and errors. The server uses localhost
port 30000 by default. Set `evaluation.inference_port` if it is occupied. The pinned
SGLang response provides acceptance counters in aggregate, and `cli.trace=true` reports
per-round tracing as unavailable.

To export weights without generating text, use the native SpecForge exporter on a
restored, registered checkpoint:

```bash
read -r -p 'Absolute path to the restored checkpoint: ' CHECKPOINT
uv run --env-file .env specforge export --to sglang \
  --checkpoint "$CHECKPOINT" \
  --draft-config configs/models/olmo2_1b_eagle3.json \
  --vocab-mapping "$CHECKPOINT/eagle3-vocab-mapping.pt" \
  --output-dir "$(dirname "$CHECKPOINT")/draft_sglang"
```

The resulting directory contains the draft weights and configuration for SGLang. Pair it
with the OLMo-2 target and tokenizer revisions recorded in the run's
`model-snapshot.json`.

## Inspect experiment metrics

Every training run writes `metrics.jsonl`, `metrics-schema.json`, and SVG plots under
`plots/` when those metrics are available. Compare `train/loss` and `eval/loss` across
optimizer steps. `eval/simulated_acc_len` estimates offline acceptance. The
`generation/*` counters record SGLang's observed draft acceptance and latency.

To inspect logged runs in a browser, start the local MLflow server and open
`http://127.0.0.1:5000`.

```bash
uv run --env-file .env sh -eu -c 'mlflow server --backend-store-uri "$MLFLOW_TRACKING_URI" --host 127.0.0.1'
```

## Run training stages separately

Use this route in place of the end-to-end pilot if you need to prepare data
on a CPU host, inspect the training plan, or start from existing features.

```bash
uv run --env-file .env eagle3-olmo2 --profile pilot prepare-data cli.split=train
uv run --env-file .env eagle3-olmo2 --profile pilot prepare-data cli.split=eval
uv run --env-file .env eagle3-olmo2 --profile pilot prepare-features cli.split=eval
uv run --env-file .env eagle3-olmo2 --profile pilot preflight
uv run --env-file .env eagle3-olmo2 --profile pilot train
```

- `prepare-data` writes JSONL conversations and checksum manifests. Train comes from
  `train_sft`. Validation (`eval`) and the final `test` partition come from `test_sft`
  and require the prepared train split for overlap checks.
- `prepare-features` supports `eval` and `test`. It restores the matching JSONL
  refs, writes hidden states and a capture manifest, and registers the split in DVC.
- `preflight` restores the train JSONL and validation features. On CPU, it tokenizes
  the train split, filters rows without usable assistant tokens, builds the
  vocabulary map, and writes `training-plan.json` with the expected optimizer steps
  and the size of the token cache. It needs access to the target snapshot and tokenizer.
- `train` uses the train JSONL and the validation features. The local preprocessing
  cache under `EAGLE3_CACHE_DIR/train-inputs` holds only token IDs, loss masks and the
  vocabulary map. A cache miss rebuilds those inputs. Existing offline train
  artifacts and their DVC objects are outside this workflow.
- Use the end-to-end pilot command to produce the generation smoke report required
  for full-run approval. For staged training with the `full` profile, pass
  `cli.approved_pilot` to `train`.

## Resume an interrupted run

Use a retained, complete live-capture `*-stepN` checkpoint from before the final
step. Keep the original train JSONL, tokenizer, target, preprocessing cache,
and training schedule. Use a fresh `infrastructure.output_dir` and set
`infrastructure.resume_from_checkpoint` to the original step directory.
Checkpoints from the earlier offline workflow remain usable for inference when
registered with their vocabulary map. Optimizer resume requires a live-capture
checkpoint whose training-input manifest matches the configured run.

## Local DVC recovery

Preparation creates a DVC workspace in each artifact's parent directory. Keep the
`*.dvc` refs and access to the `local-storage` object store together. The pipeline pulls
from that store and checks the refs before consuming data, features, or checkpoints.

If the cache or remote path moves, set `DVC_WORKSPACE` to the absolute path of the
existing artifact parent and apply the new paths from `.env`:

```bash
uv run --env-file .env eagle3-olmo2-dvc-setup "$DVC_WORKSPACE"
```

Set `DVC_REF` to the `*.dvc` filename in that workspace. Use `pull` to restore the
artifact, or `push` to finish an interrupted upload before retrying a pipeline stage.

```bash
uv run --env-file .env dvc --cd "$DVC_WORKSPACE" pull -r local-storage "$DVC_REF"
```

If a prepared artifact conflicts with a changed dataset revision, seed, or row limit,
choose new data and feature paths. The manifests record the original selection and
checksums.

## Tests and code quality

```bash
uv run --locked --extra pipeline --group dev pytest --ignore=tests/gpu
uv run --locked --group dev pre-commit run --all-files
```

On a CUDA host with model and dataset access, run the opt-in integration test for
capture, training, resume, export, and inference:

```bash
EAGLE3_GPU_TEST=1 uv run --env-file .env --locked --extra pipeline --group dev pytest tests/gpu/test_pipeline_smoke.py -v
```

The project CLI lives in
[`src/eagle3_olmo2/`](src/eagle3_olmo2/), the trainer and exporter in
[`specforge/`](specforge/), and the integration tests in [`tests/gpu/`](tests/gpu/).
