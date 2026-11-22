"""Run the modular OLMo-2/SpecForge EAGLE-3 pipeline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

from eagle3_olmo2.config import (
    DEFAULT_EXPERIMENT,
    ROOT,
    _expand_env,
    compose,
    experiment_module_path,
    validate_specforge,
)
from eagle3_olmo2.data import sha256_file
from eagle3_olmo2.dvc import ensure_dvc_artifacts_available, register_artifact
from eagle3_olmo2.metrics import METRIC_SCHEMA, canonical_metrics
from eagle3_olmo2.paths import ensure_configured_storage_dirs, expand_env, resolve_path
from eagle3_olmo2.readiness import (
    asset_identity,
    estimate_capture_storage,
    feature_manifest,
    plan_training,
    write_json,
)
from eagle3_olmo2.train_data import (
    TRAIN_INPUT_DIGEST,
    TRAIN_INPUT_MANIFEST,
    prepare_training_inputs,
)


def read_dataset_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return expand_env(yaml.safe_load(stream)["dataset"])


def _count_jsonl_rows(path: Path) -> int:
    """Count non-empty prepared rows so capture limits never exceed the file."""
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def experiment_output_dir(experiment_path: Path) -> Path:
    with experiment_path.open(encoding="utf-8") as stream:
        infrastructure = expand_env(yaml.safe_load(stream).get("infrastructure", {}))
    output_dir = infrastructure.get(
        "output_dir",
        os.getenv("EAGLE3_OUTPUT_DIR", "outputs/olmo2_1b_eagle3"),
    )
    return resolve_path(output_dir)


def _resolve_snapshot(repo_or_path: str, revision: str, *, relative_to: Path) -> str:
    candidate = Path(repo_or_path)
    if not candidate.is_absolute():
        candidate = relative_to / candidate
    if candidate.exists():
        return str(candidate.resolve())
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_or_path, revision=revision)


def _target_head_compatible_snapshot(model_path: str, output_dir: Path) -> str:
    """Add the index expected by SpecForge's TargetHead to single-file HF models.

    SpecForge's target embedding loader supports a single ``model.safetensors``
    file, while its frozen LM-head loader requires an index. Keep the downloaded
    Hugging Face snapshot immutable and create a lightweight symlink overlay in
    the experiment output directory when that legacy path is needed.
    """
    source = Path(model_path)
    if (
        list(source.glob("*.index.json"))
        or not (source / "model.safetensors").is_file()
    ):
        return model_path

    overlay = output_dir / "model-with-index"
    if (overlay / "model.safetensors.index.json").is_file():
        return str(overlay.resolve())
    from safetensors import safe_open

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".target-index-", dir=output_dir) as work:
        staging = Path(work) / overlay.name
        staging.mkdir()
        for asset in source.iterdir():
            (staging / asset.name).symlink_to(asset.resolve())
        with safe_open(str(source / "model.safetensors"), framework="pt") as weights:
            weight_map = {key: "model.safetensors" for key in weights.keys()}
        write_json(staging / "model.safetensors.index.json", {"weight_map": weight_map})
        try:
            staging.rename(overlay)
        except OSError:
            if not (overlay / "model.safetensors.index.json").is_file():
                raise
    return str(overlay.resolve())


def resolve_model_assets(
    model_config_path: Path, output_dir: Path, *, cache_dir: Path | None = None
) -> tuple[str, str]:
    """Resolve model and tokenizer revisions to immutable local snapshots."""
    with model_config_path.open(encoding="utf-8") as stream:
        model = yaml.safe_load(stream)["model"]
    model_id = model["target_model_path"]
    resolved_model = _resolve_snapshot(
        model_id, model.get("revision", "main"), relative_to=model_config_path.parent
    )
    overlay_root = output_dir
    if cache_dir is not None:
        snapshot_key = hashlib.sha256(resolved_model.encode()).hexdigest()
        overlay_root = cache_dir / "target-overlays" / snapshot_key
    indexed_model = _target_head_compatible_snapshot(resolved_model, overlay_root)
    tokenizer_id = model.get("tokenizer_path") or model_id
    resolved_tokenizer = _resolve_snapshot(
        tokenizer_id,
        model.get("tokenizer_revision", model.get("revision", "main")),
        relative_to=model_config_path.parent,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model-snapshot.json").write_text(
        json.dumps(
            {
                "model_id": model_id,
                "requested_revision": model.get("revision", "main"),
                "resolved_model_path": resolved_model,
                "training_model_path": indexed_model,
                "tokenizer_id": tokenizer_id,
                "tokenizer_revision": model.get(
                    "tokenizer_revision", model.get("revision", "main")
                ),
                "resolved_tokenizer_path": resolved_tokenizer,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return indexed_model, resolved_tokenizer


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _configure_tracking_environment(infrastructure: dict[str, object]) -> None:
    """Pass the configured MLflow URI to child processes without serializing it."""
    expanded = _expand_env(infrastructure)
    tracking_uri = expanded.get("mlflow_tracking_uri")
    if tracking_uri and not os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = str(tracking_uri)


def require_cuda() -> None:
    """Fail before downloads when a CUDA-backed pipeline stage cannot run."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch with CUDA support is required for feature capture, training, "
            "and inference. Install the project GPU environment first."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device is available. Check the NVIDIA driver and CUDA-enabled "
            "PyTorch before running feature capture, training, inference, or full."
        )


def _write_svg(path: Path, points: list[tuple[int, float]], title: str) -> None:
    if not points:
        return
    width, height, margin = 800, 360, 48
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]
    xspan = max(1, max(xs) - min(xs))
    yspan = max(1e-12, max(ys) - min(ys))
    coords = [
        (
            margin + (x - min(xs)) * (width - 2 * margin) / xspan,
            height - margin - (y - min(ys)) * (height - 2 * margin) / yspan,
        )
        for x, y in points
    ]
    polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<text x="{margin}" y="26" font-family="sans-serif" font-size="17">{title}</text>'
        f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{polyline}"/>'
        f'<text x="{margin}" y="{height-10}" font-family="sans-serif" '
        f'font-size="12">step {min(xs)}</text>'
        f'<text x="{width-margin-50}" y="{height-10}" '
        f'font-family="sans-serif" font-size="12">step {max(xs)}</text>'
        "</svg>\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _plot_name(metric: str) -> str | None:
    aliases = {
        "loss": "train_loss",
        "train/loss": "train_loss",
        "eval/loss": "validation_loss",
        "val/loss": "validation_loss",
        "lr": "learning_rate",
        "train/lr": "learning_rate",
        "eval/simulated_acc_len": "acceptance_length",
    }
    if "acceptance_rate" in metric:
        return metric.replace("/", "_")
    return aliases.get(metric)


def _write_metric_plots(
    series: dict[str, list[tuple[int, float]]], output_dir: Path
) -> None:
    for metric, points in series.items():
        name = _plot_name(metric)
        if name:
            _write_svg(output_dir / f"{name}.svg", points, metric)


def _record_training(
    command: list[str], config: dict[str, object], report_dir: Path
) -> None:
    """Stream native logs, save scalar history/charts, and sample GPU memory."""
    log_path = report_dir / "metrics.jsonl"
    write_json(report_dir / "metrics-schema.json", METRIC_SCHEMA)
    memory_result: dict[str, object] = {
        "status": "unavailable",
        "peak_used_mib": None,
        "peak_training_used_mib": None,
        "peak_evaluation_used_mib": None,
        "available_before_training_mib": None,
        "total_gpu_mib": None,
    }
    current_phase = ["train"]
    gpu_id = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
    try:
        preflight = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
                f"--id={gpu_id}",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if preflight.returncode == 0 and preflight.stdout.strip():
            free_mib, total_mib = [
                int(value.strip())
                for value in preflight.stdout.splitlines()[0].split(",")
            ]
            memory_result.update(
                status="sampling",
                available_before_training_mib=free_mib,
                total_gpu_mib=total_mib,
            )
            print(
                f"Detected {free_mib} MiB free of {total_mib} MiB "
                f"on visible GPU {gpu_id}; "
                "training starts with microbatch 1 and CPU optimizer-state offload.",
                flush=True,
            )
            if free_mib < 1024:
                raise RuntimeError(
                    "Less than 1 GiB is free on the training GPU. Close other GPU jobs "
                    "before starting the one-sample microbatch."
                )
    except (OSError, subprocess.SubprocessError):
        pass
    stop_sampler = threading.Event()

    def sample_memory() -> None:
        try:
            while not stop_sampler.is_set():
                query = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        f"--id={gpu_id}",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                if query.returncode == 0 and query.stdout.strip():
                    values = [
                        int(line.strip())
                        for line in query.stdout.splitlines()
                        if line.strip().isdigit()
                    ]
                    if values:
                        memory_result["status"] = "sampled"
                        memory_result["peak_used_mib"] = max(
                            values[0], int(memory_result["peak_used_mib"] or 0)
                        )
                        field = (
                            "peak_evaluation_used_mib"
                            if current_phase[0] == "eval"
                            else "peak_training_used_mib"
                        )
                        memory_result[field] = max(
                            values[0], int(memory_result[field] or 0)
                        )
                stop_sampler.wait(0.5)
        except (OSError, subprocess.SubprocessError):
            memory_result["status"] = "unavailable"

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    process_env = os.environ.copy()
    process_env["EAGLE3_MLFLOW_RUN_ID_FILE"] = str(report_dir / "mlflow-run-id")
    tracker_bootstrap = str(Path(__file__).resolve().parent / "mlflow_bootstrap")
    existing_python_path = process_env.get("PYTHONPATH")
    process_env["PYTHONPATH"] = os.pathsep.join(
        path for path in (tracker_bootstrap, existing_python_path) if path
    )
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=process_env,
    )
    series: dict[str, list[tuple[int, float]]] = {}
    with log_path.open("w", encoding="utf-8") as metrics_file:
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            match = re.match(r"^step\s+(\d+):\s+(\{.*\})\s*$", line)
            if not match:
                continue
            try:
                step = int(match.group(1))
                values = ast.literal_eval(match.group(2))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(values, dict):
                continue
            is_eval_log = any(str(key).startswith("eval/") for key in values)
            if is_eval_log:
                current_phase[0] = "train"
            row = {"step": step}
            for name, value in canonical_metrics(values).items():
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if number == number and abs(number) != float("inf"):
                    row[str(name)] = number
                    series.setdefault(str(name), []).append((step, number))
            metrics_file.write(json.dumps(row, sort_keys=True) + "\n")
            metrics_file.flush()
            if not is_eval_log:
                eval_interval = config["training"]["eval_interval"]
                if eval_interval > 0 and step % eval_interval == 0:
                    current_phase[0] = "eval"
    _write_metric_plots(series, report_dir / "plots")
    return_code = process.wait()
    stop_sampler.set()
    sampler.join(timeout=2)
    memory_result["peak_child_rss_kib"] = resource.getrusage(
        resource.RUSAGE_CHILDREN
    ).ru_maxrss
    (report_dir / "gpu-memory.json").write_text(
        json.dumps(memory_result, indent=2) + "\n", encoding="utf-8"
    )
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _log_mlflow_artifacts(
    config: dict[str, object], output_dir: Path, output_config: Path
) -> None:
    """Attach resolved configuration, scalar history, and SVG plots to MLflow."""
    try:
        from mlflow import MlflowClient
    except ImportError:
        print(
            "MLflow artifact upload skipped: install mlflow to enable it.",
            file=sys.stderr,
        )
        return
    uri = config["tracking"].get("mlflow_tracking_uri") or os.getenv(
        "MLFLOW_TRACKING_URI"
    )
    if uri:
        client = MlflowClient(tracking_uri=uri)
    else:
        client = MlflowClient()
    run_id = (output_dir / "mlflow-run-id").read_text(encoding="utf-8").strip()
    try:
        commit_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("Unable to determine the Git revision for MLflow") from exc
    client.log_param(run_id, "project.git_commit", commit_sha)
    client.log_artifact(run_id, str(output_config), artifact_path="configuration")
    final_step = 0
    for artifact in (
        output_dir / "metrics.jsonl",
        output_dir / "gpu-memory.json",
        output_dir / "metrics-schema.json",
        output_dir / "training-plan.json",
        output_dir / TRAIN_INPUT_MANIFEST,
        output_dir / "checkpoint-manifest.json",
        output_dir / "inference-smoke.json",
        output_dir / "generation.json",
        output_dir / "model-snapshot.json",
        output_dir / f"{config.get('run_id')}.best_meta.json",
    ):
        if artifact.exists():
            client.log_artifact(run_id, str(artifact), artifact_path="metrics")
            if artifact.name == "metrics.jsonl":
                with artifact.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            final_step = max(final_step, int(json.loads(line)["step"]))
                        except (ValueError, KeyError, TypeError):
                            continue
            elif artifact.name == "gpu-memory.json":
                try:
                    gpu_memory = json.loads(artifact.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    gpu_memory = {}
                for metric, field in (
                    ("gpu/peak_used_mib", "peak_used_mib"),
                    ("gpu/training_peak_used_mib", "peak_training_used_mib"),
                    ("gpu/evaluation_peak_used_mib", "peak_evaluation_used_mib"),
                ):
                    value = gpu_memory.get(field)
                    if isinstance(value, (int, float)):
                        client.log_metric(run_id, metric, float(value), step=final_step)
    for feature_split, feature_path in (
        ("eval", config["data"]["eval_hidden_states_path"]),
    ):
        feature_root = resolve_path(feature_path)
        memory_record = feature_root / "feature-extraction-memory.json"
        if memory_record.exists():
            client.log_artifact(
                run_id, str(memory_record), artifact_path=f"metrics/{feature_split}"
            )
            manifest = feature_root / "feature-manifest.json"
            if manifest.exists():
                client.log_artifact(
                    run_id, str(manifest), artifact_path=f"metrics/{feature_split}"
                )
            try:
                capture_memory = json.loads(memory_record.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                capture_memory = {}
            for field, metric in (
                ("peak_allocated_bytes", "allocated"),
                ("peak_reserved_bytes", "reserved"),
            ):
                value = capture_memory.get(field)
                if isinstance(value, (int, float)):
                    client.log_metric(
                        run_id,
                        f"gpu/capture_{feature_split}_peak_{metric}_mib",
                        float(value) / (1024 * 1024),
                        step=final_step,
                    )
    plot_dir = output_dir / "plots"
    if plot_dir.exists():
        for artifact in plot_dir.glob("*.svg"):
            client.log_artifact(run_id, str(artifact), artifact_path="plots")
    for key, value in config["model"].items():
        client.log_param(run_id, f"model.{key}", str(value)[:500])
    for key, value in config["training"].items():
        if value is not None:
            client.log_param(run_id, f"training.{key}", str(value)[:500])
    client.log_param(
        run_id,
        "training.optimizer",
        str(config["training"].get("optimizer", "adamw")),
    )


def _feature_command(
    split: str,
    config_path: Path,
    model_config_path: Path,
    model_path: str,
    tokenizer_path: str,
    draft_config_path: Path,
    evaluation_config_path: Path,
    sample_limit: int | None = None,
    output_override: Path | None = None,
) -> list[str]:
    if split not in ("eval", "test"):
        raise ValueError("Only validation and test hidden states can be captured")
    dataset = read_dataset_config(config_path)
    model_cfg = expand_env(
        yaml.safe_load(model_config_path.read_text(encoding="utf-8"))["model"]
    )
    source = resolve_path(dataset[f"{split}_data_path"])
    output = output_override or resolve_path(dataset[f"{split}_features_path"])
    samples = dataset[f"{split}_samples"]
    if not source.is_file():
        raise FileNotFoundError(
            f"prepared {split} dataset does not exist: {source}; run prepare-data first"
        )
    available_samples = _count_jsonl_rows(source)
    if available_samples == 0:
        raise ValueError(f"prepared {split} dataset is empty: {source}")
    samples = min(samples, available_samples) if samples is not None else None
    if sample_limit is not None:
        samples = min(samples, sample_limit) if samples is not None else sample_limit
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--module",
        "eagle3_olmo2.prepare_features",
        "--target-model-path",
        model_path,
        "--tokenizer-path",
        tokenizer_path,
        "--strategy",
        "eagle3",
        "--draft-model-config",
        str(draft_config_path),
        "--data-path",
        str(source),
        "--output-path",
        str(output),
        "--chat-template",
        dataset["chat_template"],
        "--max-length",
        str(dataset["max_length"]),
        "--capture-seed",
        str(dataset["seed"]),
        "--batch-size",
        str(dataset.get("preprocessing_batch_size", 1)),
        "--build-dataset-num-proc",
        "1",
        "--num-workers",
        "2",
        "--cache-dir",
        str(resolve_path(dataset["cache_dir"])),
        "--sglang-attention-backend",
        model_cfg.get("capture_attention_backend", "triton"),
        "--sglang-mem-fraction-static",
        str(model_cfg.get("sglang_mem_fraction_static", 0.35)),
    ]
    if split == "eval":
        with evaluation_config_path.open(encoding="utf-8") as stream:
            evaluation = yaml.safe_load(stream)["evaluation"]
        limit = evaluation.get("max_eval_samples")
        if limit is not None:
            samples = min(samples, limit) if samples is not None else limit
    if samples is not None:
        command.extend(("--num-samples", str(samples)))
    return command


def _selection(dataset: dict, split: str, limit: int | None) -> int | None:
    configured = dataset[f"{split}_samples"]
    return min(limit, configured or limit) if limit is not None else configured


def _data_artifacts(dataset: dict, split: str) -> tuple[Path, Path]:
    source = resolve_path(dataset[f"{split}_data_path"])
    return source, source.with_suffix(source.suffix + ".manifest.json")


def _bootstrap_data(args, dataset_path: Path, dataset: dict, splits) -> None:
    for split in splits:
        if split != "train" and "test_data_path" in dataset:
            ensure_dvc_artifacts_available(*_data_artifacts(dataset, "train"))
        paths = _data_artifacts(dataset, split)
        if all(path.with_name(f"{path.name}.dvc").exists() for path in paths):
            ensure_dvc_artifacts_available(*paths)
        command = [
            sys.executable,
            "-m",
            "eagle3_olmo2.data",
            "--config",
            str(dataset_path),
            "--split",
            split,
        ]
        if args.limit is not None:
            command.extend(("--limit", str(args.limit)))
        _run(command)
        for artifact in paths:
            if not artifact.with_name(f"{artifact.name}.dvc").exists():
                register_artifact(artifact)
        ensure_dvc_artifacts_available(*paths)


def _prepared_data(dataset, split, limit, *, allow_existing_limit=False):
    source, source_manifest = _data_artifacts(dataset, split)
    ensure_dvc_artifacts_available(source, source_manifest)
    data_metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    if data_metadata["sha256"] != sha256_file(source):
        raise ValueError(f"Prepared data does not match its manifest: {source}")
    if (
        data_metadata["dataset"] != dataset["name"]
        or data_metadata["requested_revision"] != dataset["revision"]
        or data_metadata["seed"] != dataset["seed"]
        or data_metadata["parquet_pattern"] != dataset.get("parquet_pattern")
        or data_metadata["split"]
        != dataset.get(f"{split}_split", dataset["eval_split"])
        or data_metadata["partition"]
        != (split if split != "train" and "test_data_path" in dataset else None)
        or (
            not allow_existing_limit
            and data_metadata["requested_samples"] != _selection(dataset, split, limit)
        )
    ):
        raise ValueError(
            f"Prepared data does not match the {split} selection: {source}"
        )
    return source, {
        "source_sha256": data_metadata["sha256"],
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_revision": data_metadata["resolved_revision"],
    }


def _capture(args, dataset_path, dataset, split, assets, identity) -> None:
    source, data_identity = _prepared_data(dataset, split, args.limit)
    output = resolve_path(dataset[f"{split}_features_path"])
    expected = {
        **identity,
        "split": split,
        **data_identity,
        "requested_samples": args.limit,
        "capture_sample_count": _count_jsonl_rows(source),
        "capture_batch_size": dataset["preprocessing_batch_size"],
        "max_eval_samples": (
            yaml.safe_load(assets[-1].read_text(encoding="utf-8"))["evaluation"].get(
                "max_eval_samples"
            )
            if split == "eval"
            else None
        ),
    }
    ref = output.with_name(f"{output.name}.dvc")
    if ref.exists():
        ensure_dvc_artifacts_available(output)
        feature_manifest(output, expected)
        return
    if output.exists():
        # After a failed push, reuse only a completed capture of the same inputs.
        feature_manifest(output, expected)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        model_config, model_path, tokenizer_path, draft_config, evaluation_path = assets
        with tempfile.TemporaryDirectory(
            prefix=f".{split}-", dir=output.parent
        ) as staging:
            working = Path(staging) / "capture"
            _run(
                _feature_command(
                    split,
                    dataset_path,
                    model_config,
                    model_path,
                    tokenizer_path,
                    draft_config,
                    evaluation_path,
                    args.limit,
                    output_override=working,
                )
            )
            statistics = json.loads(
                (working / "capture-audit.json").read_text(encoding="utf-8")
            )
            write_json(working / "feature-manifest.json", {**statistics, **expected})
            working.rename(output)
    register_artifact(output)
    ensure_dvc_artifacts_available(output)


def _checkpoint_path(config: dict, selected: Path | None) -> Path:
    if selected is not None:
        return resolve_path(selected)
    output_dir = resolve_path(config["output_dir"])
    for kind in ("best", "latest"):
        alias = output_dir / f"{config['run_id']}-{kind}"
        if alias.is_symlink() or alias.exists():
            return alias.resolve()
    raise FileNotFoundError(
        f"No SpecForge checkpoint alias in {output_dir}; supply --checkpoint"
    )


def _run_inference(
    config, evaluation_path, tokenizer_path, checkpoint, destination, args
):
    evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))[
        "evaluation"
    ]
    destination.mkdir(parents=True, exist_ok=True)
    draft_dir = destination / "draft_sglang"
    if args.mode != "target-only":
        _run(
            [
                "specforge",
                "export",
                "--to",
                "sglang",
                "--checkpoint",
                str(checkpoint),
                "--draft-config",
                config["model"]["draft_model_config"],
                "--output-dir",
                str(draft_dir),
                "--vocab-mapping",
                str(checkpoint / "eagle3-vocab-mapping.pt"),
            ]
        )
    command = [
        sys.executable,
        "-m",
        "eagle3_olmo2.inference",
        "--target-model",
        config["model"]["target_model_path"],
        "--tokenizer-path",
        tokenizer_path,
        "--mode",
        args.mode,
        "--output",
        str(
            destination
            / (
                "generation.json"
                if args.suite or args.prompt
                else "inference-smoke.json"
            )
        ),
        "--port",
        str(evaluation["inference_port"]),
        "--max-new-tokens",
        str(evaluation["inference_max_new_tokens"]),
        "--settings",
        str(evaluation_path),
    ]
    if args.mode != "target-only":
        command.extend(
            ("--draft-model", str(draft_dir), "--checkpoint", str(checkpoint))
        )
    if args.suite:
        command.extend(("--suite", str(args.suite.resolve())))
    if args.prompt:
        command.extend(("--prompt", args.prompt))
    if args.trace:
        command.append("--trace")
    _run(command)


def _checkpoint_step(checkpoint: Path) -> int:
    step = re.search(r"-step(\d+)$", checkpoint.name)
    if step is None:
        raise ValueError(
            f"Expected the original SpecForge step directory name: {checkpoint}"
        )
    return int(step[1])


def _register_checkpoint(
    checkpoint: Path, config: dict, identity: dict, train_input: Path
) -> None:
    import torch

    from specforge.training.checkpoint import STATE_FILE

    if (
        not (checkpoint / STATE_FILE).is_file()
        or not (checkpoint / "training_state_rank0.pt").is_file()
    ):
        raise ValueError(
            f"Expected a complete SpecForge checkpoint directory: {checkpoint}"
        )
    mapping = resolve_path(config["model"]["vocab_mapping_path"])
    destination = checkpoint / "eagle3-vocab-mapping.pt"
    manifest_path = checkpoint / "eagle3-manifest.json"
    input_digest = sha256_file(train_input)
    native = torch.load(
        checkpoint / STATE_FILE, map_location="cpu", weights_only=True, mmap=True
    )
    if native.get(TRAIN_INPUT_DIGEST) != input_digest:
        raise ValueError(f"Checkpoint requires different training inputs: {checkpoint}")
    manifest = {
        **identity,
        "optimizer_step": _checkpoint_step(checkpoint),
        "vocab_mapping_sha256": sha256_file(mapping),
        TRAIN_INPUT_DIGEST: input_digest,
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError(f"Checkpoint provenance differs: {checkpoint}")
    elif not destination.exists():
        shutil.copyfile(mapping, destination)
    if sha256_file(destination) != manifest["vocab_mapping_sha256"]:
        raise ValueError(f"Checkpoint vocabulary mapping differs: {checkpoint}")
    write_json(manifest_path, manifest)
    register_artifact(checkpoint)
    ensure_dvc_artifacts_available(checkpoint)


def _verify_checkpoint(checkpoint: Path, identity: dict) -> dict:
    ensure_dvc_artifacts_available(checkpoint)
    manifest = json.loads(
        (checkpoint / "eagle3-manifest.json").read_text(encoding="utf-8")
    )
    if any(manifest.get(key) != value for key, value in identity.items()):
        raise ValueError(
            f"Checkpoint is incompatible with the selected target/tokenizer: {checkpoint}"
        )
    if manifest["vocab_mapping_sha256"] != sha256_file(
        checkpoint / "eagle3-vocab-mapping.pt"
    ):
        raise ValueError(f"Checkpoint vocabulary mapping is corrupt: {checkpoint}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare-data",
            "prepare-features",
            "compose",
            "preflight",
            "train",
            "register-checkpoint",
            "validate",
            "inference",
            "full",
        ),
    )
    parser.add_argument(
        "--split", choices=("train", "eval", "test", "all"), default="all"
    )
    parser.add_argument(
        "--limit", type=int, help="Upper bound on prepared and captured rows"
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--output-config", type=Path)
    parser.add_argument("--cuda-visible-devices")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--mode", choices=("target-only", "speculative", "compare"))
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--suite", type=Path)
    inputs.add_argument("--prompt")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument(
        "--approved-pilot",
        type=Path,
        help="Completed pilot report directory after budget review",
    )
    args = parser.parse_args()
    args.mode = args.mode or (
        "compare" if args.command == "validate" else "speculative"
    )
    if args.command == "validate" and not (args.suite or args.prompt):
        suite_name = "final.jsonl" if args.split == "test" else "quick.jsonl"
        args.suite = ROOT / "configs/prompts" / suite_name
    ensure_configured_storage_dirs()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    experiment = yaml.safe_load(args.experiment.read_text(encoding="utf-8"))
    infrastructure = expand_env(experiment.get("infrastructure", {}))
    profile = infrastructure.get("profile", "baseline")
    _configure_tracking_environment(infrastructure)
    output_dir = experiment_output_dir(args.experiment)
    dataset_path = args.dataset_config or experiment_module_path(
        args.experiment, "dataset"
    )
    dataset = read_dataset_config(dataset_path)
    if args.split == "test" and "test_data_path" not in dataset:
        parser.error("This dataset configuration has no final test split")
    splits = (
        ("train", "eval", "test") if "test_data_path" in dataset else ("train", "eval")
    )
    requested = splits if args.split == "all" else (args.split,)
    if args.command == "prepare-features":
        if args.split == "train":
            parser.error("Train features are computed per batch; use prepare-data")
        requested = tuple(split for split in requested if split != "train")
    if profile == "full" and args.limit is not None:
        parser.error("The full profile uses the configured complete train split")
    if args.command == "prepare-data":
        _bootstrap_data(args, dataset_path, dataset, requested)
        return

    config = compose(args.experiment, dataset_config_path=dataset_path)
    validate_specforge(config)
    if config["training"]["strategy"] != "eagle3" or config["training"].get(
        "lk_loss_type"
    ):
        parser.error("The project metric schema requires EAGLE-3 with LK disabled")
    if args.command == "compose":
        output_config = args.output_config or output_dir / "composed.yaml"
        output_config.parent.mkdir(parents=True, exist_ok=True)
        output_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        print(f"Wrote SpecForge run config to {output_config}")
        return

    unbounded_train = (
        dataset["train_samples"] is None and config["training"].get("max_steps") is None
    )
    full_profile = profile == "full" or unbounded_train
    pilot_dir = None
    starts_full = args.command in ("train", "full")
    if full_profile and starts_full:
        if not args.approved_pilot:
            parser.error(
                "Review the pilot resource budget, then supply --approved-pilot"
            )
        pilot = json.loads(
            (args.approved_pilot / "training-status.json").read_text(encoding="utf-8")
        )
        if pilot.get("profile") != "pilot" or pilot.get("status") != "complete":
            parser.error("--approved-pilot must point to a completed pilot report")
        if not (args.approved_pilot / "inference-smoke.json").is_file():
            parser.error("An approved pilot must include a generation smoke response")
        if not (args.approved_pilot / TRAIN_INPUT_MANIFEST).is_file():
            parser.error(
                "The approved pilot must have trained with live OLMo-2 capture"
            )
        pilot_dir = args.approved_pilot.resolve()
        if (
            config["training"].get("max_steps") is not None
            or dataset["train_samples"] is not None
        ):
            parser.error(
                "The full profile must leave train rows and optimizer steps uncapped"
            )

    if args.command not in ("preflight", "register-checkpoint"):
        configured_device = infrastructure.get("device", "cuda:0")
        if configured_device != "cuda" and not configured_device.startswith("cuda:"):
            parser.error("This pipeline requires a CUDA device")
        visible = (
            args.cuda_visible_devices or configured_device.partition(":")[2] or "0"
        )
        if "," in visible:
            parser.error("Select exactly one visible CUDA device")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
        try:
            require_cuda()
        except RuntimeError as exc:
            parser.error(str(exc))

    model_config = experiment_module_path(args.experiment, "model")
    evaluation_path = experiment_module_path(args.experiment, "evaluation")
    model_options = yaml.safe_load(model_config.read_text(encoding="utf-8"))["model"]
    draft_config = Path(config["model"]["draft_model_config"])
    model_path, tokenizer_path = resolve_model_assets(
        model_config, output_dir, cache_dir=resolve_path(dataset["cache_dir"])
    )
    identity = {
        **asset_identity(model_path, tokenizer_path, draft_config),
        "target_revision": model_options["revision"],
        "tokenizer_revision": model_options.get(
            "tokenizer_revision", model_options["revision"]
        ),
        "aux_layers": config["model"]["aux_hidden_state_layer_ids"],
    }
    if full_profile and any(
        not re.fullmatch(r"[0-9a-f]{40}", revision)
        for revision in (
            dataset["revision"],
            identity["target_revision"],
            identity["tokenizer_revision"],
        )
    ):
        parser.error(
            "Full training requires immutable dataset, target and tokenizer revisions"
        )
    if identity["aux_layers"] != [1, 7, 12]:
        parser.error("OLMo-2 capture expects zero-based auxiliary layers [1, 7, 12]")
    config["model"]["target_model_path"] = model_path
    feature_identity = {
        **identity,
        "max_length": dataset["max_length"],
        "chat_template": dataset["chat_template"],
        "dataset": dataset["name"],
        "dataset_revision": dataset["revision"],
        "dataset_seed": dataset["seed"],
    }
    assets = model_config, model_path, tokenizer_path, draft_config, evaluation_path
    if args.command == "full":
        _bootstrap_data(args, dataset_path, dataset, ("train", "eval"))
    if args.command in ("prepare-features", "full"):
        capture_splits = requested if args.command == "prepare-features" else ("eval",)
        if pilot_dir:
            for split in capture_splits:
                ensure_dvc_artifacts_available(*_data_artifacts(dataset, split))
            estimate_capture_storage(pilot_dir, dataset, capture_splits)
        for split in capture_splits:
            _capture(args, dataset_path, dataset, split, assets, feature_identity)
        if args.command == "prepare-features":
            return

    if args.command in ("validate", "inference"):
        checkpoint = None
        if args.mode != "target-only" or args.command == "validate":
            checkpoint = _checkpoint_path(config, args.checkpoint)
            _verify_checkpoint(checkpoint, identity)
        destination = Path(tempfile.mkdtemp(prefix=f"{args.command}-", dir=output_dir))
        if args.command == "validate":
            split = "test" if args.split == "test" else "eval"
            features = resolve_path(dataset[f"{split}_features_path"])
            ensure_dvc_artifacts_available(features)
            feature_manifest(features, {**feature_identity, "split": split})
            config["data"]["eval_hidden_states_path"] = str(features)
            config["model"]["vocab_mapping_path"] = str(
                checkpoint / "eagle3-vocab-mapping.pt"
            )
            eval_config = destination / "evaluation-run.yaml"
            eval_config.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            _run(
                [
                    sys.executable,
                    "-m",
                    "eagle3_olmo2.evaluate",
                    str(eval_config),
                    str(checkpoint),
                    str(destination / "offline-eval.json"),
                ]
            )
        _run_inference(
            config, evaluation_path, tokenizer_path, checkpoint, destination, args
        )
        return

    source, data_identity = _prepared_data(
        dataset,
        "train",
        args.limit,
        allow_existing_limit=(
            not full_profile and dataset["train_samples"] is None and args.limit is None
        ),
    )
    config["data"]["train_data_path"] = str(source)
    train_input = prepare_training_inputs(
        config, source, tokenizer_path, {**feature_identity, **data_identity}
    )
    if args.command == "register-checkpoint":
        _register_checkpoint(
            _checkpoint_path(config, args.checkpoint), config, identity, train_input
        )
        return

    evaluation = resolve_path(dataset["eval_features_path"])
    config["data"]["eval_hidden_states_path"] = str(evaluation)
    ensure_dvc_artifacts_available(evaluation)
    feature_manifest(evaluation, {**feature_identity, "split": "eval"})
    resume = config["training"].get("resume_from")
    start_step = 0
    if resume:
        resume_path = resolve_path(resume)
        recorded = _verify_checkpoint(resume_path, identity)
        if recorded.get(TRAIN_INPUT_DIGEST) != sha256_file(train_input):
            raise ValueError(
                "Resume requires a live-capture checkpoint with the original training inputs"
            )
        if sha256_file(resume_path / "eagle3-vocab-mapping.pt") != sha256_file(
            resolve_path(config["model"]["vocab_mapping_path"])
        ):
            raise ValueError("Resume requires the original training vocabulary mapping")
        config["training"]["resume_from"] = str(resume_path)
        start_step = _checkpoint_step(resume_path)
    if args.command == "preflight":
        plan = plan_training(config, output_dir, train_input, args.approved_pilot)
        print(json.dumps(plan, indent=2))
        return

    if (output_dir / "training-status.json").exists():
        raise RuntimeError(
            "This output directory already has a training attempt; choose a fresh "
            "output directory and pass the original checkpoint for resume."
        )
    report_dir = output_dir
    output_config = args.output_config or report_dir / "run.yaml"
    status = {"profile": profile, "run_id": config["run_id"]}
    status_path = report_dir / "training-status.json"
    started = time.monotonic()
    # Exclusive creation prevents concurrent trainers from sharing aliases.
    with status_path.open("x", encoding="utf-8") as stream:
        json.dump({**status, "status": "running"}, stream)
    try:
        plan = plan_training(config, report_dir, train_input, pilot_dir)
        if start_step >= plan["optimizer_steps"]:
            raise ValueError(
                "The resume checkpoint has already reached the configured horizon"
            )
        output_config.parent.mkdir(parents=True, exist_ok=True)
        output_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        _record_training(
            [sys.executable, "-m", "eagle3_olmo2.train", str(output_config)],
            config,
            report_dir,
        )
        training_wall_seconds = time.monotonic() - started
        checkpoint = _checkpoint_path(config, None)
        latest = (output_dir / f"{config['run_id']}-latest").resolve(strict=True)
        final_step = _checkpoint_step(latest)
        if final_step != plan["optimizer_steps"]:
            raise RuntimeError(
                f"Final checkpoint step {final_step} differs from the training plan"
            )
        write_json(
            report_dir / "checkpoint-manifest.json",
            {
                **identity,
                "best": str(checkpoint),
                "latest": str(latest),
                "planned_optimizer_steps": plan["optimizer_steps"],
                "resumed_from": config["training"].get("resume_from"),
            },
        )
        for complete in dict.fromkeys((checkpoint, latest)):
            _register_checkpoint(complete, config, identity, train_input)
        if args.command == "full":
            _run_inference(
                config, evaluation_path, tokenizer_path, checkpoint, report_dir, args
            )
        if config["tracking"]["report_to"] == "mlflow":
            _log_mlflow_artifacts(config, report_dir, output_config)
    except BaseException as exc:
        write_json(
            status_path,
            {**status, "status": "failed", "error_type": type(exc).__name__},
        )
        raise
    write_json(
        status_path,
        {
            **status,
            "status": "complete",
            "training_wall_seconds": training_wall_seconds,
            "optimizer_steps_executed": final_step - start_step,
            "checkpoint": str(checkpoint),
            "latest": str(latest),
        },
    )


if __name__ == "__main__":
    main()
