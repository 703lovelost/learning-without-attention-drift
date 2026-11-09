"""Run the modular OLMo-2/SpecForge offline EAGLE-3 pipeline."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import subprocess
import sys
import threading
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
from eagle3_olmo2.dvc import ensure_dvc_artifacts_available
from eagle3_olmo2.paths import ensure_configured_storage_dirs, expand_env, resolve_path


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


def align_final_evaluation(config: dict[str, object]) -> None:
    """Make the configured evaluation cadence land on the final offline step."""
    data = config["data"]
    training = config["training"]
    feature_root = Path(data["hidden_states_path"])
    if not feature_root.is_absolute():
        feature_root = ROOT / feature_root
    if not feature_root.exists():
        return
    samples = sum(
        1
        for path in feature_root.rglob("*")
        if path.is_file()
        and (path.name.endswith(".ckpt") or path.name.endswith(".ckpt.gz"))
    )
    update_quantum = training["batch_size"] * training["accumulation_steps"]
    updates_per_epoch = samples // update_quantum
    total_updates = updates_per_epoch * training["num_epochs"]
    if training.get("max_steps") is not None:
        total_updates = min(total_updates, training["max_steps"])
    interval = training["eval_interval"]
    if interval > 0:
        training["log_interval"] = math.gcd(training["log_interval"], interval)
    if total_updates > 0 and interval > 0 and total_updates % interval:
        adjusted = math.gcd(interval, total_updates)
        print(
            f"Adjusting evaluation interval from {interval} to {adjusted} so the "
            f"final offline optimizer step {total_updates} is evaluated.",
            flush=True,
        )
        training["eval_interval"] = adjusted
        training["log_interval"] = math.gcd(training["log_interval"], adjusted)


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
    overlay.mkdir(parents=True, exist_ok=True)
    for asset in source.iterdir():
        destination = overlay / asset.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(asset.resolve())
    from safetensors import safe_open

    with safe_open(str(source / "model.safetensors"), framework="pt") as weights:
        weight_map = {key: "model.safetensors" for key in weights.keys()}
    (overlay / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": weight_map},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return str(overlay.resolve())


def resolve_model_assets(model_config_path: Path, output_dir: Path) -> tuple[str, str]:
    """Resolve model and tokenizer revisions to immutable local snapshots."""
    with model_config_path.open(encoding="utf-8") as stream:
        model = yaml.safe_load(stream)["model"]
    model_id = model["target_model_path"]
    resolved_model = _resolve_snapshot(
        model_id, model.get("revision", "main"), relative_to=model_config_path.parent
    )
    indexed_model = _target_head_compatible_snapshot(resolved_model, output_dir)
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


def _discard_future_best(output_dir: Path, run_id: str, resume_step: int) -> bool:
    """Drop a best alias from a later step when resuming an earlier branch."""
    metadata_path = output_dir / f"{run_id}.best_meta.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        best_step = int(metadata["step"])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if best_step <= resume_step:
        return False
    best_link = output_dir / f"{run_id}-best"
    if best_link.is_symlink():
        best_link.unlink()
    metadata_path.unlink()
    print(
        f"Cleared best-checkpoint metadata at step {best_step}; resuming from "
        f"step {resume_step} starts a new checkpoint timeline.",
        flush=True,
    )
    return True


def _reconcile_best_for_resume(config: dict[str, object], output_dir: Path) -> None:
    resume_from = config["training"].get("resume_from")
    if not resume_from:
        return
    from specforge.training.checkpoint import CheckpointManager

    state = CheckpointManager.read_resume_state(str(resume_from))
    _discard_future_best(output_dir, str(config["run_id"]), int(state["global_step"]))


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
        "eval/avg_loss": "validation_loss",
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
    command: list[str], config: dict[str, object], output_config: Path
) -> None:
    """Stream native logs, save scalar history/charts, and sample GPU memory."""
    output_dir = resolve_path(str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    _reconcile_best_for_resume(config, output_dir)
    log_path = output_dir / "metrics.jsonl"
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
            for name, value in values.items():
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
            _write_metric_plots(series, output_dir / "plots")
    return_code = process.wait()
    stop_sampler.set()
    sampler.join(timeout=2)
    (output_dir / "gpu-memory.json").write_text(
        json.dumps(memory_result, indent=2) + "\n", encoding="utf-8"
    )
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    if os.getenv("MLFLOW_TRACKING_URI") or config.get("tracking", {}).get(
        "mlflow_tracking_uri"
    ):
        _log_mlflow_artifacts(config, output_dir, output_config)


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
    experiment = client.get_experiment_by_name(
        config["tracking"].get("mlflow_experiment_name") or "eagle3-olmo2"
    )
    if experiment is None:
        return
    run_name = config.get("run_id", "olmo2-eagle3")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return
    run_id = runs[0].info.run_id
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
        ("train", config["data"]["hidden_states_path"]),
        ("eval", config["data"]["eval_hidden_states_path"]),
    ):
        feature_root = resolve_path(feature_path)
        memory_record = feature_root / "feature-extraction-memory.json"
        if memory_record.exists():
            client.log_artifact(run_id, str(memory_record), artifact_path="metrics")
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
) -> list[str]:
    dataset = read_dataset_config(config_path)
    model_cfg = expand_env(
        yaml.safe_load(model_config_path.read_text(encoding="utf-8"))["model"]
    )
    is_train = split == "train"
    source = resolve_path(dataset["train_data_path" if is_train else "eval_data_path"])
    output = resolve_path(
        dataset["train_features_path" if is_train else "eval_features_path"]
    )
    samples = dataset["train_samples" if is_train else "eval_samples"]
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


def _run_inference(
    config: dict[str, object],
    output_config: Path,
    evaluation_config_path: Path,
    tokenizer_path: str,
) -> None:
    output_dir = resolve_path(str(config["output_dir"]))
    run_id = str(config["run_id"])
    checkpoint_root = output_dir
    best = checkpoint_root / f"{run_id}-best"
    latest = checkpoint_root / f"{run_id}-latest"
    checkpoint = best if best.exists() else latest
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"no SpecForge best/latest checkpoint found under {checkpoint_root}"
        )
    draft_dir = output_dir / "draft_sglang"
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
            config["model"]["vocab_mapping_path"],
        ]
    )
    with evaluation_config_path.open(encoding="utf-8") as stream:
        evaluation = yaml.safe_load(stream)["evaluation"]
    target_path = config["model"]["target_model_path"]
    _run(
        [
            sys.executable,
            "-m",
            "eagle3_olmo2.inference",
            "--target-model",
            target_path,
            "--tokenizer-path",
            tokenizer_path,
            "--draft-model",
            str(draft_dir),
            "--output",
            str(output_dir / "inference-smoke.json"),
            "--port",
            str(evaluation["inference_port"]),
            "--max-new-tokens",
            str(evaluation["inference_max_new_tokens"]),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare-data",
            "prepare-features",
            "compose",
            "train",
            "inference",
            "full",
        ),
    )
    parser.add_argument("--split", choices=("train", "eval", "all"), default="all")
    parser.add_argument(
        "--limit",
        type=int,
        help="Use a deterministic small data subset for a smoke run",
    )
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument(
        "--output-config",
        type=Path,
        default=resolve_path(
            Path(os.getenv("EAGLE3_OUTPUT_DIR", "outputs/olmo2_1b_eagle3")) / "run.yaml"
        ),
    )
    parser.add_argument("--cuda-visible-devices")
    args = parser.parse_args()
    ensure_configured_storage_dirs()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    with args.experiment.open(encoding="utf-8") as stream:
        infrastructure = expand_env(yaml.safe_load(stream).get("infrastructure", {}))
    _configure_tracking_environment(infrastructure)
    configured_device = infrastructure.get("device", "cuda:0")
    if args.cuda_visible_devices:
        visible_devices = args.cuda_visible_devices
    elif configured_device.startswith("cuda:"):
        visible_devices = configured_device.split(":", 1)[1]
    elif configured_device == "cuda":
        visible_devices = "0"
    else:
        parser.error("the feature-capture pipeline requires a CUDA device (cuda:N)")
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices
    if args.command in ("prepare-features", "train", "inference", "full"):
        try:
            require_cuda()
        except RuntimeError as exc:
            parser.error(str(exc))
        ensure_dvc_artifacts_available(ROOT)

    if args.command in ("prepare-data", "full"):
        dataset_config_path = args.dataset_config or experiment_module_path(
            args.experiment, "dataset"
        )
        data_command = [
            sys.executable,
            "-m",
            "eagle3_olmo2.data",
            "--config",
            str(dataset_config_path),
        ]
        if args.command == "prepare-data":
            data_command.extend(("--split", args.split))
        if args.limit is not None:
            data_command.extend(("--limit", str(args.limit)))
        _run(data_command)
    if args.command == "prepare-features":
        model_config_path = experiment_module_path(args.experiment, "model")
        dataset_config_path = args.dataset_config or experiment_module_path(
            args.experiment, "dataset"
        )
        evaluation_config_path = experiment_module_path(args.experiment, "evaluation")
        with model_config_path.open(encoding="utf-8") as stream:
            model_options = yaml.safe_load(stream)["model"]
        model_path, tokenizer_path = resolve_model_assets(
            model_config_path, experiment_output_dir(args.experiment)
        )
        draft_config_path = (
            model_config_path.parent / model_options["draft_model_config"]
        ).resolve()
        for split in ("train", "eval") if args.split == "all" else (args.split,):
            _run(
                _feature_command(
                    split,
                    dataset_config_path,
                    model_config_path,
                    model_path,
                    tokenizer_path,
                    draft_config_path,
                    evaluation_config_path,
                    args.limit,
                )
            )
        return
    if args.command == "full":
        model_config_path = experiment_module_path(args.experiment, "model")
        dataset_config_path = args.dataset_config or experiment_module_path(
            args.experiment, "dataset"
        )
        evaluation_config_path = experiment_module_path(args.experiment, "evaluation")
        with model_config_path.open(encoding="utf-8") as stream:
            model_options = yaml.safe_load(stream)["model"]
        model_path, tokenizer_path = resolve_model_assets(
            model_config_path, experiment_output_dir(args.experiment)
        )
        draft_config_path = (
            model_config_path.parent / model_options["draft_model_config"]
        ).resolve()
        for split in ("train", "eval"):
            _run(
                _feature_command(
                    split,
                    dataset_config_path,
                    model_config_path,
                    model_path,
                    tokenizer_path,
                    draft_config_path,
                    evaluation_config_path,
                    args.limit,
                )
            )
    if args.command in ("compose", "train", "inference", "full"):
        model_config_path = experiment_module_path(args.experiment, "model")
        evaluation_config_path = experiment_module_path(args.experiment, "evaluation")
        model_path, tokenizer_path = resolve_model_assets(
            model_config_path, experiment_output_dir(args.experiment)
        )
        config = compose(args.experiment, target_model_path=model_path)
        if args.command in ("train", "inference", "full"):
            align_final_evaluation(config)
        validate_specforge(config)
        args.output_config.parent.mkdir(parents=True, exist_ok=True)
        args.output_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        print(f"Wrote SpecForge run config to {args.output_config}")
        if args.command in ("train", "full"):
            _record_training(
                ["specforge", "train", "--config", str(args.output_config)],
                config,
                args.output_config,
            )
        if args.command == "inference":
            _run_inference(
                config, args.output_config, evaluation_config_path, tokenizer_path
            )
        elif args.command == "full":
            _run_inference(
                config, args.output_config, evaluation_config_path, tokenizer_path
            )


if __name__ == "__main__":
    main()
