"""Offline capture provenance and the single-GPU training plan."""

from __future__ import annotations

import gzip
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from eagle3_olmo2.config import ROOT
from eagle3_olmo2.data import sha256_file
from eagle3_olmo2.paths import resolve_path


def write_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=directory, text=True).strip()


def specforge_revision() -> str:
    pinned = _git(ROOT, "rev-parse", "HEAD:specforge")
    actual = _git(ROOT / "specforge", "rev-parse", "HEAD")
    if actual != pinned or _git(ROOT / "specforge", "status", "--porcelain"):
        raise RuntimeError(f"SpecForge must match clean pinned revision {pinned}")
    return actual


def asset_identity(model_path: str, tokenizer_path: str, draft_path: Path) -> dict:
    target = Path(model_path)
    tokenizer = Path(tokenizer_path)
    files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    )
    return {
        "specforge_sha": specforge_revision(),
        "target_config_sha256": sha256_file(target / "config.json"),
        "tokenizer_files": {
            name: sha256_file(tokenizer / name)
            for name in files
            if (tokenizer / name).is_file()
        },
        "draft_config_sha256": sha256_file(draft_path),
    }


def feature_manifest(path: Path, expected: dict) -> dict:
    manifest = json.loads((path / "feature-manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Features are incompatible with the selected inputs: {path}")
    return manifest


def _pilot_plan(pilot_dir: Path) -> tuple[dict, dict]:
    return (
        json.loads((pilot_dir / "training-plan.json").read_text(encoding="utf-8")),
        json.loads((pilot_dir / "training-status.json").read_text(encoding="utf-8")),
    )


def estimate_capture_storage(
    pilot_dir: Path, dataset: dict, splits: tuple[str, ...]
) -> None:
    plan, _ = _pilot_plan(pilot_dir)
    needs_by_device: dict[int, tuple[Path, int]] = {}

    def reserve(path: Path, size: int) -> None:
        path.mkdir(parents=True, exist_ok=True)
        device = path.stat().st_dev
        _, reserved = needs_by_device.get(device, (path, 0))
        needs_by_device[device] = path, reserved + size

    pending_bytes = 0
    for split in splits:
        features = resolve_path(dataset[f"{split}_features_path"])
        if features.with_name(f"{features.name}.dvc").exists():
            continue
        source = resolve_path(dataset[f"{split}_data_path"])
        manifest = source.with_suffix(source.suffix + ".manifest.json")
        rows = json.loads(manifest.read_text(encoding="utf-8"))["rows_written"]
        pilot_split = "eval" if split == "test" else split
        projected = (
            rows * plan["feature_bytes"][pilot_split] // plan[f"{pilot_split}_samples"]
        )
        reserve(features.parent, projected)
        pending_bytes += projected
    if pending_bytes:
        for key in ("DVC_CACHE_DIR", "DVC_REMOTE_DIR"):
            reserve(Path(os.environ[key]).expanduser(), pending_bytes)
    for path, required in needs_by_device.values():
        if shutil.disk_usage(path).free < required:
            raise RuntimeError(
                f"Pilot-estimated capture needs {required:,} free bytes at {path}"
            )


def plan_training(
    config: dict, output_dir: Path, train_input: Path, pilot_dir: Path | None = None
) -> dict:
    from specforge.runtime.data_plane.offline_reader import list_feature_files
    from specforge.training.schedule import (
        resolve_total_steps,
        validate_fixed_accumulation_plan,
    )

    data = config["data"]
    training = config["training"]
    evaluation = resolve_path(data["eval_hidden_states_path"])
    train_metadata = json.loads(train_input.read_text(encoding="utf-8"))
    eval_files = list_feature_files(str(evaluation))
    samples = train_metadata["eligible_rows"]
    eval_samples = len(eval_files)
    if not samples or not eval_samples:
        raise ValueError(
            "Training requires eligible rows and captured validation samples"
        )
    batch_size = training["batch_size"]
    plan_args = {
        "num_samples": samples,
        "batch_size": batch_size,
        "accumulation_steps": training["accumulation_steps"],
        "num_epochs": training["num_epochs"],
        "max_steps": training.get("max_steps"),
    }
    validate_fixed_accumulation_plan(**plan_args)
    available_microbatches = (samples // batch_size) * training["num_epochs"]
    available_steps = available_microbatches // training["accumulation_steps"]
    steps = resolve_total_steps(total_steps=training.get("total_steps"), **plan_args)
    stop = min(available_steps, training.get("max_steps") or available_steps)
    if steps != stop:
        raise ValueError(
            f"Scheduler horizon {steps} differs from actual {stop} optimizer steps"
        )
    mapping = resolve_path(config["model"]["vocab_mapping_path"])
    result = {
        "run_id": config["run_id"],
        "git_sha": _git(ROOT, "rev-parse", "HEAD"),
        "dirty_tree": bool(_git(ROOT, "status", "--porcelain")),
        "specforge_sha": specforge_revision(),
        "python": sys.version.split()[0],
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "sglang", "specforge")
        },
        "seed": training["seed"],
        "train_samples": samples,
        "eval_samples": eval_samples,
        "effective_batch": batch_size * training["accumulation_steps"],
        "optimizer_steps": steps,
        "eval_interval": training["eval_interval"],
        "save_interval": training["save_interval"],
        "vocab_mapping_sha256": sha256_file(mapping),
        "train_input_manifest_sha256": sha256_file(train_input),
        "train_hidden_states_on_disk": False,
        "train_token_cache_bytes": sum(
            path.stat().st_size
            for path in train_input.parent.rglob("*")
            if path.is_file()
        ),
        "feature_manifests": {
            "eval": sha256_file(evaluation / "feature-manifest.json"),
        },
        "feature_bytes": {
            "eval": sum(Path(file).stat().st_size for file in eval_files),
        },
    }
    if pilot_dir:
        _, status = _pilot_plan(pilot_dir)
        results = status["training_wall_seconds"]
        result["pilot_projection"] = {
            "pilot_report_dir": str(pilot_dir),
            "trainer_seconds_at_pilot_mean_step_time": (
                steps * results / status["optimizer_steps_executed"]
            ),
            "assumptions": "Linear scaling at observed pilot step time; "
            "batch, evaluation cadence, length distribution and storage affect runtime",
        }
    write_json(output_dir / "training-plan.json", result)
    write_json(output_dir / train_input.name, train_metadata)
    return result


def audit_features(
    path: Path, target_config: dict, expected_rows: int, max_length: int
) -> dict:
    import torch

    from specforge.algorithms.eagle3.data import normalize_offline_sample
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    files = list_feature_files(str(path))
    if not files or len(files) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} features in {path}, got {len(files)}"
        )
    width = target_config["hidden_size"]
    vocab = target_config["vocab_size"]
    for index in sorted({0, len(files) // 2, len(files) - 1}):
        file = files[index]
        opener = gzip.open if file.endswith(".gz") else open
        with opener(file, "rb") as stream:
            raw = torch.load(stream, map_location="cpu", weights_only=True)
        sample = normalize_offline_sample(raw, max_length)
        ids = sample["input_ids"]
        mask = sample["loss_mask"]
        length = ids.size(-1)
        if (
            not 1 < length <= max_length
            or ids.min() < 0
            or ids.max() >= vocab
            or mask.shape != ids.shape
            or mask[0, -1]
            or not mask.any()
            or not torch.all((mask == 0) | (mask == 1))
            or sample["hidden_state"].shape != (1, length, 3 * width)
            or sample["target"].shape != (1, length, width)
            or not torch.isfinite(sample["target"]).all()
            or not torch.isfinite(sample["hidden_state"]).all()
        ):
            raise ValueError(f"Invalid OLMo-2 EAGLE-3 features: {file}")
    mapping_path = path / "vocab_mapping/vocab_mapping.pt"
    mapping = torch.load(mapping_path, map_location="cpu", weights_only=True)
    indices = mapping["t2d"].nonzero().flatten()
    if (
        mapping["t2d"].shape != (vocab,)
        or mapping["t2d"].dtype != torch.bool
        or not torch.equal(indices, mapping["d2t"] + torch.arange(len(indices)))
    ):
        raise ValueError(f"Invalid target-to-draft vocabulary mapping: {mapping_path}")
    return {
        "captured_rows": len(files),
        "vocab_mapping_sha256": sha256_file(mapping_path),
    }
