"""Compose project YAML modules for the single-GPU SpecForge trainer."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

from eagle3_olmo2.paths import expand_env as _expand_env
from eagle3_olmo2.paths import resolve_path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT = ROOT / "configs/experiments/olmo2_1b_baseline.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = yaml.safe_load(stream) or {}
    if not isinstance(result, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return result


def experiment_module_path(experiment_path: Path, name: str) -> Path:
    experiment_path = experiment_path.resolve()
    experiment = _read_yaml(experiment_path).get("experiment", {})
    reference = experiment.get(name)
    if not reference:
        raise ValueError(f"experiment must reference {name!r} YAML")
    return (experiment_path.parent / reference).resolve()


def compose(
    experiment_path: Path,
    *,
    dataset_config_path: Path | None = None,
    target_model_path: str | None = None,
) -> dict[str, Any]:
    experiment_path = experiment_path.resolve()
    root_config = _read_yaml(experiment_path)
    exp = root_config.get("experiment", {})
    paths = [exp.get(name) for name in ("model", "dataset", "training", "evaluation")]
    if any(not path for path in paths):
        raise ValueError(
            "experiment must reference model, dataset, training, and evaluation YAML"
        )

    module_paths = [(experiment_path.parent / path).resolve() for path in paths]
    if dataset_config_path is not None:
        module_paths[1] = dataset_config_path.resolve()
    modules = [_expand_env(_read_yaml(path)) for path in module_paths]
    model = modules[0]["model"]
    dataset = modules[1]["dataset"]
    training = copy.deepcopy(modules[2]["training"])
    optimizer = str(training.pop("optimizer", "adamw")).lower()
    if optimizer != "adamw":
        raise ValueError(
            "the pinned SpecForge EAGLE-3 trainer uses AdamW; "
            f"unsupported project optimizer={optimizer!r}"
        )
    evaluation = modules[3]["evaluation"]
    infra = _expand_env(root_config.get("infrastructure", {}))

    model_path = target_model_path or model["target_model_path"]
    data = {
        "train_data_path": dataset["train_data_path"],
        "eval_hidden_states_path": dataset["eval_features_path"],
        "max_length": dataset["max_length"],
        "chat_template": dataset["chat_template"],
        "cache_dir": dataset["cache_dir"],
        "dataloader_num_workers": training.pop("dataloader_num_workers", 0),
    }
    train_model_config = (
        module_paths[0].parent / model["draft_model_config"]
    ).resolve()
    with train_model_config.open(encoding="utf-8") as stream:
        draft_config = json.load(stream)
    configured_architectures = draft_config.get("architectures", [])
    if configured_architectures != [model["draft_architecture"]]:
        raise ValueError(
            f"draft config architecture {configured_architectures!r} does not match "
            f"model.draft_architecture={model['draft_architecture']!r}"
        )
    if int(draft_config.get("draft_vocab_size", -1)) != int(model["draft_vocab_size"]):
        raise ValueError("model.draft_vocab_size must match the selected draft config")
    model_config = {
        "target_model_path": model_path,
        "draft_model_config": str(train_model_config),
        "draft_num_hidden_layers": model["draft_num_hidden_layers"],
        "target_backend": model["target_backend"],
        "train_target_cpu_offload": model.get("train_target_cpu_offload", True),
        "trust_remote_code": model["trust_remote_code"],
        "embedding_key": model["embedding_key"],
        "lm_head_key": model["lm_head_key"],
        "vocab_mapping_path": "",
        "aux_hidden_state_layer_ids": model["aux_hidden_state_layer_ids"],
        "torch_dtype": model["torch_dtype"],
        "cache_dir": str(resolve_path(dataset["cache_dir"])),
        "load_target_embedding": True,
    }
    training["eval_interval"] = evaluation["eval_interval"]
    training["resume_from"] = infra.get("resume_from_checkpoint")

    # Keep credentials in the process environment rather than serializing them
    # into the resolved run config or MLflow parameter/artifact logs.
    tracking_enabled = bool(os.getenv("MLFLOW_TRACKING_URI"))
    tracking = {
        "report_to": "mlflow" if tracking_enabled else "none",
        "mlflow_tracking_uri": None,
        "mlflow_experiment_name": infra.get("mlflow_experiment_name") or "eagle3-olmo2",
        "mlflow_run_name": exp["name"],
    }
    run = {
        "model": model_config,
        "data": data,
        "training": training,
        "tracking": tracking,
        "run_id": exp["name"],
        "output_dir": str(resolve_path(infra["output_dir"])),
    }
    return run


def validate_specforge(config: dict[str, Any]):
    """Validate the colocated runtime and its hidden-state batch contract."""
    import sys

    specforge_root = str(ROOT / "specforge")
    if specforge_root not in sys.path:
        sys.path.insert(0, specforge_root)
    from specforge.config import Config, DataConfig, ModelConfig

    class LiveDataConfig(DataConfig):
        tokenized_train_path: str | None = None

    class LiveModelConfig(ModelConfig):
        train_target_cpu_offload: bool = True

    class LiveTrainConfig(Config):
        data: LiveDataConfig
        model: LiveModelConfig

        @property
        def mode(self) -> str:
            # SpecForge uses this to select the colocated hidden-state contract.
            # The project supplies the batches at runtime through the loader.
            return "offline"

    LiveTrainConfig.model_rebuild(
        _types_namespace={
            "LiveDataConfig": LiveDataConfig,
            "LiveModelConfig": LiveModelConfig,
        }
    )
    resolved = LiveTrainConfig.model_validate(config)
    training = resolved.training
    if (
        not resolved.data.train_data_path
        or resolved.deployment.mode != "local_colocated"
        or (training.tp_size, training.sp_ulysses_size, training.sp_ring_size)
        != (1, 1, 1)
        or training.attention_backend == "usp"
    ):
        raise ValueError("Live EAGLE-3 requires JSONL input and a single colocated GPU")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--output",
        type=Path,
        default=resolve_path(
            Path(os.getenv("EAGLE3_OUTPUT_DIR", "outputs/olmo2_1b_eagle3")) / "run.yaml"
        ),
    )
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    config = compose(args.experiment)
    if not args.no_validate:
        validate_specforge(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Wrote SpecForge run config to {args.output}")


if __name__ == "__main__":
    main()
