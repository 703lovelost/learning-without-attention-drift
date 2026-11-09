"""Hydra-driven CLI adapter for the existing SpecForge pipeline commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from eagle3_olmo2.config import ROOT
from eagle3_olmo2.paths import ensure_configured_storage_dirs, resolve_path

_COMMANDS = (
    "prepare-data",
    "prepare-features",
    "compose",
    "train",
    "inference",
    "full",
)


def compose_hydra_config(overrides: tuple[str, ...] | list[str]):
    """Compose model/data/training/evaluation groups with Hydra overrides."""
    config_dir = ROOT / "configs"
    with initialize_config_dir(
        version_base="1.3", config_dir=str(config_dir.resolve())
    ):
        return compose(config_name="hydra/olmo2_1b", overrides=list(overrides))


def write_compat_experiment(config, output_dir: Path) -> tuple[Path, Path]:
    """Write module YAMLs consumed by the established project composer."""
    output_dir.mkdir(parents=True, exist_ok=True)
    modules_dir = output_dir / "modules"
    modules_dir.mkdir(parents=True, exist_ok=True)
    resolved = OmegaConf.to_container(config, resolve=True)
    for name in ("model", "dataset", "training", "evaluation"):
        module_path = modules_dir / f"{name}.yaml"
        module_path.write_text(
            OmegaConf.to_yaml(OmegaConf.create({name: resolved[name]})),
            encoding="utf-8",
        )

    model_config = Path(resolved["model"]["draft_model_config"])
    if not model_config.is_absolute():
        model_config = (ROOT / "configs/models" / model_config).resolve()
    model_value = resolved["model"]
    model_value["draft_model_config"] = str(model_config)
    (modules_dir / "model.yaml").write_text(
        OmegaConf.to_yaml(OmegaConf.create({"model": model_value})),
        encoding="utf-8",
    )

    experiment = {
        "experiment": {
            "name": resolved["experiment"]["name"],
            "model": "modules/model.yaml",
            "dataset": "modules/dataset.yaml",
            "training": "modules/training.yaml",
            "evaluation": "modules/evaluation.yaml",
        },
        "infrastructure": resolved["infrastructure"],
    }
    experiment_path = output_dir / "experiment.yaml"
    experiment_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(experiment)), encoding="utf-8"
    )
    return experiment_path, model_config


@click.command()
@click.argument("command", type=click.Choice(_COMMANDS))
@click.argument("overrides", nargs=-1)
def main(command: str, overrides: tuple[str, ...]) -> None:
    """Run a pipeline command with Hydra config overrides.

    Example: eagle3-olmo2-hydra full training.max_steps=2 cli.limit=16
    """
    ensure_configured_storage_dirs()
    config = compose_hydra_config(overrides)
    resolved = OmegaConf.to_container(config, resolve=True)
    output_dir = resolve_path(resolved["infrastructure"]["output_dir"])
    hydra_dir = output_dir / "hydra-config"
    experiment_path, _ = write_compat_experiment(config, hydra_dir)

    cli_config_path = resolved["cli"].get("output_config")
    if cli_config_path is None:
        cli_config_path = output_dir / "run.yaml"
    else:
        cli_config_path = resolve_path(Path(cli_config_path))

    forwarded = [
        "eagle3-olmo2",
        command,
        "--experiment",
        str(experiment_path),
        "--output-config",
        str(cli_config_path),
    ]
    cli_options = resolved["cli"]
    if cli_options.get("limit") is not None:
        forwarded.extend(("--limit", str(cli_options["limit"])))
    if cli_options.get("split"):
        forwarded.extend(("--split", str(cli_options["split"])))
    if cli_options.get("cuda_visible_devices") is not None:
        forwarded.extend(
            ("--cuda-visible-devices", str(cli_options["cuda_visible_devices"]))
        )

    from eagle3_olmo2.cli import main as pipeline_main

    previous_argv = sys.argv
    try:
        sys.argv = forwarded
        pipeline_main()
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
