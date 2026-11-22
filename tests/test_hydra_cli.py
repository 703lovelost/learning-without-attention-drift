import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from eagle3_olmo2.config import compose
from eagle3_olmo2.hydra_cli import compose_hydra_config, write_compat_experiment


class HydraConfigTest(unittest.TestCase):
    def test_hydra_storage_paths_follow_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "EAGLE3_DATA_DIR": str(storage / "data"),
                    "EAGLE3_OUTPUT_DIR": str(storage / "outputs"),
                    "EAGLE3_CACHE_DIR": str(storage / "cache"),
                },
            ):
                config = compose_hydra_config([])
                resolved = OmegaConf.to_container(config, resolve=True)

            self.assertEqual(
                resolved["dataset"]["train_data_path"],
                str(storage / "data/ultrachat/train_sft.jsonl"),
            )
            self.assertEqual(
                resolved["infrastructure"]["output_dir"], str(storage / "outputs")
            )
            self.assertEqual(resolved["dataset"]["cache_dir"], str(storage / "cache"))

    def test_hierarchical_overrides_reach_existing_config_modules(self):
        config = compose_hydra_config(
            [
                "training.max_steps=3",
                "cli.limit=8",
                "model.torch_dtype=float32",
                "model.train_target_cpu_offload=false",
            ]
        )
        self.assertEqual(config.training.max_steps, 3)
        self.assertEqual(config.cli.limit, 8)
        self.assertEqual(config.model.torch_dtype, "float32")
        self.assertFalse(config.model.train_target_cpu_offload)

    def test_hydra_config_exports_compatible_project_modules(self):
        config = compose_hydra_config(["training.max_steps=3"])
        with tempfile.TemporaryDirectory() as directory:
            experiment_path, model_config = write_compat_experiment(
                config, Path(directory)
            )
            self.assertTrue(experiment_path.is_file())
            self.assertTrue(model_config.is_file())
            experiment_text = experiment_path.read_text(encoding="utf-8")
            native_config = compose(experiment_path)
            training_text = (Path(directory) / "modules/training.yaml").read_text(
                encoding="utf-8"
            )

        self.assertIn("modules/model.yaml", experiment_text)
        self.assertIn("max_steps: 3", training_text)
        self.assertEqual(native_config["training"]["max_steps"], 3)


if __name__ == "__main__":
    unittest.main()


def test_training_profiles_separate_paths_and_align_pilot_with_full():
    configs = {
        name: compose_hydra_config([], name) for name in ("smoke", "pilot", "full")
    }
    assert len({cfg.infrastructure.output_dir for cfg in configs.values()}) == 3
    assert len({cfg.dataset.train_data_path for cfg in configs.values()}) == 3
    assert all("train_features_path" not in cfg.dataset for cfg in configs.values())
    assert configs["smoke"].training.max_steps == 2
    assert configs["pilot"].dataset.train_samples == 512
    assert configs["pilot"].training.max_steps is None
    assert configs["full"].training.max_steps is None
    assert configs["full"].dataset.train_samples is None
    for key in (
        "num_epochs",
        "batch_size",
        "accumulation_steps",
        "learning_rate",
        "lr_scheduler",
        "warmup_ratio",
        "ttt_length",
    ):
        assert configs["pilot"].training[key] == configs["full"].training[key]
