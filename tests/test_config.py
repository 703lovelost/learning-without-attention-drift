import json
import os
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from eagle3_olmo2.cli import (
    _configure_tracking_environment,
    _discard_future_best,
    _feature_command,
    _log_mlflow_artifacts,
    _target_head_compatible_snapshot,
    _write_metric_plots,
    _write_svg,
    align_final_evaluation,
    require_cuda,
    resolve_model_assets,
)
from eagle3_olmo2.config import DEFAULT_EXPERIMENT, compose, validate_specforge


class ConfigCompositionTest(unittest.TestCase):
    def test_pipeline_paths_follow_storage_environment(self):
        with TemporaryDirectory() as directory:
            storage = Path(directory)
            with patch.dict(
                os.environ,
                {
                    "EAGLE3_DATA_DIR": str(storage / "data"),
                    "EAGLE3_OUTPUT_DIR": str(storage / "outputs"),
                    "EAGLE3_CACHE_DIR": str(storage / "cache"),
                },
            ):
                config = compose(DEFAULT_EXPERIMENT)

        self.assertEqual(
            config["data"]["hidden_states_path"], str(storage / "data/features/train")
        )
        self.assertEqual(
            config["data"]["eval_hidden_states_path"],
            str(storage / "data/features/eval"),
        )
        self.assertEqual(config["model"]["cache_dir"], str(storage / "cache"))
        self.assertEqual(config["output_dir"], str(storage / "outputs"))
        self.assertEqual(
            config["model"]["vocab_mapping_path"],
            str(storage / "data/features/train/vocab_mapping/vocab_mapping.pt"),
        )

    def test_experiment_tracking_uri_is_passed_to_child_environment(self):
        tracking_uri = "sqlite:////tmp/eagle3-test-mlflow.db"
        with patch.dict(os.environ, {}, clear=True):
            _configure_tracking_environment({"mlflow_tracking_uri": tracking_uri})
            self.assertEqual(os.environ["MLFLOW_TRACKING_URI"], tracking_uri)

    def test_explicit_tracking_environment_takes_precedence(self):
        configured_uri = "sqlite:////tmp/configured.db"
        environment_uri = "sqlite:////tmp/environment.db"
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": environment_uri}):
            _configure_tracking_environment({"mlflow_tracking_uri": configured_uri})
            self.assertEqual(os.environ["MLFLOW_TRACKING_URI"], environment_uri)

    def test_tracking_uri_credentials_stay_out_of_resolved_config(self):
        secret_uri = "https://user:private-token@example.invalid/mlflow"
        with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": secret_uri}):
            config = compose(DEFAULT_EXPERIMENT)
        self.assertEqual(config["tracking"]["report_to"], "mlflow")
        self.assertIsNone(config["tracking"]["mlflow_tracking_uri"])
        self.assertNotIn("private-token", repr(config))

    def test_compose_produces_offline_eagle3_config(self):
        config = compose(DEFAULT_EXPERIMENT, target_model_path="/models/olmo2")
        self.assertEqual(config["model"]["target_model_path"], "/models/olmo2")
        self.assertEqual(config["training"]["strategy"], "eagle3")
        self.assertNotIn("optimizer", config["training"])
        self.assertEqual(config["training"]["lr_scheduler"], "cosine")
        self.assertEqual(config["data"]["hidden_states_path"], "data/features/train")
        self.assertEqual(
            config["data"]["eval_hidden_states_path"], "data/features/eval"
        )
        self.assertEqual(config["training"]["ttt_length"], 7)
        expected_tracker = "mlflow" if os.getenv("MLFLOW_TRACKING_URI") else "none"
        self.assertEqual(config["tracking"]["report_to"], expected_tracker)
        self.assertIsNone(config["tracking"]["mlflow_tracking_uri"])
        validate_specforge(config)

    def test_metric_plot_is_written_as_svg(self):
        with TemporaryDirectory() as directory:
            plot = Path(directory) / "loss.svg"
            _write_svg(plot, [(1, 2.0), (2, 1.0)], "train/loss")
            self.assertIn("<svg", plot.read_text(encoding="utf-8"))

    def test_metric_plot_writer_covers_loss_acceptance_and_acceptance_rate(self):
        with TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _write_metric_plots(
                {
                    "train/loss": [(1, 2.0), (2, 1.0)],
                    "eval/avg_loss": [(1, 1.5), (2, 0.8)],
                    "eval/simulated_acc_len": [(1, 1.1), (2, 1.4)],
                    "eval/acceptance_rate_1": [(1, 0.5), (2, 0.7)],
                },
                output_dir,
            )
            plots = sorted(path.name for path in output_dir.glob("*.svg"))

        self.assertEqual(
            plots,
            [
                "acceptance_length.svg",
                "eval_acceptance_rate_1.svg",
                "train_loss.svg",
                "validation_loss.svg",
            ],
        )

    def test_cuda_preflight_fails_with_driver_guidance(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False)
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "NVIDIA driver"):
                require_cuda()

    def test_resume_from_older_checkpoint_drops_future_best_alias(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "run-step4"
            checkpoint.mkdir()
            (output / "run-best").symlink_to(checkpoint.name)
            metadata = output / "run.best_meta.json"
            metadata.write_text(json.dumps({"step": 4, "score": 2.0}))

            self.assertTrue(_discard_future_best(output, "run", resume_step=2))
            self.assertFalse(metadata.exists())
            self.assertFalse((output / "run-best").exists())
            self.assertTrue(checkpoint.exists())

            metadata.write_text(json.dumps({"step": 1, "score": 1.0}))
            self.assertFalse(_discard_future_best(output, "run", resume_step=2))
            self.assertTrue(metadata.exists())

    def test_mlflow_artifact_upload_logs_vram_as_metrics(self):
        calls = {"metrics": [], "artifacts": [], "params": []}

        class FakeClient:
            def __init__(self, tracking_uri=None):
                pass

            def get_experiment_by_name(self, name):
                return types.SimpleNamespace(experiment_id="exp")

            def search_runs(self, experiment_ids, **kwargs):
                return [types.SimpleNamespace(info=types.SimpleNamespace(run_id="run"))]

            def log_artifact(self, run_id, path, artifact_path=None):
                calls["artifacts"].append((path, artifact_path))

            def log_metric(self, run_id, key, value, step):
                calls["metrics"].append((key, value, step))

            def log_param(self, run_id, key, value):
                calls["params"].append((key, value))

        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            train_features = root / "features/train"
            eval_features = root / "features/eval"
            output.mkdir()
            train_features.mkdir(parents=True)
            eval_features.mkdir(parents=True)
            (output / "metrics.jsonl").write_text('{"step": 7, "loss": 1.0}\n')
            (output / "gpu-memory.json").write_text(
                json.dumps(
                    {
                        "peak_used_mib": 5000,
                        "peak_training_used_mib": 4500,
                        "peak_evaluation_used_mib": 4800,
                    }
                )
            )
            for features in (train_features, eval_features):
                (features / "feature-extraction-memory.json").write_text(
                    json.dumps(
                        {
                            "peak_allocated_bytes": 1048576,
                            "peak_reserved_bytes": 2097152,
                        }
                    )
                )
            config_path = root / "run.yaml"
            config_path.write_text("model: {}\ntraining: {}\n")
            config = {
                "run_id": "olmo2",
                "tracking": {
                    "mlflow_tracking_uri": None,
                    "mlflow_experiment_name": "e",
                },
                "data": {
                    "hidden_states_path": str(train_features),
                    "eval_hidden_states_path": str(eval_features),
                },
                "model": {},
                "training": {},
            }
            fake_mlflow = types.ModuleType("mlflow")
            fake_mlflow.MlflowClient = FakeClient
            with patch.dict(sys.modules, {"mlflow": fake_mlflow}):
                _log_mlflow_artifacts(config, output, config_path)

        logged = {name: (value, step) for name, value, step in calls["metrics"]}
        self.assertEqual(logged["gpu/peak_used_mib"], (5000.0, 7))
        self.assertEqual(logged["gpu/training_peak_used_mib"], (4500.0, 7))
        self.assertEqual(logged["gpu/evaluation_peak_used_mib"], (4800.0, 7))
        self.assertEqual(logged["gpu/capture_train_peak_reserved_mib"], (2.0, 7))
        self.assertEqual(logged["gpu/capture_eval_peak_reserved_mib"], (2.0, 7))
        self.assertIn(("training.optimizer", "adamw"), calls["params"])
        self.assertTrue(
            any(name == "project.git_commit" for name, _ in calls["params"])
        )

    def test_eval_interval_lands_on_final_offline_update(self):
        config = compose(DEFAULT_EXPERIMENT, target_model_path="/models/olmo2")
        with TemporaryDirectory() as directory:
            feature_dir = Path(directory) / "features"
            feature_dir.mkdir()
            for index in range(20):
                (feature_dir / f"{index}.ckpt").touch()
            config["data"]["hidden_states_path"] = str(feature_dir)
            config["training"]["max_steps"] = 100
            align_final_evaluation(config)
            self.assertEqual(config["training"]["eval_interval"], 5)
            self.assertEqual(config["training"]["log_interval"], 5)

    def test_local_model_and_tokenizer_paths_are_recorded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model"
            tokenizer_path = root / "tokenizer"
            model_path.mkdir()
            tokenizer_path.mkdir()
            config_path = root / "model.yaml"
            config_path.write_text(
                "model:\n"
                f"  target_model_path: {model_path}\n"
                "  revision: pinned-model-sha\n"
                f"  tokenizer_path: {tokenizer_path}\n"
                "  tokenizer_revision: pinned-tokenizer-sha\n",
                encoding="utf-8",
            )
            output_dir = root / "outputs"
            resolved_model, resolved_tokenizer = resolve_model_assets(
                config_path, output_dir
            )
            self.assertEqual(resolved_model, str(model_path.resolve()))
            self.assertEqual(resolved_tokenizer, str(tokenizer_path.resolve()))
            manifest = json.loads((output_dir / "model-snapshot.json").read_text())
            self.assertEqual(manifest["requested_revision"], "pinned-model-sha")
            self.assertEqual(manifest["tokenizer_revision"], "pinned-tokenizer-sha")

    def test_single_file_target_snapshot_gets_target_head_index_overlay(self):
        import torch
        from safetensors.torch import save_file

        with TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            save_file(
                {
                    "lm_head.weight": torch.zeros(2, 2),
                    "model.embed_tokens.weight": torch.zeros(2, 2),
                },
                snapshot / "model.safetensors",
            )
            (snapshot / "config.json").write_text("{}\n", encoding="utf-8")

            compatible = Path(
                _target_head_compatible_snapshot(str(snapshot), root / "out")
            )
            self.assertEqual(compatible.name, "model-with-index")
            self.assertTrue((compatible / "config.json").is_symlink())
            index = json.loads(
                (compatible / "model.safetensors.index.json").read_text()
            )
            self.assertEqual(index["weight_map"]["lm_head.weight"], "model.safetensors")
            self.assertEqual(
                index["weight_map"]["model.embed_tokens.weight"],
                "model.safetensors",
            )

    def test_feature_capture_uses_single_process_torchrun(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            train_data = root / "train.jsonl"
            eval_data = root / "eval.jsonl"
            train_data.write_text('{"conversations": []}\n' * 3, encoding="utf-8")
            eval_data.write_text('{"conversations": []}\n' * 2, encoding="utf-8")
            dataset_path = root / "dataset.yaml"
            dataset_path.write_text(
                "dataset:\n"
                f"  train_data_path: {train_data}\n"
                f"  eval_data_path: {eval_data}\n"
                "  train_features_path: features/train\n"
                "  eval_features_path: features/eval\n"
                "  train_samples: 8\n"
                "  eval_samples: 4\n"
                "  max_length: 128\n"
                "  preprocessing_batch_size: 1\n"
                "  cache_dir: .cache/huggingface\n"
                "  chat_template: olmo2\n",
                encoding="utf-8",
            )
            model_config_path = root / "model.yaml"
            model_config_path.write_text(
                "model:\n"
                "  capture_attention_backend: triton\n"
                "  sglang_mem_fraction_static: 0.35\n",
                encoding="utf-8",
            )
            draft_config_path = root / "draft.json"
            draft_config_path.touch()
            evaluation_path = root / "evaluation.yaml"
            evaluation_path.write_text("evaluation: {}\n", encoding="utf-8")

            command = _feature_command(
                "train",
                dataset_path,
                model_config_path,
                "/models/target",
                "/models/tokenizer",
                draft_config_path,
                evaluation_path,
            )

            self.assertEqual(command[1:3], ["-m", "torch.distributed.run"])
            self.assertIn("--nproc_per_node=1", command)
            self.assertIn("--module", command)
            self.assertIn("eagle3_olmo2.prepare_features", command)
            self.assertIn("--target-model-path", command)
            self.assertEqual(
                command[command.index("--sglang-attention-backend") + 1],
                "triton",
            )

            limited_command = _feature_command(
                "train",
                dataset_path,
                model_config_path,
                "/models/target",
                "/models/tokenizer",
                draft_config_path,
                evaluation_path,
                1,
            )
            sample_arg = limited_command.index("--num-samples")
            self.assertEqual(limited_command[sample_arg + 1], "1")

            eval_command = _feature_command(
                "eval",
                dataset_path,
                model_config_path,
                "/models/target",
                "/models/tokenizer",
                draft_config_path,
                evaluation_path,
            )
            sample_arg = eval_command.index("--num-samples")
            self.assertEqual(eval_command[sample_arg + 1], "2")


if __name__ == "__main__":
    unittest.main()
