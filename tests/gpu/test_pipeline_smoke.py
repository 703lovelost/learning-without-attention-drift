"""Opt-in RTX-class integration test; run with EAGLE3_GPU_TEST=1."""

from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
ENABLED = os.getenv("EAGLE3_GPU_TEST") == "1"
HAS_MLFLOW = importlib.util.find_spec("mlflow") is not None


@unittest.skipUnless(ENABLED, "set EAGLE3_GPU_TEST=1 to run model/GPU integration")
class PipelineSmokeTest(unittest.TestCase):
    def test_full_pipeline_mlflow_resume_and_inference(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"PyTorch unavailable: {exc}")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        if not HAS_MLFLOW:
            self.skipTest("install mlflow to exercise enabled/disabled tracking")

        with tempfile.TemporaryDirectory(prefix="olmo2-eagle3-gpu-") as temp:
            work = Path(temp)
            model_dir = work / "configs/models"
            model_dir.mkdir(parents=True)
            draft_config = model_dir / "olmo2_eagle3.json"
            draft_config.write_bytes(
                (ROOT / "configs/models/olmo2_1b_eagle3.json").read_bytes()
            )
            model_config = {
                "model": {
                    "target_model_path": os.getenv(
                        "EAGLE3_OLMO2_MODEL_ID", "allenai/OLMo-2-0425-1B-Instruct"
                    ),
                    "revision": os.getenv(
                        "EAGLE3_OLMO2_MODEL_REVISION",
                        "48d788eca847d4d7548f375ad03d3c9312f6139e",
                    ),
                    "tokenizer_path": os.getenv(
                        "EAGLE3_OLMO2_TOKENIZER_ID", "allenai/OLMo-2-0425-1B-Instruct"
                    ),
                    "tokenizer_revision": os.getenv(
                        "EAGLE3_OLMO2_TOKENIZER_REVISION",
                        "48d788eca847d4d7548f375ad03d3c9312f6139e",
                    ),
                    "draft_architecture": "LlamaForCausalLMEagle3",
                    "draft_model_config": str(draft_config),
                    "draft_vocab_size": 32000,
                    "draft_num_hidden_layers": 1,
                    "trust_remote_code": False,
                    "torch_dtype": "bfloat16",
                    "target_backend": "sglang",
                    "embedding_key": "model.embed_tokens.weight",
                    "lm_head_key": "lm_head.weight",
                    "aux_hidden_state_layer_ids": [1, 7, 12],
                }
            }
            dataset_config = {
                "dataset": {
                    "name": "HuggingFaceH4/ultrachat_200k",
                    "revision": "8049631c405ae6576f93f445c6b8166f76f5505a",
                    "parquet_pattern": "data/{split}-*.parquet",
                    "train_split": "train_sft",
                    "eval_split": "test_sft",
                    "seed": 42,
                    "train_samples": None,
                    "eval_samples": 8,
                    "max_length": 256,
                    "chat_template": "olmo2",
                    "train_data_path": str(work / "data/train.jsonl"),
                    "eval_data_path": str(work / "data/eval.jsonl"),
                    "eval_features_path": str(work / "features/eval"),
                    "cache_dir": str(work / "cache"),
                    "preprocessing_batch_size": 1,
                }
            }
            training_config = {
                "training": {
                    "strategy": "eagle3",
                    "num_epochs": 1,
                    "max_steps": 2,
                    "batch_size": 1,
                    "accumulation_steps": 2,
                    "learning_rate": 0.0001,
                    "lr_scheduler": "cosine",
                    "warmup_ratio": 0.0,
                    "max_grad_norm": 0.5,
                    "optimizer_cpu_offload": True,
                    "ttt_length": 3,
                    "attention_backend": "sdpa",
                    "save_interval": 1,
                    "eval_interval": 2,
                    "log_interval": 1,
                    "max_checkpoints": 2,
                    "compact_teacher": True,
                    "compact_teacher_chunk_size": 2048,
                    "seed": 42,
                    "dataloader_num_workers": 0,
                }
            }
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            evaluation_config = {
                "evaluation": {
                    "eval_interval": 2,
                    "max_eval_samples": 8,
                    "inference_max_new_tokens": 8,
                    "inference_port": port,
                }
            }
            for name, value in (
                ("model", model_config),
                ("dataset", dataset_config),
                ("training", training_config),
                ("evaluation", evaluation_config),
            ):
                path = work / f"{name}.yaml"
                path.write_text(
                    yaml.safe_dump(value, sort_keys=False), encoding="utf-8"
                )

            mlruns_uri = f"sqlite:///{work / 'mlflow.db'}"
            experiment = {
                "experiment": {
                    "name": "olmo2-eagle3-gpu-smoke",
                    "model": "model.yaml",
                    "dataset": "dataset.yaml",
                    "training": "training.yaml",
                    "evaluation": "evaluation.yaml",
                },
                "infrastructure": {
                    "device": "cuda:0",
                    "output_dir": str(work / "outputs"),
                    "resume_from_checkpoint": None,
                    "mlflow_tracking_uri": mlruns_uri,
                    "mlflow_experiment_name": "olmo2-eagle3-gpu-smoke",
                },
            }
            experiment_path = work / "experiment.yaml"
            experiment_path.write_text(
                yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8"
            )
            run_config = work / "run.yaml"
            env = os.environ.copy()
            env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
            env.pop("MLFLOW_TRACKING_URI", None)
            env["DVC_CACHE_DIR"] = str(work / "dvc-cache")
            env["DVC_REMOTE_DIR"] = str(work / "dvc-remote")
            cuda_devices = os.getenv("EAGLE3_CUDA_VISIBLE_DEVICES", "0")
            cli = [
                sys.executable,
                "-m",
                "eagle3_olmo2.cli",
                "full",
                "--experiment",
                str(experiment_path),
                "--output-config",
                str(run_config),
                "--limit",
                "8",
                "--cuda-visible-devices",
                cuda_devices,
            ]
            subprocess.run(cli, cwd=ROOT, env=env, check=True, timeout=7200)

            output_dir = work / "outputs"
            self.assertTrue((output_dir / "olmo2-eagle3-gpu-smoke-best").exists())
            self.assertTrue(
                (output_dir / "olmo2-eagle3-gpu-smoke.best_meta.json").exists()
            )
            best_meta = json.loads(
                (output_dir / "olmo2-eagle3-gpu-smoke.best_meta.json").read_text()
            )
            self.assertEqual(best_meta["metric"], "eval/simulated_acc_len")
            self.assertEqual(best_meta["step"], 2)
            self.assertTrue((output_dir / "inference-smoke.json").exists())
            inference = json.loads((output_dir / "inference-smoke.json").read_text())
            self.assertTrue(inference["response"].get("text"))
            self.assertFalse((work / "features/train").exists())
            self.assertFalse((work / "features/train.dvc").exists())
            self.assertTrue((output_dir / "train-input-manifest.json").exists())
            self.assertTrue(
                (work / "features/eval/feature-extraction-memory.json").exists()
            )
            metrics = [
                json.loads(line)
                for line in (output_dir / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any("eval/simulated_acc_len" in row for row in metrics))
            self.assertTrue(
                any(
                    key.startswith("eval/acceptance_rate_")
                    for row in metrics
                    for key in row
                )
            )
            self.assertTrue(any("eval/loss" in row for row in metrics))

            from specforge.training.checkpoint import CheckpointManager

            checkpoint_state = CheckpointManager.read_resume_state(
                str(output_dir / "olmo2-eagle3-gpu-smoke-latest")
            )
            self.assertEqual(checkpoint_state["global_step"], 2)
            self.assertIsNotNone(checkpoint_state["backend"].get("optimizer"))
            self.assertIsNotNone(checkpoint_state["backend"].get("rng"))

            resume_source = output_dir / "olmo2-eagle3-gpu-smoke-step1"
            self.assertTrue(resume_source.exists())
            resume_state = CheckpointManager.read_resume_state(str(resume_source))
            self.assertEqual(resume_state["global_step"], 1)
            self.assertIsNotNone(resume_state["backend"].get("optimizer"))
            self.assertIsNotNone(resume_state["backend"].get("rng"))
            import math

            import torch

            self.assertTrue(
                any(
                    "train/grad_norm" in row and math.isfinite(row["train/grad_norm"])
                    for row in metrics
                )
            )
            self.assertTrue(
                any(
                    not torch.equal(weights, checkpoint_state["draft_state_dict"][name])
                    for name, weights in resume_state["draft_state_dict"].items()
                )
            )

            register_cli = [
                sys.executable,
                "-m",
                "eagle3_olmo2.cli",
                "register-checkpoint",
                "--experiment",
                str(experiment_path),
                "--checkpoint",
                str(resume_source),
            ]
            subprocess.run(register_cli, cwd=ROOT, env=env, check=True, timeout=600)
            resume_output = work / "outputs-resume"
            experiment["infrastructure"]["output_dir"] = str(resume_output)
            experiment["infrastructure"]["resume_from_checkpoint"] = str(resume_source)
            experiment["infrastructure"]["mlflow_tracking_uri"] = None
            experiment_path.write_text(
                yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8"
            )
            env.pop("MLFLOW_TRACKING_URI", None)
            train_cli = [
                sys.executable,
                "-m",
                "eagle3_olmo2.cli",
                "train",
                "--experiment",
                str(experiment_path),
                "--output-config",
                str(run_config),
                "--cuda-visible-devices",
                cuda_devices,
            ]
            subprocess.run(train_cli, cwd=ROOT, env=env, check=True, timeout=1800)
            resumed_state = CheckpointManager.read_resume_state(
                str(resume_output / "olmo2-eagle3-gpu-smoke-latest")
            )
            self.assertEqual(resumed_state["global_step"], 2)
            self.assertEqual(
                resumed_state["backend"]["optimizer"]["scheduler_state_dict"],
                checkpoint_state["backend"]["optimizer"]["scheduler_state_dict"],
            )
            self.assertIsNotNone(resumed_state["backend"].get("rng"))
            resumed_best = json.loads(
                (resume_output / "olmo2-eagle3-gpu-smoke.best_meta.json").read_text()
            )
            self.assertEqual(resumed_best["step"], 2)

            local_output = work / "outputs-local"
            experiment["experiment"]["name"] = "olmo2-eagle3-local-smoke"
            experiment["infrastructure"]["output_dir"] = str(local_output)
            experiment["infrastructure"]["resume_from_checkpoint"] = None
            experiment_path.write_text(
                yaml.safe_dump(experiment, sort_keys=False), encoding="utf-8"
            )
            subprocess.run(train_cli, cwd=ROOT, env=env, check=True, timeout=1800)
            local_metrics = [
                json.loads(line)
                for line in (local_output / "metrics.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any("train/loss" in row for row in local_metrics))

            from mlflow import MlflowClient

            client = MlflowClient(tracking_uri=mlruns_uri)
            mlflow_experiment = client.get_experiment_by_name("olmo2-eagle3-gpu-smoke")
            self.assertIsNotNone(mlflow_experiment)
            runs = client.search_runs([mlflow_experiment.experiment_id], max_results=10)
            self.assertTrue(runs)
            self.assertIn("train/loss", runs[0].data.metrics)
            self.assertIn("eval/loss", runs[0].data.metrics)
            self.assertIn("eval/simulated_acc_len", runs[0].data.metrics)
            self.assertGreater(
                runs[0].data.metrics["gpu/capture_eval_peak_reserved_mib"], 0
            )
            self.assertTrue(client.list_artifacts(runs[0].info.run_id, "plots"))
            self.assertTrue(
                client.get_run(runs[0].info.run_id).data.params.get(
                    "training.optimizer"
                )
            )


if __name__ == "__main__":
    unittest.main()
