import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from eagle3_olmo2.metrics import canonical_metrics

BOOTSTRAP = Path(__file__).resolve().parents[1] / "src/eagle3_olmo2/mlflow_bootstrap"


def test_metric_aliases_preserve_source_values_and_split():
    assert canonical_metrics(
        {
            "train/loss": 0.47,
            "train/kl_loss": 0.47,
            "eval/avg_loss": 0.41,
            "eval/expected_acceptance_0": 0.7,
            "eval/acceptance_rate_0": 0.7,
            "train/acceptance_rate_0": 0.8,
            "train/optimizer_step": 11.0,
        }
    ) == {
        "train/loss": 0.47,
        "eval/loss": 0.41,
        "eval/acceptance_rate_0": 0.7,
        "train/acceptance_rate_0": 0.8,
        "train/optimizer_step": 11.0,
    }


def test_real_native_tracker_logs_each_metric_once_to_sqlite():
    with TemporaryDirectory() as folder:
        env = os.environ.copy()
        # Add the hook before starting the interpreter exactly as the runner does.
        env["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(BOOTSTRAP), env.get("PYTHONPATH")))
        )
        env["MLFLOW_TRACKING_URI"] = f"sqlite:///{folder}/mlflow.db"
        subprocess.run(
            [
                sys.executable,
                "-c",
                """
import os
from types import SimpleNamespace

from mlflow import MlflowClient
from specforge.training.tracking import create_tracker_logger

args = SimpleNamespace(
    report_to="mlflow",
    mlflow_experiment_name="metric-contract",
    mlflow_run_name="new-run",
    mlflow_tracking_uri=None,
)
with create_tracker_logger(args, ".") as logger:
    logger({"loss": 0.5, "kl_loss": 0.5, "acceptance_rate": 0.4,
            "expected_acceptance": 0.4, "expected_acceptance_0": 0.3,
            "acceptance_rate_0": 0.3}, 7)
    logger({"eval/avg_loss": 0.25, "eval/expected_acceptance_0": 0.2,
            "eval/acceptance_rate_0": 0.2}, 7)

client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
experiment = client.get_experiment_by_name("metric-contract")
run, = client.search_runs([experiment.experiment_id])
assert run.data.tags["metrics_schema_version"] == "eagle3-offline-v1"
assert "mlflow_tracking_uri" not in run.data.params
for name, value in [
    ("train/loss", 0.5), ("eval/loss", 0.25),
    ("train/acceptance_rate", 0.4), ("train/acceptance_rate_0", 0.3),
    ("eval/acceptance_rate_0", 0.2), ("train/optimizer_step", 7.0),
]:
    observations = client.get_metric_history(run.info.run_id, name)
    assert [(item.step, item.value) for item in observations] == [(7, value)]
assert not any("kl_loss" in key or "expected_acceptance" in key for key in run.data.metrics)
""",
            ],
            env=env,
            check=True,
        )
