"""Adapt the pinned SpecForge tracker and redact its tracking URI."""

import os
from pathlib import Path

import specforge.tracker
import specforge.training.tracking
from eagle3_olmo2.metrics import SCHEMA_VERSION, canonical_metrics

_native_public_config = specforge.tracker._public_config
_native_metric_names = specforge.training.tracking.training_metric_names
_native_tracker_init = specforge.tracker.MLflowTracker.__init__
_native_tracker_log = specforge.tracker.MLflowTracker.log


def _public_config_without_tracking_uri(args):
    public_config = _native_public_config(args)
    public_config.pop("mlflow_tracking_uri", None)
    return public_config


def _project_metric_names(metrics):
    return canonical_metrics(_native_metric_names(metrics))


def _tag_metrics_schema(self, args, output_dir):
    _native_tracker_init(self, args, output_dir)
    self._project_optimizer_step = None
    if self.rank == 0 and self.is_initialized:
        mlflow = specforge.tracker.mlflow
        mlflow.set_tag("metrics_schema_version", SCHEMA_VERSION)
        if run_id_file := os.getenv("EAGLE3_MLFLOW_RUN_ID_FILE"):
            Path(run_id_file).write_text(
                mlflow.active_run().info.run_id, encoding="utf-8"
            )


def _log_optimizer_step_once(self, values, step=None):
    key = "train/optimizer_step"
    if key in values and step is not None:
        if self._project_optimizer_step == step:
            values = {name: value for name, value in values.items() if name != key}
        else:
            self._project_optimizer_step = step
    return _native_tracker_log(self, values, step=step)


specforge.tracker._public_config = _public_config_without_tracking_uri
specforge.training.tracking.training_metric_names = _project_metric_names
specforge.tracker.MLflowTracker.__init__ = _tag_metrics_schema
specforge.tracker.MLflowTracker.log = _log_optimizer_step_once
