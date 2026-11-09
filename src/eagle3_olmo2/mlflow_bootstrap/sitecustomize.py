"""Redact the MLflow tracking URI from SpecForge's logged run parameters."""

from __future__ import annotations

import specforge.tracker

_native_public_config = specforge.tracker._public_config


def _public_config_without_tracking_uri(args):
    public_config = _native_public_config(args)
    public_config.pop("mlflow_tracking_uri", None)
    return public_config


specforge.tracker._public_config = _public_config_without_tracking_uri
