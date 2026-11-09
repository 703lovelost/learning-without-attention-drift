"""Resolve project paths and initialize user-configured external storage."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
_OMEGA_ENV = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")
_STORAGE_ENV_VARS = (
    "EAGLE3_DATA_DIR",
    "EAGLE3_OUTPUT_DIR",
    "EAGLE3_CACHE_DIR",
    "HF_HOME",
    "DVC_CACHE_DIR",
    "DVC_REMOTE_DIR",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "CUDA_CACHE_PATH",
    "TORCHINDUCTOR_CACHE_DIR",
)


def expand_env(value: Any) -> Any:
    """Expand ``${NAME}`` and ``${NAME:-default}`` in nested YAML values."""
    if isinstance(value, str):
        value = _OMEGA_ENV.sub(
            lambda match: os.getenv(match.group(1), match.group(2) or ""), value
        )
        return _ENV.sub(
            lambda match: os.getenv(match.group(1), match.group(2) or ""), value
        )
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def resolve_path(value: str | Path, *, base: Path = ROOT) -> Path:
    """Resolve configured paths, preserving absolute external storage paths."""
    expanded = Path(str(expand_env(value))).expanduser()
    if not expanded.is_absolute():
        expanded = base / expanded
    return expanded.resolve()


def ensure_configured_storage_dirs() -> None:
    """Create configured data/cache/temp directories before pipeline work."""
    for variable in _STORAGE_ENV_VARS:
        value = os.environ.get(variable)
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{variable} must be an absolute path: {path}")
        path.mkdir(parents=True, exist_ok=True)
