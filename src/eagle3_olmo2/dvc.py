"""DVC storage setup and retrieval for repository data and model artifacts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def _configured_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"{variable} must be set in the environment")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{variable} must be an absolute path: {path}")
    return path.resolve()


def configure_local_storage(root: Path) -> tuple[Path, Path]:
    """Set the machine-local DVC cache and local remote from environment vars."""
    cache_dir = _configured_path("DVC_CACHE_DIR")
    remote_dir = _configured_path("DVC_REMOTE_DIR")
    if (
        cache_dir == remote_dir
        or cache_dir in remote_dir.parents
        or remote_dir in cache_dir.parents
    ):
        raise RuntimeError(
            "DVC_CACHE_DIR and DVC_REMOTE_DIR must be separate directories"
        )

    dvc_config = root / ".dvc" / "config"
    if not dvc_config.is_file():
        raise RuntimeError("DVC is not initialized; run `uv run dvc init` first")

    cache_dir.mkdir(parents=True, exist_ok=True)
    remote_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["dvc", "cache", "dir", "--local", str(cache_dir)],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            "dvc",
            "remote",
            "add",
            "--local",
            "--default",
            "--force",
            "local-storage",
            str(remote_dir),
        ],
        cwd=root,
        check=True,
    )
    return cache_dir, remote_dir


def ensure_dvc_artifacts_available(root: Path) -> bool:
    """Pull repository-tracked DVC artifacts before training or inference.

    Returns False when the repository has no DVC-tracked artifacts. A broken
    DVC setup or failed pull raises instead of silently continuing with stale or
    missing local files.
    """
    ignored_roots = {".git", ".venv", "specforge"}
    references = [root / "dvc.lock"] if (root / "dvc.lock").is_file() else []
    references.extend(
        path
        for path in root.rglob("*.dvc")
        if path.relative_to(root).parts[0] not in ignored_roots
    )
    if not references:
        return False
    if not (root / ".dvc" / "config").is_file():
        raise RuntimeError("DVC references exist, but DVC is not initialized")
    subprocess.run(["dvc", "pull"], cwd=root, check=True)
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    cache_dir, remote_dir = configure_local_storage(root)
    print(f"Configured DVC cache at {cache_dir}")
    print(f"Configured local DVC remote at {remote_dir}")


if __name__ == "__main__":
    main()
