"""Version external pipeline artifacts in colocated DVC workspaces."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

REMOTE = "local-storage"


def _configured_path(variable: str) -> Path:
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"{variable} must be set in the environment")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{variable} must be an absolute path: {path}")
    return path.resolve()


def _storage_paths() -> tuple[Path, Path]:
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
    return cache_dir, remote_dir


def _dvc(root: Path, *args: str, capture: bool = False) -> str:
    env = os.environ.copy()
    env.setdefault(
        "DVC_SITE_CACHE_DIR", str(_configured_path("DVC_CACHE_DIR") / "site")
    )
    try:
        process = subprocess.run(
            ["dvc", *args],
            cwd=root,
            check=True,
            text=True,
            capture_output=capture,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Install the pipeline extra to use DVC") from exc
    except subprocess.CalledProcessError as exc:
        detail = f": {exc.stderr.strip()}" if exc.stderr else ""
        raise RuntimeError(f"DVC {args[0]} failed in {root}{detail}") from exc
    return process.stdout if capture else ""


def configure_local_storage(root: Path) -> tuple[Path, Path]:
    """Configure a DVC workspace without updating its shared defaults."""
    cache_dir, remote_dir = _storage_paths()
    if not (root / ".dvc/config").is_file():
        raise RuntimeError(f"DVC is not initialized in {root}")
    for directory in (cache_dir, remote_dir):
        directory.mkdir(parents=True, exist_ok=True)
        if not os.access(directory, os.R_OK | os.W_OK | os.X_OK):
            raise RuntimeError(f"DVC storage is inaccessible: {directory}")
    _dvc(root, "cache", "dir", "--local", str(cache_dir))
    _dvc(root, "remote", "add", "--local", "--force", REMOTE, str(remote_dir))
    return cache_dir, remote_dir


def _check_storage(root: Path) -> None:
    cache_dir, remote_dir = _storage_paths()
    for directory in (cache_dir, remote_dir):
        if not directory.is_dir() or not os.access(directory, os.R_OK | os.X_OK):
            raise RuntimeError(f"DVC storage is inaccessible: {directory}")
    for actual, expected in (
        (_dvc(root, "cache", "dir", capture=True).strip(), cache_dir),
        (
            _dvc(root, "config", f"remote.{REMOTE}.url", capture=True).strip(),
            remote_dir,
        ),
    ):
        if Path(actual) != expected:
            raise RuntimeError(
                f"DVC storage differs from the configured local path: {expected}"
            )


def ensure_dvc_artifacts_available(*artifacts: Path) -> None:
    """Require exact refs, restore them, and check cache/remote and worktree status."""
    by_workspace: dict[Path, list[Path]] = {}
    for artifact in artifacts:
        root = artifact.parent
        if not (root / ".dvc/config").is_file():
            raise RuntimeError(
                f"DVC is not initialized in {root}; bootstrap {artifact.name}"
            )
        ref = root / f"{artifact.name}.dvc"
        if not ref.is_file():
            raise RuntimeError(f"Required DVC reference is missing: {ref}")
        outputs = (yaml.safe_load(ref.read_text(encoding="utf-8")) or {}).get(
            "outs", []
        )
        if (
            len(outputs) != 1
            or outputs[0].get("path") != artifact.name
            or not outputs[0].get("md5")
        ):
            raise RuntimeError(f"DVC reference must track exactly {artifact}: {ref}")
        by_workspace.setdefault(root, []).append(artifact)
    for root, paths in by_workspace.items():
        _check_storage(root)
        refs = [f"{path.name}.dvc" for path in paths]
        _dvc(root, "pull", "-r", REMOTE, *refs)
        if not all(path.exists() for path in paths):
            raise RuntimeError(f"DVC did not restore every required artifact in {root}")
        cloud = _dvc(root, "status", "-c", "-r", REMOTE, "--json", *refs, capture=True)
        local = _dvc(root, "status", "--json", *refs, capture=True)
        if json.loads(cloud) or json.loads(local):
            raise RuntimeError(
                f"DVC workspace or local-storage disagrees with refs: {root}"
            )


def register_artifact(path: Path) -> None:
    """Bootstrap one materialized file or directory and push it to local storage."""
    if not path.exists() or path.is_symlink():
        raise RuntimeError(f"DVC registration requires a materialized artifact: {path}")
    root = path.parent
    cache_dir, remote_dir = _storage_paths()
    for directory in (cache_dir, remote_dir):
        if path == directory or path in directory.parents or directory in path.parents:
            raise RuntimeError(
                f"DVC storage overlaps with the tracked artifact: {path}"
            )
    if not (root / ".dvc/config").is_file():
        _dvc(root, "init", "--no-scm")
    configure_local_storage(root)
    # Same-filesystem copies can consume less, but DVC's default is portable
    # copy/reflink. Budget both stores up front.
    size = (
        path.stat().st_size
        if path.is_file()
        else sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    )
    for directory in (cache_dir, remote_dir):
        if shutil.disk_usage(directory).free < size:
            raise RuntimeError(f"Insufficient space to register {path} in {directory}")
    _dvc(root, "add", path.name)
    _dvc(root, "push", "-r", REMOTE, f"{path.name}.dvc")


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, nargs="?", default=Path.cwd())
    args = parser.parse_args()
    configure_local_storage(args.workspace.resolve())


if __name__ == "__main__":
    main()
