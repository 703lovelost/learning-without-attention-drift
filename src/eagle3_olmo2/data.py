"""Download and normalize UltraChat SFT splits for native SpecForge capture."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from eagle3_olmo2.paths import ensure_configured_storage_dirs, expand_env, resolve_path

ROOT = Path(__file__).resolve().parents[2]


def normalize_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages") or row.get("conversations")
    if not isinstance(messages, list):
        raise ValueError("UltraChat record has no messages list")
    normalized = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role", message.get("from"))
        content = message.get("content", message.get("value"))
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            continue
        if content is None:
            continue
        normalized.append({"role": role, "content": str(content)})
    if not any(message["role"] == "assistant" for message in normalized):
        raise ValueError("conversation has no assistant response")
    return normalized


def prepare_split(
    *,
    dataset_name: str,
    revision: str,
    split: str,
    output: Path,
    seed: int,
    limit: int | None,
    cache_dir: Path | None = None,
) -> int:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    resolved_revision = HfApi().dataset_info(dataset_name, revision=revision).sha
    dataset = load_dataset(
        dataset_name,
        split=split,
        revision=resolved_revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    dataset = dataset.shuffle(seed=seed)
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as stream:
        for index, row in enumerate(dataset):
            try:
                conversations = normalize_messages(row)
            except ValueError as exc:
                raise ValueError(f"invalid record {index} in {split}: {exc}") from exc
            stream.write(
                json.dumps({"conversations": conversations}, ensure_ascii=False) + "\n"
            )
            written += 1
    if written == 0:
        raise RuntimeError(f"dataset split {split!r} produced no valid rows")
    manifest = {
        "dataset": dataset_name,
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "split": split,
        "seed": seed,
        "requested_samples": limit,
        "rows_written": written,
        "datasets_fingerprint": getattr(dataset, "_fingerprint", None),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/datasets/ultrachat_200k.yaml"
    )
    parser.add_argument("--split", choices=("train", "eval", "all"), default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    ensure_configured_storage_dirs()
    with args.config.open(encoding="utf-8") as stream:
        cfg = expand_env(yaml.safe_load(stream)["dataset"])
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    selected = ("train", "eval") if args.split == "all" else (args.split,)
    for name in selected:
        split = cfg["train_split"] if name == "train" else cfg["eval_split"]
        out = resolve_path(
            cfg["train_data_path" if name == "train" else "eval_data_path"]
        )
        cache_dir = resolve_path(cfg["cache_dir"])
        configured_limit = cfg["train_samples" if name == "train" else "eval_samples"]
        count = prepare_split(
            dataset_name=cfg["name"],
            revision=cfg.get("revision", "main"),
            split=split,
            output=out,
            cache_dir=cache_dir,
            seed=cfg["seed"],
            limit=args.limit if args.limit is not None else configured_limit,
        )
        print(f"Wrote {count} records from {split} to {out}")


if __name__ == "__main__":
    main()
