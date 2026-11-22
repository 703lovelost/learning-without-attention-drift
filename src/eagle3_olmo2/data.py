"""Download and normalize UltraChat SFT splits for native SpecForge capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prompt_id(messages: list[dict[str, str]]) -> str:
    """Group examples with the same first user turn across partitions."""
    prompt = next(
        (message["content"] for message in messages if message["role"] == "user"),
        messages[0]["content"],
    )
    # Fold punctuation and case, and keep the first 16 words of long prompts
    # together to screen near-duplicates that add or rewrite trailing context.
    words = re.findall(r"\w+", prompt.casefold())
    return hashlib.sha256(" ".join(words[:16]).encode()).hexdigest()


def train_prompt_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as stream:
        return {prompt_id(json.loads(line)["conversations"]) for line in stream}


def prepare_split(
    *,
    dataset_name: str,
    revision: str,
    split: str,
    output: Path,
    seed: int,
    limit: int | None,
    cache_dir: Path | None = None,
    partition: str | None = None,
    exclude_path: Path | None = None,
    parquet_pattern: str | None = None,
) -> int:
    from datasets import load_dataset
    from huggingface_hub import HfApi

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    selection = {
        "dataset": dataset_name,
        "requested_revision": revision,
        "split": split,
        "seed": seed,
        "requested_samples": limit,
        "partition": partition,
        "parquet_pattern": parquet_pattern,
        "partition_policy": "sha256-normalized-first-user-prefix16-v1",
        "excluded_train_sha256": sha256_file(exclude_path) if exclude_path else None,
    }
    if output.exists() and manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            all(previous.get(key) == value for key, value in selection.items())
            and len(revision) == 40
            and previous.get("sha256") == sha256_file(output)
        ):
            return previous["rows_written"]
        raise ValueError(
            f"Existing data does not match this selection: {output}; choose a new data path"
        )

    resolved_revision = revision
    if len(revision) != 40:
        resolved_revision = HfApi().dataset_info(dataset_name, revision=revision).sha
    files = {split: parquet_pattern.format(split=split)} if parquet_pattern else None
    dataset = load_dataset(
        dataset_name,
        split=split,
        revision=resolved_revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        **({"data_files": files} if files else {}),
    ).shuffle(seed=seed)
    excluded = train_prompt_ids(exclude_path) if exclude_path else set()
    seen = set()
    rejected = {"invalid": 0, "excluded_train_prompt": 0, "duplicate": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with NamedTemporaryFile(
        mode="w", dir=output.parent, encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            for row in dataset:
                try:
                    conversations = normalize_messages(row)
                except ValueError:
                    rejected["invalid"] += 1
                    continue
                group = prompt_id(conversations)
                if partition and (int(group[:8], 16) % 2 == 0) != (partition == "eval"):
                    continue
                if group in excluded:
                    rejected["excluded_train_prompt"] += 1
                    continue
                canonical = json.dumps(
                    conversations, ensure_ascii=False, sort_keys=True
                )
                record_id = hashlib.sha256(canonical.encode()).hexdigest()
                if record_id in seen:
                    rejected["duplicate"] += 1
                    continue
                seen.add(record_id)
                stream.write(
                    json.dumps(
                        {"id": record_id, "conversations": conversations},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
                if written == limit:
                    break
            stream.flush()
            if written == 0:
                raise RuntimeError(f"dataset split {split!r} produced no valid rows")
            digest = sha256_file(temporary)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {
        **selection,
        "resolved_revision": resolved_revision,
        "sha256": digest,
        "rows_written": written,
        "rejected": rejected,
        "datasets_fingerprint": getattr(dataset, "_fingerprint", None),
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/datasets/ultrachat_200k.yaml"
    )
    parser.add_argument(
        "--split", choices=("train", "eval", "test", "all"), default="all"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    ensure_configured_storage_dirs()
    with args.config.open(encoding="utf-8") as stream:
        cfg = expand_env(yaml.safe_load(stream)["dataset"])
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    splits = ("train", "eval", "test") if "test_data_path" in cfg else ("train", "eval")
    selected = splits if args.split == "all" else (args.split,)
    for name in selected:
        split = cfg.get(f"{name}_split", cfg["eval_split"])
        out = resolve_path(cfg[f"{name}_data_path"])
        cache_dir = resolve_path(cfg["cache_dir"])
        configured_limit = cfg[f"{name}_samples"]
        partition = name if "test_data_path" in cfg and name != "train" else None
        count = prepare_split(
            dataset_name=cfg["name"],
            revision=cfg.get("revision", "main"),
            split=split,
            output=out,
            cache_dir=cache_dir,
            seed=cfg["seed"],
            limit=(
                min(args.limit, configured_limit or args.limit)
                if args.limit is not None
                else configured_limit
            ),
            partition=partition,
            parquet_pattern=cfg.get("parquet_pattern"),
            exclude_path=resolve_path(cfg["train_data_path"]) if partition else None,
        )
        print(f"Wrote {count} records from {split} to {out}")


if __name__ == "__main__":
    main()
