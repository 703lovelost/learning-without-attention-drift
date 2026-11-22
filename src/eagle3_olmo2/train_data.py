"""Prepare token IDs, loss masks, and the vocabulary map for live training."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path

from eagle3_olmo2.data import sha256_file
from eagle3_olmo2.paths import resolve_path
from eagle3_olmo2.prepare_features import eligible_examples, register_olmo2_template
from eagle3_olmo2.readiness import write_json

TRAIN_INPUT_MANIFEST = "train-input-manifest.json"
TRAIN_INPUT_DIGEST = "train_input_manifest_sha256"


def prepare_training_inputs(
    config: dict, source: Path, tokenizer_path: str, identity: dict
) -> Path:
    import torch
    from datasets import Dataset
    from transformers import AutoConfig

    from specforge.data.preprocessing import (
        build_eagle3_dataset,
        process_token_dict_to_mappings,
    )
    from specforge.utils import load_tokenizer, safe_conversations_generator

    model = config["model"]
    data = config["data"]
    identity = {
        **identity,
        "format": "olmo2-live-eagle3-v1",
        "torch_dtype": model["torch_dtype"],
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = resolve_path(data["cache_dir"]) / "train-inputs"
    cache.mkdir(parents=True, exist_ok=True)
    prepared = cache / key
    if not prepared.exists():
        register_olmo2_template()
        tokenizer = load_tokenizer(
            tokenizer_path, trust_remote_code=model["trust_remote_code"]
        )
        target = AutoConfig.from_pretrained(
            model["target_model_path"], trust_remote_code=model["trust_remote_code"]
        ).get_text_config()
        if len(tokenizer) > target.vocab_size:
            raise ValueError("Training tokenizer exceeds the OLMo-2 vocabulary")
        if tokenizer_path != model["target_model_path"]:
            target_tokenizer = load_tokenizer(
                model["target_model_path"], trust_remote_code=model["trust_remote_code"]
            )
            if (
                tokenizer.get_vocab() != target_tokenizer.get_vocab()
                or tokenizer.chat_template != target_tokenizer.chat_template
            ):
                raise ValueError("Training tokenizer must match the OLMo-2 tokenizer")
        draft = json.loads(
            Path(model["draft_model_config"]).read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory(prefix=".train-inputs-", dir=cache) as work:
            working = Path(work)
            rows = Dataset.from_generator(
                safe_conversations_generator,
                gen_kwargs={"file_path": str(source)},
                cache_dir=str(working / "raw"),
            )
            tokens = build_eagle3_dataset(
                dataset=rows,
                tokenizer=tokenizer,
                chat_template=data["chat_template"],
                max_length=data["max_length"],
                shuffle_seed=identity["dataset_seed"],
                num_proc=None,
                cache_dir=str(working / "processed"),
                cache_key=key,
            )
            tokens, statistics = eligible_examples(tokens, data["max_length"])
            tokens = tokens.select_columns(["input_ids", "loss_mask"])
            counts = Counter()
            for row in tokens:
                supervised = row["input_ids"][row["loss_mask"] == 1]
                ids, frequencies = supervised.unique(return_counts=True)
                counts.update(dict(zip(ids.tolist(), frequencies.tolist())))
            d2t, t2d = process_token_dict_to_mappings(
                counts, draft["draft_vocab_size"], target.vocab_size
            )
            result = working / "result"
            tokens.save_to_disk(str(result / "tokens"))
            mapping = result / "vocab_mapping.pt"
            with mapping.open("wb") as stream:
                torch.save({"d2t": d2t, "t2d": t2d}, stream)
            write_json(
                result / TRAIN_INPUT_MANIFEST,
                {
                    **identity,
                    **statistics,
                    "vocab_mapping_sha256": sha256_file(mapping),
                },
            )
            try:
                result.rename(prepared)
            except OSError:
                if not prepared.is_dir():
                    raise

    manifest = json.loads((prepared / TRAIN_INPUT_MANIFEST).read_text(encoding="utf-8"))
    mapping = prepared / "vocab_mapping.pt"
    if (
        any(manifest.get(key) != value for key, value in identity.items())
        or manifest["vocab_mapping_sha256"] != sha256_file(mapping)
        or not (prepared / "tokens/dataset_info.json").is_file()
    ):
        raise ValueError(f"Training inputs disagree with their provenance: {prepared}")
    config["data"]["tokenized_train_path"] = str(prepared / "tokens")
    config["model"]["vocab_mapping_path"] = str(mapping)
    return prepared / TRAIN_INPUT_MANIFEST
