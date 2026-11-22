"""Register OLMo-2's message delimiters and invoke native SpecForge capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from eagle3_olmo2.data import sha256_file
from eagle3_olmo2.paths import ensure_configured_storage_dirs, resolve_path
from eagle3_olmo2.readiness import audit_features, write_json

ROOT = Path(__file__).resolve().parents[2]
SPEC_FORGE = ROOT / "specforge"


def _limit_capture_memory(native_args: list[str], torch) -> dict[str, int | float]:
    """Cap SGLang's static allocation using current free VRAM and target size."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "OLMo-2 target feature extraction requires CUDA, but no CUDA "
            "device is available."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    target_parser = argparse.ArgumentParser(add_help=False)
    target_parser.add_argument("--target-model-path", required=True)
    fraction_parser = argparse.ArgumentParser(add_help=False)
    fraction_parser.add_argument(
        "--sglang-mem-fraction-static", type=float, default=0.4
    )
    target_args, _ = target_parser.parse_known_args(native_args)
    fraction_args, _ = fraction_parser.parse_known_args(native_args)

    target_config_path = Path(target_args.target_model_path) / "config.json"
    config = json.loads(target_config_path.read_text(encoding="utf-8"))
    text_config = config.get("text_config", config)
    hidden = int(text_config["hidden_size"])
    intermediate = int(text_config["intermediate_size"])
    layers = int(text_config["num_hidden_layers"])
    vocab = int(text_config["vocab_size"])
    parameters = 2 * vocab * hidden + layers * (
        4 * hidden * hidden + 3 * hidden * intermediate
    )
    dtype_bytes = (
        4
        if text_config.get("torch_dtype", config.get("torch_dtype")) == "float32"
        else 2
    )
    estimated_weights = int(parameters * dtype_bytes * 1.2)
    reserve_bytes = max(1 << 30, int(total_bytes * 0.1))
    available_for_static = free_bytes - estimated_weights - reserve_bytes
    safe_fraction = max(0.0, available_for_static / total_bytes)
    effective_fraction = min(fraction_args.sglang_mem_fraction_static, safe_fraction)
    if effective_fraction < 0.15:
        raise RuntimeError(
            "Insufficient free GPU memory for OLMo-2 feature capture: "
            f"{free_bytes / (1 << 30):.2f} GiB free, estimated target weights "
            f"{estimated_weights / (1 << 30):.2f} GiB, reserve "
            f"{reserve_bytes / (1 << 30):.2f} GiB. Close other GPU jobs or "
            "reduce sequence length before retrying."
        )
    found_fraction = False
    for index, token in enumerate(native_args[:-1]):
        if token == "--sglang-mem-fraction-static":
            native_args[index + 1] = str(effective_fraction)
            found_fraction = True
            break
    if not found_fraction:
        native_args.extend(("--sglang-mem-fraction-static", str(effective_fraction)))
    return {
        "available_before_capture_bytes": int(free_bytes),
        "total_gpu_bytes": int(total_bytes),
        "estimated_target_weights_bytes": estimated_weights,
        "sglang_mem_fraction_requested": fraction_args.sglang_mem_fraction_static,
        "sglang_mem_fraction_effective": effective_fraction,
    }


def register_olmo2_template() -> None:
    if str(SPEC_FORGE) not in sys.path:
        sys.path.insert(0, str(SPEC_FORGE))
    from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate

    # This OLMo-2 Instruct revision uses role tokens and endoftext rather than
    # ChatML markers. The tokenizer renders messages via apply_chat_template;
    # this registry entry supplies SpecForge's assistant-span mask boundaries.
    if "olmo2" not in TEMPLATE_REGISTRY.get_all_template_names():
        TEMPLATE_REGISTRY.register(
            "olmo2",
            ChatTemplate(
                assistant_header="<|assistant|>\n",
                user_header="<|user|>\n",
                system_prompt="",
                end_of_turn_token="<|endoftext|>",
                parser_type="general",
            ),
        )


def install_olmo2_capture_adapter() -> None:
    """Patch the pinned SGLang Olmo2 class with SpecForge's capture contract."""
    from eagle3_olmo2.sglang_adapter import install_olmo2_eagle3_capture_adapter

    install_olmo2_eagle3_capture_adapter()


def eligible_examples(dataset, max_length: int):
    """Match EAGLE-3's effective loss mask after the final token is cleared."""
    valid = []
    loss_tokens = 0
    lengths = []
    for index, row in enumerate(dataset.select_columns(["input_ids", "loss_mask"])):
        mask = row["loss_mask"].reshape(-1)
        count = int(mask[:-1].sum())
        if count:
            valid.append(index)
            loss_tokens += count
            lengths.append(row["input_ids"].numel())
    if not valid:
        raise ValueError("No examples with assistant loss tokens after truncation")
    return dataset.select(valid), {
        "preprocessed_rows": len(dataset),
        "rejected_empty_masks": len(dataset) - len(valid),
        "loss_tokens": loss_tokens,
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "rows_at_context_limit": lengths.count(max_length),
        "eligible_rows": len(valid),
    }


def _install_preprocessing_audit(source: Path, seed: int, stats: dict) -> None:
    from specforge.data import preprocessing

    native = preprocessing.build_eagle3_dataset
    source_hash = sha256_file(source)

    def audited_build(*args, **kwargs):
        tokenizer = kwargs["tokenizer"]
        template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
        # Native cache identity uses the source path. Include content and the
        # actual template so stale preprocessing cannot enter the capture.
        kwargs["cache_key"] += f"-{source_hash}-{template_hash}-{seed}"
        kwargs["shuffle_seed"] = seed
        if kwargs.get("num_proc") == 1:
            kwargs["num_proc"] = None
        dataset, statistics = eligible_examples(
            native(*args, **kwargs), kwargs["max_length"]
        )
        stats.update(
            statistics, source_sha256=source_hash, chat_template_sha256=template_hash
        )
        return dataset

    preprocessing.build_eagle3_dataset = audited_build


def main() -> None:
    ensure_configured_storage_dirs()
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--tokenizer-path")
    wrapper.add_argument("--capture-seed", type=int, default=42)
    wrapper.add_argument("--local-rank", "--local_rank", type=int)
    wrapper_args, native_args = wrapper.parse_known_args()
    sys.argv = [sys.argv[0], *native_args]
    register_olmo2_template()
    install_olmo2_capture_adapter()
    if wrapper_args.tokenizer_path:
        import specforge.utils

        native_load_tokenizer = specforge.utils.load_tokenizer
        native_parser = argparse.ArgumentParser(add_help=False)
        native_parser.add_argument("--target-model-path")
        native_args_values, _ = native_parser.parse_known_args(native_args)
        target_config_path = Path(native_args_values.target_model_path) / "config.json"
        expected_vocab_size = None
        if target_config_path.is_file():
            config_data = json.loads(target_config_path.read_text(encoding="utf-8"))
            text_config = config_data.get("text_config", config_data)
            expected_vocab_size = text_config.get("vocab_size")

        def load_configured_tokenizer(_target_model_path, **kwargs):
            tokenizer = native_load_tokenizer(wrapper_args.tokenizer_path, **kwargs)
            if str(Path(wrapper_args.tokenizer_path).resolve()) != str(
                Path(_target_model_path).resolve()
            ):
                target_tokenizer = native_load_tokenizer(_target_model_path, **kwargs)
                if tokenizer.get_vocab() != target_tokenizer.get_vocab():
                    raise ValueError(
                        "configured tokenizer token IDs differ from the target tokenizer"
                    )
                if getattr(tokenizer, "chat_template", None) != getattr(
                    target_tokenizer, "chat_template", None
                ):
                    raise ValueError(
                        "configured tokenizer chat template differs from "
                        "the target tokenizer"
                    )
            actual_vocab_size = len(tokenizer)
            if expected_vocab_size is not None and actual_vocab_size > int(
                expected_vocab_size
            ):
                raise ValueError(
                    "configured tokenizer vocabulary exceeds the target: "
                    f"len(tokenizer)={actual_vocab_size}, "
                    f"target vocab_size={expected_vocab_size}"
                )
            return tokenizer

        specforge.utils.load_tokenizer = load_configured_tokenizer
    script = SPEC_FORGE / "scripts/prepare_hidden_states.py"
    import torch

    def argument(name: str) -> str:
        return native_args[native_args.index(name) + 1]

    stats: dict = {}
    _install_preprocessing_audit(
        Path(argument("--data-path")), wrapper_args.capture_seed, stats
    )
    target_config = json.loads(
        (Path(argument("--target-model-path")) / "config.json").read_text(
            encoding="utf-8"
        )
    )
    memory = _limit_capture_memory(native_args, torch)
    started = time.monotonic()
    sys.argv = [sys.argv[0], *native_args]
    torch.cuda.reset_peak_memory_stats()
    try:
        runpy.run_path(str(script), run_name="__main__")
    except RuntimeError as exc:
        error = str(exc)
        error_lower = error.lower()
        if "out of memory" in error_lower:
            raise RuntimeError(
                "OLMo-2 feature capture ran out of GPU memory. Reduce the "
                "configured capture batch size or sequence length and check "
                "the reported available GPU memory."
            ) from exc
        if "nvcc" in error_lower or "compiler" in error_lower:
            raise RuntimeError(
                "OLMo-2 feature capture could not compile a CUDA/SGLang kernel. "
                "Install the CUDA Toolkit (including nvcc) and a C++ compiler, "
                "then ensure /usr/local/cuda/bin/nvcc exists or configure the "
                "toolchain path expected by SGLang. Original error: "
                f"{error}"
            ) from exc
        if "cuda" in error_lower:
            raise RuntimeError(
                "OLMo-2 feature capture failed in the CUDA/SGLang stage. "
                f"Original error: {error}"
            ) from exc
        raise
    argv = sys.argv[1:]
    try:
        output_path = Path(argv[argv.index("--output-path") + 1])
    except (ValueError, IndexError):
        output_path = resolve_path(
            Path(os.getenv("EAGLE3_OUTPUT_DIR", "outputs/olmo2_1b_eagle3"))
        )
    stats.update(
        audit_features(
            output_path,
            target_config,
            stats["eligible_rows"],
            int(argument("--max-length")),
        ),
        capture_wall_seconds=time.monotonic() - started,
    )
    write_json(output_path / "capture-audit.json", stats)
    memory.update(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    (output_path / "feature-extraction-memory.json").write_text(
        json.dumps(
            {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **memory},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
