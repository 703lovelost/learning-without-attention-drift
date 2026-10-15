"""Register OLMo-2's message delimiters and invoke native SpecForge capture."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC_FORGE = ROOT / "specforge"


def _limit_capture_memory(native_args: list[str], torch) -> dict[str, int | float]:
    """Cap SGLang's static allocation using current free VRAM and target size."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "OLMo-2 target feature extraction requires CUDA, but no CUDA device is available."
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    target_parser = argparse.ArgumentParser(add_help=False)
    target_parser.add_argument("--target-model-path", required=True)
    fraction_parser = argparse.ArgumentParser(add_help=False)
    fraction_parser.add_argument("--sglang-mem-fraction-static", type=float, default=0.4)
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
    dtype_bytes = 4 if text_config.get("torch_dtype", config.get("torch_dtype")) == "float32" else 2
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
        native_args.extend(
            ("--sglang-mem-fraction-static", str(effective_fraction))
        )
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
    from specforge.data.template import ChatTemplate, TEMPLATE_REGISTRY

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


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--tokenizer-path")
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
                        "configured tokenizer chat template differs from the target tokenizer"
                    )
            actual_vocab_size = len(tokenizer)
            if expected_vocab_size is not None and actual_vocab_size > int(expected_vocab_size):
                raise ValueError(
                    "configured tokenizer vocabulary exceeds the target: "
                    f"len(tokenizer)={actual_vocab_size}, target vocab_size={expected_vocab_size}"
                )
            return tokenizer

        specforge.utils.load_tokenizer = load_configured_tokenizer
    script = SPEC_FORGE / "scripts/prepare_hidden_states.py"
    import torch

    memory = _limit_capture_memory(native_args, torch)
    sys.argv = [sys.argv[0], *native_args]
    torch.cuda.reset_peak_memory_stats()
    try:
        runpy.run_path(str(script), run_name="__main__")
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() or "cuda" in str(exc).lower():
            raise RuntimeError(
                "OLMo-2 feature capture failed in the CUDA/SGLang stage. Reduce "
                "the configured capture batch size or sequence length and check "
                "the reported available GPU memory."
            ) from exc
        raise
    argv = sys.argv[1:]
    try:
        output_path = Path(argv[argv.index("--output-path") + 1])
    except (ValueError, IndexError):
        output_path = ROOT / "outputs/olmo2_1b_eagle3"
    output_path.mkdir(parents=True, exist_ok=True)
    memory.update(
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(),
    )
    (output_path / "feature-extraction-memory.json").write_text(
        json.dumps(
            {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **memory},
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
