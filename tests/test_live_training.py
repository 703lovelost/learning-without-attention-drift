import json
from dataclasses import replace

import torch
from datasets import load_from_disk
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import (
    AutoModel,
    Olmo2Config,
    Olmo2ForCausalLM,
    PreTrainedTokenizerFast,
)

from eagle3_olmo2.cli import _target_head_compatible_snapshot
from eagle3_olmo2.train import LiveOlmo2Batch, TokenStore
from eagle3_olmo2.train_data import prepare_training_inputs
from specforge.algorithms.eagle3.data import build_offline_collator
from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane import FeatureDataLoader


def test_token_only_preparation_and_live_batch_match_olmo2(tmp_path):
    snapshot = tmp_path / "target"
    words = [
        "[UNK]",
        "[PAD]",
        "<|user|>",
        "<|assistant|>",
        "<|endoftext|>",
        "ask",
        "answer",
    ]
    tokenizer_model = Tokenizer(
        WordLevel({word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer_model.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_model,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|endoftext|>",
        additional_special_tokens=["<|user|>", "<|assistant|>"],
        chat_template=(
            "{% for msg in messages %}"
            "{{ '<|' + msg.role + '|>\n' + msg.content + '<|endoftext|>' }}"
            "{% endfor %}"
        ),
    )
    tokenizer.save_pretrained(snapshot)
    Olmo2ForCausalLM(
        Olmo2Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            num_hidden_layers=4,
            eos_token_id=4,
            pad_token_id=1,
        )
    ).save_pretrained(snapshot)
    target = _target_head_compatible_snapshot(str(snapshot), tmp_path / "indexed")

    source = tmp_path / "train.jsonl"
    source.write_text(
        "".join(
            json.dumps(
                {
                    "conversations": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "answer answer"},
                    ]
                }
            )
            + "\n"
            for prompt in ("ask", "ask ask ask", "ask " * 20)
        ),
        encoding="utf-8",
    )
    draft = tmp_path / "draft.json"
    draft.write_text('{"draft_vocab_size": 8}', encoding="utf-8")
    config = {
        "model": {
            "target_model_path": target,
            "draft_model_config": str(draft),
            "trust_remote_code": False,
            "torch_dtype": "float32",
        },
        "data": {
            "cache_dir": str(tmp_path / "cache"),
            "chat_template": "olmo2",
            "max_length": 12,
        },
    }
    identity = {"dataset_seed": 42}
    manifest = prepare_training_inputs(config, source, str(snapshot), identity)
    assert prepare_training_inputs(config, source, str(snapshot), identity) == manifest
    assert json.loads(manifest.read_text())["eligible_rows"] == 2
    assert not list(tmp_path.rglob("*.ckpt"))

    tokens = load_from_disk(config["data"]["tokenized_train_path"]).with_format("torch")
    assert set(tokens.column_names) == {"input_ids", "loss_mask"}
    backbone = AutoModel.from_pretrained(target, dtype=torch.float32)
    store = TokenStore("run", tokens)
    ref = SampleRef(
        "first",
        "run",
        None,
        "tokens://0",
        {},
        {},
        "eagle3",
        metadata={"row": 0, "target_repr": "hidden_state"},
    )
    loader = FeatureDataLoader(
        store,
        refs=[
            ref,
            replace(ref, sample_id="second", metadata={**ref.metadata, "row": 1}),
        ],
        batch_size=2,
        collate_fn=LiveOlmo2Batch(
            backbone, [0, 1, 2], build_offline_collator(), torch.device("cpu")
        ),
    )
    batch = next(iter(loader)).tensors
    expected = backbone(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
        output_hidden_states=True,
    )
    real = batch["attention_mask"].bool()
    torch.testing.assert_close(batch["target"][real], expected.last_hidden_state[real])
    torch.testing.assert_close(
        batch["hidden_state"][real],
        torch.cat(expected.hidden_states[1:4], dim=-1)[real],
    )
    assert not batch["target"].requires_grad
    assert not batch["hidden_state"][~real].any()
    for mask, length in zip(batch["loss_mask"], real.sum(dim=1)):
        last = int(length) - 1
        assert not mask[last:].any()
        assert mask.any()
