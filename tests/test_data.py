import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from eagle3_olmo2.data import normalize_messages, prepare_split
from eagle3_olmo2.prepare_features import register_olmo2_template


class NormalizeMessagesTest(unittest.TestCase):
    def test_ultrachat_roles_are_normalized(self):
        actual = normalize_messages(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        )
        self.assertEqual(
            actual,
            [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
            ],
        )

    def test_sharegpt_fields_are_normalized(self):
        self.assertEqual(
            normalize_messages({"conversations": [{"from": "gpt", "value": "answer"}]}),
            [{"role": "assistant", "content": "answer"}],
        )

    def test_rejects_missing_assistant_turn(self):
        with self.assertRaisesRegex(ValueError, "no assistant"):
            normalize_messages({"messages": [{"role": "user", "content": "Hi"}]})

    def test_olmo2_mask_boundaries_match_model_format(self):
        register_olmo2_template()
        from specforge.data.template import TEMPLATE_REGISTRY

        template = TEMPLATE_REGISTRY.get("olmo2")
        self.assertEqual(template.user_header, "<|user|>\n")
        self.assertEqual(template.assistant_header, "<|assistant|>\n")
        self.assertEqual(template.end_of_turn_token, "<|endoftext|>")

    def test_dataset_revision_is_resolved_and_recorded(self):
        observed = {}

        class FakeDataset(list):
            _fingerprint = "fake-fingerprint"

            def shuffle(self, seed):
                observed["seed"] = seed
                return self

            def select(self, indices):
                return FakeDataset(self[index] for index in indices)

        class FakeHubApi:
            def dataset_info(self, name, revision):
                observed["revision_query"] = (name, revision)
                return types.SimpleNamespace(sha="immutable-sha")

        fake_datasets = types.ModuleType("datasets")

        def load_dataset(name, split, revision, cache_dir=None):
            observed["load_revision"] = revision
            observed["cache_dir"] = cache_dir
            return FakeDataset(
                [
                    {
                        "messages": [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"},
                        ]
                    }
                ]
            )

        fake_datasets.load_dataset = load_dataset
        fake_hub = types.ModuleType("huggingface_hub")
        fake_hub.HfApi = FakeHubApi
        with (
            TemporaryDirectory() as directory,
            patch.dict(
                sys.modules, {"datasets": fake_datasets, "huggingface_hub": fake_hub}
            ),
        ):
            output = Path(directory) / "train.jsonl"
            count = prepare_split(
                dataset_name="HuggingFaceH4/ultrachat_200k",
                revision="main",
                split="train_sft",
                output=output,
                seed=7,
                limit=1,
                cache_dir=Path(directory) / "hf-cache",
            )
            manifest = json.loads(
                output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
            )
        self.assertEqual(count, 1)
        self.assertEqual(observed["load_revision"], "immutable-sha")
        self.assertEqual(observed["cache_dir"], str(Path(directory) / "hf-cache"))
        self.assertEqual(manifest["resolved_revision"], "immutable-sha")
        self.assertEqual(manifest["datasets_fingerprint"], "fake-fingerprint")


if __name__ == "__main__":
    unittest.main()


def test_prompt_groups_cover_punctuation_and_rewritten_trailing_context():
    from eagle3_olmo2.data import prompt_id

    prefix = (
        "Please read the summary of the review carefully and identify "
        "the name of the author listed"
    )
    messages = [{"role": "user", "content": f"{prefix}. The report is attached."}]
    variant = [
        {"role": "user", "content": f"{prefix.upper()}! Use the corrected report."}
    ]
    other = [{"role": "user", "content": "Identify the publication date."}]
    assert prompt_id(messages) == prompt_id(variant)
    assert prompt_id(messages) != prompt_id(other)
