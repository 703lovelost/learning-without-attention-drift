from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from eagle3_olmo2.inference import _request, load_prompts

PROMPTS = Path(__file__).resolve().parents[1] / "configs/prompts"


def test_development_and_final_prompt_suites_are_disjoint():
    quick, final = (
        load_prompts(SimpleNamespace(suite=PROMPTS / f"{name}.jsonl"))
        for name in ("quick", "final")
    )
    assert len(quick) == len(final) == 24
    assert len({row["category"] for row in final}) == 12
    assert not {row["id"] for row in quick} & {row["id"] for row in final}
    assert not {str(row["messages"]) for row in quick} & {
        str(row["messages"]) for row in final
    }


def test_runtime_acceptance_uses_draft_counts_and_omits_zero_denominator():
    args = SimpleNamespace(port=30000, suite=PROMPTS / "quick.jsonl")
    prompt = load_prompts(args)[0]
    response = {
        "text": "20",
        "output_ids": [42],
        "meta_info": {
            "spec_num_correct_drafts": 2,
            "spec_num_proposed_drafts": 4,
            "spec_accept_length": 3.0,
        },
    }
    settings = {"context_length": 4096, "request_timeout": 2}
    with patch("eagle3_olmo2.inference._post", return_value=response):
        result = _request(args, "speculative", prompt, [11], settings)
        assert result["metrics"]["generation/acceptance_rate"] == 0.5
        assert result["metrics"]["generation/completion_tokens_per_verify"] == 3.0
        assert result["task_check"] is True
        response["meta_info"].clear()
        result = _request(args, "speculative", prompt, [11], settings)
        assert "generation/acceptance_rate" not in result["metrics"]
        assert "acceptance_rate_unavailable" in result
