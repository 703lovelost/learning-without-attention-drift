"""Canonical tracking keys for SpecForge's offline EAGLE-3 objective."""

import re

SCHEMA_VERSION = "eagle3-offline-v1"
METRIC_SCHEMA = {
    "version": SCHEMA_VERSION,
    "objective": "TTT-weighted soft-target cross-entropy, with LK disabled",
    "acceptance_rate": {
        "definition": "Offline probability overlap between draft and teacher",
        "train": "per-depth supervised denominators; "
        "the overall value is the mean of depth ratios",
        "eval": "native evaluator weights the per-batch depth rates by loss-token count",
    },
    "eval/simulated_acc_len": "Native simulated acceptance length across TTT depths",
    "legacy_names": {
        "eval/avg_loss": "eval/loss",
        "expected_acceptance[_depth]": "acceptance_rate[_depth]",
        "kl_loss[_depth]": "loss[_depth] where emitted as an alias",
    },
}


def canonical_metrics(values):
    """Filter aliases at the output seam, leaving native checkpoint scoring intact."""
    return {
        _canonical_name(key): value
        for key, value in values.items()
        if not re.fullmatch(
            r"(?:(?:train|eval)/)?(?:expected_acceptance|kl_loss)(?:_\d+)?", key
        )
        and not (key == "eval/avg_loss" and "eval/loss" in values)
    }


def _canonical_name(key: str) -> str:
    if key == "eval/avg_loss":
        return "eval/loss"
    if key.startswith(("train/", "eval/", "perf/", "position_")):
        return key
    return f"train/{key}"
