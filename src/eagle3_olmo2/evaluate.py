"""Evaluate one checkpoint using the pinned SpecForge offline evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from eagle3_olmo2.config import validate_specforge
from eagle3_olmo2.metrics import canonical_metrics
from eagle3_olmo2.readiness import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    from specforge.application import resolve_run
    from specforge.cli import _bootstrap_single_process_env
    from specforge.distributed import destroy_distributed, init_distributed
    from specforge.eval import Evaluator
    from specforge.launch import _make_offline_eval_data_factory
    from specforge.training.assembly import build_model_bundle

    cfg = validate_specforge(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    cfg.training.resume_from = None
    cfg.model.draft_checkpoint_path = str(args.checkpoint)
    resolved = resolve_run(cfg)
    cfg, algorithm = resolved.config, resolved.algorithm
    _bootstrap_single_process_env()
    init_distributed(timeout=cfg.training.dist_timeout, tp_size=1)
    try:
        bundle = build_model_bundle(cfg, algorithm=algorithm)
        bundle.model.eval()
        strategy = algorithm.providers.step.build(
            bundle.model,
            target_head=bundle.target_head,
            **bundle.strategy_kwargs.options,
        )
        loader = _make_offline_eval_data_factory(
            algorithm=algorithm,
            modality=cfg.model.input_modality,
            hidden_states_path=cfg.data.eval_hidden_states_path,
            run_id=cfg.run_id,
            batch_size=cfg.training.batch_size,
            max_len=cfg.data.max_length,
            ttt_length=cfg.training.ttt_length,
            use_usp_preprocess=False,
            dataloader_num_workers=cfg.data.dataloader_num_workers,
        )()
        try:
            metrics = canonical_metrics(Evaluator().run(strategy.forward_loss, loader))
        finally:
            loader.close()
        if "eval/loss" not in metrics:
            raise RuntimeError("Offline evaluation produced no loss")
        write_json(
            args.output,
            {
                "checkpoint": str(args.checkpoint),
                "features": cfg.data.eval_hidden_states_path,
                "metrics": metrics,
            },
        )
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
