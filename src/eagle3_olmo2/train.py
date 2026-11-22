"""Feed live OLMo-2 hidden states to the native SpecForge EAGLE-3 trainer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml

from eagle3_olmo2.config import validate_specforge
from eagle3_olmo2.data import sha256_file
from eagle3_olmo2.train_data import TRAIN_INPUT_DIGEST, TRAIN_INPUT_MANIFEST
from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.feature_store import LocalFeatureStore


class TokenStore(LocalFeatureStore):
    """Read token rows from Arrow through SpecForge's local feature loader."""

    get_returns_fresh_tensors = True

    def __init__(self, run_id, dataset):
        super().__init__(run_id)
        self.dataset = dataset

    def get(self, sample_ref, *, device="cpu", names=None):
        row = self.dataset[sample_ref.metadata["row"]]
        tensors = {name: row[name].to(device) for name in (names or row)}
        return tensors, self._register_file_lease(sample_ref)


class LiveOlmo2Batch:
    """Fill the trainer's usual hidden-state tensors with a frozen forward pass."""

    def __init__(self, backbone, layers, collate, device):
        self.backbone = backbone.eval().requires_grad_(False)
        self.layers = layers
        self.collate = collate
        self.device = device

    @torch.no_grad()
    def __call__(self, rows):
        examples = []
        for row in rows:
            ids = row["input_ids"].reshape(1, -1)
            loss_mask = row["loss_mask"].reshape(1, -1).clone()
            loss_mask[0, -1] = 0
            examples.append(
                {
                    "input_ids": ids,
                    "loss_mask": loss_mask,
                    "attention_mask": torch.ones_like(ids),
                }
            )
        batch = {
            name: tensor.to(self.device)
            for name, tensor in self.collate(examples).items()
            if tensor is not None
        }
        captured = {}

        def capture(module, _inputs, output):
            captured[module] = output

        modules = [self.backbone.layers[index] for index in self.layers]
        handles = [module.register_forward_hook(capture) for module in modules]
        try:
            target = self.backbone(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()
        padding = batch["attention_mask"].eq(0).unsqueeze(-1)
        batch["hidden_state"] = torch.cat(
            [captured[module] for module in modules], dim=-1
        ).masked_fill_(padding, 0)
        batch["target"] = target.masked_fill_(padding, 0)
        return batch


def build_live_trainer(cfg, algorithm):
    from accelerate import cpu_offload
    from datasets import load_from_disk
    from transformers import AutoModel

    from specforge.launch import (
        _assemble_trainer,
        _make_offline_eval_data_factory,
        _shard_offline_refs,
    )
    from specforge.runtime.control_plane import DataFlowController
    from specforge.runtime.control_plane.metadata_store import NoOpMetadataStore
    from specforge.training.assembly import (
        _close_configured_logger,
        _common_launch_kwargs,
        _configured_logger,
        _dataloader_num_workers,
        build_model_bundle,
    )
    from specforge.utils import get_local_device

    training = cfg.training
    if not cfg.data.tokenized_train_path or not cfg.model.vocab_mapping_path:
        raise ValueError(
            "Start training through the project CLI to prepare token inputs"
        )
    token_path = Path(cfg.data.tokenized_train_path)
    dataset = load_from_disk(str(token_path)).with_format("torch")
    provider = algorithm.providers.offline_for(cfg.model.input_modality)
    device = get_local_device()
    backbone = AutoModel.from_pretrained(
        cfg.model.target_model_path,
        dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation="sdpa",
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if backbone.config.model_type != "olmo2":
        raise ValueError("The live capture adapter requires an OLMo-2 backbone")
    collate = LiveOlmo2Batch(
        backbone,
        cfg.model.aux_hidden_state_layer_ids,
        provider.build_collator(),
        device,
    )
    if cfg.model.train_target_cpu_offload:
        cpu_offload(backbone, execution_device=device)
    else:
        backbone.to(device)
    refs = [
        SampleRef(
            sample_id=f"{cfg.run_id}:{index:08d}",
            run_id=cfg.run_id,
            source_task_id=None,
            feature_store_uri=f"tokens://{index}",
            feature_keys={},
            feature_specs={},
            strategy=algorithm.name,
            metadata={
                "row": index,
                "target_repr": "hidden_state",
                "ttt_length": training.ttt_length,
            },
        )
        for index in range(len(dataset))
    ]

    def refs_for_epoch(epoch):
        return _shard_offline_refs(
            refs, use_usp_preprocess=False, seed=training.seed, epoch=epoch
        )

    eval_factory = _make_offline_eval_data_factory(
        algorithm=algorithm,
        modality=cfg.model.input_modality,
        hidden_states_path=cfg.data.eval_hidden_states_path,
        run_id=cfg.run_id,
        batch_size=training.batch_size,
        max_len=cfg.data.max_length,
        ttt_length=training.ttt_length,
        use_usp_preprocess=False,
        dataloader_num_workers=_dataloader_num_workers(cfg, algorithm),
    )
    bundle = build_model_bundle(cfg, algorithm=algorithm)
    logger = _configured_logger(cfg)
    options = _common_launch_kwargs(cfg, bundle, algorithm, logger=logger)
    options.pop("modality")
    # Run target forward synchronously on the trainer's CUDA stream.
    options["dataloader_num_workers"] = 0
    try:
        return _assemble_trainer(
            **options,
            controller=DataFlowController(
                cfg.run_id,
                metadata_store=NoOpMetadataStore(),
                enable_sample_queue=False,
            ),
            store=TokenStore(cfg.run_id, dataset),
            ref_source={"refs": refs_for_epoch(0), "refs_for_epoch": refs_for_epoch},
            model=bundle.model,
            target_head=bundle.target_head,
            num_epochs=training.num_epochs,
            eval_data_factory=eval_factory,
            collate_fn=collate,
            durable_ack=False,
            resume_from=training.resume_from,
            checkpoint_extra={
                "offline_sampler_version": 1,
                "sampler_seed": training.seed,
                "source_dataset_size": len(dataset),
                TRAIN_INPUT_DIGEST: sha256_file(
                    token_path.parent / TRAIN_INPUT_MANIFEST
                ),
            },
        )
    except BaseException:
        _close_configured_logger(logger)
        raise


def main() -> None:
    from accelerate.utils import set_seed

    from specforge.application import resolve_run
    from specforge.cli import _bootstrap_single_process_env, _worker_signal_unwind
    from specforge.distributed import destroy_distributed, init_distributed

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = validate_specforge(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    resolved = resolve_run(cfg)
    os.environ["FSDP_SHARDING"] = cfg.training.fsdp_sharding
    set_seed(cfg.training.seed)
    _bootstrap_single_process_env()
    if int(os.environ["WORLD_SIZE"]) != 1:
        raise ValueError("Live EAGLE-3 training requires one process on one GPU")
    init_distributed(timeout=cfg.training.dist_timeout, tp_size=1)
    failed = True
    try:
        with _worker_signal_unwind():
            build_live_trainer(cfg, resolved.algorithm).fit()
        failed = False
    finally:
        destroy_distributed(abort=failed)


if __name__ == "__main__":
    main()
