"""Stage 2: SFT training (Ray Train + PyTorch on RayJob).

Reads the parquet from stage 1, spins up N Ray Train workers (1 GPU each) to do
SFT, and exports the checkpoint as a HuggingFace directory so that the vLLM in
stages 3/4 can load it directly.

Five easy traps:

1. Ray 2.58 enables Train V2 by default. In V2 ``TorchTrainer.restore`` /
   ``can_restore`` are deprecated; resume goes through ``resume_from_checkpoint``
   together with ``ray.train.get_checkpoint()``.
2. On resume you **must load the weights from the checkpoint**. Reading back only
   the epoch number while leaving the model on the base weights silently
   discards the training you already did — worse than not resuming at all.
3. Under FSDP, ``model.state_dict()`` gives you a shard; you must collect it
   inside a ``FSDP.state_dict_type(FULL_STATE_DICT, rank0_only=True)`` context.
   This is collective communication: every rank has to enter it, you cannot call
   it on rank 0 only.
4. Under FSDP, gradient clipping must use ``model.clip_grad_norm_()``
   (distributed global norm); ``torch.nn.utils.clip_grad_norm_`` clips only this
   rank's shard.
5. ``result.checkpoint`` is the **latest** checkpoint, not the best one; for the
   best one use ``result.get_best_checkpoint(metric, mode)``.

Verified against: Ray 2.58.0 (Train V2 on by default) and the FSDP API of the
torch 2.7 series.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ray
import ray.train
import ray.train.torch
import torch
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

SHARED_DIR = os.environ.get("SHARED_DIR", "/mnt/cluster_storage")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# Points at the symlink by default, but main() resolves it into an immutable
# version right away before handing it to the workers, so that nobody can swap
# the data out by re-running stage 1 while training is in flight.
TOKENIZED_DIR = os.environ.get("TOKENIZED_DIR", f"{SHARED_DIR}/data/tokenized/current")
MODEL_ROOT = os.environ.get("MODEL_ROOT", f"{SHARED_DIR}/models")
TRAIN_STORAGE = os.environ.get("TRAIN_STORAGE", f"{SHARED_DIR}/train")
RUN_NAME = os.environ.get("RUN_NAME", "sft")
RUN_ID = os.environ.get("RUN_ID", time.strftime("%Y%m%d-%H%M%S"))

NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
PARALLEL_STRATEGY = os.environ.get("PARALLEL_STRATEGY", "ddp")  # ddp | fsdp
EPOCHS = int(os.environ.get("EPOCHS", "2"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
LR = float(os.environ.get("LR", "1e-5"))
SHUFFLE_BUFFER = int(os.environ.get("SHUFFLE_BUFFER", "1024"))
SEED = int(os.environ.get("SEED", "42"))

_WRAPPER_PREFIXES = ("module.", "_fsdp_wrapped_module.")
STATE_FILE = "trainer_state.pt"


def _strip_wrapper_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the key prefixes left behind by DDP / FSDP wrapping so from_pretrained can read it back."""
    cleaned = {}
    for key, value in state_dict.items():
        for prefix in _WRAPPER_PREFIXES:
            while key.startswith(prefix):
                key = key[len(prefix):]
        cleaned[key] = value
    return cleaned


def _fsdp_wrap_policy(model: torch.nn.Module) -> Optional[ModuleWrapPolicy]:
    """Shard by transformer layer instead of wrapping the whole model as one flat unit.

    Without an auto_wrap_policy, FSDP has a single root unit and the forward pass
    has to all-gather every parameter at once, so the GPU memory saving is close
    to zero. Use the HF model's own ``_no_split_modules`` to resolve the decoder
    layer class names, so no specific model's class name is hardcoded here.
    """
    names = getattr(model, "_no_split_modules", None)
    if not names:
        return None
    classes = set()
    for module in model.modules():
        if type(module).__name__ in names:
            classes.add(type(module))
    return ModuleWrapPolicy(classes) if classes else None


def gather_full_state_dict(model: torch.nn.Module) -> Dict[str, Any]:
    """Return the complete state_dict on rank 0 under either wrapping (DDP / FSDP).

    The FSDP branch is a collective operation: every rank must call it together,
    otherwise it hangs on the communication. Non-rank-0 returns an empty dict.
    """
    if isinstance(model, FSDP):
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            return _strip_wrapper_prefix(model.state_dict())
    inner = getattr(model, "module", model)
    return _strip_wrapper_prefix(
        {key: value.detach().cpu() for key, value in inner.state_dict().items()}
    )


def gather_optimizer_state(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> Dict[str, Any]:
    """Under FSDP you likewise need the collective API to get the full optimizer state. Every rank must call it."""
    if isinstance(model, FSDP):
        return FSDP.optim_state_dict(model, optimizer)
    return optimizer.state_dict()


def load_optimizer_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    full_state: Dict[str, Any],
) -> None:
    """Re-shard the full optimizer state back onto the current rank. Every rank must call it."""
    if isinstance(model, FSDP):
        sharded = FSDP.optim_state_dict_to_load(model, optimizer, full_state)
        optimizer.load_state_dict(sharded)
    else:
        optimizer.load_state_dict(full_state)


def clip_gradients(model: torch.nn.Module, max_norm: float) -> None:
    """With FSDP sharding you must use its own method to compute the global norm."""
    if isinstance(model, FSDP):
        model.clip_grad_norm_(max_norm)
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def train_func(config: Dict[str, Any]) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    context = ray.train.get_context()
    rank = context.get_world_rank()

    # Resume branch: weights, optimizer state and epoch all have to come from the
    # checkpoint. Restoring only the epoch while the weights stay on the base
    # model silently discards the training results.
    checkpoint = ray.train.get_checkpoint()
    start_epoch = 0
    optimizer_state: Optional[Dict[str, Any]] = None

    if checkpoint is None:
        source = config["base_model"]
        model = AutoModelForCausalLM.from_pretrained(source)
        tokenizer = AutoTokenizer.from_pretrained(source)
    else:
        # The directory from as_directory is only valid inside the context, so
        # the loading has to happen in there.
        with checkpoint.as_directory() as ckpt_dir:
            state_path = os.path.join(ckpt_dir, STATE_FILE)
            if not os.path.exists(state_path):
                # Better to fail than to silently start over from the base weights.
                raise RuntimeError(
                    f"checkpoint is missing {STATE_FILE}, cannot determine the resume point; "
                    "refusing to continue from the base weights, which would silently "
                    "discard training progress"
                )
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            start_epoch = int(state["epoch"]) + 1
            optimizer_state = state.get("optimizer")
            model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
            tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        if rank == 0:
            print(
                f"[train] resumed from checkpoint: weights loaded, "
                f"optimizer state={'yes' if optimizer_state else 'no'}, "
                f"start epoch={start_epoch}"
            )

    # Not using from_pretrained(dtype=...): that parameter has been renamed
    # across transformers versions (torch_dtype -> dtype), an explicit .to() is
    # more robust.
    model = model.to(dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # prepare_model handles device placement + parallel wrapping, replacing
    # hand-written .to(device) and DDP(...). Note that it does model.to(device)
    # first and wraps afterwards, so the model must fit on a single GPU first;
    # fsdp saves the parameter/optimizer memory during compute, not during
    # initialization.
    strategy = config["parallel_strategy"]
    strategy_kwargs: Dict[str, Any] = {}
    if strategy == "fsdp":
        policy = _fsdp_wrap_policy(model)
        if policy is not None:
            strategy_kwargs["auto_wrap_policy"] = policy
        elif rank == 0:
            print("[train] warning: model does not declare _no_split_modules, FSDP will degenerate into a single flat unit")
    model = ray.train.torch.prepare_model(
        model, parallel_strategy=strategy, parallel_strategy_kwargs=strategy_kwargs
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    if optimizer_state is not None:
        # All ranks enter together: the FSDP branch is a collective operation.
        load_optimizer_state(model, optimizer, optimizer_state)

    shard = ray.train.get_dataset_shard("train")

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        steps, running_loss, skipped = 0, 0.0, 0
        # When feeding data with Ray Data there is no DistributedSampler; without
        # an explicit shuffle you train in write order: samples within a batch are
        # highly similar and the gradient direction gets skewed by the
        # intra-batch correlation. The seed varies with the epoch, so every round
        # has a different order while staying reproducible.
        for batch in shard.iter_torch_batches(
            batch_size=config["batch_size"],
            local_shuffle_buffer_size=config["shuffle_buffer"],
            local_shuffle_seed=config["seed"] + epoch,
        ):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            # When a whole batch has no supervised token, cross entropy is
            # 0/0 = nan. Stage 1 already filters out those samples; this is the
            # second line of defense: better to skip than to let nan into the
            # parameters.
            if not torch.isfinite(output.loss):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue
            output.loss.backward()
            clip_gradients(model, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += output.loss.item()
            steps += 1

        if steps == 0:
            raise RuntimeError(
                f"epoch {epoch} had no valid step at all ({skipped} batches skipped), "
                "check whether the labels from stage 1 are entirely masked"
            )
        mean_loss = running_loss / steps

        # Both gathers are collective operations and must live outside the rank check.
        full_state = gather_full_state_dict(model)
        full_optimizer = gather_optimizer_state(model, optimizer)

        with tempfile.TemporaryDirectory() as staging:
            reported = None
            if rank == 0:
                inner = getattr(model, "module", model)
                # Save in HF directory format: config.json + safetensors +
                # tokenizer, so vLLM can use this directory directly as
                # model_source.
                inner.save_pretrained(
                    staging, state_dict=full_state, safe_serialization=True
                )
                tokenizer.save_pretrained(staging)
                torch.save(
                    {
                        "epoch": epoch,
                        "loss": mean_loss,
                        "optimizer": full_optimizer,
                    },
                    os.path.join(staging, STATE_FILE),
                )
                reported = ray.train.Checkpoint.from_directory(staging)
            ray.train.report(
                {"epoch": epoch, "loss": mean_loss, "skipped_batches": skipped},
                checkpoint=reported,
            )

        if rank == 0:
            print(
                f"[train] epoch {epoch} loss={mean_loss:.4f} "
                f"steps={steps} skipped={skipped}"
            )


def resolve_data_version(path: str) -> Path:
    """Resolve tokenized/current into an immutable version directory.

    The whole training job resolves it exactly once; from then on every worker
    uses the resolved absolute path — so re-running stage 1 and flipping the
    symlink does not affect training that is already in progress.
    """
    resolved = Path(path).resolve(strict=True)
    if not any(resolved.glob("*.parquet")):
        raise SystemExit(f"no parquet under {resolved}, run stage 1 first")
    return resolved


def publish(export_dir: Path, root: Path) -> None:
    """Atomically point models/sft-current at this export, keeping older versions for rollback.

    Note that the symlink is just a convenience entry point for humans and for
    stage 3. Stage 4's serveConfigV2 should reference the immutable version
    number, otherwise changing the symlink will not make Serve reload the weights.
    """
    link = root / "sft-current"
    staging = root / f".sft-current-{RUN_ID}"
    staging.symlink_to(export_dir.name)
    os.replace(staging, link)
    print(f"[train] {link} -> {export_dir.name}")


def main() -> None:
    ray.init(address="auto")

    data_version = resolve_data_version(TOKENIZED_DIR)
    dataset = ray.data.read_parquet(str(data_version))
    row_count = dataset.count()
    print(f"[train] {row_count} training rows, data version {data_version}")

    trainer = TorchTrainer(
        train_func,
        train_loop_config={
            "base_model": BASE_MODEL,
            "parallel_strategy": PARALLEL_STRATEGY,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "shuffle_buffer": SHUFFLE_BUFFER,
            "seed": SEED,
        },
        scaling_config=ScalingConfig(num_workers=NUM_WORKERS, use_gpu=True),
        run_config=RunConfig(
            storage_path=TRAIN_STORAGE,
            name=f"{RUN_NAME}-{RUN_ID}",
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
            ),
        ),
        datasets={"train": dataset},
    )
    result = trainer.fit()

    if result.error is not None:
        raise RuntimeError(f"training failed: {result.error}")

    # result.checkpoint is the latest one, not the best one. Since CheckpointConfig
    # already scores by loss, take the best one explicitly; fall back to the
    # latest if that is unavailable.
    best = result.get_best_checkpoint("loss", "min")
    selection = "best-by-loss"
    if best is None:
        best = result.checkpoint
        selection = "latest"
    if best is None:
        raise RuntimeError("training finished but there is no checkpoint, check the ray.train.report call")

    print(f"[train] export strategy={selection}, metrics={result.metrics}")

    root = Path(MODEL_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    export_dir = root / f"sft-{RUN_ID}"
    # Refuse to overwrite an existing version: re-running with the same RUN_ID
    # should be an explicit act, it must not quietly overwrite a directory that
    # may be in the middle of being loaded by a service.
    if export_dir.exists():
        raise SystemExit(f"{export_dir} already exists, pick a different RUN_ID instead of overwriting a published version")
    best.to_directory(str(export_dir))

    # Provenance: which data, which base model and which config produced these
    # weights. Stage 3's evaluation report and stage 4's rollout can both
    # reference it.
    provenance = {
        "run_id": RUN_ID,
        "base_model": BASE_MODEL,
        "data_version": str(data_version),
        "data_rows": row_count,
        "num_workers": NUM_WORKERS,
        "parallel_strategy": PARALLEL_STRATEGY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "seed": SEED,
        "checkpoint_selection": selection,
        "metrics": result.metrics,
    }
    (export_dir / "pipeline_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    publish(export_dir, root)
    print(f"[train] weights exported: {export_dir}")


if __name__ == "__main__":
    main()