# Stage 2: SFT Training — Turning Fixed-Length Tensors into Weights vLLM Can Load Directly

> This article corresponds to [`train_sft.py`](train_sft.py) and [`rayjob-train-sft.yaml`](rayjob-train-sft.yaml).
> Previous article: [Stage 1, data processing](../10-data/data-processing.en.md). Baseline: Ray 2.58.0, KubeRay v1.7.0.

## 1. Where this stage sits in the LLM lifecycle

This is **the first step of post-training: supervised fine-tuning (SFT)**. There is a segment on either side of it, and the boundaries need to be clear:

```
Pretraining          SFT (this stage)        Preference alignment      Deployment
───────────          ────────────────        ────────────────────      ──────────
unlabeled text       instruction-response    human pref. pairs / RM    weights frozen
learn "world         learn "answer format    learn "which answer       forward only
knowledge"           and style"              is better"
1000s of GPUs,       a few GPUs,             a few GPUs,               continuously
months               hours to days           days                      online
```

What SFT does is very plain: **take a base model that can only continue text and teach it to produce an answer in the expected format when it sees a question, and to know when to stop**. It adds little knowledge; mostly it changes behavior. That also explains why SFT needs so little data (a few thousand to a few hundred thousand records is already effective) while pretraining needs TB-scale corpora.

Three common misconceptions to clear up first:

- **SFT does not need a thousand GPUs.** Full-parameter fine-tuning of 7B fits on 8×80GB, and 0.5B fits on a single card. What really needs scale is pretraining.
- **Falling loss does not mean SFT succeeded.** SFT loss drops easily, because answer formats are highly predictable. Effectiveness has to come from evaluation, see stage 3.
- **You do not necessarily need RLHF after SFT.** In many business scenarios SFT plus good data is enough.

**What upstream provides**: the fixed-length parquet produced by stage 1 (`input_ids` / `attention_mask` / `labels`), through the `data/tokenized/current` symlink.
**What downstream needs**: the vLLM in stages 3 and 4 wants to load with `model_source=<some directory>` directly, and **does not want to run any conversion script**.

## 2. What this stage must accomplish

| # | Job | Consequence of skipping it |
| --- | --- | --- |
| 1 | Organize N GPUs into one training group | — |
| 2 | Each worker only gets its own share of the data | Data trained repeatedly, effectively re-weighting it |
| 3 | Pick the right parallel strategy (DDP / FSDP) | Straight OOM if it does not fit on one card |
| 4 | **Checkpoint periodically and be able to resume** | One preemption means starting from scratch |
| 5 | **Export in HF directory format** | You have to maintain a conversion script between training and inference |
| 6 | Versioned publishing | No rollback, and no answer to "which training run produced what is in production" |

Items 3, 4 and 5 are where all the technical substance of this stage lives. Item 5 is especially worth stressing: many teams' pipelines break right here, joined by a hand-written `convert_checkpoint.py`, and then that script becomes the most fragile link in the whole chain.

## 3. Why the code is written this way

### 3.1 Why Ray Train instead of torchrun or Kubeflow Trainer

All three can run distributed PyTorch on K8s; the difference is **who organizes the workers**:

| Approach | Who organizes workers | Data sharding | Good for |
| --- | --- | --- | --- |
| `torchrun` + hand-written Job | You (you handle rank, MASTER_ADDR, restarts) | You write it | You already have mature scripts and do not want another runtime |
| Kubeflow Trainer | A K8s controller (TrainJob + JobSet) | You write it | Training is a standalone workload, not co-stacked with data/inference |
| **Ray Train** (this directory) | Ray (actor scheduling + fault tolerance) | **Ray Data shards automatically** | Data, training and inference share one runtime |

The decisive reason this directory picks Ray Train is the third column: the Ray Dataset from stage 1 can be handed straight to the trainer, and Ray makes sure each worker gets a non-overlapping shard. With the other two approaches you write that sharding logic yourself, and "does each rank really get non-overlapping data" is exactly the kind of thing that is hard to verify and does not raise an error when it is wrong.

Be clear about the cost: **one more runtime**. Ray has its own scheduling, its own failure semantics, its own set of concepts (actor / placement group / logical resources). If your scenario is training only, with no data processing and no online inference, Kubeflow Trainer carries less mental load. Part 7 of this repository's blog series takes the Trainer route, so you can read them side by side.

### 3.2 Why train_loop_config only holds scalars

```python
trainer = TorchTrainer(
    train_func,
    train_loop_config={
        "base_model": BASE_MODEL,
        "parallel_strategy": PARALLEL_STRATEGY,
        "epochs": EPOCHS,
        ...
    },
    ...
)
```

Whatever is in `train_loop_config` gets serialized and shipped to every worker. Putting a model object, a Dataset or a large array in there causes an unnecessary large-object transfer, or fails outright. **The rule is: config goes into config, objects are constructed inside `train_func`.**

That is why `train_func` contains `AutoModelForCausalLM.from_pretrained(...)` — each worker loads its own copy, rather than the driver loading one and shipping it over.

### 3.3 ScalingConfig and the GPU worker count must be equal

```python
scaling_config=ScalingConfig(num_workers=NUM_WORKERS, use_gpu=True)
```

`num_workers` is the **number of training processes**, and `use_gpu=True` makes each process request one GPU. It must equal the total number of GPU workers in the YAML:

```
NUM_WORKERS  ==  workerGroupSpecs[gpu-group].replicas × nvidia.com/gpu per Pod
```

Set too low: GPUs idle.
Set too high: **Ray Train never gets enough workers, the job sits in Initializing, and no error is raised.** What you observe is "the job was submitted but nothing happened", with only a waiting-for-resources message in the log. This is the number one trap of this stage.

Diagnostic command:

```bash
kubectl -n llm-pipeline exec <head-pod> -- ray status
# Check whether the GPU total under Resources equals NUM_WORKERS
```

### 3.4 prepare_model is the single switch for the parallel strategy

```python
model = ray.train.torch.prepare_model(
    model, parallel_strategy=config["parallel_strategy"]  # "ddp" | "fsdp" | None
)
```

That one line replaces this hand-written code:

```python
# What prepare_model does for you
device = ...                                    # get the assigned GPU from the Ray Train context
model = model.to(device)
model = DistributedDataParallel(model, device_ids=[device.index])
```

The benefit is not just a few fewer lines, it is that **switching parallel strategy only takes changing one environment variable** — not a single line of the training loop moves. The difference between DDP and FSDP:

| | DDP | FSDP |
| --- | --- | --- |
| Parameters | A full replica on every card | Sharded across cards |
| GPU memory | (model + optimizer state) × N cards | About 1/N |
| Communication | all-reduce gradients on backward | all-gather parameters on forward/backward + reduce-scatter gradients |
| Communication volume | Low | High |
| Good for | Models that **fit on one card** | Models that do not fit on one card |

**The criterion is simple: if it fits on one card, use DDP.** DDP is faster for 0.5B, because FSDP's extra communication is pure overhead on a small model. Full-parameter fine-tuning of 7B needs roughly `7B × 2 bytes (bf16) × (1 param + 1 grad + 2×4-byte AdamW states) ≈ 100GB+`, which does not fit on a single 80GB card, so you need FSDP to shard parameters, gradients and optimizer state.

**But there is a boundary here that has to be spelled out: `prepare_model` is not a "change one string and train a bigger model" switch.** Read the Ray 2.58.0 implementation (`train_loop_utils.py`):

```python
if move_to_device:
    model = model.to(device)          # ← move the whole thing onto the GPU first
...
if parallel_strategy and world_size > 1:
    ...
    model = DataParallel(model, **parallel_strategy_kwargs)   # ← then wrap with FSDP
```

The order is **`.to(device)` first, wrapping second**. Add to that the fact that every rank in `train_func` does its own `from_pretrained` of a full model, and this path requires that **the model already fits on one card at initialization time**. What FSDP saves here is parameter/gradient/optimizer memory during the "compute phase", not during the "initialization phase".

To really train a model that does not fit on one card, what you need is meta device initialization plus `param_init_fn` (or the sharded loading of FSDP2 / DeepSpeed ZeRO-3), and that is not a matter of changing one environment variable. The applicable range of the `fsdp` branch in this directory is: **the model fits, but once you add parameters + gradients + optimizer state you can no longer train it**.

There is also one default you have to supply yourself, or FSDP is nearly pointless:

```python
strategy_kwargs = {}
if strategy == "fsdp":
    policy = _fsdp_wrap_policy(model)      # shard by transformer layer
    if policy is not None:
        strategy_kwargs["auto_wrap_policy"] = policy
model = ray.train.torch.prepare_model(
    model, parallel_strategy=strategy, parallel_strategy_kwargs=strategy_kwargs
)
```

The `parallel_strategy_kwargs` that `prepare_model` passes by default is an empty dict, which means **no `auto_wrap_policy`**. FSDP then has a single root unit and the forward pass has to all-gather every parameter at once — stored sharded, aggregated in full during compute, so the GPU memory saving is close to zero. `_fsdp_wrap_policy` uses the HF model's own `_no_split_modules` to resolve the decoder layer classes and hands them to `ModuleWrapPolicy`, so each layer is aggregated and released separately, which is where the real benefit comes from. Using `_no_split_modules` instead of hardcoding `Qwen2DecoderLayer` means you do not have to edit code when you change models.

One last easily overlooked fact: `parallel_strategy` only takes effect when `world_size > 1`. With `NUM_WORKERS=1` the model is not wrapped at all, so `isinstance(model, FSDP)` is false and all the FSDP branches below take the ordinary path — that behavior is correct, but do not try to verify FSDP logic on a single card, you cannot.

### 3.5 Why .to(dtype=...) instead of from_pretrained(torch_dtype=...)

```python
model = AutoModelForCausalLM.from_pretrained(config["base_model"])
model = model.to(dtype=torch.bfloat16)
```

It looks like a detour, and the reason is that **the parameter name changed across transformers versions**: older versions call it `torch_dtype`, newer ones renamed it to `dtype` and put `torch_dtype` on a deprecation path. Hardcoding either one will warn or fail on some version. `.to(dtype=...)` is a stable PyTorch API and safe across versions.

The cost is that loading takes fp32 memory once before converting to bf16, so peak memory is about twice the model size. That does not matter for 0.5B; for 70B you must use `from_pretrained`'s dtype parameter (and accept the version coupling), or load with `device_map` / meta device.

Why bf16 and not fp16: bf16 has the same exponent width as fp32, so its dynamic range is sufficient and **no GradScaler is needed**. fp16 in LLM training overflows to NaN easily and needs loss scaling. The cost is that bf16 has fewer mantissa bits, but for training stability that trade is worth it.

### 3.6 Why use_cache and gradient_checkpointing show up together

```python
model.config.use_cache = False
model.gradient_checkpointing_enable()
```

**These two lines depend on each other; you cannot write only one.**

- `use_cache=False`: the KV cache is an **inference** optimization (caching K/V of past tokens to avoid recomputation). During training every step computes the full sequence, so the cache is useless and just consumes GPU memory.
- `gradient_checkpointing_enable()`: trade time for GPU memory — do not save intermediate activations on the forward pass, recompute them on the backward pass. It saves more than half of activation memory, at the cost of roughly 30% extra compute.

The point of conflict is that gradient checkpointing requires the forward pass to be replayable, while `use_cache=True` mutates cache state during the forward pass. If you do not turn the cache off, transformers warns and turns it off itself, but relying on a library's auto-correction is a bad habit — write it explicitly so whoever reads the code knows it is intentional.

The thing turned off here becomes **the entirety of capacity planning** by stage 4: the size of the KV cache decides how much concurrency one card can carry, which in turn decides how you fill in the autoscaling parameters. The formulas and measured numbers are in [section 3.3 of the serving article](../40-serve/online-serving.en.md). **Training turns it off, inference depends on it** — the same mechanism has exactly opposite standing in the two stages.

### 3.7 Why not use prepare_data_loader

Ray Train provides `prepare_data_loader`, which wraps a plain PyTorch DataLoader with a `DistributedSampler`. This directory **does not use it**:

```python
shard = ray.train.get_dataset_shard("train")
for batch in shard.iter_torch_batches(...):
```

Because the data is fed in by Ray Data, sharding is already done by `get_dataset_shard` — each worker gets its own share and does not need a sampler to split it again. Ray's official docs say so explicitly: skip `prepare_data_loader` when feeding Ray Data.

Wrapping another `DistributedSampler` around it causes a second split, so each worker only trains on `1/N²` of the data, and **no error is raised**.

`iter_torch_batches` defaults to `device="auto"`, which inside a Ray Train worker resolves to the GPU that worker was assigned. That is why there is no `.to(device)` anywhere in the code below:

```python
output = model(
    input_ids=batch["input_ids"],       # already on the right GPU
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)
```

### 3.8 Why an explicit shuffle is mandatory

```python
for batch in shard.iter_torch_batches(
    batch_size=config["batch_size"],
    local_shuffle_buffer_size=config["shuffle_buffer"],
    local_shuffle_seed=config["seed"] + epoch,
):
```

**This is the one most easily missed.** On the DataLoader + `DistributedSampler` route, `sampler.set_epoch(epoch)` takes care of reshuffling every round; once you switch to Ray Data there is no sampler, and if you do not shuffle explicitly, the training order is the parquet write order.

The consequence is that samples within a batch are highly similar (stage 1's demo data is even written out contiguously grouped by `repeat`), each batch's gradient direction is skewed by the intra-batch correlation, and training quality gets noticeably worse — while the loss curve just looks "a bit jittery", which is very hard to attribute.

Adding `epoch` to `local_shuffle_seed` makes each round's order different while keeping the whole thing reproducible. `local_shuffle_buffer_size` is the number of rows in the buffer: too small is the same as no shuffle, too large eats memory. This is a **local** shuffle (within the buffer), not a global one; if you need a global shuffle, do `ds.random_shuffle()` in stage 1 before writing.

### 3.9 Why FSDP's state_dict is such a hassle

```python
def gather_full_state_dict(model):
    if isinstance(model, FSDP):
        config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            return _strip_wrapper_prefix(model.state_dict())
    inner = getattr(model, "module", model)
    return _strip_wrapper_prefix(
        {k: v.detach().cpu() for k, v in inner.state_dict().items()}
    )
```

Under FSDP, `model.state_dict()` by default returns **the shard on this card**, not the full weights. Saving that directly gives you a mutilated file that `from_pretrained` cannot read.

You must enter the `FULL_STATE_DICT` context to make FSDP all-gather the shards into complete tensors. Each of the three parameters has its purpose:

- `StateDictType.FULL_STATE_DICT`: ask for the full weights instead of a shard.
- `rank0_only=True`: assemble only on rank 0. Without it every card holds a full copy of the weights, which for 7B on 8 cards is 8×14GB of pointless overhead.
- `offload_to_cpu=True`: put the assembled result in CPU memory. The full weights can be larger than a single card's GPU memory.

**The most critical part is where you call it:**

```python
# All ranks enter together: the FSDP gather is collective communication.
full_state = gather_full_state_dict(model)

with tempfile.TemporaryDirectory() as staging:
    reported = None
    if rank == 0:          # ← only the saving action goes inside the rank check
        ...
```

`gather_full_state_dict` is **outside** the `if rank == 0`. all-gather is a collective operation and requires every participant to call it. Written like this:

```python
if rank == 0:                                  # ✗ wrong
    full_state = gather_full_state_dict(model)
```

rank 0 waits forever for the other ranks to join that communication, while the other ranks skipped it long ago — the job **hangs permanently**, with no error at all. This is one of the hardest FSDP failures to diagnose; the symptom is "training reaches the first checkpoint and stops moving".

`_strip_wrapper_prefix` handles the key prefixes left by the wrappers: DDP adds `module.`, and FSDP in some torch versions adds `_fsdp_wrapped_module.`. Without cleaning them, `from_pretrained` cannot find the corresponding parameters, reports a pile of unexpected keys at load time, and hands you a randomly initialized model — and **by default that is only a warning, not an error**.

**The same "shard vs global" problem shows up again in gradient clipping**, and there it is even more subtle:

```python
def clip_gradients(model, max_norm):
    if isinstance(model, FSDP):
        model.clip_grad_norm_(max_norm)                    # distributed global norm
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

Under FSDP each rank only holds part of the gradients. Calling `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)` directly computes **the norm of this rank's shard**, and each rank independently scales its own part by a different coefficient — that is neither clipping the global norm, nor is the scaling consistent across ranks.

The FSDP instance's own `clip_grad_norm_()` method does a cross-rank all-reduce to get the true global norm and then scales uniformly. It is likewise a collective operation, and every rank has to call it.

This mistake does not crash and does not hang, it just quietly makes training less stable — the kind of problem that falls into "you cannot see it in the loss curve, but the result is just a bit worse".

### 3.10 Why export in HF directory format — this is the key interface of the whole chain

```python
inner = getattr(model, "module", model)
inner.save_pretrained(staging, state_dict=full_state, safe_serialization=True)
tokenizer.save_pretrained(staging)
```

The artifact is a standard HuggingFace model directory:

```
config.json                 model architecture and hyperparameters
model.safetensors           weights
generation_config.json      default generation parameters
tokenizer.json / vocab      tokenizer
```

**This format is the contract between training and inference.** Stage 3's `vLLMEngineProcessorConfig(model_source=<that directory>)` and stage 4's `model_loading_config.model_source` can both consume it directly, with no conversion step in between.

Compare with the common approach — saving only `torch.save(model.state_dict(), "model.pt")` — and you get a bare state_dict that vLLM does not understand, so you need a conversion script to assemble the config, assemble the tokenizer and rename keys. That script becomes the part of the chain most prone to rot: change the model structure and it breaks, and when it breaks it usually manifests as "the model loads but the output is gibberish".

`safe_serialization=True` uses safetensors instead of pickle: faster to load, and it does not execute arbitrary code. `tokenizer.save_pretrained` must be saved alongside — without the tokenizer, vLLM goes looking on the HF Hub, which fails outright in an offline environment and, in a connected one, may get you a different tokenizer version than the one used in training.

`state_dict=full_state` is passed explicitly. Under FSDP the parameters of `inner` are still sharded, and without this argument `save_pretrained` would read the shards.

### 3.11 The calling contract of report

```python
ray.train.report({"epoch": epoch, "loss": mean_loss}, checkpoint=reported)
```

The rule is: **every worker must call `report`, but only rank 0 passes a checkpoint (the rest pass `None`)**.

`report` has synchronization semantics internally, so if one worker does not call it the other workers get stuck. And only one copy of the checkpoint is needed — having every rank upload the full weights is pure waste, and they would overwrite each other.

`ray.train.Checkpoint.from_directory(staging)` is created inside `with tempfile.TemporaryDirectory()`, and `report` is called before leaving the context. Ray Train copies the directory contents to the persistent storage given by `RunConfig.storage_path`, after which the temporary directory can be safely deleted.

### 3.12 Resume: you cannot use restore under Train V2

Ray 2.58 **enables Train V2 by default** (`is_v2_enabled()` returns `True` by default). In V2, `TorchTrainer.restore` and `can_restore` are deprecated. Plenty of examples online, including some official documentation pages, still use this form:

```python
if TorchTrainer.can_restore(path):          # ✗ deprecated under V2
    trainer = TorchTrainer.restore(path, ...)
```

V2's approach separates the two paths:

- **Automatic restart after a worker failure**: Ray hands the most recent checkpoint to the newly started worker via `ray.train.get_checkpoint()`, and `train_func` is responsible for reading it and restoring.
- **Manually resuming a training run**: pass `TorchTrainer(..., resume_from_checkpoint=Checkpoint(path))`.

#### Resume must load the weights; reading back only the epoch number is catastrophic

Here is the **wrong** version first — it genuinely existed in this directory, and it is worse than not resuming at all:

```python
model = AutoModelForCausalLM.from_pretrained(config["base_model"])   # base weights
...
checkpoint = ray.train.get_checkpoint()
if checkpoint is not None:
    with checkpoint.as_directory() as ckpt_dir:
        state = torch.load(os.path.join(ckpt_dir, "trainer_state.pt"))
    start_epoch = int(state["epoch"]) + 1        # ✗ only the epoch jumped, the weights were not replaced
```

The consequence: after a worker restart it "continues from epoch 2", but the model is on the **base weights**. The results of the first two rounds are silently discarded, while the log says resume succeeded, the loss curve keeps going down, and the final checkpoint looks perfectly fine. **Nothing tells you that training actually only ran the second half.**

The correct version makes weights, optimizer state and epoch all come from the checkpoint:

```python
checkpoint = ray.train.get_checkpoint()
if checkpoint is None:
    model = AutoModelForCausalLM.from_pretrained(config["base_model"])
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
else:
    # the directory from as_directory is only valid inside the context, so loading must happen in there
    with checkpoint.as_directory() as ckpt_dir:
        state_path = os.path.join(ckpt_dir, STATE_FILE)
        if not os.path.exists(state_path):
            # better to fail than to silently start over from the base weights
            raise RuntimeError(f"checkpoint is missing {STATE_FILE}, refusing to continue from the base weights")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        start_epoch = int(state["epoch"]) + 1
        optimizer_state = state.get("optimizer")
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir)      # ← weights come from the checkpoint
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
```

Three details:

1. **The loading is written inside the `with`.** `as_directory()` is a context manager, and once you leave the block the directory may be cleaned up.
2. **`from_pretrained(ckpt_dir)` reads the checkpoint directory directly.** Because what we saved is exactly the HF directory format (section 3.10), resume and first-time loading go through the same API and no manual `load_state_dict` is needed. That is another side benefit of "saving in HF format".
3. **Missing `trainer_state.pt` raises, it does not degrade.** Silently degrading to the base weights is precisely the shape of the bug above; better to let the job fail so a human sees it.

#### Optimizer state also needs the collective API under FSDP

```python
def gather_optimizer_state(model, optimizer):
    if isinstance(model, FSDP):
        return FSDP.optim_state_dict(model, optimizer)
    return optimizer.state_dict()

def load_optimizer_state(model, optimizer, full_state):
    if isinstance(model, FSDP):
        optimizer.load_state_dict(
            FSDP.optim_state_dict_to_load(model, optimizer, full_state))
    else:
        optimizer.load_state_dict(full_state)
```

Same as with weights: under FSDP the optimizer state is also sharded, and `optimizer.state_dict()` only gives you this rank's share. `FSDP.optim_state_dict()` aggregates it and `optim_state_dict_to_load()` re-shards it. **Both are collective operations and every rank has to call them** (the same trap as in section 3.9).

The consequence of not restoring the optimizer state is worse than you might think: AdamW's first- and second-moment estimates start accumulating from zero, so the first several steps after resume are effectively running an unwarmed optimizer, and the loss visibly jitters.

**The resume granularity in this directory is still the epoch, not the step.** A failure in the middle of an epoch reruns the whole round, because the data iteration position is not recorded. Step-level resume would also require persisting Ray Data's iteration progress — that is a distinctly larger amount of engineering, and it is not done.

### 3.13 CheckpointConfig and the final export

```python
run_config=RunConfig(
    storage_path=TRAIN_STORAGE,
    name=f"{RUN_NAME}-{RUN_ID}",
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="loss",
        checkpoint_score_order="min",
    ),
)
```

- `storage_path` must be shared storage. Ray Train explicitly requires a shared path for multi-node training — a local path only works on a single node; across nodes the checkpoint written by rank 0 is invisible to the others.
- `num_to_keep=2` plus taking the minimum by loss: only keep the two best-scoring ones, so one per epoch does not fill up the PVC. A 7B checkpoint is in the 14GB range, and 20 epochs of training is 280GB. Note that this scoring only affects **which ones are kept**, not which one is exported (see below).
- `name` carries `RUN_ID`: every training run gets its own directory and they do not overwrite each other.

#### The data version must be pinned at the start of the job

```python
def resolve_data_version(path: str) -> Path:
    resolved = Path(path).resolve(strict=True)      # tokenized/current -> v-<RUN_ID>
    if not any(resolved.glob("*.parquet")):
        raise SystemExit(f"no parquet under {resolved}, run stage 1 first")
    return resolved

data_version = resolve_data_version(TOKENIZED_DIR)
dataset = ray.data.read_parquet(str(data_version))   # use the resolved absolute path
```

`TOKENIZED_DIR` points at the `tokenized/current` symlink by default. **Handing the symlink path straight to Ray Data is unsafe**: Ray Data reads lazily, so file enumeration and the actual read happen at different moments, and if someone re-runs stage 1 and flips the symlink in between, this training run may read data mixed from two versions, or even read a path that has already been cleaned up.

`resolve()` turns it once into the immutable `v-<RUN_ID>` absolute path, and from then on every worker uses the same definite directory. The resolved result is also written into `pipeline_provenance.json`, so "which data did this model use" is on the record.

**This is an important limitation of the "atomic symlink switch" pattern**: atomically replacing a directory entry only guarantees that a reader sees either the complete old link or the complete new link, it **does not guarantee that a series of subsequent open calls constitutes a snapshot**. For snapshot semantics you must resolve once on the consumer side and pin it.

```python
result = trainer.fit()
if result.error is not None:
    raise RuntimeError(f"training failed: {result.error}")
```

**You must check `result.error` explicitly.** `trainer.fit()` does not necessarily raise when training fails; it puts the error into `result.error`. Without the check, a failed training run finishes "successfully", the RayJob status turns SUCCEEDED, and then stage 3 goes to load a model that does not exist or is stale — the failure is deferred to downstream and diagnosis costs twice as much.

#### result.checkpoint is the latest, not the best

This is a spot where it is very easy to configure a "self-contradictory policy":

```python
# CheckpointConfig scores by loss and keeps the best two...
checkpoint_config=CheckpointConfig(num_to_keep=2,
                                   checkpoint_score_attribute="loss",
                                   checkpoint_score_order="min")
...
result.checkpoint.to_directory(export_dir)     # ✗ but this exports the latest one
```

The Ray 2.58 Train V2 `Result` documentation is quite clear: `checkpoint` is **the latest checkpoint**, and there is also a `best_checkpoints` list and `get_best_checkpoint(metric, mode)`. So the version above will export the worse one whenever "the last round happens to be worse" — and the scoring configuration for `num_to_keep` did nothing at all on this path.

The correct version takes it explicitly:

```python
best = result.get_best_checkpoint("loss", "min")
selection = "best-by-loss"
if best is None:                      # fall back to the latest when there is no scored checkpoint
    best = result.checkpoint
    selection = "latest"
if best is None:
    raise RuntimeError("training finished but there is no checkpoint, check the ray.train.report call")
```

**But be clear about how weak this "best" is**: `loss` is rank 0's unweighted average of batch losses on the training set, not a validation-set metric, and it is not weighted by token count. It can catch "the last round clearly went off the rails", it cannot replace evaluation. Real model selection needs holdout metrics, and that is stage 3's job. So the code records the selection strategy in `pipeline_provenance.json`, so downstream knows how this version was picked.

#### Record provenance on export, and refuse to overwrite

```python
export_dir = root / f"sft-{RUN_ID}"
if export_dir.exists():
    raise SystemExit(f"{export_dir} already exists, pick a different RUN_ID instead of overwriting a published version")
best.to_directory(str(export_dir))
(export_dir / "pipeline_provenance.json").write_text(json.dumps({
    "run_id": RUN_ID, "base_model": BASE_MODEL,
    "data_version": str(data_version),      # the immutable path resolved from stage 1
    "checkpoint_selection": selection, "metrics": result.metrics, ...
}))
publish(export_dir, root)      # models/sft-current -> sft-<RUN_ID>
```

The `exists()` check is not redundant: the export directory may be in the middle of being loaded by stage 4's engine, and an overwrite would let a running service read half-new, half-old safetensors. **"Do not overwrite a published version" has to be a raise in the code, not a sentence in the README.**

`pipeline_provenance.json` answers "which data, which base model and which config produced these weights". It lets stage 3's evaluation report and stage 4's rollout be reconciled against each other — without it, "is what we evaluated the thing we shipped" can only be tracked in someone's head.

## 4. Why the YAML is written this way

### 4.1 The head must explicitly declare num-gpus: "0"

```yaml
headGroupSpec:
  rayStartParams:
    num-gpus: "0"
```

`rayStartParams` are the arguments passed to `ray start`, and `num-gpus` tells Ray "how many logical GPUs this node has". The head Pod's container does not request `nvidia.com/gpu`, but if the node has GPUs and Ray auto-detects them, a training actor may get scheduled onto the head — and then fail because the container has no access to the GPU devices, with an error message that usually points at CUDA rather than at scheduling.

Writing `"0"` explicitly closes that path off. **It also shows that Ray's logical resources and K8s's physical resources are two separate books**: Ray only looks at `rayStartParams` and its auto-detection, not at the container's resources declaration. Problems arising from a mismatch between the two are very hard to pin down.

By the same reasoning, stage 4's RayService head also declares `num-gpus: "0"`, for the same reason: inference engine replicas should not land on the head.

### 4.2 /dev/shm is mandatory, not an optimization

```yaml
volumeMounts:
  - name: shared-memory
    mountPath: /dev/shm
volumes:
  - name: shared-memory
    emptyDir:
      medium: Memory
      sizeLimit: 8Gi
```

A container's `/dev/shm` is only **64MB** by default. PyTorch's DataLoader workers, NCCL's intra-node communication, and later vLLM's multiple processes all need shared memory. When 64MB is not enough, what you see is:

- training **freezes** at the first batch or the first all-reduce;
- or a `Bus error (core dumped)`;
- or NCCL reports a timeout error that has nothing to do with shared memory.

**None of the three symptoms mentions `/dev/shm`.** This is the most unreasonable trap of this stage, which is why it is hardcoded into the template.

`medium: Memory` puts the emptyDir on tmpfs (memory) instead of disk. Note that this space **counts against the container's memory limit**: the worker requests 48Gi of memory, `/dev/shm` takes 8Gi, and what is actually available for training is 40Gi. Subtract it when you compute GPU memory and RAM.

### 4.3 Why the GPU workers use Guaranteed QoS

```yaml
resources:
  requests:
    cpu: "8"
    memory: 48Gi
    nvidia.com/gpu: "1"
  limits:
    cpu: "8"
    memory: 48Gi
    nvidia.com/gpu: "1"
```

All three requests equal the limits, so the Pod lands in **Guaranteed** QoS. This is the opposite of stage 1's Burstable, and the reasons are:

- **GPUs cannot be oversubscribed.** `nvidia.com/gpu` is an incompressible integer resource, and request and limit **must be equal** (a hard K8s requirement for extended resources; writing them unequal gets rejected by the API server).
- **Getting a training job OOM-killed is expensive.** Killing a run that has been going for 3 hours costs 3 hours × N cards. Compared with that, the loss in bin-packing efficiency is negligible.
- **CPU is set equal too**, to avoid data-loading jitter from CPU contention — a GPU waiting on data is the most expensive kind of waiting.

One more thing that is easy to overlook: `nvidia.com/gpu: "1"` can only be an integer. Ray supports logically fractional GPUs (`num_gpus=0.5`), but that is only Ray's internal accounting and **provides no GPU memory isolation** — two actors each asking for 0.5 GPU genuinely share all the memory of the same card, and whoever uses more OOMs the other. For real isolation you need MIG or time-slicing, which is node-level configuration.

### 4.4 Why the autoscaler is turned off, and the gang scheduling problem it raises

```yaml
workerGroupSpecs:
  - groupName: gpu-group
    replicas: 2
    minReplicas: 2
    maxReplicas: 2
```

A training group is **all or nothing**: if only 1 of the 2 workers comes up, training does not start (`num_workers=2` never gets the second one). Having the autoscaler add Pods one at a time is pointless.

But pinning the three values only solves the Ray layer, it **does not solve the K8s layer**. The real problem is: when the cluster has only 1 free card left, K8s will schedule one Pod successfully and let it hold that card while waiting for the second Pod. The second Pod stays Pending because no card is available. The result is:

- a card is occupied but doing nothing;
- training never starts;
- and if there are two such training jobs at the same time, they each hold one card and wait for each other forever — **resource deadlock**.

Neither the Ray layer nor the KubeRay layer solves this; it needs **gang scheduling** at the K8s layer: Kueue or Volcano, with the semantics "either all N Pods get scheduled or none of them do".

This directory does not integrate them, because at the 2-card experimental scale avoiding conflicts by hand is enough. **As soon as multiple people share a GPU pool, gang scheduling is a requirement, not an optimization.**

### 4.5 activeDeadlineSeconds is the insurance policy for your GPUs

```yaml
shutdownAfterJobFinishes: true
ttlSecondsAfterFinished: 600
activeDeadlineSeconds: 14400
```

- `activeDeadlineSeconds: 14400` (4 hours): when training hangs (NCCL timeout, deadlock, blocked data read) it does not exit by itself, and the GPUs stay occupied. This ceiling lets KubeRay terminate it. **In a shared GPU cluster this is the single most important field** — without it, one hung job can block the whole pool all night.
- `ttlSecondsAfterFinished: 600`: twice as long as stage 1's 300 seconds, because training produces more logs that are more worth capturing, and a failed training run needs the scene preserved even more.

### 4.6 NCCL_DEBUG and the commented-out scheduling constraints

```yaml
env:
  - name: NCCL_DEBUG
    value: WARN
```

NCCL prints almost nothing by default. Setting it to `WARN` lets you see NIC selection and topology detection warnings when communication goes wrong — NCCL picking the wrong NIC under container networking is a common failure. You can temporarily raise it to `INFO` while diagnosing, but that produces a lot of logs and is not suitable to leave on.

```yaml
# tolerations:
#   - key: nvidia.com/gpu
#     operator: Exists
#     effect: NoSchedule
# nodeSelector:
#   node.kubernetes.io/instance-type: g5.xlarge
```

Commented out rather than deleted, because these two are **almost certainly needed, but their values vary by cluster**. GPU nodes usually carry a taint (to keep ordinary workloads off them), and a Pod without the matching toleration stays Pending forever, with the message in events being `node(s) had untolerated taint` — that message is quite clear, but you only find it if you go look at events.

## 5. How to verify it

```bash
make train        # internally deletes the old RayJob of the same name, applies, then waits via wait_rayjob.sh
```

The waiting logic of `make train` deserves its own note: **you cannot use `kubectl wait --for=condition=Complete rayjob/...`**. KubeRay v1.7.0's `RayJobStatus` has no `Conditions` field at all (only `jobStatus` and `jobDeploymentStatus`), so that condition never appears and `kubectl wait` just waits until it times out — a successful job also "fails to wait". And `jobDeploymentStatus=Complete` is not a success signal either, because `IsJobDeploymentTerminal` returns true for both `Complete` and `Failed`. The real success criterion is `status.jobStatus == SUCCEEDED`, which is exactly the field `wait_rayjob.sh` polls.

```bash
# 1. Job status (the real success signal)
kubectl -n llm-pipeline get rayjob llm-train-sft \
  -o jsonpath='{.status.jobStatus}{"\n"}'          # expect SUCCEEDED

# 2. The GPUs really are being used (run during training)
kubectl -n llm-pipeline exec <gpu-worker-pod> -- nvidia-smi \
  --query-gpu=utilization.gpu,memory.used --format=csv

# 3. Loss is going down, with no large number of skipped batches
kubectl -n llm-pipeline logs -l ray.io/group=gpu-group | grep '\[train\] epoch'
# expect something like: [train] epoch 0 loss=1.8342 steps=12 skipped=0

# 4. The artifact is a valid HF directory and carries provenance
#    (run inside a temporary Pod with the PVC mounted)
ls /mnt/cluster_storage/models/sft-current/
# expect: config.json  model.safetensors  tokenizer.json
#         trainer_state.pt  pipeline_provenance.json
cat /mnt/cluster_storage/models/sft-current/pipeline_provenance.json
```

The criteria, from least to most trustworthy:

1. `jobStatus: SUCCEEDED` — the weakest criterion, it only says the process exited normally.
2. GPU utilization is not 0 — rules out "training actually ran on the CPU".
3. Loss falls epoch over epoch and `skipped=0` — shows backpropagation took effect and there were no zero-supervision batches.
4. `config.json` + `*.safetensors` + `pipeline_provenance.json` are all there — the export format is right and you can say which data was used.
5. **Stage 3 can load it and generate fluent text** — the only criterion that truly proves "the training artifact is usable".

The first four green and the fifth failing is an entirely possible combination (for example, if the labels are entirely masked, the loss will be a beautiful small number but the model learned nothing). So the acceptance check for this stage really has to wait for stage 3.

**If this was a resumed run, there is one more log line to look at**:

```
[train] resumed from checkpoint: weights loaded, optimizer state=yes, start epoch=2
```

Seeing only the epoch number without "weights loaded" means the resume path degenerated into the bug described in section 3.12 — training is running the second half on the base weights, while every other metric looks normal.

## 6. The mistakes that bite hardest

| Symptom | Actual cause | How to locate it |
| --- | --- | --- |
| Job sits in Initializing, no error | `NUM_WORKERS` ≠ total GPU workers | `ray status`, look at GPU total |
| Freezes on the very first batch / bus error | `/dev/shm` is only 64MB | Enter the Pod, `df -h /dev/shm` |
| Freezes when it reaches the first checkpoint | The FSDP gather is written inside `if rank == 0` | Check whether all ranks entered the collective call |
| The exported model produces pure random output | The state_dict key prefixes were not stripped | The unexpected keys warning from `from_pretrained` |
| Loss is extremely low but the model cannot answer | Labels entirely masked with -100 (a stage 1 bug) | Decode the supervised part of one sample back |
| Loss NaN | A whole batch of zero-supervision samples (0/0), or fp16 without a scaler | Look at the `skipped` count; switch to bf16 |
| **Loss starts from scratch after resume / results look untrained** | **Only the epoch number was restored, not the weights (section 3.12)** | The log must contain "weights loaded" |
| Loss jitters once after resume before settling | Weights restored but optimizer momentum was not | Look for "optimizer state=yes" in the log |
| FSDP is slower than DDP and saves no GPU memory | Missing `auto_wrap_policy`, the whole model wrapped as one unit (section 3.4) | Print the FSDP module tree and count the units |
| FSDP training stability inexplicably degrades | Gradient clipping used the local norm (section 3.9) | Check whether it goes through `model.clip_grad_norm_` |
| A card is occupied but training does not start | Missing gang scheduling (section 4.4) | `kubectl get pod`, look for a Pending Pod from the same group |

A second line of defense was added in the training loop specifically against zero-supervision batches:

```python
if not torch.isfinite(output.loss):
    optimizer.zero_grad(set_to_none=True)
    skipped += 1
    continue          # better to skip than to let nan into the parameters
...
if steps == 0:
    raise RuntimeError(f"epoch {epoch} had no valid step at all ({skipped} batches skipped)")
```

Once nan gets into the parameters it never comes out — every subsequent step's loss is nan, and the weights are entirely ruined. Stage 1 already filters zero-supervision samples on the data side (section 3.6 of that article); this is the runtime backstop. `steps == 0` raises immediately, avoiding the worst kind of silent failure: "training succeeded but not a single step ran".

## 7. What this stage deliberately omits

| Omitted | Cost | When you must do it |
| --- | --- | --- |
| Step-level resume | A failure mid-epoch reruns the whole round (weights and optimizer state are restored, what is missing is the data iteration position) | When a single epoch takes more than a few hours |
| LoRA / QLoRA | Full-parameter fine-tuning is expensive, and multiple adapters cannot coexist | GPU memory is tight, or you need different adapters for several business lines |
| LR scheduling, warmup | Convergence quality takes a hit | When you seriously tune for quality (almost always needed) |
| Validation-set loss and early stopping | You can only rely on a fixed epoch count and may overfit; `get_best_checkpoint` uses training-set loss | When the data volume makes overfitting a problem |
| Gang scheduling | Deadlock is possible when several jobs share a GPU pool | When the GPU pool is shared by several people |
| Cross-node validation | Multi-GPU on one machine does not prove cross-node NCCL works | When you scale beyond one machine |
| Models that do not fit on one card | Needs meta device initialization or FSDP2/DeepSpeed; changing `PARALLEL_STRATEGY` is not enough (section 3.4) | When the model exceeds single-card capacity |

Already added (absent in earlier versions): **data version pinning and provenance recording** (`resolve_data_version` + `pipeline_provenance.json`),
**real restoration of weights and optimizer state**, **FSDP global gradient clipping**, and **explicit best-checkpoint selection**.

The most worthwhile things to add next are **LR scheduling** and **validation-set loss** — they cost almost nothing (a few lines each) but directly determine training quality. A stricter approach is to verify the hashes in the data manifest before training starts, and record the verification result in `pipeline_provenance.json` as well.

One downstream effect related to LoRA is worth mentioning: if you switch to LoRA, stage 4's serving approach changes too — Ray Serve LLM's `LLMConfig.lora_config.dynamic_lora_loading_path` supports dynamically loading multiple adapters from object storage, so one base model serves several business lines. That is LoRA's more important value, beyond saving GPU memory.

## 8. What changes when you scale up

| Model size | `PARALLEL_STRATEGY` | GPU | Is this directory's code enough |
| --- | --- | --- | --- |
| 0.5B (default) | `ddp` | 2 × 24GB | Enough |
| 7B | `fsdp` (with auto_wrap_policy) | 8 × 80GB | **Mostly enough**: the 14GB model fits on one card, and FSDP saves parameters + gradients + optimizer state. You need gang scheduling; checkpoints are in the 14GB range so `num_to_keep` has to be sized against PVC capacity; liveness and load times enter the minutes range |
| 70B | Not enough | Multi-node | **This directory's code does not apply**, see below |

We have to be honest about 70B: **changing `PARALLEL_STRATEGY` will not get you there.** Three hard blockers:

1. **Initialization does not fit.** `prepare_model` does `model.to(device)` before wrapping with FSDP (section 3.4), and every rank does its own `from_pretrained` of a full model. A 140GB model OOMs right at that step, before FSDP is even involved. You need meta device initialization + `param_init_fn`, or switch to FSDP2 / DeepSpeed ZeRO-3's sharded loading path.
2. **The export does not fit.** `FULL_STATE_DICT` + `offload_to_cpu` has to assemble the full 140GB in rank 0's CPU memory before writing to disk. At that scale you should use `SHARDED_STATE_DICT` for training checkpoints and run a separate merge job only at final publication.
3. **Resume granularity is too coarse.** A single epoch of 70B is hours or even days, so epoch-level resume means one preemption costs a whole round.

So the applicable range of this directory's approach is **models that fit on one card but cannot be trained on one card**, which is roughly 0.5B to 7B (up to around 13B on an 80GB card). This is not something "get it working first, then scale up" can extend into; beyond that you have to switch training stacks. Spelling out that boundary is more useful than handing you a 70B config table that will not run.

---

**Next article**: [Stage 3 — Offline Batch Inference and Evaluation](../30-batch/batch-inference.en.md), on why vLLM is treated as an operator of Ray Data, where the offline/online trade-offs lie, and why this step is the real acceptance check for training.