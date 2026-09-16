# A four-stage LLM pipeline on Kubernetes: Ray + PyTorch + vLLM

Runnable code and manifests for a complete LLM chain on one Kubernetes cluster —
data processing, SFT training, offline batch inference, online serving — where each
stage submits independently and artifacts are handed off through a single shared volume.

> **Verification scope, stated up front.** Every API and field here was checked against
> pinned upstream source (Ray 2.58.0, KubeRay v1.7.0) rather than paraphrased docs, the
> four Ray custom resources validate against the real KubeRay v1.7.0 CRD schemas, and
> both shell scripts are exercised against behavioural fixtures. **It has not been
> executed end to end on a real GPU cluster.** Convergence, memory headroom, load times
> and probe thresholds all need confirming on your first real run. See
> [Verification scope](#verification-scope) for the full list of what was and was not checked.

This repository is code only. The long-form explanation of *why* each line of code and
each YAML field looks the way it does lives in a separate article series (not yet
published).

## The chain

```
raw jsonl ──► Ray Data ──────► Ray Train + PyTorch ──► Ray Data + vLLM ──► RayService + vLLM
              RayJob, CPU      RayJob, GPU             RayJob, GPU          long-running, GPU
              tokenize         SFT, export as a        batch generate       OpenAI-compatible
              + holdout        HuggingFace directory   + acceptance gate    endpoint
```

Each stage's output is the next stage's **direct input — there is no conversion script
anywhere in the chain.** Three decisions make that hold:

1. Stage 1 renders prompts with the tokenizer's own chat template, so the template has a
   single source of truth and training input matches what vLLM will build at inference.
2. Stage 2 exports a HuggingFace model directory (`config.json` + safetensors +
   tokenizer), which vLLM loads as-is.
3. Stages 3 and 4 point at the same immutable version directory, so what you evaluated is
   what you serve.

Artifacts go into versioned directories created exclusively — never overwritten.
Consumers resolve the convenience symlink into an immutable path once at startup, because
atomically replacing a symlink is *not* a snapshot.

## Who owns what

Confusing these five layers is the most common source of rework:

| Layer | Component | Owns | Does not own |
| --- | --- | --- | --- |
| Resources | Kubernetes | node placement, quota, GPU devices, storage | has no concept of an epoch or a KV cache |
| Orchestration | KubeRay Operator | RayJob / RayService lifecycles | no operator-level scheduling, no model loading |
| Runtime | Ray (Core / Data / Train / Serve) | task and actor scheduling inside the cluster, data sharding, worker groups, service replicas | does not provision nodes; **logical GPU quota is not memory isolation** |
| Compute | PyTorch | forward/backward, DDP/FSDP collectives, optimizer | does not know where data comes from |
| Engine | vLLM | continuous batching, PagedAttention, KV cache, OpenAI protocol | does not decide replica counts or routing |

In one line: **Kubernetes gives resources, KubeRay gives a Ray cluster, Ray gives
parallelism, PyTorch trains, vLLM serves.**

## Layout

```
00-platform/   namespace, shared RWX PVC, HF token example
10-data/       prepare_data.py  + RayJob    stage 1
20-train/      train_sft.py     + RayJob    stage 2
30-batch/      batch_infer.py   + RayJob    stage 3
40-serve/      RayService       + smoke_test.sh   stage 4
Makefile       one target per stage
wait_rayjob.sh polls status.jobStatus (see "Non-obvious decisions" #1)
```

## Prerequisites

```bash
# GPUs are schedulable (you should see nvidia.com/gpu capacity)
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu'

# An RWX StorageClass exists
kubectl get storageclass
```

- **At least 2 GPUs** (stage 2 uses 2 workers × 1 card). With one card, set `NUM_WORKERS`
  and `replicas` to 1.
- **Fill in `00-platform/storage.yaml`**: replace `REPLACE_WITH_RWX_STORAGE_CLASS` with a
  StorageClass that really supports ReadWriteMany (EKS `efs-sc`, GKE `standard-rwx` /
  Filestore CSI). Do **not** set it to the empty string — in Kubernetes an explicit
  `storageClassName: ""` means "use no StorageClass", which *disables* dynamic
  provisioning and leaves the PVC Pending unless a matching class-less PV exists.
  (Omitting the field entirely is what selects the default class, but default classes are
  usually RWO, which fails for multi-node training.)
- The namespace is hard-coded to `llm-pipeline` in the manifests and the Makefile
  deliberately does not let you override it — without a templating engine, an override
  only produces a split where the ConfigMap lands in one namespace and the RayJobs run in
  another. Edit the manifests, or wrap them in kustomize.
- An HF token is only needed for gated models; Qwen2.5 is public:
  `kubectl -n llm-pipeline create secret generic hf-token --from-literal=hf_token="$HF_TOKEN"`

## Running it

```bash
make operator     # install KubeRay Operator v1.7.0
make platform     # namespace + shared PVC
make data         # stage 1
make train        # stage 2
make batch        # stage 3
                  # ↓ before this, replace REPLACE_WITH_SFT_RUN_ID in
                  #   40-serve/rayservice-llm.yaml with the version stage 2 exported
make serve        # stage 4
make test         # acceptance checks against the live endpoint
```

The three job stages each delete the previous same-named RayJob, apply, then poll with
`wait_rayjob.sh`. `make serve` refuses to run while the `model_source` placeholder is
still in place.

## Version baseline

| Component | Version |
| --- | --- |
| KubeRay | v1.7.0 (helm chart 1.7.0) |
| Ray | 2.58.0 (Train V2 enabled by default) |
| Image | `rayproject/ray-llm:2.58.0-py312-cu130` |
| Kubernetes | 1.28+ |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |

vLLM comes from the image; do not pin it separately. Deviating from these versions is
fine, but re-check the fields — several of the ones this code depends on moved or were
deprecated recently.

## Non-obvious decisions in this code

Read this before "fixing" anything that looks odd. Most of these were real defects in an
earlier revision, found in review and fixed; reverting them reintroduces the bug.

1. **`wait_rayjob.sh` exists because `kubectl wait --for=condition=Complete rayjob/...`
   can never succeed.** KubeRay v1.7.0's `RayJobStatus` has no `Conditions` field, so that
   condition never appears and even a successful job waits until the timeout.
   `jobDeploymentStatus=Complete` is not a success signal either —
   `IsJobDeploymentTerminal` returns true for both `Complete` and `Failed`. The only
   criterion is `status.jobStatus == SUCCEEDED`.
2. **The RayService worker's `readinessProbe` is spelled out on purpose.** KubeRay injects
   its combined probe (raylet health + Serve proxy `/-/healthz`, `failureThreshold: 1`)
   *only when the user has declared none*. Declaring your own and omitting `/-/healthz`
   admits traffic before the engine has loaded weights, so requests 5xx. Either write none
   and let KubeRay inject, or write all of it — there is no safe middle.
3. **`/dev/shm` is mounted as a memory-backed `emptyDir`** in every GPU pod. The container
   default is 64MB; the PyTorch DataLoader, NCCL and vLLM's inter-process communication all
   use shared memory. Without it, training hangs or hits a bus error, and the message
   points nowhere near the cause. Note it counts against the container memory limit.
4. **`enableInTreeAutoscaling: true` is required** for Serve to scale. The Serve
   `autoscaling_config` only decides replica counts; adding Ray worker Pods is the Ray
   autoscaler's job. Without it, `workerGroupSpecs.maxReplicas` is decorative and replicas
   sit PENDING.
5. **`model_source` must be an immutable version directory, not the `sft-current`
   symlink.** Weights already in GPU memory are never reloaded because a symlink changed;
   editing `serveConfigV2` is what triggers a Serve update. Pointing at the symlink gives
   you "I thought I shipped it" plus old and new replicas serving side by side.
6. **Resume loads weights, optimizer state and epoch — all three.** Restoring only the
   epoch number leaves the model on base weights and silently discards prior training
   while the loss curve, checkpoints and job status all look healthy.
7. **FSDP paths use `model.clip_grad_norm_()` and an `auto_wrap_policy`.**
   `torch.nn.utils.clip_grad_norm_` clips each rank's *shard* norm, not the global norm.
   Without `auto_wrap_policy`, FSDP wraps the model as one flat unit and all-gathers every
   parameter at once, so the memory saving is close to zero. Also note `prepare_model`
   moves the model to the device *before* wrapping, so this path still requires the model
   to fit on one GPU.
8. **Export uses `result.get_best_checkpoint("loss", "min")`, not `result.checkpoint`** —
   Ray Train V2 defines `checkpoint` as the *latest*, which quietly contradicts a
   `checkpoint_score_attribute` config. The score is rank 0's unweighted mean of training
   batch losses, not a validation metric; it catches a blown final epoch, nothing more.
9. **Over-length samples have an explicit policy** (`10-data/prepare_data.py`): the prompt
   is never truncated (that would break template consistency), the answer may be truncated
   but always keeps its EOS, and a sample whose supervision budget is too small is
   rejected and counted. Concatenate-then-truncate instead, and a prompt that fills
   `max_len` produces all-`-100` labels — a whole batch of those makes cross-entropy `nan`
   and destroys the weights, with no error.
10. **`smoke_test.sh` checks streaming by arrival timing, not by counting `data:` lines.**
    A proxy can buffer a complete response and deliver every event line at once; the line
    count is identical. Its predecessor passed a service that returned `{}`, emitted only
    an error event, and exited with a curl timeout.
11. **`deploymentUnhealthySecondThreshold` is deliberately absent.** KubeRay v1.7.0 marks
    it and `serviceUnhealthySecondThreshold` "Deprecated: This field is not used anymore".
    Slow model loading is protected by the worker's liveness `failureThreshold` instead.

Two more thresholds worth knowing: the worker liveness
`failureThreshold × periodSeconds` must exceed model load time or Pods restart mid-load
(looks like "the model is too big to start"), and `terminationGracePeriodSeconds` must
exceed p99 single-request generation time or a rollout cuts streams that have already
delivered half their tokens.

## Without an RWX volume

Switch to object storage; three replacements, and the PVC is no longer needed:

| Where | Change to |
| --- | --- |
| stage 1 output | `write_parquet("s3://bucket/data/tokenized/v-<id>")` |
| stage 2 `RunConfig` | `storage_path="s3://bucket/train"` |
| stage 4 `model_source` | `{bucket_uri: "s3://bucket/models/sft-<id>"}` (`CloudMirrorConfig` form) |

The cost is that there are no symlinks and no atomic rename, so "current version" has to
be expressed by an explicit version number or a pointer file, and publishing becomes an
edit to `bucket_uri` in `serveConfigV2`. Ray Train requires shared storage for multi-node
training; a local path only works on a single node.

## Scaling up

| Model | `PARALLEL_STRATEGY` | GPUs | Does this code still apply? |
| --- | --- | --- | --- |
| 0.5B (default) | `ddp` | 2 × 24GB | yes |
| 7B | `fsdp` | 8 × 80GB | mostly — add gang scheduling (Kueue/Volcano), budget PVC for 14GB checkpoints, raise liveness thresholds for minute-scale loads |
| 70B | — | multi-node | **no** — see below |

At 70B three things block this code, and none is a config change: every rank loads a full
model before FSDP wraps it, so initialization OOMs; `FULL_STATE_DICT` with CPU offload has
to assemble 140GB on rank 0 before writing; and epoch-level recovery means one preemption
costs a whole epoch. That range needs meta-device init or FSDP2/DeepSpeed ZeRO-3, sharded
checkpoints, and step-level recovery. This code targets models that fit on one card but
cannot be *trained* on one card — roughly 0.5B to 13B on 80GB.

Three numbers move together and must be recomputed as a set:

```
stage 1 MAX_LEN + MAX_TOKENS  ≤  stage 4 max_model_len
                                 └─ sets per-sequence KV cache size
                                    └─ with gpu_memory_utilization, sets the token budget
                                       └─ which bounds target_ongoing_requests
```

KV cache per token is `2 × layers × kv_heads × head_dim × dtype_bytes` — note **kv_heads**
(GQA), not attention heads; using the latter overestimates by an order of magnitude. For
Qwen2.5-7B that is 56 KiB/token, so a single 80GB card at `gpu_memory_utilization: 0.85`
holds roughly 920K tokens ≈ 900 concurrent sequences at 1K tokens each. The default
`target_ongoing_requests: 32` is tuned for the 0.5B/24GB case and is far too conservative
at 7B.

## Observability

- **Ray Dashboard**: `kubectl -n llm-pipeline port-forward svc/<head-svc> 8265:8265` for
  task/actor placement, per-GPU occupancy and Serve replica status.
- **vLLM engine metrics** require `log_engine_metrics: true` in `LLMConfig`; only then do
  you get TTFT, TPOT/ITL, engine queue depth, KV cache utilization and prefix cache hit
  rate. Capacity planning and autoscaling need these, not CPU utilization.
- **Diagnose in layers.** Pods Ready (K8s) → Ray nodes registered (Ray) → Serve
  application and replicas RUNNING (app) → the engine actually generates
  (`/v1/chat/completions`). Only the fourth means "the service works"; the first three
  green with the fourth failing is a common combination.

## Verification scope

**Checked against pinned upstream source:**

- KubeRay v1.7.0's `ray.io/v1` RayJob / RayService fields, `upgradeStrategy` values,
  `autoscalerOptions`, `enableInTreeAutoscaling`
- `RayJobStatus` has **no** `Conditions` field; `RayServiceReady = "Ready"` **does** exist
  (so RayJob cannot use `kubectl wait --for=condition`, RayService can)
- `deploymentUnhealthySecondThreshold` / `serviceUnhealthySecondThreshold` marked
  deprecated and unused
- KubeRay's probe injection: a user-declared probe wins; the RayService worker's combined
  probe is injected only when none is declared — identical logic in v1.4.2 and v1.7.0
- Ray 2.58.0 `ray.data.llm` exports `build_processor` (`build_llm_processor` is no longer
  in `__all__`); `vLLMEngineProcessorConfig` fields
- `ray.serve.llm`'s `LLMConfig` / `ModelLoadingConfig` / `CloudMirrorConfig` fields
- Ray Train V2 on by default; `restore` / `can_restore` deprecated; `Result.checkpoint` is
  the latest while `get_best_checkpoint(metric, mode)` is separate; `prepare_model` moves
  to device **before** wrapping and passes no `auto_wrap_policy` by default
- torch 2.7's `FSDP.clip_grad_norm_` / `optim_state_dict` / `optim_state_dict_to_load`,
  and `ModuleWrapPolicy`
- `rayproject/ray-llm:2.58.0-py312-cu130` and helm chart `1.7.0` exist

**Executed locally (no Ray, no GPU, no cluster):**

- all four Ray custom resources validate against the real KubeRay v1.7.0 CRD schemas
- `wait_rayjob.sh` behaves correctly across 6 status fixtures, including "deployment
  Complete but job FAILED must not be read as success"
- `smoke_test.sh` behaves correctly across 5 fixtures: empty `{}`, an error event, a curl
  timeout, a buffered burst and a missing `[DONE]` all fail; incremental and complete passes
- `Tokenize._encode` boundary behaviour: zero-supervision samples are rejected, and every
  accepted sample ends its supervised span with EOS whether or not the answer was truncated
- Python compiles, YAML parses (including the embedded `serveConfigV2`), Makefile parses

**Not verified:** never executed on a real GPU cluster. Training convergence, memory
headroom, load times, whether the probe thresholds are right, the real behaviour of FSDP
state-dict and optimizer-state collectives on a specific torch build, and the actual
memory saving from `auto_wrap_policy` all need a first real run. The bundled demo dataset
is 8 template rows (6 train / 2 eval) — a falling loss proves the chain is wired, not that
training works.

## License

[Apache-2.0](LICENSE). Some code follows patterns from the Ray documentation, which is
Apache-2.0 licensed.