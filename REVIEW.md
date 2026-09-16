# Technical review

Date: 2026-09-14  
Reviewed baseline: all 17 files present in this directory, including the Makefile, three Python programs, seven YAML files, smoke test, overview, and four stage articles.  
API baseline checked: Ray **2.58.0**, KubeRay **1.7.0**.  
Verdict: **Changes requested before describing this as a working end-to-end example.**

The component boundaries and HF-format model handoff are useful. However, the default Make workflow waits on a nonexistent RayJob condition; recovery does not restore training state; serving cannot acquire the additional GPU workers advertised by its autoscaling settings; and the smoke test can declare a broken stream successful. These are correctness problems independent of production features such as authentication or a sophisticated evaluator.

No implementation files were changed by this review.

## Severity and evidence conventions

- **P1:** blocks a documented workflow or can silently invalidate training, model selection, or acceptance.
- **P2:** incorrect behavior in a supported mode, operational/configuration defects, or inaccurate implementation guidance that should be corrected.
- **Reproduced locally:** exercised actual repository logic or a filesystem/control fixture without Ray, PyTorch, GPU, or a cluster.
- **Source-confirmed:** established from local code and, where necessary, the pinned upstream implementation. This is not a claim of GPU execution.

## Findings

### R1 — P1: All three Make targets wait for a RayJob condition that does not exist

**Locations:** [Makefile](Makefile), lines 37–47; data article §5; corresponding train/batch run instructions.

`kubectl wait --for=condition=Complete rayjob/...` looks for `status.conditions` with type `Complete`. KubeRay 1.7.0's `RayJobStatus` has `jobStatus` and `jobDeploymentStatus`, not that condition array. `Complete` is a deployment-status value, not a Kubernetes condition.

Consequently, a successfully completed data job can leave `make data` waiting until its one-hour timeout; training and batch targets have the same defect. Merely replacing the wait with `jobDeploymentStatus=Complete` is insufficient: deployment completion also covers terminal failed/stopped jobs.

**Fix:** poll `status.jobStatus`, succeed only on `SUCCEEDED`, and exit promptly on `FAILED`, `STOPPED`, or controller submission/validation failure. Retain an overall timeout and print status/message on failure. A JSONPath success-only wait can be a minimal first step, but still needs a failure monitor.

Also define rerun behavior: reapplying an already completed, fixed-name RayJob is not a fresh execution. Use unique names or explicitly retire the completed object before recreating it.

**Evidence:** local Make dry-run; pinned [`RayJobStatus`](https://github.com/ray-project/kuberay/blob/v1.7.0/ray-operator/apis/ray/v1/rayjob_types.go), including the comment explaining terminal `Complete`.

**Acceptance:** a successful fake/status fixture returns promptly; failed/stopped fixtures fail promptly; a second intentional run actually creates new work.

### R2 — P1: “Resume” skips epochs while retaining newly loaded base weights

**Locations:** [train_sft.py](20-train/train_sft.py), lines 87–113, 149–158, 179–201; [training article](20-train/sft-training.en.md), §3.12 and §7; overview §6.

Every worker loads `config["base_model"]` and constructs a new optimizer. The checkpoint branch reads only `trainer_state.pt` and updates `start_epoch`; it never loads the saved HF weights into the model. The checkpoint metadata contains only epoch/loss, without optimizer or RNG state.

For example, restoring an epoch-0 checkpoint in the default two-epoch run starts epoch 1 using the original base model. This does not resume at an epoch boundary: it discards the learned parameters while claiming previous epochs are complete. Epoch-boundary recovery still needs model state; continuation of the same AdamW process also needs optimizer state.

Two additional parts of the advertised recovery path are broken:

1. No `FailureConfig` is supplied. Ray 2.58 Train V2 defaults ordinary worker-error retries to `max_failures=0`. Controller and recognized preemption policies are separate; do not generalize their defaults into automatic recovery from arbitrary worker errors.
2. The article recommends `TorchTrainer(..., resume_from_checkpoint=...)`, but the pinned Train V2 constructor **raises `DeprecationWarning` as an exception** when this argument is non-null. It is not just a logged warning. The program also exposes no manual checkpoint input.

**Fix:** implement a version-appropriate initial checkpoint input through application configuration and load it inside the training function; use Ray's provided recovery checkpoint for actual retries. Restore weights before continuing, restore optimizer/RNG and the declared data position, validate world size/data identity, and configure the intended failure policy. FSDP optimizer restoration needs its own supported state-dict handling.

**Evidence:** local control-flow inspection; pinned [Train V2 constructor](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/train/v2/api/data_parallel_trainer.py) and [FailureConfig](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/train/v2/api/config.py).

**Acceptance:** inject failure after a saved epoch; verify loaded parameter/state identity and resumed progress against a no-failure control within declared tolerances. Test manual continuation separately. A log that prints a resumed epoch alone is not evidence.

### R3 — P1: Serve autoscaling has no mechanism to increase Ray GPU workers

**Locations:** [rayservice-llm.yaml](40-serve/rayservice-llm.yaml), lines 42–48, 51–52, 116–122; [online article](40-serve/online-serving.en.md), §3.3; data article §4.4.

Serve is allowed to request four replicas, but the RayCluster starts one worker with one GPU and omits `enableInTreeAutoscaling: true`. `minReplicas: 1` / `maxReplicas: 4` constrain autoscaling when enabled; they do not enable it.

Serve can therefore request another engine replica while its placement group waits for a GPU that this manifest never provisions. Spare GPUs in Kubernetes do not fix the missing Ray worker control loop.

**Fix:** explicitly enable and configure Ray in-tree autoscaling, including the operator-generated sidecar/RBAC and tested resource-demand path, or pre-provision enough fixed workers and document that tradeoff. Kubernetes node autoscaling is a further, separate dependency.

**Evidence:** parsed manifest has no autoscaling enable field; pinned [RayClusterSpec](https://github.com/ray-project/kuberay/blob/v1.7.0/ray-operator/apis/ray/v1/raycluster_types.go).

**Acceptance:** sustained demand results in an additional healthy model replica **and** GPU worker Pod; distinguish desired replicas, pending placement groups, provisioned nodes, and requests actually served.

### R4 — P1: The smoke test can pass empty responses and failed streams

**Locations:** [smoke_test.sh](40-serve/smoke_test.sh), lines 73–108; [online article](40-serve/online-serving.en.md), §5; overview stage-4 acceptance.

Non-streaming acceptance only checks that the response string is nonempty, so `{}` passes. Streaming acceptance counts `data:` lines and suppresses the pipeline's failure with `|| true`; it does not require content, a valid finish reason, or `[DONE]`.

**Local reproduction:** ran the actual shell script with fake `kubectl` and `curl` executables. `/v1/models` returned the expected ID; non-streaming returned `{}`; streaming returned one assistant-role event and one error event, no content or `[DONE]`, and curl exited **28**. The script exited **0** and printed `All three checks passed`.

Counting SSE lines also cannot detect buffering: a proxy can buffer an entire valid multi-event response and deliver all lines together. Conversely, a short valid answer need not produce multiple content chunks.

**Fix:** parse JSON/SSE; require the exact model ID and valid generation fields; reject error events, missing completion, and transport failures; use connect/overall timeouts. Track first content time and subsequent delivery times for incremental-delivery checks, separately from protocol completeness.

Construct request JSON with a JSON encoder: the current direct interpolation breaks when `PROMPT` contains quotes, backslashes, or newlines. Verify the port-forward process remains alive so an existing unrelated local server cannot satisfy the test.

**Acceptance:** the reproduced failed stream and `{}` both fail; complete short output passes; a buffered multi-event fixture cannot be declared incrementally delivered based only on line count.

### R5 — P1: Atomic symlink updates do not pin the data or model used by a consumer

**Locations:** [prepare_data.py](10-data/prepare_data.py), lines 151–168, 196–210; [train_sft.py](20-train/train_sft.py), lines 36, 164–176, 211–215; [batch_infer.py](30-batch/batch_infer.py), lines 28–29, 40–44; serving `model_source`, line 35; overview §2.

The overview promises that reruns cannot change data under a running training job and that batch evaluation and serving use identical bytes. Both consume mutable `current` paths without pinning a resolved version. Atomic replacement of a directory entry does not make a sequence of later opens a snapshot.

A consumer can resolve one file before a switch and another afterward. Existing engines retain old in-memory weights while newly created replicas load the new target. Batch and serving can load different versions even if their configured path strings are identical.

`eval.jsonl` is an additional unversioned output: preparation truncates and rewrites it **before** tokenization/publication completes. A failed preparation can therefore replace evaluation data while leaving `tokenized/current` on the old version.

`RUN_ID` has second-level precision and data/output directories use `exist_ok=True`; repeated explicit IDs or concurrent starts also do not enforce the stated non-overwrite contract.

**Fix:** resolve and record immutable versions at stage submission/start, pass those exact paths to all workers, version the evaluation data with its dataset, and make new-version creation exclusive. Pin HF revisions as well: data tokenizers and training models independently load an unpinned repository ID today. Record data/model/config identity in exported artifacts and evaluation output.

The batch article §3.3 already acknowledges part of this limitation. Align the overview and serving claims with that admission rather than retaining contradictory guarantees.

**Evidence:** local filesystem fixture demonstrated that a retained `current/weights` path reads v2 after a link switch while a previously resolved path reads v1. This demonstrates pathname semantics, not a measured Ray race.

**Acceptance:** switch `current` during a multi-file consumer run and verify every read/replica in that run remains on one recorded version. Failed preparation must not alter the active evaluation dataset.

### R6 — P1: The documented model upgrade/rollback command is a no-op

**Locations:** [online article](40-serve/online-serving.en.md), lines 385–403; [rayservice-llm.yaml](40-serve/rayservice-llm.yaml), line 52.

The upgrade example patches `rayClusterConfig.rayVersion` to `"2.58.0"`, exactly its current value. This is not a desired-spec change and does not reliably trigger cluster replacement or model reload. After switching `sft-current`, the running engine can continue serving the old weights indefinitely; rollback has the same problem.

**Fix:** preferably put the immutable model version directly in `serveConfigV2` and change that value to update the application. If testing NewCluster replacement, make an actual, supported Pod-template/configuration change and verify a new RayCluster is created. Do not fabricate a Ray version merely to force a rollout, and do not assume a metadata-only change necessarily triggers one.

**Evidence:** parsed current value equals the documented patch value.

**Acceptance:** observe the expected application/cluster revision change and independently verify the served model release identity, including after rollback.

### R7 — P1: Long prompts can eliminate every supervised token

**Locations:** [prepare_data.py](10-data/prepare_data.py), lines 114–133; [train_sft.py](20-train/train_sft.py), lines 126–135; data article §3.6.

The program concatenates prompt and answer, then truncates both inputs and labels. If the prompt fills `MAX_LEN`, all labels are `-100`. There is no rejection or minimum-supervision check. A batch with no effective target tokens can yield undefined/NaN mean cross entropy, and the training loop does not reject non-finite loss or gradients.

Even when part of the answer survives, truncation can remove EOS, contradicting the unconditional claim that EOS is learned.

**Local reproduction:** executed the actual `_encode` method with a synthetic tokenizer returning a ten-token prompt and `max_len=8`. Output labels were eight `-100` values: **zero supervised tokens**. No real tokenizer or GPU was used.

**Fix:** define an explicit overlength policy, preserve required answer/EOS tokens or reject the sample, and check the effective shifted labels contain targets. Add finite-loss checks before updating parameters. Report accepted/rejected counts.

**Acceptance:** long-prompt and boundary-length fixtures cannot silently enter training with no targets; intended EOS behavior is checked.

### R8 — P2: FSDP uses local gradient clipping rather than the distributed norm

**Location:** [train_sft.py](20-train/train_sft.py), lines 95–100 and 131–133.

The advertised FSDP branch always calls `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`. Under sharded gradients, each rank sees only a portion of the gradient. Clipping those local norms independently is not clipping the global model norm.

**Fix:** use the FSDP instance's supported distributed `clip_grad_norm_` path on every participating rank when sharded; retain the ordinary helper for DDP. Validate against the Torch version actually inside the image.

Also qualify the scaling recipe: Ray's pinned `prepare_model` moves the model to the device before wrapping it. Combined with loading a full base model on every rank and no transformer-layer auto-wrap policy, changing only `PARALLEL_STRATEGY` is not a demonstrated route to a model that cannot fit on one GPU. Full CPU state collection has a separate peak.

**Evidence:** local call site and pinned [Ray Torch preparation](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/train/torch/train_loop_utils.py). No FSDP GPU reproduction was performed.

**Acceptance:** a two-rank clipping fixture agrees with the unsharded global norm; measure initialization, forward/backward, and export peaks before publishing the 7B/70B sizing recipe.

### R9 — P2: The configured unhealthy-deployment timeout is ignored

**Locations:** [rayservice-llm.yaml](40-serve/rayservice-llm.yaml), lines 21–22; [online article](40-serve/online-serving.en.md), §3.5.

KubeRay 1.7.0 explicitly marks `DeploymentUnhealthySecondThreshold` as deprecated and **not used anymore**. Setting it to 900 does not provide the described protection against slow-loading models or alter recovery timing as the article claims.

**Fix:** remove the ineffective knob and document the actual initialization/application-health behavior of the pinned controller. If exposing a supported timeout, trace it to the code path it controls before recommending it.

**Evidence:** pinned [RayServiceSpec](https://github.com/ray-project/kuberay/blob/v1.7.0/ray-operator/apis/ray/v1/rayservice_types.go), comments immediately above this field.

### R10 — P2: “Best checkpoint” export actually selects the latest checkpoint

**Locations:** [train_sft.py](20-train/train_sft.py), lines 138, 158, 194–215; training article §3.13.

Retention is scored by `loss`, but export uses `result.checkpoint`. Ray 2.58 Train V2 defines this as the **latest** checkpoint; scored retained checkpoints are available separately. A worse final epoch can therefore be exported despite the stated best-checkpoint policy.

The score is also rank 0's unweighted average of batch losses, not a globally aggregated validation metric. That limitation should be explicit even for a toy example.

**Fix:** either declare that the latest checkpoint is intentionally exported, or explicitly call the supported best-checkpoint selection API and record the selected metric/version. For quality selection, use a held-out metric with the appropriate token/rank aggregation.

**Evidence:** pinned [Train V2 Result](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/train/v2/api/result.py) documents `checkpoint`, `best_checkpoints`, and `get_best_checkpoint`.

### R11 — P2: Batch “nonempty output” acceptance never fails the job

**Locations:** [batch_infer.py](30-batch/batch_infer.py), lines 89–99; batch article §5.

The code counts empty strings in five sampled rows and prints the count, but never raises. Even if every sampled response is empty, the job can finish successfully and the next stage can be started.

This is narrower than adding the explicitly deferred production quality gate: the current code does not enforce its own minimal nonempty-output acceptance. `str(None)` also becomes `"None"`, which counts as nonempty.

**Fix:** reject missing/non-string/empty generated text according to an explicit policy; report total input/output/error counts; exit nonzero when the declared acceptance fails. Persist the smoke evaluation and exact model/data versions. Keep this distinct from claims of model quality.

**Acceptance:** empty, null, missing, and zero-row fixtures fail; successful text fixtures pass; application exit status reflects the result.

### R12 — P2: Empty storage class disables default-class selection

**Location:** [storage.yaml](00-platform/storage.yaml), lines 26–29.

The comment says leaving the class blank uses the default StorageClass. In Kubernetes, an explicitly supplied `storageClassName: ""` requests no storage class and disables default dynamic provisioning. Omitting the field is different.

The README does tell readers to supply a real RWX class, so this is not a failure for a reader who fills it in correctly. It is still an incorrect fallback instruction that can leave the PVC Pending when no matching classless PV exists.

**Fix:** require a real RWX class with an unmistakable placeholder/preflight check, or document classless static provisioning accurately. Do not suggest a default class necessarily supports RWX.

### R13 — P2: The Makefile namespace override does not affect resource namespaces

**Locations:** [Makefile](Makefile), lines 1, 24–47, 52–60; all resource manifests.

`NAMESPACE` controls ConfigMap creation, waits, status, test, and cleanup. The manifests still explicitly target `llm-pipeline`. With an alternate namespace, scripts can be created in one namespace while RayJobs run in another and cannot mount that ConfigMap; waits and cleanup also target the wrong resources.

**Evidence:** `make -n data train batch test NAMESPACE=review-alt` shows the split behavior without executing any cluster commands.

**Fix:** remove unsupported namespace configurability or render namespace consistently with a single mechanism, including Namespace/PVC/Secret/ConfigMap and all CRs.

## Acknowledged scope gaps and documentation corrections

These should not be presented as newly discovered missing production features: the stage articles explicitly defer a proper holdout, lineage manifest, deployment gate, gateway authentication, and detailed draining. The review does not require implementing all of those to retain a clearly labeled teaching example. It does require removing contradictory guarantees.

| File/section | Correction |
| --- | --- |
| Data article §3.9/§5; `prepare_data.py` | The evaluation data is a subset of **all** training rows, not a holdout. The demo starts with 128 rows, deduplicates to 8, and `take(32)` returns at most 8; acceptance cannot require exactly 32. Unique real input need not shrink during deduplication. |
| Overview §7, object storage | “Three replacements” do not remove the PVC dependency: raw/eval paths, existence checks, local export, symlink publication, training input, and artifact distribution still need changes. |
| Training article §3.4/§3.5 | Parameter/gradient/optimizer precision must be accounted for separately. `model.to(bfloat16)` plus ordinary AdamW does not establish FP32 optimizer moments; the displayed memory equation mixes units. Add actual state dtype/size measurements. |
| Training article §3.9 | FSDP's state-dict behavior depends on the selected API/configuration; do not claim that an unconfigured `state_dict()` universally returns shards. Explicitly requesting a full export is useful regardless. |
| Training article §3.13 | In the inspected Train V2 `fit()`, a returned execution error is raised. Checking `result.error` defensively is harmless, but the article's stated failure-return behavior is not the pinned V2 path. |
| Training article §4.2 | An 8 GiB tmpfs `sizeLimit` is a cap, not 8 GiB pre-reserved/always consumed RAM. Actual tmpfs use contributes to memory pressure. |
| Training article §4.3 | NVIDIA time-slicing does **not** provide MIG-like memory isolation. Guaranteed QoS also does not prevent a container exceeding its memory limit or exhausting GPU memory. |
| Training article §4.4; overview sizing | Kueue admission and scheduler Pod placement are different operations. Do not promise generic atomic gang placement merely by naming Kueue. Worker demand need not equal every GPU in the cluster; it must fit available compatible resources. |
| Online article §4 | Proxy/node health is not proof that the intended model has loaded or meets an SLO. A node-health liveness endpoint should not fail merely because model loading takes time. Preserve application-level generation acceptance separately. |
| Online article §5 | SSE event count proves neither token count nor incremental arrival. More than one event can arrive fully buffered; this also motivates R4. |
| Batch article §3.4/§8; online sizing | `max_model_len` is a limit, not a promise to allocate that many KV tokens for every active request. Cache allocation, token lengths, and batching matter. Chunked prefill, prefix reuse, and length ordering need workload measurements rather than unconditional speedup claims. |
| Batch article §3.6/§5 | Temperature zero does not guarantee bitwise reproducibility across kernels/topologies. Fluent output from an already instruction-tuned base does not prove SFT ran correctly or improved it. |
| Overview/stage headers | “API checked” is distinct from “executable end-to-end verified.” Retain the honest no-GPU-run disclaimer and soften the stronger runnable/recovery/scaling assertions until the tests below pass. |

## Positive findings

- Clear separation of batch RayJobs and the long-lived RayService.
- Shared PVC mounted into both drivers and workers for local-path handoffs.
- Model/tokenizer exported together in HF format.
- FSDP full-state collection is correctly invoked outside the rank-0-only save branch.
- Prompt/padding masking and explicit actor-based tokenizer reuse are sensible foundations, subject to the truncation fix.
- GPU workers mount `/dev/shm`; job deadlines and delayed cleanup are explicit.
- Worker readiness includes both raylet and Serve proxy checks; the pinned KubeRay source confirms that user-provided probes bypass default injection. The **meaning** attributed to proxy health still needs correction.
- Credentials are placeholders and optional where declared; the review found no reason to change the namespace/Secret example beyond documenting its intended use.

## Verification performed

1. Read every source, manifest, Makefile, and article listed in the coverage table.
2. Parsed all three Python programs with `ast.parse`.
3. Parsed seven YAML files plus the nested `serveConfigV2` YAML.
4. Ran `bash -n 40-serve/smoke_test.sh`.
5. Ran Make dry-runs only; no Make deployment target was executed.
6. Executed the repository's `_encode` method with a synthetic tokenizer: all-masked long-prompt case reproduced.
7. Executed the actual smoke script with mocked executables: invalid JSON generation plus failed/incomplete SSE incorrectly passed.
8. Exercised atomic symlink replacement using temporary files: mutable path and pinned path read different versions.
9. Inspected version-pinned upstream Ray/KubeRay APIs for status, recovery, checkpoint selection, autoscaling configuration, and probes.

**Not performed:** image pull/build; actual image package inventory; dependency imports against Ray 2.58; real tokenizer/weights; GPU training; DDP/FSDP collectives; live RayJob/RayService reconciliation; Kubernetes server-side validation; node autoscaling; networked HTTP generation; performance or convergence tests.

Syntax success is not runtime approval. The local fixtures establish the stated Python/shell/path behavior only.

## Suggested correction and validation order

1. Fix RayJob waits and fail-fast orchestration. Demonstrate data → train → batch progression and a deliberate stage failure.
2. Correct training recovery and overlength handling. Run one short baseline and one recoverable failure with state comparison.
3. Pin consumer artifact versions; separate candidate export from serving promotion.
4. Make batch/smoke acceptance meaningful; run malformed-response and incomplete-stream fixtures before a real generation.
5. Enable/test GPU worker autoscaling and replace the no-op rollout recipe.
6. Correct best-checkpoint selection, FSDP handling, storage/namespace behavior, and unsupported documentation claims.
7. Capture the actual image digest/package versions and execute the minimal GPU pipeline. Test each optional large-model/FSDP path separately before calling it supported.

## File-by-file coverage

The following snapshot identifies the files reviewed before this review file was added. Hashes are SHA-256 prefixes, for change detection rather than artifact authentication.

| File | Lines | SHA-256 prefix | Coverage |
| --- | ---: | --- | --- |
| [00-platform/hf-token.example.yaml](00-platform/hf-token.example.yaml) | 18 | `1e6fbcbd8163` | Storage and namespace contracts: R12–R13; Secret placeholder checked |
| [00-platform/namespace.yaml](00-platform/namespace.yaml) | 8 | `86f1acfdfeda` | Storage and namespace contracts: R12–R13; Secret placeholder checked |
| [00-platform/storage.yaml](00-platform/storage.yaml) | 32 | `02ae9bce3ae9` | Storage and namespace contracts: R12–R13; Secret placeholder checked |
| [10-data/data-processing.en.md](10-data/data-processing.en.md) | 463 | `527bf759af51` | Data contracts, truncation and publication: R5, R7; YAML integration: R1, R13 |
| [10-data/prepare_data.py](10-data/prepare_data.py) | 215 | `efa8fc8d4ba0` | Data contracts, truncation and publication: R5, R7; YAML integration: R1, R13 |
| [10-data/rayjob-data-prep.yaml](10-data/rayjob-data-prep.yaml) | 129 | `0e4de33f7e9a` | Data contracts, truncation and publication: R5, R7; YAML integration: R1, R13 |
| [20-train/rayjob-train-sft.yaml](20-train/rayjob-train-sft.yaml) | 163 | `389b5478fa17` | Recovery, FSDP, checkpoint selection: R2, R8, R10; resources and handoff checked |
| [20-train/sft-training.en.md](20-train/sft-training.en.md) | 528 | `2c9343593aa6` | Recovery, FSDP, checkpoint selection: R2, R8, R10; resources and handoff checked |
| [20-train/train_sft.py](20-train/train_sft.py) | 220 | `2705468f5bd9` | Recovery, FSDP, checkpoint selection: R2, R8, R10; resources and handoff checked |
| [30-batch/batch-inference.en.md](30-batch/batch-inference.en.md) | 380 | `4bb74fe8b59f` | Artifact identity and output acceptance: R5, R11; resource/mount settings checked |
| [30-batch/batch_infer.py](30-batch/batch_infer.py) | 103 | `a409f93af68b` | Artifact identity and output acceptance: R5, R11; resource/mount settings checked |
| [30-batch/rayjob-batch-infer.yaml](30-batch/rayjob-batch-infer.yaml) | 130 | `2f8fff9de092` | Artifact identity and output acceptance: R5, R11; resource/mount settings checked |
| [40-serve/online-serving.en.md](40-serve/online-serving.en.md) | 486 | `a79f9bceeefb` | Autoscaling, streaming, rollout and health: R3–R6, R9 |
| [40-serve/rayservice-llm.yaml](40-serve/rayservice-llm.yaml) | 202 | `d6a6b00951b7` | Autoscaling, streaming, rollout and health: R3–R6, R9 |
| [40-serve/smoke_test.sh](40-serve/smoke_test.sh) | 110 | `d7329f4c8cfd` | Autoscaling, streaming, rollout and health: R3–R6, R9 |
| [Makefile](Makefile) | 61 | `dde524b40ee1` | Wait/rerun and namespace behavior: R1, R13 |
| [README.md](README.md) | 288 | `9b2e33b48141` | Cross-stage guarantees, workflow, storage and sizing claims |

## Additional pinned source references

- [Ray 2.58 Train V2 configuration](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/train/v2/api/config.py)
- [Ray 2.58 legacy FailureConfig inherited by V2](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/air/config.py)
- [Ray 2.58 trainer deprecation definitions](https://github.com/ray-project/ray/blob/ray-2.58.0/python/ray/train/base_trainer.py)
- [KubeRay 1.7 Pod/probe construction](https://github.com/ray-project/kuberay/blob/v1.7.0/ray-operator/controllers/ray/common/pod.go)

## Reproduce the two local logic failures

Run from this directory. The first fixture extracts only the actual encoding method, so it does not import Ray/NumPy/Transformers:

```python
import ast
from pathlib import Path
from types import SimpleNamespace

tree = ast.parse(Path("10-data/prepare_data.py").read_text())
klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tokenize")
method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "_encode")
namespace = {}
exec(compile(ast.Module(body=[method], type_ignores=[]), "fixture", "exec"), namespace)

class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0
    def apply_chat_template(self, *args, **kwargs):
        return "prompt"
    def __call__(self, text, **kwargs):
        return {"input_ids": list(range(10)) if text == "prompt" else [8, 9]}

_, _, labels = namespace["_encode"](
    SimpleNamespace(tokenizer=Tokenizer(), max_len=8), "question", "answer"
)
print(labels)
assert all(label == -100 for label in labels)
```

The second runs the actual shell script with local command stubs. It makes no Kubernetes or network requests:

```python
import os
from pathlib import Path
import subprocess
import tempfile

script = Path("40-serve/smoke_test.sh").resolve()
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "kubectl").write_text("#!/bin/sh\nexit 0\n")
    (root / "curl").write_text("""#!/bin/sh
case "$*" in
 */v1/models*) printf '%s\\n' '{"data":[{"id":"sft-qwen"}]}' ;;
 *'"stream":true'*) printf '%s\\n' \
'data: {"choices":[{"delta":{"role":"assistant"}}]}' \
'data: {"error":{"message":"engine failed"}}'; exit 28 ;;
 *) printf '%s\\n' '{}' ;;
esac
""")
    for path in root.iterdir():
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script)], capture_output=True, text=True, timeout=10,
        env={**os.environ, "PATH": str(root) + os.pathsep + os.environ["PATH"]},
    )
    print(result.stdout)
    print("exit =", result.returncode)
    assert result.returncode == 0  # Demonstrates the defect; should fail after correction.
```
