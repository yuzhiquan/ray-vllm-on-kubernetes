# Stage 4: online serving — turning vLLM into an endpoint that can scale, upgrade, and roll back

> This article corresponds to [`rayservice-llm.yaml`](rayservice-llm.yaml) and [`smoke_test.sh`](smoke_test.sh).
> Previous article: [Stage 3, offline batch inference](../30-batch/batch-inference.en.md). Baseline: Ray 2.58.0, KubeRay v1.7.0.

## 1. Where this stage sits in the LLM lifecycle

This is the **delivery point**. The first three stages can be rerun, can fail, can be slow; the moment something goes wrong in this stage it is a user-visible outage. What makes it special in the lifecycle:

```
Stages 1-3: batch semantics             Stage 4: serving semantics
──────────────────────────             ──────────────────────────
destroyed once finished                 long-running, must stay alive
failure means rerun                     failure means removing from traffic immediately
resource needs are computable            load is unpredictable, needs feedback control
version switch = change a path           version switch = zero-downtime upgrade + rollback
no "partially complete"                  streaming responses are inherently "partially complete"
```

That last line is the biggest difference between LLM serving and traditional web serving, and it is the source of most of this stage's complexity. An ordinary HTTP request either succeeds or fails; a streaming generation request can **have already emitted 80 tokens to the user and then have the connection drop** — that state is neither success nor failure, and none of the existing K8s release mechanisms (rolling update, graceful termination) were designed for it.

From the model's point of view, this stage is "weights frozen, forward pass only." But from the platform's point of view, it is the only place among the four stages where you have to think about being **stateful** — the KV cache lives in the GPU memory of one specific instance, which breaks the assumption that "a stateless service can be scaled freely."

**What upstream provides**: the `models/sft-current` symlink points at a directory in HF format, and it has already passed the stage 3 checks.
**What downstream needs**: an OpenAI-compatible HTTP endpoint that clients can use by changing one `base_url`.

## 2. What this stage must accomplish: six things

| # | Task | Consequence of skipping it |
| --- | --- | --- |
| 1 | Expose a standard protocol (OpenAI-compatible) | Every caller has to write an adapter layer |
| 2 | **Take traffic only after the engine is truly ready** | Requests hit an instance still loading the model, 5xx |
| 3 | Autoscale on load feedback | Either waste GPUs or get overloaded |
| 4 | Zero-downtime upgrade | Swapping models requires downtime |
| 5 | Rollback | A bad new version can only be fixed by rolling forward |
| 6 | Clear drain and cancellation semantics for streaming requests | Releases truncate output the user is reading |

Item 2 is the **most dangerous** configuration in this stage, because it has a "writing it makes things worse" trap (see section 4.3). Item 6 is the one most likely to be ignored completely.

## 3. Why the configuration is written this way

Stage 4 is characterized by **almost zero code** — there is no `serve_app.py`; all the logic lives in the YAML's `serveConfigV2`. So this section is about that embedded config.

### 3.1 Why import_path points at Ray's own builder

```yaml
serveConfigV2: |
  applications:
    - name: llm
      import_path: ray.serve.llm:build_openai_app
      route_prefix: "/"
      args:
        llm_configs:
          - model_loading_config: ...
```

The format of `import_path` is `module:callable`. Ray Serve imports it, passes `args` in as parameters, and gets back a Serve application.

The `ray.serve.llm:build_openai_app` referenced here is Ray's own builder, and it assembles three layers for you:

```
HTTP layer      OpenAI-compatible routes (/v1/models, /v1/chat/completions, /v1/completions, /v1/embeddings)
  ↓
Routing layer   LLMRouter: multi-model dispatch, request queuing, (optionally) prefix-aware routing
  ↓
Engine layer    LLMServer × N replicas, each running a vLLM engine internally
```

Writing your own `@serve.deployment` wrapper around vLLM also works, but then you have to implement OpenAI's request/response models, the SSE streaming protocol, multi-model dispatch, and metadata endpoints like `/v1/models` yourself. All of that is pure protocol plumbing with no business value.

**The price is flexibility**: to insert custom logic before or after generation (rewriting prompts, filtering output, audit logging) you have to switch to a custom service class via `server_cls`, or add another Serve deployment of your own in front. `build_openai_app` means "zero code for the standard case," not "everything is customizable."

The contents of `args` are exactly the fields of `LLMConfig`; writing them in YAML is completely equivalent to writing `LLMConfig(...)` in Python — Ray uses pydantic for validation, and the field names map one to one. **The benefit is separating config from code**: swapping models, tuning replica counts, and changing engine parameters all touch only YAML, with no image rebuild.

### 3.2 model_id and model_source are two different things

```yaml
model_loading_config:
  model_id: sft-qwen
  model_source: /mnt/cluster_storage/models/sft-current
```

- **`model_id`** is the **external name**. It is what the client puts in `"model": "sft-qwen"`, and what `/v1/models` returns. It is a stable API contract and should not change as the underlying version changes.
- **`model_source`** is **where the weights come from**. It can be an HF model id, a local directory, or a `CloudMirrorConfig` in the form `{bucket_uri: "s3://..."}`.

The value of separating the two is that **switching model versions does not affect callers**. The underlying weights move from `sft-20260914` to `sft-20260915` while `model_id` stays `sft-qwen`, so not one line of client code changes.

#### model_source must be an immutable version, never a symlink

This is the opposite of stage 3, for entirely different reasons, and it is worth spelling out on its own.

Stages 1 and 2 established the `models/sft-current` symlink, and it is very natural to want to use it here too — "serve whatever training released." But doing that gets you a service that **looks like it is working and in fact never updates**:

```yaml
# ✗ Wrong: changing the symlink does not make the engine reload the weights
model_source: /mnt/cluster_storage/models/sft-current
```

Weights already loaded into GPU memory do not change because a symlink on disk was repointed. Changing the symlink only affects **the next engine start**. So what you see is:

1. training produces a new version, the symlink is repointed at it;
2. the service keeps running on the old weights, and `/v1/models` still shows the same `model_id`;
3. no errors, no status change — **you think you released, but you did not**.

And worse: after a replica restarts due to scale-up or a failure, it loads the new weights, so **two versions, old and new, are serving inside the same service at once**, and identical requests get answers in different styles.

The correct approach is to write the immutable version into `serveConfigV2`:

```yaml
model_source: /mnt/cluster_storage/models/REPLACE_WITH_SFT_RUN_ID
```

This way, the act of "releasing a new version" **is** "change this line and apply," and changing `serveConfigV2` itself triggers a Serve application update. Releasing no longer depends on any implicit side effect.

`make serve` blocks the case where the placeholder has not been replaced:

```makefile
@grep -q REPLACE_WITH_SFT_RUN_ID 40-serve/rayservice-llm.yaml && { \
    echo "ERROR: replace model_source with the version directory exported by stage 2 first"; exit 1; } || true
```

The price is that the manifest has to change per release — which is exactly how it should be: **a release is an explicit, reviewable, revertible configuration change, not a symlink edit.** The symlink stays for humans and for stage 3 (which `resolve()`s it into an immutable path itself), but it is not the service's loading source.

### 3.3 The KV cache determines the concurrency ceiling

In the first three stages the KV cache was a bit player: stage 2 explicitly **turns it off**
([training article, section 3.6](../20-train/sft-training.en.md); during training it is useless and eats GPU memory),
and stage 3 casually pushes it near the limit. In online serving it becomes **the entirety of capacity planning** —
`target_ongoing_requests` in section 3.4 below, the metrics in section 3.5, and the three numbers
that have to be computed together when scaling up in section 8 all follow from the arithmetic in this section.

#### What it is, and why it makes the service stateful

Autoregressive generation produces one token at a time, but every step needs the new token to attend to the K and V of **all preceding tokens**. Those K/V depend only on the historical tokens themselves and never change — so cache them, and each step only computes the new token's share. That is the KV cache.

Without caching, generating the nth token means recomputing the K/V of the previous n-1, taking total cost from O(n) to O(n²). So it is not an optional optimization; it is the precondition for autoregressive inference being usable at all.

The price is that **this cache lives in the GPU memory of one specific instance**. Therefore:

- GPU memory usage grows with "concurrency × sequence length" rather than being a fixed model size;
- if a follow-up request in the same session is routed to a different replica, the cache is wasted and has to be prefilled again;
- the assumption that "a stateless service can be scaled and routed freely" does not hold here.

This is the most fundamental architectural difference between stage 4 and the first three stages.

#### The arithmetic and the measured numbers

```
KV cache per token = 2 (K and V) × layers × KV heads × head_dim × bytes per element
```

Note that it is the **number of KV heads** (`num_key_value_heads`), not the number of attention heads (`num_attention_heads`). Modern models generally use GQA (grouped-query attention), where multiple query heads share one group of KV heads. Computing with the attention head count **overestimates by an order of magnitude**:

| Model | Layers | Attention heads | **KV heads** | head_dim | Per token |
| --- | --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | 24 | 14 | **2** | 64 | 12 KiB |
| Qwen2.5-7B-Instruct | 28 | 28 | **4** | 128 | 56 KiB |

(Numbers taken from `config.json` in the two models' HF repos, bf16 precision. The 0.5B's 14 query heads share 2 KV groups, and the 7B's 28 share 4 — GQA shrinks the KV cache by 7x outright.)

A 7B on a single 80GB card with `gpu_memory_utilization: 0.85`:

```
Reserved GPU memory  80 GB × 0.85            ≈ 63.3 GiB
Minus weights        7.62B params × 2 bytes  ≈ 14.2 GiB
──────────────────────────────────────────────────────────
KV cache pool                                ≈ 49 GiB      (activations and CUDA context overhead still to be deducted)
token budget         49 GiB ÷ 56 KiB         ≈ 920 K tokens
```

**The key point is that the budget's unit is a total token count, not a sequence count.** vLLM's PagedAttention slices the cache into fixed-size blocks allocated on demand, so a 200-token sequence occupies only 200 tokens' worth of space rather than reserving `max_model_len`. Therefore:

| Average sequence length (prompt + output) | Concurrency ceiling |
| --- | --- |
| 512 token | ~1800 requests |
| 1024 token | ~900 requests |
| 2048 token | ~450 requests |
| 4096 token | ~225 requests |

Only with this table can you judge whether the parameters are reasonable. The `target_ongoing_requests: 32` set in this directory (next section) is **extremely conservative** for 7B/80GB — it was chosen for an experimental environment running 0.5B on a single 24GB card. When scaling to 7B this value should go up accordingly, otherwise you trigger scale-up while 90% of the KV cache is still free, buying GPUs for nothing.

#### How the three parameters interact

```
max_model_len            ── ceiling for a single sequence, decides whether overlong requests are rejected
       ×
gpu_memory_utilization   ── how large the KV pool is
       ↓
   token budget  ÷  average sequence length  =  real concurrency ceiling
       ↓
target_ongoing_requests  ── must be below it, otherwise scale-up is always too late
```

All three must be computed together. The most common mistake is changing only one:

- **Only raising `max_model_len`**: from 4096 to 32768, the single-sequence ceiling grows 8x, but the token
  budget is unchanged — long requests coming in drain the pool quickly, and short requests are forced to queue
  or even get preempted and recomputed. The symptom is degraded latency **with no error**.
- **Only raising `target_ongoing_requests`**: Serve keeps dispatching requests to an engine that cannot hold
  that many sequences, and the requests queue inside the engine. Serve-layer metrics look healthy (in-flight
  request count near the target), while TTFT has already collapsed.
- **Only raising `gpu_memory_utilization`**: the headroom left for activations and the CUDA context gets
  squeezed out, showing up as OOM at engine startup, or OOM on some long request after running for a while.

#### prefill and decode use the KV cache in completely different ways

This is the prerequisite for understanding why TTFT and TPOT are **two independent metrics**:

| | prefill (processing input) | decode (token-by-token generation) |
| --- | --- | --- |
| KV cache operation | **writes** the whole prompt's K/V in one go | **appends** one token's K/V per step |
| Bottleneck | compute (matrix multiply) | memory bandwidth (must read all historical K/V) |
| Parallelism | high, the whole sequence is computed together | low, inherently serial |
| Corresponding metric | **TTFT** | **TPOT / ITL** |
| Optimizations | prefix caching, chunked prefill | continuous batching, larger batches |

The two phases compete for the same card, and their optimization directions are opposite — decode wants to accumulate large batches to improve bandwidth utilization, while prefill wants to run immediately to lower TTFT. `enable_chunked_prefill` is the reconciliation of that conflict: it chops a long prompt's prefill into chunks interleaved between decode steps, so one long prompt does not stall every request currently generating.

This also explains why "the service is slow" cannot be diagnosed from a single metric: high TTFT is a prefill or queuing problem, high TPOT is a decode or too-large-batch problem, and the treatments are completely different.

#### prefix caching is nearly free on this pipeline

```yaml
enable_prefix_caching: true
```

The hit condition is that **the request prefix is exactly identical**. The hit rate is naturally high in this directory's scenario, because stage 1 generates training data with `apply_chat_template` and stages 3/4 have the engine render requests with the same template, so every request begins with the same template header:

```
<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n     ← identical for everyone, always a hit
{the user's actual question}                              ← only from here does it diverge
```

The template header is usually a few dozen tokens, a small fraction, so the benefit is limited but genuinely positive. What makes prefix caching pay off by an order of magnitude are two other scenarios: **long system prompts** (hundreds to thousands of tokens of role setup shared by all requests) and **multi-turn conversations** (the prefix of turn n is everything from the previous n-1 turns).

The latter raises a platform problem: with multiple replicas, if the second turn of the same session is routed to another replica, the cache is entirely wasted. Solving that requires **KV-aware routing** (sending the request back to the replica that has its prefix cached); see section 7 and article 6 of this repository's blog.

### 3.4 Why autoscaling cannot use CPU utilization

```yaml
deployment_config:
  autoscaling_config:
    min_replicas: 1
    max_replicas: 4
    target_ongoing_requests: 32
  max_ongoing_requests: 64
```

**Using CPU utilization for HPA on an LLM inference service is wrong.** The reason is that the bottleneck is elsewhere:

| Phase | GPU | CPU | Observed |
| --- | --- | --- | --- |
| prefill (processing input) | compute saturated | mostly idle | CPU metrics show no load |
| decode (token-by-token generation) | memory bandwidth saturated, compute not necessarily | mostly idle | CPU metrics show no load |

The GPU already has a long queue and users are waiting, while CPU utilization may be only 15%. A CPU-based HPA concludes that everything is fine and does not scale up.

The correct signal is **queuing**. `target_ongoing_requests: 32` means "each replica handling 32 requests at once is the healthy state" — above that, scale up; below it, scale down. It is a quantity Serve can observe directly, with no extra metrics pipeline needed.

The relationship between the two parameters has to be clear:

- `target_ongoing_requests: 32` — the **target value** for autoscaling (a soft threshold).
- `max_ongoing_requests: 64` — the **hard ceiling** per replica. Beyond that number, Serve's routing layer stops dispatching to this replica, and requests queue at the routing layer.

`max` must be greater than `target`, and the gap between them is "the buffer for while scale-up has not completed." Too small a buffer and requests get blocked at the routing layer during scale-up; too large and a single replica accumulates too many requests, degrading latency. 2x is a common starting point, but **the correct value must be measured** — it depends on your input length distribution and latency targets.

**The upper bound on both of these numbers is determined by the KV cache; you cannot fill them in by feel.** As computed in the previous section: a 7B on an 80GB card with an average sequence of 1024 tokens has a real concurrency ceiling of roughly 900 requests. This directory writes 32 because it targets an experimental environment running 0.5B on a single 24GB card; on 7B that is far too conservative — it would trigger scale-up while 90% of the KV cache is still free. Conversely, if you set `target` above the concurrency the KV cache can hold, Serve keeps dispatching to an engine that cannot fit it, and every Serve-layer metric looks fine while TTFT has already collapsed. Compute the token budget from section 3.3 first, then set these two values.

One fact you must know: **new replicas are not immediately available.** Scaling up has to go all the way through K8s scheduling the Pod → pulling the image → Ray joining the cluster → loading the weights → allocating the KV cache, and for a 7B model that chain takes minutes. So LLM autoscaling can only respond to **minute-scale trends**, not second-scale spikes. Spikes are handled by `max_ongoing_requests` queuing and rate limiting, not by scaling up.

#### Configuring autoscaling_config alone will not scale anything: you also have to enable the Ray autoscaler

This one is easy to miss, and once missed the symptoms are very misleading. The `autoscaling_config` above only governs the **Serve replica count**, and replicas have to land on Ray workers. **Adding Ray worker Pods is the Ray autoscaler's job, and it is off by default.**

So configuring only the Serve-side parameters gets you:

```yaml
workerGroupSpecs:
  - replicas: 1
    maxReplicas: 4          # ← decorative, it will never reach 4
```

The cluster only ever has 1 GPU worker; when Serve tries to scale to a second replica it cannot get a GPU, the replica stays PENDING, and the `max_replicas: 4` written in `autoscaling_config` never takes effect. The symptom is "autoscaling is configured but never scales," and the Serve-layer logs only say there are insufficient resources.

You have to enable it explicitly in `rayClusterConfig`:

```yaml
rayClusterConfig:
  rayVersion: "2.58.0"
  enableInTreeAutoscaling: true
  autoscalerOptions:
    idleTimeoutSeconds: 300        # idle wait before scaling down
    upscalingMode: Default
    resources:
      requests: {cpu: 200m, memory: 256Mi}
      limits: {cpu: "1", memory: 1Gi}
```

`idleTimeoutSeconds: 300` is much larger than the default because model loading is expensive: scaling a replica down and back up costs minutes, which is not worth flapping over for a traffic trough of a few dozen seconds.

This is also the full picture of **three-layer autoscaling**, where each layer covers one segment and nothing scales if any layer is missing:

```
Serve autoscaler   decides how many replicas are needed from target_ongoing_requests
        ↓ needs more GPUs
Ray autoscaler     enableInTreeAutoscaling, decides how many worker Pods are needed
        ↓ needs more nodes
Cluster autoscaler / Karpenter   decides how many GPU nodes are needed
```

This directory only configures the second layer. The third (node level) is out of scope for the example, but if your GPU node pool is fixed, the ceiling of `maxReplicas: 4` is actually determined by the node pool's capacity rather than by that number.

### 3.5 log_engine_metrics decides whether you can do capacity planning

```yaml
log_engine_metrics: true
```

Without this switch you can only see Serve-layer metrics (replica count, in-flight requests). Only with it on do you get vLLM's internal engine metrics:

- **TTFT** (time to first token) — the response speed users perceive, mainly affected by prefill and queuing;
- **TPOT / ITL** (time per output token) — generation speed;
- **engine queue depth** — closer to the real bottleneck than the Serve-layer in-flight count;
- **KV cache utilization** — decides whether more concurrency can still fit;
- **prefix cache hit rate** — decides whether prefix caching is actually working.

**You must have this set of metrics to do capacity planning and troubleshooting.** For the symptom "the service is slow," the Serve layer alone cannot distinguish requests queuing at the routing layer, queuing in the engine, or a single request simply being slow — and the three have completely different treatments (add replicas / tune batch parameters / reduce max_model_len).

### 3.6 upgradeStrategy: the default strategy doubles GPU demand

```yaml
upgradeStrategy:
  type: NewCluster
```

`NewCluster` is the default strategy, and the flow is:

```
old cluster serving → bring up a complete new cluster → new cluster ready → switch traffic → reclaim old cluster
                     └─ during this window both clusters exist ─┘
```

**Inside that window, GPU demand is twice the normal level.** For this directory (1 GPU worker) that means an upgrade needs 2 cards; for a production service with 4 replicas it means the upgrade momentarily needs 8 cards.

If the cluster does not have that much headroom, the consequence is not "the upgrade is a bit slow" but **the new cluster's Pods stay Pending forever, the service is stuck on the old version, and the RayService status looks like it is "upgrading."** This failure mode is well hidden, because the old version is still serving normally and nobody immediately notices that the upgrade never happened at all.

v1.7.0 offers `NewClusterWithIncrementalUpgrade` (beta, enabled by default), which migrates traffic gradually in proportion and does not need double resources all at once, but it depends on the Gateway API. There is also `None` (in-place update, with downtime).

**The key point is: do not assume an upgrade always costs zero extra GPUs.** Compute the resource demand inside the window before upgrading.

#### A knob that is already deprecated and has no effect even when set

Many tutorials (including earlier versions of this directory) configure one item here:

```yaml
# ✗ Completely ineffective on KubeRay v1.7.0
deploymentUnhealthySecondThreshold: 900
```

The motivation for setting it is right — "LLMs load slowly, don't let the controller declare the deployment failed too early." But look at v1.7.0's `rayservice_types.go` and this field is explicitly labeled:

```go
// Deprecated: This field is not used anymore. ref: .../kuberay/issues/1685
DeploymentUnhealthySecondThreshold *int32 `json:"deploymentUnhealthySecondThreshold,omitempty"`
```

`serviceUnhealthySecondThreshold` is deprecated the same way. Setting it raises no error, fails no validation, and has no effect — which is the worst class of configuration: **you think you added a layer of protection, and in reality there is nothing there.**

So this directory deleted it. Protection for slow loading actually comes from the worker's **liveness `failureThreshold`** (see section 4.4); that is what really decides "how long loading has to take before it counts as dead."

The general lesson: **before adding a field to a CRD, go check the types file to confirm it is still being read.** A CRD accepts any schema-valid field, and acceptance does not mean it takes effect.

### 3.7 terminationGracePeriodSeconds and streaming responses

```yaml
terminationGracePeriodSeconds: 300
```

This value must be **greater than the p99 generation time of a single request**.

K8s terminates a Pod by sending SIGTERM → waiting `terminationGracePeriodSeconds` → sending SIGKILL. For an ordinary HTTP service the 30-second default is plenty (requests take milliseconds). For streaming generation, one request can last tens of seconds to minutes.

Too short a grace period causes this: during a rolling release, a connection that is emitting tokens to a user gets cut off by SIGKILL. What the user sees is **output stopping halfway**, with no error message and no terminator.

But even with a generous grace period, it only makes a **best effort** to let in-flight requests finish — it cannot guarantee:

- a request that happens to start right at the grace-period boundary will still be truncated;
- whether the client can distinguish "normal end" from "truncated" (in the SSE protocol the only difference is whether `data: [DONE]` was received);
- whether retrying is safe (the user has already seen partial output, and retrying produces duplicate content).

**These semantics have to be defined at the application layer; the platform layer cannot give them to you.** This directory only goes as far as "give a generous grace period"; it does not implement drain markers, per-request cancellation tracking, or retry deduplication. That is a deliberate boundary; see section 7.

## 4. Why the YAML is written this way: probes are the most dangerous configuration in this stage

This section is broken out on its own because it is **the one place in the whole pipeline where following conventional best practice introduces a failure**.

### 4.1 KubeRay's probe injection rules

When creating a Pod, KubeRay checks whether the container declares probes (`ray-operator/controllers/ray/common/pod.go`):

```go
if rayContainer.LivenessProbe == nil {
    // inject default liveness
}
if rayContainer.ReadinessProbe == nil {
    // inject default readiness
    if creatorCRDType == utils.RayServiceCRD && rayNodeType == rayv1.WorkerNode {
        // additionally append the Serve proxy health check, and set failureThreshold to 1
    }
}
```

The rule is: **user-declared probes win, and KubeRay does not touch them at all.**

The logic is the same on v1.4.2 and v1.7.0; this is not the special behavior of one version.

For Ray ≥ 2.53 the injected probes use the unified health endpoint `httpGet :52365/api/healthz`; earlier versions use an exec probe with wget against the raylet.

### 4.2 The trap: writing a probe for a RayService worker loses the Serve check

Note the nested if in the code above: **only when the user has not written a `readinessProbe` does KubeRay append the Serve proxy health check.**

Which produces this scenario:

1. following K8s best practice, you add a `readinessProbe` to the RayService's worker container, checking raylet health;
2. KubeRay sees `ReadinessProbe != nil` and skips the entire injection branch;
3. **the Serve proxy's `/-/healthz` check is gone**;
4. result: the Pod is Ready as soon as the raylet is up, but the vLLM engine is still loading weights (several minutes);
5. the Pod is added to the Service's Endpoints, requests come in → 5xx.

**You did something "more correct" and introduced an intermittent failure that only shows up during scale-up and releases.**

So there are exactly two correct approaches, with no third:

- **either write no probes at all** and let KubeRay inject them (that is what stages 1, 2, and 3 do);
- **or write them completely**, including the Serve proxy check.

This directory picks the second, because `failureThreshold` has to be tuned (see 4.4 for why):

```yaml
readinessProbe:
  exec:
    command:
      - bash
      - -c
      - >-
        wget --tries 1 -T 2 -q -O-
        http://localhost:52365/api/local_raylet_healthz | grep success &&
        wget --tries 1 -T 5 -q -O-
        http://localhost:8000/-/healthz | grep success
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 10
  failureThreshold: 1
```

Two checks in series, with `&&` ensuring both must pass to count as Ready:

- `:52365/api/local_raylet_healthz` — the Ray node (raylet) is alive;
- `:8000/-/healthz` — Serve's HTTP proxy can handle requests.

`exec` is used rather than `httpGet` because a probe can only have one `httpGet`, and here two endpoints have to be checked. `grep success` is KubeRay's own approach — those two endpoints include `success` in the response body, and the HTTP status code alone is not enough (KubeRay's `BaseWgetHealthCommand` constant is literally `wget ... | grep success`).

### 4.3 Why failureThreshold: 1 is right

```yaml
failureThreshold: 1
```

Removing from the endpoint set after a single failure looks aggressive. But for readiness it is correct, because **the cost of a readiness failure is extremely low, while the cost of wrongly being judged Ready is extremely high**:

| | readiness failure | wrongly judged Ready |
| --- | --- | --- |
| Consequence | the Pod is removed from Endpoints and takes no new requests | requests hit an unprepared engine and users get 5xx |
| Restarts the container? | **No** | — |
| Recovery | added back automatically once the probe passes again | requires manual intervention |

Readiness does not restart the container (that is liveness's job), so "a brief NotReady is preferable" costs almost nothing. KubeRay's own injection also uses `ServeReadinessProbeFailureThreshold = 1`; this directory just replicates that value explicitly.

It also means: **the Pod will be NotReady for the whole model-loading period, and that is expected behavior, not a failure.** When a newly scaled-out Pod stays not Ready for several minutes, first confirm whether it is still loading weights.

### 4.4 The liveness failureThreshold must cover the model load time

```yaml
livenessProbe:
  httpGet:
    path: /api/healthz
    port: 52365
  initialDelaySeconds: 60
  periodSeconds: 5
  timeoutSeconds: 5
  failureThreshold: 120
```

`120 × 5s = 600s`, that is, a 10-minute tolerance window (plus the 60-second initialDelay).

**Liveness and readiness have exactly opposite natures**: a liveness failure **restarts the container**. For a vLLM engine in the middle of loading 14GB of weights, being restarted means starting over. If the threshold is smaller than the load time, you get:

```
Pod starts → loads weights (5 minutes) → liveness declares failure at minute 2 → restart
          → reloads weights → killed again at minute 2 → infinite loop
```

The symptom is "the Pod keeps going into CrashLoopBackOff, and the logs always stop at loading the model," which is easily misdiagnosed as "the model is too big to start" or "not enough GPU memory."

**So the bigger the model, the bigger liveness's `failureThreshold × periodSeconds` has to be.** Pulling a 70B model's weights from object storage can take more than 10 minutes, and this value has to be scaled up accordingly. This is a parameter that must be adjusted with model size; do not copy the template blindly.

`httpGet` is used here rather than exec because liveness only needs to confirm the Ray node is alive and does not need to check Serve — the health of the Serve application is checked separately by the RayService controller on every reconcile.

### 4.5 Why the head uses relatively relaxed probes

```yaml
# head
livenessProbe:
  httpGet: { path: /api/healthz, port: 52365 }
  failureThreshold: 120
readinessProbe:
  httpGet: { path: /api/healthz, port: 52365 }
  failureThreshold: 10
```

The head does not run an inference engine (`num-gpus: "0"`); it only does GCS, the dashboard, and Serve routing, and it starts in seconds, so a plain `failureThreshold: 10` is enough for readiness.

The head's readiness does **not** need the Serve proxy check — KubeRay's injection only adds it for workers. The reason is that the RayService controller actively checks the health of the HTTP proxy on the head on every reconcile, going through controller logic rather than a kubelet probe.

### 4.6 Why GPU workers use Guaranteed QoS

```yaml
resources:
  requests: { cpu: "8", memory: 32Gi, nvidia.com/gpu: "1" }
  limits:   { cpu: "8", memory: 32Gi, nvidia.com/gpu: "1" }
```

The reasoning is the same as in stage 2 (GPUs cannot be oversubscribed, and being OOM-killed is expensive), but **online serving is stricter**: an OOM-killed batch job just reruns, while an OOM-killed online service is a user-visible outage — and it usually happens at peak traffic, precisely the moment when nothing should go wrong.

The head uses `requests: 8Gi / limits: 12Gi` (Burstable) instead. The head is not on the critical compute path, and its memory usage fluctuates mainly because of the dashboard and log aggregation, so allowing bursts is the better deal.

### 4.7 Service topology: how requests actually get in

KubeRay creates two Services for one RayService:

| Service | Name | Purpose |
| --- | --- | --- |
| head service | `<cluster>-head-svc` | dashboard(8265), GCS(6379), client(10001) |
| **serve service** | `<name>-serve-svc` | **application traffic(8000)** |

The naming rule comes from `GenerateServeServiceName`: `<rayservice-name>-serve-svc`. So since this directory's RayService is named `llm-serve`, the serve service is `llm-serve-serve-svc` — which is exactly `smoke_test.sh`'s default.

The serve service's Endpoints contain only **Ready** Pods, and that is why the probe trap in section 4.2 is fatal: readiness is the only gate on traffic.

By default the head Pod is also in the serve service (the head runs an HTTP proxy too). `spec.excludeHeadPodFromServeSvc: true` can exclude it so application traffic goes only to workers. That is worth considering when the GPU workers and the head differ a lot in spec; this directory does not set it.

**This directory configures neither Ingress nor Gateway**; acceptance relies on `kubectl port-forward`. Production needs a gateway in front, and you have to be careful: **many gateways buffer response bodies by default**, which turns a streaming response into "wait for the whole generation, then return it all at once," making TTFT meaningless. Nginx needs `proxy_buffering off`; for Envoy, confirm the buffer filter is not enabled.

## 5. How to verify it

```bash
make serve
make test          # equivalent to ./40-serve/smoke_test.sh
```

`smoke_test.sh` does four checks. First a cautionary example, because it has more teaching value than the correct version.

### A cautionary example: an acceptance script that hands a broken service a free pass

An earlier version of this directory was written like this:

```bash
# Check 2: non-streaming
response="$(curl -fsS ... -d "{...}")"
if [[ -z "$response" ]]; then exit 1; fi          # ✗ only tests for an empty string

# Check 3: streaming
chunks="$(curl -fsS -N ... | grep -c '^data: ' || true)"   # ✗ || true swallows transfer failures
if [[ "${chunks:-0}" -lt 2 ]]; then exit 1; fi              # ✗ counts lines
```

I ran this script with fake `kubectl` and `curl`, feeding it a **completely broken** service:

- the non-streaming call returned `{}`;
- the streaming call returned only one role event plus one `error` event, with no content at all and no `[DONE]`;
- `curl` exited with code 28 (timeout).

The script printed "**all three checks passed**" and exited 0.

Four independent defects stacked together:

| Defect | Consequence |
| --- | --- |
| `[[ -z "$response" ]]` only tests for an empty string | `{}` and `{"error":...}` all count as passing |
| `\|\| true` | curl's transfer failures (timeout, connection drop) are swallowed entirely |
| counting `data:` lines | passes even with only protocol frames and no content; `error` events are counted too |
| not checking `[DONE]` | a stream cut off midway counts as complete |

And the most fundamental one is: **counting SSE lines simply cannot detect buffering.** A proxy can buffer the complete response and then deliver every event line at once — the line count is identical. Conversely, a very short correct answer may genuinely have only one content chunk. There is no relationship between line count and "incremental arrival."

### The four checks after the fix

**Check 1: `/v1/models` contains model_id exactly.** Parse with `json.load` and compare `data[].id`, rather than `grep`. Failure here is usually a wrong indent or field name in `serveConfigV2` — and that class of error **does not crash the Pod**; the Pod stays Running while the application status is DEPLOY_FAILED:

```bash
kubectl -n llm-pipeline get rayservice llm-serve \
  -o jsonpath='{.status.activeServiceStatus.applicationStatuses}' | python3 -m json.tool
```

**Check 2: non-streaming has real content.** It requires `choices[0].message.content` to be a non-empty string with a `finish_reason`, and fails immediately if an `error` field appears. The output of this step should be **stylistically consistent** with stage 3's sampled output — inconsistency means the two stages did not load the same model version.

**Check 3: stream completeness.** It checks `curl`'s exit code separately (no more `|| true`) and requires: at least one content delta, `[DONE]` must be received, no `error` events, and every SSE payload must be valid JSON.

**Check 4: incremental delivery — judged by time.**

```python
events.append((time.monotonic(), piece))     # record the arrival time of each content chunk
...
span = events[-1][0] - events[0][0]
if span < span_min:                          # default 0.15s
    sys.exit("The content chunks arrived almost simultaneously; the response was very likely buffered by an intermediate layer")
```

The criterion changes from "how many chunks" to "how long from the first chunk to the last." In a fully buffered response every line arrives at once, so the span is close to 0.

There is an implementation detail worth mentioning here: **one python process must timestamp as it reads**; you cannot fork a `python3 -c 'print(time.time())'` per line in the shell — interpreter startup alone is tens of milliseconds, which inflates every inter-line gap and makes buffering detection useless. The overhead of the measuring tool itself must be far smaller than the quantity being measured.

When there is only one content chunk, this check is skipped with an explanation (normal for a short answer) rather than raising an error.

**These four checks were validated with five fixtures**: empty `{}` + error event + curl 28 → fail; stream buffered and delivered all at once → fail; missing `[DONE]` → fail; only protocol frames and no content → fail; incremental arrival and complete → pass. **The acceptance script itself also needs to be accepted** — a check that never fails is worse than no check, because it gives false confidence.

The characteristic of the buffering failure is: **everything works functionally, only TTFT has become the full generation time**. It raises no error, triggers no alert, and merely makes users feel it is slow, while all Serve-layer metrics look healthy. So it has to be tested specifically.

## 6. Upgrade and rollback

**Swapping model versions**: change `model_source` inside `serveConfigV2`, then apply. That is the whole step.

```bash
# 1. Check which version stage 2 exported
kubectl -n llm-pipeline exec <any pod that mounts the PVC> -- \
  cat /mnt/cluster_storage/models/sft-current/pipeline_provenance.json

# 2. Replace the version in the manifest
sed -i 's|models/sft-20260914-2130|models/sft-20260915-0900|' 40-serve/rayservice-llm.yaml

# 3. apply. Changing serveConfigV2 triggers a Serve application update
kubectl apply -f 40-serve/rayservice-llm.yaml

# 4. Confirm it really switched
kubectl -n llm-pipeline get rayservice llm-serve \
  -o jsonpath='{.status.activeServiceStatus.applicationStatuses}' | python3 -m json.tool
```

#### A wrong approach that used to be written here

An earlier version gave "change the symlink + patch a field to trigger the upgrade," and that patch was:

```bash
# ✗ This command does nothing at all
kubectl -n llm-pipeline patch rayservice llm-serve --type=merge \
  -p '{"spec":{"rayClusterConfig":{"rayVersion":"2.58.0"}}}'
```

`rayVersion` in the manifest was already `"2.58.0"`. **Patching a field to its current value does not constitute a spec change**, does not produce a new generation, and therefore triggers no rollout at all. So: the symlink changed but the engine did not reload, the patch executed successfully but had no effect, and `kubectl` returned 0 — a triple "success," with the model not swapped in the slightest.

This mistake has two lessons:

1. **Do not rely on "change some field, any field" to trigger a release.** Even if you pick a field that really does change (adding an annotation, say), you are depending on a side effect rather than expressing intent. Once the version is written into `serveConfigV2`, "changing the config" and "swapping the model" are the same action.
2. **Do not fake a Ray version number in order to trigger a rollout.** That makes the manifest's `rayVersion` inconsistent with the actual version in the image, and that field also determines the autoscaler sidecar's image.

**Rollback**: change `model_source` back to the old version, then apply. The old version directory is always kept (stage 2's `publish` only adds and never deletes, and it refuses to overwrite outright when `export_dir.exists()`), which is the direct payoff of versioned releases. Rollback takes exactly the same path as a release, and that is why it is reliable.

**Two different upgrade paths** must be kept distinct:

| Change | KubeRay behavior |
| --- | --- |
| Only `serveConfigV2` changes | Updates the Serve application on the existing cluster (no cluster swap) |
| `rayClusterConfig` changes | Follows `upgradeStrategy`: bring up a new cluster, then switch traffic |

Only the second needs double GPUs. Tuning Serve-layer parameters like `num_replicas` or `target_ongoing_requests` takes the first path and costs far less.

**One last thing that has to be stated clearly**: whichever path you take and however long the grace period is, **a streaming request that has already been interrupted cannot be recovered**. The partial output the user has already seen is just there. "Zero-downtime upgrade" means new requests are unaffected, not "every in-flight streaming request can finish completely." Those two things are frequently conflated.

## 7. What this stage deliberately omits

Online serving has the biggest production gap of the four stages:

| Omitted | Cost | Priority |
| --- | --- | --- |
| **Release gate** (check stage 3's evaluation verdict before loading) | Models that failed evaluation can still go live | Highest |
| Ingress / Gateway + TLS | port-forward only | High |
| Authentication and quotas | The endpoint is fully open inside the cluster | High |
| Artifact hash verification | Tampered weights are loaded just the same | Medium |
| PodDisruptionBudget | Node maintenance may take out every replica at once | Medium |
| Prometheus + Grafana wiring | Metrics are enabled but nobody collects them | Medium |
| canary / gradual rollout | Only all-at-once switching | Medium |
| Drain markers and cancellation tracking for streaming requests | Cannot answer "how many requests did this release truncate" | Medium |
| KV-aware routing | Low prefix cache hit rate with multiple replicas | Low (limited benefit with few replicas) |

Already added (absent in earlier versions): **`enableInTreeAutoscaling`** (otherwise replicas cannot scale out),
**writing an immutable version into `model_source`** (otherwise changing the symlink has no effect and old and new replicas run mixed),
**deleting the deprecated `deploymentUnhealthySecondThreshold`**,
and **an acceptance script that can actually fail** (including time-based buffering detection).

**The first item in the table is the most worth adding first, and the cheapest.** The idea is to change `import_path` from `ray.serve.llm:build_openai_app` to a thin wrapper of your own:

```python
# serve_app.py (this file does not exist in this directory)
def build(args):
    release = read_json(Path(args["model_dir"]) / "_READY.json")
    if not release["evaluation"]["passed"]:
        raise ValueError("the model did not pass the offline evaluation gate")
    # verify artifact hashes ...
    return build_openai_app({"llm_configs": [LLMConfig(...)]})
```

This way, "a model that failed evaluation may not go live" is no longer a sentence in a process document but **a raise at deploy time** — the Serve application fails to come up, the RayService status becomes DEPLOY_FAILED, and the old version keeps serving. The contract itself is small: once stage 3 passes, write a `_READY.json` into the model directory recording the evaluation verdict and per-file hashes; here, read and verify it before loading, and `raise` if anything fails to match.

On KV-aware routing: Ray 2.58's release notes mention that KV cache- and token-aware request routing is complete. What it solves is "a follow-up request in the same session should go back to the replica that has its prefix cached." The benefit is limited with few replicas, and significant with many replicas and long sessions. Article 6 of this repository's blog is dedicated to this topic (using the llm-d Router as its case study) and is worth reading alongside.

## 8. What changes when you scale up

| Model | `tensor_parallel_size` | worker spec | Must be adjusted in step |
| --- | --- | --- | --- |
| 0.5B (this directory) | 1 | 1 × 24GB | — |
| 7B | 1 | 1 × 80GB | Increase liveness `failureThreshold` (loading takes minutes) |
| 70B | 4–8 | multi-GPU single host, or `numOfHosts > 1` multi-host groups | Increase the liveness threshold substantially; `upgradeStrategy`'s double resources become a hard constraint, so incremental is essentially mandatory; cross-node TP requires confirming interconnect bandwidth |

Three numbers have to be computed together and cannot be tuned in isolation:

```
stage 1's MAX_LEN  +  MAX_TOKENS   ≤   stage 4's max_model_len
                                        ↓ determines the KV cache size of a single sequence
                                        ↓ together with gpu_memory_utilization determines the concurrency ceiling
                                        ↓ determines the reasonable value of target_ongoing_requests
```

Raising `max_model_len` from 4096 to 32768 grows the single-sequence KV cache ceiling 8x (from 224 MiB to
1.75 GiB for 7B), while the token budget is unchanged — long requests will drain the pool fast. And if
`target_ongoing_requests: 32` is not recomputed along with it, Serve keeps dispatching requests to an engine
that cannot fit them, showing up as queuing and degraded latency but **without raising an error**. Recomputing
the token budget with the arithmetic in section 3.3 is the only correct way to change these three values.

`gpu_memory_utilization` must be lowered when multiple models share a card. This directory's 0.85 assumes "one model owns the whole card." Two models sharing a card with 0.85 each will OOM outright — and it happens while the second model is loading, so it looks like the second model's problem.

---

## The pipeline in review

Four articles in, the shape of the pipeline is:

```
Stage 1  text → fixed-length tensors   Ray Data           contract: iter_torch_batches can consume it directly
Stage 2  tensors → HF weight directory Ray Train+PyTorch  contract: vLLM can use it as model_source directly
Stage 3  weights → generated results   Ray Data+vLLM      contract: proves the weights are usable
Stage 4  weights → HTTP endpoint       Ray Serve+vLLM     contract: OpenAI-compatible
```

Each segment's output is the **direct input** of the next, **with no conversion script** — that is the only design goal this directory truly cares about. Three things make it work: stage 1 using `apply_chat_template` so the template has a single source, stage 2 exporting the HF directory format, and stages 3 and 4 pointing at the same `model_source`.

The pattern shared by all four stages: **write artifacts versioned, switch symlinks atomically, never overwrite an old version**. It simultaneously provides "safe to rerun," "revertible," and "traceable," at the cost of depending on POSIX semantics (moving to object storage requires redesigning the release protocol).

And the biggest overall gap in this pipeline is the **gate**: stage 3 has no quantitative metrics or regression criteria, and stage 4 does not check the evaluation verdict before loading. Fill that one hole and it goes from "it runs" to "we dare ship it."